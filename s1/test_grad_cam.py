import sys

sys.path.insert(0, "./")

import os
import os.path as osp
import glob
import numpy as np
import torch
import ut
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network

import pickle
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ----------------------- GRAD-CAM CLASS -----------------------
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = dict([*model.named_modules()])[target_layer_name]

        self.gradients = None
        self.activations = None

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def save_activation(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_backward_hook(save_gradient)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        logits, _ = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)

        loss = logits[:, class_idx]
        loss.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )

        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.detach().cpu().numpy()


# # --------------- SAVE GRAD-CAM HEATMAP ----------------
# def save_gradcam(img_tensor, cam_map, step, pred, gt):
#     img = img_tensor.squeeze().detach().cpu().numpy()
#     img = np.transpose(img, (1, 2, 0))
#     img = (img - img.min()) / (img.max() - img.min())

#     cam = cv2.applyColorMap((cam_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
#     cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
#     cam = 0.3 * cam / 255.0 + 0.7 * img

#     plt.figure(figsize=(3, 3))
#     plt.imshow(cam)
#     plt.title(f"Pred: {pred} | GT: {gt}")  # ✅ SHOW PRED & GT
#     plt.axis("off")

#     out_dir = "gradcam_outputs"
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = f"{out_dir}/gradcam_step_{step}.png"
#     plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
#     plt.close()
#     print(f"[GradCAM] Saved: {out_path} (Pred={pred}, GT={gt})")


# --------------- SAVE GRAD-CAM HEATMAP ----------------
def save_gradcam(img_tensor, cam_map, step, pred, gt):
    img = img_tensor.squeeze().detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min())

    # ✅ Intensificar colores del heatmap
    cam_map = cam_map**2.0  # Aumenta intensidad
    cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min())

    cam = cv2.applyColorMap((cam_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

    # ✅ Mezcla más fuerte: más heatmap, menos imagen original
    cam = 0.6 * cam / 255.0 + 0.4 * img  # 0.6 puede subir hasta 0.8

    plt.figure(figsize=(3, 3))
    plt.imshow(cam)
    plt.title(f"Pred: {pred} | GT: {gt}")
    plt.axis("off")

    out_dir = "gradcam_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/gradcam_step_{step}.png"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[GradCAM] Saved: {out_path} (Pred={pred}, GT={gt})")


# ----------------------- ORIGINAL SCRIPT -----------------------
parser = argparse.ArgumentParser("cifar")
parser.add_argument(
    "--data", type=str, default="../data", help="location of the data corpus"
)
parser.add_argument("--dir", type=str, default=None, help="location of population")
parser.add_argument("--data_dir", type=str, default=None, help="path of data")
parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument(
    "--init_channels", type=int, default=36, help="num of init channels"
)
parser.add_argument("--layers", type=int, default=20, help="total number of layers")
parser.add_argument(
    "--model_path", type=str, default="EXP/model.pt", help="path of pretrained model"
)
parser.add_argument("--log_path", type=str, default=None, help="path of log file")
parser.add_argument(
    "--auxiliary", action="store_true", default=False, help="use auxiliary tower"
)
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
parser.add_argument(
    "--drop_path_prob", type=float, default=0.2, help="drop path probability"
)
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--arch", type=str, default=None, help="which architecture to use")
args = parser.parse_args()

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)

if args.log_path is not None:
    fh = logging.FileHandler(os.path.join(args.log_path, "evaluate.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

tmp = ""
for arg in sys.argv:
    tmp += " {}".format(arg)
logging.info(f"python{tmp}")

CIFAR_CLASSES = 7


def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    logging.info("gpu device = %d" % args.gpu)
    logging.info("args = %s", args)

    print("---------Genotype---------")
    if args.arch is not None:
        genotype = eval("genotypes.%s" % args.arch)
    if args.dir is not None:
        if "pickle" in args.dir:
            with open(os.path.join(args.dir), "rb") as f:
                genotype = pickle.load(f)
        else:
            with open(os.path.join(args.dir, "genotype.pickle"), "rb") as f:
                genotype = pickle.load(f)
        logging.info("Unpickling genotype.pickle")
    logging.info(genotype)
    print("--------------------------")

    model = Network(
        args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype
    )
    ut.load(model, args.model_path, args.gpu)
    model = model.cuda()

    logging.info("param size = %fMB", ut.count_parameters_in_MB(model))
    print("param size = %fMB", ut.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().cuda()

    _, test_transform = ut._data_transforms_ckplus(args)
    folder_path = args.data_dir
    test_data = dset.ImageFolder(osp.join(folder_path, "test"), test_transform)

    test_queue = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    model.drop_path_prob = 0.0
    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info("test_acc %f", test_acc)
    print(f"test_acc {test_acc}")


# def infer(test_queue, model, criterion):
#     objs = ut.AvgrageMeter()
#     top1 = ut.AvgrageMeter()
#     top5 = ut.AvgrageMeter()

#     model.eval()

#     # ✅ attach GradCAM to last cell
#     target_layer = f"cells.{args.layers - 1}"
#     print(f"[GradCAM] Using layer: {target_layer}")
#     gradcam = GradCAM(model, target_layer)

#     img_counter = 0  # for unique naming

#     for step, (input, target) in enumerate(test_queue):

#         input = input.cuda()
#         target = target.cuda()

#         logits, _ = model(input)
#         loss = criterion(logits, target)

#         prec1, prec5 = ut.accuracy(logits, target, topk=(1, 2))
#         n = input.size(0)
#         objs.update(loss.item(), n)
#         top1.update(prec1.item(), n)
#         top5.update(prec5.item(), n)

#         # ✅ Grad-CAM for EVERY IMAGE in the batch
#         batch_size = input.size(0)
#         for i in range(batch_size):
#             img = input[i].unsqueeze(0)

#             pred = torch.argmax(logits[i]).item()
#             gt = target[i].item()

#             cam_map = gradcam.generate(img, pred)[0, 0]

#             print(
#                 f"[IMG global={img_counter} | batch={step} idx={i}] Pred={pred} | GT={gt}"
#             )
#             save_gradcam(img, cam_map, f"{step}_{i}", pred, gt)

#             img_counter += 1

#         if step % args.report_freq == 0:
#             logging.info("test %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

#     return top1.avg, objs.avg


def infer(test_queue, model, criterion):
    objs = ut.AvgrageMeter()
    top1 = ut.AvgrageMeter()
    top5 = ut.AvgrageMeter()
    model.eval()

    target_layer = f"cells.{args.layers - 1}"
    print(f"[GradCAM] Using layer: {target_layer}")
    gradcam = GradCAM(model, target_layer)

    img_counter = 0

    # ✅ Store predictions + GT
    all_preds = []
    all_targets = []

    # ✅ CUDA timing setup
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    total_images = 0

    # ✅ start timer
    start_time.record()

    for step, (input, target) in enumerate(test_queue):
        input = input.cuda()
        target = target.cuda()
        batch_size = input.size(0)

        # ✅ Forward pass
        logits, _ = model(input)

        loss = criterion(logits, target)
        prec1, prec5 = ut.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)

        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # ✅ Track how many images processed
        total_images += batch_size

        # ✅ Save predictions
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        batch_targets = target.cpu().numpy()

        all_preds.extend(batch_preds.tolist())
        all_targets.extend(batch_targets.tolist())

        # ✅ GradCAM for each image
        for i in range(batch_size):
            img = input[i].unsqueeze(0)

            pred = int(batch_preds[i])
            gt = int(batch_targets[i])

            cam_map = gradcam.generate(img, pred)[0, 0]
            print(
                f"[IMG global={img_counter} | batch={step} idx={i}] Pred={pred} | GT={gt}"
            )
            save_gradcam(img, cam_map, f"{step}_{i}", pred, gt)
            img_counter += 1

        if step % args.report_freq == 0:
            logging.info("test %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

    # ✅ stop timer
    end_time.record()
    torch.cuda.synchronize()

    # ✅ compute total time in seconds
    total_time_ms = start_time.elapsed_time(end_time)  # ms
    total_time_s = total_time_ms / 1000.0

    fps = total_images / total_time_s
    print(f"\n[STATS] Processed {total_images} images in {total_time_s:.3f} sec")
    print(f"[STATS] FPS: {fps:.2f}")

    # ✅ Save predictions/GT
    results = {"preds": all_preds, "targets": all_targets}
    with open("predictions_gt.pkl", "wb") as f:
        pickle.dump(results, f)

    print("[INFO] Saved predictions & ground-truth → predictions_gt.pkl")
    return top1.avg, objs.avg


if __name__ == "__main__":
    main()
