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
    # model.load_state_dict(torch.load(args.model_path)['state_dict'], strict = False)
    model = model.cuda()

    logging.info("param size = %fMB", ut.count_parameters_in_MB(model))
    print("param size = %fMB", ut.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

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
    # model.drop_path_prob = args.drop_path_prob
    test_acc, test_obj, fps = infer_n_times(test_queue, model, criterion)
    logging.info("test_acc %f", test_acc)
    print(f"test_acc {test_acc}")


# def infer(test_queue, model, criterion):
#     objs = ut.AvgrageMeter()
#     top1 = ut.AvgrageMeter()
#     top5 = ut.AvgrageMeter()
#     model.eval()

#     with torch.no_grad():
#         for step, (input, target) in enumerate(test_queue):
#             input = Variable(input, volatile=True).cuda()
#             target = Variable(target, volatile=True).cuda()

#             logits, _ = model(input)
#             loss = criterion(logits, target)

#             prec1, prec5 = ut.accuracy(logits, target, topk=(1, 2))
#             n = input.size(0)
#             objs.update(loss.data.item(), n)
#             top1.update(prec1.data.item(), n)
#             top5.update(prec5.data.item(), n)

#             if step % args.report_freq == 0:
#                 logging.info("test %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

#     return top1.avg, objs.avg


# def infer(test_queue, model, criterion):
#     objs = ut.AvgrageMeter()
#     top1 = ut.AvgrageMeter()
#     top5 = ut.AvgrageMeter()
#     model.eval()

#     total_images = 0
#     total_time_ms = 0.0  # milliseconds

#     with torch.no_grad():
#         for step, (input, target) in enumerate(test_queue):

#             input = input.cuda()
#             target = target.cuda()

#             # ----- CUDA Timing -----
#             start_time = torch.cuda.Event(enable_timing=True)
#             end_time = torch.cuda.Event(enable_timing=True)

#             torch.cuda.synchronize()
#             start_time.record()

#             logits, _ = model(input)

#             end_time.record()
#             torch.cuda.synchronize()
#             # -----------------------

#             batch_time = start_time.elapsed_time(end_time)  # ms
#             total_time_ms += batch_time
#             total_images += input.size(0)

#             loss = criterion(logits, target)
#             prec1, prec5 = ut.accuracy(logits, target, topk=(1, 2))
#             n = input.size(0)

#             objs.update(loss.item(), n)
#             top1.update(prec1.item(), n)
#             top5.update(prec5.item(), n)

#             if step % args.report_freq == 0:
#                 logging.info("test %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

#     # Compute FPS
#     total_time_sec = total_time_ms / 1000.0
#     fps = total_images / total_time_sec if total_time_sec > 0 else 0

#     logging.info(f"[STATS] Processed {total_images} images in {total_time_sec:.3f} sec")
#     logging.info(f"[STATS] FPS: {fps:.2f}")
#     print(f"[STATS] Processed {total_images} images in {total_time_sec:.3f} sec")
#     print(f"[STATS] FPS: {fps:.2f}")

#     return top1.avg, objs.avg


def infer_n_times(test_queue, model, criterion, runs=5):
    model.eval()

    total_images = len(test_queue.dataset)

    avg_top1 = 0.0
    avg_loss = 0.0
    total_time_sec = 0.0

    with torch.no_grad():
        for r in range(runs):
            objs = ut.AvgrageMeter()
            top1 = ut.AvgrageMeter()
            top5 = ut.AvgrageMeter()

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            for input, target in test_queue:
                input = input.cuda()
                target = target.cuda()

                logits, _ = model(input)
                loss = criterion(logits, target)

                prec1, prec5 = ut.accuracy(logits, target, topk=(1, 2))
                n = input.size(0)

                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

            end.record()
            torch.cuda.synchronize()

            run_time_ms = start.elapsed_time(end)
            run_time_sec = run_time_ms / 1000.0
            run_fps = total_images / run_time_sec

            avg_top1 += top1.avg
            avg_loss += objs.avg
            total_time_sec += run_time_sec

            print(
                f"[RUN {r+1}] Acc@1: {top1.avg:.2f} | Loss: {objs.avg:.4f} | Time: {run_time_sec:.3f}s | FPS: {run_fps:.2f}"
            )

    # Final averages
    avg_top1 /= runs
    avg_loss /= runs
    avg_time = total_time_sec / runs
    avg_fps = total_images / avg_time

    print("--------------------------------------------------")
    print(f"[FINAL AVERAGE] over {runs} runs:")
    print(f"Top-1 Accuracy: {avg_top1:.2f}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Avg Time: {avg_time:.3f} sec")
    print(f"Avg FPS: {avg_fps:.2f}")
    print("--------------------------------------------------")

    return avg_top1, avg_loss, avg_fps


if __name__ == "__main__":
    main()
