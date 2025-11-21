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
parser.add_argument(
    "--model_path_2", type=str, default=None, help="path to second pretrained model"
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
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--arch", type=str, default=None, help="which architecture to use")
args = parser.parse_args()

# Logging
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


# ---------------------------------------------------------------------
#   TWO-MODEL SOFT VOTING INFERENCE
# ---------------------------------------------------------------------
def infer_voting(test_queue, model1, model2, criterion):
    model1.eval()
    model2.eval()

    objs = ut.AvgrageMeter()
    top1 = ut.AvgrageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for step, (input, target) in enumerate(test_queue):
            input = input.cuda()
            target = target.cuda()

            # Forward pass
            logits1, _ = model1(input)
            logits2, _ = model2(input)

            # SOFT VOTING
            logits = (logits1 + logits2) / 2.0

            # Predictions
            preds = torch.argmax(logits, dim=1)

            # Save predictions + targets
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

            # Loss and accuracy
            loss = criterion(logits, target)
            prec1, _ = ut.accuracy(logits, target, topk=(1, 2))
            n = input.size(0)

            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)

            if step % args.report_freq == 0:
                logging.info("test %03d %e %f", step, objs.avg, top1.avg)

    print(f"\n[VOTING] Final Accuracy: {top1.avg:.2f} | Loss: {objs.avg:.4f}")

    # --------------------------------------------------------
    # SAVE PREDS + TARGETS INTO PICKLE
    # --------------------------------------------------------
    out_path = os.path.join(args.dir if args.dir else ".", "voting_preds_targets.pkl")

    with open(out_path, "wb") as f:
        pickle.dump({"preds": all_preds, "targets": all_targets}, f)

    print(f"[INFO] Saved predictions + targets → {out_path}")

    return top1.avg, objs.avg


# ---------------------------------------------------------------------


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

    # ------------------ MODEL 1 ------------------
    model1 = Network(
        args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype
    )
    ut.load(model1, args.model_path, args.gpu)
    model1 = model1.cuda()

    # ------------------ MODEL 2 ------------------
    if args.model_path_2 is None:
        raise ValueError("You must provide --model_path_2 for voting")

    model2 = Network(
        args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype
    )
    ut.load(model2, args.model_path_2, args.gpu)
    model2 = model2.cuda()

    logging.info("param size MODEL 1 = %fMB", ut.count_parameters_in_MB(model1))
    logging.info("param size MODEL 2 = %fMB", ut.count_parameters_in_MB(model2))

    criterion = nn.CrossEntropyLoss().cuda()

    # Dataset
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

    # Disable stochastic depth
    model1.drop_path_prob = 0.0
    model2.drop_path_prob = 0.0

    # Run ensemble voting
    test_acc, test_loss = infer_voting(test_queue, model1, model2, criterion)

    logging.info("FINAL VOTING test_acc %f", test_acc)
    print(f"FINAL VOTING test_acc {test_acc}")


if __name__ == "__main__":
    main()
