import os
import sys

# Must be set before any torch/cuda imports
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

sys.path.insert(0, "./")

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
from procedures import seed_everything

parser = argparse.ArgumentParser("cifar")
parser.add_argument("--data", type=str, default="../data")
parser.add_argument("--dir", type=str, default=None)
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--report_freq", type=float, default=50)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--init_channels", type=int, default=36)
parser.add_argument("--layers", type=int, default=20)
parser.add_argument("--arch", type=str, default=None)
parser.add_argument("--log_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--auxiliary", action="store_true", default=False)
parser.add_argument("--cutout", action="store_true", default=False)
parser.add_argument("--cutout_length", type=int, default=16)
parser.add_argument("--drop_path_prob", type=float, default=0.2)

# ⭐ NEW: list of model paths
parser.add_argument(
    "--model_paths",
    nargs="+",
    required=True,
    help="List of pretrained model paths for N-model voting",
)

args = parser.parse_args()

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
if args.log_path is not None:
    fh = logging.FileHandler(os.path.join(args.log_path, "evaluate.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 7


# ---------------------------------------------------------------------
#   N-MODEL SOFT VOTING
# ---------------------------------------------------------------------
def infer_voting(test_queue, models, criterion):

    for m in models:
        m.eval()

    objs = ut.AvgrageMeter()
    top1 = ut.AvgrageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for step, (inp, target) in enumerate(test_queue):

            inp = inp.cuda()
            target = target.cuda()

            # Collect logits from all N models
            logits_list = []
            for model in models:
                logits, _ = model(inp)
                logits_list.append(logits)

            # ⭐ SOFT VOTING: average all logits
            logits = sum(logits_list) / float(len(logits_list))

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

            loss = criterion(logits, target)
            prec1, _ = ut.accuracy(logits, target, topk=(1, 2))

            objs.update(loss.item(), inp.size(0))
            top1.update(prec1.item(), inp.size(0))

            if step % args.report_freq == 0:
                logging.info("test %03d %e %f", step, objs.avg, top1.avg)

    print(f"\n[VOTING] Final Accuracy: {top1.avg:.2f} | Loss: {objs.avg:.4f}")

    # -------------------------
    # Save preds + targets
    # -------------------------
    run_ids = [
        os.path.basename(os.path.dirname(p)).replace("eval-EXP-", "")
        for p in args.model_paths
    ]

    out_name = f"voting_{len(models)}_preds_targets_" + "_".join(run_ids) + ".pkl"
    out_path = os.path.join(args.dir if args.dir else ".", out_name)

    with open(out_path, "wb") as f:
        pickle.dump({"preds": all_preds, "targets": all_targets}, f)

    print(f"[INFO] Saved predictions + targets → {out_path}")

    return top1.avg, objs.avg


# ---------------------------------------------------------------------
def main():

    if not torch.cuda.is_available():
        logging.info("No GPU available")
        sys.exit(1)

    logging.info(f"Setting Global Seed: {args.seed}")
    seed_everything(args.seed)
    # This ensures the DataLoader shuffle is isolated from other random calls
    g = torch.Generator()
    g.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    # Load genotype
    if args.arch is not None:
        genotype = eval("genotypes.%s" % args.arch)
    else:
        with open(os.path.join(args.dir, "genotype.pickle"), "rb") as f:
            genotype = pickle.load(f)

    print("---------Genotype---------")
    logging.info(genotype)
    print("--------------------------")

    # ---------------------------------------------------------
    # LOAD N MODELS
    # ---------------------------------------------------------
    models = []
    for model_path in args.model_paths:
        m = Network(
            args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype
        )
        ut.load(m, model_path, args.gpu)
        m = m.cuda()
        m.drop_path_prob = 0.0
        models.append(m)

    logging.info(f"Loaded {len(models)} models for voting")

    for i, m in enumerate(models):
        logging.info(f"Model {i} params = {ut.count_parameters_in_MB(m)} MB")

    criterion = nn.CrossEntropyLoss().cuda()

    # Dataset
    _, test_transform = ut._data_transforms_ckplus(args)
    test_data = dset.ImageFolder(osp.join(args.data_dir, "test"), test_transform)

    test_queue = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        generator=g,
    )

    # Run ensemble voting
    test_acc, test_loss = infer_voting(test_queue, models, criterion)

    logging.info("FINAL VOTING ACC %f", test_acc)
    print(f"FINAL VOTING ACC {test_acc}")


if __name__ == "__main__":
    main()
