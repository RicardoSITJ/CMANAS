"""Train/fine-tune a lightweight baseline backbone under the SAME protocol as CMANAS-FER.

Mirrors `train_finetune_jaffe.py` exactly (transforms, seeds, optimizer, LR schedule,
label smoothing, grad clip, dataloader seeding, per-epoch best-on-val checkpointing) so the
only variable vs the NAS model is the architecture. Answers reviewers R1.4 / R3 / R4.Q2.

Two-stage pipeline, matching ours (supernet/retrain on CK+ -> fine-tune on JAFFE under LOSO):
  * --stage ckplus : ImageNet-pretrained backbone -> train on CK+ (train/val) -> best_weights.pt
  * --stage jaffe  : load CK+ baseline weights (--init_weights) -> fine-tune on one JAFFE
                     LOSO fold (train/val) -> best_weights.pt (per subject x seed)

Evaluate with `voting_baseline.py` (same metric code as voting_n.py: acc/FPS/params/disk).
"""

import os
import sys

# Must be set before any torch/cuda imports (parity with train_finetune_jaffe.py)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

sys.path.insert(0, "./")

import time
import random
import pickle
import logging
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dset
from codecarbon import EmissionsTracker
import gc
from procedures import seed_everything, seed_worker
import ut
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from baseline_models import build_baseline

parser = argparse.ArgumentParser("cmanas-fer-baseline")
parser.add_argument("--backbone", type=str, required=True,
                    help="mobilenet_v3_small | mobilenet_v3_large | efficientnet_b0 | resnet18")
parser.add_argument("--stage", type=str, default="jaffe", choices=["ckplus", "jaffe"],
                    help="ckplus: pretrain on CK+ | jaffe: fine-tune on a JAFFE LOSO fold")
parser.add_argument("--data", type=str, default="../data")
parser.add_argument("--dir", type=str, default=None, help="parent output dir")
parser.add_argument("--data_dir", type=str, default=None, help="path of data (CK+ root or JAFFE fold)")
parser.add_argument("--init_weights", type=str, default=None,
                    help="path to CK+ baseline best_weights.pt to load before the JAFFE stage")
parser.add_argument("--pretrained", type=lambda x: x.lower() == "true", default=True,
                    help="load ImageNet-pretrained backbone weights (ckplus stage)")
parser.add_argument("--num_classes", type=int, default=7)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=0.025)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=3e-4)
parser.add_argument("--report_freq", type=float, default=50)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--cutout", action="store_true", default=False)
parser.add_argument("--cutout_length", type=int, default=16)
parser.add_argument("--save", type=str, default="EXP")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--grad_clip", type=float, default=5)
args = parser.parse_args()

if args.seed is None or args.seed < 0:
    args.seed = random.randint(1, 100000)

if args.stage == "jaffe":
    subject = args.data_dir.rstrip("/").split("/")[-1]
    args.save = f"eval-{args.save}-{args.backbone}-{subject}-{args.seed}-{args.epochs}"
else:
    args.save = f"ckplus-{args.save}-{args.backbone}-{args.seed}-{args.epochs}"
if args.dir is not None:
    args.save = os.path.join(args.dir, args.save)
ut.create_exp_dir(args.save)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format,
                    datefmt="%m/%d %I:%M:%S %p")
fh = logging.FileHandler(os.path.join(args.save, "eval_log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info(f"[INFO] torch {torch.__version__}, torchvision {torchvision.__version__}")

writer = SummaryWriter(os.path.join(args.save, "runs"))


def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    logging.info(f"Setting Global Seed: {args.seed}")
    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    logging.info(f"gpu device = {args.gpu}")
    logging.info(f"args = {args}")

    # Model: torchvision backbone (ImageNet-pretrained for ckplus stage), wrapped to the NAS interface
    model = build_baseline(args.backbone, args.num_classes, pretrained=args.pretrained).cuda()
    logging.info("backbone = %s | param size = %fMB", args.backbone,
                 ut.count_parameters_in_MB(model))

    # Load CK+ baseline init before the JAFFE stage (mirrors CK+ -> JAFFE transfer)
    if args.stage == "jaffe":
        if args.init_weights and os.path.exists(args.init_weights):
            logging.info(f"[INFO] Loading CK+ baseline weights from {args.init_weights}")
            ut.load(model, args.init_weights, args.gpu)
        else:
            logging.warning(f"[WARN] init_weights not found ({args.init_weights}); "
                            f"fine-tuning from ImageNet init.")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    train_transform, valid_transform = ut._data_transforms_ckplus(args)
    train_data = dset.ImageFolder(os.path.join(args.data_dir, "train"), train_transform)
    valid_data = dset.ImageFolder(os.path.join(args.data_dir, "val"), valid_transform)
    logging.info(f"[INFO] len(train_data): {len(train_data)}, len(valid_data): {len(valid_data)}")

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True,
        num_workers=0, generator=g, worker_init_fn=seed_worker)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True,
        num_workers=0, generator=g, worker_init_fn=seed_worker)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    test_error = []
    best_acc_top1 = 0.0

    tracker = EmissionsTracker(
        project_name=f"train_baseline_{args.backbone}_{args.stage}",
        output_dir="carbon_logs",
        output_file=f"train_baseline_{args.backbone}_{args.stage}.csv")
    tracker.start()
    for epoch in range(args.epochs):
        logging.info(f"[INFO] epoch ({epoch + 1}/{args.epochs}) lr {scheduler.get_last_lr()[0]:e}")
        epoch_start = time.time()
        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info(f"[INFO] train_acc {train_acc.item():.4f} finished in "
                     f"{(time.time() - epoch_start) / 60:.2f} minutes")
        writer.add_scalar("train_acc", train_acc, epoch + 1)
        writer.add_scalar("train_obj", train_obj, epoch + 1)
        scheduler.step()

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info(f"[INFO] valid_acc {valid_acc.item():.4f}")
        writer.add_scalar("valid_acc", valid_acc, epoch + 1)
        writer.add_scalar("valid_obj", valid_obj, epoch + 1)
        writer.add_scalar("test_error", 100 - valid_acc, epoch + 1)

        ut.save(model, os.path.join(args.save, "weights.pt"))
        test_error.append(100 - valid_acc)

        if valid_acc > best_acc_top1:
            ut.save(model, os.path.join(args.save, "best_weights.pt"))
            best_acc_top1 = valid_acc
            logging.info(f"[INFO] New best model saved with acc {best_acc_top1.item():.4f}")
        writer.add_scalar("best_acc", best_acc_top1, epoch + 1)
        logging.info(f"[INFO] Epoch finished in {(time.time() - epoch_start) / 60:.2f} minutes")
        logging.info("=" * 100)

    emissions = tracker.stop()
    logging.info(f"[INFO] Estimated emissions (kg CO2): {emissions:.5f}")
    logging.info(f"best_acc: {best_acc_top1.item():.4f}, valid_acc: {valid_acc.item():.4f}")
    print(f"best_acc: {best_acc_top1.item():.4f}, valid_acc: {valid_acc.item():.4f}")

    with open(os.path.join(args.save, "test_error.pickle"), "wb") as f:
        pickle.dump(test_error, f)


def train(train_queue, model, criterion, optimizer):
    objs = ut.AvgrageMeter()
    top1 = ut.AvgrageMeter()
    top5 = ut.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda()
        optimizer.zero_grad()
        logits, _ = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1, prec5 = ut.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
        if step % args.report_freq == 0:
            logging.info("train %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)
    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = ut.AvgrageMeter()
    top1 = ut.AvgrageMeter()
    top5 = ut.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            logits, _ = model(input)
            loss = criterion(logits, target)
            prec1, prec5 = ut.accuracy(logits, target, topk=(1, 2))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
            if step % args.report_freq == 0:
                logging.info("valid %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)
    return top1.avg, objs.avg


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_time = time.time()
    main()
    logging.info(f"[INFO] Finished in {(time.time() - start_time) / 3600:.2f} hours")
    writer.close()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
