import os
import sys

# Must be set before any torch/cuda imports
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

sys.path.insert(0, "./")

import time
import glob
import random
import pickle
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision
from codecarbon import EmissionsTracker
import gc
from procedures import seed_everything, seed_worker
import ut
import visualize
import genotypes
from model import NetworkCIFAR as Network
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser("cifar10")
parser.add_argument("--data", type=str, default="../data")
parser.add_argument("--dir", type=str, default=None, help="location of population")
parser.add_argument("--genotype_dir", type=str, default=None, help="path of genotype")
parser.add_argument("--data_dir", type=str, default=None, help="path of data")
parser.add_argument("--batch_size", type=int, default=96)
parser.add_argument("--learning_rate", type=float, default=0.025)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=3e-4)
parser.add_argument("--report_freq", type=float, default=50)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--epochs", type=int, default=600)
parser.add_argument("--init_channels", type=int, default=36)
parser.add_argument("--layers", type=int, default=20)
parser.add_argument("--auxiliary", action="store_true", default=False)
parser.add_argument("--auxiliary_weight", type=float, default=0.4)
parser.add_argument("--cutout", action="store_true", default=False)
parser.add_argument("--cutout_length", type=int, default=16)
parser.add_argument("--drop_path_prob", type=float, default=0.2)
parser.add_argument("--save", type=str, default="EXP")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--arch", type=str, default="DARTS")
parser.add_argument("--grad_clip", type=float, default=5)
parser.add_argument(
    "--finetune",
    type=lambda x: x.lower() == "true",
    default=True,
    help="True/False: load best_weights.pt and continue training",
)
args = parser.parse_args()

if args.seed is None or args.seed < 0:
    args.seed = random.randint(1, 100000)

subject = args.data_dir.rstrip("/").split("/")[-1]
args.save = f"eval-{args.save}-{subject}-{args.seed}-{args.epochs}"
if args.dir is not None:
    args.save = os.path.join(args.dir, args.save)
ut.create_exp_dir(args.save)

# Logging setup
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "eval_log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info(
    f"[INFO] torch version: {torch.__version__}, torchvision version: {torchvision.__version__}"
)

CIFAR_CLASSES = 7
writer = SummaryWriter(os.path.join(args.save, "runs"))


def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    logging.info(f"Setting Global Seed: {args.seed}")
    seed_everything(args.seed)
    # This ensures the DataLoader shuffle is isolated from other random calls
    g = torch.Generator()
    g.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)
    logging.info(f"gpu device = {args.gpu}")
    logging.info(f"args = {args}")

    # Load genotype
    genotype_path = os.path.join(args.genotype_dir, "genotype.pickle")
    if not os.path.exists(genotype_path):
        logging.error(f"No genotype found at {genotype_path}")
        return
    with open(genotype_path, "rb") as f:
        genotype = pickle.load(f)
    # visualize.plot(genotype.normal, os.path.join(args.save, "normal"), False)
    # visualize.plot(genotype.reduce, os.path.join(args.save, "reduction"), False)
    logging.info(genotype)

    # Model setup
    model = Network(
        args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype
    ).cuda()
    logging.info("param size = %fMB", ut.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,  # Lower LR is safer for small datasets to prevent "jitter"
        weight_decay=0.1,  # Slightly higher weight decay helps prevent overfitting
        amsgrad=True,  # Set to True only if you see divergent behavior
    )

    # ============================
    #  LOAD FINE-TUNED WEIGHTS
    # ============================
    best_model_path = os.path.join(args.dir, "best_weights.pt")
    if args.finetune and os.path.exists(best_model_path):
        logging.info(f"[INFO] Loading fine-tuned weights from {best_model_path}")
        print(f"[INFO] Loading fine-tuned weights from {best_model_path}")
        ut.load(model, best_model_path)
    else:
        logging.warning(
            f"[WARN] No best_weights.pt found at {best_model_path}, starting from scratch."
        )

    # Dataset
    train_transform, valid_transform = ut._data_transforms_ckplus(args)
    folder_path = args.data_dir
    train_data = dset.ImageFolder(os.path.join(folder_path, "train"), train_transform)
    valid_data = dset.ImageFolder(os.path.join(folder_path, "val"), valid_transform)
    logging.info(
        f"[INFO] len(train_data): {len(train_data)}, len(valid_data): {len(valid_data)}"
    )

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        generator=g,
        worker_init_fn=seed_worker,
    )
    valid_queue = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        generator=g,
        worker_init_fn=seed_worker,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=float(args.epochs), eta_min=1e-6
    )

    test_error = []
    best_acc_top1 = 0.0

    # ============================
    #  TRAIN / FINE-TUNE
    # ============================
    tracker = EmissionsTracker(
        project_name="train_finetune_jaffe",
        output_dir="carbon_logs",
        output_file="train_finetune_jaffe.csv",
    )
    tracker.start()
    for epoch in range(args.epochs):
        logging.info(
            f"[INFO] epoch ({epoch + 1}/{args.epochs}) lr {scheduler.get_last_lr()[0]:e}"
        )
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        epoch_start = time.time()
        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        # logging.info(
        #     f"[INFO] train_acc {train_acc:.4f} finished in {(time.time() - epoch_start) / 60:.2f} minutes"
        # )
        logging.info(
            f"[INFO] train_acc {train_acc.item():.4f} finished in {(time.time() - epoch_start) / 60:.2f} minutes"
        )

        writer.add_scalar("train_acc", train_acc, epoch + 1)
        writer.add_scalar("train_obj", train_obj, epoch + 1)
        scheduler.step()

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        # logging.info(f"[INFO] valid_acc {valid_acc:.4f}")
        logging.info(f"[INFO] valid_acc {valid_acc.item():.4f}")
        writer.add_scalar("valid_acc", valid_acc, epoch + 1)
        writer.add_scalar("valid_obj", valid_obj, epoch + 1)
        writer.add_scalar("test_error", 100 - valid_acc, epoch + 1)

        ut.save(model, os.path.join(args.save, "weights.pt"))
        test_error.append(100 - valid_acc)

        if valid_acc > best_acc_top1:
            ut.save(model, os.path.join(args.save, "best_weights.pt"))
            best_acc_top1 = valid_acc
            # logging.info(f"[INFO] New best model saved with acc {best_acc_top1:.4f}")
            logging.info(
                f"[INFO] New best model saved with acc {best_acc_top1.item():.4f}"
            )

        writer.add_scalar("best_acc", best_acc_top1, epoch + 1)
        writer.add_scalar("best_test_error", 100 - best_acc_top1, epoch + 1)
        logging.info(
            f"[INFO] Epoch finished in {(time.time() - epoch_start) / 60:.2f} minutes"
        )
        logging.info("=" * 100)

    emissions = tracker.stop()
    logging.info(f"[INFO] Estimated emissions (kg CO₂): {emissions:.5f}")
    logging.info(
        f"best_acc: {best_acc_top1.item():.4f}, valid_acc: {valid_acc.item():.4f}"
    )
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
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
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
    logging.info(
        f"[INFO] Training/Fine-tuning finished in {(time.time() - start_time) / 3600:.2f} hours"
    )
    writer.close()
    # Delete model and data loaders if they exist
    variables_to_delete = [
        "model",
        "train_queue",
        "valid_queue",
        "optimizer",
        "criterion",
    ]
    gl = globals()
    for var in variables_to_delete:
        if var in gl:
            del gl[var]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
