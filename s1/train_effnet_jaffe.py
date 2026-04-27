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
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from codecarbon import EmissionsTracker
import gc
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, Subset

# Bypass de módulos externos que podrían faltar o causar error
from types import ModuleType
fake_genotypes = ModuleType('genotypes')
sys.modules['genotypes'] = fake_genotypes
import ut 

parser = argparse.ArgumentParser("efficientnet_b0_train")
parser.add_argument("--data_dir", type=str, required=True, help="path of data")
parser.add_argument("--dir", type=str, default=".", help="directory to save results")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=3e-4)
parser.add_argument("--report_freq", type=float, default=50)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--save", type=str, default="EFFNET_EXP")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--grad_clip", type=float, default=5)
parser.add_argument("--num_classes", type=int, default=7)
parser.add_argument("--finetune", type=lambda x: x.lower() == "true", default=True)
args = parser.parse_args()

# Setup de carpetas
subject = args.data_dir.rstrip("/").split("/")[-1]
args.save = f"eval-{args.save}-{subject}-{args.seed}-{args.epochs}"
args.save = os.path.join(args.dir, args.save)
ut.create_exp_dir(args.save)

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format="%(asctime)s %(message)s", datefmt="%m/%d %I:%M:%S %p"
)
fh = logging.FileHandler(os.path.join(args.save, "train_log.txt"))
logging.getLogger().addHandler(fh)

writer = SummaryWriter(os.path.join(args.save, "runs"))

def main():
    if not torch.cuda.is_available():
        sys.exit(1)

    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    # --- Modelo EfficientNet-B0 ---
    logging.info("Cargando EfficientNet-B0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, args.num_classes)
    model = model.cuda()

    # Cargar pesos si existe fine-tune previo
    best_model_path = os.path.join(args.dir, "best_weights.pt")
    if args.finetune and os.path.exists(best_model_path):
        logging.info(f"Cargando pesos previos de {best_model_path}")
        ut.load(model, best_model_path)

    logging.info("param size = %fMB", ut.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # --- Datasets con Resize a 224 ---
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        train_data = dset.ImageFolder(os.path.join(args.data_dir, "train"), train_transform)
        valid_data = dset.ImageFolder(os.path.join(args.data_dir, "val"), valid_transform)
    except FileNotFoundError:
        logging.warning("Split manual de carpeta train...")
        full_dataset = dset.ImageFolder(os.path.join(args.data_dir, "train"), train_transform)
        train_size = int(0.8 * len(full_dataset))
        indices = list(range(len(full_dataset)))
        train_idx, val_idx = indices[:train_size], indices[train_size:]
        train_data = Subset(full_dataset, train_idx)
        # Aplicamos el transform de validación manualmente al subset si es posible
        valid_data = Subset(dset.ImageFolder(os.path.join(args.data_dir, "train"), valid_transform), val_idx)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2, worker_init_fn=seed_worker)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2, worker_init_fn=seed_worker)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    best_acc_top1 = 0.0
    tracker = EmissionsTracker(project_name="effnet_train", output_dir=args.save)
    tracker.start()

    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs} - LR: {scheduler.get_last_lr()[0]:e}")
        
        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        
        scheduler.step()

        writer.add_scalar("train_acc", train_acc, epoch + 1)
        writer.add_scalar("valid_acc", valid_acc, epoch + 1)

        if valid_acc > best_acc_top1:
            best_acc_top1 = valid_acc
            ut.save(model, os.path.join(args.save, "best_weights.pt"))
            logging.info(f"Nuevo mejor modelo: {best_acc_top1:.4f}")

        ut.save(model, os.path.join(args.save, "weights.pt"))
        logging.info(f"Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f}")

    tracker.stop()
    logging.info(f"Mejor Accuracy final: {best_acc_top1:.4f}")

def train(train_queue, model, criterion, optimizer):
    objs, top1 = ut.AvgrageMeter(), ut.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input, target = input.cuda(), target.cuda()
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, _ = ut.accuracy(logits, target, topk=(1, 2))
        objs.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
    return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
    objs, top1 = ut.AvgrageMeter(), ut.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for input, target in valid_queue:
            input, target = input.cuda(), target.cuda()
            logits = model(input)
            loss = criterion(logits, target)
            prec1, _ = ut.accuracy(logits, target, topk=(1, 2))
            objs.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
    return top1.avg, objs.avg

if __name__ == "__main__":
    main()
