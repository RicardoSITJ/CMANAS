import os
import sys
import time
import glob
import numpy as np
import torch
import ut
import logging
import argparse
import torch.nn as nn
import random
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from codecarbon import EmissionsTracker
import gc
from procedures import seed_everything, seed_worker
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# Bypass de genotypes para evitar errores si el archivo no existe
from types import ModuleType
fake_genotypes = ModuleType('genotypes')
sys.modules['genotypes'] = fake_genotypes

parser = argparse.ArgumentParser("ckplus_effnet")
parser.add_argument("--data", type=str, default="../data", help="location of the data corpus")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.01, help="init learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--epochs", type=int, default=100, help="num of training epochs")
parser.add_argument("--save", type=str, default="EFFNET_EXP", help="experiment name")
parser.add_argument("--seed", type=int, default=4000, help="random seed")
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
args = parser.parse_args()

# Configuración de directorios
args.save = f"eval-{args.save}-{args.seed}-{args.epochs}"
ut.create_exp_dir(args.save)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
fh = logging.FileHandler(os.path.join(args.save, "train_log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 7
writer = SummaryWriter(os.path.join(args.save, "runs"))

def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    
    logging.info("gpu device = %d" % args.gpu)
    logging.info("args = %s", args)

    # --- Inicialización de EfficientNet-B0 ---
    logging.info("Instanciando EfficientNet-B0 preentrenada...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Ajustar cabezal para 7 clases (CK+)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, CIFAR_CLASSES)
    model = model.cuda()

    logging.info("param size = %fMB", ut.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # --- Transforms con Resize a 224 para EfficientNet ---
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

    train_data = dset.ImageFolder(os.path.join(args.data, "train"), train_transform)
    valid_data = dset.ImageFolder(os.path.join(args.data, "val"), valid_transform)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    best_acc_top1 = 0.0
    tracker = EmissionsTracker(project_name="train_effnet_ckplus", output_dir=args.save)
    tracker.start()

    for epoch in range(args.epochs):
        logging.info("[INFO] epoch (%d/%d) lr %e", epoch + 1, args.epochs, scheduler.get_last_lr()[0])
        
        # Train
        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info("[INFO] train_acc %f", train_acc)
        writer.add_scalar("train_acc", train_acc, epoch + 1)
        
        # Valid
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info("[INFO] valid_acc %f", valid_acc)
        writer.add_scalar("valid_acc", valid_acc, epoch + 1)

        scheduler.step()

        # --- Guardar el MEJOR modelo ---
        if valid_acc > best_acc_top1:
            best_acc_top1 = valid_acc
            logging.info(f"Nuevo récord de Accuracy: {best_acc_top1:.4f}. Guardando best_weights.pt")
            ut.save(model, os.path.join(args.save, "best_weights.pt"))
        
        # Guardar pesos actuales (última época)
        ut.save(model, os.path.join(args.save, "weights.pt"))

    tracker.stop()
    logging.info(f"Entrenamiento finalizado. Mejor Accuracy: {best_acc_top1:.4f}")

def train(train_queue, model, criterion, optimizer):
    objs = ut.AvgrageMeter()
    top1 = ut.AvgrageMeter()
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
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        if step % args.report_freq == 0:
            logging.info("train %03d loss: %e acc: %f", step, objs.avg, top1.avg)
    return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
    objs = ut.AvgrageMeter()
    top1 = ut.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input, target = input.cuda(), target.cuda()
            logits = model(input)
            loss = criterion(logits, target)
            prec1, _ = ut.accuracy(logits, target, topk=(1, 2))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
    return top1.avg, objs.avg

if __name__ == "__main__":
    main()
    writer.close()
