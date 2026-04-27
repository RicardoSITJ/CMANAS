import sys
import os
from types import ModuleType

# --- STEP 1: Fix the 'genotypes' error without editing ut.py ---
# This creates a dummy module in memory so 'import ut' doesn't fail
fake_genotypes = ModuleType('genotypes')
sys.modules['genotypes'] = fake_genotypes

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn  # FIXED: torch, not torchvision
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

# Now we can safely import ut
import ut

import argparse
import logging

parser = argparse.ArgumentParser("cifar")
parser.add_argument("--data", type=str, default="../data", help="location of the data corpus")
parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--model_path", type=str, default=None, help="path of pretrained model")
parser.add_argument("--seed", type=int, default=0, help="random seed")
args = parser.parse_args()

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")

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
    
    logging.info("gpu device = %d" % args.gpu)

    # --- STEP 2: Initialize EfficientNet-B0 ---
    # We use weights=None because you are likely loading your own .pt file
    model = models.efficientnet_b0(weights=None)
    
    # Adjust the classifier for CK+ (7 classes)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, CIFAR_CLASSES)

    if args.model_path:
        # Using the utility load function from your project
        ut.load(model, args.model_path, args.gpu)
    
    model = model.cuda()

    logging.info("param size = %fMB", ut.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().cuda()

    # --- STEP 3: Handle the 48x48 resolution ---
    # We define the transform here to ensure it resizes to 224 for EfficientNet
    test_transform = transforms.Compose([
        transforms.Resize(48), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    folder_path = "/kaggle/working/CMANAS/datasets/ckplus_split/7_class"
    test_data = dset.ImageFolder(osp.join(folder_path, "test"), test_transform)

    test_queue = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info("test_acc %f", test_acc)
    print(f"test_acc {test_acc}")


def infer(test_queue, model, criterion):
    objs = ut.AvgrageMeter()
    top1 = ut.AvgrageMeter()
    top5 = ut.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(test_queue):
            input = input.cuda()
            target = target.cuda()

            # Standard EfficientNet returns a single tensor (logits)
            logits = model(input)
            loss = criterion(logits, target)

            # Using accuracy from your ut.py
            prec1, prec5 = ut.accuracy(logits, target, topk=(1, 2))
            
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info("test %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

if __name__ == "__main__":
    main()
