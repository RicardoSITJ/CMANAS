import sys
import os
import os.path as osp
import numpy as np
import torch
import ut
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.backends.cudnn as cudnn
import torchvision.models as models  # Added torchvision models
from torch.autograd import Variable

# from model import NetworkCIFAR as Network  # No longer needed for EfficientNet

parser = argparse.ArgumentParser("cifar")
parser.add_argument("--data", type=str, default="../data", help="location of the data corpus")
parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--model_path", type=str, default=None, help="path of pretrained weights (optional)")
parser.add_argument("--log_path", type=str, default=None, help="path of log file")
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
    torch.cuda.manual_seed(args.seed)

    logging.info("gpu device = %d" % args.gpu)
    logging.info("args = %s", args)

    # --- Initialize EfficientNet-B0 ---
    logging.info("Initializing EfficientNet-B0")
    # Set weights=None for training from scratch, or weights='DEFAULT' for ImageNet pre-training
    model = models.efficientnet_b0(weights=None) 
    
    # Adjust the final fully connected layer for 7 classes
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, CIFAR_CLASSES)

    # Load weights if path is provided
    if args.model_path:
        ut.load(model, args.model_path, args.gpu)
        logging.info(f"Loaded model from {args.model_path}")

    model = model.cuda()

    logging.info("param size = %fMB", ut.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().cuda()

    _, test_transform = ut._data_transforms_ckplus(args)
    folder_path = "/kaggle/working/CMANAS/datasets/ckplus_split/7_class"
    test_data = dset.ImageFolder(osp.join(folder_path, "test"), test_transform)

    test_queue = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4, # Increased for efficiency
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

            # EfficientNet returns only logits, unlike the previous DARTS model
            logits = model(input)
            loss = criterion(logits, target)

            # Note: topk=(1, 2) used because CIFAR_CLASSES is small. 
            # If classes < 5, topk=5 will crash.
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
