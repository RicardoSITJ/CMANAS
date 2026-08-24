import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import pickle
import genotypes
import torch.nn.functional as F


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_jaffe7(args):
    CIFAR_MEAN = [x / 255 for x in [0.4369, 0.4369, 0.4369]]
    CIFAR_STD = [x / 255 for x in [0.2356, 0.2356, 0.2356]]

    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_ckplus(args):
    # 1. FIX: Use neutral Mean/Std.
    # CIFAR stats (trucks/frogs) hurt face model convergence.
    FACE_MEAN = [0.5, 0.5, 0.5]
    FACE_STD = [0.5, 0.5, 0.5]

    # 2. FIX: Increase resolution.
    # 32x32 is too small to see mouth/eye details. 48x48 is the standard minimum for FER.
    # If your model architecture specifically requires 32x32 input, change this back to 32.
    # IMG_SIZE = 48
    IMG_SIZE = 48

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # Optional: Forces model to focus on structure (smile shape), not skin tone.
            # Helps prevent overfitting to specific subjects.
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            # 3. FIX: Use RandomAffine instead of just Rotation.
            # This adds Shift (translate) and Zoom (scale) which helps when
            # the test subject's face isn't perfectly centered.
            transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
            ),
            # Reduced jitter slightly so we don't lose shadow details
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # RandomCrop with padding helps robustness
            transforms.RandomCrop(IMG_SIZE, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(FACE_MEAN, FACE_STD),
        ]
    )

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(num_output_channels=3),  # Match training channels
            transforms.ToTensor(),
            transforms.Normalize(FACE_MEAN, FACE_STD),
        ]
    )

    return train_transform, valid_transform


def _data_transforms_ckplus1(args):
    # 1. FIX: Use neutral Mean/Std.
    # CIFAR stats (trucks/frogs) hurt face model convergence.
    FACE_MEAN = [0.5, 0.5, 0.5]
    FACE_STD = [0.5, 0.5, 0.5]

    # 2. FIX: Increase resolution.
    # 32x32 is too small to see mouth/eye details. 48x48 is the standard minimum for FER.
    # If your model architecture specifically requires 32x32 input, change this back to 32.
    # IMG_SIZE = 48
    IMG_SIZE = 96

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # Optional: Forces model to focus on structure (smile shape), not skin tone.
            # Helps prevent overfitting to specific subjects.
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            # 3. FIX: Use RandomAffine instead of just Rotation.
            # This adds Shift (translate) and Zoom (scale) which helps when
            # the test subject's face isn't perfectly centered.
            transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
            ),
            # Reduced jitter slightly so we don't lose shadow details
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # RandomCrop with padding helps robustness
            transforms.RandomCrop(IMG_SIZE, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(FACE_MEAN, FACE_STD),
        ]
    )

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(num_output_channels=3),  # Match training channels
            transforms.ToTensor(),
            transforms.Normalize(FACE_MEAN, FACE_STD),
        ]
    )

    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return (
        np.sum(
            np.prod(v.size())
            for name, v in model.named_parameters()
            if "auxiliary" not in name
        )
        / 1e6
    )


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, "model_best.pth.tar")
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path, gpu=0):
    model.load_state_dict(
        torch.load(model_path, map_location="cuda:{}".format(gpu)), strict=False
    )


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = Variable(
            torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        )
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # makedirs: also create parent dirs (needed for nested output dirs)
    print("Experiment dir : {}".format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)
