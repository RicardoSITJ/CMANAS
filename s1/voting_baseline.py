"""Evaluate baseline backbones with the SAME metric code as voting_n.py.

Identical to voting_n.py (soft-voting accuracy, CUDA-event FPS timing, theoretical params,
on-disk size, same output pickle schema) EXCEPT the model is a torchvision baseline backbone
instead of NetworkCIFAR(genotype). This guarantees the baseline metrics are byte-for-byte
comparable to the NAS model's (reviewers R1.4 / R3 / R4.Q2).
"""

import os
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, "./")

import os.path as osp
import numpy as np
import torch
import ut
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset

from procedures import seed_everything
import pickle
from baseline_models import build_baseline

parser = argparse.ArgumentParser("cmanas-fer-baseline-voting")
parser.add_argument("--backbone", type=str, required=True,
                    help="mobilenet_v3_small | mobilenet_v3_large | efficientnet_b0 | resnet18")
parser.add_argument("--data", type=str, default="../data")
parser.add_argument("--dir", type=str, default=None)
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--report_freq", type=float, default=50)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--log_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_classes", type=int, default=7)
parser.add_argument("--model_paths", nargs="+", required=True,
                    help="List of fine-tuned baseline model paths for N-model voting")
args = parser.parse_args()

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
if args.log_path is not None:
    fh = logging.FileHandler(os.path.join(args.log_path, "evaluate.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def infer_voting(test_queue, models, criterion):
    for m in models:
        m.eval()

    objs = ut.AvgrageMeter()
    top1 = ut.AvgrageMeter()

    all_preds = []
    all_targets = []

    total_images = len(test_queue.dataset)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    with torch.no_grad():
        for step, (inp, target) in enumerate(test_queue):
            inp = inp.cuda()
            target = target.cuda()

            logits_list = []
            for model in models:
                logits, _ = model(inp)
                logits_list.append(logits)

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

    end_event.record()
    torch.cuda.synchronize()

    run_time_ms = start_event.elapsed_time(end_event)
    run_time_sec = run_time_ms / 1000.0
    fps = total_images / run_time_sec

    total_params_mb = sum(ut.count_parameters_in_MB(m) for m in models)

    total_disk_mb = 0.0
    for path in args.model_paths:
        if os.path.exists(path):
            total_disk_mb += os.path.getsize(path) / (1024.0 * 1024.0)

    print(f"\n[VOTING] Final Accuracy: {top1.avg:.2f} | Loss: {objs.avg:.4f}")
    print(f"[TIMING] Time: {run_time_sec:.3f}s | FPS: {fps:.2f}")
    print(f"[SIZE]   Params: {total_params_mb:.2f} MB | Disk: {total_disk_mb:.2f} MB")

    run_ids = [
        os.path.basename(os.path.dirname(p)).replace("eval-EXP-", "")
        for p in args.model_paths
    ]
    out_name = f"voting_{len(models)}_preds_targets_" + "_".join(run_ids) + ".pkl"
    out_path = os.path.join(args.dir if args.dir else ".", out_name)

    save_dict = {
        "preds": all_preds,
        "targets": all_targets,
        "fps": fps,
        "total_time_sec": run_time_sec,
        "acc": top1.avg,
        "loss": objs.avg,
        "params_mb": total_params_mb,
        "disk_mb": total_disk_mb,
        "backbone": args.backbone,
    }
    with open(out_path, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"[INFO] Saved predictions + stats -> {out_path}")

    return top1.avg, objs.avg, fps


def main():
    if not torch.cuda.is_available():
        logging.info("No GPU available")
        sys.exit(1)

    logging.info(f"Setting Global Seed: {args.seed}")
    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    models = []
    for model_path in args.model_paths:
        m = build_baseline(args.backbone, args.num_classes, pretrained=False)
        ut.load(m, model_path, args.gpu)
        m = m.cuda()
        models.append(m)
    logging.info(f"Loaded {len(models)} '{args.backbone}' models for voting")
    for i, m in enumerate(models):
        logging.info(f"Model {i} params = {ut.count_parameters_in_MB(m)} MB")

    criterion = nn.CrossEntropyLoss().cuda()

    _, test_transform = ut._data_transforms_ckplus(args)
    test_data = dset.ImageFolder(osp.join(args.data_dir, "test"), test_transform)
    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=0, generator=g)

    test_acc, test_loss, test_fps = infer_voting(test_queue, models, criterion)
    logging.info("FINAL VOTING ACC %f", test_acc)
    print(f"FINAL VOTING ACC {test_acc}")
    print(f"FINAL FPS {test_fps:.2f}")


if __name__ == "__main__":
    main()
