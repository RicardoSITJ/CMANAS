"""Efficiency benchmark: FPS / latency / params / FLOPs at a given input resolution.

Rebuts reviewer R1.1: the 45 FPS at 48x48 is not just an artifact of the tiny resolution.
Measures the CMANAS-FER architecture AND standard lightweight backbones at the SAME resolution
(e.g. 48 and 224) with identical timing code, so efficiency is compared like-for-like.
No dataset and no training required (random-input forward passes).

Usage:
  python ./s1/bench_latency.py --model nas --img_size 224 --out efficiency_summary.csv
  python ./s1/bench_latency.py --model mobilenet_v3_small --img_size 224 --out efficiency_summary.csv
"""

import os
import sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, "./")

import csv
import argparse
import torch
import ut

parser = argparse.ArgumentParser("cmanas-fer-latency-bench")
parser.add_argument("--model", type=str, required=True,
                    help="'nas' (frozen CMANAS-FER genotype) or a torchvision backbone name")
parser.add_argument("--img_size", type=int, default=48)
parser.add_argument("--num_classes", type=int, default=7)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--iters", type=int, default=100)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--out", type=str, default="efficiency_summary.csv")
args = parser.parse_args()


def build_model():
    if args.model == "nas":
        from model import NetworkCIFAR as Network
        from frozen_genotype import CMANAS_FER, INIT_CHANNELS, LAYERS
        m = Network(INIT_CHANNELS, args.num_classes, LAYERS, False, CMANAS_FER)
        m.drop_path_prob = 0.0
        return m
    else:
        from baseline_models import build_baseline
        return build_baseline(args.model, args.num_classes, pretrained=False)


def measure_flops(model, h, w):
    x = torch.randn(1, 3, h, w).cuda()
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return 2.0 * macs / 1e9  # GFLOPs (2*MACs)
    except Exception as e:
        print(f"[WARN] FLOPs unavailable ({e})")
        return float("nan")


def main():
    if not torch.cuda.is_available():
        print("No GPU available")
        sys.exit(1)
    torch.cuda.set_device(args.gpu)

    model = build_model().cuda().eval()
    params = ut.count_parameters_in_MB(model) * 1e6  # count (excludes 'auxiliary')

    gflops = measure_flops(model, args.img_size, args.img_size)

    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size).cuda()
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            model(x)
        end.record()
        torch.cuda.synchronize()
    elapsed_s = start.elapsed_time(end) / 1000.0
    n_images = args.batch_size * args.iters
    fps = n_images / elapsed_s
    latency_ms = 1000.0 * elapsed_s / n_images  # per image, at this batch size

    row = {
        "model": args.model,
        "img_size": args.img_size,
        "params": int(params),
        "gflops": round(gflops, 4) if gflops == gflops else "nan",
        "fps": round(fps, 2),
        "latency_ms_per_img": round(latency_ms, 4),
        "batch_size": args.batch_size,
    }
    print(row)

    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[INFO] appended to {args.out}")


if __name__ == "__main__":
    main()
