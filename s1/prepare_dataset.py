"""Deterministically prepare cross-dataset FER data into a uniform ImageFolder layout.

Produces  <out>/{train,val,test}/<emotion>/*  with the SAME 7 emotion class names across datasets
(anger, disgust, fear, happiness, neutral, sadness, surprise) so evaluation is identical everywhere
(see revision/EVAL-PROTOCOL.md). The prepared folders are meant to be uploaded as an OWN, versioned
Kaggle dataset (reproducibility: won't disappear/change, pin version + SHA256).

Determinism: fixed --seed for the stratified val split; sorted file lists; copy (not move).

Supported:
  --dataset fer2013plus  : src has fer2013plus/fer2013/{train,test}/<8 classes>/*.png (drops 'contempt')
  --dataset rafdb        : src has dataset/aligned/{train,test}/*_images/*.jpg + dataset/{train,test}_labels.csv
"""

import os
import csv
import glob
import random
import shutil
import argparse

# Unified 7-emotion class names (used by every dataset for a consistent evaluation).
CANON = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

# FER2013+ folder names -> canonical (drop 'contempt' to align with CK+/JAFFE/RAF-DB 7 classes).
FERPLUS_MAP = {
    "anger": "anger", "disgust": "disgust", "fear": "fear", "happiness": "happiness",
    "neutral": "neutral", "sadness": "sadness", "surprise": "surprise",
    # 'contempt' intentionally dropped
}

# RAF-DB numeric label -> canonical name (official RAF-DB basic-emotion mapping).
RAFDB_MAP = {
    1: "surprise", 2: "fear", 3: "disgust", 4: "happiness",
    5: "sadness", 6: "anger", 7: "neutral",
}


def _copy(files, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for src in files:
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))


def _stratified_val(train_by_class, val_frac, seed):
    """Split {class: [files]} into (train, val) deterministically, stratified per class."""
    rng = random.Random(seed)
    tr, va = {}, {}
    for cls, files in train_by_class.items():
        files = sorted(files)               # deterministic order
        rng.shuffle(files)                  # seed-fixed shuffle
        n_val = max(1, int(round(len(files) * val_frac))) if files else 0
        va[cls] = files[:n_val]
        tr[cls] = files[n_val:]
    return tr, va


def prepare_ferplus(src, out, val_frac, seed):
    root = os.path.join(src, "fer2013plus", "fer2013")
    # test: copy directly (official test split)
    test_by_class = {}
    for folder, canon in FERPLUS_MAP.items():
        test_by_class[canon] = sorted(glob.glob(os.path.join(root, "test", folder, "*.png")))
    # train -> stratified train/val
    train_src = {}
    for folder, canon in FERPLUS_MAP.items():
        train_src[canon] = sorted(glob.glob(os.path.join(root, "train", folder, "*.png")))
    tr, va = _stratified_val(train_src, val_frac, seed)
    return tr, va, test_by_class


def _rafdb_split(src, csv_name, img_subdir):
    labels = {}
    with open(os.path.join(src, "dataset", csv_name)) as f:
        for row in csv.DictReader(f):
            labels[row["image"]] = int(row["label"])
    by_class = {c: [] for c in CANON}
    img_dir = os.path.join(src, "dataset", "aligned", img_subdir)
    for img, lab in labels.items():
        p = os.path.join(img_dir, img)
        if os.path.exists(p):
            by_class[RAFDB_MAP[lab]].append(p)
    return by_class


def prepare_rafdb(src, out, val_frac, seed):
    test_by_class = _rafdb_split(src, "test_labels.csv", os.path.join("test", "test_images"))
    train_src = _rafdb_split(src, "train_labels.csv", os.path.join("train", "train_images"))
    tr, va = _stratified_val(train_src, val_frac, seed)
    return tr, va, test_by_class


def main():
    ap = argparse.ArgumentParser("prepare_dataset")
    ap.add_argument("--dataset", required=True, choices=["fer2013plus", "rafdb"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dataset == "fer2013plus":
        tr, va, te = prepare_ferplus(args.src, args.out, args.val_frac, args.seed)
    else:
        tr, va, te = prepare_rafdb(args.src, args.out, args.val_frac, args.seed)

    for split, by_class in [("train", tr), ("val", va), ("test", te)]:
        for cls, files in by_class.items():
            _copy(files, os.path.join(args.out, split, cls))

    # Report counts (fundamentar / manifest)
    print(f"[{args.dataset}] seed={args.seed} val_frac={args.val_frac}  out={args.out}")
    for split, by_class in [("train", tr), ("val", va), ("test", te)]:
        total = sum(len(v) for v in by_class.values())
        per = {c: len(by_class.get(c, [])) for c in CANON}
        print(f"  {split:5s} total={total:6d}  {per}")


if __name__ == "__main__":
    main()
