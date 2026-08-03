"""
Resolve train / val / test data sources for every stage of the pipeline.

Two modes, chosen automatically from the CLI paths you pass:

1. OFFICIAL splits (recommended, reproduces the paper's numbers)
   Provide the official Validation and/or Test archives. Each split reads its
   own images / concept JSONs / label CSV. This is what makes your macro-F1
   directly comparable to Table 1 (the paper reports on the 600-image Test v2).

2. INTERNAL stratified split (fallback)
   If no official val/test archives are given, we carve a stratified
   train/val/test split out of your 2000 training images. This yields a
   *legitimate* held-out metric but on a different test set than the paper, so
   the absolute number will differ from 71.5.

Every stage (phase1/2/3 + evaluate) imports `resolve_splits(args)` so the exact
same partition is used throughout a run (seed-controlled).
"""
from dataclasses import dataclass
from typing import Optional, List

from dataset import stratified_split


@dataclass
class SplitSpec:
    name: str
    img_source: str
    part2_source: str
    part3_csv: str
    ids: Optional[List[str]] = None  # None -> use everything available in the source


def add_data_args(parser):
    """Shared CLI args for locating the ISIC-2017 splits."""
    # Training source (what you already have).
    parser.add_argument("--img_dir", required=True, help="Training images (dir or .zip)")
    parser.add_argument("--task2_dir", required=True, help="Training concept JSONs (dir or .zip)")
    parser.add_argument("--task3_csv", required=True, help="Training diagnosis label CSV")
    # Optional official validation split.
    parser.add_argument("--val_img", default=None, help="Official validation images (dir or .zip)")
    parser.add_argument("--val_task2", default=None, help="Official validation concept JSONs")
    parser.add_argument("--val_task3", default=None, help="Official validation label CSV")
    # Optional official test split (the paper's reporting set).
    parser.add_argument("--test_img", default=None, help="Official test images (dir or .zip)")
    parser.add_argument("--test_task2", default=None, help="Official test concept JSONs")
    parser.add_argument("--test_task3", default=None, help="Official test label CSV")
    # Fallback internal-split fractions.
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)


def resolve_splits(args):
    """Return dict {'train':SplitSpec,'val':SplitSpec or None,'test':SplitSpec or None}."""
    have_val = args.val_img and args.val_task2 and args.val_task3
    have_test = args.test_img and args.test_task2 and args.test_task3

    train_src = dict(img=args.img_dir, p2=args.task2_dir, p3=args.task3_csv)

    if have_val or have_test:
        # OFFICIAL mode. Training uses all training ids; val/test read their own archives.
        train = SplitSpec("train", train_src["img"], train_src["p2"], train_src["p3"], ids=None)
        val = (SplitSpec("val", args.val_img, args.val_task2, args.val_task3)
               if have_val else None)
        test = (SplitSpec("test", args.test_img, args.test_task2, args.test_task3)
                if have_test else None)
        mode = "official"
    else:
        # INTERNAL stratified split of the training set.
        tr, va, te = stratified_split(args.task3_csv, val_frac=args.val_frac,
                                      test_frac=args.test_frac, seed=args.split_seed)
        train = SplitSpec("train", train_src["img"], train_src["p2"], train_src["p3"], ids=tr)
        val = SplitSpec("val", train_src["img"], train_src["p2"], train_src["p3"], ids=va)
        test = SplitSpec("test", train_src["img"], train_src["p2"], train_src["p3"], ids=te)
        mode = "internal"

    print(f"[splits] mode = {mode}")
    return {"train": train, "val": val, "test": test, "mode": mode}
