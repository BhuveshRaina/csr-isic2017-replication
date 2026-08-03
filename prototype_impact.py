"""
Per-prototype impact scan (doctor-in-the-loop safety map).

Motivation
----------
Atlas hit_count measures how often a prototype WINS the nearest-match competition
during Phase-2 assignment. It does NOT measure how much the Phase-3 task head
*depends* on that prototype's similarity score when classifying. We learned these
are different: pigment_network m=33 had only an 11.8% atlas share yet removing it
dropped macro-F1 by 10.85, because the task head had learned a strong
nevus-vs-melanoma weight on it.

So "safe to reject?" cannot be read off the atlas. This script measures it
directly: for every prototype, zero its column in the task head's effective
linear map (exact ablation -- same math as evaluate_curated --true_ablation) and
record the change in macro-F1 on a chosen split. The result is an honest safety
map the doctor UI can surface: large negative delta = costly to reject (the model
genuinely relies on it), near-zero or positive = safe / beneficial to reject
(a candidate shortcut).

Runs in a single forward pass over the split: we compute the raw similarity
scores once, then re-score 400 times in logit space (cheap matrix ops), so it is
fast even on CPU.

Usage (score on validation so we are not tuning on test):
  python prototype_impact.py \
    --weights checkpoints_revive/csr_network_final_best.pth \
    --atlas   checkpoints_revive/atlas.pt \
    --img_dir data/isic_224x224.zip \
    --task2_dir data/ISIC-2017_Training_Part2_GroundTruth.zip \
    --task3_csv data/ISIC-2017_Training_Part3_GroundTruth.csv \
    --val_img data/ISIC-2017_Validation_Data.zip \
    --val_task2 data/ISIC-2017_Validation_Part2_GroundTruth.zip \
    --val_task3 data/ISIC-2017_Validation_Part3_GroundTruth.csv \
    --split val --out checkpoints_revive/impact.pt
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import get_dataloader, CONCEPT_KEYS
from evaluate import build_csr_from_state
from models.concept_model import DEFAULT_BACKBONE
from splits import add_data_args, resolve_splits
from utils import seed_everything, resolve_device, amp_autocast


@torch.no_grad()
def collect_scores(net, loader, device):
    """One pass: raw similarity vectors s (N,400) and ground-truth labels."""
    S, gts = [], []
    for images, _, targets in tqdm(loader, desc="scoring"):
        images = images.to(device)
        with amp_autocast(device):
            _, s_flat, _ = net(images)          # (B, K*M) raw max-pooled sims
        S.append(s_flat.float().cpu())
        gts.append(targets.numpy())
    return torch.cat(S), np.concatenate(gts)


def main(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    splits = resolve_splits(args)
    spec = splits[args.split]
    if spec is None:
        raise SystemExit(f"No '{args.split}' split available.")

    loader = get_dataloader(spec.img_source, spec.part3_csv, spec.part2_source,
                            batch_size=args.batch_size, is_train=False,
                            ids=spec.ids, image_size=args.image_size)

    state = torch.load(args.weights, map_location=device)
    net, K, M = build_csr_from_state(state, args.backbone, device)
    net.eval()

    # Fold BN+Linear into one exact linear map on the raw scores.
    W_eff, b_eff = net.task_head.effective_linear()     # (C, K*M), (C,)
    W_eff, b_eff = W_eff.cpu(), b_eff.cpu()

    S, gts = collect_scores(net, loader, device)        # (N, K*M), (N,)
    base_logits = S @ W_eff.t() + b_eff
    base_pred = base_logits.argmax(1).numpy()
    base_f1 = f1_score(gts, base_pred, average="macro", zero_division=0)

    atlas = torch.load(args.atlas, map_location="cpu")
    hits = atlas["hit_count"].reshape(-1)               # (K*M,)

    # Ablate each prototype: zero its W column, re-argmax. Vectorised per column.
    deltas = np.zeros(K * M, dtype=np.float32)
    for j in tqdm(range(K * M), desc="ablating"):
        w_save = W_eff[:, j].clone()
        if w_save.abs().sum() == 0:
            deltas[j] = 0.0
            continue
        W_eff[:, j] = 0.0
        pred = (S @ W_eff.t() + b_eff).argmax(1).numpy()
        deltas[j] = f1_score(gts, pred, average="macro", zero_division=0) - base_f1
        W_eff[:, j] = w_save

    result = {"delta_f1": torch.from_numpy(deltas), "base_f1": base_f1,
              "hit_count": hits, "K": K, "M": M, "split": args.split}
    torch.save(result, args.out)

    print(f"\nBaseline macro-F1 ({args.split}, {len(gts)} imgs): {base_f1*100:.2f}")
    print(f"Saved impact map -> {args.out}\n")

    wnorm = W_eff.norm(dim=0).numpy()                    # note: after loop W_eff restored
    print("Most COSTLY prototypes to reject (model relies on them):")
    print(f"{'k,m':>8} {'concept':>18} {'hits':>6} {'dF1':>8} {'|w|':>6}")
    for j in np.argsort(deltas)[:15]:
        k, m = divmod(int(j), M)
        print(f"{k},{m:<5} {CONCEPT_KEYS[k]:>18} {int(hits[j]):>6} "
              f"{deltas[j]*100:>7.2f} {wnorm[j]:>6.2f}")

    print("\nSAFEST / most BENEFICIAL to reject (candidate shortcuts):")
    for j in np.argsort(deltas)[::-1][:15]:
        k, m = divmod(int(j), M)
        print(f"{k},{m:<5} {CONCEPT_KEYS[k]:>18} {int(hits[j]):>6} "
              f"{deltas[j]*100:>7.2f} {wnorm[j]:>6.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Per-prototype ablation impact map")
    add_data_args(p)
    p.add_argument("--weights", required=True)
    p.add_argument("--atlas", required=True)
    p.add_argument("--split", default="val", choices=["val", "test", "train"],
                   help="which split to score impact on (default val -- avoids "
                        "tuning decisions on the test set)")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--out", default="checkpoints/impact.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
