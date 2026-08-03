"""
Improving concept-level interaction: in-distribution rejection.

The problem
-----------
The paper's concept rejection sets s_km = 0 for the rejected concept. But the
task head was trained on similarity scores that live in a narrow band (mean
~0.09, and "present" prototypes fire up to ~0.7-0.8). It NEVER saw an exact
zero during training. So rejecting a concept -- especially one firing high, like
milia_like_cyst at 0.777 -- forces the classifier to evaluate an input pattern
it has never encountered (all 100 scores of a block = 0). That out-of-distribution
jolt is what causes the violent confidence swings we observed (a correct case
flipping to 98% wrong off a single rejection).

Notice the paper's own PROSE says the model should "suppress its confidence" in
the concept -- but its EQUATION annihilates it to zero. This script implements
what the prose actually describes.

The idea
--------
When a doctor says "concept k is absent," the faithful thing to tell the
classifier is not "pretend you saw nothing" (zero) but "make the evidence for
concept k look like it does on patients who genuinely DON'T have concept k."
We build that reference from the TRAINING set only (no test leakage): for every
prototype (k,m), the average / median similarity it produces on training images
where concept k's ground-truth label is 'absent'. That is the in-distribution
"this concept is absent" signal.

Strategies compared (all applied only to truly-absent concepts)
---------------------------------------------------------------
  zero (paper)   s_km = 0                                  (out of distribution)
  absent_mean    s_km = mean_km   over absent-k train imgs (in distribution)
  absent_median  s_km = median_km over absent-k train imgs (robust in-dist)
  clip_to_mean   s_km = min(s_km, mean_km)                 (only pull DOWN
                 over-firing prototypes to normal-absent level; never raise --
                 this is literally "suppress the hallucination", the paper's
                 stated intent)
  shrink_0.5     s_km = 0.5 * s_km                          (gentle scaling toward 0)

For each strategy we reject one absent concept at a time (the realistic doctor
action) and measure, stratified by the model's top1-top2 margin (paper's
intervention zone is margin < 0.3):
  fixed   wrong -> correct   (good)
  broke   correct -> wrong   (bad)
  mean confidence delta on the TRUE class (did we make the right answer more or
  less likely, even when the label didn't flip)

The winning strategy is the one that turns concept rejection from net-negative
into net-positive (or least harmful) in the margin < 0.3 zone.

Usage:
  python improve_concept_rejection.py \
    --weights checkpoints_revive/csr_network_final_best.pth \
    --img_dir data/isic_224x224.zip \
    --task2_dir data/ISIC-2017_Training_Part2_GroundTruth.zip \
    --task3_csv data/ISIC-2017_Training_Part3_GroundTruth.csv \
    --val_img data/ISIC-2017_Validation_Data.zip \
    --val_task2 data/ISIC-2017_Validation_Part2_GroundTruth.zip \
    --val_task3 data/ISIC-2017_Validation_Part3_GroundTruth.csv \
    --test_img data/ISIC-2017_Test_v2_Data.zip \
    --test_task2 data/ISIC-2017_Test_v2_Part2_GroundTruth.zip \
    --test_task3 data/ISIC-2017_Test_v2_Part3_GroundTruth.csv \
    --out checkpoints_revive/concept_rejection_profiles.pt
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import get_dataloader, CLASS_NAMES, CONCEPT_KEYS
from evaluate import build_csr_from_state
from models.concept_model import DEFAULT_BACKBONE
from splits import add_data_args, resolve_splits
from utils import seed_everything, resolve_device, amp_autocast


@torch.no_grad()
def collect(net, loader, device):
    """One forward pass. Returns S3 (N,K,M), gts (N,), concept_gt (N,K)."""
    S, gts, cgt = [], [], []
    for images, concepts, targets in tqdm(loader, desc="forward"):
        images = images.to(device)
        with amp_autocast(device):
            _, s_flat, _ = net(images)
        S.append(s_flat.float().cpu())
        gts.append(targets.numpy())
        cgt.append(concepts.numpy())
    S = torch.cat(S)
    return S, np.concatenate(gts), np.concatenate(cgt)


def build_absent_profile(S3_train, cgt_train, K, M):
    """
    Per-prototype reference score for 'concept genuinely absent', from TRAINING
    data only. Returns (mean, median) each shaped (K, M).
    """
    absent = torch.from_numpy(cgt_train < 0.5)          # (Ntr, K)
    prof_mean = torch.zeros(K, M)
    prof_med = torch.zeros(K, M)
    print("\nPer-concept 'absent' reference (built from training set):")
    for k in range(K):
        sel = absent[:, k]
        n = int(sel.sum())
        block = S3_train[sel, k, :]                     # (n, M)
        prof_mean[k] = block.mean(0)
        prof_med[k] = block.median(0).values
        print(f"  {CONCEPT_KEYS[k]:>18}: {n:4d} absent train imgs | "
              f"mean score {block.mean():.3f}  max-proto-mean {prof_mean[k].max():.3f}")
    return prof_mean, prof_med


def summarize(delta, was_correct, now_correct, tag):
    if len(delta) == 0:
        print(f"    {tag:>28}: no cases")
        return None
    improved = int((delta > 1e-6).sum())
    worsened = int((delta < -1e-6).sum())
    fixed = int((~was_correct & now_correct).sum())
    broke = int((was_correct & ~now_correct).sum())
    net = fixed - broke
    print(f"    {tag:>28}: n={len(delta):4d}  mean_dP_true={delta.mean()*100:+6.2f}pp  "
          f"fixed={fixed:3d} broke={broke:3d} net={net:+4d}  "
          f"improved={improved:4d} worsened={worsened:4d}")
    return net


def main(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    splits = resolve_splits(args)

    for name in ("train", "test"):
        if splits.get(name) is None:
            raise SystemExit(f"No '{name}' split available (needed to build/eval).")

    state = torch.load(args.weights, map_location=device)
    net, K, M = build_csr_from_state(state, args.backbone, device)
    net.eval()
    W, b = net.task_head.effective_linear()
    W, b = W.cpu(), b.cpu()

    def probs_of(S3):
        return torch.softmax(S3.reshape(S3.shape[0], K * M) @ W.t() + b, dim=1)

    # ---- 1. build the 'absent' reference from TRAINING -------------------
    tr = splits["train"]
    tr_loader = get_dataloader(tr.img_source, tr.part3_csv, tr.part2_source,
                               batch_size=args.batch_size, is_train=False,
                               ids=tr.ids, image_size=args.image_size)
    S_tr, _, cgt_tr = collect(net, tr_loader, device)
    S3_tr = S_tr.view(-1, K, M)
    prof_mean, prof_med = build_absent_profile(S3_tr, cgt_tr, K, M)

    # ---- 2. evaluate on TEST --------------------------------------------
    te = splits["test"]
    te_loader = get_dataloader(te.img_source, te.part3_csv, te.part2_source,
                               batch_size=args.batch_size, is_train=False,
                               ids=te.ids, image_size=args.image_size)
    S_te, gts, cgt = collect(net, te_loader, device)
    N = S_te.shape[0]
    S3 = S_te.view(N, K, M)

    base_probs = probs_of(S3)
    base_pred = base_probs.argmax(1).numpy()
    gts_t = torch.from_numpy(gts)
    p_true_before = base_probs[torch.arange(N), gts_t].numpy()
    was_correct_all = base_pred == gts
    base_f1 = f1_score(gts, base_pred, average="macro", zero_division=0)

    sorted_p, _ = base_probs.sort(dim=1, descending=True)
    margin = (sorted_p[:, 0] - sorted_p[:, 1]).numpy()
    absent = cgt < 0.5

    print(f"\nTest: {N} images | baseline macro-F1 {base_f1*100:.2f} | "
          f"{int((margin<0.30).sum())} in margin<0.30 zone")

    def replace(block, k, strategy):
        """block: (n, M) scores for concept k on selected images."""
        if strategy == "zero":
            return torch.zeros_like(block)
        if strategy == "absent_mean":
            return prof_mean[k].unsqueeze(0).expand_as(block).clone()
        if strategy == "absent_median":
            return prof_med[k].unsqueeze(0).expand_as(block).clone()
        if strategy == "clip_to_mean":
            return torch.minimum(block, prof_mean[k].unsqueeze(0))
        if strategy == "shrink_0.5":
            return 0.5 * block
        raise ValueError(strategy)

    strategies = ["zero", "absent_mean", "absent_median", "clip_to_mean", "shrink_0.5"]

    # For each strategy: reject one absent concept at a time, pool all such
    # (image, concept) events, and report stratified by margin.
    results = {}
    for strat in strategies:
        deltas, wasc, nowc, marg = [], [], [], []
        for k in range(K):
            sel = np.where(absent[:, k])[0]
            if len(sel) == 0:
                continue
            Sk = S3.clone()
            Sk[sel, k, :] = replace(S3[sel, k, :], k, strat)
            pk = probs_of(Sk)
            p_after = pk[torch.from_numpy(sel), gts_t[sel]].numpy()
            pred_after = pk.argmax(1).numpy()[sel]
            deltas.append(p_after - p_true_before[sel])
            wasc.append(was_correct_all[sel])
            nowc.append(pred_after == gts[sel])
            marg.append(margin[sel])
        deltas = np.concatenate(deltas); wasc = np.concatenate(wasc)
        nowc = np.concatenate(nowc); marg = np.concatenate(marg)
        results[strat] = (deltas, wasc, nowc, marg)

    label = {"zero": "zero (PAPER)", "absent_mean": "absent_mean",
             "absent_median": "absent_median", "clip_to_mean": "clip_to_mean",
             "shrink_0.5": "shrink_0.5"}

    print("\n" + "=" * 78)
    print("SINGLE-CONCEPT REJECTION, replacement-strategy comparison")
    print("(dP_true = change in probability of the CORRECT class; higher = better)")
    print("=" * 78)
    for strat in strategies:
        d, wc, nc, mg = results[strat]
        print(f"\n  >>> {label[strat]}")
        summarize(d, wc, nc, "ALL absent cases")
        lo = mg < 0.30
        summarize(d[lo], wc[lo], nc[lo], "margin < 0.30 (act here)")
        summarize(d[~lo], wc[~lo], nc[~lo], "margin >= 0.30 (skip)")

    # ---- headline: net (fixed-broke) in the decision zone, per strategy ---
    print("\n" + "=" * 78)
    print("HEADLINE  --  margin < 0.30 zone (where the paper says to intervene)")
    print("=" * 78)
    print(f"  {'strategy':>16} {'net(fixed-broke)':>18} {'mean_dP_true':>14}")
    best = None
    for strat in strategies:
        d, wc, nc, mg = results[strat]
        lo = mg < 0.30
        fixed = int((~wc[lo] & nc[lo]).sum())
        broke = int((wc[lo] & ~nc[lo]).sum())
        net = fixed - broke
        print(f"  {label[strat]:>16} {net:>+18} {d[lo].mean()*100:>+13.2f}pp")
        if best is None or net > best[1]:
            best = (strat, net)
    print(f"\n  Best in decision zone: {label[best[0]]}  (net {best[1]:+d})")
    print("  (zero is the paper's method; anything above it is an improvement.)")

    torch.save({"prof_mean": prof_mean, "prof_med": prof_med,
                "results": {s: [x.tolist() if isinstance(x, np.ndarray) else x
                                for x in results[s]] for s in strategies},
                "K": K, "M": M}, args.out)
    print(f"\nSaved profiles + results -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="In-distribution concept rejection")
    add_data_args(p)
    p.add_argument("--weights", required=True)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="checkpoints/concept_rejection_profiles.pt")
    main(p.parse_args())
