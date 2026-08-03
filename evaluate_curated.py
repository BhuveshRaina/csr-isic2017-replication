"""
Measure the effect of train-time atlas refinement (paper Sec. 3.2).

The doctor reviews the prototype atlas in doctor_ui.py and discards prototypes
whose exemplar image shows a shortcut (ruler marks, ink/marker strokes,
dermoscope vignette) rather than genuine pathology. Those prototypes have
s_km forced to 0 permanently. This script re-runs the full test set with that
mask applied and reports the macro-F1 delta against the uncurated baseline.

This is the quantitative counterpart to the paper's Figure 5: it answers
"does clinician curation actually change the numbers, and in which direction?"

Usage:
  python evaluate_curated.py \
    --weights checkpoints_deduped/csr_network_final_best.pth \
    --mask checkpoints_deduped/discarded_prototypes.pt \
    --img_dir data/isic_224x224.zip \
    --task2_dir data/ISIC-2017_Training_Part2_GroundTruth.zip \
    --task3_csv data/ISIC-2017_Training_Part3_GroundTruth.csv \
    --test_img data/ISIC-2017_Test_v2_Data.zip \
    --test_task2 data/ISIC-2017_Test_v2_Part2_GroundTruth.zip \
    --test_task3 data/ISIC-2017_Test_v2_Part3_GroundTruth.csv
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

from dataset import get_dataloader, CLASS_NAMES
from evaluate import build_csr_from_state
from models.concept_model import DEFAULT_BACKBONE
from splits import add_data_args, resolve_splits
from utils import seed_everything, resolve_device, amp_autocast


@torch.no_grad()
def run(net, loader, device, prototype_mask=None, desc="eval"):
    """Discard via BN-input zeroing (net.predict_interactive). See caveat below."""
    preds, gts = [], []
    for images, _, targets in tqdm(loader, desc=desc):
        images = images.to(device)
        with amp_autocast(device):
            if prototype_mask is None:
                logits, _, _ = net(images)
            else:
                logits, _, _ = net.predict_interactive(
                    images, prototype_mask=prototype_mask)
        preds.append(logits.argmax(1).cpu().numpy())
        gts.append(targets.numpy())
    return np.concatenate(preds), np.concatenate(gts)


@torch.no_grad()
def run_true_ablation(net, loader, device, discard_pairs, desc="eval"):
    """
    True zero-weight ablation.

    net.predict_interactive zeroes s_km *before* BatchNorm, so BN's running
    mean/var re-centers that zero into a fixed constant -- the discarded
    prototype still contributes a bias term to every prediction, it's just
    no longer patient-specific. That's why the earlier BN-zeroing experiment
    only moved macro-F1 by +0.72 despite discarding the 5 highest-traffic
    prototypes: it wasn't really removing their influence, just freezing it.

    Task head is Linear(BatchNorm(s)), and at inference BatchNorm is a fixed
    per-feature affine map, so the whole head is exactly equivalent to a
    single linear map on the *raw* similarity scores s:
        logits = s @ W_eff.T + b_eff        (TaskHead.effective_linear)
    To genuinely remove a prototype's influence -- contributes 0 to every
    logit, for every patient, regardless of its raw similarity value -- we
    zero its column in W_eff directly, instead of touching s.
    """
    W_eff, b_eff = net.task_head.effective_linear()   # (num_classes, K*M), (num_classes,)
    W_eff = W_eff.clone()
    K, M = net.prototypes.num_concepts, net.prototypes.M
    for k, m in discard_pairs:
        W_eff[:, k * M + m] = 0.0

    preds, gts = [], []
    for images, _, targets in tqdm(loader, desc=desc):
        images = images.to(device)
        with amp_autocast(device):
            _, s_flat, _ = net(images)             # raw (unmasked) similarity scores
            logits = s_flat.float() @ W_eff.t() + b_eff
        preds.append(logits.argmax(1).cpu().numpy())
        gts.append(targets.numpy())
    return np.concatenate(preds), np.concatenate(gts)


def report(tag, preds, gts):
    macro = f1_score(gts, preds, average="macro", zero_division=0)
    per_class = f1_score(gts, preds, average=None, zero_division=0)
    acc = (preds == gts).mean() * 100
    print(f"\n--- {tag} ---")
    print(f"Macro F1 : {macro*100:.2f}   Accuracy: {acc:.2f}%")
    for name, v in zip(CLASS_NAMES, per_class):
        print(f"   {name:>22s}: {v*100:.2f}")
    return macro


def main(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    splits = resolve_splits(args)
    if splits["test"] is None:
        raise SystemExit("No test split available.")
    spec = splits["test"]
    loader = get_dataloader(spec.img_source, spec.part3_csv, spec.part2_source,
                            batch_size=args.batch_size, is_train=False,
                            ids=spec.ids, image_size=args.image_size)

    state = torch.load(args.weights, map_location=device)
    net, K, M = build_csr_from_state(state, args.backbone, device)
    net.eval()

    blob = torch.load(args.mask, map_location="cpu")
    mask = blob["prototype_mask"].to(device)
    discarded = blob.get("discarded", [])
    if tuple(mask.shape) != (K, M):
        raise ValueError(f"Mask is {tuple(mask.shape)} but network is ({K}, {M}).")

    print(f"Test images: {len(loader.dataset)}")
    print(f"Discarded prototypes: {len(discarded)} / {K*M}")
    if discarded:
        print("  " + ", ".join(f"k={k},m={m}" for k, m in discarded[:20])
              + (" ..." if len(discarded) > 20 else ""))

    base_preds, gts = run(net, loader, device, None, desc="baseline")

    if args.true_ablation:
        cur_preds, _ = run_true_ablation(net, loader, device, discarded, desc="curated (true ablation)")
        tag = "CURATED -- true zero-weight ablation (Discard column zeroed in effective head)"
    else:
        cur_preds, _ = run(net, loader, device, mask, desc="curated (BN-zeroing)")
        tag = "CURATED -- BN-input zeroing (see run_true_ablation docstring for caveat)"

    b = report("BASELINE (no curation)", base_preds, gts)
    c = report(tag, cur_preds, gts)

    changed = int((base_preds != cur_preds).sum())
    print(f"\nDelta macro-F1 : {(c-b)*100:+.2f} points")
    print(f"Predictions changed by curation: {changed} / {len(gts)}")
    print("\nCurated confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(gts, cur_preds))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate CSR with a curated prototype mask")
    add_data_args(p)
    p.add_argument("--weights", required=True)
    p.add_argument("--mask", default="checkpoints_deduped/discarded_prototypes.pt")
    p.add_argument("--true_ablation", action="store_true",
                   help="zero the discarded prototypes' columns in the effective "
                        "linear head (exact removal) instead of zeroing s_km before "
                        "BatchNorm (which only fixes it to a constant bias)")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
