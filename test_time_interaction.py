"""
Test-time doctor-in-the-loop evaluation (paper Sec. 3.3) -- the MAIN mechanism.

Train-time atlas curation (Sec. 3.2) is optional and one-time: the paper only
discards prototypes that are clear shortcuts. The mechanism that actually runs
per patient is test-time interaction, and this script measures whether it
improves predictions across the whole test set instead of anecdotally.

We simulate an ORACLE clinician using the Part-2 ground-truth concept
annotations. For each test image the doctor sees which concepts the model is
using as evidence and rejects the ones that are genuinely absent from this
patient (Eqn: s_km = 0 for all m in that concept). This is the upper bound of
what concept-level interaction can deliver -- a real clinician is not a perfect
oracle, but if even the oracle does not help, the mechanism is not useful.

Three settings are reported:
  baseline        no interaction
  oracle_all      reject EVERY concept whose ground truth is 'absent'
  oracle_one      reject only the single most-wrongly-used concept
                  (highest model similarity among the truly-absent ones) --
                  a realistic "doctor makes one correction" scenario

Efficiency note: the backbone forward pass dominates cost, and concept
rejection is just zeroing entries of s_scores. So we run the network ONCE per
batch and then apply every rejection variant vectorised in score space.

Usage:
  python test_time_interaction.py \
    --weights checkpoints_revive/csr_network_final_best.pth \
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

from dataset import get_dataloader, CLASS_NAMES, CONCEPT_KEYS
from evaluate import build_csr_from_state
from models.concept_model import DEFAULT_BACKBONE
from splits import add_data_args, resolve_splits
from utils import seed_everything, resolve_device, amp_autocast


@torch.no_grad()
def run(net, loader, device):
    """
    One pass. Returns per-image:
      s_scores (N,K,M) raw similarity, gts (N,), concept_gt (N,K)
    """
    S, gts, cgt = [], [], []
    for images, concepts, targets in tqdm(loader, desc="forward"):
        images = images.to(device)
        with amp_autocast(device):
            _, s_flat, _ = net(images)
        S.append(s_flat.float().cpu())
        gts.append(targets.numpy())
        cgt.append(concepts.numpy())
    return torch.cat(S), np.concatenate(gts), np.concatenate(cgt)


def classify(net, s_flat):
    """Apply the task head to (possibly modified) similarity scores."""
    W, b = net.task_head.effective_linear()
    return (s_flat @ W.cpu().t() + b.cpu()).argmax(1).numpy()


def report(tag, preds, gts, base=None):
    macro = f1_score(gts, preds, average="macro", zero_division=0)
    per = f1_score(gts, preds, average=None, zero_division=0)
    acc = (preds == gts).mean() * 100
    line = f"\n--- {tag} ---\nMacro F1 : {macro*100:.2f}   Accuracy: {acc:.2f}%"
    if base is not None:
        line += f"   (delta {(macro-base)*100:+.2f})"
    print(line)
    for name, v in zip(CLASS_NAMES, per):
        print(f"   {name:>22s}: {v*100:.2f}")
    return macro


def main(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    splits = resolve_splits(args)
    spec = splits["test"]
    if spec is None:
        raise SystemExit("No test split available.")

    loader = get_dataloader(spec.img_source, spec.part3_csv, spec.part2_source,
                            batch_size=args.batch_size, is_train=False,
                            ids=spec.ids, image_size=args.image_size)

    state = torch.load(args.weights, map_location=device)
    net, K, M = build_csr_from_state(state, args.backbone, device)
    net.eval()

    S, gts, cgt = run(net, loader, device)          # (N,K*M), (N,), (N,K)
    N = S.shape[0]
    S3 = S.view(N, K, M)

    print(f"\nTest images: {N} | K={K} concepts, M={M} prototypes")
    absent = cgt < 0.5                               # (N,K) True where concept truly absent
    print(f"Truly-absent concept slots: {int(absent.sum())} / {N*K} "
          f"({100*absent.mean():.1f}%) -- these are what a doctor could reject")

    # ---- baseline -------------------------------------------------------
    base_pred = classify(net, S)
    base = report("BASELINE (no interaction)", base_pred, gts)

    # ---- oracle: reject every truly-absent concept ----------------------
    mask_all = torch.from_numpy((~absent).astype(np.float32))      # (N,K) keep=1
    S_all = (S3 * mask_all.unsqueeze(-1)).reshape(N, K * M)
    pred_all = classify(net, S_all)
    report("ORACLE-ALL (reject every absent concept)", pred_all, gts, base)
    print(f"   predictions changed: {(pred_all != base_pred).sum()} / {N}")

    # ---- oracle: reject only the single worst-offending concept ---------
    # "worst" = the truly-absent concept the model scores highest on.
    peak = S3.max(dim=-1).values.numpy()                            # (N,K)
    peak_absent = np.where(absent, peak, -np.inf)
    worst = peak_absent.argmax(axis=1)                              # (N,)
    has_absent = np.isfinite(peak_absent.max(axis=1))
    mask_one = torch.ones(N, K)
    rows = np.arange(N)[has_absent]
    mask_one[rows, worst[has_absent]] = 0.0
    S_one = (S3 * mask_one.unsqueeze(-1)).reshape(N, K * M)
    pred_one = classify(net, S_one)
    report("ORACLE-ONE (reject single worst absent concept)", pred_one, gts, base)
    print(f"   predictions changed: {(pred_one != base_pred).sum()} / {N}")
    print(f"   images where a correction was available: {int(has_absent.sum())} / {N}")

    # ---- per-concept breakdown: which concept helps most when rejected --
    print("\nRejecting ONE concept at a time (only where truly absent):")
    for k in range(K):
        mk = torch.ones(N, K)
        sel = absent[:, k]
        mk[torch.from_numpy(sel), k] = 0.0
        Sk = (S3 * mk.unsqueeze(-1)).reshape(N, K * M)
        pk = classify(net, Sk)
        f1k = f1_score(gts, pk, average="macro", zero_division=0)
        print(f"   {CONCEPT_KEYS[k]:>18}: dF1 {100*(f1k-base):+6.2f}  "
              f"(applied to {int(sel.sum())} images, "
              f"{(pk != base_pred).sum()} predictions changed)")

    print("\nConfusion matrix, ORACLE-ALL (rows=true, cols=pred):")
    print(confusion_matrix(gts, pred_all))

    # ---- TARGETED intervention (the paper's actual intent) --------------
    # Concept rejection is meant for cases where the model HALLUCINATES a
    # concept: it fires strongly on a concept that is truly absent AND that
    # firing is driving a wrong prediction. Blanket-rejecting all absent
    # concepts (above) deletes information the classifier legitimately uses
    # and collapses it. Here we intervene ONLY on currently-wrong images, and
    # only reject the single truly-absent concept the model scores highest on.
    print("\n" + "=" * 60)
    print("TARGETED intervention: only on images the model gets WRONG,")
    print("reject the single truly-absent concept it fires highest on.")
    print("=" * 60)
    wrong = base_pred != gts
    peak = S3.max(dim=-1).values.numpy()
    peak_absent = np.where(absent, peak, -np.inf)
    worst = peak_absent.argmax(axis=1)
    has_absent = np.isfinite(peak_absent.max(axis=1))
    target_rows = np.where(wrong & has_absent)[0]

    S_t = S3.clone()
    for i in target_rows:
        S_t[i, worst[i], :] = 0.0
    pred_t = classify(net, S_t.reshape(N, K * M))

    # accounting: of the images we touched, how many flipped and in which direction
    fixed = int(((pred_t == gts) & wrong)[target_rows].sum())
    broke = int(((pred_t != gts) & ~wrong)[target_rows].sum())  # 0 by construction (we only touch wrong)
    still_wrong = len(target_rows) - fixed
    f1_t = f1_score(gts, pred_t, average="macro", zero_division=0)
    print(f"\nWrong images with a rejectable absent concept: {len(target_rows)} "
          f"/ {int(wrong.sum())} wrong")
    print(f"   fixed by the intervention : {fixed}")
    print(f"   still wrong               : {still_wrong}")
    report("TARGETED (reject on wrong images only)", pred_t, gts, base)
    print(f"\nNet: a doctor intervening only when the model is wrong and only on a "
          f"clearly-absent\nover-fired concept would correct {fixed} of "
          f"{len(target_rows)} such cases ({100*fixed/max(len(target_rows),1):.0f}%).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test-time interaction evaluation (Sec 3.3)")
    add_data_args(p)
    p.add_argument("--weights", required=True)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
