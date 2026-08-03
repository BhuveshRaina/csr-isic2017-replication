"""
Confidence-gated doctor-in-the-loop (the deployable version).

The targeted experiment in test_time_interaction.py cheated twice: it knew the
true concept labels AND it knew which images the model got wrong. A real
clinician knows neither. This script removes the second, more important cheat.

Idea: the model's own prediction confidence (max softmax probability) is
available at inference with no ground truth. If correctly-classified images are
confident and wrongly-classified ones are hesitant, then "confidence < T" is a
legitimate trigger for review -- the doctor only looks at flagged cases.

We still model the doctor's clinical judgement of "is this concept actually
present?" with the ground-truth concept labels (a real doctor judges by eye;
that judgement is what the tool is FOR, so simulating a competent one is fair).
The thing we are NOT allowed to fake -- knowing the prediction is wrong -- is
exactly what the confidence gate replaces.

Outputs:
  1. Confidence separation: correct vs wrong.
  2. Confidence as a "needs review" detector (recall of wrong / review workload).
  3. Gated intervention sweep: reject one concept only on flagged low-confidence
     images, measuring how many wrong get FIXED vs how many correct get BROKEN.

Usage:
  python confidence_gate.py \
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
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import get_dataloader, CLASS_NAMES
from evaluate import build_csr_from_state
from models.concept_model import DEFAULT_BACKBONE
from splits import add_data_args, resolve_splits
from utils import seed_everything, resolve_device, amp_autocast


@torch.no_grad()
def run(net, loader, device):
    S, gts, cgt = [], [], []
    for images, concepts, targets in tqdm(loader, desc="forward"):
        images = images.to(device)
        with amp_autocast(device):
            _, s_flat, _ = net(images)
        S.append(s_flat.float().cpu())
        gts.append(targets.numpy())
        cgt.append(concepts.numpy())
    return torch.cat(S), np.concatenate(gts), np.concatenate(cgt)


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

    S, gts, cgt = run(net, loader, device)
    N = S.shape[0]
    S3 = S.view(N, K, M)
    W, b = net.task_head.effective_linear()
    W, b = W.cpu(), b.cpu()

    logits = S @ W.t() + b
    probs = torch.softmax(logits, dim=1)
    # Paper's actual trigger (Sec. 5, Interactivity): "we only require human
    # interaction when the output probability of the model is indecisive: when
    # the difference of top-1 and top-2 class probability is less than 0.3."
    # NOT max-probability (what this script originally used) -- margin. A case
    # at 0.50/0.45/0.05 is highly ambiguous by margin (0.05) but unremarkable
    # by max-prob (0.50), so the two statistics disagree exactly on the
    # borderline cases that matter most.
    sorted_probs, _ = probs.sort(dim=1, descending=True)
    margin = (sorted_probs[:, 0] - sorted_probs[:, 1]).numpy()   # small margin = indecisive
    base_pred = logits.argmax(1).numpy()
    correct = base_pred == gts
    base_f1 = f1_score(gts, base_pred, average="macro", zero_division=0)

    # ---- 1. confidence separation ---------------------------------------
    print(f"\nBaseline macro-F1: {base_f1*100:.2f} | {int(correct.sum())} correct, "
          f"{int((~correct).sum())} wrong of {N}")
    print("\n=== 1. Top1-Top2 margin: correct vs wrong (small margin = indecisive) ===")
    print(f"  correct: mean {margin[correct].mean():.3f}  median {np.median(margin[correct]):.3f}")
    print(f"  wrong  : mean {margin[~correct].mean():.3f}  median {np.median(margin[~correct]):.3f}")

    # ---- 2. confidence as a 'needs review' detector ---------------------
    absent = cgt < 0.5
    peak = np.where(absent, S3.max(-1).values.numpy(), -np.inf)
    worst = peak.argmax(1)
    has_absent = np.isfinite(peak.max(1))

    print("\n=== 2. 'Review if margin < T' as a wrong-case detector ===")
    print("(paper's own criterion: T=0.3)")
    print(f"{'T':>6} {'flagged':>8} {'wrong_caught':>13} {'correct_flagged':>16} "
          f"{'%wrong_caught':>13}")
    n_wrong = int((~correct).sum())
    for T in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        flag = margin < T
        wc = int((flag & ~correct).sum())
        cf = int((flag & correct).sum())
        tag = "  <-- paper's T" if abs(T - 0.30) < 1e-6 else ""
        print(f"{T:>6.2f} {int(flag.sum()):>8} {wc:>13} {cf:>16} "
              f"{100*wc/max(n_wrong,1):>12.1f}%{tag}")

    # ---- 3. gated intervention sweep ------------------------------------
    # For flagged (low-conf) images that have a rejectable absent concept,
    # reject the single highest-firing absent concept. Measure fixed vs broke.
    print("\n=== 3. Gated intervention: reject 1 concept on flagged images ===")
    print(f"(baseline macro-F1 {base_f1*100:.2f}; paper's own criterion: T=0.3)")
    print(f"{'T':>6} {'treated':>8} {'fixed':>6} {'broke':>6} {'net':>6} "
          f"{'final_F1':>9} {'delta':>7}")
    best = (base_f1, None)
    for T in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 2.0]:
        flag = (margin < T) & has_absent
        St = S3.clone()
        for i in np.where(flag)[0]:
            St[i, worst[i], :] = 0.0
        new_pred = (St.reshape(N, K * M) @ W.t() + b).argmax(1).numpy()
        fixed = int((flag & ~correct & (new_pred == gts)).sum())
        broke = int((flag & correct & (new_pred != gts)).sum())
        f1_new = f1_score(gts, new_pred, average="macro", zero_division=0)
        tag = " <-- reject-all (no gate)" if T > 1.0 else (
              "  <-- paper's T" if abs(T - 0.30) < 1e-6 else "")
        print(f"{T:>6.2f} {int(flag.sum()):>8} {fixed:>6} {broke:>6} "
              f"{fixed-broke:>+6} {f1_new*100:>8.2f} {(f1_new-base_f1)*100:>+6.2f}{tag}")
        if f1_new > best[0]:
            best = (f1_new, T)

    if best[1] is not None:
        print(f"\nBest gate: T={best[1]:.2f} -> macro-F1 {best[0]*100:.2f} "
              f"(+{(best[0]-base_f1)*100:.2f} over baseline)")
    else:
        print("\nNo threshold beat baseline -- confidence gating did not help on this model.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Confidence-gated interaction")
    add_data_args(p)
    p.add_argument("--weights", required=True)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
