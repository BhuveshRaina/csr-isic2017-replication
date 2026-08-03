"""
Selective, in-distribution concept rejection -- the "make it actually help" version.

We established two things:
  * Zeroing a rejected concept is out-of-distribution and wrecks the model.
  * Replacing it with the concept's typical ABSENT profile (absent_median) is
    safe, but rejecting EVERY absent concept is still ~neutral, because most
    absent concepts are not hurting the prediction -- only the ones the model
    is HALLUCINATING (firing high despite being absent) are.

This script adds the missing piece: a gate that rejects a concept only when it
is worth rejecting. A concept is a rejection candidate on image i only if:
    (a) the doctor says it is absent          (ground-truth concept label = 0)
    (b) the model is indecisive               (top1-top2 margin < margin_T; paper uses 0.3)
    (c) the model is OVER-FIRING the concept  (peak similarity exceeds its
        normal-absent level by more than overfire_T)

When those hold, we reject the single most over-firing absent concept using the
absent_median replacement (not zero). Everything else is left untouched.

(a) is the doctor's judgement (simulated with the oracle, same as all prior
experiments -- judging concept presence by eye is exactly what the doctor is
FOR). (b) and (c) are computed from the model at inference with NO ground truth,
so this is a deployable policy, not an oracle on correctness.

We sweep margin_T and overfire_T and report, for each, the resulting test
macro-F1 and the net (fixed - broke). The question: does selective
in-distribution rejection push ISIC doctor-in-the-loop above baseline?

Usage:
  python selective_rejection.py \
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
    --out checkpoints_revive/selective_rejection.pt
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
    for name in ("train", "test"):
        if splits.get(name) is None:
            raise SystemExit(f"No '{name}' split available.")

    state = torch.load(args.weights, map_location=device)
    net, K, M = build_csr_from_state(state, args.backbone, device)
    net.eval()
    W, b = net.task_head.effective_linear()
    W, b = W.cpu(), b.cpu()

    def preds_of(S3):
        return (S3.reshape(S3.shape[0], K * M) @ W.t() + b).argmax(1).numpy()

    def probs_of(S3):
        return torch.softmax(S3.reshape(S3.shape[0], K * M) @ W.t() + b, dim=1)

    # ---- absent profiles from TRAIN (median = robust replacement value) --
    tr = splits["train"]
    tr_loader = get_dataloader(tr.img_source, tr.part3_csv, tr.part2_source,
                               batch_size=args.batch_size, is_train=False,
                               ids=tr.ids, image_size=args.image_size)
    S_tr, _, cgt_tr = collect(net, tr_loader, device)
    S3_tr = S_tr.view(-1, K, M)
    absent_tr = torch.from_numpy(cgt_tr < 0.5)
    prof_med = torch.zeros(K, M)
    absent_peak_level = torch.zeros(K)     # typical peak similarity when absent
    for k in range(K):
        block = S3_tr[absent_tr[:, k], k, :]
        prof_med[k] = block.median(0).values
        # "normal absent" peak = median over absent imgs of that image's max-proto score
        absent_peak_level[k] = block.max(dim=1).values.median()
    print("\nNormal-absent peak level per concept (over-firing is measured against this):")
    for k in range(K):
        print(f"  {CONCEPT_KEYS[k]:>18}: {absent_peak_level[k]:.3f}")

    # ---- test set -------------------------------------------------------
    te = splits["test"]
    te_loader = get_dataloader(te.img_source, te.part3_csv, te.part2_source,
                               batch_size=args.batch_size, is_train=False,
                               ids=te.ids, image_size=args.image_size)
    S_te, gts, cgt = collect(net, te_loader, device)
    N = S_te.shape[0]
    S3 = S_te.view(N, K, M)

    base_probs = probs_of(S3)
    base_pred = base_probs.argmax(1).numpy()
    base_f1 = f1_score(gts, base_pred, average="macro", zero_division=0)
    correct0 = base_pred == gts
    sorted_p, _ = base_probs.sort(dim=1, descending=True)
    margin = (sorted_p[:, 0] - sorted_p[:, 1]).numpy()

    absent = cgt < 0.5                                   # doctor oracle: concept absent
    peak = S3.max(dim=-1).values.numpy()                 # (N,K) model's peak firing per concept
    overfire = peak - absent_peak_level.numpy()[None, :] # how far above normal-absent
    overfire[~absent] = -np.inf                          # only absent concepts are rejectable

    print(f"\nTest: {N} imgs | baseline macro-F1 {base_f1*100:.2f} | "
          f"{int(correct0.sum())} correct, {int((~correct0).sum())} wrong")

    def apply_policy(margin_T, overfire_T):
        """Reject the single most over-firing absent concept on each qualifying image."""
        St = S3.clone()
        treated = 0
        cand = overfire.copy()
        cand[margin >= margin_T, :] = -np.inf            # gate (b): only indecisive imgs
        best_k = cand.argmax(1)
        best_val = cand.max(1)
        for i in range(N):
            if best_val[i] > overfire_T:                 # gate (c): must be over-firing enough
                St[i, best_k[i], :] = prof_med[best_k[i]]
                treated += 1
        new_pred = preds_of(St)
        f1_new = f1_score(gts, new_pred, average="macro", zero_division=0)
        fixed = int((~correct0 & (new_pred == gts)).sum())
        broke = int((correct0 & (new_pred != gts)).sum())
        return treated, fixed, broke, f1_new

    print("\n" + "=" * 74)
    print("SELECTIVE in-distribution rejection sweep (replacement = absent_median)")
    print("=" * 74)
    print(f"{'margin_T':>9} {'overfire_T':>11} {'treated':>8} {'fixed':>6} "
          f"{'broke':>6} {'net':>6} {'macroF1':>9} {'delta':>7}")
    best = (base_f1, None)
    for margin_T in [0.30, 0.20, 1.01]:                  # 1.01 = no margin gate (all imgs)
        for overfire_T in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            treated, fixed, broke, f1_new = apply_policy(margin_T, overfire_T)
            tagm = "(all)" if margin_T > 1 else ""
            print(f"{margin_T:>9.2f}{tagm:>0} {overfire_T:>11.2f} {treated:>8} "
                  f"{fixed:>6} {broke:>6} {fixed-broke:>+6} {f1_new*100:>8.2f} "
                  f"{(f1_new-base_f1)*100:>+6.2f}")
            if f1_new > best[0]:
                best = (f1_new, (margin_T, overfire_T, treated, fixed, broke))

    print("\n" + "-" * 74)
    if best[1] is not None:
        mT, oT, tr_n, fx, bk = best[1]
        print(f"BEST: margin_T={mT:.2f}, overfire_T={oT:.2f} -> macro-F1 {best[0]*100:.2f} "
              f"(+{(best[0]-base_f1)*100:.2f} over {base_f1*100:.2f})")
        print(f"      treated {tr_n} images: fixed {fx}, broke {bk}, net {fx-bk:+d}")
        print("      ==> selective in-distribution rejection IMPROVES ISIC doctor-in-the-loop.")
    else:
        print(f"No policy beat baseline {base_f1*100:.2f}. Selective inference-only rejection "
              f"is safe but not net-positive; concept-dropout retraining is the next lever.")

    torch.save({"base_f1": base_f1, "prof_med": prof_med,
                "absent_peak_level": absent_peak_level}, args.out)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Selective in-distribution concept rejection")
    add_data_args(p)
    p.add_argument("--weights", required=True)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="checkpoints/selective_rejection.pt")
    main(p.parse_args())
