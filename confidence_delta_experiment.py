"""
Per-image, per-concept confidence-delta experiment.

Question: for every test image, if we reject (zero out) a concept that is
genuinely ABSENT (ground truth), does the model become MORE confident in the
correct class, or does confidence swing around unpredictably / toward the
wrong class?

This is a finer-grained probe than the macro-F1 sweeps in
test_time_interaction.py and confidence_gate.py: it does not require the
prediction to FLIP to register an effect. It directly measures

    delta = P(true class | after rejection) - P(true class | before)

for every (image, absent concept) pair, so it can see partial harm or partial
benefit that a flip-only metric misses -- exactly the "confidence went from
64% to 98.9% but the label didn't change" case you just saw manually with
pigment_network.

It also stratifies every result by the model's own top1-top2 margin BEFORE
intervention, split at the paper's own threshold (0.3). This directly tests
the paper's underlying assumption -- that indecisive (margin < 0.3) cases
benefit more from concept rejection than already-confident (margin >= 0.3)
cases -- with data, instead of a single hand-picked example.

Two intervention modes per image, for every concept k where ground truth says
absent:
  single  -- zero ONLY concept k's 100 prototype scores, one concept at a time
             (this is what you did by hand in the Streamlit app)
  all     -- zero EVERY truly-absent concept simultaneously (one row/image)
             (this is the ORACLE-ALL setting from test_time_interaction.py,
             but scored on confidence delta instead of macro-F1)

Usage:
  python confidence_delta_experiment.py \
    --weights checkpoints_revive/csr_network_final_best.pth \
    --img_dir data/isic_224x224.zip \
    --task2_dir data/ISIC-2017_Training_Part2_GroundTruth.zip \
    --task3_csv data/ISIC-2017_Training_Part3_GroundTruth.csv \
    --test_img data/ISIC-2017_Test_v2_Data.zip \
    --test_task2 data/ISIC-2017_Test_v2_Part2_GroundTruth.zip \
    --test_task3 data/ISIC-2017_Test_v2_Part3_GroundTruth.csv \
    --out checkpoints_revive/confidence_delta.pt
"""
import argparse
import collections

import numpy as np
import torch
from tqdm import tqdm

from dataset import get_dataloader, CLASS_NAMES, CONCEPT_KEYS
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


def summarize(rows_subset, tag):
    if not rows_subset:
        print(f"  {tag}: no cases")
        return
    d = np.array([r["delta_p_correct"] for r in rows_subset])
    improved = int((d > 1e-6).sum())
    worsened = int((d < -1e-6).sum())
    unchanged = len(d) - improved - worsened
    flips_to_wrong = sum(r["was_correct"] and not r["now_correct"] for r in rows_subset)
    flips_to_right = sum((not r["was_correct"]) and r["now_correct"] for r in rows_subset)
    print(f"  {tag:>32}: n={len(d):4d}  mean_delta={d.mean()*100:+6.2f}pp  "
          f"median={np.median(d)*100:+6.2f}pp  "
          f"improved={improved:4d} worsened={worsened:4d} unchanged={unchanged:4d}  "
          f"flips: correct->wrong={flips_to_wrong:3d}  wrong->correct={flips_to_right:3d}")


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

    S, gts, cgt = run(net, loader, device)
    N = S.shape[0]
    S3 = S.view(N, K, M)
    W, b = net.task_head.effective_linear()
    W, b = W.cpu(), b.cpu()

    def probs_of(S_mod):
        logits = S_mod.reshape(N, K * M) @ W.t() + b
        return torch.softmax(logits, dim=1)

    base_probs = probs_of(S3)
    base_pred = base_probs.argmax(1).numpy()
    gts_t = torch.from_numpy(gts)
    p_correct_before = base_probs[torch.arange(N), gts_t].numpy()

    sorted_p, _ = base_probs.sort(dim=1, descending=True)
    margin = (sorted_p[:, 0] - sorted_p[:, 1]).numpy()
    low_margin = margin < 0.30   # paper's own "indecisive" threshold

    absent = cgt < 0.5   # (N, K) True where concept is genuinely absent

    print(f"\n{N} images | {int(low_margin.sum())} below margin 0.30 "
          f"(paper's indecisive zone), {int((~low_margin).sum())} at/above")

    rows = []

    # ---- SINGLE: reject one absent concept at a time ---------------------
    for k in range(K):
        sel = np.where(absent[:, k])[0]
        if len(sel) == 0:
            continue
        Sk = S3.clone()
        Sk[sel, k, :] = 0.0
        probs_k = probs_of(Sk)
        p_correct_after = probs_k[torch.from_numpy(sel), gts_t[sel]].numpy()
        pred_after = probs_k.argmax(1).numpy()[sel]
        delta = p_correct_after - p_correct_before[sel]
        for j, i in enumerate(sel):
            rows.append({
                "img_idx": int(i), "concept": CONCEPT_KEYS[k], "mode": "single",
                "margin_before": float(margin[i]), "delta_p_correct": float(delta[j]),
                "was_correct": bool(base_pred[i] == gts[i]),
                "now_correct": bool(pred_after[j] == gts[i]),
            })

    # ---- ALL: reject every truly-absent concept simultaneously -----------
    mask_keep = torch.from_numpy((~absent).astype(np.float32))          # (N,K)
    S_all = S3 * mask_keep.unsqueeze(-1)
    probs_all = probs_of(S_all)
    p_correct_all = probs_all[torch.arange(N), gts_t].numpy()
    pred_all = probs_all.argmax(1).numpy()
    delta_all = p_correct_all - p_correct_before
    for i in range(N):
        if absent[i].sum() == 0:
            continue
        rows.append({
            "img_idx": i, "concept": "ALL_ABSENT", "mode": "all",
            "margin_before": float(margin[i]), "delta_p_correct": float(delta_all[i]),
            "was_correct": bool(base_pred[i] == gts[i]),
            "now_correct": bool(pred_all[i] == gts[i]),
        })

    # ---- report ------------------------------------------------------------
    for mode in ["single", "all"]:
        mode_rows = [r for r in rows if r["mode"] == mode]
        print(f"\n=== mode={mode} ===")
        summarize(mode_rows, "ALL cases")
        summarize([r for r in mode_rows if r["margin_before"] < 0.30],
                   "margin < 0.30 (paper's zone)")
        summarize([r for r in mode_rows if r["margin_before"] >= 0.30],
                   "margin >= 0.30 (paper says skip)")

        if mode == "single":
            print("\n  per-concept breakdown, margin < 0.30 only:")
            for k in range(K):
                sub = [r for r in mode_rows
                       if r["concept"] == CONCEPT_KEYS[k] and r["margin_before"] < 0.30]
                summarize(sub, f"    {CONCEPT_KEYS[k]}")
            print("\n  per-concept breakdown, margin >= 0.30 only:")
            for k in range(K):
                sub = [r for r in mode_rows
                       if r["concept"] == CONCEPT_KEYS[k] and r["margin_before"] >= 0.30]
                summarize(sub, f"    {CONCEPT_KEYS[k]}")

    torch.save({"rows": rows, "margin": margin, "gts": gts, "base_pred": base_pred},
               args.out)
    print(f"\nSaved raw per-case records -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Per-image confidence-delta concept-rejection experiment")
    add_data_args(p)
    p.add_argument("--weights", required=True)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--split", default="test", choices=["val", "test", "train"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="checkpoints/confidence_delta.pt")
    main(p.parse_args())
