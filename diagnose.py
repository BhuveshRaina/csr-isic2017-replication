"""
Diagnostic for the CSR pipeline: find WHERE the signal is lost.

Run this after training. It answers four questions in order:

  Q1. Did the Phase-1 concept model learn anything?
      -> per-concept F1/AUC on the test set. If this is near chance, every
         downstream stage is doomed and Phase 1 is the thing to fix.

  Q2. Does the backbone contain disease information at all?
      -> logistic-regression probe on pooled backbone features (GAP(f)).
         This is the CEILING for anything built on this frozen backbone.

  Q3. Do the 400 similarity scores contain disease information?
      -> statistics (saturation, dynamic range, per-class separation) plus a
         balanced logistic-regression probe on s. If the probe reaches a decent
         macro-F1 but your trained task head did not, the head/loss is the
         problem (class imbalance). If the probe ALSO collapses, the similarity
         features themselves are uninformative -> Phase 2 / projector problem.

  Q4. Is there a train/test distribution shift in the projector?
      -> compares projector outputs on concept vectors (what Phase 2 trained on)
         vs. raw feature patches (what inference feeds it). A large gap points at
         the IBN BatchNorm running statistics.

Usage (Colab):
  python diagnose.py \
    --img_dir data/isic_224x224.zip \
    --task2_dir data/ISIC-2017_Training_Part2_GroundTruth.zip \
    --task3_csv data/ISIC-2017_Training_Part3_GroundTruth.csv \
    --test_img data/ISIC-2017_Test_v2_Data.zip \
    --test_task2 data/ISIC-2017_Test_v2_Part2_GroundTruth.zip \
    --test_task3 data/ISIC-2017_Test_v2_Part3_GroundTruth.csv \
    --phase1_weights checkpoints/concept_model_phase1_best.pth \
    --phase2_weights checkpoints/phase2_best.pth \
    --csr_weights checkpoints/csr_network_final_best.pth
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

from dataset import get_dataloader, CLASS_NAMES, CONCEPT_KEYS
from models.concept_model import ConceptModel, DEFAULT_BACKBONE
from models.projector_model import Projector, ConceptPrototypes
from models.csr_network import TaskHead, CSRNetwork
from train_phase2 import extract_local_concept_vectors
from utils import resolve_device, seed_everything


def hr(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74, flush=True)


@torch.no_grad()
def collect(concept_model, projector, prototypes, loader, device, max_batches=None):
    """Gather backbone features, concept preds, similarity scores, labels."""
    feats, cpred, cgt, sims, ys = [], [], [], [], []
    protos = prototypes.get_normalized_prototypes()
    K, M, _ = protos.shape
    for i, (images, concepts, targets) in enumerate(tqdm(loader, desc="collect")):
        if max_batches and i >= max_batches:
            break
        images = images.to(device)
        # fp32 everywhere: we are diagnosing numerics, no AMP.
        _, c_prob, cam, f = concept_model(images)
        B, C, H, W = f.shape
        feats.append(f.mean(dim=(2, 3)).cpu().numpy())          # GAP(f)
        cpred.append(c_prob.cpu().numpy())
        cgt.append(concepts.numpy())

        patches = f.view(B, C, H * W).permute(0, 2, 1).reshape(B * H * W, C, 1)
        fp = projector(patches).view(B, H * W, -1)
        S = torch.einsum("bnc,kmc->bnkm", fp, protos)            # (B,HW,K,M)
        s = S.amax(dim=1).reshape(B, K * M)                      # (B,K*M)
        sims.append(s.cpu().numpy())
        ys.append(targets.numpy())
    return (np.concatenate(feats), np.concatenate(cpred), np.concatenate(cgt),
            np.concatenate(sims), np.concatenate(ys))


def probe(X_tr, y_tr, X_te, y_te, tag, balanced=True):
    """Balanced logistic-regression probe -> macro F1 achievable from X."""
    mu, sd = X_tr.mean(0, keepdims=True), X_tr.std(0, keepdims=True) + 1e-8
    clf = LogisticRegression(max_iter=3000, multi_class="multinomial",
                             class_weight="balanced" if balanced else None)
    clf.fit((X_tr - mu) / sd, y_tr)
    pred = clf.predict((X_te - mu) / sd)
    f1 = f1_score(y_te, pred, average="macro", zero_division=0)
    print(f"  {tag:<52s} macro-F1 = {f1*100:5.2f}%")
    print(f"  {'':<52s} preds/class = {np.bincount(pred, minlength=3).tolist()}")
    return f1


def main(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"device = {device}", flush=True)

    # ---------------- models ----------------
    ck2 = torch.load(args.phase2_weights, map_location=device)
    proj_dim, M = ck2.get("proj_dim", 256), ck2.get("M", 100)
    concept_model = ConceptModel(num_concepts=4, backbone=args.backbone,
                                 pretrained=False).to(device).eval()
    concept_model.load_state_dict(torch.load(args.phase1_weights, map_location=device))
    projector = Projector(in_dim=concept_model.feature_dim, out_dim=proj_dim).to(device).eval()
    prototypes = ConceptPrototypes(num_concepts=4, M=M, dim=proj_dim).to(device).eval()
    projector.load_state_dict(ck2["projector"])
    prototypes.load_state_dict(ck2["prototypes"])

    # ---------------- data ----------------
    train_loader = get_dataloader(args.img_dir, args.task3_csv, args.task2_dir,
                                  batch_size=args.batch_size, is_train=False,
                                  image_size=args.image_size, shuffle=False)
    test_loader = get_dataloader(args.test_img, args.test_task3, args.test_task2,
                                 batch_size=args.batch_size, is_train=False,
                                 image_size=args.image_size, shuffle=False)

    hr("COLLECTING FEATURES")
    Ftr, Ctr, Gtr, Str, Ytr = collect(concept_model, projector, prototypes,
                                      train_loader, device, args.max_batches)
    Fte, Cte, Gte, Ste, Yte = collect(concept_model, projector, prototypes,
                                      test_loader, device, args.max_batches)
    print(f"train {Ftr.shape[0]} imgs | test {Fte.shape[0]} imgs")
    print(f"test class counts {CLASS_NAMES} = {np.bincount(Yte, minlength=3).tolist()}")

    # ---------------- Q1: concept model ----------------
    hr("Q1. Phase-1 concept model quality (test set)")
    bad = 0
    for k, name in enumerate(CONCEPT_KEYS):
        gt, pr = Gte[:, k], Cte[:, k]
        if gt.sum() == 0 or gt.sum() == len(gt):
            print(f"  {name:>20s}: (degenerate on test)")
            continue
        auc = roc_auc_score(gt, pr)
        f1 = f1_score(gt, (pr > 0.5).astype(float), zero_division=0)
        flag = "  <-- weak" if auc < 0.65 else ""
        bad += auc < 0.65
        print(f"  {name:>20s}: AUC={auc:.3f}  F1={f1*100:5.1f}%  "
              f"prevalence={gt.mean()*100:4.1f}%{flag}")
    print(f"\n  concept prob range: [{Cte.min():.3f}, {Cte.max():.3f}]  mean={Cte.mean():.3f}")
    if bad >= 3:
        print("  VERDICT: concept model is near chance -> fix Phase 1 first.")
    else:
        print("  VERDICT: concept model has signal.")

    # ---------------- Q2: backbone ceiling ----------------
    hr("Q2. Does the frozen backbone contain DISEASE signal? (ceiling)")
    probe(Ftr, Ytr, Fte, Yte, "logreg on GAP(backbone features)")

    # ---------------- Q3: similarity features ----------------
    hr("Q3. Do the 400 similarity scores contain disease signal?")
    print(f"  s range   : [{Ste.min():.4f}, {Ste.max():.4f}]")
    print(f"  s mean/std: {Ste.mean():.4f} / {Ste.std():.4f}")
    print(f"  per-image std of s (mean): {Ste.std(axis=1).mean():.4f}")
    print(f"  per-feature std across imgs (mean): {Ste.std(axis=0).mean():.4f}")
    dead = int((Ste.std(axis=0) < 1e-4).sum())
    print(f"  near-constant features: {dead}/{Ste.shape[1]}")
    if Ste.std(axis=0).mean() < 1e-3:
        print("  !! similarity scores barely vary across images -> collapsed features")
    # class separation
    for k, name in enumerate(CLASS_NAMES):
        m = Ste[Yte == k].mean()
        print(f"  mean s | {name:>22s} = {m:.4f}")
    print()
    probe(Str, Ytr, Ste, Yte, "logreg on similarity scores s (balanced)")
    probe(Str, Ytr, Ste, Yte, "logreg on similarity scores s (unweighted)", balanced=False)

    # ---------------- your trained head ----------------
    if args.csr_weights:
        hr("Your trained task head, for comparison")
        state = torch.load(args.csr_weights, map_location=device)
        th = TaskHead(in_features=4 * M, num_classes=len(CLASS_NAMES),
                      normalize_input="task_head.norm.weight" in state).to(device)
        th.load_state_dict({k.split("task_head.")[1]: v for k, v in state.items()
                            if k.startswith("task_head.")})
        th.eval()
        with torch.no_grad():
            logits = th(torch.tensor(Ste, dtype=torch.float32, device=device))
        pred = logits.argmax(1).cpu().numpy()
        print(f"  macro-F1 = {f1_score(Yte, pred, average='macro', zero_division=0)*100:.2f}%")
        print(f"  preds/class = {np.bincount(pred, minlength=3).tolist()}")
        W = th.linear.weight.detach().cpu().numpy()
        print(f"  head weight |W| mean={np.abs(W).mean():.5f}  bias={th.linear.bias.detach().cpu().numpy().round(3).tolist()}")

    # ---------------- Q4: projector distribution shift ----------------
    hr("Q4. Projector train/test distribution shift (IBN BatchNorm)")
    images, concepts, _ = next(iter(train_loader))
    images, concepts = images.to(device), concepts.to(device)
    with torch.no_grad():
        _, _, cam, f = concept_model(images)
        v, labels = extract_local_concept_vectors(cam.float(), f.float(), concepts)
        B, C, H, W = f.shape
        patches = f.view(B, C, H * W).permute(0, 2, 1).reshape(B * H * W, C, 1)
        pv = projector(v.unsqueeze(-1)) if v.numel() else None
        pp = projector(patches)
        # pre-projection input statistics
        print(f"  INPUT  concept vectors v : mean={v.mean():.4f} std={v.std():.4f}"
              if v.numel() else "  (no concept vectors in this batch)")
        print(f"  INPUT  feature patches   : mean={patches.mean():.4f} std={patches.std():.4f}")
        if pv is not None:
            # cosine between the two projected populations' centroids
            cv = F.normalize(pv.mean(0), dim=0)
            cp = F.normalize(pp.mean(0), dim=0)
            print(f"  OUTPUT centroid cosine(v', patch') = {(cv*cp).sum().item():.4f}")
            print(f"  OUTPUT v'  std across dims = {pv.std(0).mean():.4f}")
        print(f"  OUTPUT patch' std across dims = {pp.std(0).mean():.4f}")
        # how well do prototypes match each population?
        protos = prototypes.get_normalized_prototypes()
        if pv is not None:
            sv = torch.einsum("nc,kmc->nkm", pv, protos).amax(dim=(1, 2))
            print(f"  max sim to any prototype | concept vectors : {sv.mean().item():.4f}")
        sp = torch.einsum("nc,kmc->nkm", pp, protos).amax(dim=(1, 2))
        print(f"  max sim to any prototype | feature patches  : {sp.mean().item():.4f}")
        print("\n  If the two 'max sim' numbers differ a lot, the prototypes were")
        print("  learned in a different regime than inference operates in ->")
        print("  BatchNorm running-stat mismatch is the culprit.")

    hr("SUMMARY / WHAT TO FIX")
    print("""
 Read the numbers above in this order:

  * Q1 weak (AUC ~0.5)      -> Phase 1 failed. Fix concepts before anything else.
  * Q2 low (<45%)           -> the frozen backbone has no disease signal. Concept-only
                               pretraining is too weak here; ISIC-2019 pretraining or
                               unfreezing the backbone is required.
  * Q3 probe HIGH, your head LOW  -> features are fine, the TASK HEAD collapsed:
                               retrain Phase 3 with --class_weighted.
  * Q3 probe LOW too        -> the similarity features are uninformative: Phase 2 /
                               projector problem (see Q4).
  * Q4 'max sim' very different between concept vectors and patches
                            -> BatchNorm mismatch; run fix_bn.py to recalibrate.
""")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", required=True)
    p.add_argument("--task2_dir", required=True)
    p.add_argument("--task3_csv", required=True)
    p.add_argument("--test_img", required=True)
    p.add_argument("--test_task2", required=True)
    p.add_argument("--test_task3", required=True)
    p.add_argument("--phase1_weights", required=True)
    p.add_argument("--phase2_weights", required=True)
    p.add_argument("--csr_weights", default=None)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_batches", type=int, default=None,
                   help="limit batches for a quick run")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
