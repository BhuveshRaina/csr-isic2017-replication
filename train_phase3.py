"""
Phase 3 - Task head training.

Paper: "the task head is trained to predict the target from similarity scores
with the concept prototypes, using cross-entropy." (Sec. 2.2). Everything except
the single linear task head H is frozen.

Model selection is on VALIDATION macro-F1 (the paper's metric, chosen because of
heavy class imbalance), NOT on training loss. This is the key correctness fix:
the original script measured F1 on the training set it was fitting, which is not
comparable to the paper's test-set number.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import get_dataloader, CLASS_NAMES
from models.concept_model import ConceptModel, DEFAULT_BACKBONE
from models.projector_model import Projector, ConceptPrototypes
from models.csr_network import TaskHead, CSRNetwork
from splits import add_data_args, resolve_splits
from utils import seed_everything, resolve_device, amp_autocast, make_grad_scaler


def loader_from_spec(spec, batch_size, image_size, is_train):
    return get_dataloader(spec.img_source, spec.part3_csv, spec.part2_source,
                          batch_size=batch_size, is_train=is_train, ids=spec.ids,
                          image_size=image_size)


@torch.no_grad()
def evaluate(csr_net, loader, device):
    csr_net.eval()
    preds, gts = [], []
    for images, _, targets in loader:
        images = images.to(device)
        with amp_autocast(device):
            logits, _, _ = csr_net(images)
        preds.append(logits.argmax(1).cpu().numpy())
        gts.append(targets.numpy())
    preds, gts = np.concatenate(preds), np.concatenate(gts)
    macro_f1 = f1_score(gts, preds, average="macro", zero_division=0)
    acc = (preds == gts).mean() * 100
    return macro_f1, acc


@torch.no_grad()
def build_absent_profile(csr_net, loader, device, K, M, fill="median"):
    """
    Per-prototype 'typical absent' similarity, from the training set. This is
    the value we substitute for a dropped concept -- an in-distribution stand-in
    for 'this concept is absent', instead of an out-of-distribution zero.
    """
    if fill == "zero":
        return torch.zeros(K, M)
    csr_net.eval()
    S, C = [], []
    for images, concepts, _ in tqdm(loader, desc="[P3] absent-profile"):
        _, s_flat, _ = csr_net(images.to(device))
        S.append(s_flat.float().cpu())
        C.append(concepts)
    S = torch.cat(S).view(-1, K, M)
    C = torch.cat(C)
    absent = C < 0.5
    prof = torch.zeros(K, M)
    for k in range(K):
        block = S[absent[:, k], k, :]
        prof[k] = block.median(0).values if fill == "median" else block.mean(0)
    return prof


def apply_concept_dropout(s_flat, concepts, prof, p_drop, n_drop, K, M):
    """
    Concept-dropout augmentation. For each image (with probability p_drop),
    pick n_drop of its TRULY-ABSENT concepts (ground-truth label 0) and replace
    their M similarity scores with the 'typical absent' profile. Training the
    task head to still predict the correct disease under this teaches it to (a)
    not over-rely on any single concept's scores and (b) redistribute weight to
    the remaining concepts when one is removed -- i.e. to actually re-reason when
    a doctor rejects a concept, which a plain linear head never learns to do.
    """
    B = s_flat.shape[0]
    s = s_flat.float().view(B, K, M).clone()
    absent = (concepts < 0.5).cpu()                      # (B, K)
    for i in range(B):
        if torch.rand(()).item() >= p_drop:
            continue
        ks = torch.nonzero(absent[i], as_tuple=False).flatten()
        if ks.numel() == 0:
            continue
        sel = ks[torch.randperm(ks.numel())[:n_drop]].to(s.device)
        s[i, sel, :] = prof[sel].to(s.dtype)
    return s.view(B, K * M)


def main(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    splits = resolve_splits(args)
    train_loader = loader_from_spec(splits["train"], args.batch_size, args.image_size, True)
    val_spec = splits["val"]
    val_loader = loader_from_spec(val_spec, args.batch_size, args.image_size, False) if val_spec else None

    num_classes = len(CLASS_NAMES)
    counts = np.bincount(np.array(train_loader.dataset.targets), minlength=num_classes)
    print(f"Train class distribution {CLASS_NAMES} = {counts.tolist()}")

    # --- sub-models (frozen) ---
    concept_model = ConceptModel(num_concepts=4, backbone=args.backbone).to(device)
    ckpt2 = torch.load(args.phase2_weights, map_location=device)
    proj_dim = ckpt2.get("proj_dim", 256)
    M = ckpt2.get("M", 100)
    projector = Projector(in_dim=concept_model.feature_dim, out_dim=proj_dim).to(device)
    prototypes = ConceptPrototypes(num_concepts=4, M=M, dim=proj_dim).to(device)
    task_head = TaskHead(in_features=4 * M, num_classes=num_classes,
                         normalize_input=not args.no_input_norm).to(device)

    concept_model.load_state_dict(torch.load(args.phase1_weights, map_location=device))
    projector.load_state_dict(ckpt2["projector"])
    prototypes.load_state_dict(ckpt2["prototypes"])
    for m in (concept_model, projector, prototypes):
        m.eval()
        for prm in m.parameters():
            prm.requires_grad = False

    csr_net = CSRNetwork(concept_model, projector, prototypes, task_head).to(device)

    K = 4
    absent_profile = None
    if args.concept_dropout > 0.0:
        absent_profile = build_absent_profile(csr_net, train_loader, device, K, M,
                                              fill=args.dropout_fill).to(device)
        print(f"Concept-dropout ON: p={args.concept_dropout}, n_drop={args.dropout_n}, "
              f"fill={args.dropout_fill}. Model learns to predict with a rejected "
              f"concept removed (delivers usable doctor-in-the-loop).")

    # Faithful default: plain cross-entropy. Optional class weighting for imbalance.
    weight = None
    if args.class_weighted:
        w = counts.sum() / (num_classes * np.clip(counts, 1, None))
        weight = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"Using class-weighted CE: {w.round(3).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weight)
    # Weight decay on a 400->3 head actively suppresses an already-weak gradient
    # signal; default to 0 (override with --weight_decay if you want it back).
    optimizer = optim.AdamW(csr_net.task_head.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = make_grad_scaler(device)

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "csr_network_final_best.pth")
    final_path = os.path.join(args.save_dir, "csr_network_final.pth")
    best_f1 = -1.0

    for epoch in range(args.epochs):
        csr_net.task_head.train()
        running, preds, gts = 0.0, [], []
        pbar = tqdm(train_loader, desc=f"[P3] Epoch {epoch+1}/{args.epochs}")
        for images, concepts, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device):
                logits, s_flat, _ = csr_net(images)
                if absent_profile is not None:
                    s_in = apply_concept_dropout(s_flat, concepts, absent_profile,
                                                 args.concept_dropout, args.dropout_n, K, M)
                    logits = csr_net.task_head(s_in.to(images.dtype))
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            preds.append(logits.argmax(1).detach().cpu().numpy())
            gts.append(targets.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step()

        tr_f1 = f1_score(np.concatenate(gts), np.concatenate(preds), average="macro", zero_division=0)
        if val_loader is not None:
            val_f1, val_acc = evaluate(csr_net, val_loader, device)
            print(f"[P3] Epoch {epoch+1}: train_loss={running/len(train_loader):.4f} "
                  f"train_F1={tr_f1:.4f} | val_F1={val_f1:.4f} val_acc={val_acc:.2f}%")
            metric = val_f1
        else:
            print(f"[P3] Epoch {epoch+1}: train_loss={running/len(train_loader):.4f} train_F1={tr_f1:.4f}")
            metric = tr_f1

        if metric > best_f1:
            best_f1 = metric
            torch.save(csr_net.state_dict(), best_path)
            print(f"    --> saved best (macro-F1={metric:.4f})")

    torch.save(csr_net.state_dict(), final_path)
    print(f"[P3] done. best val macro-F1={best_f1:.4f} -> {best_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 3: Task head (cross-entropy)")
    add_data_args(p)
    p.add_argument("--phase1_weights", required=True)
    p.add_argument("--phase2_weights", required=True)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--no_input_norm", action="store_true",
                   help="disable BatchNorm standardisation of the similarity scores "
                        "(reproduces the collapsed-head behaviour; not recommended)")
    p.add_argument("--class_weighted", action="store_true",
                   help="recommended: training is 1372/374/254 imbalanced and the "
                        "metric is macro-F1")
    p.add_argument("--concept_dropout", type=float, default=0.0,
                   help="probability of removing a truly-absent concept from an "
                        "image during training (0=off). Teaches the head to support "
                        "doctor-in-the-loop concept rejection. Try 0.5.")
    p.add_argument("--dropout_n", type=int, default=1,
                   help="how many absent concepts to drop per selected image")
    p.add_argument("--dropout_fill", default="median", choices=["median", "mean", "zero"],
                   help="value substituted for a dropped concept (median=safe, "
                        "in-distribution; zero=paper's brittle version)")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
