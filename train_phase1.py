"""
Phase 1 - Concept model training (multi-label BCE over the K=4 concepts).

Paper: "Initially, we train the Concept model for multi-label classification
using binary cross-entropy, predicting one-hot encoded vector of ground truth
concepts." (Sec. 2.2, Training details)

Improvements vs. the original script:
  * held-out validation each epoch (concept macro-F1 + BCE), model selection on
    the validation metric instead of the training loss;
  * reproducible seeding, CPU/MPS-safe AMP, consistent split resolution.
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

from dataset import get_dataloader
from models.concept_model import ConceptModel, DEFAULT_BACKBONE
from splits import add_data_args, resolve_splits
from utils import seed_everything, resolve_device, amp_autocast, make_grad_scaler


def loader_from_spec(spec, batch_size, image_size, is_train):
    return get_dataloader(spec.img_source, spec.part3_csv, spec.part2_source,
                          batch_size=batch_size, is_train=is_train, ids=spec.ids,
                          image_size=image_size)


@torch.no_grad()
def evaluate_concepts(model, loader, device, criterion):
    model.eval()
    losses, preds, gts = [], [], []
    for images, concepts, _ in loader:
        images, concepts = images.to(device), concepts.to(device)
        with amp_autocast(device):
            logits, probs, _, _ = model(images)
            losses.append(criterion(logits, concepts).item())
        preds.append((probs > 0.5).float().cpu().numpy())
        gts.append(concepts.cpu().numpy())
    preds, gts = np.concatenate(preds), np.concatenate(gts)
    # Macro-F1 across the K concept columns.
    f1 = f1_score(gts, preds, average="macro", zero_division=0)
    return float(np.mean(losses)), float(f1)


def main(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    splits = resolve_splits(args)
    train_loader = loader_from_spec(splits["train"], args.batch_size, args.image_size, True)
    val_spec = splits["val"]
    val_loader = loader_from_spec(val_spec, args.batch_size, args.image_size, False) if val_spec else None

    model = ConceptModel(num_concepts=4, backbone=args.backbone).to(device)
    print(f"Backbone: {model.backbone_name} | feature_dim={model.feature_dim}")
    if getattr(args, "init_backbone", None):
        missing, unexpected = model.F.load_state_dict(
            torch.load(args.init_backbone, map_location=device), strict=False)
        print(f"Loaded ISIC-2019 backbone from {args.init_backbone} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # Optional per-concept positive-class weighting. Concepts are imbalanced (e.g.
    # streaks are rare), so a flat BCE lets the model minimise loss by mostly
    # predicting "absent" -- low loss, poor recall, capped F1. pos_weight upweights
    # the rarer positive class per concept so missing it costs more than a false
    # positive does, directly targeting recall on rare concepts.
    if args.class_weighted:
        ds = train_loader.dataset
        concept_sum = torch.zeros(4)
        for img_id in ds.img_names:
            concept_sum += ds._get_concepts(img_id)
        # _get_concepts lazily caches an open ZipFile on the dataset object. On
        # Linux the DataLoader forks, so every worker would inherit that same
        # open handle and read through it concurrently, corrupting the shared
        # file position/central-directory state -> BadZipFile "Overlapped
        # entries (possible zip bomb)". Close and clear so each worker opens
        # its own handle lazily, as it does on a normal run.
        for attr in ("_concept_zip", "_image_zip"):
            handle = getattr(ds, attr, None)
            if handle is not None:
                handle.close()
                setattr(ds, attr, None)
        n = len(ds.img_names)
        pos = concept_sum.clamp(min=1.0)
        neg = n - concept_sum
        pos_weight = (neg / pos).clamp(max=args.max_pos_weight)
        print(f"Concept positive counts (/{n}): {concept_sum.tolist()}")
        print(f"BCE pos_weight per concept: {pos_weight.round(decimals=2).tolist()}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = make_grad_scaler(device)

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "concept_model_phase1_best.pth")
    final_path = os.path.join(args.save_dir, "concept_model_phase1_final.pth")
    best_metric = -1.0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"[P1] Epoch {epoch+1}/{args.epochs}")
        for images, concepts, _ in pbar:
            images, concepts = images.to(device), concepts.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device):
                logits, _, _, _ = model(images)
                loss = criterion(logits, concepts)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step()
        train_loss = running / len(train_loader)

        if val_loader is not None:
            val_loss, val_f1 = evaluate_concepts(model, val_loader, device, criterion)
            print(f"[P1] Epoch {epoch+1}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_conceptF1={val_f1:.4f}")
            metric = val_f1
        else:
            print(f"[P1] Epoch {epoch+1}: train_loss={train_loss:.4f} (no val split)")
            metric = -train_loss  # fall back to training loss for selection

        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), best_path)
            print(f"    --> saved best (metric={metric:.4f})")

    torch.save(model.state_dict(), final_path)
    print(f"[P1] done. best -> {best_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 1: Concept model (multi-label BCE)")
    add_data_args(p)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   help="timm backbone; paper uses ConvNeXt-T ImageNet (default). "
                        "Try 'convnext_tiny.in12k_ft_in1k' for a stronger init.")
    p.add_argument("--init_backbone", default=None,
                   help="path to backbone_isic2019.pth to initialise F from ISIC-2019 pretraining")
    p.add_argument("--class_weighted", action="store_true",
                   help="weight BCE by inverse concept prevalence (recommended: "
                        "concepts like streaks are rare, flat BCE under-predicts them)")
    p.add_argument("--max_pos_weight", type=float, default=20.0,
                   help="cap on pos_weight so a very rare concept doesn't destabilise training")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
