"""
Phase 2 - Concept prototypes learning (projector P + prototypes p).

Paper (Sec. 2.2):
  1. Freeze the Phase-1 Concept model.
  2. For each concept k present in an image, build the local concept vector
       v_k = sum_{h,w} softmax(cam_k) * f            (Eqn 2, spatial softmax).
  3. Project v' = P(v) and pull it toward its concept's prototypes while pushing
     from other concepts' prototypes with the multi-prototype contrastive loss
     (Eqns 6-9; lambda=20, gamma=1000, delta=0.1).

Improvements: vectorised local-vector extraction (no python double loop),
validation loss + prototype-assignment accuracy, model selection on validation.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloader
from models.concept_model import ConceptModel, DEFAULT_BACKBONE
from models.projector_model import Projector, ConceptPrototypes, MultiPrototypeContrastiveLoss
from splits import add_data_args, resolve_splits
from utils import seed_everything, resolve_device, amp_autocast, make_grad_scaler


def loader_from_spec(spec, batch_size, image_size, is_train):
    return get_dataloader(spec.img_source, spec.part3_csv, spec.part2_source,
                          batch_size=batch_size, is_train=is_train, ids=spec.ids,
                          image_size=image_size)


def extract_local_concept_vectors(cam, f, concept_targets):
    """
    Vectorised Eqn 2.
      cam:(B,K,H,W)  f:(B,C,H,W)  concept_targets:(B,K) in {0,1}
    Returns v:(N,C) and labels:(N,) for every (image, present-concept) pair.
    """
    B, K, H, W = cam.shape
    soft = F.softmax(cam.reshape(B, K, H * W), dim=-1).reshape(B, K, H, W)  # spatial softmax
    v_all = torch.einsum("bkhw,bchw->bkc", soft, f)                        # (B, K, C)
    mask = concept_targets > 0.5                                          # (B, K)
    v = v_all[mask]                                                       # (N, C)
    labels = mask.nonzero(as_tuple=False)[:, 1]                          # concept index k
    return v, labels


@torch.no_grad()
def revive_dead_prototypes(concept_model, projector, prototypes, loader, device,
                           optimizer=None, noise_std=0.01):
    """
    Dead-prototype revival -- fixes winner-take-all collapse.

    WHY COLLAPSE HAPPENS (traced in the loss, not guessed):
      In MultiPrototypeContrastiveLoss, q_m = softmax(gamma * cos) with
      gamma=1000 is numerically an argmax. sim_k is therefore ~ max_m cos, so
      within a concept ONLY the single winning prototype receives gradient.
      A prototype that loses early never wins again, never gets a gradient, and
      stays frozen at its random init for the rest of training. This is a
      rich-get-richer ratchet, and it is why one prototype ends up the nearest
      match for 92-99% of a concept's training vectors while the other ~95 are
      never used.

      Note we already tested the obvious knob: lowering gamma to 100 made it
      WORSE (32 alive vs 42). That is expected in hindsight -- soft assignment
      drags every prototype toward the concept mean, which is a different
      collapse. So the fix must not touch the paper's loss at all.

    THE FIX (standard for the identical failure in VQ-VAE / Jukebox codebooks,
    where it is called "random restarts"): periodically find prototypes that no
    training vector maps to, and re-seed them at real data points so they
    re-enter the competition where data actually lives.

      WHERE to re-seed matters. We place each revived prototype at the training
      vector that is currently WORST-COVERED by the existing prototype set
      (lowest max cosine similarity), then update coverage before choosing the
      next one. That is farthest-point / k-means++ style selection: revived
      prototypes spread into the regions the concept currently fails to
      represent, instead of piling back onto the dominant cluster and dying
      again immediately.

    Returns (n_revived, alive_before, alive_after).
    """
    was_training = projector.training
    projector.eval()

    protos = prototypes.get_normalized_prototypes().float()      # (K, M, C)
    K, M, C = protos.shape

    # One pass to collect every projected training vector and its concept.
    all_v, all_lab = [], []
    for images, concept_targets, _ in loader:
        images, concept_targets = images.to(device), concept_targets.to(device)
        with amp_autocast(device):
            _, _, cam, f = concept_model(images)
        v, labels = extract_local_concept_vectors(cam.float(), f.float(), concept_targets)
        if v.numel() == 0:
            continue
        all_v.append(projector(v.unsqueeze(-1)).float())
        all_lab.append(labels)
    if not all_v:
        if was_training:
            projector.train()
        return 0, 0, 0
    v_all, lab_all = torch.cat(all_v), torch.cat(all_lab)

    n_revived = alive_before = alive_after = 0
    for k in range(K):
        sel = lab_all == k
        if sel.sum() == 0:
            continue
        vk = v_all[sel]                                  # (n_k, C) unit-norm
        sim = vk @ protos[k].t()                         # (n_k, M)
        coverage, winner = sim.max(dim=1)                # best proto per vector

        used = torch.zeros(M, dtype=torch.bool, device=device)
        used[winner.unique()] = True
        alive_before += int(used.sum())
        dead = (~used).nonzero(as_tuple=False).flatten()
        if dead.numel() == 0:
            alive_after += int(used.sum())
            continue

        # Greedy farthest-point: repeatedly seed at the worst-covered vector,
        # then refresh coverage so the next pick lands somewhere genuinely new.
        picks = []
        cover = coverage.clone()
        for _ in range(min(dead.numel(), vk.shape[0])):
            idx = int(cover.argmin())
            picks.append(idx)
            cover = torch.maximum(cover, vk @ vk[idx])   # new proto covers its neighbourhood
        if len(picks) < dead.numel():                    # more dead slots than vectors
            reps = -(-dead.numel() // max(len(picks), 1))
            picks = (picks * reps)[:dead.numel()]
        dead = dead[:len(picks)]

        new_dirs = vk[torch.tensor(picks, device=device)]
        new_dirs = new_dirs + noise_std * torch.randn_like(new_dirs)
        prototypes.prototypes.data[k, dead] = new_dirs

        # Stale AdamW moments would immediately drag a revived prototype back
        # toward wherever its dead predecessor was heading. Clear them.
        if optimizer is not None:
            state = optimizer.state.get(prototypes.prototypes, None)
            if state:
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        state[key][k, dead] = 0.0

        n_revived += dead.numel()
        alive_after += int(used.sum()) + dead.numel()

    if was_training:
        projector.train()
    return n_revived, alive_before, alive_after


@torch.no_grad()
def evaluate(concept_model, projector, prototypes, criterion, loader, device):
    projector.eval()
    losses, correct, total = [], 0, 0
    for images, concept_targets, _ in loader:
        images, concept_targets = images.to(device), concept_targets.to(device)
        with amp_autocast(device):
            _, _, cam, f = concept_model(images)
        v, labels = extract_local_concept_vectors(cam.float(), f.float(), concept_targets)
        if v.numel() == 0:
            continue
        with amp_autocast(device):
            v_prime = projector(v.unsqueeze(-1))
            protos = prototypes.get_normalized_prototypes()
            losses.append(criterion(v_prime, labels, protos).item())
        # prototype-assignment accuracy: does the nearest prototype belong to the true concept?
        cos = torch.einsum("nc,kmc->nkm", v_prime.float(), protos.float())
        pred_concept = cos.amax(dim=-1).argmax(dim=-1)
        correct += (pred_concept == labels).sum().item()
        total += labels.numel()
    acc = correct / total if total else 0.0
    return float(np.mean(losses)) if losses else 0.0, acc


def main(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    splits = resolve_splits(args)
    train_loader = loader_from_spec(splits["train"], args.batch_size, args.image_size, True)
    val_spec = splits["val"]
    val_loader = loader_from_spec(val_spec, args.batch_size, args.image_size, False) if val_spec else None

    concept_model = ConceptModel(num_concepts=4, backbone=args.backbone).to(device)
    concept_model.load_state_dict(torch.load(args.phase1_weights, map_location=device))
    concept_model.eval()
    for prm in concept_model.parameters():
        prm.requires_grad = False

    projector = Projector(in_dim=concept_model.feature_dim, out_dim=args.proj_dim).to(device)
    prototypes = ConceptPrototypes(num_concepts=4, M=args.M, dim=args.proj_dim).to(device)

    # Prototypes must NOT get weight decay: they are learned reference points, not
    # regularized weights. Decaying them shrinks rarely-winning prototypes toward the
    # origin, where after L2-normalization they scatter and die -- a ratchet that
    # causes prototype collapse (few "alive" prototypes, most unused). Only the
    # projector gets weight decay.
    optimizer = optim.AdamW(
        [{"params": projector.parameters(), "weight_decay": args.weight_decay},
         {"params": prototypes.parameters(), "weight_decay": 0.0}],
        lr=args.lr)

    class_weight = None
    if args.class_weighted:
        # One training vector per (image, present-concept) pair, so counting
        # concept presence across images = counting the vectors themselves.
        ds = train_loader.dataset
        counts = torch.zeros(4)
        for img_id in ds.img_names:
            counts += ds._get_concepts(img_id)
        # Same fork hazard as Phase 1: close any zip handle opened by the loop
        # above before the DataLoader forks workers, or they inherit and race
        # on it -> BadZipFile "Overlapped entries (possible zip bomb)".
        for attr in ("_concept_zip", "_image_zip"):
            handle = getattr(ds, attr, None)
            if handle is not None:
                handle.close()
                setattr(ds, attr, None)
        total = counts.sum().item()
        class_weight = (total / (4 * counts.clamp(min=1.0))).to(device)
        print(f"Concept vector counts: {counts.tolist()}")
        print(f"Phase-2 class weight per concept: {class_weight.round(decimals=3).tolist()}")

    criterion = MultiPrototypeContrastiveLoss(lambda_scale=args.lambda_scale,
                                              gamma=args.gamma, delta=args.delta,
                                              class_weight=class_weight,
                                              diversity_beta=args.diversity_beta,
                                              diversity_gamma=args.diversity_gamma)
    scaler = make_grad_scaler(device)

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "phase2_best.pth")
    best_metric = -1.0

    for epoch in range(args.epochs):
        projector.train(); prototypes.train()
        running, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"[P2] Epoch {epoch+1}/{args.epochs}")
        for images, concept_targets, _ in pbar:
            images, concept_targets = images.to(device), concept_targets.to(device)
            with torch.no_grad(), amp_autocast(device):
                _, _, cam, f = concept_model(images)
            v, labels = extract_local_concept_vectors(cam.float(), f.float(), concept_targets)
            if v.numel() == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device):
                v_prime = projector(v.unsqueeze(-1))
                protos = prototypes.get_normalized_prototypes()
                loss = criterion(v_prime, labels, protos)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * labels.numel()
            seen += labels.numel()
            pbar.set_postfix(loss=f"{loss.item():.4f}", vectors=seen)
        train_loss = running / seen if seen else 0.0

        # Revive dead prototypes. Skipped in the final stretch so that anything
        # revived still gets enough epochs to actually train before we stop.
        if (args.revive_every > 0
                and (epoch + 1) % args.revive_every == 0
                and (args.epochs - (epoch + 1)) >= args.revive_every):
            n_rev, alive_b, alive_a = revive_dead_prototypes(
                concept_model, projector, prototypes, train_loader, device,
                optimizer=optimizer, noise_std=args.revive_noise)
            print(f"    [revive] alive {alive_b}/{4*args.M} -> re-seeded {n_rev} "
                  f"dead prototypes at worst-covered training vectors")

        if val_loader is not None:
            val_loss, val_acc = evaluate(concept_model, projector, prototypes,
                                         criterion, val_loader, device)
            div_str = (f" div={criterion.last_diversity:.3f}"
                       if args.diversity_beta > 0 else "")
            print(f"[P2] Epoch {epoch+1}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_protoAcc={val_acc:.4f}{div_str}")
            metric = val_acc
        else:
            print(f"[P2] Epoch {epoch+1}: train_loss={train_loss:.4f} (no val split)")
            metric = -train_loss

        if metric > best_metric:
            best_metric = metric
            torch.save({"projector": projector.state_dict(),
                        "prototypes": prototypes.state_dict(),
                        "proj_dim": args.proj_dim, "M": args.M}, best_path)
            print(f"    --> saved best (metric={metric:.4f})")

    print(f"[P2] done. best -> {best_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 2: Projector + prototypes (contrastive)")
    add_data_args(p)
    p.add_argument("--phase1_weights", required=True)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--M", type=int, default=100, help="prototypes per concept (paper: 100)")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--gamma", type=float, default=1000.0,
                   help="prototype-assignment softmax sharpness (paper: 1000; "
                        "lower (e.g. 100) reduces winner-take-all prototype collapse)")
    p.add_argument("--lambda_scale", type=float, default=20.0,
                   help="contrastive logit scale (paper: 20)")
    p.add_argument("--delta", type=float, default=0.1, help="margin (paper: 0.1)")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="weight decay for the projector only; prototypes always get 0")
    p.add_argument("--revive_every", type=int, default=0,
                   help="re-seed unused prototypes every N epochs (0=off). Fixes "
                        "winner-take-all collapse; try 3. See revive_dead_prototypes.")
    p.add_argument("--revive_noise", type=float, default=0.01,
                   help="gaussian jitter added when re-seeding a revived prototype")
    p.add_argument("--diversity_beta", type=float, default=0.0,
                   help="load-balancing penalty weight (0=off; try 0.01). Spreads a "
                        "concept's vectors across prototypes so none is load-bearing, "
                        "making doctor rejection safe. See MultiPrototypeContrastiveLoss.")
    p.add_argument("--diversity_gamma", type=float, default=10.0,
                   help="soft softmax temperature for the diversity term (must be << "
                        "gamma=1000, whose softmax is saturated and has no gradient)")
    p.add_argument("--class_weighted", action="store_true",
                   help="weight the Phase-2 cross-entropy by inverse concept-vector "
                        "frequency (rare concepts like streaks/negative_network are "
                        "~6%% of training vectors each and get undertrained otherwise)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
