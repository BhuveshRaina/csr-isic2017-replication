"""
Prototype projection / concept atlas construction (Sec. 3.1, Eqn 11).

Each learned prototype p_km is linked to the training image whose projected local
concept vector v' is *nearest* to it (highest cosine similarity). This produces
the interpretable atlas  I(p_km) = x_i  used for explanations and for train-time
atlas refinement (Sec. 3.2).

Note on Eqn 11: the paper prints `argmin<v', p>` but "nearest" prototype means
the LARGEST cosine similarity, so we use argmax (a sign typo in the paper).

Outputs checkpoints/atlas.pt with, per (k, m):
    image_id, similarity, concept index, and how many vectors mapped to it
    (a prototype that nothing maps to is a candidate to discard when refining).
"""
import argparse

import torch
from tqdm import tqdm

from dataset import get_dataloader, CONCEPT_KEYS, CLASS_NAMES  # noqa: F401
from models.concept_model import ConceptModel, DEFAULT_BACKBONE
from models.projector_model import Projector, ConceptPrototypes
from train_phase2 import extract_local_concept_vectors
from splits import add_data_args, resolve_splits
from utils import seed_everything, resolve_device, amp_autocast


@torch.no_grad()
def main(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    splits = resolve_splits(args)
    train = splits["train"]

    loader = get_dataloader(train.img_source, train.part3_csv, train.part2_source,
                            batch_size=args.batch_size, is_train=False, ids=train.ids,
                            image_size=args.image_size, shuffle=False)
    image_ids = loader.dataset.img_names

    concept_model = ConceptModel(num_concepts=4, backbone=args.backbone).to(device).eval()
    concept_model.load_state_dict(torch.load(args.phase1_weights, map_location=device))
    ckpt2 = torch.load(args.phase2_weights, map_location=device)
    proj_dim, M = ckpt2.get("proj_dim", 256), ckpt2.get("M", 100)
    projector = Projector(in_dim=concept_model.feature_dim, out_dim=proj_dim).to(device).eval()
    prototypes = ConceptPrototypes(num_concepts=4, M=M, dim=proj_dim).to(device).eval()
    projector.load_state_dict(ckpt2["projector"])
    prototypes.load_state_dict(ckpt2["prototypes"])
    protos = prototypes.get_normalized_prototypes()          # (K, M, C')
    K = protos.shape[0]

    best_sim = torch.full((K, M), -2.0, device=device)
    best_img = torch.full((K, M), -1, dtype=torch.long, device=device)
    hit_count = torch.zeros((K, M), dtype=torch.long, device=device)

    sample_offset = 0
    for images, concept_targets, _ in tqdm(loader, desc="Building atlas"):
        images, concept_targets = images.to(device), concept_targets.to(device)
        with amp_autocast(device):
            _, _, cam, f = concept_model(images)
        v, labels = extract_local_concept_vectors(cam.float(), f.float(), concept_targets)
        # map vector position back to its source image index within the batch
        mask = concept_targets > 0.5
        batch_img_idx = mask.nonzero(as_tuple=False)[:, 0] + sample_offset
        if v.numel() > 0:
            v_prime = projector(v.unsqueeze(-1)).float()      # (N, C')
            # per concept, compare only against that concept's prototypes
            for k in range(K):
                sel = labels == k
                if sel.sum() == 0:
                    continue
                vp = v_prime[sel]                              # (n_k, C')
                sim = vp @ protos[k].t()                       # (n_k, M)
                max_sim, arg = sim.max(dim=0)                  # best vector per prototype
                nearest_proto = sim.argmax(dim=1)              # each vector's nearest proto
                for m in nearest_proto.tolist():
                    hit_count[k, m] += 1
                improve = max_sim > best_sim[k]
                best_sim[k][improve] = max_sim[improve]
                src = batch_img_idx[sel][arg]
                best_img[k][improve] = src[improve]
        sample_offset += images.size(0)

    atlas = {"image_ids": image_ids, "K": K, "M": M,
             "best_image_idx": best_img.cpu(), "best_similarity": best_sim.cpu(),
             "hit_count": hit_count.cpu(),
             "concept_keys": CONCEPT_KEYS}
    torch.save(atlas, args.out)
    dead = int((hit_count == 0).sum().item())
    print(f"Atlas saved to {args.out}. {dead}/{K*M} prototypes had no nearest vector "
          f"(candidates to discard when refining).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the concept prototype atlas (Eqn 11)")
    add_data_args(p)
    p.add_argument("--phase1_weights", required=True)
    p.add_argument("--phase2_weights", required=True)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--out", default="checkpoints/atlas.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
