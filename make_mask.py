"""
Build a prototype mask for evaluate_curated.py without clicking through the UI.

Two ways to specify what to discard:

  --discard "0:46,1:3,1:44,2:87,3:23"
      Explicit (k, m) pairs. Use for the clinician-curation experiment:
      discard prototypes whose atlas exemplar shows an acquisition artefact
      (ruler, ink marker, vignette) rather than skin pathology.

  --min_hits 2
      Discard every prototype whose atlas hit_count is below the threshold.
      With 2, this removes the "won exactly one training vector" prototypes,
      which carry almost no traffic. Useful as a CONTROL: if removing these
      barely moves macro-F1, then any drop from the explicit artefact list
      is attributable to artefact reliance, not to "removing prototypes hurts".

Both can be combined. Prints a summary of what survives.

Example:
  python make_mask.py --atlas checkpoints_deduped/atlas.pt \
      --discard "0:46,1:3,1:44,2:87,3:23" \
      --out checkpoints_deduped/mask_artifacts.pt
"""
import argparse

import torch

from dataset import CONCEPT_KEYS


def parse_pairs(text):
    pairs = set()
    if not text:
        return pairs
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        k, m = chunk.split(":")
        pairs.add((int(k), int(m)))
    return pairs


def main(args):
    atlas = torch.load(args.atlas, map_location="cpu")
    K, M = int(atlas["K"]), int(atlas["M"])
    hits = atlas["hit_count"]
    ids = list(atlas["image_ids"])
    best = atlas["best_image_idx"]

    discard = parse_pairs(args.discard)
    if args.min_hits > 0:
        for k in range(K):
            for m in range(M):
                h = int(hits[k, m])
                if 0 < h < args.min_hits:      # alive but low-traffic
                    discard.add((k, m))

    mask = torch.ones(K, M)
    for k, m in discard:
        if not (0 <= k < K and 0 <= m < M):
            raise ValueError(f"prototype ({k},{m}) out of range for K={K}, M={M}")
        mask[k, m] = 0.0

    print(f"Atlas: K={K}, M={M}")
    print(f"Discarding {len(discard)} prototypes\n")
    for k in range(K):
        alive = [m for m in range(M) if int(hits[k, m]) > 0]
        kept = [m for m in alive if (k, m) not in discard]
        kept_traffic = sum(int(hits[k, m]) for m in kept)
        total_traffic = sum(int(hits[k, m]) for m in alive)
        print(f"[k={k}] {CONCEPT_KEYS[k]:<18s} alive {len(alive):>3d} -> kept {len(kept):>3d}"
              f" | training vectors covered {kept_traffic}/{total_traffic}")
        for m in sorted(alive, key=lambda m: -int(hits[k, m]))[:args.show]:
            idx = int(best[k, m])
            img = ids[idx] if 0 <= idx < len(ids) else "?"
            flag = "DISCARD" if (k, m) in discard else "keep   "
            print(f"      m={m:<3d} hits={int(hits[k,m]):>4d}  {flag}  {img}")
        print()

    torch.save({"prototype_mask": mask, "discarded": sorted(discard)}, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build a curated prototype mask")
    p.add_argument("--atlas", default="checkpoints_deduped/atlas.pt")
    p.add_argument("--out", default="checkpoints_deduped/discarded_prototypes.pt")
    p.add_argument("--discard", default="", help='e.g. "0:46,1:3,1:44"')
    p.add_argument("--min_hits", type=int, default=0,
                   help="discard alive prototypes with hit_count below this")
    p.add_argument("--show", type=int, default=6, help="rows to print per concept")
    main(p.parse_args())
