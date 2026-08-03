"""
Model explanations (Sec. 3.1).

For an input image, CSR predicts a class and, for each ACTIVE concept, presents:
  * the similarity map S_km of the highest-similarity prototype (where the
    concept is activated on the image), and
  * the concept similarity score s_km.

This script overlays those similarity maps on the image and saves a figure.
Optionally, if an atlas (build_atlas.py) is supplied, it also prints the source
"prototype image" I(p_km) for each shown prototype.

Usage:
  python explain.py --weights checkpoints/csr_network_final_best.pth \
      --image data/some_image.jpg --out explanation.png
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import CLASS_NAMES, CONCEPT_KEYS
from evaluate import build_csr_from_state
from utils import resolve_device, build_transforms


def load_image(path, image_size):
    img = Image.open(path).convert("RGB")
    tensor = build_transforms(image_size, is_train=False)(img).unsqueeze(0)
    disp = np.array(img.resize((image_size, image_size))) / 255.0
    return tensor, disp


@torch.no_grad()
def main(args):
    device = resolve_device(args.device)
    state = torch.load(args.weights, map_location=device)
    net, K, M = build_csr_from_state(state, args.backbone, device)

    x, disp = load_image(args.image, args.image_size)
    x = x.to(device)
    logits, s_flat, S_maps = net(x)                      # S_maps:(1,K,M,H,W)
    pred = int(logits.argmax(1))
    s = s_flat.view(K, M)
    print(f"Predicted class: {CLASS_NAMES[pred]}")

    # top prototype per concept + its score
    top_m = s.argmax(dim=1)                              # (K,)
    top_s = s.max(dim=1).values
    active = (top_s > args.concept_threshold).nonzero(as_tuple=False).flatten().tolist()
    print("Active concepts (by top prototype similarity):")
    for k in range(K):
        flag = "*" if k in active else " "
        print(f"  {flag} {CONCEPT_KEYS[k]:>18s}: max sim = {top_s[k].item():.3f} (proto #{int(top_m[k])})")

    atlas = torch.load(args.atlas, map_location="cpu") if args.atlas else None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure. (pip install matplotlib)")
        return

    show = active if active else list(range(K))
    ncol = len(show) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(3.2 * ncol, 3.4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(disp); axes[0].set_title(f"input -> {CLASS_NAMES[pred]}"); axes[0].axis("off")
    for i, k in enumerate(show, start=1):
        m = int(top_m[k])
        heat = S_maps[0, k, m].clamp(min=0).cpu().numpy()
        heat = np.array(Image.fromarray((heat / (heat.max() + 1e-8) * 255).astype(np.uint8))
                        .resize((args.image_size, args.image_size), Image.BILINEAR)) / 255.0
        axes[i].imshow(disp); axes[i].imshow(heat, cmap="jet", alpha=0.45)
        title = f"{CONCEPT_KEYS[k]}\nsim={top_s[k].item():.2f}"
        if atlas is not None:
            idx = int(atlas["best_image_idx"][k, m])
            if idx >= 0:
                title += f"\nI(p)= {atlas['image_ids'][idx]}"
        axes[i].set_title(title, fontsize=8); axes[i].axis("off")
    plt.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Explanation figure saved to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CSR similarity-map explanations")
    p.add_argument("--weights", required=True)
    p.add_argument("--image", required=True, help="path to a .jpg image")
    p.add_argument("--atlas", default=None, help="optional checkpoints/atlas.pt")
    p.add_argument("--backbone", default="convnext_tiny.fb_in1k")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--concept_threshold", type=float, default=0.3)
    p.add_argument("--out", default="explanation.png")
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
