"""
Doctor-in-the-loop test-time interaction demo (Sec. 3.3).

Shows how the SAME image's prediction changes under:
  * spatial-level interaction  : positive boxes {bb+}=1 (focus), negative boxes
                                 {bb-}=0 (ignore), neutral weight alpha elsewhere
                                 -> importance map A (Eqn 12), reweight (Eqn 13);
  * concept-level interaction  : reject a concept k -> s_km = 0 for all m.

Boxes are given in normalised [x1 y1 x2 y2] coordinates (0..1) and mapped onto
the HxW feature grid. alpha defaults to 0.2 (the paper's setting).

Example:
  python interaction_demo.py --weights checkpoints/csr_network_final_best.pth \
      --image data/img.jpg --pos_box 0.1 0.1 0.5 0.5 --reject_concept 2
"""
import argparse

import torch

from dataset import CLASS_NAMES, CONCEPT_KEYS
from evaluate import build_csr_from_state
from explain import load_image
from utils import resolve_device


def build_importance_map(H, W, pos_boxes, neg_boxes, alpha, device):
    """A(h,w): 1 inside positive boxes, 0 inside negative boxes, alpha elsewhere."""
    A = torch.full((H, W), float(alpha), device=device)

    def to_grid(box):
        x1, y1, x2, y2 = box
        c1, c2 = int(x1 * W), int(round(x2 * W))
        r1, r2 = int(y1 * H), int(round(y2 * H))
        return max(r1, 0), min(max(r2, r1 + 1), H), max(c1, 0), min(max(c2, c1 + 1), W)

    for b in neg_boxes or []:
        r1, r2, c1, c2 = to_grid(b); A[r1:r2, c1:c2] = 0.0
    for b in pos_boxes or []:
        r1, r2, c1, c2 = to_grid(b); A[r1:r2, c1:c2] = 1.0
    return A


def softmax_probs(logits):
    return torch.softmax(logits, dim=1).flatten().tolist()


@torch.no_grad()
def main(args):
    device = resolve_device(args.device)
    state = torch.load(args.weights, map_location=device)
    net, K, M = build_csr_from_state(state, args.backbone, device)

    x, _ = load_image(args.image, args.image_size)
    x = x.to(device)

    base_logits, _, S_maps = net(x)
    H, W = S_maps.shape[-2:]
    base = softmax_probs(base_logits)
    print("Baseline prediction:")
    for name, p in zip(CLASS_NAMES, base):
        print(f"  {name:>22s}: {p:.3f}")
    print(f"  -> {CLASS_NAMES[int(base_logits.argmax())]}\n")

    pos = [args.pos_box] if args.pos_box else None
    neg = [args.neg_box] if args.neg_box else None
    A = None
    if pos or neg:
        A = build_importance_map(H, W, pos, neg, args.alpha, device)

    inter_logits, _, _ = net.predict_interactive(
        x, importance_map=A, rejected_concepts=args.reject_concept or None)
    inter = softmax_probs(inter_logits)

    print("After interaction:")
    if pos: print(f"  + positive box {args.pos_box}")
    if neg: print(f"  - negative box {args.neg_box}")
    if args.reject_concept:
        print(f"  x rejected concepts: {[CONCEPT_KEYS[k] for k in args.reject_concept]}")
    for name, p0, p1 in zip(CLASS_NAMES, base, inter):
        print(f"  {name:>22s}: {p0:.3f} -> {p1:.3f}  ({p1-p0:+.3f})")
    print(f"  -> {CLASS_NAMES[int(inter_logits.argmax())]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CSR interaction demo (Eqn 12-13)")
    p.add_argument("--weights", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--backbone", default="convnext_tiny.fb_in1k")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--pos_box", type=float, nargs=4, metavar=("x1", "y1", "x2", "y2"))
    p.add_argument("--neg_box", type=float, nargs=4, metavar=("x1", "y1", "x2", "y2"))
    p.add_argument("--reject_concept", type=int, nargs="*", help="concept indices to reject (0-3)")
    p.add_argument("--alpha", type=float, default=0.2, help="neutral weight (paper: 0.2)")
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
