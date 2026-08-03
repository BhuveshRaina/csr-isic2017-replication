"""
List wrongly-predicted test images, with the exact patient index the
Streamlit app's "Patient index" sidebar field expects -- so you can pick one,
type its index into the app, and try to correct it interactively.

For each wrong prediction we also show:
  margin        top1-top2 probability gap (paper says only intervene if < 0.3)
  worst concept the truly-absent concept the model is firing highest on --
                the single best candidate to reject in the app

Sorted with low-margin (paper's "indecisive", most fixable) cases first, since
those are the realistic candidates for a doctor to intervene on.

Usage:
  python find_wrong_predictions.py \
    --weights checkpoints_revive/csr_network_final_best.pth \
    --img_dir data/isic_224x224.zip \
    --task2_dir data/ISIC-2017_Training_Part2_GroundTruth.zip \
    --task3_csv data/ISIC-2017_Training_Part3_GroundTruth.csv \
    --test_img data/ISIC-2017_Test_v2_Data.zip \
    --test_task2 data/ISIC-2017_Test_v2_Part2_GroundTruth.zip \
    --test_task3 data/ISIC-2017_Test_v2_Part3_GroundTruth.csv \
    --n 20
"""
import argparse

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


def main(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    splits = resolve_splits(args)
    spec = splits["test"]
    if spec is None:
        raise SystemExit("No test split available.")

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

    logits = S @ W.t() + b
    probs = torch.softmax(logits, dim=1)
    base_pred = logits.argmax(1).numpy()
    sorted_p, _ = probs.sort(dim=1, descending=True)
    margin = (sorted_p[:, 0] - sorted_p[:, 1]).numpy()

    absent = cgt < 0.5
    peak = np.where(absent, S3.max(-1).values.numpy(), -np.inf)
    worst = peak.argmax(1)
    has_absent = np.isfinite(peak.max(1))

    wrong = np.where(base_pred != gts)[0]
    print(f"\n{N} test images | {len(wrong)} wrong ({100*len(wrong)/N:.1f}%)")
    print(f"{'idx':>5} {'true':>18} {'pred':>18} {'margin':>7} {'has_fix':>8} "
          f"{'reject_concept':>16} {'conf_true':>10}")

    order = wrong[np.argsort(margin[wrong])]   # lowest margin (most indecisive) first
    for i in order[:args.n]:
        fix_concept = CONCEPT_KEYS[worst[i]] if has_absent[i] else "-"
        print(f"{i:>5} {CLASS_NAMES[gts[i]]:>18} {CLASS_NAMES[base_pred[i]]:>18} "
              f"{margin[i]:>7.3f} {str(bool(has_absent[i])):>8} {fix_concept:>16} "
              f"{probs[i, gts[i]].item():>10.3f}")

    print(f"\nType one of the 'idx' values above into the Streamlit app's "
          f"'Patient index' field, then try rejecting the concept listed under "
          f"'reject_concept' for that row.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Find wrong predictions for interactive testing")
    add_data_args(p)
    p.add_argument("--weights", required=True)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=20, help="how many wrong cases to list")
    main(p.parse_args())
