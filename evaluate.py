"""
Evaluate a trained CSR network on the held-out TEST split and report the same
quantities as Table 1 of the paper: Macro F1-score, per-class F1, accuracy,
confusion matrix, and the explanation size (#exp) / #prototypes.

For ISIC-2017: #concepts K=4, M=100 -> #prototypes = 400, explanation size = 4
(CSR presents only the single highest-similarity prototype per active concept).

Usage (official test set):
  python evaluate.py \
    --img_dir data/isic_224x224.zip --task2_dir data/ISIC-2017_Training_Part2_GroundTruth.zip \
    --task3_csv data/ISIC-2017_Training_Part3_GroundTruth.csv \
    --test_img data/ISIC-2017_Test_v2_Data.zip \
    --test_task2 data/ISIC-2017_Test_v2_Part2_GroundTruth.zip \
    --test_task3 data/ISIC-2017_Test_v2_Part3_GroundTruth.csv \
    --weights checkpoints/csr_network_final_best.pth
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from dataset import get_dataloader, CLASS_NAMES
from models.concept_model import ConceptModel, DEFAULT_BACKBONE
from models.projector_model import Projector, ConceptPrototypes
from models.csr_network import TaskHead, CSRNetwork
from splits import add_data_args, resolve_splits
from utils import seed_everything, resolve_device, amp_autocast


def build_csr_from_state(state, backbone, device):
    """Reconstruct the CSR network with dimensions inferred from the checkpoint."""
    proto = state["prototypes.prototypes"]           # (K, M, dim)
    K, M, dim = proto.shape
    num_classes = state["task_head.linear.weight"].shape[0]
    # Older checkpoints have no BatchNorm in the task head; detect and match.
    normalize_input = "task_head.norm.weight" in state
    concept_model = ConceptModel(num_concepts=K, backbone=backbone, pretrained=False)
    projector = Projector(in_dim=concept_model.feature_dim, out_dim=dim)
    prototypes = ConceptPrototypes(num_concepts=K, M=M, dim=dim)
    task_head = TaskHead(in_features=K * M, num_classes=num_classes,
                         normalize_input=normalize_input)
    net = CSRNetwork(concept_model, projector, prototypes, task_head)
    net.load_state_dict(state)
    return net.to(device).eval(), K, M


@torch.no_grad()
def run(net, loader, device):
    preds, gts = [], []
    for images, _, targets in loader:
        images = images.to(device)
        with amp_autocast(device):
            logits, _, _ = net(images)
        preds.append(logits.argmax(1).cpu().numpy())
        gts.append(targets.numpy())
    return np.concatenate(preds), np.concatenate(gts)


def main(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    splits = resolve_splits(args)
    test_spec = splits["test"]
    if test_spec is None:
        raise SystemExit("No test split available. Provide --test_img/--test_task2/--test_task3 "
                         "or rely on the internal split (omit official val/test paths).")

    test_loader = get_dataloader(test_spec.img_source, test_spec.part3_csv, test_spec.part2_source,
                                 batch_size=args.batch_size, is_train=False, ids=test_spec.ids,
                                 image_size=args.image_size)

    state = torch.load(args.weights, map_location=device)
    net, K, M = build_csr_from_state(state, args.backbone, device)

    preds, gts = run(net, test_loader, device)
    macro_f1 = f1_score(gts, preds, average="macro", zero_division=0)
    per_class = f1_score(gts, preds, average=None, zero_division=0)
    acc = (preds == gts).mean() * 100

    print("\n================ CSR TEST RESULTS (ISIC-2017) ================")
    print(f"Split mode        : {splits['mode']}  ({len(gts)} test images)")
    print(f"Macro F1-score    : {macro_f1*100:.2f}   (paper Table 1 CSR = 71.5)")
    print(f"Accuracy          : {acc:.2f}%")
    print(f"#prototypes        : {K*M}   |  explanation size (#exp): {K}")
    print("\nPer-class F1:")
    for name, f1v in zip(CLASS_NAMES, per_class):
        print(f"  {name:>22s}: {f1v*100:.2f}")
    print("\nClassification report:")
    print(classification_report(gts, preds, target_names=CLASS_NAMES, zero_division=0, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(gts, preds))
    print("==============================================================\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate CSR (Table 1 macro-F1)")
    add_data_args(p)
    p.add_argument("--weights", required=True, help="csr_network_final_best.pth")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
