"""
Pointing Game evaluation (Sec. 5, Table 2).

A "hit" occurs when the point of maximum value in a concept's similarity map
falls inside that concept's ground-truth region. For ISIC-2017 the concept
ground-truth region is reconstructed from the official superpixel PNG
(ISIC_xxx_superpixels.png, present in the full-resolution Data archives) and the
Part-2 feature JSON, which marks which superpixels belong to each concept.

We report the CSR hit-rate: for every (image, active-concept) pair we take the
concept's best prototype similarity map, upsample it to image resolution, and
check whether its argmax lies inside the concept's superpixel mask.

Requires the FULL-RES official Data zip (it contains the superpixel PNGs). The
resized 224px training zip does NOT contain superpixels, so run this on
--test_img = ISIC-2017_Test_v2_Data.zip.
"""
import argparse
import io
import json
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dataset import CONCEPT_KEYS
from evaluate import build_csr_from_state
from utils import resolve_device, build_transforms


def decode_superpixels(png_bytes):
    """ISIC encodes the superpixel index across RGB channels: idx = R + G*256 + B*65536."""
    arr = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB")).astype(np.int64)
    return arr[..., 0] + arr[..., 1] * 256 + arr[..., 2] * 65536


def concept_mask(superpix, feature_list):
    """Binary mask (H,W) = union of superpixels flagged for this concept."""
    pos_ids = {i for i, v in enumerate(feature_list) if v == 1}
    if not pos_ids:
        return None
    return np.isin(superpix, list(pos_ids))


def index_zip(zip_path, suffix):
    members = {}
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():
            if m.lower().endswith(suffix):
                stem = m.split("/")[-1]
                members[stem] = m
    return members


@torch.no_grad()
def main(args):
    device = resolve_device(args.device)
    state = torch.load(args.weights, map_location=device)
    net, K, M = build_csr_from_state(state, args.backbone, device)
    tfm = build_transforms(args.image_size, is_train=False)

    img_zip = zipfile.ZipFile(args.test_img)
    sp_members = index_zip(args.test_img, "_superpixels.png")
    if not sp_members:
        raise SystemExit("No *_superpixels.png found in --test_img. Use the full-resolution "
                         "official Data zip (e.g. ISIC-2017_Test_v2_Data.zip); the resized "
                         "224px training zip has no superpixels, so the Pointing Game "
                         "cannot be run on it.")
    jpg_members = index_zip(args.test_img, ".jpg")
    con_zip = zipfile.ZipFile(args.test_task2)
    con_members = {m.split("/")[-1].replace("_features.json", ""): m
                   for m in con_zip.namelist() if m.endswith(".json")}

    hits = total = 0
    ids = [stem.replace("_superpixels.png", "") for stem in sp_members]
    for img_id in tqdm(ids, desc="Pointing Game"):
        if img_id not in jpg_members or img_id not in con_members:
            continue
        feats = json.load(con_zip.open(con_members[img_id]))
        superpix = decode_superpixels(img_zip.read(sp_members[f"{img_id}_superpixels.png"]))
        Himg, Wimg = superpix.shape

        pil = Image.open(io.BytesIO(img_zip.read(jpg_members[img_id]))).convert("RGB")
        x = tfm(pil).unsqueeze(0).to(device)
        _, _, S_maps = net(x)                      # (1,K,M,h,w)
        # per-concept map = best prototype at each location
        concept_map = S_maps[0].amax(dim=1)        # (K, h, w)

        for k, key in enumerate(CONCEPT_KEYS):
            mask = concept_mask(superpix, feats.get(key, []))
            if mask is None or mask.sum() == 0:
                continue                            # concept absent -> skip
            heat = concept_map[k].clamp(min=0)[None, None]
            heat = F.interpolate(heat, size=(Himg, Wimg), mode="bilinear",
                                 align_corners=False)[0, 0].cpu().numpy()
            r, c = np.unravel_index(np.argmax(heat), heat.shape)
            hits += int(mask[r, c])
            total += 1

    rate = 100.0 * hits / total if total else 0.0
    print(f"\nPointing Game hit-rate (CSR): {rate:.1f}%  over {total} (image, concept) pairs")
    print("(Paper Table 2 reports PG on TBX11K/Tuberculosis; this is the ISIC concept-level analogue.)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pointing Game hit-rate for CSR concepts")
    p.add_argument("--weights", required=True)
    p.add_argument("--test_img", required=True, help="full-res Data zip with *_superpixels.png")
    p.add_argument("--test_task2", required=True, help="Part2 concept JSON zip")
    p.add_argument("--backbone", default="convnext_tiny.fb_in1k")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
