"""
ISIC-2017 dataset for CSR.

Concepts (Task 2 dermoscopic features): milia_like_cyst, pigment_network,
negative_network, streaks  -> 4 concepts (K=4).

Targets (Task 3 diagnosis): melanoma, seborrheic_keratosis, nevus  -> 3 classes.
The Part3 CSV encodes melanoma / seborrheic_keratosis one-hot; everything else
is nevus. (The supplementary phrases the 3 classes loosely; the ISIC-2017
challenge uses exactly {melanoma, seborrheic keratosis, nevus}.)

Key features vs. the original code:
  * accepts an explicit list of image ids (``ids=``) so we can build clean
    train / val / test splits from separate official archives OR by carving a
    stratified split out of the training set;
  * only keeps ids that have BOTH an image and a concept file, so mismatched
    partial downloads never crash training;
  * image size is configurable (ISIC uses 224).
Sources can each be either an extracted directory or a .zip archive.
"""
import os
import io
import json
import glob
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils import build_transforms

CONCEPT_KEYS = ["milia_like_cyst", "pigment_network", "negative_network", "streaks"]
CLASS_NAMES = ["seborrheic_keratosis", "melanoma", "nevus"]  # indices 0,1,2


def _target_from_row(melanoma, seborrheic) -> int:
    if float(melanoma) == 1.0:
        return 1  # melanoma
    if float(seborrheic) == 1.0:
        return 0  # seborrheic keratosis
    return 2      # nevus (neither)


class ISIC2017Dataset(Dataset):
    def __init__(self, img_source, part3_csv, part2_source, transform=None,
                 ids=None, concept_keys=CONCEPT_KEYS):
        self.img_source = img_source
        self.part2_source = part2_source
        self.transform = transform
        self.concept_keys = list(concept_keys)
        self._image_zip = None
        self._concept_zip = None

        for path, name in [(part3_csv, "Task 3 CSV"), (img_source, "image source"),
                           (part2_source, "Task 2 concept source")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found: {path}")

        # ---- resolve zip vs directory sources -----------------------------
        self.image_zip_path = img_source if zipfile.is_zipfile(img_source) else None
        self.concept_zip_path = part2_source if zipfile.is_zipfile(part2_source) else None
        if self.image_zip_path is None and not os.path.isdir(img_source):
            raise ValueError(f"Image source must be a directory or .zip: {img_source}")
        if self.concept_zip_path is None and not os.path.isdir(part2_source):
            raise ValueError(f"Concept source must be a directory or .zip: {part2_source}")

        self.image_members = self._index_zip(self.image_zip_path, ".jpg") if self.image_zip_path else None
        self.concept_members = self._index_zip(self.concept_zip_path, ".json") if self.concept_zip_path else None

        # ---- labels --------------------------------------------------------
        df = pd.read_csv(part3_csv)
        required = {"image_id", "melanoma", "seborrheic_keratosis"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Task 3 CSV missing columns: {sorted(missing)}")
        label_map = {r.image_id: _target_from_row(r.melanoma, r.seborrheic_keratosis)
                     for r in df.itertuples(index=False)}

        # ---- keep only ids that are fully available ------------------------
        candidate_ids = list(ids) if ids is not None else list(label_map.keys())
        self.img_names, self.targets = [], []
        skipped = 0
        for img_id in candidate_ids:
            if img_id not in label_map:
                skipped += 1
                continue
            if not self._has_image(img_id) or not self._has_concept(img_id):
                skipped += 1
                continue
            self.img_names.append(img_id)
            self.targets.append(label_map[img_id])
        if len(self.img_names) == 0:
            raise RuntimeError("No usable samples found (image + concept + label). "
                               "Check that the image / concept / label sources match.")
        if skipped:
            print(f"[ISIC2017Dataset] kept {len(self.img_names)} samples, skipped {skipped} "
                  f"(missing image/concept/label).")

    # ------------------------------------------------------------------ zips
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_zip"] = None
        state["_concept_zip"] = None
        return state

    @staticmethod
    def _index_zip(zip_path, suffix):
        members = {}
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if not member.lower().endswith(suffix):
                    continue
                stem = os.path.splitext(os.path.basename(member))[0]
                if stem.endswith("_features"):
                    stem = stem[:-len("_features")]
                members[stem] = member
        return members

    def _has_image(self, img_id):
        if self.image_members is not None:
            return img_id in self.image_members
        return os.path.exists(os.path.join(self.img_source, f"{img_id}.jpg"))

    def _has_concept(self, img_id):
        if self.concept_members is not None:
            return img_id in self.concept_members
        if os.path.exists(os.path.join(self.part2_source, f"{img_id}_features.json")):
            return True
        return bool(glob.glob(os.path.join(self.part2_source, "**", f"{img_id}*_features.json"),
                              recursive=True))

    def _get_image_zip(self):
        if self._image_zip is None:
            self._image_zip = zipfile.ZipFile(self.image_zip_path)
        return self._image_zip

    def _get_concept_zip(self):
        if self._concept_zip is None:
            self._concept_zip = zipfile.ZipFile(self.concept_zip_path)
        return self._concept_zip

    def _open_image(self, img_id):
        from PIL import Image
        if self.image_zip_path:
            with self._get_image_zip().open(self.image_members[img_id]) as fh:
                return Image.open(io.BytesIO(fh.read())).convert("RGB")
        return Image.open(os.path.join(self.img_source, f"{img_id}.jpg")).convert("RGB")

    def _load_concept_json(self, img_id):
        if self.concept_zip_path:
            with self._get_concept_zip().open(self.concept_members[img_id]) as fh:
                return json.load(fh)
        exact = os.path.join(self.part2_source, f"{img_id}_features.json")
        if os.path.exists(exact):
            with open(exact) as fh:
                return json.load(fh)
        found = glob.glob(os.path.join(self.part2_source, "**", f"{img_id}*_features.json"),
                          recursive=True)
        with open(found[0]) as fh:
            return json.load(fh)

    def _get_concepts(self, img_id):
        data = self._load_concept_json(img_id)
        vec = []
        for key in self.concept_keys:
            superpixels = data.get(key, [])
            vec.append(1.0 if any(v == 1 for v in superpixels) else 0.0)
        return torch.tensor(vec, dtype=torch.float32)

    # --------------------------------------------------------------- dataset
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_id = self.img_names[idx]
        image = self._open_image(img_id)
        if self.transform:
            image = self.transform(image)
        concept_labels = self._get_concepts(img_id)
        target_label = torch.tensor(self.targets[idx], dtype=torch.long)
        return image, concept_labels, target_label


# ----------------------------------------------------------------- splitting
def stratified_split(part3_csv, ids=None, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Stratified train/val/test split over image ids by target class.
    Used only when official validation/test archives are not supplied.
    Returns (train_ids, val_ids, test_ids).
    """
    df = pd.read_csv(part3_csv)
    df["target"] = [_target_from_row(m, s) for m, s in zip(df.melanoma, df.seborrheic_keratosis)]
    if ids is not None:
        df = df[df.image_id.isin(set(ids))]
    rng = np.random.default_rng(seed)
    train_ids, val_ids, test_ids = [], [], []
    for _, grp in df.groupby("target"):
        g = grp.image_id.to_numpy()
        rng.shuffle(g)
        n = len(g)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test_ids += g[:n_test].tolist()
        val_ids += g[n_test:n_test + n_val].tolist()
        train_ids += g[n_test + n_val:].tolist()
    return train_ids, val_ids, test_ids


def get_dataloader(img_source, part3_csv, part2_source, batch_size=32, is_train=True,
                   ids=None, image_size=224, num_workers=2, shuffle=None):
    transform = build_transforms(image_size=image_size, is_train=is_train)
    dataset = ISIC2017Dataset(img_source, part3_csv, part2_source, transform, ids=ids)
    if shuffle is None:
        shuffle = is_train
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)
