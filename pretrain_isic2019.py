"""ISIC-2019 backbone pretraining for CSR (see project README)."""
import argparse, io, os, zipfile
import numpy as np, pandas as pd, timm, torch
import torch.nn as nn, torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from tqdm import tqdm
from utils import (seed_everything, resolve_device, amp_autocast, make_grad_scaler, build_transforms)
from models.concept_model import DEFAULT_BACKBONE


class ISIC2019Dataset(Dataset):
    def __init__(self, img_source, label_csv, transform=None, ids=None):
        self.img_source = img_source; self.transform = transform
        self.zip_path = img_source if zipfile.is_zipfile(img_source) else None
        if self.zip_path is None and not os.path.isdir(img_source):
            raise ValueError(f"Image source must be a dir or .zip: {img_source}")
        self.members = self._index_zip(self.zip_path) if self.zip_path else None
        self._zip = None
        df = pd.read_csv(label_csv)
        self.class_cols = [c for c in df.columns if c.lower() not in ("image", "image_id")]
        id_col = "image" if "image" in df.columns else df.columns[0]
        targets = df[self.class_cols].to_numpy().argmax(axis=1)
        self.img_names, self.targets = [], []
        keep = set(ids) if ids is not None else None
        for name, tgt in zip(df[id_col].tolist(), targets.tolist()):
            if keep is not None and name not in keep: continue
            if self._has_image(name):
                self.img_names.append(name); self.targets.append(int(tgt))
        if not self.img_names: raise RuntimeError("No ISIC-2019 images found.")
        self.num_classes = len(self.class_cols)
        print(f"[ISIC2019] {len(self.img_names)} images, {self.num_classes} classes {self.class_cols}")
    def __getstate__(self):
        s = self.__dict__.copy(); s["_zip"] = None; return s
    @staticmethod
    def _index_zip(zp):
        members = {}
        with zipfile.ZipFile(zp) as a:
            for m in a.namelist():
                if m.lower().endswith(".jpg"):
                    members[os.path.splitext(os.path.basename(m))[0]] = m
        return members
    def _has_image(self, name):
        if self.members is not None: return name in self.members
        return os.path.exists(os.path.join(self.img_source, f"{name}.jpg"))
    def _get_zip(self):
        if self._zip is None: self._zip = zipfile.ZipFile(self.zip_path)
        return self._zip
    def _open(self, name):
        if self.zip_path:
            with self._get_zip().open(self.members[name]) as fh:
                return Image.open(io.BytesIO(fh.read())).convert("RGB")
        return Image.open(os.path.join(self.img_source, f"{name}.jpg")).convert("RGB")
    def __len__(self): return len(self.img_names)
    def __getitem__(self, idx):
        img = self._open(self.img_names[idx])
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.targets[idx], dtype=torch.long)


def main(args):
    seed_everything(args.seed); device = resolve_device(args.device)
    print(f"Using device: {device}")
    full = pd.read_csv(args.label_csv)
    id_col = "image" if "image" in full.columns else full.columns[0]
    class_cols = [c for c in full.columns if c.lower() not in ("image", "image_id")]
    tgt = full[class_cols].to_numpy().argmax(1)
    rng = np.random.default_rng(args.seed); val_ids = set()
    for k in np.unique(tgt):
        ids_k = full[id_col].to_numpy()[tgt == k]; rng.shuffle(ids_k)
        val_ids.update(ids_k[: max(1, int(0.1 * len(ids_k)))].tolist())
    train_ids = set(full[id_col]) - val_ids
    tfm_tr = build_transforms(args.image_size, is_train=True)
    tfm_ev = build_transforms(args.image_size, is_train=False)
    train_ds = ISIC2019Dataset(args.img_dir, args.label_csv, tfm_tr, ids=train_ids)
    val_ds = ISIC2019Dataset(args.img_dir, args.label_csv, tfm_ev, ids=val_ids)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    model = timm.create_model(args.backbone, pretrained=True, num_classes=train_ds.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = make_grad_scaler(device)
    os.makedirs(args.save_dir, exist_ok=True); best_f1 = -1.0
    best_path = os.path.join(args.save_dir, "backbone_isic2019.pth")
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        pbar = tqdm(train_loader, desc=f"[PRE] Epoch {epoch+1}/{args.epochs}")
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += loss.item(); pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                with amp_autocast(device):
                    preds.append(model(images).argmax(1).cpu().numpy())
                gts.append(targets.numpy())
        f1 = f1_score(np.concatenate(gts), np.concatenate(preds), average="macro", zero_division=0)
        print(f"[PRE] Epoch {epoch+1}: train_loss={running/len(train_loader):.4f} val_macroF1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            fe = timm.create_model(args.backbone, pretrained=False, num_classes=0)
            fe.load_state_dict(model.state_dict(), strict=False)
            torch.save(fe.state_dict(), best_path)
            print(f"    --> saved backbone (val_macroF1={f1:.4f}) to {best_path}")
    print(f"[PRE] done. Best val macro-F1={best_f1:.4f}. Backbone -> {best_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", required=True)
    p.add_argument("--label_csv", required=True)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
