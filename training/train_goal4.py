#!/usr/bin/env python3
"""Train the Goal4 multi-head MedicalNet baseline on available sequences."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from torch import nn
from training.models.medicalnet import MedicalNet3DEncoder
from training.sequence_dataset import assign_accession_splits, canonical_modality, normalize
from training.resize import resize_volume


HEADS = {
    "location": ("check__location_of_lesion", "cat"),
    "morphology": ("check__lesion_morphology", "cat"),
    "who_grade": ("check__who_grade", "who"),
    "enhancement": ("check__tumor_t1wi_c_enhan__std", "bin"),
    "enhancement_pattern": ("check__tumor_t1wi_c_enhan_pattern", "cat"),
    "necrosis": ("check__tumor_feature_necrosis__std", "bin"),
    "cystic_change": ("check__tumor_feature_change__std", "bin"),
    "hemorrhage": ("check__tumor_feature_hemorrhage__std", "bin"),
    "calcification": ("check__tumor_feature_calcification__std", "bin"),
    "margin_clear": ("check__lesion_morph_feature_boundary__std", "bin"),
    "lobulation": ("check__lesion_mor_feature_lobulation__std", "bin"),
    "signal_t2wi": ("check__tumor_t2wi_signal_intensity", "cat"),
    "signal_flair": ("check__tumor_t2_flair_sign_intensity", "cat"),
}

LOCATION = ("Brainstem", "RightParietal", "RightFrontal", "RightBasalGanglia", "RightTemporal", "RightCerebellar", "RightOccipital", "LeftParietal", "LeftFrontal", "LeftBasalGanglia", "LeftTemporal", "LeftCerebellar", "LeftOccipital", "Other", "Unknown")
MORPH = ("Regular", "Irregular")
WHO = ("1", "2", "3", "4")
PATTERN = ("None", "Ring", "RimEnhancing", "Nodular", "GroundGlass", "Gyriform", "Multifocal", "Other")
SIGNAL = ("Low", "Iso", "High")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def norm(value: str) -> str:
    return "".join(str(value).lower().split()).replace("_", "")


def encode(head: str, value: str) -> int:
    if not value or value.lower() in {"na", "unk", "unknown", "none", "nan"}:
        return -1
    kind = HEADS[head][1]
    if kind == "bin":
        return 1 if value.lower() in {"1", "true", "yes", "有", "是"} else 0
    choices = LOCATION if head == "location" else MORPH if head == "morphology" else WHO if head == "who_grade" else PATTERN if head == "enhancement_pattern" else SIGNAL
    aliases = {norm(item): i for i, item in enumerate(choices)}
    aliases.update({"regular": 0, "规则": 0, "irregular": 1, "不规则": 1, "high": 2, "高": 2, "low": 0, "低": 0, "iso": 1, "等": 1, "ring": 1, "环形": 1, "rimenhancement": 2, "花环状": 2, "nodular": 3, "结节状": 3, "multifocal": 6, "多灶状": 6})
    return aliases.get(norm(value), -1)


class Goal4Dataset(Dataset):
    def __init__(self, rows, labels, split, shape):
        self.rows = [r for r in rows if r["split"] == split]
        self.labels = labels; self.shape = shape

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = normalize(np.asarray(nib.load(row["series_path"]).dataobj, dtype=np.float32))
        image = resize_volume(image, self.shape, is_mask=False)
        label = self.labels[(row["AccessionNumber"], row["SeriesUid"])]
        return {"image": torch.from_numpy(np.asarray(image, dtype=np.float32))[None], "targets": torch.tensor(label, dtype=torch.long), "accession": row["AccessionNumber"]}


def loss_and_metrics(outputs, targets):
    total = outputs["location"].new_zeros(())
    used = 0
    for head, (_, kind) in HEADS.items():
        target = targets[:, list(HEADS).index(head)]
        valid = target >= 0
        if not valid.any(): continue
        if kind == "bin":
            total = total + BCEWithLogitsLoss()(outputs[head][valid, 0], target[valid].float())
        else:
            total = total + CrossEntropyLoss()(outputs[head][valid], target[valid])
        used += 1
    return total / max(used, 1), used


class Goal4MedicalNet(nn.Module):
    """MedicalNet encoder with the same multi-head names as inference Goal4."""

    HEAD_SIZES = {"location": 15, "morphology": 2, "who_grade": 4, "enhancement": 1, "enhancement_pattern": 8, "necrosis": 1, "cystic_change": 1, "hemorrhage": 1, "calcification": 1, "margin_clear": 1, "lobulation": 1, "signal_t2wi": 3, "signal_flair": 3}

    def __init__(self, checkpoint: Path | None = None):
        super().__init__()
        self.encoder = MedicalNet3DEncoder(depth=50, in_channels=1, checkpoint=checkpoint)
        self.heads = nn.ModuleDict({name: nn.Linear(self.encoder.feature_dim, size) for name, size in self.HEAD_SIZES.items()})

    def forward(self, x):
        features = self.encoder(x)
        return {name: head(features) for name, head in self.heads.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-index", type=Path, required=True); parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--medicalnet-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=40); parser.add_argument("--batch-size", type=int, default=2); parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-shape", nargs=3, type=int, default=(32, 32, 16)); parser.add_argument("--val-fraction", type=float, default=0.2); parser.add_argument("--test-fraction", type=float, default=0.1); parser.add_argument("--patience", type=int, default=8); parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--output", type=Path, default=Path("checkpoint_goal4"))
    args = parser.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    file_rows = read_csv(args.file_index); label_rows = read_csv(args.labels_csv)
    labels = {}
    for row in label_rows:
        labels[(row.get("AccessionNumber", ""), row.get("SeriesUid", ""))] = [encode(head, row.get(column, "")) for head, (column, _) in HEADS.items()]
    rows = [r for r in file_rows if r.get("status") in {"series_and_mask_found", "series_found_mask_missing"} and (r.get("AccessionNumber", ""), r.get("SeriesUid", "")) in labels]
    for r in rows: r["split"] = "train"
    # Reuse accession-safe split logic without duplicating sequence records.
    accessions = sorted({r["AccessionNumber"] for r in rows}); rng = random.Random(args.seed); rng.shuffle(accessions)
    n_test = round(len(accessions) * args.test_fraction); n_val = round(len(accessions) * args.val_fraction)
    split_map = {a: "test" if i < n_test else "val" if i < n_test + n_val else "train" for i, a in enumerate(accessions)}
    for r in rows: r["split"] = split_map[r["AccessionNumber"]]
    args.output.mkdir(parents=True, exist_ok=True); shape = tuple(args.target_shape)
    loaders = {s: DataLoader(Goal4Dataset(rows, labels, s, shape), batch_size=args.batch_size, shuffle=s == "train", num_workers=args.num_workers) for s in ("train", "val", "test")}
    if not len(loaders["train"].dataset) or not len(loaders["val"].dataset): raise RuntimeError("train/val split is empty")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = Goal4MedicalNet(checkpoint=args.medicalnet_checkpoint).to(device); optimizer = AdamW(model.parameters(), lr=args.lr); best = float("inf"); stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True); loss, _ = loss_and_metrics(model(batch["image"].to(device)), batch["targets"].to(device)); loss.backward(); optimizer.step(); running += float(loss.item())
        model.eval(); val_loss = 0.0
        with torch.inference_mode():
            for batch in loaders["val"]: val_loss += float(loss_and_metrics(model(batch["image"].to(device)), batch["targets"].to(device))[0].item())
        val_loss /= max(len(loaders["val"]), 1); row = {"epoch": epoch, "train_loss": running / max(len(loaders["train"]), 1), "val_loss": val_loss}; history.append(row); print(json.dumps(row), flush=True)
        if val_loss < best - 1e-4: best = val_loss; stale = 0; torch.save({"model": model.state_dict(), "backend": "medicalnet", "in_channels": 1}, args.output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience: break
    (args.output / "summary.json").write_text(json.dumps({"sequences": len(rows), "cases": len(accessions), "history": history, "best_val_loss": best}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
