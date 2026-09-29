#!/usr/bin/env python3
"""Train Goal4 multi-head predictions at AccessionNumber level."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from training.models.medicalnet import MedicalNet3DEncoder
from training.resize import resize_volume
from training.study_dataset import MODALITIES, StudyNiftiDataset, StudyRecord, build_study_records


HEADS = {
    "location": ("check__location_of_lesion", "cat"), "morphology": ("check__lesion_morphology", "cat"), "who_grade": ("check__who_grade", "who"),
    "enhancement": ("check__tumor_t1wi_c_enhan__std", "bin"), "enhancement_pattern": ("check__tumor_t1wi_c_enhan_pattern", "cat"), "necrosis": ("check__tumor_feature_necrosis__std", "bin"),
    "cystic_change": ("check__tumor_feature_change__std", "bin"), "hemorrhage": ("check__tumor_feature_hemorrhage__std", "bin"), "calcification": ("check__tumor_feature_calcification__std", "bin"),
    "margin_clear": ("check__lesion_morph_feature_boundary__std", "bin"), "lobulation": ("check__lesion_mor_feature_lobulation__std", "bin"), "signal_t2wi": ("check__tumor_t2wi_signal_intensity", "cat"), "signal_flair": ("check__tumor_t2_flair_sign_intensity", "cat"),
}
LOCATION = ("Brainstem", "RightParietal", "RightFrontal", "RightBasalGanglia", "RightTemporal", "RightCerebellar", "RightOccipital", "LeftParietal", "LeftFrontal", "LeftBasalGanglia", "LeftTemporal", "LeftCerebellar", "LeftOccipital", "Other", "Unknown")
MORPH = ("Regular", "Irregular"); WHO = ("1", "2", "3", "4"); PATTERN = ("None", "Ring", "RimEnhancing", "Nodular", "GroundGlass", "Gyriform", "Multifocal", "Other"); SIGNAL = ("Low", "Iso", "High")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def norm(value: str) -> str:
    return "".join(str(value).lower().split()).replace("_", "")


def encode(head: str, value: str) -> int:
    if not value or value.lower() in {"na", "unk", "unknown", "none", "nan"}: return -1
    kind = HEADS[head][1]
    if kind == "bin": return 1 if value.lower() in {"1", "true", "yes", "有", "是"} else 0
    choices = LOCATION if head == "location" else MORPH if head == "morphology" else WHO if head == "who_grade" else PATTERN if head == "enhancement_pattern" else SIGNAL
    aliases = {norm(item): i for i, item in enumerate(choices)}
    aliases.update({"regular": 0, "规则": 0, "irregular": 1, "不规则": 1, "high": 2, "高": 2, "low": 0, "低": 0, "iso": 1, "等": 1, "ring": 1, "环形": 1, "rimenhancement": 2, "花环状": 2, "nodular": 3, "结节状": 3, "multifocal": 6, "多灶状": 6})
    return aliases.get(norm(value), -1)


def split_records(records: list[StudyRecord], val_fraction: float, test_fraction: float, seed: int):
    accessions = sorted({record.accession for record in records}); rng = random.Random(seed); rng.shuffle(accessions)
    n_test = round(len(accessions) * test_fraction); n_val = round(len(accessions) * val_fraction)
    split = {a: "test" if i < n_test else "val" if i < n_test + n_val else "train" for i, a in enumerate(accessions)}
    return ([r for r in records if split[r.accession] == "train"], [r for r in records if split[r.accession] == "val"], [r for r in records if split[r.accession] == "test"], split)


class StudyGoal4Dataset(Dataset):
    def __init__(self, records, labels, shape):
        self.base = StudyNiftiDataset(records, target_shape=shape, require_label=False)
        self.labels = labels
    def __len__(self): return len(self.base)
    def __getitem__(self, index):
        item = self.base[index]; item["targets"] = torch.tensor(self.labels[item["AccessionNumber"]], dtype=torch.long); return item


def loader(dataset, batch_size, shuffle, num_workers):
    kwargs = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": num_workers}
    if torch.cuda.is_available(): kwargs["pin_memory"] = True
    if num_workers > 0: kwargs["persistent_workers"] = True; kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def loss_and_metrics(outputs, targets):
    total = outputs["location"].new_zeros(()); used = 0
    for index, (head, (_, kind)) in enumerate(HEADS.items()):
        target = targets[:, index]; valid = target >= 0
        if not valid.any(): continue
        if kind == "bin": total = total + BCEWithLogitsLoss()(outputs[head][valid, 0], target[valid].float())
        else: total = total + CrossEntropyLoss()(outputs[head][valid], target[valid])
        used += 1
    return total / max(used, 1), used


class Goal4MedicalNet(nn.Module):
    HEAD_SIZES = {"location": 15, "morphology": 2, "who_grade": 4, "enhancement": 1, "enhancement_pattern": 8, "necrosis": 1, "cystic_change": 1, "hemorrhage": 1, "calcification": 1, "margin_clear": 1, "lobulation": 1, "signal_t2wi": 3, "signal_flair": 3}
    def __init__(self, checkpoint: Path | None = None):
        super().__init__(); self.encoder = MedicalNet3DEncoder(depth=50, in_channels=len(MODALITIES), checkpoint=checkpoint); self.heads = nn.ModuleDict({name: nn.Linear(self.encoder.feature_dim, size) for name, size in self.HEAD_SIZES.items()})
    def forward(self, x):
        features = self.encoder(x); return {name: head(features) for name, head in self.heads.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-index", type=Path, required=True); parser.add_argument("--labels-csv", type=Path, required=True); parser.add_argument("--medicalnet-checkpoint", type=Path, default=None); parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--target-shape", nargs=3, type=int, default=(64, 64, 32)); parser.add_argument("--val-fraction", type=float, default=0.2); parser.add_argument("--test-fraction", type=float, default=0.1); parser.add_argument("--patience", type=int, default=12); parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--lr-factor", type=float, default=0.5); parser.add_argument("--lr-patience", type=int, default=3); parser.add_argument("--min-lr", type=float, default=1e-6); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--output", type=Path, default=Path("checkpoint_goal4"))
    args = parser.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    label_rows = read_csv(args.labels_csv); labels = {}
    for row in label_rows:
        accession = row.get("AccessionNumber", "")
        if accession and accession not in labels: labels[accession] = [encode(head, row.get(column, "")) for head, (column, _) in HEADS.items()]
    records = build_study_records(args.file_index, args.labels_csv, require_all_modalities=False)
    records = [r for r in records if r.accession in labels]
    train_records, val_records, test_records, split_map = split_records(records, args.val_fraction, args.test_fraction, args.seed)
    if not train_records or not val_records: raise RuntimeError("train/val is empty; check index and split fractions")
    args.output.mkdir(parents=True, exist_ok=True); shape = tuple(args.target_shape)
    (args.output / "splits.json").write_text(json.dumps({s: sorted(a for a, v in split_map.items() if v == s) for s in ("train", "val", "test")}, ensure_ascii=False, indent=2), encoding="utf-8")
    loaders = {s: loader(StudyGoal4Dataset(rs, labels, shape), args.batch_size, s == "train", args.num_workers) for s, rs in (("train", train_records), ("val", val_records), ("test", test_records))}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = Goal4MedicalNet(checkpoint=args.medicalnet_checkpoint).to(device); optimizer = AdamW(model.parameters(), lr=args.lr); scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr); best = float("inf"); stale = 0; history = []; start_epoch = 1
    if args.resume:
        payload = torch.load(args.resume, map_location=device); missing, unexpected = model.load_state_dict(payload.get("model", payload), strict=False)
        if payload.get("optimizer"):
            try: optimizer.load_state_dict(payload["optimizer"])
            except ValueError: pass
        if payload.get("scheduler"):
            try: scheduler.load_state_dict(payload["scheduler"])
            except ValueError: pass
        start_epoch = int(payload.get("epoch", 0)) + 1; best = float(payload.get("best", best)); stale = int(payload.get("stale", 0)); history = list(payload.get("history", [])); print(json.dumps({"resumed_from": str(args.resume), "start_epoch": start_epoch, "missing": list(missing), "unexpected": list(unexpected)}), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); running = 0.0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True); loss, _ = loss_and_metrics(model(batch["image"].to(device, non_blocking=True)), batch["targets"].to(device, non_blocking=True)); loss.backward(); optimizer.step(); running += float(loss.item())
        model.eval(); val_loss = 0.0
        with torch.inference_mode():
            for batch in loaders["val"]: val_loss += float(loss_and_metrics(model(batch["image"].to(device, non_blocking=True)), batch["targets"].to(device, non_blocking=True))[0].item())
        val_loss /= max(len(loaders["val"]), 1); scheduler.step(val_loss); row = {"epoch": epoch, "train_loss": running / max(len(loaders["train"]), 1), "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}; history.append(row); print(json.dumps(row), flush=True)
        if val_loss < best - 1e-4: best = val_loss; stale = 0; torch.save({"model": model.state_dict(), "backend": "medicalnet", "in_channels": len(MODALITIES), "epoch": epoch}, args.output / "best.pt")
        else: stale += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "backend": "medicalnet", "in_channels": len(MODALITIES), "epoch": epoch, "best": best, "stale": stale, "history": history}, args.output / "last.pt")
        if stale >= args.patience: break
    summary = {"cases": len(records), "case_counts": {s: sum(v == s for v in split_map.values()) for s in ("train", "val", "test")}, "history": history, "best_val_loss": best, "target_shape": shape}; (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
