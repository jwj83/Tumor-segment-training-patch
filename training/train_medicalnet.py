#!/usr/bin/env python3
"""Train one shared MedicalNet on available sequences; score at accession level."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from training.models.medicalnet import MedicalNet3DClassifier
from training.sequence_dataset import SequenceNiftiDataset, assign_accession_splits, build_sequence_records


def auc(labels, scores):
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
        return None
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    p = y.sum(); n = y.size - p
    ranks = np.flatnonzero(y == 1) + 1
    return float((ranks.sum() - p * (p + 1) / 2) / (p * n))


def evaluate(model, loader, device, criterion):
    model.eval(); total = 0.0; count = 0
    seq_y, seq_s, case_s, case_y = [], [], {}, {}
    with torch.inference_mode():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = model(x).reshape(-1)
            total += float(criterion(logits, y).item()) * x.shape[0]
            count += x.shape[0]
            scores = torch.sigmoid(logits).cpu().tolist()
            labels = y.cpu().tolist()
            for accession, score, label in zip(batch["accession"], scores, labels):
                case_s[accession] = max(case_s.get(accession, 0.0), float(score))
                case_y[accession] = float(label)
                seq_s.append(score); seq_y.append(label)
    names = sorted(case_s)
    return {
        "loss": total / max(count, 1),
        "sequence_roc_auc": auc(seq_y, seq_s),
        "case_roc_auc": auc([case_y[n] for n in names], [case_s[n] for n in names]),
        "sequences": count, "cases": len(names),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-index", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--label-column", default="check__glioma_with_label__std")
    parser.add_argument("--medicalnet-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-shape", nargs=3, type=int, default=(32, 32, 16))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("checkpoint_medicalnet"))
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    records = build_sequence_records(args.file_index, args.labels_csv, args.label_column)
    assign_accession_splits(records, args.val_fraction, args.test_fraction, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    split_ids = {split: sorted({r.accession for r in records if r.split == split}) for split in ("train", "val", "test")}
    (args.output / "splits.json").write_text(json.dumps(split_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    shape = tuple(args.target_shape)
    train_loader = DataLoader(SequenceNiftiDataset(records, "train", shape), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(SequenceNiftiDataset(records, "val", shape), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(SequenceNiftiDataset(records, "test", shape), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if not len(train_loader.dataset) or not len(val_loader.dataset):
        raise RuntimeError("train/val is empty; check index and split fractions")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedicalNet3DClassifier(depth=50, in_channels=1, num_classes=1, checkpoint=args.medicalnet_checkpoint).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = BCEWithLogitsLoss()
    best = -float("inf"); stale = 0; best_epoch = 0; history = []; start_epoch = 1
    if args.resume:
        payload = torch.load(args.resume, map_location=device)
        model.load_state_dict(payload.get("model", payload))
        if payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best = float(payload.get("best", best))
        best_epoch = int(payload.get("best_epoch", 0))
        stale = int(payload.get("stale", 0))
        history = list(payload.get("history", []))
        print(json.dumps({"resumed_from": str(args.resume), "start_epoch": start_epoch}, ensure_ascii=False), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); running = 0.0; seen = 0
        for batch in train_loader:
            x, y = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x).reshape(-1), y)
            loss.backward(); optimizer.step()
            running += float(loss.item()) * x.shape[0]; seen += x.shape[0]
        metrics = evaluate(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train_loss": running / max(seen, 1), **{f"val_{k}": v for k, v in metrics.items()}}
        history.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
        score = metrics["case_roc_auc"]
        score = -float("inf") if score is None else float(score)
        if score > best + 1e-4:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "backend": "medicalnet", "in_channels": 1, "aggregation": "case_max", "epoch": epoch}, args.output / "best.pt")
        else:
            stale += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "backend": "medicalnet", "in_channels": 1, "aggregation": "case_max", "epoch": epoch, "best": best, "best_epoch": best_epoch, "stale": stale, "history": history}, args.output / "last.pt")
        if stale >= args.patience:
            break

    if (args.output / "best.pt").is_file():
        model.load_state_dict(torch.load(args.output / "best.pt", map_location=device)["model"])
    summary = {"records": len(records), "sequence_counts": {s: sum(r.split == s for r in records) for s in ("train", "val", "test")}, "best_epoch": best_epoch, "best_val_case_roc_auc": None if best == -float("inf") else best, "test": evaluate(model, test_loader, device, criterion) if len(test_loader.dataset) else None, "history": history}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
