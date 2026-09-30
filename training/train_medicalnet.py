#!/usr/bin/env python3
"""Train a case-level MedicalNet classifier for Goal3."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from training.models.medicalnet import MedicalNet3DClassifier
from training.runtime import amp_tools, tune_backend
from training.study_dataset import MODALITIES, StudyNiftiDataset, StudyRecord, build_study_records


def auc(labels, scores):
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
        return None
    # Rank positives in ascending score order; the rank-sum formula below
    # otherwise reports 1 - ROC-AUC when scores are sorted descending.
    order = np.argsort(s, kind="mergesort")
    y = y[order]
    positives = y.sum()
    negatives = y.size - positives
    ranks = np.flatnonzero(y == 1) + 1
    return float((ranks.sum() - positives * (positives + 1) / 2) / (positives * negatives))


def split_records(records: list[StudyRecord], val_fraction: float, test_fraction: float, seed: int):
    accessions = sorted({record.accession for record in records})
    rng = random.Random(seed); rng.shuffle(accessions)
    n_test = round(len(accessions) * test_fraction); n_val = round(len(accessions) * val_fraction)
    split = {a: "test" if i < n_test else "val" if i < n_test + n_val else "train" for i, a in enumerate(accessions)}
    return ([r for r in records if split[r.accession] == "train"], [r for r in records if split[r.accession] == "val"], [r for r in records if split[r.accession] == "test"], split)


def loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    kwargs = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": num_workers}
    if torch.cuda.is_available(): kwargs["pin_memory"] = True
    if num_workers > 0:
        kwargs["persistent_workers"] = True; kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def evaluate(model, data_loader, device, criterion, autocast):
    model.eval(); total = 0.0; count = 0; labels = []; scores = []
    with torch.inference_mode():
        for batch in data_loader:
            x = batch["image"].to(device, non_blocking=True)
            # collate turns Python floats into float64; BCEWithLogitsLoss wants float32.
            y = batch["label"].to(device, non_blocking=True).float()
            with autocast():
                logits = model(x).reshape(-1)
            logits = logits.float()
            total += float(criterion(logits, y).item()) * x.shape[0]; count += x.shape[0]
            labels.extend(y.cpu().tolist()); scores.extend(torch.sigmoid(logits).cpu().tolist())
    return {"loss": total / max(count, 1), "case_roc_auc": auc(labels, scores), "cases": count}


def load_resume(model, optimizer, scheduler, scaler, path: Path, device):
    payload = torch.load(path, map_location=device); state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if payload.get("optimizer"):
        try: optimizer.load_state_dict(payload["optimizer"])
        except ValueError: pass
    if payload.get("scheduler"):
        try: scheduler.load_state_dict(payload["scheduler"])
        except ValueError: pass
    if payload.get("scaler"):
        try: scaler.load_state_dict(payload["scaler"])
        except (ValueError, KeyError, RuntimeError): pass
    return payload, {"missing": list(missing), "unexpected": list(unexpected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-index", type=Path, required=True); parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--label-column", default="check__glioma_with_label__std"); parser.add_argument("--medicalnet-checkpoint", type=Path, default=None); parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-shape", nargs=3, type=int, default=(64, 64, 32)); parser.add_argument("--val-fraction", type=float, default=0.2); parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Reuse preprocessed study tensors; fill it once with scripts/precache_studies.py.")
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--amp", choices=("auto", "on", "off"), default="auto", help="Mixed precision; auto enables it on CUDA.")
    parser.add_argument("--save-last-every", type=int, default=5, help="last.pt carries optimizer state (~3x the model size); writing it every epoch is slow on network storage.")
    parser.add_argument("--patience", type=int, default=12); parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--lr-factor", type=float, default=0.5); parser.add_argument("--lr-patience", type=int, default=3); parser.add_argument("--min-lr", type=float, default=1e-6); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--output", type=Path, default=Path("checkpoint_medicalnet"))
    args = parser.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    records = build_study_records(args.file_index, args.labels_csv, require_all_modalities=False, label_column=args.label_column)
    records = [r for r in records if r.label is not None]
    train_records, val_records, test_records, split_map = split_records(records, args.val_fraction, args.test_fraction, args.seed)
    if not train_records or not val_records: raise RuntimeError("train/val is empty; check index and split fractions")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "splits.json").write_text(json.dumps({s: sorted(a for a, v in split_map.items() if v == s) for s in ("train", "val", "test")}, ensure_ascii=False, indent=2), encoding="utf-8")
    shape = tuple(args.target_shape)
    dataset_kwargs = {"target_shape": shape, "cache_dir": args.cache_dir, "cache_dtype": args.cache_dtype}
    train_loader = loader(StudyNiftiDataset(train_records, **dataset_kwargs), args.batch_size, True, args.num_workers)
    # Validation is a fifth of the work; half the workers keep up with it and
    # leave CPU for the training loader.
    val_workers = max(args.num_workers // 2, 0)
    val_loader = loader(StudyNiftiDataset(val_records, **dataset_kwargs), args.batch_size, False, val_workers)
    positives = sum(float(r.label == 1) for r in train_records); negatives = sum(float(r.label == 0) for r in train_records)
    if positives == 0 or negatives == 0: raise RuntimeError(f"train split must contain both classes, got positives={positives}, negatives={negatives}")
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); tune_backend(device)
    autocast, scaler, amp_enabled = amp_tools(args.amp, device)
    model = MedicalNet3DClassifier(depth=50, in_channels=len(MODALITIES), num_classes=1, checkpoint=args.medicalnet_checkpoint).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr); criterion = BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
    best = -float("inf"); stale = 0; best_epoch = 0; history = []; start_epoch = 1
    print(json.dumps({"device": str(device), "amp": amp_enabled, "cache_dir": str(args.cache_dir) if args.cache_dir else None, "cases": {"train": len(train_records), "val": len(val_records), "test": len(test_records)}}, ensure_ascii=False), flush=True)
    if args.resume:
        payload, report = load_resume(model, optimizer, scheduler, scaler, args.resume, device); start_epoch = int(payload.get("epoch", 0)) + 1; best = float(payload.get("best", best)); best_epoch = int(payload.get("best_epoch", 0)); stale = int(payload.get("stale", 0)); history = list(payload.get("history", []))
        print(json.dumps({"resumed_from": str(args.resume), "start_epoch": start_epoch, "load_report": report}, ensure_ascii=False), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); running = 0.0; seen = 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True); y = batch["label"].to(device, non_blocking=True).float(); optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss = criterion(model(x).reshape(-1).float(), y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); running += float(loss.item()) * x.shape[0]; seen += x.shape[0]
        metrics = evaluate(model, val_loader, device, criterion, autocast); score = -float("inf") if metrics["case_roc_auc"] is None else float(metrics["case_roc_auc"]); scheduler.step(score)
        row = {"epoch": epoch, "train_loss": running / max(seen, 1), "lr": optimizer.param_groups[0]["lr"], **{f"val_{k}": v for k, v in metrics.items()}}; history.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
        if score > best + 1e-4:
            best, best_epoch, stale = score, epoch, 0; torch.save({"model": model.state_dict(), "backend": "medicalnet", "in_channels": len(MODALITIES), "epoch": epoch}, args.output / "best.pt")
        else: stale += 1
        final_epoch = epoch == args.epochs or stale >= args.patience
        if final_epoch or args.save_last_every <= 1 or epoch % args.save_last_every == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "backend": "medicalnet", "in_channels": len(MODALITIES), "epoch": epoch, "best": best, "best_epoch": best_epoch, "stale": stale, "history": history}, args.output / "last.pt")
        if stale >= args.patience: break
    if (args.output / "best.pt").is_file(): model.load_state_dict(torch.load(args.output / "best.pt", map_location=device)["model"], strict=False)
    # Built only now: a persistent test loader would hold workers for the whole run.
    test_metrics = evaluate(model, loader(StudyNiftiDataset(test_records, **dataset_kwargs), args.batch_size, False, val_workers), device, criterion, autocast) if test_records else None
    summary = {"records": len(records), "case_counts": {s: sum(v == s for v in split_map.values()) for s in ("train", "val", "test")}, "best_epoch": best_epoch, "best_val_case_roc_auc": None if best == -float("inf") else best, "test": test_metrics, "history": history, "pos_weight": float(pos_weight.item()), "target_shape": shape, "amp": amp_enabled, "cache_dir": str(args.cache_dir) if args.cache_dir else None}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
