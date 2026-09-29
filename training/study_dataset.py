"""Study-level (four-modality) lazy dataset.

``file_index.csv`` has one row per sequence.  This module groups those rows by
AccessionNumber so the model sees one study at a time with channels
T1/T1CE/T2/FLAIR, matching the competition inference path.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from training.dataset import choose_mask_path, load_nifti
from training.resize import resize_volume


MODALITIES = ("t1", "t1ce", "t2", "flair")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def canonical_modality(value: str) -> str | None:
    text = re.sub(r"[^a-z0-9]+", "", value.lower())
    if "flair" in text or "t2flair" in text:
        return "flair"
    if "t1ce" in text or "t1c" in text or "enh" in text or "t1plusc" in text:
        return "t1ce"
    if text == "t1" or text.startswith("t1_") or text.endswith("t1"):
        return "t1"
    if "t2" in text:
        return "t2"
    return None


def normalize_volume(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.float32)
    values = np.nan_to_num(values, copy=True)
    lo, hi = np.percentile(values[finite], (1.0, 99.0))
    values = np.clip(values, lo, hi)
    return ((values - float(lo)) / max(float(hi - lo), 1e-6)).astype(np.float32)


@dataclass(frozen=True)
class StudyRecord:
    accession: str
    modalities: dict[str, str | None]
    label: float | None
    core_mask: str | None
    abnormal_mask: str | None


def _pick_mask(series_dir: str, name: str) -> str | None:
    if not series_dir:
        return None
    selected = choose_mask_path(
        sorted(Path(series_dir).glob("*_mask.nii.gz")), name
    )
    return str(selected) if selected else None


def build_study_records(
    file_index: str | Path,
    labels_csv: str | Path | None = None,
    *,
    require_all_modalities: bool = True,
    label_column: str = "check__glioma_with_label__std",
) -> list[StudyRecord]:
    rows = _read_csv(Path(file_index))
    labels: dict[str, dict[str, str]] = {}
    if labels_csv:
        for row in _read_csv(Path(labels_csv)):
            labels.setdefault(row.get("AccessionNumber", ""), row)

    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        accession = row.get("AccessionNumber", "")
        if not accession or row.get("status") not in {"series_and_mask_found", "series_found_mask_missing"}:
            continue
        modality = canonical_modality(row.get("SeriesType", ""))
        if modality is None or not row.get("series_path"):
            continue
        grouped.setdefault(accession, {}).setdefault(modality, row)

    records: list[StudyRecord] = []
    for accession, modality_rows in sorted(grouped.items()):
        if require_all_modalities and any(modality not in modality_rows for modality in MODALITIES):
            continue
        label_row = labels.get(accession, {})
        raw_label = label_row.get(label_column, "")
        try:
            label = float(raw_label) if raw_label != "" else None
        except ValueError:
            label = None
        t1ce_row = modality_rows.get("t1ce")
        flair_row = modality_rows.get("flair")
        t2_row = modality_rows.get("t2")
        core_mask = _pick_mask(t1ce_row.get("series_dir", ""), "t1ce_auto") if t1ce_row else None
        abnormal_mask = _pick_mask(flair_row.get("series_dir", ""), "t2_auto") if flair_row else None
        if abnormal_mask is None and t2_row:
            abnormal_mask = _pick_mask(t2_row.get("series_dir", ""), "t2_auto")
        records.append(
            StudyRecord(
                accession=accession,
                modalities={modality: modality_rows.get(modality, {}).get("series_path") for modality in MODALITIES},
                label=label,
                core_mask=core_mask,
                abnormal_mask=abnormal_mask,
            )
        )
    return records


class StudyNiftiDataset(Dataset):
    """One accession per sample; NIfTI files are loaded inside ``__getitem__``."""

    def __init__(
        self,
        records: Iterable[StudyRecord],
        *,
        target_shape: tuple[int, int, int] = (32, 32, 16),
        task: str = "classification",
        require_label: bool = True,
        require_masks: bool = False,
    ) -> None:
        self.records = [record for record in records if (not require_label or record.label is not None)]
        if require_masks:
            self.records = [record for record in self.records if record.core_mask or record.abnormal_mask]
        self.target_shape = target_shape
        self.task = task

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        channels = []
        present = []
        for modality in MODALITIES:
            path = record.modalities.get(modality)
            if path:
                values = normalize_volume(np.asarray(nib.load(path).dataobj, dtype=np.float32))
                # Modalities from one study can have different native grids.
                # Resize each volume before stacking; stacking raw arrays first
                # fails as soon as T1/T1CE/T2/FLAIR shapes differ.
                values = resize_volume(values, self.target_shape, is_mask=False)
                channels.append(torch.from_numpy(np.asarray(values, dtype=np.float32)))
                present.append(1.0)
            else:
                channels.append(None)
                present.append(0.0)
        if not any(channel is not None for channel in channels):
            raise RuntimeError(f"study {record.accession} has no readable modality")
        channels = [
            channel if channel is not None else torch.zeros(self.target_shape, dtype=torch.float32)
            for channel in channels
        ]
        image = torch.stack(channels, dim=0)
        item: dict[str, Any] = {
            "image": image,
            "AccessionNumber": record.accession,
            "label": "" if record.label is None else record.label,
            "modality_present": torch.tensor(present, dtype=torch.float32),
        }
        if self.task == "segmentation":
            if record.core_mask:
                item["core_mask"] = resize_volume(load_nifti(Path(record.core_mask), is_mask=True), self.target_shape, is_mask=True)
            if record.abnormal_mask:
                item["abnormal_mask"] = resize_volume(load_nifti(Path(record.abnormal_mask), is_mask=True), self.target_shape, is_mask=True)
        return item


__all__ = ["MODALITIES", "StudyRecord", "StudyNiftiDataset", "build_study_records", "canonical_modality"]
