"""Study-level (four-modality) lazy dataset.

``file_index.csv`` has one row per sequence.  This module groups those rows by
AccessionNumber so the model sees one study at a time with channels
T1/T1CE/T2/FLAIR, matching the competition inference path.

Reading and resampling the volumes lives in ``training.cache``, which can keep
the finished ``(4, *target_shape)`` tensor on disk.  ``__getitem__`` has no
augmentation and no randomness, so every epoch after the first recomputes bytes
it already produced; pass ``cache_dir`` to skip that work.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from training.cache import MODALITIES, load_study_image, normalize_volume
from training.dataset import choose_mask_path, load_nifti
from training.resize import resize_volume


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
    """One accession per sample.

    ``cache_dir`` stores each finished study tensor as a ~1 MB ``.npz``; the
    first epoch fills it and later epochs read it instead of decoding four
    gzipped NIfTI volumes.  Leave it as ``None`` for the original behaviour.
    """

    def __init__(
        self,
        records: Iterable[StudyRecord],
        *,
        target_shape: tuple[int, int, int] = (32, 32, 16),
        task: str = "classification",
        require_label: bool = True,
        require_masks: bool = False,
        cache_dir: str | Path | None = None,
        cache_dtype: str = "float16",
    ) -> None:
        self.records = [record for record in records if (not require_label or record.label is not None)]
        if require_masks:
            self.records = [record for record in self.records if record.core_mask or record.abnormal_mask]
        self.target_shape = tuple(int(x) for x in target_shape)
        self.task = task
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_dtype = np.float32 if str(cache_dtype) == "float32" else np.float16

    def __len__(self) -> int:
        return len(self.records)

    def modality_paths(self, record: StudyRecord) -> tuple[str | None, ...]:
        return tuple(record.modalities.get(modality) for modality in MODALITIES)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        try:
            image, present = load_study_image(
                self.modality_paths(record),
                self.target_shape,
                accession=record.accession,
                cache_dir=self.cache_dir,
                cache_dtype=self.cache_dtype,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"study {record.accession}: {exc}") from exc
        item: dict[str, Any] = {
            "image": torch.from_numpy(image),
            "AccessionNumber": record.accession,
            # NaN rather than "": a batch mixing str and float breaks
            # default_collate, and require_label=False lets unlabelled studies in.
            "label": float("nan") if record.label is None else float(record.label),
            "has_label": 0.0 if record.label is None else 1.0,
            "modality_present": torch.from_numpy(present),
        }
        if self.task == "segmentation":
            if record.core_mask:
                item["core_mask"] = resize_volume(load_nifti(Path(record.core_mask), is_mask=True), self.target_shape, is_mask=True)
            if record.abnormal_mask:
                item["abnormal_mask"] = resize_volume(load_nifti(Path(record.abnormal_mask), is_mask=True), self.target_shape, is_mask=True)
        return item


__all__ = [
    "MODALITIES",
    "StudyRecord",
    "StudyNiftiDataset",
    "build_study_records",
    "canonical_modality",
    "normalize_volume",
]
