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
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from training.cache import MODALITIES, load_study_image, normalize_volume
from training.dataset import load_nifti, mask_label_from_path, normalize_mask_name
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
    # Which modality each mask was drawn on.  Goal5 must use that series as the
    # reference grid, otherwise the annotation is resampled across series and
    # can be clipped by a smaller FOV.
    core_mask_modality: str | None = None
    abnormal_mask_modality: str | None = None
    # Every mask file for the target.  Multifocal lesions are annotated as
    # 水肿_1_mask.nii.gz, 水肿_2_mask.nii.gz, ... so a target is a list, not one
    # file; core_mask/abnormal_mask keep the first entry for older callers.
    core_masks: tuple[str, ...] = ()
    abnormal_masks: tuple[str, ...] = ()


# Goal5 targets, following the competition annotation protocol:
#   T1CE            -> 肿瘤瘤体             task A, "T1增强核心区"
#   FLAIR else T2   -> 瘤体 / 水肿 / 全肿瘤  task B, "Flair/T2总异常区"
# Task B is a union.  The protocol lets an annotator either draw 瘤体 and 水肿
# separately or merge them into one 全肿瘤 when the two cannot be told apart, so
# picking a single file by priority would make those two conventions disagree:
# a study with both files would lose its tumour body and teach the model that
# the core is *not* part of the total abnormality.
CORE_MASK_LABELS = ("肿瘤瘤体", "瘤体", "全肿瘤")
ABNORMAL_MASK_LABELS = ("瘤体", "水肿", "全肿瘤")


def _labelled_masks(series_dir: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    if not series_dir:
        return grouped
    for path in sorted(Path(series_dir).glob("*_mask.nii.gz")):
        grouped.setdefault(normalize_mask_name(mask_label_from_path(path)), []).append(str(path))
    return grouped


def collect_masks(series_dir: str, labels: Sequence[str], *, union: bool) -> tuple[str, ...]:
    """Every mask file belonging to one target.

    ``union`` merges all listed labels; otherwise the first label that has any
    file wins.  Either way all files carrying that label are returned, so a
    multifocal lesion does not silently lose its second and third component.
    """

    grouped = _labelled_masks(series_dir)
    if union:
        return tuple(path for label in labels for path in grouped.get(normalize_mask_name(label), []))
    for label in labels:
        hits = grouped.get(normalize_mask_name(label), [])
        if hits:
            return tuple(hits)
    return ()


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
        core_masks = collect_masks(t1ce_row.get("series_dir", ""), CORE_MASK_LABELS, union=False) if t1ce_row else ()
        core_mask_modality = "t1ce" if core_masks else None
        abnormal_masks = collect_masks(flair_row.get("series_dir", ""), ABNORMAL_MASK_LABELS, union=True) if flair_row else ()
        abnormal_mask_modality = "flair" if abnormal_masks else None
        if not abnormal_masks and t2_row:
            abnormal_masks = collect_masks(t2_row.get("series_dir", ""), ABNORMAL_MASK_LABELS, union=True)
            abnormal_mask_modality = "t2" if abnormal_masks else None
        records.append(
            StudyRecord(
                accession=accession,
                modalities={modality: modality_rows.get(modality, {}).get("series_path") for modality in MODALITIES},
                label=label,
                core_mask=core_masks[0] if core_masks else None,
                abnormal_mask=abnormal_masks[0] if abnormal_masks else None,
                core_mask_modality=core_mask_modality,
                abnormal_mask_modality=abnormal_mask_modality,
                core_masks=core_masks,
                abnormal_masks=abnormal_masks,
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
    "ABNORMAL_MASK_LABELS",
    "CORE_MASK_LABELS",
    "MODALITIES",
    "StudyRecord",
    "StudyNiftiDataset",
    "build_study_records",
    "canonical_modality",
    "collect_masks",
    "normalize_volume",
]
