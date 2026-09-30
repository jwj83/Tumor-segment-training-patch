#!/usr/bin/env python3
"""Export one four-channel nnUNet dataset per Goal5 target.

Two targets, matching the competition spec:

    core      task A, "T1增强核心区"    -> 肿瘤瘤体 drawn on T1CE
    abnormal  task B, "Flair/T2总异常区" -> 瘤体 ∪ 水肿 ∪ 全肿瘤 on FLAIR, else T2

Three properties this export guarantees:

1. The reference grid is **the series the mask was drawn on**.  Resampling a
   label onto another series interpolates it and silently clips whatever falls
   outside that series' FOV; here the label is only ever resampled along with
   its own image, and the other three modalities come to it.

2. Every axis is at least ``--min-spacing`` but is **never upsampled**.  A
   clinical 0.45x0.45x6 mm axial series becomes 1.5x1.5x6 mm, an 11x drop in
   voxels, while a 1 mm isotropic 3-D series drops 3.4x.  Forcing true isotropy
   would instead interpolate 6 mm slices up to 1.5 mm and inflate the volume.

3. Cases are keyed by AccessionNumber and skipped when already complete, so an
   interrupted export resumes instead of renumbering and orphaning cases.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from training.study_dataset import MODALITIES, StudyRecord, build_study_records


TARGETS = ("core", "abnormal")


def sanitize(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_")
    return cleaned or "case"


def reference_grid(image: nib.Nifti1Image, min_spacing) -> tuple[tuple[int, int, int], np.ndarray]:
    """Coarsest grid with every axis at or above ``min_spacing``; never finer."""

    zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
    zooms[~np.isfinite(zooms) | (zooms <= 0)] = 1.0
    scale = np.maximum(np.asarray(min_spacing, dtype=float) / zooms, 1.0)
    shape = np.maximum(np.ceil(np.asarray(image.shape[:3], dtype=float) / scale), 1.0).astype(int)
    affine = np.array(image.affine, dtype=float)
    affine[:3, :3] = affine[:3, :3] * scale[np.newaxis, :]
    return (int(shape[0]), int(shape[1]), int(shape[2])), affine


def to_grid(image: nib.Nifti1Image, grid, *, is_mask: bool) -> np.ndarray:
    shape, affine = grid
    if image.shape[:3] == shape and np.allclose(image.affine, affine, atol=1e-4):
        data = np.asarray(image.dataobj, dtype=np.float32)
    else:
        resampled = resample_from_to(image, (shape, affine), order=0 if is_mask else 1, cval=0.0)
        data = np.asarray(resampled.dataobj, dtype=np.float32)
    np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return data


def save(path: Path, data: np.ndarray, affine: np.ndarray, dtype) -> None:
    """Write with an explicit dtype.

    Reusing the reference header would inherit its data type, so a float32
    volume would be quantised back into the reference's int16 and a uint8 label
    would be stored as int16.
    """

    image = nib.Nifti1Image(np.ascontiguousarray(data, dtype=dtype), affine)
    image.header.set_data_dtype(dtype)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def load_3d(path: str | Path) -> nib.Nifti1Image | None:
    image = nib.load(str(path))
    if len(image.shape) < 3:
        return None
    if len(image.shape) > 3:
        # Dynamic or multi-b-value series: nnUNet takes 3-D channels only.
        if int(np.prod(image.shape[3:])) != 1:
            return None
        image = image.slicer[:, :, :, 0]
    return image


def target_masks(record: StudyRecord, target: str) -> tuple[tuple[str, ...], str | None]:
    if target == "core":
        return record.core_masks, record.core_mask_modality
    return record.abnormal_masks, record.abnormal_mask_modality


def build_case(record: StudyRecord, target: str, min_spacing, out_dir: Path, case_id: str) -> dict:
    masks, mask_modality = target_masks(record, target)
    if not masks or not mask_modality:
        return {"skip": "no_mask"}
    reference_path = record.modalities.get(mask_modality)
    if not reference_path:
        return {"skip": "mask_modality_missing"}
    reference = load_3d(reference_path)
    if reference is None:
        return {"skip": "reference_not_3d"}

    grid = reference_grid(reference, min_spacing)
    shape, affine = grid

    union = np.zeros(shape, dtype=bool)
    for mask_path in masks:
        mask_image = load_3d(mask_path)
        if mask_image is None:
            continue
        union |= to_grid(mask_image, grid, is_mask=True) > 0.5
    foreground = int(union.sum())
    if foreground == 0:
        # Either an empty annotation or one that fell outside the grid; training
        # on it would teach the model that this study has nothing to find.
        return {"skip": "empty_mask_after_resample"}

    channels = []
    for modality in MODALITIES:
        path = record.modalities.get(modality)
        image = load_3d(path) if path else None
        channels.append(np.zeros(shape, dtype=np.float32) if image is None else to_grid(image, grid, is_mask=False))

    save(out_dir / "labelsTr" / f"{case_id}.nii.gz", union, affine, np.uint8)
    for index, values in enumerate(channels):
        save(out_dir / "imagesTr" / f"{case_id}_{index:04d}.nii.gz", values, affine, np.float32)
    return {
        "case": case_id,
        "AccessionNumber": record.accession,
        "target": target,
        "reference_modality": mask_modality,
        "reference_series": str(reference_path),
        "masks": list(masks),
        "native_shape": [int(x) for x in reference.shape[:3]],
        "native_spacing": [round(float(z), 4) for z in reference.header.get_zooms()[:3]],
        "exported_shape": list(shape),
        "foreground_voxels": foreground,
    }


def case_files(out_dir: Path, case_id: str) -> list[Path]:
    return [out_dir / "labelsTr" / f"{case_id}.nii.gz"] + [
        out_dir / "imagesTr" / f"{case_id}_{index:04d}.nii.gz" for index in range(len(MODALITIES))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-id", type=int, default=0, help="Recorded in dataset.json only; the directory name carries the real id.")
    parser.add_argument("--min-spacing", nargs=3, type=float, default=(1.5, 1.5, 1.5), help="Lower bound per axis in mm. Axes already coarser than this are left alone.")
    parser.add_argument("--force", action="store_true", help="Rebuild cases whose files already exist.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    records = build_study_records(args.index, args.labels_csv, require_all_modalities=False)
    if args.limit:
        records = records[: args.limit]

    selected: list[dict] = []
    skipped: list[dict] = []
    reused = 0
    for record in records:
        case_id = f"case_{sanitize(record.accession)}"
        masks, mask_modality = target_masks(record, args.target)
        if not masks or not mask_modality:
            skipped.append({"AccessionNumber": record.accession, "reason": "no_mask"})
            continue
        if not args.force and all(path.is_file() for path in case_files(args.out_dir, case_id)):
            selected.append({"case": case_id, "AccessionNumber": record.accession, "target": args.target, "reused": True})
            reused += 1
            continue
        result = build_case(record, args.target, tuple(args.min_spacing), args.out_dir, case_id)
        if "skip" in result:
            skipped.append({"AccessionNumber": record.accession, "reason": result["skip"]})
            continue
        selected.append(result)
        if len(selected) % 50 == 0:
            print(json.dumps({"exported": len(selected), "skipped": len(skipped), "of": len(records)}), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = {
        "channel_names": {str(index): modality.upper() for index, modality in enumerate(MODALITIES)},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(selected),
        "file_ending": ".nii.gz",
        "name": args.dataset_name,
        "description": f"Goal5 four-channel dataset; target={args.target}; min_spacing={list(args.min_spacing)}",
    }
    (args.out_dir / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "case_map.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "skipped.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")

    built = [row for row in selected if not row.get("reused")]
    voxels = [int(np.prod(row["exported_shape"])) for row in built if "exported_shape" in row]
    reasons: dict[str, int] = {}
    for row in skipped:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    summary = {
        "out_dir": str(args.out_dir),
        "target": args.target,
        "studies": len(records),
        "cases": len(selected),
        "reused": reused,
        "skipped": len(skipped),
        "skip_reasons": reasons,
        "min_spacing": list(args.min_spacing),
        "median_exported_voxels": int(np.median(voxels)) if voxels else 0,
        "estimated_imagesTr_gb": round(sum(voxels) * len(MODALITIES) * 4 / 1e9, 2) if voxels else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
