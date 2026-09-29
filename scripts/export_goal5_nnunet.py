#!/usr/bin/env python3
"""Export one geometry-consistent four-channel nnUNet dataset per Goal5 target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from training.study_dataset import MODALITIES, StudyRecord, build_study_records


def load_resampled(path: Path, reference: nib.Nifti1Image, *, is_mask: bool) -> nib.Nifti1Image:
    source = nib.load(str(path))
    if source.shape == reference.shape and np.allclose(source.affine, reference.affine):
        image = source
    else:
        image = resample_from_to(source, reference, order=0 if is_mask else 1)
    data = np.asarray(image.dataobj, dtype=np.float32)
    data = np.nan_to_num(data, copy=False)
    if is_mask:
        data = (data > 0).astype(np.uint8)
    return nib.Nifti1Image(data, reference.affine, reference.header)


def write_image(path: Path, image: nib.Nifti1Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def reference_for(record: StudyRecord, target: str) -> Path | None:
    if target == "core":
        preferred = ("t1ce", "t1", "flair", "t2")
    else:
        preferred = ("flair", "t2", "t1ce", "t1")
    return next((Path(record.modalities[m]) for m in preferred if record.modalities.get(m)), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("core", "abnormal"), required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()

    records = build_study_records(args.index, args.labels_csv, require_all_modalities=False)
    images_dir = args.out_dir / "imagesTr"
    labels_dir = args.out_dir / "labelsTr"
    selected = []
    for record in records:
        mask_path = record.core_mask if args.target == "core" else record.abnormal_mask
        reference_path = reference_for(record, args.target)
        if not mask_path or not Path(mask_path).is_file() or reference_path is None:
            continue
        reference = nib.load(str(reference_path))
        case_id = f"case_{len(selected):06d}"
        for channel, modality in enumerate(MODALITIES):
            source_path = record.modalities.get(modality)
            destination = images_dir / f"{case_id}_{channel:04d}.nii.gz"
            if source_path:
                write_image(destination, load_resampled(Path(source_path), reference, is_mask=False))
            else:
                write_image(destination, nib.Nifti1Image(np.zeros(reference.shape, dtype=np.float32), reference.affine, reference.header))
        write_image(labels_dir / f"{case_id}.nii.gz", load_resampled(Path(mask_path), reference, is_mask=True))
        selected.append({"case": case_id, "AccessionNumber": record.accession, "target": args.target, "reference": str(reference_path)})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = {
        "channel_names": {str(i): modality.upper() for i, modality in enumerate(MODALITIES)},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(selected),
        "file_ending": ".nii.gz",
        "name": args.dataset_name,
        "description": f"Goal5 four-channel dataset; target={args.target}; dataset_id={args.dataset_id}",
    }
    (args.out_dir / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "case_map.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "target": args.target, "cases": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
