#!/usr/bin/env python3
"""Build the study tensor cache once, so training epochs stop re-decoding NIfTI.

``StudyNiftiDataset.__getitem__`` is deterministic - no augmentation, fixed
``target_shape`` - so the tensor it produces in epoch 1 is bit-identical to the
one it produces in epoch 50.  Decoding four gzipped volumes, normalizing them at
full resolution and resampling them costs 0.3-1.3 s per study, which dwarfs the
GPU step at 64x64x32.

Run this once; then pass the same ``--cache-dir`` to train_medicalnet.py and
train_goal4.py.  Both share the cache when ``--target-shape`` matches.

    python -m scripts.precache_studies \
      --file-index ./file_check/file_index.csv \
      --labels-csv ./readout/series_merged.csv \
      --cache-dir /2026aicompetition/workspace/cache/study_64x64x32 \
      --target-shape 64 64 32 --workers 8

Each cached study is 4 * prod(target_shape) * 2 bytes in float16: 1.05 MB at
64x64x32.  Re-running is cheap - finished studies are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from training.cache import MODALITIES, cache_path, load_study_image
from training.study_dataset import build_study_records


def _init_worker() -> None:
    # Each worker resizes with torch; without this they each spawn a full thread
    # pool and fight over the same cores.
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # pragma: no cover - torch is always present in practice
        pass


def _build_one(payload: tuple[str, tuple[str | None, ...], tuple[int, ...], str, str, bool]) -> tuple[str, str | None]:
    accession, paths, shape, cache_dir, dtype, force = payload
    try:
        load_study_image(
            paths,
            shape,
            accession=accession,
            cache_dir=cache_dir,
            cache_dtype=np.float32 if dtype == "float32" else np.float16,
            force=force,
        )
        return accession, None
    except Exception as exc:
        return accession, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file-index", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--target-shape", nargs=3, type=int, default=(64, 64, 32))
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--label-column", default="check__glioma_with_label__std")
    parser.add_argument("--require-all-modalities", action="store_true")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="Rebuild even when the cache file already exists.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N studies (smoke test).")
    args = parser.parse_args()

    shape = tuple(int(x) for x in args.target_shape)
    records = build_study_records(
        args.file_index,
        args.labels_csv,
        require_all_modalities=args.require_all_modalities,
        label_column=args.label_column,
    )
    if args.limit:
        records = records[: args.limit]
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    skipped = 0
    for record in records:
        paths = tuple(record.modalities.get(modality) for modality in MODALITIES)
        if not any(paths):
            continue
        if not args.force and cache_path(paths, shape, args.cache_dir, accession=record.accession).is_file():
            skipped += 1
            continue
        pending.append((record.accession, paths, shape, str(args.cache_dir), args.cache_dtype, args.force))

    per_study_mb = 4 * shape[0] * shape[1] * shape[2] * (4 if args.cache_dtype == "float32" else 2) / 1e6
    print(
        json.dumps(
            {
                "studies": len(records),
                "already_cached": skipped,
                "to_build": len(pending),
                "workers": args.workers,
                "target_shape": list(shape),
                "cache_dir": str(args.cache_dir),
                "estimated_total_mb": round(len(records) * per_study_mb, 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not pending:
        return

    started = time.perf_counter()
    done = 0
    failures: list[dict[str, str]] = []
    workers = max(1, args.workers)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = [pool.submit(_build_one, payload) for payload in pending]
        for future in as_completed(futures):
            accession, error = future.result()
            done += 1
            if error:
                failures.append({"AccessionNumber": accession, "error": error})
            if done % 25 == 0 or done == len(pending):
                elapsed = time.perf_counter() - started
                rate = done / max(elapsed, 1e-6)
                print(
                    json.dumps(
                        {
                            "done": done,
                            "total": len(pending),
                            "failed": len(failures),
                            "studies_per_s": round(rate, 2),
                            "eta_s": round((len(pending) - done) / max(rate, 1e-6)),
                        }
                    ),
                    flush=True,
                )

    report = {
        "built": done - len(failures),
        "failed": len(failures),
        "seconds": round(time.perf_counter() - started, 1),
        "failures": failures[:20],
    }
    (args.cache_dir / "precache_report.json").write_text(
        json.dumps({**report, "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
