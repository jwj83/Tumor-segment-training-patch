"""Deterministic preprocessed-study cache.

``StudyNiftiDataset.__getitem__`` is a pure function of the four modality paths
and ``target_shape``: there is no augmentation and no randomness, so epoch 2
recomputes exactly the bytes epoch 1 already produced.  Decoding four gzipped
NIfTI volumes, normalizing them at full resolution and trilinearly resampling
them costs 0.3-1.3 s per study, which is 10-100x the GPU step for a 64x64x32
input.  Caching the finished ``(4, *target_shape)`` tensor turns every epoch
after the first into a ~1 MB read.

This module owns ``MODALITIES`` and ``normalize_volume`` so it stays free of any
import from ``training.study_dataset`` (which imports *this*).  Both names are
re-exported there, so existing ``from training.study_dataset import MODALITIES``
callers are unaffected.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

from training.resize import resize_volume


MODALITIES = ("t1", "t1ce", "t2", "flair")

# Bump whenever ``normalize_volume`` or ``build_study_image`` changes meaning.
# The version is part of the cache file name, so stale caches are never read.
PREPROC_VERSION = 2

# Upper bound on the voxel count fed to np.percentile.  A 1/99 percentile over a
# 1M-voxel stride sample matches the full-volume value to ~1e-4 while running
# ~40x faster; np.percentile over 11M voxels alone cost 100 ms per modality.
PERCENTILE_SAMPLE = 1_000_000


def normalize_volume(array: np.ndarray) -> np.ndarray:
    """Clip to the 1/99 percentile range and rescale to [0, 1].

    Allocates one buffer and mutates it in place.  The previous implementation
    made five full-volume passes (isfinite mask, boolean-indexed copy,
    nan_to_num copy, clip copy, rescale copy) plus a full-volume percentile.
    """

    values = np.array(array, dtype=np.float32, copy=True)
    np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    flat = values.reshape(-1)
    stride = max(1, flat.size // PERCENTILE_SAMPLE)
    lo, hi = np.percentile(flat[::stride], (1.0, 99.0))
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        # All-zero, constant, or fully non-finite volume: nothing to rescale.
        return np.zeros(values.shape, dtype=np.float32)
    values -= lo
    values *= 1.0 / (hi - lo)
    np.clip(values, 0.0, 1.0, out=values)
    return values


def build_study_image(
    paths: Sequence[str | os.PathLike[str] | None],
    target_shape: Sequence[int],
    *,
    dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Read, normalize and resize one study into ``(4, *target_shape)``.

    Each modality is resized on its own before stacking: volumes inside one
    study routinely disagree on the native grid (320x320x20 vs 310x320x20), so
    stacking first fails.  A missing modality becomes a zero channel already at
    ``target_shape``.
    """

    shape = tuple(int(x) for x in target_shape)
    channels: list[np.ndarray] = []
    present: list[float] = []
    for path in paths:
        if path:
            values = normalize_volume(np.asarray(nib.load(str(path)).dataobj, dtype=np.float32))
            values = np.asarray(resize_volume(values, shape, is_mask=False), dtype=np.float32)
            channels.append(values)
            present.append(1.0)
        else:
            channels.append(np.zeros(shape, dtype=np.float32))
            present.append(0.0)
    if not any(present):
        raise RuntimeError("study has no readable modality")
    image = np.stack(channels, axis=0).astype(dtype, copy=False)
    return image, np.asarray(present, dtype=np.float32)


def _fingerprint(paths: Sequence[str | os.PathLike[str] | None], shape: tuple[int, ...]) -> str:
    payload = "|".join(
        [f"v{PREPROC_VERSION}", "x".join(str(x) for x in shape)]
        + [str(path or "") for path in paths]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def cache_path(
    paths: Sequence[str | os.PathLike[str] | None],
    target_shape: Sequence[int],
    cache_dir: str | os.PathLike[str],
    *,
    accession: str = "study",
) -> Path:
    """Cache file name; changing any source path or the version renames it."""

    shape = tuple(int(x) for x in target_shape)
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", str(accession)) or "study"
    tag = "x".join(str(x) for x in shape)
    return Path(cache_dir) / f"{safe}__{tag}__{_fingerprint(paths, shape)}.npz"


def _write_atomic(path: Path, image: np.ndarray, present: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp name: several DataLoader workers may build the same study.
    temp = path.parent / f".{path.name}.{os.getpid()}.tmp.npz"
    try:
        with temp.open("wb") as handle:
            np.savez(handle, image=image, present=present)
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _read(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        with np.load(path) as data:
            return np.asarray(data["image"]), np.asarray(data["present"], dtype=np.float32)
    except Exception:
        # Truncated or half-written file from an interrupted run: rebuild it.
        return None


def load_study_image(
    paths: Sequence[str | os.PathLike[str] | None],
    target_shape: Sequence[int],
    *,
    accession: str = "study",
    cache_dir: str | os.PathLike[str] | None = None,
    cache_dtype: np.dtype | type = np.float16,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(image float32 (4, *shape), present float32 (4,))``.

    Without ``cache_dir`` this is the plain on-the-fly path.  With it, the first
    call writes the cache and every later call reads ~1 MB instead of decoding
    tens of MB of gzipped NIfTI.
    """

    shape = tuple(int(x) for x in target_shape)
    if cache_dir is None:
        return build_study_image(paths, shape, dtype=np.float32)

    path = cache_path(paths, shape, cache_dir, accession=accession)
    if not force and path.is_file():
        cached = _read(path)
        if cached is not None and cached[0].shape == (len(MODALITIES), *shape):
            return np.asarray(cached[0], dtype=np.float32), cached[1]

    image, present = build_study_image(paths, shape, dtype=cache_dtype)
    _write_atomic(path, image, present)
    return np.asarray(image, dtype=np.float32), present


__all__ = [
    "MODALITIES",
    "PREPROC_VERSION",
    "build_study_image",
    "cache_path",
    "load_study_image",
    "normalize_volume",
]
