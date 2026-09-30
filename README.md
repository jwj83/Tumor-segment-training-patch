# Tumor-segment training patch

Files to copy into the existing training checkout (`/2026aicompetition/workspace/Tumor-segment-training`).

## New in this round: study tensor cache

`StudyNiftiDataset.__getitem__` is deterministic — no augmentation, fixed
`target_shape` — so every epoch after the first re-decodes four gzipped NIfTI
volumes to rebuild bytes it already produced. That decode + percentile
normalization + trilinear resample costs roughly 0.3–1.3 s per study, against a
GPU step of tens of milliseconds at 64x64x32. The training loop was waiting on
the DataLoader, not the model.

`training/cache.py` stores each finished `(4, *target_shape)` tensor as one
`.npz` (1.05 MB in float16 at 64x64x32). Pass `--cache-dir` to either trainer
and the second epoch onward reads that instead.

Fill the cache once, in parallel, before training:

```bash
python -m scripts.precache_studies \
  --file-index ./file_check/file_index.csv \
  --labels-csv ./readout/series_merged.csv \
  --cache-dir /2026aicompetition/workspace/cache/study_64x64x32 \
  --target-shape 64 64 32 --workers 8
```

It is idempotent (finished studies are skipped) and writes
`precache_report.json` listing any study it could not read. Skipping this step
is fine — the cache fills itself during epoch 1 — but precaching uses more
processes than the DataLoader has workers, and goal3 and goal4 then share one
cache instead of each building it.

Then add the same directory to both training commands:

```bash
python -u -m training.train_medicalnet ... \
  --cache-dir /2026aicompetition/workspace/cache/study_64x64x32

python -u -m training.train_goal4 ... \
  --cache-dir /2026aicompetition/workspace/cache/study_64x64x32
```

The cache key covers `target_shape`, the four source paths and a preprocessing
version, so changing any of them writes a new file rather than reading a stale
one. Put the cache on local disk, not on the dataset mount.

## Files

New:

- `training/cache.py` — preprocessing + the on-disk cache. Owns `MODALITIES` and
  `normalize_volume`, both still importable from `training/study_dataset.py`.
- `training/runtime.py` — AMP helpers and cuDNN autotuning.
- `scripts/precache_studies.py` — one-shot parallel cache builder.
- `tests/test_study_cache.py` — copy into `tests/`.

Changed:

- `training/study_dataset.py` — reads through the cache; `normalize_volume`
  takes the 1/99 percentile over a strided sample instead of the full volume
  (same result to ~1e-4, several times faster) and works in place; an
  unlabelled study now collates as `NaN` with a `has_label` flag instead of `""`,
  which would have broken `default_collate` in a mixed batch.
- `training/train_medicalnet.py`, `training/train_goal4.py` — `--cache-dir`,
  `--cache-dtype`, `--amp`, `--save-last-every`; the test DataLoader is built
  after training instead of holding persistent workers throughout; validation
  uses half the workers; labels are cast to float32 before the loss; goal4 skips
  a batch in which every head is unlabelled (the loss would otherwise have no
  `grad_fn` and `backward()` would raise).

Unchanged by this round: `training/models/medicalnet.py`, `training/resize.py`,
`scripts/export_goal5_nnunet.py`, `scripts/train_goal5_nnunet.sh`.

## New options

```text
--cache-dir DIR          reuse preprocessed study tensors (the main speedup)
--cache-dtype float16    float32 doubles the cache size, no accuracy benefit here
--amp auto|on|off        auto = mixed precision on CUDA; use off if the loss goes NaN
--save-last-every 5      last.pt carries optimizer state (~3x the model, ~554 MB
                         for ResNet50-3D); writing it every epoch dominates a
                         cached epoch. best.pt is still written on every
                         improvement and holds the model only.
```

Resuming from a `last.pt` written before this patch still works; the scaler
state is simply absent.

## Earlier in this patch

The classification and Goal4 training entries use one `AccessionNumber` per
sample with up to four modality channels (T1, T1CE, T2, FLAIR), zero filling
missing modalities. They support the official MedicalNet
`models.resnet.resnet50` API, optional pretrained checkpoints, 64x64x32 inputs,
class weighting, ReduceLROnPlateau, early stopping, and `last.pt` resume.

```text
--medicalnet-checkpoint /path/to/medicalnet_weight.pth
--num-workers 4
--target-shape 64 64 32
--lr 1e-4
--patience 12
```

Goal5 nnUNet training is provided by:

- `scripts/export_goal5_nnunet.py`: creates geometry-consistent four-channel
  datasets for the core and abnormal targets, resampling each case to its T1CE
  or FLAIR/T2 reference grid.
- `scripts/train_goal5_nnunet.sh`: plans, verifies, preprocesses, and trains the
  two nnUNet datasets.

Run it from the training repository after setting `nnUNet_raw`,
`nnUNet_preprocessed`, and `nnUNet_results`.

## Verification before this commit

Against a synthetic dataset (24 studies with deliberately mismatched native
grids and missing modalities) and a stub matching the official MedicalNet
`resnet50` signature:

```text
python -m py_compile <all patched files>   OK
bash -n scripts/train_goal5_nnunet.sh      OK
python -m unittest discover -s tests       25 tests OK
precache + 3-epoch train_medicalnet run, with and without --cache-dir   OK
```

That is a correctness smoke test only. The timings quoted above are extrapolated
from per-volume measurements, not measured on the competition hardware.
