"""Regression tests for the study cache and the multi-shape stacking path."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from training.cache import cache_path, load_study_image, normalize_volume
from training.study_dataset import MODALITIES, StudyNiftiDataset, build_study_records


SHAPE = (64, 64, 32)

# One accession whose four modalities disagree on the native grid, and one that
# is missing FLAIR entirely.  The missing-modality study is listed first so the
# collate path sees it as batch element 0.
STUDIES = {
    "ACC_MISSING": {"T1": (40, 44, 9), "T1CE": (40, 44, 9), "T2": (36, 38, 7)},
    "ACC_FULL": {"T1": (52, 50, 11), "T1CE": (48, 50, 11), "T2": (44, 46, 8), "FLAIR": (40, 42, 6)},
}


def _write_dataset(root: Path) -> tuple[Path, Path]:
    rows = []
    rng = np.random.default_rng(0)
    for accession, modalities in STUDIES.items():
        for series_type, shape in modalities.items():
            uid = f"{accession}_{series_type}"
            series_dir = root / "data" / accession / uid
            series_dir.mkdir(parents=True, exist_ok=True)
            series_path = series_dir / f"{uid}.nii.gz"
            nib.save(nib.Nifti1Image((rng.random(shape) * 1000).astype(np.float32), np.eye(4)), series_path)
            mask = np.zeros(shape, dtype=np.uint8)
            mask[2:6, 2:6, 1:3] = 1
            nib.save(nib.Nifti1Image(mask, np.eye(4)), series_dir / "水肿_1_mask.nii.gz")
            nib.save(nib.Nifti1Image(mask, np.eye(4)), series_dir / "肿瘤瘤体_1_mask.nii.gz")
            rows.append(
                {
                    "AccessionNumber": accession,
                    "SeriesUid": uid,
                    "SeriesType": series_type,
                    "series_dir": str(series_dir),
                    "series_path": str(series_path),
                    "series_file_exists": "true",
                    "status": "series_and_mask_found",
                }
            )

    index_path = root / "file_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels_path = root / "labels.csv"
    with labels_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["AccessionNumber", "check__glioma_with_label__std"])
        writer.writeheader()
        writer.writerow({"AccessionNumber": "ACC_FULL", "check__glioma_with_label__std": "1"})
        writer.writerow({"AccessionNumber": "ACC_MISSING", "check__glioma_with_label__std": "0"})
    return index_path, labels_path


class StudyCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.index_path, self.labels_path = _write_dataset(self.root)
        self.records = build_study_records(self.index_path, self.labels_path, require_all_modalities=False)
        self.assertEqual([r.accession for r in self.records], ["ACC_FULL", "ACC_MISSING"])

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_mixed_native_shapes_stack_to_target_shape(self) -> None:
        dataset = StudyNiftiDataset(self.records, target_shape=SHAPE)
        for index in range(len(dataset)):
            self.assertEqual(tuple(dataset[index]["image"].shape), (len(MODALITIES), *SHAPE))

    def test_missing_modality_is_a_zero_channel(self) -> None:
        dataset = StudyNiftiDataset(self.records, target_shape=SHAPE)
        item = dataset[[r.accession for r in self.records].index("ACC_MISSING")]
        self.assertEqual(item["modality_present"].tolist(), [1.0, 1.0, 1.0, 0.0])
        self.assertTrue(torch.all(item["image"][MODALITIES.index("flair")] == 0))

    def test_batches_collate_when_the_first_study_misses_a_modality(self) -> None:
        # Regression: a None or str/float mix in the item dict makes
        # default_collate raise as soon as it lands in batch element 0.
        ordered = sorted(self.records, key=lambda r: r.accession != "ACC_MISSING")
        dataset = StudyNiftiDataset(ordered, target_shape=SHAPE)
        batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
        self.assertEqual(tuple(batch["image"].shape), (2, len(MODALITIES), *SHAPE))
        self.assertEqual(batch["label"].float().tolist(), [0.0, 1.0])

    def test_unlabelled_study_collates_as_nan(self) -> None:
        records = [self.records[0], self.records[1].__class__(**{**self.records[1].__dict__, "label": None})]
        dataset = StudyNiftiDataset(records, target_shape=SHAPE, require_label=False)
        batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
        self.assertEqual(batch["has_label"].tolist(), [1.0, 0.0])
        self.assertTrue(torch.isnan(batch["label"][1]))

    def test_float32_cache_round_trips_exactly(self) -> None:
        cache_dir = self.root / "cache32"
        plain = StudyNiftiDataset(self.records, target_shape=SHAPE)
        cached = StudyNiftiDataset(self.records, target_shape=SHAPE, cache_dir=cache_dir, cache_dtype="float32")
        for index in range(len(plain)):
            torch.testing.assert_close(cached[index]["image"], plain[index]["image"], rtol=0, atol=0)

    def test_float16_cache_matches_within_half_precision(self) -> None:
        cache_dir = self.root / "cache16"
        plain = StudyNiftiDataset(self.records, target_shape=SHAPE)
        cached = StudyNiftiDataset(self.records, target_shape=SHAPE, cache_dir=cache_dir, cache_dtype="float16")
        for index in range(len(plain)):
            torch.testing.assert_close(cached[index]["image"], plain[index]["image"], rtol=0, atol=1e-3)

    def test_second_read_does_not_touch_the_nifti_files(self) -> None:
        cache_dir = self.root / "cache_hit"
        dataset = StudyNiftiDataset(self.records, target_shape=SHAPE, cache_dir=cache_dir, cache_dtype="float32")
        first = dataset[0]["image"].clone()

        # Move the source volumes away: a cache hit must not need them.
        moved = self.root / "data_moved"
        (self.root / "data").rename(moved)
        try:
            second = dataset[0]["image"]
            torch.testing.assert_close(second, first, rtol=0, atol=0)
        finally:
            moved.rename(self.root / "data")

    def test_cache_key_changes_with_shape_and_paths(self) -> None:
        paths = tuple(self.records[0].modalities.get(m) for m in MODALITIES)
        base = cache_path(paths, SHAPE, "/tmp/cache", accession="ACC_FULL")
        self.assertNotEqual(base, cache_path(paths, (32, 32, 16), "/tmp/cache", accession="ACC_FULL"))
        self.assertNotEqual(base, cache_path((*paths[:3], None), SHAPE, "/tmp/cache", accession="ACC_FULL"))

    def test_load_study_image_without_cache_dir_writes_nothing(self) -> None:
        paths = tuple(self.records[0].modalities.get(m) for m in MODALITIES)
        image, present = load_study_image(paths, SHAPE)
        self.assertEqual(image.shape, (len(MODALITIES), *SHAPE))
        self.assertEqual(image.dtype, np.float32)
        self.assertEqual(present.tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_study_without_any_modality_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            load_study_image((None, None, None, None), SHAPE)


class NormalizeVolumeTest(unittest.TestCase):
    def test_rescales_to_unit_range(self) -> None:
        values = np.linspace(-500, 1500, 4000, dtype=np.float32).reshape(20, 20, 10)
        result = normalize_volume(values)
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)
        # Monotone input stays monotone after clipping and rescaling.
        flat = result.reshape(-1)
        self.assertTrue(np.all(np.diff(flat) >= -1e-6))

    def test_matches_the_full_volume_percentile_reference(self) -> None:
        rng = np.random.default_rng(1)
        values = (rng.random((96, 96, 40)) * 1000).astype(np.float32)
        finite = np.isfinite(values)
        lo, hi = np.percentile(values[finite], (1.0, 99.0))
        reference = ((np.clip(values, lo, hi) - float(lo)) / max(float(hi - lo), 1e-6)).astype(np.float32)
        np.testing.assert_allclose(normalize_volume(values), reference, atol=2e-3)

    def test_degenerate_volumes_return_zeros(self) -> None:
        for values in (
            np.zeros((8, 8, 4), dtype=np.float32),
            np.full((8, 8, 4), 7.5, dtype=np.float32),
            np.full((8, 8, 4), np.nan, dtype=np.float32),
        ):
            result = normalize_volume(values)
            self.assertEqual(result.shape, values.shape)
            self.assertTrue(np.all(result == 0))

    def test_does_not_mutate_the_input(self) -> None:
        values = (np.arange(64, dtype=np.float32) * 10).reshape(4, 4, 4)
        before = values.copy()
        normalize_volume(values)
        np.testing.assert_array_equal(values, before)


if __name__ == "__main__":
    unittest.main()
