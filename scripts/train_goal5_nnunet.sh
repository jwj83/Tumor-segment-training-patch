#!/usr/bin/env bash
# Goal5: two nnUNet models (core / abnormal), in three resumable stages.
#
#   bash scripts/train_goal5_nnunet.sh export
#   bash scripts/train_goal5_nnunet.sh preprocess
#   bash scripts/train_goal5_nnunet.sh train
#   bash scripts/train_goal5_nnunet.sh all
#
# Each stage is safe to re-run.  Export skips cases whose files already exist,
# preprocess skips a dataset that already has preprocessed data, and training
# always passes --c so it resumes from the latest checkpoint.  The previous
# single pipeline redid the export whenever training died 10 hours in.
set -euo pipefail

STAGE="${1:-all}"

INDEX="${INDEX:-./file_check/file_index.csv}"
LABELS="${LABELS:-./readout/series_merged.csv}"
NNUNET_RAW="${nnUNet_raw:?Set nnUNet_raw to a writable nnUNet raw directory}"
NNUNET_PREPROCESSED="${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
NNUNET_DATASET_CORE="${NNUNET_DATASET_CORE:-501}"
NNUNET_DATASET_ABNORMAL="${NNUNET_DATASET_ABNORMAL:-502}"
NNUNET_CONFIG="${NNUNET_CONFIG:-3d_fullres}"
NNUNET_PLANS="${NNUNET_PLANS:-nnUNetPlans}"
NNUNET_TRAINER="${NNUNET_TRAINER:-nnUNetTrainer_100epochs}"
NNUNET_FOLD="${NNUNET_FOLD:-0}"
# Each preprocessing worker holds a whole four-channel case in RAM; the nnUNet
# default of 4 is what ran the box out of memory.
NNUNET_NP="${NNUNET_NP:-2}"
NNUNET_NPFP="${NNUNET_NPFP:-2}"
MIN_SPACING="${MIN_SPACING:-1.5 1.5 1.5}"
# Two trainings share the GPU, so halve nnUNet's augmentation workers.
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-6}"

CORE_DIR="$NNUNET_RAW/Dataset${NNUNET_DATASET_CORE}_GliomaCore"
ABNORMAL_DIR="$NNUNET_RAW/Dataset${NNUNET_DATASET_ABNORMAL}_GliomaAbnormal"

run_export() {
  test -f "$INDEX"
  test -f "$LABELS"
  python -m scripts.export_goal5_nnunet \
    --index "$INDEX" --labels-csv "$LABELS" \
    --out-dir "$CORE_DIR" --target core \
    --dataset-id "$NNUNET_DATASET_CORE" --dataset-name GliomaCore \
    --min-spacing $MIN_SPACING
  python -m scripts.export_goal5_nnunet \
    --index "$INDEX" --labels-csv "$LABELS" \
    --out-dir "$ABNORMAL_DIR" --target abnormal \
    --dataset-id "$NNUNET_DATASET_ABNORMAL" --dataset-name GliomaAbnormal \
    --min-spacing $MIN_SPACING
  du -sh "$CORE_DIR" "$ABNORMAL_DIR"
  df -h "$NNUNET_RAW" | tail -1
}

preprocess_one() {
  local dataset_id="$1" name="$2"
  local target_dir="$NNUNET_PREPROCESSED/Dataset$(printf '%03d' "$dataset_id")_${name}/${NNUNET_PLANS}_${NNUNET_CONFIG}"
  if [ -d "$target_dir" ] && [ -n "$(ls -A "$target_dir" 2>/dev/null)" ]; then
    printf '%s\n' "preprocessed data already present for $dataset_id, skipping: $target_dir"
    return 0
  fi
  # -c limits this to the one configuration that gets trained.  The default is
  # ['2d', '3d_fullres', '3d_lowres'], so leaving it out preprocesses the whole
  # dataset three times and stores three copies, two of which are never read.
  nnUNetv2_plan_and_preprocess -d "$dataset_id" \
    -c "$NNUNET_CONFIG" -np "$NNUNET_NP" -npfp "$NNUNET_NPFP" \
    --verify_dataset_integrity
}

run_preprocess() {
  command -v nnUNetv2_plan_and_preprocess >/dev/null
  preprocess_one "$NNUNET_DATASET_CORE" GliomaCore
  preprocess_one "$NNUNET_DATASET_ABNORMAL" GliomaAbnormal
  du -sh "$NNUNET_PREPROCESSED" || true
}

run_train() {
  command -v nnUNetv2_train >/dev/null
  # Both models share one GPU.  --c resumes from checkpoint_latest.pth and just
  # warns when there is nothing to resume, so it is safe on a first run.
  nnUNetv2_train "$NNUNET_DATASET_CORE" "$NNUNET_CONFIG" "$NNUNET_FOLD" \
    -tr "$NNUNET_TRAINER" -p "$NNUNET_PLANS" --c &
  local pid_core=$!
  nnUNetv2_train "$NNUNET_DATASET_ABNORMAL" "$NNUNET_CONFIG" "$NNUNET_FOLD" \
    -tr "$NNUNET_TRAINER" -p "$NNUNET_PLANS" --c &
  local pid_abnormal=$!

  local status=0
  wait "$pid_core" || status=$?
  if [ "$status" -ne 0 ]; then
    printf '%s\n' "core training exited with $status" >&2
  fi
  local status_abnormal=0
  wait "$pid_abnormal" || status_abnormal=$?
  if [ "$status_abnormal" -ne 0 ]; then
    printf '%s\n' "abnormal training exited with $status_abnormal" >&2
  fi
  if [ "$status" -ne 0 ] || [ "$status_abnormal" -ne 0 ]; then
    return 1
  fi
  printf '%s\n' "Goal5 core and abnormal training completed." \
    "trainer: $NNUNET_TRAINER" \
    "results root: ${nnUNet_results:-unset}"
}

case "$STAGE" in
  export) run_export ;;
  preprocess) run_preprocess ;;
  train) run_train ;;
  all) run_export; run_preprocess; run_train ;;
  *) printf '%s\n' "usage: $0 [export|preprocess|train|all]" >&2; exit 2 ;;
esac
