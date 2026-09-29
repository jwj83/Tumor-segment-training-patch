#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/2026aicompetition/datasets/training/annotation}"
INDEX="${INDEX:-./file_check/file_index.csv}"
LABELS="${LABELS:-./readout/series_merged.csv}"
NNUNET_RAW="${nnUNet_raw:?Set nnUNet_raw to a writable nnUNet raw directory}"
NNUNET_DATASET_CORE="${NNUNET_DATASET_CORE:-501}"
NNUNET_DATASET_ABNORMAL="${NNUNET_DATASET_ABNORMAL:-502}"
NNUNET_CONFIG="${NNUNET_CONFIG:-3d_fullres}"
NNUNET_PLANS="${NNUNET_PLANS:-nnUNetPlans}"
NNUNET_TRAINER="${NNUNET_TRAINER:-nnUNetTrainer}"
NNUNET_FOLD="${NNUNET_FOLD:-0}"

command -v nnUNetv2_plan_and_preprocess >/dev/null
command -v nnUNetv2_train >/dev/null
test -f "$INDEX"
test -f "$LABELS"

CORE_DIR="$NNUNET_RAW/Dataset${NNUNET_DATASET_CORE}_GliomaCore"
ABNORMAL_DIR="$NNUNET_RAW/Dataset${NNUNET_DATASET_ABNORMAL}_GliomaAbnormal"
mkdir -p "$NNUNET_RAW"

python -m scripts.export_goal5_nnunet \
  --index "$INDEX" --labels-csv "$LABELS" \
  --out-dir "$CORE_DIR" --target core \
  --dataset-id "$NNUNET_DATASET_CORE" --dataset-name GliomaCore

python -m scripts.export_goal5_nnunet \
  --index "$INDEX" --labels-csv "$LABELS" \
  --out-dir "$ABNORMAL_DIR" --target abnormal \
  --dataset-id "$NNUNET_DATASET_ABNORMAL" --dataset-name GliomaAbnormal

nnUNetv2_plan_and_preprocess -d "$NNUNET_DATASET_CORE" --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d "$NNUNET_DATASET_ABNORMAL" --verify_dataset_integrity

nnUNetv2_train "$NNUNET_DATASET_CORE" "$NNUNET_CONFIG" "$NNUNET_FOLD" -tr "$NNUNET_TRAINER" -p "$NNUNET_PLANS" --npz
nnUNetv2_train "$NNUNET_DATASET_ABNORMAL" "$NNUNET_CONFIG" "$NNUNET_FOLD" -tr "$NNUNET_TRAINER" -p "$NNUNET_PLANS" --npz

printf '%s\n' "Goal5 core and abnormal nnUNet training completed." \
  "core dataset: $NNUNET_DATASET_CORE" \
  "abnormal dataset: $NNUNET_DATASET_ABNORMAL" \
  "results root: ${nnUNet_results:-unset}"
