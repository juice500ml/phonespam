#!/usr/bin/env bash
# Sample-efficiency ablations on wavlm-large, layer 24: fit on a fraction of
# training utterances (audio_paths). Pushes each to
# juice500/wavlm-24-efficiency-<1_N>-phonemodel (private).
#
# Variants:
#   - 1_2  (50%)   - 1_4  (25%)   - 1_8  (12.5%)
#   - 1_16 (6.25%) - 1_32 (3.125%)
#
# Reuses ${WORK_DIR}/wavlm-24.feats.pkl from model_ablation.sh if present.
#
# Requirements:
#   - DATASET_CSV: per-phone CSV from training/prepare_datasets.py.
#   - `huggingface-cli login` for juice500.
#   - GPU recommended (DEVICE=cuda:0).

set -euo pipefail

: "${DATASET_CSV:?set DATASET_CSV=path/to/<dataset>.csv}"
DEVICE="${DEVICE:-cuda:0}"
WORK_DIR="${WORK_DIR:-exp/phonemodel_grid}"
SEED="${SEED:-42}"
mkdir -p "$WORK_DIR"

MODEL=microsoft/wavlm-large
SHORT=wavlm
LAYER=24
FEATS_PKL="${WORK_DIR}/${SHORT}-${LAYER}.feats.pkl"

if [[ ! -f "$FEATS_PKL" ]]; then
  python -m phonological_posteriogram.training.extract_features \
    --model "$MODEL" \
    --dataset_csv "$DATASET_CSV" \
    --split train \
    --layer_index "$LAYER" \
    --pool center \
    --sr 16000 \
    --device "$DEVICE" \
    --output_path "$FEATS_PKL"
fi

train_fraction() {
  local label="$1" frac="$2"
  local tag="${SHORT}-${LAYER}-efficiency-${label}-phonemodel"
  local out_dir="${WORK_DIR}/${tag}"
  local repo_id="juice500/${tag}"

  echo
  echo "=== ${repo_id}  (audio_fraction=${frac}, seed=${SEED}) ==="

  python -m phonological_posteriogram.training.train \
    --features_pkl "$FEATS_PKL" \
    --output_dir "$out_dir" \
    --push_to_hub "$repo_id" \
    --hub_private \
    --audio_fraction "$frac" \
    --seed "$SEED"
}

train_fraction 1_2  0.5
train_fraction 1_4  0.25
train_fraction 1_8  0.125
train_fraction 1_16 0.0625
train_fraction 1_32 0.03125

echo
echo "efficiency_ablation: 5 runs completed."
