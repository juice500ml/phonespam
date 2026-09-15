#!/usr/bin/env bash
# Sample-efficiency ablations on wavlm-large, layer 24: fit on a fraction of
# training utterances (audio_paths). Writes each to
# ${WORK_DIR}/wavlm-24-efficiency-<1_N>-phonemodel/model.pt.
#
# Variants: 1_2, 1_4, ..., 1_1024 (1/N of the TIMIT training utterances).
#
# Reuses ${WORK_DIR}/wavlm-24.feats.pkl from model_ablation.sh if present.
#
# Requirements:
#   - DATASET_CSV: timit-raw.csv from training/prepare_datasets.py.
#   - GPU recommended (DEVICE=cuda:0).
#   - Optional: HF_ORG=<org> to also push each model (private) to <org>/<run
#     name>; needs `huggingface-cli login`.

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
  python -m phonespam.training.extract_features \
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
  local push_args=()
  if [[ -n "${HF_ORG:-}" ]]; then
    push_args=(--push_to_hub "${HF_ORG}/${tag}" --hub_private)
  fi

  echo
  echo "=== ${tag}  (audio_fraction=${frac}, seed=${SEED}) ==="

  python -m phonespam.training.train \
    --features_pkl "$FEATS_PKL" \
    --output_dir "$out_dir" \
    ${push_args[@]+"${push_args[@]}"} \
    --audio_fraction "$frac" \
    --seed "$SEED"
}

train_fraction 1_2  0.5
train_fraction 1_4  0.25
train_fraction 1_8  0.125
train_fraction 1_16 0.0625
train_fraction 1_32 0.03125
train_fraction 1_64 0.015625
train_fraction 1_128 0.0078125
train_fraction 1_256 0.00390625
train_fraction 1_512 0.001953125
train_fraction 1_1024 0.0009765625

echo
echo "efficiency_ablation: 10 runs completed."
