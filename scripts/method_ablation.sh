#!/usr/bin/env bash
# Structural ablations of the boundary-signal stack on wavlm-large, layer 24.
# Pushes each to juice500/wavlm-24-ablation-<name>-phonemodel (private).
#
# Variants:
#   - oneframedelta        : single frame_delta(offset=1).
#   - stackframedelta      : 3 frame_deltas (no bwd_contrast, no mel_svf).
#   - stackframebwddelta   : default minus mel_svf (frame_deltas + bwd_contrasts).
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

train_ablation() {
  local name="$1" overrides="$2"
  local tag="${SHORT}-${LAYER}-ablation-${name}-phonemodel"
  local out_dir="${WORK_DIR}/${tag}"
  local repo_id="juice500/${tag}"

  echo
  echo "=== ${repo_id} ==="

  python -m phonespam.training.train \
    --features_pkl "$FEATS_PKL" \
    --output_dir "$out_dir" \
    --push_to_hub "$repo_id" \
    --hub_private \
    --hparams_overrides "$overrides"
}

# 1) Single frame_delta(offset=1).
train_ablation oneframedelta '
{"combined_signals":[
  {"name":"frame_delta","kwargs":{"offset":1},"shift":1}
]}'

# 2) Stack of frame_deltas (no bwd_contrast, no mel_svf).
train_ablation stackframedelta '
{"combined_signals":[
  {"name":"frame_delta","kwargs":{"offset":3},"shift":2},
  {"name":"frame_delta","kwargs":{"offset":2},"shift":1},
  {"name":"frame_delta","kwargs":{"offset":1},"shift":1}
]}'

# 3) Default signals minus mel_svf (frame_deltas + bwd_contrasts).
train_ablation stackframebwddelta '
{"combined_signals":[
  {"name":"frame_delta","kwargs":{"offset":3},"shift":2},
  {"name":"frame_delta","kwargs":{"offset":2},"shift":1},
  {"name":"frame_delta","kwargs":{"offset":1},"shift":1},
  {"name":"bwd_contrast","kwargs":{"lookbehind":2},"shift":-1},
  {"name":"bwd_contrast","kwargs":{"lookbehind":3},"shift":-1},
  {"name":"bwd_contrast","kwargs":{"lookbehind":1},"shift":0}
]}'

echo
echo "method_ablation: 3 runs completed."
