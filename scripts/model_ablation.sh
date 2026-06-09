#!/usr/bin/env bash
# Train PhoneModels across {wavlm, hubert, w2v2, xls-r} × {0,3,...,24} layers
# and push each to juice500/<short>-<layer>-phonemodel (private).
#
# Per run:
#   1) training/extract_features.py  → ${WORK_DIR}/<short>-<layer>.feats.pkl
#   2) training/train.py              → ${WORK_DIR}/<short>-<layer>-phonemodel/model.pt
#      with --push_to_hub --hub_private → juice500/<short>-<layer>-phonemodel
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

MODELS=(
  microsoft/wavlm-large
  facebook/hubert-large-ll60k
  facebook/wav2vec2-large-lv60
  facebook/wav2vec2-xls-r-300m
)
SHORTS=(wavlm hubert w2v2 xls-r)
LAYERS=(0 3 6 9 12 15 18 21 24)

train_one() {
  local model="$1" short="$2" layer="$3"
  local tag="${short}-${layer}-phonemodel"
  local feats_pkl="${WORK_DIR}/${short}-${layer}.feats.pkl"
  local out_dir="${WORK_DIR}/${tag}"
  local repo_id="juice500/${tag}"

  echo
  echo "=== ${repo_id} (model=${model}, layer=${layer}) ==="

  if [[ ! -f "$feats_pkl" ]]; then
    python -m phonological_posteriogram.training.extract_features \
      --model "$model" \
      --dataset_csv "$DATASET_CSV" \
      --split train \
      --layer_index "$layer" \
      --pool center \
      --sr 16000 \
      --device "$DEVICE" \
      --output_path "$feats_pkl"
  fi

  python -m phonological_posteriogram.training.train \
    --features_pkl "$feats_pkl" \
    --output_dir "$out_dir" \
    --push_to_hub "$repo_id" \
    --hub_private
}

for i in "${!MODELS[@]}"; do
  for layer in "${LAYERS[@]}"; do
    train_one "${MODELS[$i]}" "${SHORTS[$i]}" "$layer"
  done
done

echo
echo "model_ablation: all $(( ${#MODELS[@]} * ${#LAYERS[@]} )) runs completed."
