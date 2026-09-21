#!/usr/bin/env bash
# Reproduces the paper's main result: raw corpora -> per-phone CSVs -> S3M
# features -> a trained PhoneModel -> the reported metrics. The wavlm/24 run
# here is the released juice500/wavlm-24-phonemodel.
#
# Stages (each is skipped if its output already exists, so the script is
# resumable; pass FORCE=1 to redo everything):
#   1) training/prepare_datasets  -> ${DATA_DIR}/{timit-raw,timit-merged,voxangeles}.csv
#   2) training/extract_features  -> ${FEATS_DIR}/${TAG}.pkl
#   3) training/train             -> ${MODELS_DIR}/${TAG}/model.pt
#   4) evaluate_metrics.py        -> TIMIT + VoxAngeles segmentation/recognition
#
# Requirements:
#   - TIMIT_ROOT / VOX_ROOT pointing at the distributed corpora.
#   - pip install -e ".[train]"
#   - A GPU is strongly recommended for stages 2 and 4.
#
# Usage (runnable from anywhere; outputs land under the repo root by default):
#   TIMIT_ROOT=/path/to/TIMIT VOX_ROOT=/path/to/voxangeles \
#     scripts/reproduce_main_result.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

: "${TIMIT_ROOT:?set TIMIT_ROOT=/path/to/TIMIT}"
: "${VOX_ROOT:?set VOX_ROOT=/path/to/voxangeles}"

DEVICE="${DEVICE:-cuda:0}"
SSL_MODEL="${SSL_MODEL:-microsoft/wavlm-large}"
LAYER="${LAYER:--1}"          # -1 == last hidden state (layer 24 for wavlm-large)
TAG="${TAG:-wavlm-24-timit}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
FEATS_DIR="${FEATS_DIR:-${REPO_ROOT}/feats}"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models}"
# Memoizes per-utterance inference in stage 4 so metrics can be re-derived
# without re-running the encoder. Keyed by model + segmentation config.
export CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/exp/eval_cache}"

mkdir -p "$DATA_DIR" "$FEATS_DIR" "$MODELS_DIR" "$CACHE_DIR"

TRAIN_CSV="${DATA_DIR}/timit-raw.csv"
FEATS_PKL="${FEATS_DIR}/${TAG}.pkl"
MODEL_DIR="${MODELS_DIR}/${TAG}"

# Re-run a stage when FORCE=1, or when its output is missing.
stale() { [ "${FORCE:-0}" = "1" ] || [ ! -e "$1" ]; }

# --- 1. corpora -> per-phone CSVs ----------------------------------------- #
# TIMIT emits two CSVs: -raw keeps every interval with closures unlabeled (the
# training substrate for closure/release channels), -merged folds closures into
# the release (the evaluation ground truth).
if stale "$TRAIN_CSV"; then
  python3 -m phonespam.training.prepare_datasets \
    --dataset_path "$TIMIT_ROOT" --dataset_type timit --output_dir "$DATA_DIR"
fi
if stale "${DATA_DIR}/voxangeles.csv"; then
  python3 -m phonespam.training.prepare_datasets \
    --dataset_path "$VOX_ROOT" --dataset_type voxangeles --output_dir "$DATA_DIR"
fi

# --- 2. SSL features ------------------------------------------------------ #
if stale "$FEATS_PKL"; then
  python3 -m phonespam.training.extract_features \
    --model "$SSL_MODEL" \
    --dataset_csv "$TRAIN_CSV" \
    --split train \
    --layer_index "$LAYER" \
    --pool center \
    --output_path "$FEATS_PKL" \
    --device "$DEVICE"
fi

# --- 3. fit the phonological activation map ------------------------------- #
if stale "${MODEL_DIR}/model.pt"; then
  python3 -m phonespam.training.train \
    --features_pkl "$FEATS_PKL" \
    --output_dir "$MODEL_DIR" \
    --split train
fi

# --- 4. metrics ----------------------------------------------------------- #
# Reports segmentation (R-value, precision/recall) and recognition (PER/PFER)
# on TIMIT test and VoxAngeles.
python3 "${HERE}/evaluate_metrics.py" \
  --model "$MODEL_DIR" \
  --timit_root "$TIMIT_ROOT" \
  --vox_root "$VOX_ROOT" \
  --device "$DEVICE"
