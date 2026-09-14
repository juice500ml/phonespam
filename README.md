# SPAM

Code for **Phone Segmentation and Recognition through Phonological Activation
Mapping** (SLT 2026).

Pretrained model: [`juice500/wavlm-24-phonemodel`](https://huggingface.co/juice500/wavlm-24-phonemodel)
(WavLM-large, layer 24, fit on the TIMIT training set)

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/juice500ml/spam.git
cd spam
uv venv && uv pip install -e ".[train]"
# or: python -m venv .venv && source .venv/bin/activate && pip install -e ".[train]"
```

The `train` extra adds dataset preparation and evaluation (via
`phone-metrics`). For inference only, install `.` without extras.

## Usage

```python
from phonological_posteriogram import PhoneModel
from phonological_posteriogram.phoible import vocab_for_language

model = PhoneModel.from_pretrained("juice500/wavlm-24-phonemodel")  # device="cuda" for GPU

wav = model.load_audio("utt.wav")  # mono float32, resampled to the model's rate
feats = model.extract_features(wav)  # (T, D) S3M features

# Phonological activation map, shape (T, n_features).
# Columns are named by model.posteriogram.featnames.
post = model.posteriogram.project(feats, view="ipa", act="sigmoid")

# Segmentation: boundaries between phones, as frame indices.
boundaries = model.segment(feats, wav)
times = model.encoder.frame_to_time(boundaries)  # in seconds

# Recognition: one label per segment ("_" is silence). `vocab` is optional:
# without it any PanPhon phone can be predicted; with it, output is restricted
# to that inventory (here, English from PHOIBLE).
phones = model.recognize(post, boundaries, vocab=vocab_for_language("eng"))
```

## Training and evaluation

### Data

- **TIMIT** is distributed by the LDC ([LDC93S1](https://catalog.ldc.upenn.edu/LDC93S1)).
  Place the root of the official distribution at `data/TIMIT`.
- **VoxAngeles** (evaluation only):

  ```bash
  git clone --branch main --depth 1 https://github.com/pacscilab/voxangeles.git data/voxangeles
  (cd data/voxangeles/data/audited_aligned && for f in *.zip; do unzip -o "$f"; done)
  ```

### Train the released model

```bash
# 1. Per-phone CSVs. Training uses timit-raw.csv, which keeps stop closures.
python -m phonological_posteriogram.training.prepare_datasets \
    --dataset_type timit --dataset_path data/TIMIT --output_dir data/csv

# 2. Per-phone S3M features for the training split.
python -m phonological_posteriogram.training.extract_features \
    --model microsoft/wavlm-large --layer_index 24 --pool center --sr 16000 \
    --dataset_csv data/csv/timit-raw.csv --split train \
    --device cuda:0 --output_path exp/wavlm-24.feats.pkl

# 3. Fit the phonological vectors and save exp/wavlm-24-phonemodel/model.pt.
python -m phonological_posteriogram.training.train \
    --features_pkl exp/wavlm-24.feats.pkl --output_dir exp/wavlm-24-phonemodel
```

### Evaluate

```bash
python scripts/evaluate_metrics.py --model exp/wavlm-24-phonemodel \
    --timit_root data/TIMIT --vox_root data/voxangeles
```

`--model` also accepts a Hub id such as `juice500/wavlm-24-phonemodel`. For
TIMIT test and VoxAngeles, the script reports boundary R-value (20 ms, strict)
and PER/TER/PFER with both ground-truth ("oracle") and predicted ("pipeline")
segmentation. On VoxAngeles, oracle PFER is also split into phones seen and
unseen in TIMIT. Set `CACHE_DIR=.cache/eval` to cache per-utterance inference
across runs.

## Development

```bash
uv pip install -e ".[train,dev]"
pre-commit install
pytest
```

## Citation

```bibtex
@inproceedings{spam2026,
  title     = {Phone Segmentation and Recognition through Phonological Activation Mapping},
  author    = {Bharadwaj, Shikhar and Choi, Kwanghee and McIntosh, Stephen and Li, Chin-Jou and
               Yeo, Eunjung and Saito, Daisuke and Minematsu, Nobuaki and Watanabe, Shinji and
               Zhu, Jian and Harwath, David and Mortensen, David R.},
  booktitle = {Proc. IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026}
}
```

## License

The code is MIT-licensed. `phonological_posteriogram/data/phoible.csv` is
[PHOIBLE 2.0](https://phoible.org) data, licensed separately under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); see
`phonological_posteriogram/data/README.md`.
