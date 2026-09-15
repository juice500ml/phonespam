# SPAM

Code for **Phone Segmentation and Recognition through Phonological Activation
Mapping** (SLT 2026).

Pretrained model: [`juice500/wavlm-24-phonemodel`](https://huggingface.co/juice500/wavlm-24-phonemodel)
(WavLM-large, layer 24, fit on the TIMIT training set)

## Installation

Requires Python 3.10+.

For inference:

```bash
pip install phonespam
```

To train or evaluate, clone the repo and install the `train` extra:

```bash
git clone https://github.com/juice500ml/phonespam.git
cd phonespam
uv venv && uv pip install -e ".[train]" && uv sync --group train
# or: python -m venv .venv && source .venv/bin/activate && pip install -e ".[train]"
```

Dataset preparation and evaluation also need
[`phone-metrics`](https://github.com/stephenmac7/phone-metrics). It is not on
PyPI, and PyPI rejects direct URL requirements in published metadata, so it
cannot be named in the `train` extra. `uv sync --group train` installs it; with
plain pip:

```bash
pip install "phone-metrics @ git+https://github.com/stephenmac7/phone-metrics@v0.1.0"
```

## Usage

```python
from phonespam import PhoneModel, vocab_for_language

model = PhoneModel.from_pretrained("juice500/wavlm-24-phonemodel")  # device="cuda" for GPU

for seg in model.transcribe("utt.wav", vocab=vocab_for_language("eng")):
    print(f"{seg.start:.2f}-{seg.end:.2f}  {seg.label}")
```

`transcribe` runs the whole pipeline and returns contiguous
`Segment(start, end, label)` tuples with times in seconds. `vocab` is optional:
without it any PanPhon phone can be predicted; with it, output is restricted to
that inventory. To drive the stages yourself:

```python
from phonespam import PhoneModel, vocab_for_language

model = PhoneModel.from_pretrained("juice500/wavlm-24-phonemodel")

wav = model.load_audio("utt.wav")  # mono float32, resampled to the model's rate
feats = model.extract_features(wav)  # (T, D) S3M features

# Phonological activation map (SPAM), shape (T, n_features). Carries its own
# feature names and frame times: spam.featnames, spam.times,
# spam.select([...]), spam.to_frame(). np.asarray(spam) gives the raw matrix.
post = model.spam_from_features(feats)  # or model.spam("utt.wav")

# Segmentation: boundaries between phones, as frame indices.
boundaries = model.segment(feats, wav)
times = model.encoder.frame_to_time(boundaries)  # in seconds

# Recognition: one label per segment, len(boundaries) + 1 ("_" is silence).
# recognize() takes frame indices and converts them for you.
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
python -m phonespam.training.prepare_datasets \
    --dataset_type timit --dataset_path data/TIMIT --output_dir data/csv

# 2. Per-phone S3M features for the training split.
python -m phonespam.training.extract_features \
    --model microsoft/wavlm-large --layer_index 24 --pool center --sr 16000 \
    --dataset_csv data/csv/timit-raw.csv --split train \
    --device cuda:0 --output_path exp/wavlm-24.feats.pkl

# 3. Fit the phonological vectors and save exp/wavlm-24-phonemodel/model.pt.
python -m phonespam.training.train \
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

The code is MIT-licensed. The phone inventory table it downloads at runtime is
[PHOIBLE 2.0](https://phoible.org) data, licensed separately under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/):

> Moran, Steven & McCloy, Daniel (eds.) 2019. PHOIBLE 2.0. Jena: Max Planck
> Institute for the Science of Human History. DOI: 10.5281/zenodo.2626687

The table is a ~26 MB CSV and is **not** bundled in the wheel. It is downloaded
once from a pinned, checksum-verified upstream commit and cached under
`~/.cache/phonespam/` (or `$XDG_CACHE_HOME/phonespam`). Set
`PHONESPAM_CACHE_DIR` to relocate the cache, or `PHONESPAM_PHOIBLE_CSV` to use
a copy you already have (and skip the download entirely).
