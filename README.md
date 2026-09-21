# SPAM
- SPAM means Self-supervised speech model-based Phonological Activation Mapping.
- Paper accepted to SLT 2026: [Phone Segmentation and Recognition through Phonological Activation Mapping](https://arxiv.org/abs/2607.09020) (Shikhar Bharadwaj∗, Kwanghee Choi∗, Stephen McIntosh∗, Chin-Jou Li, Eunjung Yeo,
Daisuke Saito, Nobuaki Minematsu, Shinji Watanabe, Jian Zhu, David Harwath and David R. Mortensen)
- Phonological vector first came from [[b]=[d]-[t]+[p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](https://aclanthology.org/2026.findings-acl.537/), and SPAM first came from [Self-Supervised Speech Models Encode Phonetic Context via Position-dependent Orthogonal Subspaces](https://arxiv.org/abs/2603.12642) as a visualization tool.

## Pretrained models
- Best model so far: [`juice500/wavlm-24-phonemodel`](https://huggingface.co/juice500/wavlm-24-phonemodel) (WavLM-large, layer 24, fit on the TIMIT training set)
- Different models and layers also available: Swap above with `model = wavlm/hubert/w2v2/xls-r`, `layer = 3/6/.../24`
- Efficiency ablation (Figure 3 in the paper): `juice500/wavlm-24-efficiency-{EFF}-phonemodel` where `EFF=1_2/1_4/.../1_1024` (using 1/2, 1/4, ..., 1/1024 of TIMIT training set)
- Segmentation ablation (Table 3 in the paper): `juice500/wavlm-24-ablation-oneframedelta-phonemodel`, `juice500/wavlm-24-ablation-stackframedelta-phonemodel`, `juice500/wavlm-24-ablation-stackframebwddelta-phonemodel`

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
uv venv && uv pip install -e ".[train]"
# or: python -m venv .venv && source .venv/bin/activate && pip install -e ".[train]"
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
that inventory.

### The activation map

`model.spam(...)` returns the SPAM itself — the intermediate both the segmenter
and the recognizer consume, one row per encoder frame and one column per
phonological feature.

```python
spam = model.spam("utt.wav")  # normalized to [0, 1]
spam.shape  # (n_frames, n_features)
spam.featnames  # ("silence+", "syl+", "son+", ...)
spam.times  # frame start times in seconds

heat = spam.select(["silence+", "son+", "cons+", "strid+"])
plt.imshow(np.asarray(heat).T, aspect="auto", vmin=0, vmax=1)

spam.to_frame()  # pandas DataFrame indexed by time
```

`np.asarray(spam)` gives the bare matrix, and a `Spam` can be passed straight to `model.recognize`. Pass `act="none"` for the raw, unbounded projections instead of the sigmoid-normalized ones.

### Segmentation only

```python
model.boundary_times("utt.wav")  # array of boundary times in seconds
```

`model.segment(features, waveform)` is the lower-level call: it takes features
you already have and returns **frame indices** rather than seconds.

### Driving the stages yourself
```python
wav = model.load_audio("utt.wav")  # mono float32, resampled to the model's rate
feats = model.extract_features(wav)  # (T, D) S3M features — one encoder pass
spam = model.spam_from_features(feats)  # reuses feats, no second pass

boundaries = model.segment(feats, wav)  # frame indices
times = model.encoder.frame_to_time(boundaries)  # seconds
phones = model.recognize(spam, boundaries, vocab=vocab_for_language("eng"))

model.sr, model.frame_shift  # 16000, 320 — one frame is frame_shift/sr seconds
```

### Saving and sharing a model

```python
model.to("cuda")
model.save_pretrained("exp/my-phonemodel")  # writes model.pt
model.push_to_hub("me/my-phonemodel", private=True)  # needs `huggingface-cli login`
```

### Constraining the phone inventory

`vocab_for_language` resolves a language name, ISO 639-3 code, or Glottocode to that language's PHOIBLE inventory:

```python
from phonespam import vocab_for_inventory, vocab_for_language

vocab = vocab_for_language("eng")  # or "deu", "stan1293", "German", ...
segs = model.transcribe("utt.wav", vocab=vocab)

# Some languages map to several PHOIBLE inventories; a warning names them all
# and one is chosen. Pick explicitly to silence it:
vocab = vocab_for_inventory(2175)  # Western/Mid-Western US English

# By default these are the inventory's *allophones*. Not every PHOIBLE source
# transcribes them; for those, ask for the phoneme inventory instead:
vocab = vocab_for_inventory(2252, phoneme=True)  # English (RP)
```

The inventory table (~26 MB) is downloaded on first use and cached under
`~/.cache/phonespam/`. Set `PHONESPAM_CACHE_DIR` to relocate that cache, or
`PHONESPAM_PHOIBLE_CSV` to point at a copy you already have and skip the
download entirely.

### Tuning the segmenter

`hparam_overrides` merges onto the model's stored hyperparameters for a single
call. `Segmenter.default_hparams()` lists them all:

```python
segs = model.transcribe("utt.wav", hparam_overrides={"combined_prominence": 0.005})
segs = model.transcribe("utt.wav", snap_silence=False)
```

For a sweep, copy the segmenter instead — the copy shares the posteriogram
weights, so nothing is refit:

```python
wav = model.load_audio("utt.wav")
feats = model.extract_features(wav)

tuned = model.segmenter.with_hparams({"distance": "l2"})
boundaries = tuned.segment(feats, wav)  # frame indices
```


## Training and evaluation

The steps below train the released model. [`scripts/`](scripts/) holds the
rest of the paper reproduction — ablations, error analysis, and the figure
notebooks — and is not part of the installed package; see
[`scripts/README.md`](scripts/README.md).

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

The code is MIT-licensed.

The phone inventory table it downloads at runtime (for `vocab_for_language`) is
[PHOIBLE 2.0](https://phoible.org) data, licensed separately under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/):

> Moran, Steven & McCloy, Daniel (eds.) 2019. PHOIBLE 2.0. Jena: Max Planck
> Institute for the Science of Human History. DOI: 10.5281/zenodo.2626687
