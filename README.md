# phonological-posteriogram

Phone recognition and segmentation using phonological posteriograms over
self-supervised speech features.

## Install

Inference only — load a pretrained model and run it on audio:

```bash
pip install phonological-posteriogram
```

Training extras — adds dataset preparation (TIMIT, VoxAngeles), SSL feature
dumping, and the scikit-learn silence detector. Training itself only requires
running SSL inference, dumping features, and fitting on them (similar in
spirit to k-means clustering on SSL features):

```bash
pip install "phonological-posteriogram[train]"
```

## Usage

HuggingFace-style loading from the Hub or a local path:

```python
from phonological_posteriogram import PhoneModel

model = PhoneModel.from_pretrained("user/phonpost-wavlm-large")

# Load audio (resampled to the model's rate) and run the SSL encoder.
wav = model.load_audio("utt.wav")
feats = model.extract_features(wav)

# Per-frame phonological posteriogram / activation map, shape (T, n_features).
# Columns are named by model.posteriogram.featnames.
post = model.posteriogram.project(feats, view="ipa", act="sigmoid")

# Phone boundaries as frame indices; convert to seconds via the encoder.
boundaries = model.segmenter().segment(feats, wav)
boundary_times = model.encoder.frame_to_time(boundaries)
```

The encoder is lazy-loaded on first call, so `from_pretrained` is cheap if
you just want to inspect or save the artifact.

## Versioning

Git tags are the single source of truth. Builds get their version via
`setuptools-scm` at build time; `python-semantic-release` creates tags from
conventional commits on push to `main`. Releases are then published to PyPI
automatically by `.github/workflows/publish.yml`.
