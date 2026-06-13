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
import librosa
from phonological_posteriogram import PhonologicalPosteriogram

model = PhonologicalPosteriogram.from_pretrained("user/phonpost-wavlm-large")
y, _ = librosa.load("utt.wav", sr=model.net_spec["sr"], mono=True)

# Phone boundary times (seconds).
boundaries = model.segment_seconds(y)

# Per-frame phonological posteriogram (T, n_features).
post = model.posteriogram(y)
```

The encoder is lazy-loaded on first call, so `from_pretrained` is cheap if
you just want to inspect or save the artifact.

## Versioning

Git tags are the single source of truth. Builds get their version via
`setuptools-scm` at build time; `python-semantic-release` creates tags from
conventional commits on push to `main`. Releases are then published to PyPI
automatically by `.github/workflows/publish.yml`.
