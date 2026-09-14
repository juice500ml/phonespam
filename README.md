# phonespam

> [**Phone Segmentation and Recognition through Phonological Activation Mapping**](https://arxiv.org/abs/2607.09020)
> Shikhar Bharadwaj*, Kwanghee Choi*, Stephen McIntosh*, Chin-Jou Li, Eunjung Yeo,
> Daisuke Saito, Nobuaki Minematsu, Shinji Watanabe, Jian Zhu, David Harwath, David R. Mortensen.
> Accepted at **SLT 2026**. [arXiv:2607.09020](https://arxiv.org/abs/2607.09020)

## Install

Inference only — load a pretrained model and run it on audio:

```bash
pip install phonespam
```

Training extras — adds dataset preparation (TIMIT, VoxAngeles) and SSL
feature dumping. Training itself only requires running SSL inference, dumping
features, and fitting on them (similar in spirit to k-means clustering on SSL
features):

```bash
pip install "phonespam[train]"
```

Dataset preparation (`phonespam.training.prepare_datasets`) and metric
evaluation (`scripts/evaluate_metrics.py`) additionally need
[phone-metrics](https://github.com/stephenmac7/phone-metrics), which is not
published on PyPI and so cannot be named as an extra:

```bash
pip install "phone-metrics @ git+https://github.com/stephenmac7/phone-metrics.git"
```

With `uv`, `uv sync --group train` pulls it in automatically.

## Usage

```python
from phonespam import PhoneModel

model = PhoneModel.from_pretrained("juice500/wavlm-24-phonemodel")

for seg in model.transcribe("utt.wav"):
    print(f"{seg.start:.2f}-{seg.end:.2f}  {seg.label}")
```

`transcribe` runs the whole pipeline — load audio, encode, project, segment,
label — and returns contiguous `Segment(start, end, label)` tuples with times
in seconds. Constrain the output to one language's phone inventory with
`vocab=`, or override segmenter hyperparameters for a single call:

```python
from phonespam import vocab_for_language

segs = model.transcribe("utt.wav", vocab=vocab_for_language("deu"))
segs = model.transcribe("utt.wav", hparam_overrides={"combined_prominence": 0.005})
```

If you only need boundaries, `model.boundaries("utt.wav")` returns them as
times in seconds.

## The activation map

`model.spam(...)` returns the SPAM itself — the intermediate the segmenter and
recognizer both consume, one row per encoder frame and one column per
phonological feature:

```python
spam = model.spam("utt.wav")  # normalized to [0, 1]
spam = model.spam("utt.wav", act="none")  # raw projections, unbounded

spam.shape  # (n_frames, n_features)
spam.featnames  # ("silence+", "cons+", "cons-", ...)
spam.times  # frame start times in seconds
```

It carries its feature names and frame times, so picking channels out for a
plot doesn't mean looking up column positions:

```python
heat = spam.select(["silence+", "hi+", "hi-", "strid+", "strid-"])
plt.imshow(np.asarray(heat).T, aspect="auto", vmin=0, vmax=1)

spam.to_frame()  # pandas DataFrame indexed by time
```

`np.asarray(spam)` gives the bare matrix back, and a `Spam` can be passed
straight to `Recognizer.recognize`. If you want both the map and a
transcription of the same audio, extract features once and reuse them:

```python
feats = model.extract_features(model.load_audio("utt.wav"))
spam = model.spam_from_features(feats)
```

### Driving the stages directly

The stages stay available for research use. If you wire them yourself, two
conventions are easy to miss, and both fail *silently* — the output keeps the
right shape and only the values are wrong:

- `Segmenter.segment` returns **frame indices**, while `Recognizer.recognize`
  takes **times in seconds**. Convert with `model.encoder.frame_to_time(...)`.
- `Recognizer.recognize` scores a **sigmoid** posteriogram, but
  `project` defaults to `act="none"`.

`recognize` warns when it detects either mistake, but `transcribe` is the way
to avoid them.

```python
wav = model.load_audio("utt.wav")
feats = model.extract_features(wav)

# Per-frame posteriogram, shape (T, n_features).
# Columns are named by model.posteriogram.featnames.
post = model.posteriogram.project(feats, view="ipa", act="sigmoid")

frames = model.segmenter.segment(feats, wav)  # frame indices
times = model.encoder.frame_to_time(frames)  # seconds
labels = model.recognizer.recognize(post, times, sr=model.sr, frame_shift=model.frame_shift)
```

The encoder is lazy-loaded on first use, so `from_pretrained` is cheap if you
only want to inspect or re-save the artifact.

## Phone inventories (PHOIBLE)

`vocab_for_language` resolves a language name, ISO 639-3 code, or Glottocode
to that language's phone inventory, for use as `transcribe(..., vocab=...)`:

```python
from phonespam import vocab_for_language, vocab_for_inventory

vocab = vocab_for_language("eng")  # or "deu", "stan1293", "German", ...
segs = model.transcribe("utt.wav", vocab=vocab)

# When a language maps to several PHOIBLE inventories, pick one explicitly:
vocab = vocab_for_inventory(2252)  # English (RP)
```

The inventory table is a ~26 MB CSV and is **not** bundled in the wheel. It is
downloaded once from a pinned, checksum-verified upstream commit and cached
under `~/.cache/phonespam/` (or `$XDG_CACHE_HOME/phonespam`). Two environment
variables control this:

| Variable | Effect |
| --- | --- |
| `PHONESPAM_CACHE_DIR` | Relocate the download cache. |
| `PHONESPAM_PHOIBLE_CSV` | Use this `phoible.csv` verbatim; nothing is downloaded. Use it on offline machines or to pin a different snapshot. |

## Licensing

phonespam itself is MIT (see `LICENSE`). The PHOIBLE inventory data it downloads
at runtime is **not** — it is CC BY-SA 3.0 and must be attributed separately:

> Moran, Steven & McCloy, Daniel (eds.) 2019. PHOIBLE 2.0. Jena: Max Planck
> Institute for the Science of Human History. Available online at
> <http://phoible.org>. DOI: 10.5281/zenodo.2626687

## Citation

```bibtex
@inproceedings{bharadwaj2026spam,
  title     = {Phone Segmentation and Recognition through Phonological Activation Mapping},
  author    = {Bharadwaj, Shikhar and Choi, Kwanghee and McIntosh, Stephen and
               Li, Chin-Jou and Yeo, Eunjung and Saito, Daisuke and
               Minematsu, Nobuaki and Watanabe, Shinji and Zhu, Jian and
               Harwath, David and Mortensen, David R.},
  booktitle = {IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026},
  eprint    = {2607.09020},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url       = {https://arxiv.org/abs/2607.09020}
}
```
