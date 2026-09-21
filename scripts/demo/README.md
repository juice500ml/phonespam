---
title: SPAM Phone Segmentation and Recognition
emoji: 🗣️
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# SPAM demo

Interactive demo for [Phone Segmentation and Recognition through Phonological
Activation Mapping](https://arxiv.org/abs/2607.09020) (SLT 2026), running
[`juice500/wavlm-24-phonemodel`](https://huggingface.co/juice500/wavlm-24-phonemodel).

Give it the example utterance or record your own, and it draws five panels on a
single time axis, with the predicted boundaries shared across all of them:

1. **Spectrogram** — wideband, via [specplotter](https://github.com/juice500ml/specplotter).
2. **SPAM** — all 43 phonological feature activations, normalized to `[0, 1]`.
3. **Frame-wise labels** — the recognition head alone, one label per frame,
   with no segmentation.
4. **Combined boundary signal** — the fuzzy-AND of the selected segmentation
   signals, with the peaks that became boundaries marked.
5. **Phones** — both heads: one label per segment.

Controls:

- **Segmentation signals** — the released model's ensemble, grouped into the
  four families the paper ablates. Turning one off re-runs segmentation
  without it, so you can watch the boundary signal and the phones change.
- **Phone inventory** — restrict recognition to any of PHOIBLE's ~2700
  languages, listed as `Name (Glottocode)`.

## Running it

```bash
pip install -r requirements.txt
GRADIO_TEMP_DIR=$PWD/.gradio_tmp python app.py
```

This folder is a demo. It is not part of the `phonespam` package and nothing in the library imports from it — see [`../README.md`](../README.md).

## Deploying as a Space

The YAML header above is the Space configuration, so the folder can be pushed to a HuggingFace Space as-is:

```bash
git clone https://huggingface.co/spaces/<user>/<space> hf-space
cp -r scripts/demo/* hf-space/
cd hf-space && git add -A && git commit -m "SPAM demo" && git push
```

CPU is enough — inference is roughly real-time — but the Space needs disk for
the encoder and enough RAM to hold it.

## Example audio

`examples/LDC93S1.wav` is the TIMIT sample utterance LDC publishes as a free
[catalogue addendum](https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav):
*"She had your dark suit in greasy wash water all year."*
