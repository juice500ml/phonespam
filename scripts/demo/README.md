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

Give it the example utterance or record your own, and it draws four panels on a
single time axis, with the predicted boundaries shared across all of them:

1. **Spectrogram** — wideband, via [specplotter](https://github.com/juice500ml/specplotter).
2. **SPAM** — all 43 phonological feature activations, normalized to `[0, 1]`.
   This is the representation both heads read from.
3. **Combined boundary signal** — the fuzzy-AND of the selected segmentation
   signals, with the peaks that became boundaries marked.
4. **Phones** — the recognition head's label for each span.

Controls:

- **Segmentation signals** — the released model's ensemble, grouped into the
  four families the paper ablates. Turning one off re-runs segmentation
  without it, so you can watch the boundary signal and the phones change.
- **Phone inventory** — restrict recognition to one language's PHOIBLE
  inventory. The model is fit on TIMIT, so the non-English options show its
  cross-lingual behaviour.
- **Boundary sensitivity** — peak prominence required for a boundary. Lower
  finds more; the released model uses `0.001`.

## Running it

```bash
pip install -r requirements.txt
python app.py
```

First start downloads WavLM-large (~1.2 GB), the phone model, and the PHOIBLE
inventory table (~26 MB). They are fetched once at import, not per request, so
startup is slow and clicking **Run** is not.

This folder is a demo. It is not part of the `phonespam` package and nothing in
the library imports from it — see [`../README.md`](../README.md).

## Deploying as a Space

The YAML header above is the Space configuration, so the folder can be pushed
to a HuggingFace Space as-is:

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
*"She had your dark suit in greasy wash water all year."* No TIMIT licence is
needed to run the demo.
