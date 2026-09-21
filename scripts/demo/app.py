"""Gradio demo: spectrogram, SPAM, boundary signal and phones on one time axis.

Run locally with `python app.py`, or deploy the folder as a HuggingFace Space.
Not part of the phonespam package -- see ../README.md.
"""

from itertools import groupby
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from specplotter import SpecPlotter  # noqa: E402

from phonespam import PhoneModel, vocab_for_language  # noqa: E402
from phonespam.phoible import _phoible, phoible_csv_path  # noqa: E402
from phonespam.segmenter import Segmenter  # noqa: E402

MODEL_ID = "juice500/wavlm-24-phonemodel"
EXAMPLE_AUDIO = Path(__file__).parent / "examples" / "LDC93S1.wav"
EXAMPLE_TEXT = "She had your dark suit in greasy wash water all year."

UNCONSTRAINED = "Unconstrained (any PanPhon phone)"


def _languages():
    """Every PHOIBLE language as ``{"Name (Glottocode)": glottocode}``."""
    rows = _phoible()[["LanguageName", "Glottocode"]].dropna().drop_duplicates()
    rows = rows.sort_values("LanguageName", key=lambda c: c.str.lower())
    return {UNCONSTRAINED: None} | {
        f"{r.LanguageName} ({r.Glottocode})": r.Glottocode for r in rows.itertuples()
    }


def _vocab(glottocode):
    if glottocode is None:
        return None
    try:
        return vocab_for_language(glottocode)
    except ValueError:
        # Some PHOIBLE sources transcribe no allophones; use their phonemes.
        return vocab_for_language(glottocode, phoneme=True)


def _paired_order(featnames):
    """Row order putting each feature's + and - channels next to each other."""
    index = {name: i for i, name in enumerate(featnames)}
    order, placed = [], set()
    for i, name in enumerate(featnames):
        if i in placed:
            continue
        order.append(i)
        placed.add(i)
        partner = index.get(name[:-1] + "-") if name.endswith("+") else None
        if partner is not None and partner not in placed:
            order.append(partner)
            placed.add(partner)
    return order


def _signal_groups():
    """Group DEFAULT_COMBINED_SIGNALS under the UI's four labels.

    Derived rather than hardcoded so the checkboxes cannot drift from the
    ensemble. Mirrors the ablation ladder in ``scripts/method_ablation.sh``.
    """
    groups = {
        "Diff. between frames": [],
        "Multi-scale diff.": [],
        "Backward contrast": [],
        "Mel spectrogram": [],
    }
    for spec in Segmenter.DEFAULT_COMBINED_SIGNALS:
        if spec["name"] == "frame_delta":
            key = (
                "Diff. between frames" if spec["kwargs"].get("offset") == 1 else "Multi-scale diff."
            )
        elif spec["name"] == "bwd_contrast":
            key = "Backward contrast"
        else:
            key = "Mel spectrogram"
        groups[key].append(dict(spec))
    return groups


SIGNAL_GROUPS = _signal_groups()
SIGNAL_CHOICES = list(SIGNAL_GROUPS)

# Loaded once at startup, not per request.
print(f"loading {MODEL_ID} ...")
MODEL = PhoneModel.from_pretrained(MODEL_ID)
_ = MODEL.encoder  # force the lazy SSL encoder download now, not on first click
print("warming the PHOIBLE inventory table ...")
phoible_csv_path()
LANGUAGES = _languages()
print(f"ready ({len(LANGUAGES) - 1} languages).")


def analyze(audio_path, signals, language):
    """Run the model and draw the four aligned panels."""
    if not audio_path:
        raise gr.Error("Provide some audio: upload a file, record, or use the example.")
    if not signals:
        raise gr.Error(
            "Select at least one segmentation signal -- with none of them there is "
            "no boundary signal to find peaks in."
        )

    specs = [spec for name in signals for spec in SIGNAL_GROUPS[name]]
    segmenter = MODEL.segmenter.with_hparams(
        {"combined_signals": specs, "combined_prominence": 0.001}
    )

    wav = MODEL.load_audio(audio_path)
    feats = MODEL.extract_features(wav)
    spam = MODEL.spam_from_features(feats)  # normalized to [0, 1]
    duration = len(wav) / MODEL.sr

    frames, signal = segmenter.segment(feats, wav, return_signal=True)
    times = MODEL.encoder.frame_to_time(frames)

    vocab = _vocab(LANGUAGES[language])
    kw = dict(sr=MODEL.sr, frame_shift=MODEL.frame_shift, vocab=vocab)
    labels = MODEL.recognizer.recognize(spam, times, **kw)
    edges = np.concatenate([[0.0], times, [duration]])
    stepwise = _runs(MODEL.recognizer.recognize(spam, spam.times[1:], **kw), spam.times, duration)

    return _figure(wav, spam, signal, times, edges, labels, stepwise, duration)


def _runs(labels, frame_times, duration):
    """Collapse per-frame labels into (start, end, label) spans."""
    spans, i = [], 0
    for label, group in groupby(labels):
        n = len(list(group))
        end = frame_times[i + n] if i + n < len(frame_times) else duration
        spans.append((frame_times[i], end, label))
        i += n
    return spans


def _strip(ax, spans, colour, edge, fontsize=9):
    for start, end, label in spans:
        ax.add_patch(
            plt.Rectangle(
                (start, 0.05), max(end - start, 1e-4), 0.9, facecolor=colour, edgecolor=edge
            )
        )
        ax.text((start + end) / 2, 0.5, label, ha="center", va="center", fontsize=fontsize)
    ax.set_ylim(0, 1)
    ax.set_yticks([])


def _figure(wav, spam, signal, times, edges, labels, stepwise, duration):
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(13, 9.5),
        height_ratios=[3, 4.5, 0.7, 1.3, 0.7],
        sharex=True,
        layout="constrained",
    )
    ax_spec, ax_spam, ax_step, ax_sig, ax_phn = axes

    SpecPlotter(sample_rate=MODEL.sr).plot_spectrogram(wav, ax=ax_spec, show_annotation=False)
    ax_spec.set_title("Spectrogram", loc="left", fontsize=10)
    # SpecPlotter draws its own ticks and grid; sharex makes them redundant.
    ax_spec.set_xlabel("")
    ax_spec.grid(False)
    plt.setp(ax_spec.get_xticklabels(), visible=False)

    order = _paired_order(spam.featnames)
    ax_spam.imshow(
        np.asarray(spam).T[order],
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=1,
        origin="upper",
        extent=[0, duration, len(order) - 0.5, -0.5],
        interpolation="nearest",
    )
    ax_spam.set_yticks(range(len(order)), [spam.featnames[i] for i in order], fontsize=6)
    ax_spam.set_ylabel("SPAM")
    ax_spam.set_title(
        f"SPAM: S3M-based Phonological Activation Map ({len(spam.featnames)} channels)",
        loc="left",
        fontsize=10,
    )

    # Edge frames are NaN (the delta/contrast windows have no room there).
    frame_times = spam.times
    ax_sig.plot(frame_times, signal, lw=1.0, color="#1f77b4")
    peak_idx = MODEL.encoder.time_to_frame(times)
    ax_sig.plot(times, signal[np.clip(peak_idx, 0, len(signal) - 1)], "o", ms=3, color="#d62728")
    finite = signal[np.isfinite(signal)]
    if finite.size:
        pad = 0.05 * (finite.max() - finite.min() or 1.0)
        ax_sig.set_ylim(finite.min() - pad, finite.max() + pad)
    ax_sig.set_ylabel("signal")
    ax_sig.set_title("Combined boundary signal", loc="left", fontsize=10)
    ax_sig.margins(x=0)

    _strip(ax_step, stepwise, "#f3e7dc", "#bb9c82", fontsize=6)
    ax_step.set_ylabel("frame-wise")
    ax_step.set_title("Recognition head alone, per frame", loc="left", fontsize=10)

    _strip(ax_phn, list(zip(edges[:-1], edges[1:], labels, strict=True)), "#dfe7f5", "#8296bb")
    ax_phn.set_ylabel("phones")
    ax_phn.set_title(
        "Final prediction (using both recognition and segmentation heads)", loc="left", fontsize=10
    )
    ax_phn.set_xlabel("Time [s]")

    for ax in axes:
        if ax is ax_step:
            continue
        colour = "white" if ax is ax_spam else "black"
        for t in times:
            ax.axvline(t, color=colour, lw=0.8, alpha=0.75)
    ax_spec.set_xlim(0, duration)
    return fig


with gr.Blocks(title="SPAM Demo") as demo:
    gr.Markdown(f"""
## 🗣️ SPAM — Phone Segmentation and Recognition Demo

Demo for [Phone Segmentation and Recognition through Phonological Activation
Mapping](https://arxiv.org/abs/2607.09020) (SLT 2026), using
[`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}).

Upload audio, record your own, or use the example, then click **Run**.
""")

    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(
                label="Input Audio",
                type="filepath",
                sources=["upload", "microphone"],
                value=str(EXAMPLE_AUDIO),
            )
            gr.Markdown(f"""
The example is TIMIT's `LDC93S1`: *"{EXAMPLE_TEXT}"*
""")
        with gr.Column(scale=1):
            signal_boxes = gr.CheckboxGroup(
                choices=SIGNAL_CHOICES,
                value=SIGNAL_CHOICES,
                label="Segmentation signals",
                info="The released model combines all four as a fuzzy-AND.",
                interactive=True,
            )
            language = gr.Dropdown(
                choices=list(LANGUAGES),
                value=UNCONSTRAINED,
                label="Phone inventory",
                info=("Restrict recognition to one language's PHOIBLE inventory."),
                interactive=True,
            )
            run_btn = gr.Button("▶ Run", variant="primary")

    plot = gr.Plot(show_label=False)

    run_btn.click(fn=analyze, inputs=[audio, signal_boxes, language], outputs=plot)
    demo.load(fn=analyze, inputs=[audio, signal_boxes, language], outputs=plot)

if __name__ == "__main__":
    demo.launch()
