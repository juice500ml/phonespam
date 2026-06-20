"""Evaluate phone-classification metrics from a library artifact.

Loads a :class:`~phonological_posteriogram.phone_model.PhoneModel` (trained by
``training/train.py``), hardcodes the best segmentation config, and reports,
for both datasets, the best config's boundary R-value and the **oracle** +
**full-pipeline** phone-recognition metrics.

Scoring uses the external ``phone_metrics`` package. GT segmentation/labels
come from ``phone_metrics`` loaders, so labels are canonicalized consistently.

Run::

    uv run python scripts/evaluate_metrics.py \\
        --model trained/model.pt \\
        --timit_root data/TIMIT --vox_root data/voxangeles

Notes
-----
Both the oracle and pipeline paths recognize on the **sigmoid** posteriogram
(``project(act="sigmoid")``). The segmentation signals themselves still use the
raw projection internally (inside the library ``Segmenter``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import panphon
from phone_metrics import (
    PrecisionRecallMetric,
    load_timit,
    load_voxangeles,
    oracle_phone_accuracy,
    phone_error_rates,
)
from tqdm import tqdm

from phonological_posteriogram.phone_model import PhoneModel
from phonological_posteriogram.recognizer import Recognizer, panphon_featmap

# The best segmentation configuration (grid.json[0], also the library
# Segmenter's DEFAULT_COMBINED_SIGNALS). Hardcoded so the script is
# self-contained and independent of the artifact's stored hparams.
BEST_COMBINED_SIGNALS = [
    {"name": "frame_delta", "kwargs": {"offset": 3}, "shift": 2},
    {"name": "frame_delta", "kwargs": {"offset": 2}, "shift": 1},
    {"name": "frame_delta", "kwargs": {"offset": 1}, "shift": 1},
    {"name": "bwd_contrast", "kwargs": {"lookbehind": 2}, "shift": -1},
    {"name": "bwd_contrast", "kwargs": {"lookbehind": 3}, "shift": -1},
    {"name": "bwd_contrast", "kwargs": {"lookbehind": 1}, "shift": 0},
    {"name": "mel_svf", "kwargs": {"left": 2, "right": 1}, "shift": 0},
]


def _print_segmentation(name, seg_raw, seg_snap):
    """Best-config boundary R-value (raw + snapped), with precision/recall
    reported from the raw (non-snapped) boundaries."""
    r, rs = seg_raw.compute(), seg_snap.compute()
    print(
        f"{'best':16s} {name:12s} RV={r['rval']:.3f} (snapped={rs['rval']:.3f})  "
        f"P={r['precision']:.3f} R={r['recall']:.3f}"
    )


def evaluate_timit(model, utts, recognizer, segmenter, *, sr, frame_shift):
    """Best segmentation R-value + oracle and full-pipeline phone metrics."""
    frame_sec = frame_shift / sr
    oracle_utts, oracle_preds = [], []
    oracle_seqs, pipeline_seqs = [], []
    seg_raw = PrecisionRecallMetric(tolerance=0.02)
    seg_snap = PrecisionRecallMetric(tolerance=0.02)

    for utt in tqdm(utts, desc="TIMIT test", leave=False):
        wav = model.load_audio(utt.audio_path)
        feats = model.extract_features(wav)
        post = model.posteriogram.project(feats, view="ipa", act="sigmoid")
        segs = utt.segments

        bt = [s.end for s in segs[:-1]]  # inter-segment boundary times (seconds)
        labs = recognizer.recognize(post, bt, sr=sr, frame_shift=frame_shift)
        assert len(labs) == len(segs), (len(labs), len(segs))
        oracle_preds.extend(labs)
        oracle_seqs.append(list(labs))
        oracle_utts.append(utt)

        raw_bt = np.asarray(segmenter.segment(feats, wav, snap_silence=False)) * frame_sec
        pipe_bt = np.asarray(segmenter.segment(feats, wav, snap_silence=True)) * frame_sec
        seg_raw.update(utt.boundaries, raw_bt)
        seg_snap.update(utt.boundaries, pipe_bt)
        pipeline_seqs.append(
            recognizer.recognize(post, pipe_bt, sr=sr, frame_shift=frame_shift)
        )

    _print_segmentation("TIMIT_test", seg_raw, seg_snap)

    acc = oracle_phone_accuracy(oracle_utts, oracle_preds, label="ipa")
    er = phone_error_rates(oracle_utts, oracle_seqs, label="ipa")
    pipe_er = phone_error_rates(oracle_utts, pipeline_seqs, label="ipa")
    print(
        f"TIMIT merged-phone center-frame accuracy "
        f"(oracle, n={acc.total}): {acc.accuracy:.4f}"
    )
    print(
        f"TIMIT merged-phone center-frame error rates "
        f"(oracle, n={er.reference_total}): PER={er.per:.4f}  PFER={er.pfer:.4f}"
    )
    print(
        f"TIMIT full-pipeline phone error rates (best segmentation + recognizer, "
        f"n={pipe_er.reference_total}): PER={pipe_er.per:.4f}  PFER={pipe_er.pfer:.4f}"
    )


def evaluate_vox(model, utts, segmenter, *, featnames, sr, frame_shift):
    """Best segmentation R-value + VoxAngeles within-language oracle,
    panphon-unrestricted PFER, and full pipeline (per-language vocab)."""
    frame_sec = frame_shift / sr
    ft = panphon.FeatureTable()

    # Per-language candidate vocab = representable non-silence phones attested in
    # that language's test set, plus silence; one recognizer per language.
    lang_phones: dict[str, set[str]] = {}
    for utt in utts:
        for seg in utt.segments:
            if seg.ipa_label != "_" and ft.seg_known(seg.ipa_label):
                lang_phones.setdefault(utt.language, set()).add(seg.ipa_label)
    lang_rec = {
        lg: Recognizer(featnames=featnames, featmap=panphon_featmap([*sorted(ps), "_"], featnames))
        for lg, ps in lang_phones.items()
    }
    panphon_rec = Recognizer(featnames=featnames)

    oracle_utts, oracle_preds = [], []
    oracle_seqs, panphon_seqs, pipeline_seqs = [], [], []
    seg_raw = PrecisionRecallMetric(tolerance=0.02)
    seg_snap = PrecisionRecallMetric(tolerance=0.02)
    for utt in tqdm(utts, desc="VoxAngeles", leave=False):
        wav = model.load_audio(utt.audio_path)
        feats = model.extract_features(wav)
        post = model.posteriogram.project(feats, view="ipa", act="sigmoid")
        segs = utt.segments
        rec = lang_rec[utt.language]

        bt = [s.end for s in segs[:-1]]
        labs = rec.recognize(post, bt, sr=sr, frame_shift=frame_shift)
        panphon_labs = panphon_rec.recognize(post, bt, sr=sr, frame_shift=frame_shift)
        assert len(labs) == len(segs) == len(panphon_labs)
        oracle_preds.extend(labs)
        oracle_seqs.append(list(labs))
        panphon_seqs.append(list(panphon_labs))
        oracle_utts.append(utt)

        raw_bt = np.asarray(segmenter.segment(feats, wav, snap_silence=False)) * frame_sec
        pipe_bt = np.asarray(segmenter.segment(feats, wav, snap_silence=True)) * frame_sec
        seg_raw.update(utt.boundaries, raw_bt)
        seg_snap.update(utt.boundaries, pipe_bt)
        pipeline_seqs.append(
            rec.recognize(post, pipe_bt, sr=sr, frame_shift=frame_shift)
        )

    _print_segmentation("VoxAngeles", seg_raw, seg_snap)

    acc = oracle_phone_accuracy(oracle_utts, oracle_preds, label="ipa")
    er = phone_error_rates(oracle_utts, oracle_seqs, label="ipa")
    panphon_er = phone_error_rates(oracle_utts, panphon_seqs, label="ipa")
    pipe_er = phone_error_rates(oracle_utts, pipeline_seqs, label="ipa")
    n_langs = len({u.language for u in oracle_utts})
    print(
        f"VoxAngeles within-language phone accuracy "
        f"(oracle, {n_langs} languages, n={acc.total}): "
        f"micro={acc.accuracy:.4f}  macro_lang={acc.macro_language:.4f}"
    )
    print(
        f"VoxAngeles within-language phone error rates "
        f"(oracle, {n_langs} languages, n={er.reference_total}): "
        f"PER={er.per:.4f}  PFER={er.pfer:.4f}  "
        f"macro_lang_PER={er.macro_language_per:.4f}  "
        f"macro_lang_PFER={er.macro_language_pfer:.4f}"
    )
    print(
        f"VoxAngeles panphon-unrestricted PFER "
        f"(oracle, n={panphon_er.reference_total}): {panphon_er.pfer:.4f}"
    )
    print(
        f"VoxAngeles full-pipeline phone error rates (best segmentation + recognizer, "
        f"{n_langs} languages, n={pipe_er.reference_total}): "
        f"PER={pipe_er.per:.4f}  PFER={pipe_er.pfer:.4f}  "
        f"macro_lang_PER={pipe_er.macro_language_per:.4f}  "
        f"macro_lang_PFER={pipe_er.macro_language_pfer:.4f}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="PhoneModel artifact path or dir.")
    parser.add_argument("--timit_root", type=Path, default=Path("data/TIMIT"))
    parser.add_argument("--vox_root", type=Path, default=Path("data/voxangeles"))
    parser.add_argument("--device", default="cuda", help="torch device for the SSL encoder.")
    parser.add_argument("--skip_timit", action="store_true")
    parser.add_argument("--skip_vox", action="store_true")
    args = parser.parse_args(argv)

    model = PhoneModel.from_pretrained(args.model, device=args.device)
    sr = model.net_spec["sr"]
    frame_shift = model.encoder.stride_size
    featnames = model.posteriogram.featnames
    segmenter = model.segmenter({"combined_signals": BEST_COMBINED_SIGNALS})

    if not args.skip_timit:
        timit_rec = Recognizer(
            featnames=featnames, featmap=model.posteriogram.views["ipa"].featmap
        )
        evaluate_timit(
            model,
            load_timit(args.timit_root, split="test"),
            timit_rec,
            segmenter,
            sr=sr,
            frame_shift=frame_shift,
        )

    if not args.skip_vox:
        evaluate_vox(
            model,
            load_voxangeles(args.vox_root),
            segmenter,
            featnames=featnames,
            sr=sr,
            frame_shift=frame_shift,
        )


if __name__ == "__main__":
    main()
