"""Evaluate a pretrained model against a `prepare_datasets` CSV.

Runs segmentation **and** phone recognition end-to-end on every utterance
in the CSV and reports both:

* Boundary metrics (precision / recall / F1 / R-value) via
  :class:`SegmentationEvaluator`.
* Phone recognition (PER / PFER) via :class:`PhoneRecognitionEvaluator`.

By default the recognizer outputs the full panphon phone vocab; pass
``--vocab`` (an explicit comma-separated phone list) **or** ``--lang`` /
``--phoible_id`` (with optional ``--phoneme``) to constrain the output
vocabulary.

Run as::

    python -m phonological_posteriogram.training.evaluate \\
        --model user/phonpost-wavlm-large \\
        --dataset_csv data/timit-merged.csv \\
        --split test

For TIMIT, ``prepare_datasets`` writes both ``data/timit-merged.csv`` and
``data/timit-raw.csv``. Use the merged CSV for evaluation ground truth; the
raw CSV is intended for training features (its closures feed the
closure/release fit).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..evaluation import (
    PhoneRecognitionEvaluator,
    SegmentationEvaluator,
    SegmentationUnit,
)
from ..phone_model import PhoneModel


def _get_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace repo id or local path/dir for the pretrained model.",
    )
    parser.add_argument(
        "--dataset_csv",
        type=Path,
        required=True,
        help="CSV produced by training/prepare_datasets.py.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "test", "both"),
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--tolerance_ms",
        type=int,
        default=20,
        help="Boundary tolerance for true-positive counting.",
    )
    parser.add_argument(
        "--match_mode",
        default="lenient",
        choices=("lenient", "strict"),
        help="Boundary match policy.",
    )
    parser.add_argument(
        "--forced",
        action="store_true",
        help=(
            "Also compute per-phone error stats. Only meaningful when "
            "predicted and GT segment counts match closely; otherwise the "
            "1-to-1 zip is truncated and the numbers are misleading."
        ),
    )
    parser.add_argument(
        "--no_snap_silence",
        action="store_true",
        help=(
            "Disable silence-snapping in the segmenter (overrides the "
            "model's saved ``snap_silence`` hparam for this run)."
        ),
    )
    parser.add_argument(
        "--vocab",
        default=None,
        help=(
            "Comma-separated list of phones to constrain the recognizer's "
            "output vocabulary. Mutually exclusive with --lang/--phoible_id."
        ),
    )
    parser.add_argument(
        "--lang",
        default=None,
        help=(
            "Constrain the recognizer's vocab to one Phoible language "
            "(LanguageName / ISO 639-3 / Glottocode)."
        ),
    )
    parser.add_argument(
        "--phoible_id",
        type=int,
        default=None,
        help="Constrain the recognizer's vocab to one Phoible InventoryID.",
    )
    parser.add_argument(
        "--phoneme",
        action="store_true",
        help=(
            "When --lang/--phoible_id is set, use the inventory's abstract "
            "Phoneme set rather than the surface Allophones."
        ),
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device (cpu, cuda:0, ...)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on number of utterances; useful for quick sanity checks.",
    )
    return parser.parse_args(argv)


def _gt_units(group_df):
    """Build a sorted list of GT :class:`SegmentationUnit` from a CSV group."""
    g = group_df.sort_values("min")
    return [
        SegmentationUnit(float(row.min), float(row.max), row.ipa)
        for row in g.itertuples()
    ]


def _print_recog_results(r):
    if not r:
        print("\nPhone Recognition: no utterances scored.")
        return
    print(
        f"\nPhone Recognition:  PER={r['per']:.4f}  PFER={r['pfer']:.4f}  "
        f"(utterances={r['n_utterances']})"
    )


def run(args):
    model = PhoneModel.from_pretrained(args.model, device=args.device)
    if args.no_snap_silence:
        # The new ``PhoneModel.recognize`` doesn't expose per-call segmenter
        # toggles, so override the hparam in place for this run.
        model.hparams["snap_silence"] = False

    vocab = (
        [p for p in args.vocab.split(",") if p] if args.vocab is not None else None
    )

    df = pd.read_csv(args.dataset_csv)
    if args.split != "both":
        df = df[df.split == args.split]
    df = df[df.ipa.notna()].reset_index(drop=True)

    audio_paths = df.audio_path.unique()
    if args.limit is not None:
        audio_paths = audio_paths[: args.limit]

    predictions = {}
    ground_truth = {}
    symbols_dict = {} if args.forced else None

    for audio_path in tqdm(audio_paths, desc="Evaluating"):
        units = model.recognize(
            audio_path,
            lang=args.lang,
            phoible_id=args.phoible_id,
            phoneme=args.phoneme,
            vocab=vocab,
        )

        gt = _gt_units(df[df.audio_path == audio_path])
        ground_truth[audio_path] = gt
        predictions[audio_path] = units
        if args.forced:
            symbols_dict[audio_path] = [u.label for u in gt]

    seg_eval = SegmentationEvaluator(
        tolerance_ms=args.tolerance_ms,
        forced=args.forced,
        match_mode=args.match_mode,
    )
    seg_results = seg_eval.evaluate_batch(
        predictions, ground_truth, symbols_dict=symbols_dict
    )
    seg_eval.pretty_print(seg_results)

    rec_eval = PhoneRecognitionEvaluator()
    rec_results = rec_eval.evaluate_batch(predictions, ground_truth)
    _print_recog_results(rec_results)

    return {**seg_results, **rec_results}


def main(argv=None):
    run(_get_args(argv))


if __name__ == "__main__":
    sys.exit(main())
