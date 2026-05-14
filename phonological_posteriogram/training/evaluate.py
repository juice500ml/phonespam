"""Evaluate a pretrained model against a `prepare_datasets` CSV.

Loads each utterance referenced by the CSV, runs ``model.segment_seconds``
to get predicted boundary times, and pipes the results through
:class:`phonological_posteriogram.evaluation.SegmentationEvaluator` so
boundary-level metrics (precision / recall / F1 / R-value) can be read
directly from the same artifact + CSV pair that training produced.

Run as::

    python -m phonological_posteriogram.training.evaluate \\
        --model user/phonpost-wavlm-large \\
        --dataset_csv timit.csv \\
        --split test
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..evaluation import SegmentationEvaluator, SegmentationUnit
from ..pretrained import PhonologicalPosteriogram


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
        help="Disable silence-snapping in the segmenter.",
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


def _pred_units(boundary_times, audio_duration):
    """Wrap predicted boundary times into per-segment units.

    Pads with 0 and ``audio_duration`` so the units cover the full audio,
    matching how GT rows tile a TIMIT utterance.
    """
    bs = np.unique(
        np.concatenate(
            [
                np.asarray([0.0]),
                np.asarray(boundary_times, dtype=float),
                np.asarray([float(audio_duration)]),
            ]
        )
    )
    return [SegmentationUnit(float(bs[i]), float(bs[i + 1])) for i in range(len(bs) - 1)]


def run(args):
    model = PhonologicalPosteriogram.from_pretrained(args.model, device=args.device)
    sr = model.net_spec["sr"]

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
        try:
            x, _ = librosa.load(audio_path, sr=sr, mono=True)
            pred_times = model.segment_seconds(
                x, snap_silence=not args.no_snap_silence
            )
        except Exception:
            print(f"Failed on {audio_path}:", file=sys.stderr)
            traceback.print_exc()
            continue

        audio_duration = len(x) / sr
        group = df[df.audio_path == audio_path]
        gt = _gt_units(group)
        ground_truth[audio_path] = gt
        predictions[audio_path] = _pred_units(pred_times, audio_duration)
        if args.forced:
            symbols_dict[audio_path] = [u.label for u in gt]

    evaluator = SegmentationEvaluator(
        tolerance_ms=args.tolerance_ms,
        forced=args.forced,
        match_mode=args.match_mode,
    )
    results = evaluator.evaluate_batch(
        predictions,
        ground_truth,
        symbols_dict=symbols_dict,
    )
    evaluator.pretty_print(results)
    return results


def main(argv=None):
    run(_get_args(argv))


if __name__ == "__main__":
    sys.exit(main())
