"""Compute-efficient hyperparameter search for the Segmenter.

The SSL encoder is the expensive part of the pipeline, and its per-frame
features depend only on the audio — never on the Segmenter hparams. So this
script runs the encoder **once** per utterance, caches the features, then
sweeps a list of candidate hparam configurations over the cached features,
scoring each against the GT boundaries from a ``prepare_datasets`` CSV.

The candidate list is a JSON file: a list whose items are hparam-override
dicts (each merged on top of the model's default hparams). For example::

    [
      {"drop_k": 1},
      {"combined_signals": [
          {"name": "fwd_contrast", "kwargs": {"lookahead": 1}, "shift": 1},
          {"name": "fwd_contrast", "kwargs": {"lookahead": 2}, "shift": 1}
       ], "drop_k": 0}
    ]

Run as::

    python -m phonological_posteriogram.training.tune \\
        --model user/phonpost-wavlm-large \\
        --dataset_csv timit.csv \\
        --hparams_json grid.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..evaluation import SegmentationEvaluator
from ..phone_model import PhoneModel
from .evaluate import _gt_units, _pred_units


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
        "--hparams_json",
        type=Path,
        required=True,
        help="JSON file: a list of hparam-override dicts to sweep.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "test", "both"),
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--metric",
        default="rval",
        choices=("rval", "f1", "precision", "recall"),
        help="Metric to rank candidate hparams by.",
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
        "--device", default="cpu", help="Torch device (cpu, cuda:0, ...)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on number of utterances; useful for quick sanity checks.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to write the ranked results as JSON.",
    )
    parser.add_argument(
        "--save_best_dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to save the model with the best hparams "
            "baked in (as a PhoneModel artifact)."
        ),
    )
    args = parser.parse_args(argv)
    print(args)
    return args


def _extract_feature_cache(model, df, audio_paths):
    """Run the SSL encoder ONCE per utterance; return cache + GT units.

    cache: audio_path -> (per-frame feats, waveform). The expensive step.
    """
    sr = model.net_spec["sr"]
    cache = {}
    ground_truth = {}
    for audio_path in tqdm(audio_paths, desc="Encoding (once)"):
        try:
            x, _ = librosa.load(audio_path, sr=sr, mono=True)
            feats = model.extract_features(x)
        except Exception:
            print(f"Failed to encode {audio_path}:", file=sys.stderr)
            traceback.print_exc()
            continue
        cache[audio_path] = (feats, np.asarray(x, dtype=np.float32))
        ground_truth[audio_path] = _gt_units(df[df.audio_path == audio_path])
    return cache, ground_truth


def _score_hparams(model, overrides, cache, ground_truth, evaluator):
    """Run the Segmenter with one hparam override over the cached features."""
    seg = model.segmenter(overrides)
    frame_to_sec = model.net_spec["frame_shift"] / model.net_spec["sr"]
    predictions = {}
    for path, (feats, waveform) in cache.items():
        frames = seg.segment(feats, waveform)
        seconds = frames * frame_to_sec
        predictions[path] = _pred_units(seconds, len(waveform) / model.net_spec["sr"])
    return evaluator.evaluate_batch(predictions, ground_truth)


def run(args):
    model = PhoneModel.from_pretrained(args.model, device=args.device)

    hparam_list = json.loads(Path(args.hparams_json).read_text())
    if not isinstance(hparam_list, list) or not hparam_list:
        raise ValueError(
            f"{args.hparams_json} must contain a non-empty JSON list of "
            "hparam-override dicts."
        )

    df = pd.read_csv(args.dataset_csv)
    if args.split != "both":
        df = df[df.split == args.split]
    df = df[df.ipa.notna()].reset_index(drop=True)

    audio_paths = df.audio_path.unique()
    if args.limit is not None:
        audio_paths = audio_paths[: args.limit]

    # Expensive step — run the encoder once and cache.
    cache, ground_truth = _extract_feature_cache(model, df, audio_paths)
    if not cache:
        raise RuntimeError("No utterances could be encoded; nothing to tune.")

    evaluator = SegmentationEvaluator(
        tolerance_ms=args.tolerance_ms, match_mode=args.match_mode
    )

    # Cheap step — sweep hparams over the cached features.
    results = []
    for i, overrides in enumerate(hparam_list):
        try:
            metrics = _score_hparams(
                model, overrides, cache, ground_truth, evaluator
            )
            score = float(metrics.get(args.metric, 0.0))
            results.append(
                {"index": i, "overrides": overrides, "score": score,
                 "metrics": metrics}
            )
            print(f"[{i:3d}] {args.metric}={score:.4f}  {overrides}")
        except Exception as exc:  # one bad config shouldn't kill the sweep
            print(f"[{i:3d}] FAILED: {exc}", file=sys.stderr)
            results.append(
                {"index": i, "overrides": overrides, "score": float("-inf"),
                 "error": str(exc)}
            )

    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]
    print(
        f"\nBest: index {best['index']}  {args.metric}={best['score']:.4f}\n"
        f"  overrides: {best['overrides']}"
    )

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(results, indent=2, default=str))
        print(f"Wrote ranked results to {args.output_json}")

    if args.save_best_dir is not None and best["score"] != float("-inf"):
        best_model = PhoneModel(
            posteriogram=model.posteriogram,
            net_spec=model.net_spec,
            hparams={**model.hparams, **best["overrides"]},
        )
        out = best_model.save_pretrained(args.save_best_dir)
        print(f"Saved best model to {out}")

    return results


def main(argv=None):
    run(_get_args(argv))


if __name__ == "__main__":
    sys.exit(main())
