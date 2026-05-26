"""Compute-efficient hyperparameter search for the Segmenter.

The SSL encoder is the expensive part of the pipeline, and its per-frame
features depend only on the audio — never on the Segmenter hparams. So
this script runs the encoder **once** per utterance, caches the features
**and** the posteriogram, then sweeps a list of candidate hparam
configurations over the cached state, scoring each against the GT
boundaries and labels from a ``prepare_datasets`` CSV.

Every candidate is scored under **two** recognition conditions and the
two PER/PFER values are reported side-by-side:

- **known-language**: the per-utterance phone vocab from Phoible, resolved
  from the CSV's ``language`` column (an ISO 639-3 code per row, as
  written by :mod:`phonological_posteriogram.training.prepare_datasets`).
- **unknown-language**: no vocab constraint at all — the full panphon
  phone vocab.

Boundary metrics (rval / f1 / precision / recall) are vocab-independent,
so they're computed once and shown unprefixed. Recognition metrics appear
twice: ``known_per`` / ``known_pfer`` and ``unknown_per`` / ``unknown_pfer``.

The ``--metric`` arg picks which value to rank candidates by:
higher-is-better for boundary metrics, lower-is-better for per/pfer.

The candidate list is a JSON file: a list whose items are hparam-override
dicts (each merged on top of the model's default hparams).

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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..evaluation import (
    PhoneRecognitionEvaluator,
    SegmentationEvaluator,
    SegmentationUnit,
)
from ..phone_model import PhoneModel
from ..recognizer import _resolve_lang
from .evaluate import _gt_units

LOWER_IS_BETTER = frozenset({"known_per", "known_pfer", "unknown_per", "unknown_pfer"})
METRIC_CHOICES = (
    "rval", "f1", "precision", "recall",
    "known_per", "known_pfer", "unknown_per", "unknown_pfer",
)
_NO_CONSTRAINT = dict(lang=None, phoible_id=None, phoneme=False, vocab=None)


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
        choices=METRIC_CHOICES,
        help=(
            "Metric to rank candidates by. Boundary metrics "
            "(rval/f1/precision/recall) are higher-is-better and "
            "vocab-independent; known_/unknown_per/pfer are "
            "lower-is-better and split by recognition condition."
        ),
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
    """Run the SSL encoder ONCE per utterance; cache feats, waveform,
    AND posteriogram (the posteriogram is hparam-independent, so it's
    safe to cache alongside the features)."""
    cache = {}
    ground_truth = {}
    for audio_path in tqdm(audio_paths, desc="Encoding (once)"):
        try:
            x = model.load_audio(audio_path)
            feats = model.extract_features(x)
            posteriogram = model.posteriogram.project(
                feats, view="ipa", act="sigmoid"
            )
        except Exception:
            print(f"Failed to encode {audio_path}:", file=sys.stderr)
            traceback.print_exc()
            continue
        cache[audio_path] = (feats, x, posteriogram)
        ground_truth[audio_path] = _gt_units(df[df.audio_path == audio_path])
    return cache, ground_truth


def _triples_to_units(model, triples):
    """Wrap frame-based (start, end, label) triples from
    :meth:`Recognizer.recognize` into seconds-based SegmentationUnits via
    the encoder's frame→time map."""
    if not triples:
        return []
    starts = model.encoder.frame_to_time(np.array([t[0] for t in triples]))
    ends = model.encoder.frame_to_time(np.array([t[1] for t in triples]))
    return [
        SegmentationUnit(float(starts[i]), float(ends[i]), triples[i][2])
        for i in range(len(triples))
    ]


def _resolve_known_lang_kwargs(df, audio_paths):
    """Per-utterance recognize() kwargs for the **known-language** run.

    The dataset CSV must carry a ``language`` column (ISO 639-3 codes, as
    written by :mod:`prepare_datasets`). Each utterance's code is mapped
    to a Phoible InventoryID via :func:`_resolve_lang`. Codes that don't
    resolve fall back to no constraint (full panphon) for those
    utterances; a single warning collects them.
    """
    if "language" not in df.columns:
        raise ValueError(
            "tune.py requires a `language` column in the dataset CSV "
            "(an ISO 639-3 code per row). Re-run "
            "`prepare_datasets.py` to regenerate the CSV."
        )
    path_to_lang = (
        df.drop_duplicates("audio_path")
          .set_index("audio_path")["language"]
          .to_dict()
    )
    resolved = {}
    unresolved: set = set()
    for p in audio_paths:
        try:
            pid = _resolve_lang(str(path_to_lang[p]))
            resolved[p] = dict(
                lang=None, phoible_id=pid, phoneme=False, vocab=None
            )
        except ValueError:
            unresolved.add(str(path_to_lang[p]))
            resolved[p] = dict(_NO_CONSTRAINT)
    if unresolved:
        warnings.warn(
            f"Could not resolve {len(unresolved)} language(s) to a Phoible "
            f"InventoryID: {sorted(unresolved)[:10]}"
            f"{'...' if len(unresolved) > 10 else ''}; those utterances run "
            "with no vocab constraint in the known-language pass.",
            stacklevel=2,
        )
    return resolved


def _score_hparams(
    model, overrides, kwargs_known,
    cache, ground_truth, seg_evaluator, rec_evaluator,
):
    """Score one hparam override on the cached features under BOTH
    recognition conditions (known-language + unknown-language).

    Boundaries are vocab-independent, so we segment once and feed the
    same boundaries through the recognizer twice (constrained + free).
    Returns a flat metrics dict with seg metrics unprefixed and
    recognition metrics under ``known_*`` / ``unknown_*`` keys.
    """
    seg = model.segmenter(overrides)
    preds_known, preds_unknown = {}, {}
    for path, (feats, waveform, posteriogram) in cache.items():
        boundaries = seg.segment(feats, waveform)
        triples_known = model.recognizer.recognize(
            posteriogram, boundaries, **kwargs_known[path]
        )
        triples_unknown = model.recognizer.recognize(
            posteriogram, boundaries, **_NO_CONSTRAINT
        )
        preds_known[path] = _triples_to_units(model, triples_known)
        preds_unknown[path] = _triples_to_units(model, triples_unknown)

    # Boundaries (and thus seg metrics) are identical across the two
    # predictions — same segmenter, only the labels differ. Compute once.
    seg_metrics = seg_evaluator.evaluate_batch(preds_known, ground_truth)
    rec_known = rec_evaluator.evaluate_batch(preds_known, ground_truth)
    rec_unknown = rec_evaluator.evaluate_batch(preds_unknown, ground_truth)
    return {
        **seg_metrics,
        **{f"known_{k}": v for k, v in rec_known.items()},
        **{f"unknown_{k}": v for k, v in rec_unknown.items()},
    }


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

    # Expensive step — run the encoder + posteriogram once and cache.
    cache, ground_truth = _extract_feature_cache(model, df, audio_paths)
    if not cache:
        raise RuntimeError("No utterances could be encoded; nothing to tune.")

    seg_evaluator = SegmentationEvaluator(
        tolerance_ms=args.tolerance_ms, match_mode=args.match_mode
    )
    rec_evaluator = PhoneRecognitionEvaluator()

    # Known-language kwargs are resolved once and reused across all sweeps.
    kwargs_known = _resolve_known_lang_kwargs(df, list(cache.keys()))

    lower_better = args.metric in LOWER_IS_BETTER
    failed_score = float("inf") if lower_better else float("-inf")

    # Cheap step — sweep hparams over the cached features. The recognizer
    # (and its predmat) is the model's embedded one; vocab masks are
    # applied per-call, so the sweep doesn't rebuild them.
    results = []
    for i, overrides in enumerate(hparam_list):
        try:
            metrics = _score_hparams(
                model, overrides, kwargs_known,
                cache, ground_truth,
                seg_evaluator, rec_evaluator,
            )
            score = float(metrics.get(args.metric, failed_score))
            results.append(
                {"index": i, "overrides": overrides, "score": score,
                 "metrics": metrics}
            )
            print(
                f"[{i:3d}] {rval}={metrics.get('rval', failed_score):.4f}  "
                f"[{i:3d}] {precision}={metrics.get('precision', failed_score):.4f}  "
                f"[{i:3d}] {recall}={metrics.get('recall', failed_score):.4f}  "
                f"known_per={metrics.get('known_per', float('nan')):.4f}  "
                f"unknown_per={metrics.get('unknown_per', float('nan')):.4f}  "
                f"{overrides}"
            )
        except Exception as exc:  # one bad config shouldn't kill the sweep
            print(f"[{i:3d}] FAILED: {exc}", file=sys.stderr)
            results.append(
                {"index": i, "overrides": overrides,
                 "score": failed_score, "error": str(exc)}
            )

    results.sort(key=lambda r: r["score"], reverse=not lower_better)
    best = results[0]
    print(
        f"\nBest: index {best['index']}  {args.metric}={best['score']:.4f}\n"
        f"  overrides: {best['overrides']}"
    )

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(results, indent=2, default=str))
        print(f"Wrote ranked results to {args.output_json}")

    if args.save_best_dir is not None and best["score"] not in (
        float("inf"), float("-inf")
    ):
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
