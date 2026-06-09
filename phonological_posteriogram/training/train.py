"""Fit a PhonologicalPosteriogram on per-phone SSL features and save a model.

Reads a pickled DataFrame produced by ``training/extract_features.py`` (its
``df.attrs`` carries ``hf_repo``, ``encoder_layer``, ``pool``, ``sr``), fits
the ``ipa`` phonological-vector view and the backward regressor under the
notebook's TIMIT closure/release label scheme, wraps them in a
:class:`PhoneModel`, and saves the artifact so it can be reloaded with
``PhoneModel.from_pretrained``.

Run as::

    python -m phonological_posteriogram.training.train \\
        --features_pkl feats.pkl \\
        --output_dir ./trained
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..phone_model import PhoneModel
from ..posteriogram import PhonologicalPosteriogram
from ..segmenter import Segmenter

REQUIRED_ATTRS = ("hf_repo", "encoder_layer", "sr")


def _get_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    parser.add_argument(
        "--features_pkl",
        type=Path,
        required=True,
        help="Pickled DataFrame from training/extract_features.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write the trained model artifact into.",
    )
    parser.add_argument(
        "--filename",
        default="model.pt",
        help="Filename for the saved artifact (within --output_dir).",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "test", "both"),
        help=(
            "Feature split to fit on when the pickle contains a split column. "
            "Use train for release artifacts."
        ),
    )
    parser.add_argument(
        "--mel_frame_shift_ms",
        type=int,
        default=10,
        help=(
            "Mel-spectrogram hop in ms used by the mel_svf signal at "
            "segmentation time. Stored in the artifact's hparams."
        ),
    )
    parser.add_argument(
        "--push_to_hub",
        default=None,
        help=(
            "Optional HF Hub model repo id (e.g. 'org/name'). When set, the "
            "trained artifact is pushed there via PhoneModel.push_to_hub. "
            "Requires `huggingface-cli login`."
        ),
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help=(
            "When --push_to_hub creates the repo, mark it private. Ignored "
            "if the repo already exists."
        ),
    )
    parser.add_argument(
        "--hparams_overrides",
        default=None,
        help=(
            "Optional JSON string of segmenter hparam overrides merged on top "
            "of Segmenter.default_hparams() (e.g. structural ablations of "
            "``combined_signals``). Stored as-is in the artifact."
        ),
    )
    parser.add_argument(
        "--audio_fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction (0, 1] of training utterances (audio_paths) to subsample "
            "before fitting — for sample-efficiency ablations. Whole utterances "
            "are kept or dropped (not individual phone rows)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the --audio_fraction utterance subsample (reproducible).",
    )
    args = parser.parse_args(argv)
    print(args)
    return args


def run(args):
    df = pd.read_pickle(args.features_pkl)
    attrs = dict(df.attrs)

    missing = [k for k in REQUIRED_ATTRS if k not in attrs]
    if missing:
        raise ValueError(
            f"Features pkl is missing required df.attrs keys: {missing}. "
            "Re-run training/extract_features.py to regenerate it."
        )

    if args.split != "both":
        if "split" not in df.columns:
            raise ValueError(
                "--split was set but the features pickle has no `split` column."
            )
        df = df[df.split == args.split].reset_index(drop=True)

    # Sample-efficiency ablations: drop a deterministic fraction of whole
    # utterances (audio_paths) before fitting. Individual phone rows within
    # a kept utterance are all retained.
    if args.audio_fraction < 1.0:
        if not 0.0 < args.audio_fraction <= 1.0:
            raise ValueError(
                f"--audio_fraction must be in (0, 1]; got {args.audio_fraction}"
            )
        paths = np.sort(df.audio_path.unique())
        k = max(1, int(round(len(paths) * args.audio_fraction)))
        rng = np.random.default_rng(args.seed)
        keep = set(rng.choice(paths, size=k, replace=False))
        df = df[df.audio_path.isin(keep)].reset_index(drop=True)
        print(
            f"Subsampled to {k}/{len(paths)} utterances "
            f"({args.audio_fraction:g} fraction, seed={args.seed})."
        )

    # The expensive step: fit the weights-only PhonologicalPosteriogram on the
    # notebook's TIMIT closure/release label scheme from raw TIMIT features.
    posteriogram = PhonologicalPosteriogram.fit_timit_closure_release(df)

    net_spec = {
        "hf_repo": attrs["hf_repo"],
        "encoder_layer": int(attrs["encoder_layer"]),
        "sr": int(attrs["sr"]),
    }
    hparams = Segmenter.default_hparams()
    hparams["mel_frame_shift_ms"] = int(args.mel_frame_shift_ms)
    if args.hparams_overrides:
        overrides = json.loads(args.hparams_overrides)
        if not isinstance(overrides, dict):
            raise ValueError(
                "--hparams_overrides must be a JSON object (dict), got "
                f"{type(overrides).__name__}."
            )
        hparams.update(overrides)
        print(f"Applied hparam overrides: {sorted(overrides)}")

    model = PhoneModel(
        posteriogram=posteriogram, net_spec=net_spec, hparams=hparams
    )
    out = model.save_pretrained(args.output_dir, filename=args.filename)
    print(f"Saved model artifact to {out}")
    if args.push_to_hub:
        url = model.push_to_hub(
            args.push_to_hub,
            filename=args.filename,
            private=args.hub_private,
        )
        print(f"Pushed artifact to {url}")
    return out


def main(argv=None):
    run(_get_args(argv))


if __name__ == "__main__":
    sys.exit(main())
