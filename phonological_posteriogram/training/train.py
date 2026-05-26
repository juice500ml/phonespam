"""Fit a PhonologicalPosteriogram on per-phone SSL features and save a model.

Reads a pickled DataFrame produced by ``training/extract_features.py`` (its
``df.attrs`` carries ``hf_repo``, ``encoder_layer``, ``pool``, ``sr``), fits
the three phonological-vector views and the two forward/backward regressors,
wraps them in a :class:`PhoneModel`, and saves the artifact so it can be
reloaded with ``PhoneModel.from_pretrained``.

Run as::

    python -m phonological_posteriogram.training.train \\
        --features_pkl feats.pkl \\
        --output_dir ./trained
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
        "--mel_frame_shift_ms",
        type=int,
        default=10,
        help=(
            "Mel-spectrogram hop in ms used by the mel_svf signal at "
            "segmentation time. Stored in the artifact's hparams."
        ),
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

    # The expensive step: fit the weights-only PhonologicalPosteriogram.
    posteriogram = PhonologicalPosteriogram.fit(df)

    net_spec = {
        "hf_repo": attrs["hf_repo"],
        "encoder_layer": int(attrs["encoder_layer"]),
        "sr": int(attrs["sr"]),
    }
    hparams = Segmenter.default_hparams()
    hparams["mel_frame_shift_ms"] = int(args.mel_frame_shift_ms)

    model = PhoneModel(
        posteriogram=posteriogram, net_spec=net_spec, hparams=hparams
    )
    out = model.save_pretrained(args.output_dir, filename=args.filename)
    print(f"Saved model artifact to {out}")
    return out


def main(argv=None):
    run(_get_args(argv))


if __name__ == "__main__":
    sys.exit(main())
