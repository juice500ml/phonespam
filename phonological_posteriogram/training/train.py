"""Fit a Segmenter on per-phone SSL features and save the model.

Reads a pickled DataFrame produced by ``training/extract_features.py`` (its
``df.attrs`` carries ``hf_repo``, ``encoder_layer``, ``pool``, ``sr``,
``frame_shift``), fits the phonological-vector projections and the
forward/backward regressors, and saves the artifact via
``PhonologicalPosteriogram.save_pretrained`` so it can be reloaded with
``PhonologicalPosteriogram.from_pretrained``.

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

from ..model import Segmenter
from ..pretrained import PhonologicalPosteriogram

REQUIRED_ATTRS = ("hf_repo", "encoder_layer", "sr", "frame_shift")


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
            "segmentation time. Recorded in the artifact's net_spec."
        ),
    )
    parser.add_argument(
        "--silence_backend",
        default="speech_plus",
        choices=("speech_plus", "logreg"),
        help=(
            "How to build the silence detector. 'speech_plus' (default) uses "
            "pv_ipa's 'speech+' projection — no extra training. 'logreg' "
            "fits a scikit-learn LogisticRegression on the per-phone "
            "features (silence = ipa == '_')."
        ),
    )
    parser.add_argument(
        "--silence_detector",
        type=Path,
        default=None,
        help=(
            "Optional override: load a pre-trained sklearn silence "
            "detector from this joblib path instead of fitting one. Takes "
            "precedence over --silence_backend."
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

    segmenter = Segmenter.fit(
        df,
        frame_shift=int(attrs["frame_shift"]),
        sr=int(attrs["sr"]),
        mel_frame_shift_ms=int(args.mel_frame_shift_ms),
        silence_backend=args.silence_backend,
        silence_detector_path=args.silence_detector,
    )

    net_spec = {
        "hf_repo": attrs["hf_repo"],
        "encoder_layer": int(attrs["encoder_layer"]),
        "frame_shift": int(attrs["frame_shift"]),
        "sr": int(attrs["sr"]),
        "mel_frame_shift_ms": int(args.mel_frame_shift_ms),
    }
    model = PhonologicalPosteriogram(segmenter=segmenter, net_spec=net_spec)
    out = model.save_pretrained(args.output_dir, filename=args.filename)
    print(f"Saved model artifact to {out}")
    return out


def main(argv=None):
    run(_get_args(argv))


if __name__ == "__main__":
    sys.exit(main())
