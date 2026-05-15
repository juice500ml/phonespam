"""Dump per-segment SSL features for a dataset CSV.

Runs an SSL speech encoder on each utterance, slices the per-frame features
to each labeled phone span, pools each slice to a single vector, and writes
out a pickled pandas DataFrame. The encoder/layer/pool/sr settings are
written into ``df.attrs`` so the same configuration can be reproduced at
inference time.

Note on input sample rate:
    SSL speech encoders are trained at 16 kHz, but the convolutional feature
    extractor processes raw float samples and is agnostic to the real-world
    sample rate. Feeding the model a higher-rate signal (e.g. 32 kHz) makes
    each output frame cover 320 input samples = 10 ms of real time instead
    of 20 ms. We always tell the HF processor ``sampling_rate ==
    processor.sampling_rate`` (its native 16 kHz) to skip its validation
    check, and record the *real* input sampling rate in ``df.attrs["sr"]``.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..features import SSLEncoder

# Internal stride of every wav2vec2-family SSL encoder, in input samples.
SSL_FRAME_SHIFT = 320

POOL_CHOICES = ("center", "average")


def _get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="microsoft/wavlm-large",
        help="HuggingFace SSL speech encoder repo id.",
    )
    parser.add_argument(
        "--dataset_csv",
        type=Path,
        required=True,
        help="Per-phone dataset CSV produced by prepare_datasets.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "test", "both"),
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Output .pkl path."
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device, e.g. cpu or cuda:0."
    )
    parser.add_argument(
        "--layer_index",
        type=int,
        default=-1,
        help="Hidden layer index to pull features from. -1 == last_hidden_state.",
    )
    parser.add_argument(
        "--pool",
        default="center",
        choices=POOL_CHOICES,
        help="Pooling method for each per-phone feature slice.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help=(
            "Input audio sample rate. The model still treats every 320 samples "
            "as one frame, so e.g. sr=32000 gives 10 ms hops instead of 20 ms."
        ),
    )
    args = parser.parse_args(argv)
    print(args)
    return args


def _slice_feats(row, feats):
    f = feats[row.audio_path]
    i = int(row["_frame_min"])
    j = int(row["_frame_max"])
    if j <= i:
        j = i + 1
    i = min(i, max(0, len(f) - 1))
    j = min(j, len(f))
    return f[i:j]


def _pool_feats(feat, pool):
    if pool == "center":
        return feat[len(feat) // 2]
    if pool == "average":
        return feat.mean(0)
    raise ValueError(f"Unknown pool: {pool}")


def run(args):
    np.random.seed(42)

    df = pd.read_csv(args.dataset_csv)
    if args.split != "both":
        df = df[df.split == args.split]
    df = df.reset_index(drop=True)

    print("Extracting features...")
    encoder = SSLEncoder(
        hf_repo=args.model,
        encoder_layer=args.layer_index,
        sr=args.sr,
        device=args.device,
    )

    data = {}
    for path in tqdm(df.audio_path.unique()):
        x, _ = librosa.load(path, sr=args.sr, mono=True)
        data[path] = encoder(x)

    # One batched call to the encoder's conv-stack accounting per column.
    df["_frame_min"] = encoder.time_to_frame(df["min"].to_numpy())
    df["_frame_max"] = encoder.time_to_frame(df["max"].to_numpy())

    df["feat"] = df.apply(functools.partial(_slice_feats, feats=data), axis=1)
    df = df.drop(columns=["_frame_min", "_frame_max"])
    df["feat"] = df["feat"].apply(functools.partial(_pool_feats, pool=args.pool))

    df.attrs["hf_repo"] = args.model
    df.attrs["encoder_layer"] = args.layer_index
    df.attrs["pool"] = args.pool
    df.attrs["sr"] = args.sr
    df.attrs["frame_shift"] = SSL_FRAME_SHIFT

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.output_path)


def main(argv=None):
    args = _get_args(argv)
    run(args)


if __name__ == "__main__":
    sys.exit(main())
