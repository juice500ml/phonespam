"""Command-line interface for phonological-posteriogram inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .pretrained import PhonologicalPosteriogram


def _load_audio(path: Path, sr: int) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a pretrained phonological-posteriogram model on an audio file.",
    )
    parser.add_argument(
        "audio", type=Path, help="Path to an audio file (any librosa-readable format)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace repo id or local path/dir containing the model artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write boundary times (in seconds) as a text file.",
    )
    parser.add_argument(
        "--posteriogram",
        type=Path,
        help="Optional path to write the per-frame posteriogram as a .npy file.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cpu, cuda:0, ...).")
    parser.add_argument(
        "--no-snap-silence",
        action="store_true",
        help="Disable silence-snapping of boundary predictions.",
    )
    args = parser.parse_args(argv)

    model = PhonologicalPosteriogram.from_pretrained(args.model, device=args.device)
    waveform = _load_audio(args.audio, sr=model.net_spec["sr"])

    boundary_seconds = model.segment_seconds(
        waveform, snap_silence=not args.no_snap_silence
    )

    if args.output is not None:
        np.savetxt(args.output, boundary_seconds, fmt="%.4f")
    else:
        for t in boundary_seconds:
            print(f"{t:.4f}")

    if args.posteriogram is not None:
        post = model.posteriogram(waveform)
        np.save(args.posteriogram, post)


if __name__ == "__main__":
    sys.exit(main())
