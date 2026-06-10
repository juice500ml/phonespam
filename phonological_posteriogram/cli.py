"""Command-line interface for phonological-posteriogram inference.

Loads a pretrained :class:`PhoneModel` and runs end-to-end phone
recognition on one audio file, printing one ``start_sec end_sec label``
line per predicted phone segment.

Constrain the output vocabulary with ``--vocab phone1,phone2,...``. If the
vocab should come from Phoible, resolve it with
``phonological_posteriogram.phoible`` and pass the resulting phones here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .phone_model import PhoneModel


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run a pretrained phonological-posteriogram model on an audio "
            "file and print per-segment phone predictions."
        ),
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to an audio file (any librosa-readable format).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace repo id or local path/dir containing the model artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional path to write the per-segment predictions as a "
            "tab-separated file (start_sec, end_sec, label)."
        ),
    )
    parser.add_argument(
        "--posteriogram",
        type=Path,
        help="Optional path to write the per-frame posteriogram as a .npy file.",
    )
    parser.add_argument(
        "--vocab",
        default=None,
        help=(
            "Comma-separated list of phones to constrain the recognizer's "
            "output vocabulary (e.g. 'p,t,k,a,i,u')."
        ),
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cpu, cuda:0, ...).")
    args = parser.parse_args(argv)

    model = PhoneModel.from_pretrained(args.model, device=args.device)

    vocab = [p for p in args.vocab.split(",") if p] if args.vocab is not None else None

    units = model.recognize(args.audio, vocab=vocab)

    lines = [f"{u.start:.4f}\t{u.end:.4f}\t{u.label}" for u in units]
    if args.output is not None:
        args.output.write_text("\n".join(lines) + ("\n" if lines else ""))
    else:
        for line in lines:
            print(line)

    if args.posteriogram is not None:
        waveform = model.load_audio(args.audio)
        np.save(args.posteriogram, model.compute_posteriogram(waveform))


if __name__ == "__main__":
    sys.exit(main())
