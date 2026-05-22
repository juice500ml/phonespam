"""Command-line interface for phonological-posteriogram inference.

Loads a pretrained :class:`PhoneModel` and runs end-to-end phone
recognition on one audio file, printing one ``start_sec end_sec label``
line per predicted phone segment.

Constrain the output vocabulary at most one of three ways:

- ``--vocab phone1,phone2,...`` — explicit phone list.
- ``--lang Korean`` / ``--phoible_id 423`` — pick a Phoible inventory.
  Add ``--phoneme`` to use that inventory's abstract Phoneme set rather
  than its surface Allophones.
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
            "output vocabulary (e.g. 'p,t,k,a,i,u'). Mutually exclusive with "
            "--lang / --phoible_id."
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
            "Use the inventory's abstract Phoneme set rather than the "
            "surface Allophones (requires --lang or --phoible_id)."
        ),
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device (cpu, cuda:0, ...)."
    )
    args = parser.parse_args(argv)

    model = PhoneModel.from_pretrained(args.model, device=args.device)

    vocab = (
        [p for p in args.vocab.split(",") if p] if args.vocab is not None else None
    )

    units = model.recognize(
        args.audio,
        lang=args.lang,
        phoible_id=args.phoible_id,
        phoneme=args.phoneme,
        vocab=vocab,
    )

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
