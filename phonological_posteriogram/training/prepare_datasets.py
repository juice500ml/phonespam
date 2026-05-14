"""Convert raw datasets (TIMIT, VoxAngeles) into a uniform per-phone CSV."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


TIMIT_TO_IPA = {
    # Stops
    "b": "b", "d": "d", "g": "ɡ", "p": "p",
    "t": "t", "k": "k", "dx": "ɾ", "q": "ʔ",
    # Affricates
    "jh": "d͡ʒ", "ch": "t͡ʃ",
    # Fricatives
    "s": "s", "sh": "ʃ", "z": "z", "zh": "ʒ",
    "f": "f", "th": "θ", "v": "v", "dh": "ð",
    # Nasals
    "m": "m", "n": "n", "ng": "ŋ", "em": "m̩",
    "en": "n̩", "eng": "ŋ̍", "nx": "ɾ̃",
    # Semivowels and glides
    "l": "l", "r": "ɹ", "w": "w", "y": "j",
    "hh": "h", "hv": "ɦ", "el": "l̩",
    # Vowels
    "iy": "i", "ih": "ɪ", "eh": "ɛ", "ae": "æ",
    "aa": "ɑ", "ah": "ʌ", "ao": "ɔ", "uh": "ʊ",
    "uw": "u", "ux": "ʉ", "er": "ɝ", "ax": "ə",
    "ix": "ɨ", "axr": "ɚ", "ax-h": "ə̯",
    # Diphthongs
    "ey": "eɪ", "aw": "aʊ", "ay": "aɪ", "oy": "ɔɪ", "ow": "oʊ",
    # Stop closures: dropped after being merged into the succeeding stop.
    "bcl": None, "dcl": None, "gcl": None,
    "pcl": None, "tcl": None, "kcl": None,
    # Non-speech: kept as the silence token "_" so the segmenter sees silence
    # frames during training.
    "pau": "_", "epi": "_", "h#": "_",
}
TIMIT_CLOSURE_OF = {
    "b": "bcl",
    "d": "dcl",
    "g": "gcl",
    "p": "pcl",
    "t": "tcl",
    "k": "kcl",
    "jh": "dcl",
    "ch": "tcl",
}


def _get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset")
    parser.add_argument(
        "--dataset_type", type=str, choices=["timit", "voxangeles"]
    )
    parser.add_argument("--output_path", type=Path, help="Output csv folder")
    return parser.parse_args(argv)


def _add_phone_context(df, n=5):
    for i in range(1, n + 1):
        df[f"l_{i}"] = pd.Series(dtype="object")
        df[f"r_{i}"] = pd.Series(dtype="object")

    for _, group in tqdm(df.groupby("audio_path")):
        group = group[~group.ipa.isna()].sort_values("min")

        for i in range(1, n + 1):
            if len(group) > i:
                df.loc[group.index[i:], f"l_{i}"] = group.ipa.iloc[:-i].to_numpy()
                df.loc[group.index[:-i], f"r_{i}"] = group.ipa.iloc[i:].to_numpy()

    for i in range(1, n + 1):
        df.loc[df["ipa"].isna(), f"l_{i}"] = np.nan
        df.loc[df["ipa"].isna(), f"r_{i}"] = np.nan

    return df


def _prepare_timit(timit_path: Path):
    """Read TIMIT directly off disk by globbing .WAV/.PHN file pairs.

    Adapted from https://github.com/juice500ml/phonetic-arithmetic/blob/main/prepare_datasets.py
    Avoids any dependency on the HuggingFace `datasets` library.

    `timit_path` is the TIMIT root directory containing TRAIN/ and TEST/.
    Each .WAV file has a sibling .PHN file with `start stop phone` lines
    (sample indices at 16 kHz).
    """
    rows = []
    for split in ("TRAIN", "TEST"):
        wav_paths = sorted(timit_path.glob(f"**/{split}/**/*.WAV"))
        for audio_path in tqdm(wav_paths, desc=f"TIMIT {split}"):
            audio_path_str = str(audio_path)
            phn_path = audio_path.with_suffix(".PHN")
            if not phn_path.exists():
                continue
            with open(phn_path) as f:
                for line in f:
                    start_str, stop_str, phn = line.strip().split()
                    start = int(start_str)
                    stop = int(stop_str)
                    ipa = TIMIT_TO_IPA[phn]

                    # Stop closures (bcl, dcl, ...) get merged into the
                    # succeeding stop by extending that stop's start time
                    # backwards to the closure's start.
                    if phn in TIMIT_CLOSURE_OF:
                        closure = TIMIT_CLOSURE_OF[phn]
                        if (
                            rows
                            and rows[-1]["timit_phn"] == closure
                            and rows[-1]["audio_path"] == audio_path_str
                        ):
                            start = int(rows[-1]["min"] * 16000)

                    rows.append(
                        {
                            "audio_path": audio_path_str,
                            "min": start / 16000,
                            "max": stop / 16000,
                            "timit_phn": phn,
                            "ipa": ipa,
                            "split": split.lower(),
                        }
                    )

    df = pd.DataFrame(rows)
    df = df[df.ipa.notna()].reset_index(drop=True)
    return _add_phone_context(df)


def _prepare_voxangeles(root_path: Path):
    import praatio.textgrid

    rows = []
    for path in tqdm((root_path / "data/audited_aligned").glob("**/*.TextGrid")):
        grid = praatio.textgrid.openTextgrid(path, includeEmptyIntervals=False)
        tier_name = next(
            x for x in grid.tierNames if x in ("phone", "phones", "Narrow")
        )
        for entry in grid.getTier(tier_name).entries:
            rows.append(
                {
                    "audio_path": str(path.with_suffix(".wav")),
                    "min": entry.start,
                    "max": entry.end,
                    "ipa": entry.label,
                    "split": "test",
                    "language": path.parent.name,
                }
            )
    return _add_phone_context(pd.DataFrame(rows))


def run(args):
    print(args)
    prep = {
        "timit": _prepare_timit,
        "voxangeles": _prepare_voxangeles,
    }[args.dataset_type]
    df = prep(args.dataset_path)

    os.makedirs(args.output_path, exist_ok=True)
    csv_path = args.output_path / f"{args.dataset_type}.csv"
    df.to_csv(str(csv_path), index=False)
    print("Stored to", csv_path)


def main(argv=None):
    args = _get_args(argv)
    run(args)


if __name__ == "__main__":
    sys.exit(main())
