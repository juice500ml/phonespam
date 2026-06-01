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
    "uw": "u", "ux": "ʉ", "er": "ɜ˞", "ax": "ə",
    "ix": "ɨ", "axr": "ə˞", "ax-h": "ə̥",
    # Diphthongs
    "ey": "eɪ", "aw": "aʊ", "ay": "aɪ", "oy": "ɔɪ", "ow": "oʊ",
    # Stop closures: will be dropped after being merged into the succeeding stop.
    "bcl": "b", "dcl": "d", "gcl": "ɡ",
    "pcl": "p", "tcl": "t", "kcl": "k",
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
# The closure tokens themselves (the values of TIMIT_CLOSURE_OF). Used when the
# closure->release merge is disabled, to drop the standalone closure rows.
TIMIT_CLOSURE_PHNS = frozenset(TIMIT_CLOSURE_OF.values())

# Diphthongs are kept as a single row in the dataframe (so the segmenter can
# pool features over the full glide), but they expand to two phones when
# computing the *context* of neighboring rows so that l_n/r_n look up known
# panphon segments. E.g. for [p, eɪ, t]: l_1 of t is ɪ, l_2 is e, l_3 is p.
DIPHTHONG_PARTS = {
    "eɪ": ("e", "ɪ"),
    "aʊ": ("a", "ʊ"),
    "aɪ": ("a", "ɪ"),
    "ɔɪ": ("ɔ", "ɪ"),
    "oʊ": ("o", "ʊ"),
}


def _get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset")
    parser.add_argument(
        "--dataset_type", type=str, choices=["timit", "voxangeles"]
    )
    parser.add_argument("--output_dir", type=Path, help="Output csv folder")
    return parser.parse_args(argv)


def _add_phone_context(df, n=5):
    for i in range(1, n + 1):
        df[f"l_{i}"] = pd.Series(dtype="object")
        df[f"r_{i}"] = pd.Series(dtype="object")

    for _, group in tqdm(df.groupby("audio_path")):
        group = group[~group.ipa.isna()].sort_values("min")
        if len(group) == 0:
            continue

        # Build a per-utterance "expanded" label sequence where each
        # diphthong row contributes both of its component phones, and record
        # the [start, end] span each original row occupies in that sequence.
        expanded = []
        orig_indices = []
        starts = []
        ends = []
        for orig_idx, ipa in zip(group.index, group.ipa):
            start = len(expanded)
            expanded.extend(DIPHTHONG_PARTS.get(ipa, (ipa,)))
            orig_indices.append(orig_idx)
            starts.append(start)
            ends.append(len(expanded) - 1)

        expanded_arr = np.asarray(expanded, dtype=object)
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        orig_indices = np.asarray(orig_indices)
        E = len(expanded_arr)

        for i in range(1, n + 1):
            left_pos = starts - i
            right_pos = ends + i
            l_vals = expanded_arr[np.clip(left_pos, 0, E - 1)].copy()
            r_vals = expanded_arr[np.clip(right_pos, 0, E - 1)].copy()
            l_vals[left_pos < 0] = np.nan
            r_vals[right_pos >= E] = np.nan
            df.loc[orig_indices, f"l_{i}"] = l_vals
            df.loc[orig_indices, f"r_{i}"] = r_vals

    for i in range(1, n + 1):
        df.loc[df["ipa"].isna(), f"l_{i}"] = np.nan
        df.loc[df["ipa"].isna(), f"r_{i}"] = np.nan

    return df


def _merge_stop_closures(utt_rows):
    """Fold each stop closure into the stop/affricate release after it.

    TIMIT writes most stops and affricates as a *closure* interval (``bcl``,
    ``dcl``, ...) immediately followed by a *release* (``b``, ``jh``, ...).
    When that pair occurs, we merge them into a single segment spanning
    ``[closure.min, release.max]``, labeled as the release, and drop the
    closure row.

    A closure that stands on its own — no matching release immediately after
    it — is kept and falls back to the bare stop: ``TIMIT_TO_IPA`` already
    maps e.g. ``"bcl" -> "b"``, so a lone closure surfaces as the stop, never
    as a distinct closure symbol.

    ``utt_rows`` must be time-ordered and belong to a single utterance.
    """
    merged = []
    for row in utt_rows:
        phn = row["timit_phn"]
        if (
            merged
            and phn in TIMIT_CLOSURE_OF
            and merged[-1]["timit_phn"] == TIMIT_CLOSURE_OF[phn]
        ):
            # Previous row is this release's closure: extend the release back
            # over the closure's span and replace the closure row with it.
            merged[-1] = {**row, "min": merged[-1]["min"]}
        else:
            merged.append(row)
    return merged


def _prepare_timit(timit_path: Path, merge_closures: bool = True):
    """Read TIMIT directly off disk by globbing .WAV/.PHN file pairs.

    Adapted from https://github.com/juice500ml/phonetic-arithmetic/blob/main/prepare_datasets.py
    Avoids any dependency on the HuggingFace `datasets` library.

    `timit_path` is the TIMIT root directory containing TRAIN/ and TEST/.
    Each .WAV file has a sibling .PHN file with `start stop phone` lines
    (sample indices at 16 kHz). Each utterance is parsed in full, then its
    stop closures are merged into the following release in a separate
    utterance-wise pass (see :func:`_merge_stop_closures`).
    """
    # Different TIMIT distributions use different casing (LDC ships uppercase
    # TRAIN/TEST and .WAV/.PHN; other copies are lowercase), so match case
    # insensitively. Note glob(case_sensitive=False) requires Python 3.12+.
    rows = []
    for split in ("TRAIN", "TEST"):
        wav_paths = sorted(
            timit_path.glob(f"**/{split}/**/*.WAV", case_sensitive=False)
        )
        for audio_path in tqdm(wav_paths, desc=f"TIMIT {split}"):
            phn_path = next(
                audio_path.parent.glob(
                    f"{audio_path.stem}.PHN", case_sensitive=False
                ),
                None,
            )
            if phn_path is None:
                continue
            utt_rows = []
            with open(phn_path) as f:
                for line in f:
                    start_str, stop_str, phn = line.strip().split()
                    ipa = TIMIT_TO_IPA[phn]
                    if not merge_closures and phn in TIMIT_CLOSURE_PHNS:
                        # Leave the closure unmerged and unlabeled so it is
                        # dropped before fitting; the release then keeps its
                        # own [start, end] span instead of absorbing the
                        # closure.
                        ipa = np.nan
                    utt_rows.append(
                        {
                            "audio_path": str(audio_path),
                            "min": int(start_str) / 16000,
                            "max": int(stop_str) / 16000,
                            "timit_phn": phn,
                            "ipa": ipa,
                            "split": split.lower(),
                            "language": "eng",
                        }
                    )
            if merge_closures:
                utt_rows = _merge_stop_closures(utt_rows)
            rows.extend(utt_rows)

    df = pd.DataFrame(rows)
    df = df[df.ipa.notna()].reset_index(drop=True)
    return _add_phone_context(df)


def _prepare_voxangeles(root_path: Path):
    import praatio.textgrid

    rows = []
    for path in tqdm((root_path / "data/audited_aligned").glob("**/*.TextGrid")):
        # Keep empty intervals so the phone rows tile the full audio
        # (leading/trailing silence and internal gaps included).
        grid = praatio.textgrid.openTextgrid(path, includeEmptyIntervals=True)
        tier_name = next(
            x for x in grid.tierNames if x in ("phone", "phones", "Narrow")
        )
        for entry in grid.getTier(tier_name).entries:
            label = (entry.label or "").strip()
            rows.append(
                {
                    "audio_path": str(path.with_suffix(".wav")),
                    "min": entry.start,
                    "max": entry.end,
                    "ipa": label if label else "_",
                    "split": "test",
                    "language": path.parent.name,
                }
            )
    return _add_phone_context(pd.DataFrame(rows))


def run(args):
    print(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset_type == "timit":
        # TIMIT stop/affricate closures can either be folded into the following
        # release (``-merged``, the standard evaluation-boundary convention,
        # matching prior work) or dropped so each release keeps just its own span
        # (``-dropped-closures``, cleaner release-only features for fitting). The
        # two give different segment boundaries, so we always emit both and let
        # downstream steps pick: ``-merged`` for evaluation GT, ``-dropped-closures``
        # for training.
        for tag, merge_closures in (("merged", True), ("dropped-closures", False)):
            df = _prepare_timit(args.dataset_path, merge_closures=merge_closures)
            csv_path = args.output_dir / f"{args.dataset_type}-{tag}.csv"
            df.to_csv(str(csv_path), index=False)
            print("Stored to", csv_path)
    else:
        df = _prepare_voxangeles(args.dataset_path)
        csv_path = args.output_dir / f"{args.dataset_type}.csv"
        df.to_csv(str(csv_path), index=False)
        print("Stored to", csv_path)


def main(argv=None):
    args = _get_args(argv)
    run(args)


if __name__ == "__main__":
    sys.exit(main())
