"""Convert raw datasets (TIMIT, VoxAngeles) into a uniform per-phone CSV.

Parsing of the distributed datasets (TIMIT ``.phn`` files, VoxAngeles
``.TextGrid`` files) and the TIMIT stop-closure merge logic now live in the
standalone :mod:`phone_metrics` package, so this repository never reads those
formats directly. This module is a thin adapter: it loads utterances via
``phone_metrics`` and flattens them into this repo's CSV schema, which the
downstream steps (``extract_features``, ``evaluate``, ``tune``) read.

CSV columns:

* TIMIT:  ``audio_path, min, max, timit_phn, ipa, split, language``
* VoxAngeles: ``audio_path, min, max, ipa, split, language``

``ipa`` is empty for unmerged TIMIT closures in the ``timit-raw`` CSV (the
substrate for explicit closure/release modeling). ``min``/``max`` are segment
start/end in seconds. Paths are taken verbatim from ``--dataset_path``, so pass
a repo-relative path (e.g. ``data/TIMIT``) to get repo-relative ``audio_path``.

For TIMIT, both ``timit-merged.csv`` (closures folded into the release;
evaluation ground truth) and ``timit-raw.csv`` (every interval kept, closures
unlabeled; training substrate) are emitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from phone_metrics import load_timit, load_voxangeles

# The two TIMIT closure-handling variants emitted side by side.
TIMIT_MODES = ("raw", "merged")


def _get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset")
    parser.add_argument("--dataset_type", type=str, choices=["timit", "voxangeles"])
    parser.add_argument("--output_dir", type=Path, help="Output csv folder")
    return parser.parse_args(argv)


def _timit_dataframe(root: Path, merge_closures: bool) -> pd.DataFrame:
    rows = []
    for split in ("train", "test"):
        for utt in load_timit(root, split=split, merge_closures=merge_closures):
            for seg in utt.segments:
                rows.append(
                    {
                        "audio_path": utt.audio_path,
                        "min": seg.start,
                        "max": seg.end,
                        "timit_phn": seg.raw_label,
                        "ipa": seg.ipa_label,
                        "split": utt.split,
                        "language": utt.language,
                    }
                )
    df = pd.DataFrame(rows)
    # "merged" carries no unlabeled rows; "raw" keeps closures with ipa empty.
    if merge_closures:
        df = df[df.ipa.notna()]
    return df.reset_index(drop=True)


def _voxangeles_dataframe(root: Path) -> pd.DataFrame:
    rows = []
    for utt in load_voxangeles(root):
        for seg in utt.segments:
            rows.append(
                {
                    "audio_path": utt.audio_path,
                    "min": seg.start,
                    "max": seg.end,
                    "ipa": seg.ipa_label,
                    "split": utt.split,
                    "language": utt.language,
                }
            )
    return pd.DataFrame(rows).reset_index(drop=True)


def run(args):
    print(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset_type == "timit":
        for tag in TIMIT_MODES:
            df = _timit_dataframe(args.dataset_path, merge_closures=(tag == "merged"))
            csv_path = args.output_dir / f"{args.dataset_type}-{tag}.csv"
            df.to_csv(str(csv_path), index=False)
            print("Stored to", csv_path)
    else:
        df = _voxangeles_dataframe(args.dataset_path)
        csv_path = args.output_dir / f"{args.dataset_type}.csv"
        df.to_csv(str(csv_path), index=False)
        print("Stored to", csv_path)


def main(argv=None):
    args = _get_args(argv)
    run(args)


if __name__ == "__main__":
    main()
