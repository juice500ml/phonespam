"""Evaluate phone-classification metrics from a library artifact.

Loads a :class:`~phonological_posteriogram.phone_model.PhoneModel` (trained by
``training/train.py``), uses a fixed segmentation config, and reports, for both
datasets, the boundary R-value and the **oracle** + **full-pipeline**
phone-recognition metrics (PER, TER, PFER).

PER and PFER are silence-free; TER is the token error rate that scores silence
as an ordinary token. Scoring uses the external ``phone_metrics`` package. GT
segmentation/labels come from ``phone_metrics`` loaders, so labels are
canonicalized consistently.

Run::

    CACHE_DIR=.cache/eval uv run python scripts/evaluate_metrics.py \\
        --model trained/model.pt \\
        --timit_root data/TIMIT --vox_root data/voxangeles

Set the ``CACHE_DIR`` environment variable to memoize the expensive per-utterance
inference (the sigmoid posteriogram and the raw/snapped boundary times) to disk;
a second run with the same model and segmentation config reuses it and only
re-does the cheap recognition + scoring (e.g. when iterating on metrics).

Notes
-----
Both the oracle and pipeline paths recognize on the **sigmoid** posteriogram
(``project(act="sigmoid")``). The segmentation signals themselves still use the
raw projection internally (inside the library ``Segmenter``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import panphon
import panphon.distance
from phone_metrics import (
    PrecisionRecallMetric,
    load_timit,
    load_voxangeles,
    oracle_phone_accuracy,
    phone_error_rates,
    tokenize_ipa,
)
from tqdm import tqdm

from phonological_posteriogram.phone_model import PhoneModel
from phonological_posteriogram.recognizer import Recognizer, panphon_featmap

# The default segmentation configuration (grid.json[0], also the library
# Segmenter's DEFAULT_COMBINED_SIGNALS). Hardcoded so the script is
# self-contained and independent of the artifact's stored hparams.
DEFAULT_COMBINED_SIGNALS = [
    {"name": "frame_delta", "kwargs": {"offset": 3}, "shift": 2},
    {"name": "frame_delta", "kwargs": {"offset": 2}, "shift": 1},
    {"name": "frame_delta", "kwargs": {"offset": 1}, "shift": 1},
    {"name": "bwd_contrast", "kwargs": {"lookbehind": 2}, "shift": -1},
    {"name": "bwd_contrast", "kwargs": {"lookbehind": 3}, "shift": -1},
    {"name": "bwd_contrast", "kwargs": {"lookbehind": 1}, "shift": 0},
    {"name": "mel_svf", "kwargs": {"left": 2, "right": 1}, "shift": 0},
]


_LABEL = 16  # left-column width for aligned per-dataset output
_SILENCE = "_"


def _model_tag(model_arg) -> str:
    """Stable identity for the model artifact: resolved path + mtime + size when
    it lives on disk, so a re-trained/overwritten model misses the cache."""
    p = Path(model_arg)
    if p.exists():
        st = p.stat()
        return f"{p.resolve()}:{int(st.st_mtime)}:{st.st_size}"
    return str(model_arg)


class InferenceCache:
    """On-disk cache of the expensive per-utterance inference outputs.

    Only the SSL encoder + segmenter are costly; recognition and scoring are
    cheap. This caches their products — the sigmoid ``ipa`` posteriogram and the
    raw/snapped boundary times (seconds) — so the whole script can be re-run
    (different recognizers, vocab splits, metrics) without re-running inference.

    Entries are keyed by ``tag`` (model identity + segmentation config + frame
    geometry) and the utterance audio path, so a changed model or config
    transparently misses. Disabled (pass-through) when ``cache_dir`` is None.
    """

    def __init__(self, cache_dir, tag: str):
        self.dir = Path(cache_dir) / "pp-eval" if cache_dir else None
        self._prefix = hashlib.sha1(tag.encode()).hexdigest()[:16]
        self.hits = self.misses = 0
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, audio_path: str) -> Path:
        key = hashlib.sha1(f"{self._prefix}\x00{audio_path}".encode()).hexdigest()
        return self.dir / f"{key}.npz"

    def infer(self, model, utt, segmenter, frame_sec):
        """Return ``(post, raw_bt, pipe_bt)`` for ``utt``, from disk if cached."""
        path = self._path(utt.audio_path) if self.dir is not None else None
        if path is not None and path.exists():
            self.hits += 1
            with np.load(path) as z:
                return z["post"], z["raw_bt"], z["pipe_bt"]

        self.misses += 1
        wav = model.load_audio(utt.audio_path)
        feats = model.extract_features(wav)
        post = model.posteriogram.project(feats, view="ipa", act="sigmoid")
        raw_bt = np.asarray(segmenter.segment(feats, wav, snap_silence=False)) * frame_sec
        pipe_bt = np.asarray(segmenter.segment(feats, wav, snap_silence=True)) * frame_sec
        if path is not None:
            # Native dtypes preserved so cached and live runs score identically.
            np.savez_compressed(path, post=np.asarray(post), raw_bt=raw_bt, pipe_bt=pipe_bt)
        return post, raw_bt, pipe_bt


def _row(label, body):
    print(f"  {label:{_LABEL}s}{body}")


def _expand_label(label):
    """Split an IPA label into its component phones (silence passes through).

    Mirrors ``phone_metrics``' phone expansion: a diphthong ``aɪ`` becomes the
    two tokens ``a``, ``ɪ``; a tie-barred affricate stays one token; an
    unparseable label is kept whole so it still counts as a scorable phone.
    """
    if label == _SILENCE:
        return [label]
    toks = tokenize_ipa(label)
    return toks if toks else [label]


def _split_pfer_by_vocab(utts, pred_seqs, vocab, *, dist):
    """Per-segment PFER split by reference-phone membership in ``vocab``.

    With oracle boundaries the recognizer emits exactly one label per reference
    segment, so each non-silence segment is scored independently with panphon's
    feature edit distance and bucketed by whether its reference phone(s) appear
    in ``vocab`` (the TIMIT training vocabulary). A multi-phone reference segment
    counts as in-vocab only if every component phone is in ``vocab``.

    Returns ``(cost, total)`` dicts keyed by ``"in"`` / ``"oov"``; PFER for a
    bucket is ``cost[k] / total[k]``. Summing the buckets reproduces a
    per-segment PFER, which for oracle (1:1) alignment matches the
    utterance-level PFER up to panphon's cross-segment alignment slack.
    """
    cost = {"in": 0.0, "oov": 0.0}
    total = {"in": 0, "oov": 0}
    for utt, preds in zip(utts, pred_seqs):
        assert len(preds) == len(utt.segments)
        for seg, pred in zip(utt.segments, preds):
            if seg.ipa_label == _SILENCE:
                continue
            ref_toks = _expand_label(seg.ipa_label)
            pred_toks = [t for t in _expand_label(pred) if t != _SILENCE]
            bucket = "in" if all(t in vocab for t in ref_toks) else "oov"
            cost[bucket] += float(
                dist.feature_edit_distance("".join(pred_toks), "".join(ref_toks))
            )
            total[bucket] += len(ref_toks)
    return cost, total


def _silence_pr(utts, pred_seqs):
    """Per-segment precision/recall for the silence class under oracle boundaries.

    Each reference segment has exactly one predicted label, so silence (``"_"``)
    is a binary decision per segment. Precision is the share of predicted-silence
    segments that are truly silence; recall is the share of true-silence segments
    predicted as silence. Returns ``(precision, recall, tp, fp, fn)`` (micro).
    """
    tp = fp = fn = 0
    for utt, preds in zip(utts, pred_seqs):
        assert len(preds) == len(utt.segments)
        for seg, pred in zip(utt.segments, preds):
            ref_sil, pred_sil = seg.ipa_label == _SILENCE, pred == _SILENCE
            tp += ref_sil and pred_sil
            fp += pred_sil and not ref_sil
            fn += ref_sil and not pred_sil
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return precision, recall, tp, fp, fn


def _silence_body(utts, pred_seqs):
    p, r, tp, fp, fn = _silence_pr(utts, pred_seqs)
    return f"P={p:.3f}  R={r:.3f}  (tp={tp} fp={fp} fn={fn})"


def _segmentation_body(seg_raw, seg_snap):
    """Boundary R-value (raw + snapped); precision/recall from raw boundaries."""
    r, rs = seg_raw.compute(), seg_snap.compute()
    return (
        f"RV={r['rval']:.3f}  snapped={rs['rval']:.3f}  "
        f"P={r['precision']:.3f}  R={r['recall']:.3f}"
    )


def _rates_body(er):
    return f"PER={er.per:.4f}  TER={er.ter:.4f}  PFER={er.pfer:.4f}"


def _macro_body(er):
    return f"macro_lang  PER={er.macro_language_per:.4f}  PFER={er.macro_language_pfer:.4f}"


def evaluate_timit(model, utts, recognizer, segmenter, *, featnames, sr, frame_shift, cache):
    """Segmentation R-value + oracle and full-pipeline phone metrics."""
    frame_sec = frame_shift / sr
    panphon_rec = Recognizer(featnames=featnames)
    oracle_utts, oracle_preds = [], []
    oracle_seqs, panphon_seqs, pipeline_seqs, panphon_pipeline_seqs = [], [], [], []
    seg_raw = PrecisionRecallMetric(tolerance=0.02, mode="strict")
    seg_snap = PrecisionRecallMetric(tolerance=0.02, mode="strict")

    for utt in tqdm(utts, desc="TIMIT test", leave=False):
        post, raw_bt, pipe_bt = cache.infer(model, utt, segmenter, frame_sec)
        segs = utt.segments

        bt = [s.end for s in segs[:-1]]  # inter-segment boundary times (seconds)
        labs = recognizer.recognize(post, bt, sr=sr, frame_shift=frame_shift)
        panphon_labs = panphon_rec.recognize(post, bt, sr=sr, frame_shift=frame_shift)
        assert len(labs) == len(segs) == len(panphon_labs), (len(labs), len(segs))
        oracle_preds.extend(labs)
        oracle_seqs.append(list(labs))
        panphon_seqs.append(list(panphon_labs))
        oracle_utts.append(utt)

        seg_raw.update(utt.boundaries, raw_bt)
        seg_snap.update(utt.boundaries, pipe_bt)
        pipeline_seqs.append(
            recognizer.recognize(post, pipe_bt, sr=sr, frame_shift=frame_shift)
        )
        panphon_pipeline_seqs.append(
            panphon_rec.recognize(post, pipe_bt, sr=sr, frame_shift=frame_shift)
        )

    acc = oracle_phone_accuracy(oracle_utts, oracle_preds, label="ipa")
    er = phone_error_rates(oracle_utts, oracle_seqs, label="ipa")
    panphon_er = phone_error_rates(oracle_utts, panphon_seqs, label="ipa")
    pipe_er = phone_error_rates(oracle_utts, pipeline_seqs, label="ipa")
    panphon_pipe_er = phone_error_rates(oracle_utts, panphon_pipeline_seqs, label="ipa")

    print(f"\nTIMIT test  ({er.token_total} ref tokens)")
    _row("segmentation", _segmentation_body(seg_raw, seg_snap))
    _row("oracle acc", f"{acc.accuracy:.4f}  (n={acc.total} merged phones)")
    _row("oracle rates", _rates_body(er))
    _row("oracle PFER", f"{panphon_er.pfer:.4f}  (panphon-unrestricted)")
    _row("oracle silence", _silence_body(oracle_utts, panphon_seqs))
    _row("pipeline rates", _rates_body(pipe_er))
    _row("pipeline PFER", f"{panphon_pipe_er.pfer:.4f}  (panphon-unrestricted)")


def evaluate_vox(model, utts, segmenter, *, featnames, sr, frame_shift, timit_vocab, cache):
    """Segmentation R-value + VoxAngeles within-language oracle,
    panphon-unrestricted PFER, and full pipeline (per-language vocab).

    ``timit_vocab`` is the set of TIMIT training phones (the model's ``ipa`` view
    vocabulary); the oracle PFER is additionally reported split by whether each
    VoxAngeles reference phone is in-vocab (seen in TIMIT) or OOV."""
    frame_sec = frame_shift / sr
    ft = panphon.FeatureTable()

    # Per-language candidate vocab = representable non-silence phones attested in
    # that language's test set, plus silence; one recognizer per language.
    lang_phones: dict[str, set[str]] = {}
    for utt in utts:
        for seg in utt.segments:
            if seg.ipa_label != "_" and ft.seg_known(seg.ipa_label):
                lang_phones.setdefault(utt.language, set()).add(seg.ipa_label)
    lang_rec = {
        lg: Recognizer(featnames=featnames, featmap=panphon_featmap([*sorted(ps), "_"], featnames))
        for lg, ps in lang_phones.items()
    }
    panphon_rec = Recognizer(featnames=featnames)

    oracle_utts, oracle_preds = [], []
    oracle_seqs, panphon_seqs, pipeline_seqs, panphon_pipeline_seqs = [], [], [], []
    seg_raw = PrecisionRecallMetric(tolerance=0.02, mode="strict")
    seg_snap = PrecisionRecallMetric(tolerance=0.02, mode="strict")
    for utt in tqdm(utts, desc="VoxAngeles", leave=False):
        post, raw_bt, pipe_bt = cache.infer(model, utt, segmenter, frame_sec)
        segs = utt.segments
        rec = lang_rec[utt.language]

        bt = [s.end for s in segs[:-1]]
        labs = rec.recognize(post, bt, sr=sr, frame_shift=frame_shift)
        panphon_labs = panphon_rec.recognize(post, bt, sr=sr, frame_shift=frame_shift)
        assert len(labs) == len(segs) == len(panphon_labs)
        oracle_preds.extend(labs)
        oracle_seqs.append(list(labs))
        panphon_seqs.append(list(panphon_labs))
        oracle_utts.append(utt)

        seg_raw.update(utt.boundaries, raw_bt)
        seg_snap.update(utt.boundaries, pipe_bt)
        pipeline_seqs.append(
            rec.recognize(post, pipe_bt, sr=sr, frame_shift=frame_shift)
        )
        panphon_pipeline_seqs.append(
            panphon_rec.recognize(post, pipe_bt, sr=sr, frame_shift=frame_shift)
        )

    acc = oracle_phone_accuracy(oracle_utts, oracle_preds, label="ipa")
    er = phone_error_rates(oracle_utts, oracle_seqs, label="ipa")
    panphon_er = phone_error_rates(oracle_utts, panphon_seqs, label="ipa")
    pipe_er = phone_error_rates(oracle_utts, pipeline_seqs, label="ipa")
    panphon_pipe_er = phone_error_rates(oracle_utts, panphon_pipeline_seqs, label="ipa")
    n_langs = len({u.language for u in oracle_utts})

    # Oracle PFER split by whether the reference phone appears in TIMIT, scored
    # per-segment on the panphon-unrestricted predictions (the "oracle PFER" row).
    dist = panphon.distance.Distance()
    cost, total = _split_pfer_by_vocab(oracle_utts, panphon_seqs, timit_vocab, dist=dist)
    pfer_iv = cost["in"] / total["in"] if total["in"] else float("nan")
    pfer_oov = cost["oov"] / total["oov"] if total["oov"] else float("nan")
    n_oov_types = len(
        {
            t
            for u in oracle_utts
            for s in u.segments
            if s.ipa_label != _SILENCE
            for t in _expand_label(s.ipa_label)
            if t not in timit_vocab
        }
    )

    print(f"\nVoxAngeles  ({n_langs} languages, {er.token_total} ref tokens)")
    _row("segmentation", _segmentation_body(seg_raw, seg_snap))
    _row("oracle acc", f"micro={acc.accuracy:.4f}  macro_lang={acc.macro_language:.4f}  (n={acc.total})")
    _row("oracle rates", _rates_body(er))
    _row("", _macro_body(er))
    _row("oracle PFER", f"{panphon_er.pfer:.4f}  (panphon-unrestricted)")
    _row("  in-vocab", f"{pfer_iv:.4f}  (n={total['in']} TIMIT phones)")
    _row("  OOV", f"{pfer_oov:.4f}  (n={total['oov']} phones, {n_oov_types} OOV types)")
    _row("oracle silence", _silence_body(oracle_utts, panphon_seqs))
    _row("pipeline rates", _rates_body(pipe_er))
    _row("", _macro_body(pipe_er))
    _row("pipeline PFER", f"{panphon_pipe_er.pfer:.4f}  (panphon-unrestricted)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="PhoneModel artifact path or dir.")
    parser.add_argument("--timit_root", type=Path, default=Path("data/TIMIT"))
    parser.add_argument("--vox_root", type=Path, default=Path("data/voxangeles"))
    parser.add_argument("--device", default="cuda", help="torch device for the SSL encoder.")
    parser.add_argument("--skip_timit", action="store_true")
    parser.add_argument("--skip_vox", action="store_true")
    args = parser.parse_args(argv)

    # Set the CACHE_DIR env var to memoize per-utterance inference (posteriogram
    # + boundaries) there, so metrics can be re-derived without re-running the
    # encoder/segmenter. Keyed by model + segmentation config; stale entries miss.
    cache_dir = os.environ.get("CACHE_DIR") or None

    model = PhoneModel.from_pretrained(args.model, device=args.device)
    sr = model.net_spec["sr"]
    frame_shift = model.encoder.stride_size
    featnames = model.posteriogram.featnames
    segmenter = model.segmenter({"combined_signals": DEFAULT_COMBINED_SIGNALS})
    # The model's ipa-view vocabulary is exactly the TIMIT training phones; a
    # VoxAngeles phone is OOV iff it (or a component of it) is absent here.
    timit_vocab = {p for p in model.posteriogram.views["ipa"].featmap if p != "_"}

    # Cache identity: model artifact + segmentation config + frame geometry. Any
    # change to these invalidates entries (the posteriogram/boundaries depend on
    # all of them); the audio path distinguishes utterances within a tag.
    cache_tag = _model_tag(args.model) + "\x00" + json.dumps(
        {"signals": DEFAULT_COMBINED_SIGNALS, "sr": sr, "frame_shift": frame_shift},
        sort_keys=True,
    )
    cache = InferenceCache(cache_dir, cache_tag)

    if not args.skip_timit:
        timit_rec = Recognizer(
            featnames=featnames, featmap=model.posteriogram.views["ipa"].featmap
        )
        evaluate_timit(
            model,
            load_timit(args.timit_root, split="test"),
            timit_rec,
            segmenter,
            featnames=featnames,
            sr=sr,
            frame_shift=frame_shift,
            cache=cache,
        )

    if not args.skip_vox:
        evaluate_vox(
            model,
            load_voxangeles(args.vox_root),
            segmenter,
            featnames=featnames,
            sr=sr,
            frame_shift=frame_shift,
            timit_vocab=timit_vocab,
            cache=cache,
        )

    if cache.dir is not None:
        print(f"\ninference cache: {cache.hits} hits, {cache.misses} misses ({cache.dir})")


if __name__ == "__main__":
    main()
