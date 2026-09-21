"""Boundary error analysis: where does the segmenter delete and insert boundaries?

Scores the snapped predicted boundaries with the same one-to-one matching as
``phone_metrics.PrecisionRecallMetric`` (strict, 20 ms), then attributes each
error to its phonetic context:

* a **deletion** (unmatched ground-truth boundary) to the phones on either
  side of it: a silence edge, a vowel-approximant (either order) or
  vowel-vowel transition, or any other transition;
* an **insertion** (unmatched predicted boundary) to the broad class of the
  ground-truth segment it falls inside, alongside each class's split rate
  (share of its segments containing one).

On TIMIT, insertions inside stops are also checked against the unmerged
closure/release annotation: whether silence snapping added them, whether they
fall inside the closure, and whether they sit at the release.

Finally, it estimates how much of VoxAngeles's lower recall is explained by its
mix of boundary contexts, and how much of its lower precision by its extra
splitting of non-silence segments (see :func:`print_gap_attribution`).

Run::

    CACHE_DIR=.cache/eval python scripts/boundary_errors.py \\
        --model exp/wavlm-24-phonemodel \\
        --timit_root ../data/TIMIT --vox_root ../data/voxangeles

Inference is read from the ``evaluate_metrics.py`` cache, so pass the same
``--model`` / data-root spellings used there (the cache is keyed on them).
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from functools import cache
from itertools import pairwise
from pathlib import Path

import numpy as np
import panphon
from evaluate_metrics import DEFAULT_COMBINED_SIGNALS, InferenceCache, inference_cache_tag

try:
    from phone_metrics import PrecisionRecallMetric, load_timit, load_voxangeles, tokenize_ipa
    from phone_metrics.timit import TIMIT_CLOSURE_OF, timit_segments
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError(
        "phone-metrics is required for boundary error analysis:\n    pip install 'phonespam[train]'"
    ) from e
from tqdm import tqdm

from phonespam.phone_model import PhoneModel
from phonespam.segmenter import S3M_FRAME_SHIFT

TOLERANCE = 0.02  # boundary hit tolerance (s)
_SILENCE = "_"
_EPS = 1e-9

CLASSES = (
    "silence",
    "vowel",
    "approximant",
    "nasal",
    "tap/trill",
    "stop",
    "fricative/affricate",
    "other",
)
CONTEXT_GROUPS = ("silence edge", "vowel-approximant or vowel-vowel", "other")

# PanPhon marks glottals as [+son] (and h/ɦ as [+cons]); ʡ/ʜ/ɚ are not in
# PanPhon at all; the lateral flap ɺ would read as a lateral approximant. Keyed
# on the base character, so diacritic variants follow.
_BASE_OVERRIDES = {
    "ʔ": "stop",
    "ʡ": "stop",
    "h": "fricative/affricate",
    "ɦ": "fricative/affricate",
    "ʜ": "fricative/affricate",
    "ʢ": "fricative/affricate",
    "ɺ": "tap/trill",
    "ɚ": "vowel",
    "ɝ": "vowel",
}

_FT = panphon.FeatureTable()


def _token_class(tok: str) -> str:
    if tok[0] in _BASE_OVERRIDES:
        return _BASE_OVERRIDES[tok[0]]
    fts = _FT.word_fts(tok)
    if not fts:
        return "other"
    v = fts[0]
    if v["syl"] > 0 and v["cons"] < 0:
        return "vowel"
    if v["son"] > 0:
        if v["nas"] > 0 and v["cons"] > 0:
            return "nasal"
        # PanPhon leaves delayed release unspecified (0) for taps and trills,
        # but also for laterals.
        if v["cons"] > 0 and v["delrel"] == 0 and v["lat"] < 0:
            return "tap/trill"
        return "approximant"
    if v["cont"] < 0 and v["delrel"] <= 0:
        return "stop"
    return "fricative/affricate"


@cache
def phone_class(label: str | None) -> str:
    """Broad class of an IPA segment label (one of :data:`CLASSES`).

    A multi-phone label (diphthong ``aɪ``, untied ``ts``) is a vowel if all its
    components are; a stop+fricative pair is an affricate; otherwise it takes
    the class of its first component.
    """
    if label is None or label == _SILENCE:
        return "silence"
    toks = tokenize_ipa(label)
    if not toks:
        return _BASE_OVERRIDES.get(label[0], "other")
    classes = [_token_class(t) for t in toks]
    if all(c == "vowel" for c in classes):
        return "vowel"
    if classes == ["stop", "fricative/affricate"]:
        return "fricative/affricate"
    return classes[0]


def context_group(prev_label: str | None, next_label: str | None) -> str:
    """Coarse context of the boundary between two segments (:data:`CONTEXT_GROUPS`)."""
    classes = {phone_class(prev_label), phone_class(next_label)}
    if "silence" in classes:
        return "silence edge"
    if "vowel" in classes and classes <= {"vowel", "approximant"}:
        return "vowel-approximant or vowel-vowel"
    return "other"


@cache
def _voiceless(label: str) -> bool:
    fts = _FT.word_fts(label)
    return bool(fts) and fts[0]["voi"] < 0


def _timit_closures(wav_path) -> list[tuple[float, float]]:
    """``(closure_start, release_start)`` for each TIMIT closure + release pair.

    Read from the unmerged ``.phn``; with merged closures (the scoring
    convention) each pair spans one stop/affricate segment.
    """
    wav = Path(wav_path)
    phn = next(p for p in (wav.with_suffix(".PHN"), wav.with_suffix(".phn")) if p.exists())
    segs = timit_segments(phn, merge_closures=False)
    return [
        (a.start, b.start)
        for a, b in pairwise(segs)
        if TIMIT_CLOSURE_OF.get(b.raw_label) == a.raw_label
    ]


def _matched(targets, queries):
    """Mask over ``queries``: matched by ``PrecisionRecallMetric.get_counts``.

    Mirrors its strict-mode greedy: queries in order, each taking the
    lowest-index still-unused target within the tolerance.
    """
    targets = np.asarray(targets, dtype=float)
    used = np.zeros(len(targets), dtype=bool)
    hit = np.zeros(len(queries), dtype=bool)
    for i, q in enumerate(queries):
        for c in np.flatnonzero(np.abs(targets - q) <= TOLERANCE + _EPS):
            if not used[c]:
                used[c] = hit[i] = True
                break
    return hit


def _boundary_contexts(segments):
    """``(prev_label, next_label)`` for each boundary of ``boundary_secs``.

    Same order as ``Utterance.boundaries``: outer silence stripped, inner
    segment starts, then the final inner end. An utterance edge with no
    silence segment counts as silence.
    """
    lo, hi = 0, len(segments)
    while lo < hi and segments[lo].ipa_label == _SILENCE:
        lo += 1
    while hi > lo and segments[hi - 1].ipa_label == _SILENCE:
        hi -= 1
    if lo == hi:
        return []
    ctx = [
        (segments[k - 1].ipa_label if k > 0 else _SILENCE, segments[k].ipa_label)
        for k in range(lo, hi)
    ]
    ctx.append(
        (segments[hi - 1].ipa_label, segments[hi].ipa_label if hi < len(segments) else _SILENCE)
    )
    return ctx


class BoundaryErrors:
    """Accumulates context-attributed boundary errors for one dataset."""

    def __init__(self):
        self.metric = PrecisionRecallMetric(tolerance=TOLERANCE, mode="strict")
        self.n_gt = self.n_pred = self.gt_hits = self.pred_hits = 0
        self.boundaries = Counter()  # context group -> ground-truth boundaries
        self.deletions = Counter()  # context group -> deleted boundaries
        self.insertions = Counter()  # segment class -> inserted boundaries
        self.segments = Counter()  # segment class -> ground-truth segments
        self.split = Counter()  # segment class -> segments with >= 1 insertion
        self.stops = Counter()  # closure check of insertions inside stops

    def add(self, utt, pred, raw, closures=None):
        """Score one utterance's snapped boundaries ``pred`` (seconds).

        ``raw`` holds the same utterance's boundaries before silence snapping;
        ``closures`` its TIMIT ``(closure_start, release_start)`` pairs, for
        the stop-closure check.
        """
        gt = utt.boundaries
        pred = np.asarray(pred, dtype=float)
        self.metric.update(gt, pred)
        gt_hit = _matched(pred, gt)  # recall side
        pred_hit = _matched(gt, pred)  # precision side
        self.n_gt += len(gt)
        self.n_pred += len(pred)
        self.gt_hits += int(gt_hit.sum())
        self.pred_hits += int(pred_hit.sum())

        for (prev, nxt), hit in zip(_boundary_contexts(utt.segments), gt_hit, strict=True):
            group = context_group(prev, nxt)
            self.boundaries[group] += 1
            self.deletions[group] += int(not hit)

        segs = utt.segments
        starts = np.array([s.start for s in segs])
        inserted = np.zeros(len(segs), dtype=int)
        for t in pred[~pred_hit]:
            j = int(np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(segs) - 1))
            inserted[j] += 1
            if closures is not None and phone_class(segs[j].ipa_label) == "stop":
                self._add_stop_insertion(t, segs[j], np.asarray(raw, dtype=float), closures)
        for seg, n in zip(segs, inserted, strict=True):
            cls = phone_class(seg.ipa_label)
            self.segments[cls] += 1
            self.split[cls] += int(n > 0)
            self.insertions[cls] += int(n)

    def _add_stop_insertion(self, t, seg, raw, closures):
        c = self.stops
        c["n"] += 1
        c["snap_added"] += int(not np.any(np.abs(raw - t) < 1e-6))
        # A merged stop segment starts at its closure.
        release = next((r for start, r in closures if abs(start - seg.start) < 1e-6), None)
        if release is None:
            return
        c["at_release"] += int(abs(t - release) <= TOLERANCE + _EPS)
        if t < release:
            c["in_closure"] += 1
            c["in_voiceless_closure"] += int(_voiceless(seg.ipa_label))

    def report(self, title):
        m = self.metric.compute()
        p, r = self.pred_hits / self.n_pred, self.gt_hits / self.n_gt
        # The attribution below is only valid if it uses the metric's matching.
        assert np.isclose(p, m["precision"]) and np.isclose(r, m["recall"]), (p, r, m)
        print(f"\n{title}\n  P={p:.0%}  R={r:.0%}  RV={100 * m['rval']:.1f}")

        n_del = sum(self.deletions.values())
        print(
            f"  {'deletions by context':36s}{'% of boundaries':>16s}"
            f"{'miss rate':>12s}{'% of deletions':>16s}"
        )
        for g in CONTEXT_GROUPS:
            rate = self.deletions[g] / max(self.boundaries[g], 1)
            print(
                f"    {g:34s}{self.boundaries[g] / self.n_gt:16.0%}"
                f"{rate:12.0%}{self.deletions[g] / n_del:16.0%}"
            )

        if self.insertions:
            n_ins, n_seg = sum(self.insertions.values()), sum(self.segments.values())
            print(
                f"  {'insertions by segment class':36s}{'% of segments':>16s}"
                f"{'split rate':>12s}{'% of insertions':>16s}"
            )
            for k in CLASSES:
                if self.segments[k]:
                    rate = self.split[k] / self.segments[k]
                    print(
                        f"    {k:34s}{self.segments[k] / n_seg:16.0%}"
                        f"{rate:12.0%}{self.insertions[k] / n_ins:16.0%}"
                    )

        c = self.stops
        if c["n"]:
            print(f"  insertions inside stops (n={c['n']}), vs. closure annotation")
            print(f"    {'added by silence snapping':44s}{c['snap_added'] / c['n']:.0%}")
            print(f"    {'inside the closure':44s}{c['in_closure'] / c['n']:.0%}")
            voiceless = c["in_voiceless_closure"] / max(c["in_closure"], 1)
            print(f"      {'of which in a voiceless stop':42s}{voiceless:.0%}")
            release = f"within {TOLERANCE * 1000:.0f} ms of the release"
            print(f"    {release:44s}{c['at_release'] / c['n']:.0%}")


def print_gap_attribution(timit, vox):
    """How much of VoxAngeles's lower recall and precision vs. TIMIT is explained.

    Recall: VoxAngeles's mix of boundary contexts scored at TIMIT's per-context
    miss rates, i.e. what its recall would be if only the mix differed.
    Precision: VoxAngeles's non-silence segments given TIMIT's per-class
    insertions per segment (silence insertions and hits kept as observed),
    i.e. what its precision would be without its extra non-silence splitting.
    Each is reported as the share of the TIMIT-VoxAngeles gap it closes.
    """
    r_timit, r_vox = timit.gt_hits / timit.n_gt, vox.gt_hits / vox.n_gt
    p_timit, p_vox = timit.pred_hits / timit.n_pred, vox.pred_hits / vox.n_pred

    hits = sum(
        vox.boundaries[g] * (1 - timit.deletions[g] / max(timit.boundaries[g], 1))
        for g in CONTEXT_GROUPS
    )
    r_mix = hits / vox.n_gt

    insertions = sum(
        vox.segments[k] * timit.insertions[k] / timit.segments[k]
        if k != "silence" and timit.segments[k]
        else vox.insertions[k]
        for k in vox.segments
    )
    p_split = vox.pred_hits / (vox.pred_hits + insertions)

    print("\nVoxAngeles vs. TIMIT")
    print(
        f"  recall at TIMIT per-context miss rates           R={r_mix:.0%}  "
        f"(VoxAngeles {r_vox:.0%}, TIMIT {r_timit:.0%}): "
        f"boundary mix explains {(r_timit - r_mix) / (r_timit - r_vox):.0%} of the gap"
    )
    print(
        f"  precision at TIMIT non-silence insertion rates   P={p_split:.0%}  "
        f"(VoxAngeles {p_vox:.0%}, TIMIT {p_timit:.0%}): "
        f"non-silence splitting explains {(p_split - p_vox) / (p_timit - p_vox):.0%} of the gap"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="PhoneModel artifact path or dir.")
    parser.add_argument("--timit_root", type=Path, default=Path("data/TIMIT"))
    parser.add_argument("--vox_root", type=Path, default=Path("data/voxangeles"))
    parser.add_argument(
        "--device", default="cuda", help="torch device (only used on cache misses)."
    )
    args = parser.parse_args(argv)

    model = PhoneModel.from_pretrained(args.model, device=args.device)
    sr = model.net_spec["sr"]
    # model.encoder.stride_size would load the SSL encoder; every wav2vec2-family
    # encoder has this stride, and cache hits never need the encoder.
    frame_shift = S3M_FRAME_SHIFT
    signals = DEFAULT_COMBINED_SIGNALS
    segmenter = model.segmenter({"combined_signals": signals})
    cache_dir = os.environ.get("CACHE_DIR") or None
    cache = InferenceCache(cache_dir, inference_cache_tag(args.model, signals, sr, frame_shift))

    datasets = (
        ("TIMIT test", load_timit(args.timit_root, split="test"), True),
        ("VoxAngeles", load_voxangeles(args.vox_root), False),
    )
    results = []
    for title, utts, is_timit in datasets:
        errs = BoundaryErrors()
        for utt in tqdm(utts, desc=title, leave=False):
            _post, raw_bt, pipe_bt = cache.infer(model, utt, segmenter, frame_shift / sr)
            closures = _timit_closures(utt.audio_path) if is_timit else None
            errs.add(utt, pipe_bt, raw_bt, closures)
        errs.report(f"{title}  ({len(utts)} utts, snapped boundaries, 20 ms tolerance)")
        results.append(errs)
    print_gap_attribution(*results)

    if cache.dir is not None:
        print(f"\ninference cache: {cache.hits} hits, {cache.misses} misses ({cache.dir})")


if __name__ == "__main__":
    main()
