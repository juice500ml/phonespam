"""Boundary-level evaluation for phone segmentation.

Boundaries are scored against ground truth with a configurable temporal
tolerance. Two match policies:

- ``"lenient"`` (default): a boundary counts as a true positive iff *any*
  counterpart is within tolerance — no exclusivity, matching Jian et al.
  and earlier work.
- ``"strict"``: greedy one-to-one assignment, matching Strgar & Harwath.

Free mode (``forced=False``) reports boundary precision / recall / F1 /
R-value only. Forced mode pairs predicted and GT segments 1-to-1 and
additionally reports per-phone start/end/duration error statistics,
optionally broken down by IPA symbol.
"""

# NOTE: This does not yet do the equidistant splitting between two close
# boundaries that the R-value paper recommends
# (https://www.isca-archive.org/interspeech_2009/rasanen09_interspeech.pdf).
# At low tolerance the impact is small, and results stay comparable to
# Jian et al.'s earlier papers.

from __future__ import annotations

import csv
import io
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import panphon.distance

log = logging.getLogger(__name__)


@dataclass
class SegmentationUnit:
    """A single aligned unit (e.g., one phone)."""

    start: int | float
    end: int | float
    label: str | int | None = None


class SegmentationEvaluator:
    """Evaluate predicted phone boundaries against ground truth.

    Args:
        tolerance_ms: Boundary tolerance in milliseconds (default 20).
        forced: If True, requires paired segments and computes per-phone
            error statistics in addition to boundary P/R/F1/R-value. If
            False (default), accepts any segment counts and returns only
            boundary-level metrics.
        match_mode: How to count boundary TPs.
            ``"strict"``: greedy one-to-one assignment — each boundary can
            be claimed by at most one counterpart (Strgar & Harwath).
            ``"lenient"`` (default): independent nearest-neighbour — every
            boundary counts as TP iff any counterpart lies within
            tolerance, with no exclusivity.
        strip_endpoints: If True (default), drop each utterance's first and
            last boundary (the start of the first segment and the end of the
            last — i.e. frame 0 and frame T) before scoring. These are
            trivially known (the recognizer always emits them and the GT
            always tiles to them), so scoring them inflates the metrics.
            Pass False to score the raw boundary set (e.g. when testing the
            matching logic directly).
    """

    def __init__(
        self,
        tolerance_ms: int = 20,
        forced: bool = False,
        match_mode: str = "lenient",
        strip_endpoints: bool = True,
    ):
        if match_mode not in {"strict", "lenient"}:
            raise ValueError(
                f"match_mode must be 'strict' or 'lenient', got {match_mode!r}"
            )
        self.tolerance_sec = tolerance_ms / 1000.0
        self._tol_eps = 1e-9  # Float-precision slack on the boundary check.
        self.forced = forced
        self.match_mode = match_mode
        self.strip_endpoints = strip_endpoints

    def evaluate_boundaries(
        self,
        predicted: List[SegmentationUnit],
        ground_truth: List[SegmentationUnit],
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate predicted boundaries against ground truth (one utterance)."""
        if not predicted or not ground_truth:
            return {}
        counts = self._get_boundary_counts(predicted, ground_truth)
        precision, recall, f1, rval = self._get_boundary_metrics(
            counts["precision_counter"],
            counts["recall_counter"],
            counts["pred_counter"],
            counts["gt_counter"],
        )
        results = {
            "n_pred": len(predicted),
            "n_gt": len(ground_truth),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rval": rval,
        }
        if not self.forced:
            return results
        return self._add_forced_alignment_metrics(
            results, predicted, ground_truth, symbols
        )

    def _get_boundary_counts(
        self,
        predicted: List[SegmentationUnit],
        ground_truth: List[SegmentationUnit],
    ) -> Dict[str, int]:
        pred_times = self._extract_boundary_times(predicted)
        gt_times = self._extract_boundary_times(ground_truth)

        tol = self.tolerance_sec + self._tol_eps
        if self.match_mode == "strict":
            precision_counter = self._greedy_match_count(gt_times, pred_times)
            recall_counter = self._greedy_match_count(pred_times, gt_times)
        else:
            # Guard the per-query .min() against an empty reference array
            # (possible once endpoints are stripped from a short utterance).
            precision_counter = (
                sum(np.abs(gt_times - t).min() <= tol for t in pred_times)
                if len(gt_times)
                else 0
            )
            recall_counter = (
                sum(np.abs(pred_times - t).min() <= tol for t in gt_times)
                if len(pred_times)
                else 0
            )

        return {
            "precision_counter": int(precision_counter),
            "recall_counter": int(recall_counter),
            "pred_counter": int(len(pred_times)),
            "gt_counter": int(len(gt_times)),
        }

    def _greedy_match_count(
        self, ref_times: np.ndarray, query_times: np.ndarray
    ) -> int:
        """Count queries that uniquely match a ref boundary within tolerance."""
        if len(ref_times) == 0 or len(query_times) == 0:
            return 0
        tol = self.tolerance_sec + self._tol_eps
        matches: Dict[int, List[int]] = {}
        for i, q in enumerate(query_times):
            dists = np.abs(ref_times - q)
            idxs = np.argsort(dists)
            matches[i] = [int(idx) for idx in idxs if dists[idx] <= tol]

        used: set = set()
        count = 0
        for vs in matches.values():
            for v in sorted(vs):
                if v not in used:
                    used.add(v)
                    count += 1
                    break
        return count

    def _add_forced_alignment_metrics(
        self,
        results: Dict[str, float],
        predicted: List[SegmentationUnit],
        ground_truth: List[SegmentationUnit],
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        n = min(len(predicted), len(ground_truth))
        metrics = np.array(
            [
                self._compute_metrics(p.start, p.end, g.start, g.end)
                for p, g in zip(predicted, ground_truth)
            ]
        )
        start_err, end_err, pbe, dur_err, gt_dur, pred_dur = metrics.T

        percentiles = [5, 50, 95]
        results.update({"n": n})

        error_types = [
            ("start_err", start_err),
            ("end_err", end_err),
            ("pbe", pbe),
            ("boundary_err", np.concatenate([start_err, end_err])),
            ("dur_err", dur_err),
            ("gt_dur", gt_dur),
            ("pred_dur", pred_dur),
        ]
        for name, data in error_types:
            results.update(self._compute_stats(data * 1000, name, percentiles))

        if symbols:
            results["symbol_errors"] = self._analyze_by_symbol(
                symbols, start_err, end_err, pbe, dur_err
            )

        return results

    # ------------------------------------------------------------------ #
    # Boundary helpers                                                   #
    # ------------------------------------------------------------------ #

    def _extract_boundary_times(
        self, units: List[SegmentationUnit]
    ) -> np.ndarray:
        """Unique boundary times for an utterance: every segment ``start``
        plus the single final ``end``.

        Adjacent segments share a boundary (one's ``end`` == the next's
        ``start``); collecting only starts + the final end (then
        ``np.unique``) represents each boundary exactly once — no double
        counting.

        With ``strip_endpoints`` the utterance start and end (the smallest
        and largest times — frame 0 and frame T) are dropped so only
        *internal* boundaries are scored.
        """
        times = [u.start for u in units] + [units[-1].end]
        if self.strip_endpoints and len(times) >= 2:
            times = times[1:-1]
        return np.array(times)

    def _get_boundary_metrics(
        self,
        precision_counter: float,
        recall_counter: float,
        pred_counter: int,
        gt_counter: int,
    ):
        """Precision, recall, F1, and R-value from boundary match counts.

        R-value adapted from UnsupSeg (github.com/felixkreuk/UnsupSeg).
        """
        eps = 1e-7
        precision = precision_counter / (pred_counter + eps)
        recall = recall_counter / (gt_counter + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        os = recall / (precision + eps) - 1
        r1 = np.sqrt((1 - recall) ** 2 + os**2)
        r2 = (-os + recall - 1) / np.sqrt(2)
        rval = 1 - (np.abs(r1) + np.abs(r2)) / 2
        return precision, recall, f1, rval

    def _compute_metrics(self, ps, pe, gs, ge):
        start_err = abs(ps - gs)
        end_err = abs(pe - ge)
        return (
            start_err,
            end_err,
            0.5 * (start_err + end_err),  # pbe
            abs((pe - ps) - (ge - gs)),  # dur_err
            ge - gs,  # gt_dur
            pe - ps,  # pred_dur
        )

    def _compute_stats(self, data, prefix, percentiles):
        stats = {
            f"{prefix}_mean": np.mean(data),
            f"{prefix}_std": np.std(data),
            f"{prefix}_median": np.median(data),
        }
        stats.update(
            {f"{prefix}_p{p}": np.percentile(data, p) for p in percentiles}
        )
        return stats

    def _analyze_by_symbol(self, symbols, start_err, end_err, pbe, dur_err):
        symbol_data = defaultdict(
            lambda: {"start": [], "end": [], "pbe": [], "dur": []}
        )
        for sym, se, ee, pb, de in zip(
            symbols, start_err, end_err, pbe, dur_err
        ):
            symbol_data[sym]["start"].append(se * 1000)
            symbol_data[sym]["end"].append(ee * 1000)
            symbol_data[sym]["pbe"].append(pb * 1000)
            symbol_data[sym]["dur"].append(de * 1000)

        return {
            sym: {
                "count": len(data["pbe"]),
                "pbe_mean": np.mean(data["pbe"]),
                "pbe_std": np.std(data["pbe"]),
                "start_mean": np.mean(data["start"]),
                "end_mean": np.mean(data["end"]),
                "dur_mean": np.mean(data["dur"]),
            }
            for sym, data in symbol_data.items()
        }

    def _get_metric(
        self, results: Dict, key: str, default: float = 0.0
    ) -> float:
        """Get metric, falling back to mean_<key> for batch results."""
        if key in results and results[key] is not None:
            return results[key]
        mean_key = f"mean_{key}"
        if mean_key in results and results[mean_key] is not None:
            return results[mean_key]
        return default

    def pretty_print(self, results: Dict) -> None:
        """Print results as a rich table followed by a CSV dump."""
        if not results:
            print("No results")
            return

        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            self._plain_print(results)
            return

        console = Console()
        g = self._get_metric
        samples = results.get(
            "n", results.get("total_samples", results.get("n_gt", 0))
        )
        segments = results.get("total_segments")

        table = Table(title="Segmentation Evaluation Results", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        count_row = f"{samples} samples"
        if segments is not None:
            count_row += f", {segments} segments"
        table.add_row("Count", count_row)
        table.add_section()
        table.add_row("F1", f"{g(results, 'f1'):.3f}")
        table.add_row("Precision", f"{g(results, 'precision'):.3f}")
        table.add_row("Recall", f"{g(results, 'recall'):.3f}")
        table.add_row("R-value", f"{g(results, 'rval'):.3f}")

        has_per_phone = (
            "start_err_mean" in results
            or "mean_start_err" in results
            or "mean_start_err_mean" in results
        )
        if has_per_phone:
            table.add_section()
            for label, prefix in [
                ("Start Error (ms)", "start_err"),
                ("End Error (ms)", "end_err"),
                ("Phone Boundary Error (ms)", "pbe"),
                ("Duration Error (ms)", "dur_err"),
                ("GT Duration (ms)", "gt_dur"),
                ("Pred Duration (ms)", "pred_dur"),
            ]:
                mean = g(results, f"{prefix}_mean")
                std = g(results, f"{prefix}_std")
                table.add_row(label, f"{mean:.2f} ± {std:.2f}")

        sym_table = None
        if "symbol_errors" in results:
            sym_table = Table(title="Per-Symbol PBE (top 20)", show_lines=True)
            for col in (
                "Symbol",
                "Count",
                "PBE Mean",
                "PBE Std",
                "Start",
                "End",
                "Dur",
            ):
                sym_table.add_column(col, justify="right")
            for sym, s in sorted(
                results["symbol_errors"].items(), key=lambda x: x[1]["pbe_mean"]
            )[:20]:
                sym_table.add_row(
                    str(sym)[:8],
                    str(s["count"]),
                    f"{s['pbe_mean']:.1f}",
                    f"{s['pbe_std']:.1f}",
                    f"{s['start_mean']:.1f}",
                    f"{s['end_mean']:.1f}",
                    f"{s['dur_mean']:.1f}",
                )

        console.print(table)
        if sym_table is not None:
            console.print(sym_table)

        flat = {k: v for k, v in results.items() if k != "symbol_errors"}
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(flat.keys())
        writer.writerow(flat.values())
        console.print(buf.getvalue())

    def _plain_print(self, results: Dict) -> None:
        """Fallback when rich is not installed."""
        g = self._get_metric
        print("Segmentation Evaluation Results")
        print(f"  F1:        {g(results, 'f1'):.3f}")
        print(f"  Precision: {g(results, 'precision'):.3f}")
        print(f"  Recall:    {g(results, 'recall'):.3f}")
        print(f"  R-value:   {g(results, 'rval'):.3f}")
        for prefix, label in [
            ("start_err", "Start Error (ms)"),
            ("end_err", "End Error (ms)"),
            ("pbe", "Phone Boundary Error (ms)"),
            ("dur_err", "Duration Error (ms)"),
        ]:
            mean = g(results, f"{prefix}_mean", default=None)
            if mean is not None:
                std = g(results, f"{prefix}_std", default=0.0)
                print(f"  {label}: {mean:.2f} ± {std:.2f}")

    # -------------------------- batch evaluation ---------------------------- #

    def evaluate_batch(
        self,
        predictions: Dict[str, List[SegmentationUnit]],
        ground_truth: Dict[str, List[SegmentationUnit]],
        symbols_dict: Optional[Dict[str, List[str]]] = None,
        skip_symbols: Optional[set] = None,
    ) -> Dict:
        """Evaluate a batch of predictions against ground truth.

        ``predictions`` and ``ground_truth`` are dicts keyed by utterance id.
        ``skip_symbols`` drops phone-pairs whose label is in the set (useful
        for ignoring an UNK symbol).
        """
        all_results = []

        micro_precision_counter = 0
        micro_recall_counter = 0
        micro_pred_counter = 0
        micro_gt_counter = 0

        assert any(seg_id in predictions for seg_id in ground_truth), (
            "No ground_truth seg_id matched any prediction key — likely a "
            "key-scheme mismatch (e.g. utt_id vs str(b)). Sample keys: "
            f"gt={list(ground_truth)[:3]}, pred={list(predictions)[:3]}"
        )
        for seg_id in ground_truth:
            if seg_id not in predictions:
                log.warning(
                    "Segment ID %s missing in predictions; skipping.", seg_id
                )
                continue

            preds = predictions[seg_id]
            gts = ground_truth[seg_id]
            syms = symbols_dict.get(seg_id) if symbols_dict else None

            if skip_symbols:
                preds_, gts_ = [], []
                for p, g in zip(preds, gts):
                    if (
                        g.label not in skip_symbols
                        and p.label not in skip_symbols
                    ):
                        preds_.append(p)
                        gts_.append(g)
                preds = preds_
                gts = gts_
                # NOTE: syms is filtered independently of the paired
                # preds/gts filtering above, which may cause misalignment
                # if labels diverge.
                if syms:
                    syms = [s for s in syms if s not in skip_symbols]

            res = self.evaluate_boundaries(preds, gts, syms)
            if not res:
                continue
            all_results.append(res)

            counts = self._get_boundary_counts(preds, gts)
            micro_precision_counter += counts["precision_counter"]
            micro_recall_counter += counts["recall_counter"]
            micro_pred_counter += counts["pred_counter"]
            micro_gt_counter += counts["gt_counter"]

        if not all_results:
            return {}

        boundary_metrics = {"f1", "precision", "recall", "rval"}
        count_keys = {"n", "n_pred", "n_gt"}
        metric_names = [
            k
            for k in all_results[0]
            if k not in ("symbol_errors", *count_keys, *boundary_metrics)
        ]
        aggregated = {
            "total_segments": len(all_results),
            "total_samples": sum(
                r.get("n", r.get("n_gt", 0)) for r in all_results
            ),
            **{
                f"mean_{metric}": np.mean([r[metric] for r in all_results])
                for metric in metric_names
            },
        }

        precision, recall, f1, rval = self._get_boundary_metrics(
            micro_precision_counter,
            micro_recall_counter,
            micro_pred_counter,
            micro_gt_counter,
        )
        aggregated.update(
            {
                "precision_counter": micro_precision_counter,
                "recall_counter": micro_recall_counter,
                "pred_counter": micro_pred_counter,
                "gt_counter": micro_gt_counter,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "rval": rval,
            }
        )

        if "symbol_errors" in all_results[0]:
            merged_symbols = defaultdict(
                lambda: {
                    "count": 0,
                    "pbe_all": [],
                    "start_all": [],
                    "end_all": [],
                    "dur_all": [],
                }
            )
            for r in all_results:
                for sym, stats in r.get("symbol_errors", {}).items():
                    merged_symbols[sym]["count"] += stats["count"]
                    merged_symbols[sym]["pbe_all"].append(stats["pbe_mean"])
                    merged_symbols[sym]["start_all"].append(stats["start_mean"])
                    merged_symbols[sym]["end_all"].append(stats["end_mean"])
                    merged_symbols[sym]["dur_all"].append(stats["dur_mean"])

            # NOTE: Unweighted mean-of-means — each utterance's per-symbol
            # mean has equal weight regardless of instance count. Weighting
            # by count would give a more accurate aggregate.
            aggregated["symbol_errors"] = {
                sym: {
                    "count": data["count"],
                    "pbe_mean": np.mean(data["pbe_all"]),
                    "pbe_std": np.std(data["pbe_all"]),
                    "start_mean": np.mean(data["start_all"]),
                    "end_mean": np.mean(data["end_all"]),
                    "dur_mean": np.mean(data["dur_all"]),
                }
                for sym, data in merged_symbols.items()
            }

        return aggregated


# ---------------------------------------------------------------------- #
# Phone recognition evaluation                                            #
# ---------------------------------------------------------------------- #


def _levenshtein(a, b):
    """Standard Levenshtein distance between two token lists."""
    a = list(a)
    b = list(b)
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        curr = [i]
        for j, y in enumerate(b, 1):
            cost = 0 if x == y else 1
            curr.append(
                min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost)
            )
        prev = curr
    return prev[-1]


def _as_labels(items):
    """Accept a list of label strings or a list of SegmentationUnit; return labels."""
    if not items:
        return []
    if hasattr(items[0], "label"):
        return [u.label for u in items]
    return list(items)


class PhoneRecognitionEvaluator:
    """Evaluate per-segment phone predictions against ground truth.

    Two metrics:
      * **PER** (Phone Error Rate) — token-level Levenshtein distance
        between the predicted and reference label sequences, normalized by
        the reference length. Each item in the predicted/reference list is
        treated as one token (so a compound label like ``"eɪ"`` counts as
        one token, not two).
      * **PFER** (Phonological Feature Error Rate) — uses panphon's
        :func:`panphon.distance.Distance.feature_edit_distance`, which
        weights substitutions by phonological-feature dissimilarity (a
        ``p``→``b`` substitution costs less than ``p``→``i``). The
        labels are joined into IPA strings and panphon re-segments them
        internally; the result is normalized by the reference length.

    Both ``evaluate`` and ``evaluate_batch`` accept either a list of label
    strings or a list of :class:`SegmentationUnit` (whose ``.label`` is
    used). By default the silence token ``"_"`` is dropped from both sides
    before scoring (silence is trivial to "recognize" and standard PER
    excludes it).
    """

    def __init__(self, *, skip_labels=frozenset({"_"})):
        self.skip_labels = frozenset(skip_labels) if skip_labels else frozenset()
        self._dist = panphon.distance.Distance()

    def _filter(self, labels):
        if not self.skip_labels:
            return labels
        return [l for l in labels if l not in self.skip_labels]

    def per(self, predicted, reference) -> float:
        """Phone Error Rate for one utterance."""
        pred = self._filter(_as_labels(predicted))
        ref = self._filter(_as_labels(reference))
        if not ref:
            return 0.0
        return _levenshtein(pred, ref) / len(ref)

    def pfer(self, predicted, reference) -> float:
        """Phonological Feature Error Rate for one utterance.

        Uses panphon's ``feature_edit_distance`` over the joined IPA
        strings, divided by the reference token count.
        """
        pred = self._filter(_as_labels(predicted))
        ref = self._filter(_as_labels(reference))
        if not ref:
            return 0.0
        cost = self._dist.feature_edit_distance(
            "".join(pred), "".join(ref)
        )
        return float(cost) / len(ref)

    def evaluate(self, predicted, reference) -> Dict[str, float]:
        return {"per": self.per(predicted, reference),
                "pfer": self.pfer(predicted, reference)}

    def evaluate_batch(
        self,
        predictions: Dict[str, List],
        ground_truth: Dict[str, List],
    ) -> Dict:
        """Micro-averaged PER and PFER over a batch of utterances.

        Both dicts map utterance id to either a list of label strings or a
        list of :class:`SegmentationUnit`. PER and PFER are aggregated by
        summing per-utterance distances and reference counts (rather than
        averaging per-utterance rates), so each token contributes equally.
        """
        assert any(uid in predictions for uid in ground_truth), (
            "No ground_truth id matched any prediction key — likely a "
            f"key-scheme mismatch. Sample keys: "
            f"gt={list(ground_truth)[:3]}, pred={list(predictions)[:3]}"
        )

        total_per_d = 0
        total_pfer_d = 0.0
        total_ref_len = 0
        n_utts = 0
        for uid, ref_units in ground_truth.items():
            if uid not in predictions:
                log.warning("Utterance %s missing in predictions; skipping.", uid)
                continue
            pred = self._filter(_as_labels(predictions[uid]))
            ref = self._filter(_as_labels(ref_units))
            if not ref:
                continue
            total_per_d += _levenshtein(pred, ref)
            total_pfer_d += float(
                self._dist.feature_edit_distance(
                    "".join(pred), "".join(ref)
                )
            )
            total_ref_len += len(ref)
            n_utts += 1

        if total_ref_len == 0:
            return {}
        return {
            "per": total_per_d / total_ref_len,
            "pfer": total_pfer_d / total_ref_len,
            "n_utterances": n_utts,
            "total_ref_phones": total_ref_len,
        }
