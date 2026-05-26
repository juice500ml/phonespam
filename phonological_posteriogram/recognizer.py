"""Per-segment phone recognition with optional vocab constraint.

The recognizer is a pure mapping ``(posteriogram, boundaries) → labels``
(frame-based). It always carries the **full** panphon phone vocab in its
predmat; any output-vocab constraint is applied as a mask at the very
last moment, just before the argmax. This means switching vocabularies
between calls only requires a tiny masked argmax — the expensive predmat
build only happens once.

Three (mutually exclusive) ways to constrain the output vocabulary at
:meth:`Recognizer.recognize` time:

- ``vocab=[...]`` — pass a list of phones directly (panphon-known phones only;
  unknown phones are warned about and dropped).
- ``lang=...`` / ``phoible_id=...`` — pull a Phoible inventory; ``phoneme=True``
  selects the abstract ``Phoneme`` set rather than the surface ``Allophones``.

Frame→time conversion is intentionally *not* the recognizer's job — it
returns ``(start_frame, end_frame, label)`` triples and the caller (typically
:class:`~phonological_posteriogram.phone_model.PhoneModel`) wraps those into
seconds-based :class:`~phonological_posteriogram.evaluation.SegmentationUnit`.
"""

from __future__ import annotations

import functools
import os
import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
import panphon
import pandas as pd

_PHOIBLE_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "phoible.csv"
)


@functools.lru_cache(maxsize=1)
def _phoible():
    """Lazily load (and cache) the packaged Phoible inventory CSV."""
    df = pd.read_csv(
        _PHOIBLE_CSV,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    df["InventoryID"] = df["InventoryID"].astype(int)
    return df


@functools.lru_cache(maxsize=None)
def _load_inventory(phoible_id: int, phoneme: bool = False) -> Tuple[str, ...]:
    """Load one Phoible inventory's phone list, filtered to panphon-known.

    Cached per ``(phoible_id, phoneme)`` so the CSV is read once and the
    ``ft.seg_known`` filter is applied once. Warns (once) about phones the
    inventory mentions that panphon doesn't recognize — those are dropped.
    """
    df = _phoible()
    rows = df[df["InventoryID"] == int(phoible_id)]
    if rows.empty:
        raise ValueError(
            f"Phoible InventoryID {phoible_id} not found in the packaged "
            "phoible.csv."
        )
    if phoneme:
        phones = sorted(rows["Phoneme"].dropna().unique())
    else:
        # Allophones is a space-separated list per phoneme entry.
        phones_set: set = set()
        for s in rows["Allophones"].dropna():
            phones_set.update(s.split())
        phones = sorted(phones_set)

    return _filter_panphon_known(
        tuple(phones), context=f"InventoryID {phoible_id}"
    )


@functools.lru_cache(maxsize=None)
def _validate_vocab(vocab: Tuple[str, ...]) -> Tuple[str, ...]:
    """Filter a user-supplied vocab to panphon-known phones (cached).

    Warns (once per unique input) about phones panphon doesn't recognize;
    those are dropped. Raises if nothing remains.
    """
    phones = _filter_panphon_known(vocab, context="user-supplied vocab")
    if not phones:
        raise ValueError(
            "vocab has no panphon-known phones after filtering."
        )
    return phones


def _filter_panphon_known(
    phones: Sequence[str], *, context: str
) -> Tuple[str, ...]:
    """Reduce a phone list to the panphon segments it's composed of.

    Phoible sometimes spells a "phone" as several IPA segments (e.g. the
    affricate ``ts`` -> ``t`` + ``s``, the prenasalized ``mb`` -> ``m`` +
    ``b``). ``ft.ipa_segs(p)`` splits such a string into the panphon
    segments it recognizes, silently dropping any characters it doesn't.

    Every recognized segment is added to the vocabulary (a single known
    segment is just a length-1 decomposition). A phone that does **not**
    round-trip — ``"".join(ft.ipa_segs(p)) != p`` — had an unrecognized
    character dropped; its recognized parts are still kept, but it's
    collected for a warning so the caller knows something was lost.

    Because multi-segment phones flatten to their parts, the same segment
    can arrive from several phones, so the result is de-duplicated
    (first-occurrence order preserved).
    """
    ft = panphon.FeatureTable()
    segments, unknown = [], []
    for p in phones:
        segs = ft.ipa_segs(p)
        segments += segs
        if "".join(segs) != p:
            unknown.append(p)
    if unknown:
        preview = unknown[:10]
        warnings.warn(
            f"Ignoring {len(unknown)} phone(s) with segment(s) not "
            f"recognized by panphon for {context}: {preview}"
            f"{'...' if len(unknown) > len(preview) else ''}",
            stacklevel=3,
        )
    # De-dup while keeping first-occurrence order.
    return tuple(dict.fromkeys(segments))


def _inventory_dialect(series):
    """An inventory's SpecificDialect (it's constant within an InventoryID).

    Returns the first specified (non-NaN) value if there is one; otherwise
    (the inventory has only NaN rows) returns the first index, which is NaN
    — never a value borrowed from another inventory.
    """
    specified = series.dropna()
    return specified.iloc[0] if not specified.empty else series.iloc[0]


@functools.lru_cache(maxsize=None)
def _resolve_lang(lang: str) -> int:
    """Resolve a language name to a single Phoible InventoryID.

    Searches LanguageName, then ISO6393, then Glottocode (all
    case-insensitive). When a language matches multiple inventories, the
    **dialect-free** inventory (``SpecificDialect`` is NaN) is preferred —
    it's the most generic representation of the language. If several are
    dialect-free, the smallest such InventoryID wins; if none is, the
    smallest InventoryID overall is used. A warning lists every candidate's
    SpecificDialect; pass ``phoible_id=`` explicitly to override.
    """
    df = _phoible()
    target = str(lang).strip().lower()
    for col in ("LanguageName", "ISO6393", "Glottocode"):
        mask = df[col].astype(str).str.lower() == target
        rows = df[mask]
        if rows.empty:
            continue
        inv_ids = sorted(int(i) for i in rows["InventoryID"].unique())
        if len(inv_ids) == 1:
            return inv_ids[0]

        glottocode = rows["Glottocode"].iloc[0]
        dialect_per_id = {
            int(rid): _inventory_dialect(rsub["SpecificDialect"])
            for rid, rsub in rows.groupby("InventoryID")
        }
        # Prefer the dialect-free inventory (SpecificDialect is NaN). Among
        # several dialect-free ones take the smallest InventoryID; if none
        # is dialect-free, fall back to the smallest InventoryID overall.
        dialect_free = [i for i in inv_ids if pd.isna(dialect_per_id[i])]
        chosen = dialect_free[0] if dialect_free else inv_ids[0]
        id_dialect_lines = ", ".join(
            f"{i}: {dialect_per_id[i]!r}" for i in inv_ids
        )
        warnings.warn(
            f"Phoible has {len(inv_ids)} inventories for {lang!r} "
            f"({id_dialect_lines}); picking {chosen} "
            f"(dialect {dialect_per_id[chosen]!r}). Pass `phoible_id=...` "
            f"to override. See https://phoible.org/languages/{glottocode}",
            stacklevel=3,
        )
        return chosen
    raise ValueError(
        f"Language {lang!r} not found in Phoible "
        "(LanguageName / ISO6393 / Glottocode)."
    )


class Recognizer:
    """Pure mapping ``(posteriogram, boundaries) → frame-based triples``.

    Always carries the full panphon predmat; vocab constraints mask the
    argmax per :meth:`recognize` call. Construct with the posteriogram's
    ``featnames`` only — frame↔time conversion is the caller's job.
    """

    @classmethod
    def default_hparams(cls):
        return {}

    def __init__(self, *, featnames, hparams=None):
        self.vocab, self.predmat = self._build_predmat(featnames)
        self._vocab_to_idx = {p: i for i, p in enumerate(self.vocab)}
        self._mask_cache: dict = {}
        self.hparams = {**self.default_hparams(), **(hparams or {})}

    @staticmethod
    def _build_predmat(featnames) -> Tuple[List[str], np.ndarray]:
        """Build the full-panphon predmat aligned to ``featnames``.

        Always uses every panphon-known phone with ``cons != 0``; the
        per-call vocab restriction is applied via :meth:`_phones_to_mask`,
        not by re-building the predmat.
        """
        ft = panphon.FeatureTable()
        panphon_names = ft.fts("a").names
        full_featnames = (
            ["silence+"]
            + [f"{n}+" for n in panphon_names]
            + [f"{n}-" for n in panphon_names]
        )
        name_to_full_idx = {n: i for i, n in enumerate(full_featnames)}
        missing = [n for n in featnames if n not in name_to_full_idx]
        if missing:
            raise ValueError(
                "Featnames not derivable from panphon's feature table: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        idx_in_full = np.asarray(
            [name_to_full_idx[n] for n in featnames], dtype=np.int64
        )

        vocab = ["_"]
        rows = [[1] + [0] * (len(panphon_names) * 2)]
        for k, v in ft.seg_dict.items():
            if v["cons"] != 0:
                vocab.append(k)
                feats = v.numeric()
                rows.append(
                    [0]
                    + [1 if n == 1 else 0 for n in feats]
                    + [1 if n == -1 else 0 for n in feats]
                )
        full_predmat = np.asarray(rows, dtype=np.float32)

        predmat = full_predmat[:, idx_in_full]
        sums = predmat.sum(1, keepdims=True)
        sums = np.where(sums > 0, sums, 1.0)
        return vocab, predmat / sums

    def _phones_to_mask(self, phones: Tuple[str, ...]) -> np.ndarray:
        """Vocab indices for the given phones; cached per phones-tuple.

        Silence ``"_"`` is always included so silence-snapped frames can
        label as silence regardless of the input vocab.
        """
        cached = self._mask_cache.get(phones)
        if cached is not None:
            return cached
        idxs = {
            self._vocab_to_idx[p]
            for p in phones
            if p in self._vocab_to_idx
        }
        if "_" in self._vocab_to_idx:
            idxs.add(self._vocab_to_idx["_"])
        mask = np.array(sorted(idxs), dtype=np.int64)
        self._mask_cache[phones] = mask
        return mask

    def _resolve_vocab(
        self,
        *,
        lang: Optional[str],
        phoible_id: Optional[int],
        phoneme: bool,
        vocab: Optional[Sequence[str]],
    ) -> Optional[Tuple[str, ...]]:
        """Pick the constrained phone set from one of the vocab args.

        Returns ``None`` for "no constraint" (full panphon vocab). Raises
        on conflicting / inconsistent argument combinations.
        """
        sources = sum(
            1
            for x in (lang, phoible_id, vocab)
            if x is not None
        )
        if sources > 1:
            raise ValueError(
                "Pass at most one of `vocab=`, `lang=`, or `phoible_id=`."
            )
        if phoneme and vocab is not None:
            raise ValueError(
                "phoneme=True is only meaningful with a Phoible language "
                "constraint (`lang=` / `phoible_id=`), not with an explicit "
                "`vocab=` list."
            )
        if vocab is not None:
            return _validate_vocab(tuple(vocab))
        if lang is not None or phoible_id is not None:
            if phoible_id is None:
                phoible_id = _resolve_lang(lang)
            return _load_inventory(phoible_id, phoneme=phoneme)
        # No constraint, but phoneme=True without lang/phoible_id is still
        # a user mistake (it has no effect).
        if phoneme:
            raise ValueError(
                "phoneme=True requires a language constraint "
                "(`lang=...` or `phoible_id=...`). With no language "
                "this is a phone recognizer over the full panphon "
                "vocab, not a phoneme recognizer."
            )
        return None

    def recognize(
        self,
        posteriogram,
        boundaries,
        *,
        lang: Optional[str] = None,
        phoible_id: Optional[int] = None,
        phoneme: bool = False,
        vocab: Optional[Sequence[str]] = None,
    ) -> List[Tuple[int, int, str]]:
        """Label each segment defined by ``boundaries`` via center pooling.

        Args:
            posteriogram: ``(T, n_featnames)`` per-frame sigmoid posteriogram.
            boundaries: 1D array of segmentation boundary frame indices.
            vocab: optional list of phones to constrain the output to. Phones
                panphon doesn't recognize are warned about and dropped.
            lang / phoible_id / phoneme: optional Phoible-inventory vocab
                constraint (mutually exclusive with ``vocab``). ``phoneme=True``
                requires one of ``lang`` / ``phoible_id``.

        Returns ``list[(start_frame, end_frame, label)]`` (frame-based; the
        caller handles frame→time conversion).
        """
        phones = self._resolve_vocab(
            lang=lang, phoible_id=phoible_id, phoneme=phoneme, vocab=vocab
        )

        T = len(posteriogram)
        if T == 0:
            return []

        bs = np.unique(
            np.concatenate(
                [
                    np.asarray([0]),
                    np.asarray(boundaries, dtype=int),
                    np.asarray([T]),
                ]
            )
        )
        bs = bs[(bs >= 0) & (bs <= T)]

        centers = np.clip((bs[:-1] + bs[1:]) // 2, 0, T - 1)
        center_post = np.asarray(posteriogram)[centers]
        logits = center_post @ self.predmat.T

        if phones is None:
            idxs = logits.argmax(axis=1)
        else:
            mask = self._phones_to_mask(phones)
            masked = logits[:, mask]
            idxs = mask[masked.argmax(axis=1)]

        return [
            (int(bs[i]), int(bs[i + 1]), self.vocab[int(idxs[i])])
            for i in range(len(centers))
        ]
