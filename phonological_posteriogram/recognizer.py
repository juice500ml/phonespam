"""Per-segment phone recognition with optional vocab constraint.

The recognizer is a pure mapping ``(posteriogram, boundaries) → labels``
(one label per segment). By default it carries a full panphon consonant predmat;
when a fitted feature map is supplied, it classifies over that map instead. Any
output-vocab constraint is applied as a mask at the very last moment, just before
the argmax. This means switching vocabularies between calls only requires a tiny
masked argmax — the expensive predmat build only happens once.

Constrain the output vocabulary with ``vocab=[...]``. If you want a Phoible
inventory, resolve it first with :mod:`phonological_posteriogram.phoible` and
pass the returned tuple as ``vocab``.

Frame→time conversion is intentionally *not* the recognizer's job. Boundaries
define the segments; the recognizer returns only labels for those segments.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Sequence

import numpy as np
import panphon

_CR_RELEASE_IPA = ("p", "t", "k", "b", "d", "ɡ", "t͡ʃ", "d͡ʒ")


def _default_output_label(label: str) -> str:
    if "|" in label:
        return label.split("|", 1)[0]
    if label.endswith("_cl") or label.endswith("_rl"):
        return label[:-3]
    return label


@functools.cache
def _validate_vocab(vocab: tuple[str, ...]) -> tuple[str, ...]:
    """Filter a user-supplied vocab to panphon-known phones (cached).

    Warns (once per unique input) about phones panphon doesn't recognize;
    those are dropped. Raises if nothing remains.
    """
    phones = _filter_panphon_known(vocab, context="user-supplied vocab")
    if not phones:
        raise ValueError("vocab has no panphon-known phones after filtering.")
    return phones


def _filter_panphon_known(phones: Sequence[str], *, context: str) -> tuple[str, ...]:
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


@functools.cache
def _panphon_table():
    return panphon.FeatureTable()


@functools.cache
def _panphon_base_names() -> tuple[str, ...]:
    return tuple(_panphon_table().fts("a").names)


@functools.cache
def _panphon_numeric(phone: str) -> tuple[int, ...]:
    ft = _panphon_table()
    if not ft.seg_known(phone):
        raise ValueError(f"panphon does not know phone {phone!r}.")
    return tuple(ft.fts(phone).numeric())


@functools.cache
def _active_panphon_featnames(featnames: tuple[str, ...]) -> tuple[str, ...]:
    active = tuple(
        f"{name}{sign}"
        for name in _panphon_base_names()
        for sign in ("+", "-")
        if f"{name}{sign}" in featnames
    )
    if not active:
        raise ValueError("featnames has no panphon +/- dimensions.")
    return active


def panphon_featmap(
    phones: Sequence[str],
    featnames: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Build a sparse panphon +/- feature map for ``Recognizer``.

    The returned rows contain only panphon ``feature+`` / ``feature-`` names
    that are present in ``featnames``. If ``"_"`` is included in ``phones``,
    it is treated as the silence token: ``silence+`` is added to every row,
    non-silence phones receive ``0``, and ``"_"`` receives ``1`` with all
    panphon dimensions set to ``0``. Other recognizer dimensions, such as
    closure/release state channels, are intentionally omitted and therefore
    ignored by sparse feature-map scoring.
    """
    base_names = _panphon_base_names()
    phones = tuple(phones)
    if not phones:
        raise ValueError("phones is empty.")
    featnames = tuple(featnames)
    includes_silence = "_" in phones
    non_silence_phones = tuple(phone for phone in phones if phone != "_")
    if non_silence_phones:
        active = _active_panphon_featnames(featnames)
    else:
        active = ()
    if includes_silence:
        if "silence+" not in featnames:
            raise ValueError('phones includes "_" but featnames has no "silence+".')
        active = ("silence+",) + active

    featmap = {}
    for phone in phones:
        if phone == "_":
            featmap[phone] = {name: 0 for name in active}
            featmap[phone]["silence+"] = 1
            continue
        values = dict(zip(base_names, _panphon_numeric(phone), strict=True))
        row = {
            name: int(
                (name.endswith("+") and values[name[:-1]] == 1)
                or (name.endswith("-") and values[name[:-1]] == -1)
            )
            for name in active
            if name != "silence+"
        }
        if includes_silence:
            row["silence+"] = 0
        featmap[phone] = row
    return featmap


class Recognizer:
    """Pure mapping ``(posteriogram, boundaries) → labels``.

    Carries either the full panphon predmat or a caller-supplied fitted feature
    map; vocab constraints mask the argmax per :meth:`recognize` call.
    Construct with the posteriogram's ``featnames`` only — frame↔time
    conversion is the caller's job.
    """

    @classmethod
    def default_hparams(cls):
        return {}

    def __init__(self, *, featnames, hparams=None, featmap=None):
        self._uses_fitted_featmap = featmap is not None
        self._featnames = list(featnames)
        self.vocab, self.predmat, self._feat_idxs = self._build_predmat(
            self._featnames, featmap=featmap
        )
        self._output_labels = [_default_output_label(label) for label in self.vocab]
        self._vocab_to_idx = {p: i for i, p in enumerate(self.vocab)}
        self._output_to_idxs = {}
        for i, label in enumerate(self._output_labels):
            self._output_to_idxs.setdefault(label, []).append(i)
        self._mask_cache: dict = {}
        self.hparams = {**self.default_hparams(), **(hparams or {})}

    @staticmethod
    def _build_predmat(featnames, *, featmap=None) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Build prototypes aligned to ``featnames``.

        With no fitted ``featmap``, this uses every panphon-known phone with
        ``cons != 0``; the per-call vocab restriction is applied via
        :meth:`_phones_to_mask`, not by re-building the predmat.

        Caller-supplied fitted feature maps may be dense rows aligned to every
        ``featname`` (the format used by fitted posteriograms), or sparse
        dictionaries mapping only the dimensions to score. Sparse maps let
        callers exclude inert dimensions such as closure/release state
        channels.

        If the posteriogram was trained with TIMIT raw closure/release state
        channels, the classifier adds closure/release variants for stops and
        affricates. Those variants are internal prototypes; returned labels
        are merged back to their base phone, matching the oracle evaluation.
        """
        if featmap is not None:
            vocab = list(featmap)
            first = featmap[vocab[0]]
            if isinstance(first, dict):
                provided = set(first)
                assert provided, "sparse featmap rows must provide at least one feature"
                for label in vocab[1:]:
                    assert set(featmap[label]) == provided, (
                        "sparse featmap rows must all provide the same feature names",
                        label,
                    )
                featname_to_idx = {name: i for i, name in enumerate(featnames)}
                unknown = sorted(provided - set(featname_to_idx))
                assert not unknown, (
                    "sparse featmap provides features absent from featnames",
                    unknown,
                )
                feat_idxs = np.asarray(
                    [featname_to_idx[name] for name in featnames if name in provided],
                    dtype=np.int64,
                )
                active_names = [featnames[i] for i in feat_idxs]
                predmat = np.asarray(
                    [[featmap[label][name] for name in active_names] for label in vocab],
                    dtype=np.float32,
                )
            else:
                predmat = np.asarray([featmap[label] for label in vocab], dtype=np.float32)
                feat_idxs = np.arange(len(featnames), dtype=np.int64)
            assert predmat.ndim == 2
            assert predmat.shape[1] == len(feat_idxs)
            sums = predmat.sum(1, keepdims=True)
            sums = np.where(sums > 0, sums, 1.0)
            return vocab, predmat / sums, feat_idxs

        ft = panphon.FeatureTable()
        panphon_names = ft.fts("a").names
        full_featnames = (
            ["silence+"]
            + [f"{n}+" for n in panphon_names]
            + [f"{n}-" for n in panphon_names]
            + ["closure+", "closure-", "release+", "release-"]
        )
        name_to_full_idx = {n: i for i, n in enumerate(full_featnames)}
        missing = [n for n in featnames if n not in name_to_full_idx]
        if missing:
            raise ValueError(
                "Featnames not derivable from panphon's feature table: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        vocab = ["_"]
        rows = [[1] + [0] * (len(panphon_names) * 2) + [0, 0, 0, 0]]

        def phone_row(phone, state):
            feats = ft.fts(phone).numeric()
            return (
                [0]
                + [1 if n == 1 else 0 for n in feats]
                + [1 if n == -1 else 0 for n in feats]
                + state
            )

        for k, v in ft.seg_dict.items():
            if v["cons"] != 0:
                vocab.append(k)
                rows.append(phone_row(k, [0, 0, 0, 0]))

        if any(n in featnames for n in ("closure+", "closure-", "release+", "release-")):
            for phone in _CR_RELEASE_IPA:
                assert ft.seg_known(phone), f"panphon does not know closure/release phone {phone!r}"
                vocab.append(phone + "_cl")
                rows.append(phone_row(phone, [1, 0, 0, 1]))
                vocab.append(phone + "_rl")
                rows.append(phone_row(phone, [0, 1, 1, 0]))

        full_predmat = np.asarray(rows, dtype=np.float32)

        predmat_cols = []
        for name in featnames:
            predmat_cols.append(full_predmat[:, name_to_full_idx[name]])
        predmat = np.stack(predmat_cols, axis=1)
        sums = predmat.sum(1, keepdims=True)
        sums = np.where(sums > 0, sums, 1.0)
        return vocab, predmat / sums, np.arange(len(featnames), dtype=np.int64)

    def _phones_to_mask(self, phones: tuple[str, ...]) -> np.ndarray:
        """Vocab indices for the given phones; cached per phones-tuple.

        Silence ``"_"`` is always included so silence-snapped frames can
        label as silence regardless of the input vocab.
        """
        cached = self._mask_cache.get(phones)
        if cached is not None:
            return cached
        output_to_idxs = getattr(self, "_output_to_idxs", {})
        idxs = {self._vocab_to_idx[p] for p in phones if p in self._vocab_to_idx}
        for phone in phones:
            idxs.update(output_to_idxs.get(phone, ()))
        if "_" in self._vocab_to_idx:
            idxs.add(self._vocab_to_idx["_"])
        mask = np.array(sorted(idxs), dtype=np.int64)
        self._mask_cache[phones] = mask
        return mask

    def recognize(
        self,
        posteriogram,
        boundaries,
        *,
        sr: int,
        frame_shift: int = 320,
        vocab: Sequence[str] | None = None,
    ) -> list[str]:
        """Label each segment defined by ``boundaries`` via center pooling.

        Args:
            posteriogram: ``(T, n_featnames)`` per-frame sigmoid posteriogram.
            boundaries: 1D array of inter-segment boundary *times in seconds*.
                Feature frames are hop-aligned, so each segment is pooled at the
                frame containing its midpoint time,
                ``floor(t_mid * sr / frame_shift)`` — the exact center
                convention used at feature-extraction time
                (:mod:`training.extract_features`), which avoids the frame
                drift of pooling at the midpoint of independently-floored
                boundary frames.
            sr: sample rate the posteriogram frames were produced at.
            frame_shift: SSL encoder stride in input samples (320 for the
                wav2vec2 family) — one frame spans ``frame_shift / sr`` seconds.
            vocab: optional list of phones to constrain the output to. Phones
                panphon doesn't recognize are warned about and dropped. For a
                Phoible inventory, use :mod:`phonological_posteriogram.phoible`
                to create this tuple.

        Returns one label per segment defined by ``boundaries``.
        """
        if vocab is None:
            phones = None
        elif self._uses_fitted_featmap:
            phones = tuple(vocab)
        else:
            phones = _validate_vocab(tuple(vocab))

        T = len(posteriogram)
        if T == 0:
            return []

        # Boundaries are inter-segment times (seconds). Bracket with [0,
        # frame_to_time(T)] and pool each segment at the frame holding its
        # midpoint time: floor(t_mid * sr / frame_shift). Sorting and clipping
        # keeps segment centers inside the utterance span while preserving one
        # output label per input-defined segment.
        end_t = T * frame_shift / sr
        bt = np.clip(
            np.sort(np.concatenate([[0.0], np.asarray(boundaries, dtype=float), [end_t]])),
            0.0,
            end_t,
        )
        centers = np.clip(np.floor((bt[:-1] + bt[1:]) / 2 * sr / frame_shift).astype(int), 0, T - 1)
        posteriogram = np.asarray(posteriogram)
        assert posteriogram.ndim == 2
        assert posteriogram.shape[1] == len(self._featnames)
        center_post = posteriogram[centers][:, self._feat_idxs]
        logits = center_post @ self.predmat.T

        if phones is None:
            idxs = logits.argmax(axis=1)
        else:
            mask = self._phones_to_mask(phones)
            masked = logits[:, mask]
            idxs = mask[masked.argmax(axis=1)]

        output_labels = getattr(self, "_output_labels", self.vocab)
        return [output_labels[int(idx)] for idx in idxs]
