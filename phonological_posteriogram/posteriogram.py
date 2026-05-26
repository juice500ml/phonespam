"""The fitted, weights-only part of the phone pipeline.

:class:`PhonologicalPosteriogram` holds everything that is expensive to fit
and never changes once trained:

- three phonological-vector "views" — ``ipa`` (current phone), ``l_1``
  (preceding phone), ``r_1`` (following phone),
- the two linear regressors ``W_r1_to_ipa`` / ``W_l1_to_ipa`` that map the
  ``r_1`` / ``l_1`` projections into the ``ipa`` projection space.

Algorithm classes (``Segmenter``, and later ``Recognizer``) are constructed
on top of a fitted ``PhonologicalPosteriogram`` and carry only
hyperparameters, so different configurations can be compared without
refitting these weights.
"""

from __future__ import annotations

import numpy as np
import panphon

VIEWS = ("ipa", "l_1", "r_1")


def _fill_gaps(raw_mask):
    """Fill single-frame gaps between two silent frames in a boolean mask."""
    neighbors_silent = np.logical_and(
        np.concatenate(([False], raw_mask[:-1])),
        np.concatenate((raw_mask[1:], [False])),
    )
    return np.logical_or(raw_mask, neighbors_silent)


ACTIVATIONS = ("none", "sigmoid")


def _apply_activation(raw, act):
    """Apply an output activation to a raw projection."""
    if act == "none":
        return raw
    if act == "sigmoid":
        return 1.0 / (1.0 + np.exp(-raw))
    raise ValueError(
        f"Unknown activation {act!r}; choose one of {ACTIVATIONS}."
    )


class _VectorView:
    """One phonological-vector view: per-feature pos/zero vectors + calibration.

    A view is fit by grouping the per-phone training rows on a single label
    column (``ipa``, ``l_1``, or ``r_1``). Internal to
    :class:`PhonologicalPosteriogram`; not part of the public API.
    """

    def __init__(self, *, featnames, featmap, pos_vecs, zero_vecs, scales, biases):
        self.featnames = list(featnames)
        self.featmap = {k: list(v) for k, v in featmap.items()}
        self.pos_vecs = np.asarray(pos_vecs)
        self.zero_vecs = np.asarray(zero_vecs)
        self.scales = np.asarray(scales)
        self.biases = np.asarray(biases)
        self.in_dim = self.pos_vecs.shape[1]

    # -- fitting ---------------------------------------------------------- #

    @classmethod
    def fit(cls, df, vocab, group_col, filter_features=True):
        ft = panphon.FeatureTable()
        featnames, featmap = cls._prep_featmap(vocab, ft)
        pos_vecs, zero_vecs, scales, biases = cls._calc_phnvectors(
            df, group_col, featnames, featmap
        )
        view = cls(
            featnames=featnames,
            featmap=featmap,
            pos_vecs=pos_vecs,
            zero_vecs=zero_vecs,
            scales=scales,
            biases=biases,
        )
        if filter_features:
            view._filter_features()
        return view

    @staticmethod
    def _prep_featmap(vocab, ft):
        names = (
            ["speech+"]
            + [f"{n}+" for n in ft.fts("a").names]
            + [f"{n}-" for n in ft.fts("a").names]
        )
        featmap = {}
        for v in vocab:
            if v == "_":
                featmap[v] = [1] + ([0] * (len(names) - 1))
            elif ft.seg_known(v):
                feats = ft.fts(v).numeric()
                featmap[v] = (
                    [0]
                    + [1 if n == 1 else 0 for n in feats]
                    + [1 if n == -1 else 0 for n in feats]
                )
        return names, featmap

    @staticmethod
    def _split_phns(featname, featnames, featmap):
        speech_plus_index = featnames.index("speech+")
        if featname == "speech+":
            pos_phns = {p for p, v in featmap.items() if v[speech_plus_index] == 1}
            zero_phns = {p for p, v in featmap.items() if v[speech_plus_index] == 0}
        else:
            index = featnames.index(featname)
            pos_phns = {p for p, v in featmap.items() if (v[index] == 1) & (v[speech_plus_index] == 0)}
            zero_phns = {p for p, v in featmap.items() if (v[index] == 0) & (v[speech_plus_index] == 0)}
        return pos_phns, zero_phns

    @classmethod
    def _calc_phnvectors(cls, df, group_col, featnames, featmap):
        pos_vecs, zero_vecs, scales, biases = [], [], [], []
        in_dim = len(df[~df.feat.isna()].iloc[0].feat)

        for featname in featnames:
            pos_phns, zero_phns = cls._split_phns(featname, featnames, featmap)
            if len(pos_phns) > 0 and len(zero_phns) > 0:
                pos_samples = np.stack(
                    df[df[group_col].isin(pos_phns)].feat.tolist()
                )
                zero_samples = np.stack(
                    df[df[group_col].isin(zero_phns)].feat.tolist()
                )
                pos_vec = pos_samples.mean(0)
                zero_vec = zero_samples.mean(0)
                w = pos_vec - zero_vec
                pos_center = (pos_samples @ w.T).mean(0)
                zero_center = (zero_samples @ w.T).mean(0)
                bias = -(zero_center + pos_center) / 2.0
                scale = 4.0 / (pos_center - zero_center)
            else:
                pos_vec = np.zeros(in_dim)
                zero_vec = np.zeros(in_dim)
                bias = 0.0
                scale = 1.0

            pos_vecs.append(pos_vec)
            zero_vecs.append(zero_vec)
            scales.append(scale)
            biases.append(bias)

        return (
            np.stack(pos_vecs),
            np.stack(zero_vecs),
            np.stack(scales),
            np.stack(biases),
        )

    def _filter_features(self):
        """Drop dead/degenerate features.

        Dead: no phones in the + or 0 class. Degenerate: the +/0 partition
        is identical to ``speech+`` (i.e. just a silence detector). ``speech+``
        itself is always kept.
        """
        speech_pos, speech_zero = self._split_phns(
            "speech+", self.featnames, self.featmap
        )
        keep = []
        for i, name in enumerate(self.featnames):
            pos_phns, zero_phns = self._split_phns(
                name, self.featnames, self.featmap
            )
            if len(pos_phns) == 0 or len(zero_phns) == 0:
                continue
            if (
                name != "speech+"
                and pos_phns == speech_pos
                and zero_phns == speech_zero
            ):
                continue
            keep.append(i)

        keep = np.array(keep)
        self.featnames = [self.featnames[i] for i in keep]
        self.pos_vecs = self.pos_vecs[keep]
        self.zero_vecs = self.zero_vecs[keep]
        self.scales = self.scales[keep]
        self.biases = self.biases[keep]
        self.featmap = {
            phone: [vals[i] for i in keep]
            for phone, vals in self.featmap.items()
        }

    # -- projection ------------------------------------------------------- #

    def project(self, feats, act="none"):
        W = self.pos_vecs - self.zero_vecs
        raw = feats @ W.T + self.biases[None, :]
        raw = raw * self.scales[None, :]
        return _apply_activation(raw, act)

    # -- serialization ---------------------------------------------------- #

    def to_state(self):
        return {
            "featnames": list(self.featnames),
            "featmap": {k: list(v) for k, v in self.featmap.items()},
            "pos_vecs": self.pos_vecs,
            "zero_vecs": self.zero_vecs,
            "scales": self.scales,
            "biases": self.biases,
        }

    @classmethod
    def from_state(cls, state):
        return cls(
            featnames=state["featnames"],
            featmap=state["featmap"],
            pos_vecs=state["pos_vecs"],
            zero_vecs=state["zero_vecs"],
            scales=state["scales"],
            biases=state["biases"],
        )


class PhonologicalPosteriogram:
    """Fitted phonological-vector projections for an SSL encoder.

    Bundles the three views (``ipa``/``l_1``/``r_1``) and the two regressors.
    Construct with :meth:`fit` (the one expensive training step) or
    :meth:`from_state`; query with :meth:`project` (``act='none'`` for the
    raw projection, ``act='sigmoid'`` for the posteriogram).
    """

    def __init__(self, *, views, W_r1_to_ipa, W_l1_to_ipa):
        missing = set(VIEWS) - set(views)
        assert not missing, f"PhonologicalPosteriogram missing views: {missing}"
        self.views = dict(views)
        self.W_r1_to_ipa = np.asarray(W_r1_to_ipa)
        self.W_l1_to_ipa = np.asarray(W_l1_to_ipa)

    # -- fitting ---------------------------------------------------------- #

    @classmethod
    def fit(cls, train_df, *, filter_features=True):
        """Fit the three views and both regressors from per-phone features.

        ``train_df`` needs columns ``feat`` (per-phone SSL feature vector),
        ``ipa`` / ``l_1`` / ``r_1`` (phone labels), and ``audio_path`` / ``min``
        (to order phones within each utterance for the regressors).
        """
        labeled = train_df[~train_df.ipa.isna()]
        vocab = labeled.ipa.unique().tolist()

        views = {
            name: _VectorView.fit(
                labeled, vocab, group_col=name, filter_features=filter_features
            )
            for name in VIEWS
        }

        df_sorted = labeled.sort_values(["audio_path", "min"])
        same_utt = (
            df_sorted.audio_path.values[:-1]
            == df_sorted.audio_path.values[1:]
        )
        prev_feats = np.stack(df_sorted.feat.values[:-1][same_utt])
        curr_feats = np.stack(df_sorted.feat.values[1:][same_utt])

        # Regressors are fit on the raw (un-activated) projections.
        proj_ipa_curr = views["ipa"].project(curr_feats, act="none")
        proj_ipa_prev = views["ipa"].project(prev_feats, act="none")
        proj_r1_prev = views["r_1"].project(prev_feats, act="none")
        proj_l1_next = views["l_1"].project(curr_feats, act="none")

        W_r1_to_ipa = np.linalg.lstsq(
            proj_r1_prev, proj_ipa_curr, rcond=None
        )[0]
        W_l1_to_ipa = np.linalg.lstsq(
            proj_l1_next, proj_ipa_prev, rcond=None
        )[0]

        return cls(
            views=views, W_r1_to_ipa=W_r1_to_ipa, W_l1_to_ipa=W_l1_to_ipa
        )

    # -- projection ------------------------------------------------------- #

    def project(self, feats, view="ipa", act="none"):
        """Project per-frame features onto a view.

        ``act`` selects the output activation: ``"none"`` returns the raw
        (calibrated linear) projection; ``"sigmoid"`` squashes it to (0, 1).
        For ``view='ipa'`` with ``act='sigmoid'`` this is the phonological
        posteriogram.
        """
        return self.views[view].project(feats, act=act)

    def predict_silence_mask(self, feats, threshold=0.5):
        """Per-frame silence mask from the ``speech+`` posteriogram channel.

        ``speech+`` is positive only for the silence token ``"_"`` in
        :meth:`_VectorView._prep_featmap`, so its projected (sigmoid) value
        is HIGH for silence-like frames. Returns a boolean array with
        single-frame gaps between silent frames filled in.
        """
        if "speech+" not in self.featnames:
            raise ValueError(
                "posteriogram has no 'speech+' feature; the training vocab "
                "must include the silence token '_'."
            )
        idx = self.featnames.index("speech+")
        # The threshold is calibrated against the (0, 1) sigmoid output.
        proj = self.project(feats, view="ipa", act="sigmoid")
        return _fill_gaps(proj[:, idx] > threshold)

    @property
    def featnames(self):
        """Feature names of the ``ipa`` view (the posteriogram axes)."""
        return self.views["ipa"].featnames

    # -- serialization ---------------------------------------------------- #

    def to_state(self):
        return {
            "views": {name: v.to_state() for name, v in self.views.items()},
            "W_r1_to_ipa": self.W_r1_to_ipa,
            "W_l1_to_ipa": self.W_l1_to_ipa,
        }

    @classmethod
    def from_state(cls, state):
        return cls(
            views={
                name: _VectorView.from_state(s)
                for name, s in state["views"].items()
            },
            W_r1_to_ipa=state["W_r1_to_ipa"],
            W_l1_to_ipa=state["W_l1_to_ipa"],
        )
