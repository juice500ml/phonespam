"""The fitted, weights-only part of the phone pipeline.

:class:`PhonologicalPosteriogram` holds everything that is expensive to fit
and never changes once trained:

- the ``ipa`` phonological-vector view,
- the linear regressor ``W_bwd`` that maps raw SSL features onto the ``ipa``
  projection of the preceding phone.

Algorithm classes (``Segmenter``, and later ``Recognizer``) are constructed
on top of a fitted ``PhonologicalPosteriogram`` and carry only
hyperparameters, so different configurations can be compared without
refitting these weights.
"""

from __future__ import annotations

import numpy as np
import panphon

_CR_CLOSURE_TO_STOP = {
    "pcl": "p",
    "tcl": "t",
    "kcl": "k",
    "bcl": "b",
    "dcl": "d",
    "gcl": "ɡ",
}
_CR_RELEASE_IPA = {"p", "t", "k", "b", "d", "ɡ", "t͡ʃ", "d͡ʒ"}
_CR_DIPHTHONGS = {"aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ"}


def _fill_gaps(raw_mask):
    """Fill single-frame gaps between two silent frames in a boolean mask."""
    neighbors_silent = np.logical_and(
        np.concatenate(([False], raw_mask[:-1])),
        np.concatenate((raw_mask[1:], [False])),
    )
    return np.logical_or(raw_mask, neighbors_silent)


def _feature_label_sort_key(label):
    if label == "_":
        return label
    return label.split("|", 1)[1] if "|" in label else label


ACTIVATIONS = ("none", "sigmoid")


def _apply_activation(raw, act):
    """Apply an output activation to a raw projection."""
    if act == "none":
        return raw
    if act == "sigmoid":
        return 1.0 / (1.0 + np.exp(-raw))
    raise ValueError(f"Unknown activation {act!r}; choose one of {ACTIVATIONS}.")


class _VectorView:
    """One phonological-vector view: per-feature pos/zero vectors + calibration.

    Internal to :class:`PhonologicalPosteriogram`; not part of the public API.
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
    def fit(
        cls,
        df,
        vocab,
        group_col,
        filter_features=True,
        prep_featmap=None,
    ):
        ft = panphon.FeatureTable()
        if prep_featmap is None:
            featnames, featmap = cls._prep_featmap(vocab, ft)
        else:
            featnames, featmap = prep_featmap(vocab, ft)
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
            ["silence+"]
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
                    [0] + [1 if n == 1 else 0 for n in feats] + [1 if n == -1 else 0 for n in feats]
                )
        return names, featmap

    @staticmethod
    def _split_phns(featname, featnames, featmap):
        silence_plus_index = featnames.index("silence+")
        if featname == "silence+":
            pos_phns = {p for p, v in featmap.items() if v[silence_plus_index] == 1}
            zero_phns = {p for p, v in featmap.items() if v[silence_plus_index] == 0}
        else:
            index = featnames.index(featname)
            pos_phns = {
                p for p, v in featmap.items() if (v[index] == 1) & (v[silence_plus_index] == 0)
            }
            zero_phns = {
                p for p, v in featmap.items() if (v[index] == 0) & (v[silence_plus_index] == 0)
            }
        return pos_phns, zero_phns

    @classmethod
    def _calc_phnvectors(cls, df, group_col, featnames, featmap):
        pos_vecs, zero_vecs, scales, biases = [], [], [], []
        in_dim = len(df[~df.feat.isna()].iloc[0].feat)

        for featname in featnames:
            pos_phns, zero_phns = cls._split_phns(featname, featnames, featmap)
            if len(pos_phns) > 0 and len(zero_phns) > 0:
                pos_samples = np.stack(df[df[group_col].isin(pos_phns)].feat.tolist())
                zero_samples = np.stack(df[df[group_col].isin(zero_phns)].feat.tolist())
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
        is identical to ``silence+`` (i.e. just a silence detector). ``silence+``
        itself is always kept — it is the silence channel the segmenter
        thresholds in :meth:`PhonologicalPosteriogram.predict_silence_mask`.
        """
        silence_pos, silence_zero = self._split_phns("silence+", self.featnames, self.featmap)
        keep = []
        for i, name in enumerate(self.featnames):
            pos_phns, zero_phns = self._split_phns(name, self.featnames, self.featmap)
            if len(pos_phns) == 0 or len(zero_phns) == 0:
                continue
            if name != "silence+" and pos_phns == silence_pos and zero_phns == silence_zero:
                continue
            keep.append(i)

        keep = np.array(keep)
        self.featnames = [self.featnames[i] for i in keep]
        self.pos_vecs = self.pos_vecs[keep]
        self.zero_vecs = self.zero_vecs[keep]
        self.scales = self.scales[keep]
        self.biases = self.biases[keep]
        self.featmap = {phone: [vals[i] for i in keep] for phone, vals in self.featmap.items()}

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
    """Fitted phonological-vector projection for an SSL encoder.

    Construct with :meth:`fit` (the one expensive training step) or
    :meth:`from_state`; query with :meth:`project` (``act='none'`` for the raw
    projection, ``act='sigmoid'`` for the posteriogram).
    """

    def __init__(self, *, view, W_bwd):
        self.views = {"ipa": view}
        # Raw-feature regressor: a frame's SSL features predict the ipa-view
        # projection of the preceding phone.
        self.W_bwd = np.asarray(W_bwd)

    # -- fitting ---------------------------------------------------------- #

    @classmethod
    def fit(
        cls,
        train_df,
        *,
        filter_features=True,
        group_col: str = "ipa",
        vocab=None,
        prep_featmap=None,
    ):
        """Fit the ipa view and backward regressor from per-phone features.

        ``train_df`` needs columns ``feat`` (per-phone SSL feature vector),
        ``group_col`` (phone labels), and ``audio_path`` / ``min`` (to order
        phones within each utterance for the regressor).
        """
        labeled = train_df[~train_df[group_col].isna()]
        if vocab is None:
            vocab = labeled[group_col].unique().tolist()

        view = _VectorView.fit(
            labeled,
            vocab,
            group_col=group_col,
            filter_features=filter_features,
            prep_featmap=prep_featmap,
        )

        df_sorted = labeled.sort_values(["audio_path", "min"])
        same_utt = df_sorted.audio_path.values[:-1] == df_sorted.audio_path.values[1:]
        prev_feats = np.stack(df_sorted.feat.values[:-1][same_utt])
        curr_feats = np.stack(df_sorted.feat.values[1:][same_utt])

        # W_bwd: current frame's raw SSL features -> previous phone's ipa
        # projection, fit on the raw (un-activated) projection.
        proj_ipa_prev = view.project(prev_feats, act="none")
        W_bwd = np.linalg.lstsq(curr_feats, proj_ipa_prev, rcond=None)[0]

        return cls(view=view, W_bwd=W_bwd)

    @classmethod
    def fit_timit_closure_release(cls, train_df, *, filter_features=True):
        """Fit the notebook's TIMIT closure/release feature scheme.

        Raw TIMIT intervals retain stop closures that merged evaluation labels
        drop. The fitted labels distinguish closure/release state while
        preserving the ordinary IPA feature geometry.
        """
        df = _add_timit_closure_release_labels(train_df)
        vocab = sorted(df.cr_ipa.unique(), key=_feature_label_sort_key)
        return cls.fit(
            df,
            filter_features=filter_features,
            group_col="cr_ipa",
            vocab=vocab,
            prep_featmap=_prep_timit_closure_release_featmap,
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
        """Per-frame silence mask from the ``silence+`` posteriogram channel.

        ``silence+`` is positive only for the silence token ``"_"`` in
        :meth:`_VectorView._prep_featmap`, so its projected (sigmoid) value
        is HIGH for silence-like frames. Returns a boolean array with
        single-frame gaps between silent frames filled in.
        """
        if "silence+" not in self.featnames:
            raise ValueError(
                "posteriogram has no 'silence+' feature; the training vocab "
                "must include the silence token '_'."
            )
        idx = self.featnames.index("silence+")
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
            "view": self.views["ipa"].to_state(),
            "W_bwd": self.W_bwd,
        }

    @classmethod
    def from_state(cls, state):
        missing = {"view", "W_bwd"} - set(state)
        assert not missing, f"posteriogram state missing regressors: {missing}"
        return cls(
            view=_VectorView.from_state(state["view"]),
            W_bwd=state["W_bwd"],
        )


def _timit_closure_release_label(timit_phn, ipa, next_timit_phn):
    if timit_phn in _CR_CLOSURE_TO_STOP:
        if timit_phn == "tcl" and next_timit_phn == "ch":
            return "t͡ʃ_cl"
        if timit_phn == "dcl" and next_timit_phn == "jh":
            return "d͡ʒ_cl"
        return _CR_CLOSURE_TO_STOP[timit_phn] + "_cl"
    if isinstance(ipa, str) and ipa in _CR_RELEASE_IPA:
        return ipa + "_rl"
    return ipa


def _add_timit_closure_release_labels(df):
    required = {"audio_path", "min", "timit_phn", "ipa", "feat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "TIMIT closure/release training requires columns: "
            f"{sorted(required)}; missing {sorted(missing)}"
        )
    out = df.sort_values(["audio_path", "min"]).copy()
    out["next_timit_phn"] = out.groupby("audio_path").timit_phn.shift(-1)
    # cr_ipa is the IPA prototype label (closure/release suffixed): the
    # recognizer collapses ``_cl``/``_rl`` back to the base IPA phone on output,
    # so the model speaks IPA uniformly across datasets. (Provenance back to the
    # native TIMIT token is intentionally not encoded in the label.)
    out["cr_ipa"] = [
        _timit_closure_release_label(tp, ip, nx)
        for tp, ip, nx in zip(out.timit_phn, out.ipa, out.next_timit_phn, strict=True)
    ]
    return out[out.cr_ipa.notna()].copy()


def _prep_timit_closure_release_featmap(vocab, ft):
    base = ft.fts("a").names
    names = (
        ["silence+"]
        + [f"{n}+" for n in base]
        + [f"{n}-" for n in base]
        + ["closure+", "closure-", "release+", "release-"]
    )

    def panphon_onehot(seg):
        feats = ft.fts(seg).numeric()
        return [1 if n == 1 else 0 for n in feats] + [1 if n == -1 else 0 for n in feats]

    featmap = {}
    for v in vocab:
        if v == "_":
            featmap[v] = [1] + [0] * (len(base) * 2) + [0, 0, 0, 0]
            continue
        spec = v.split("|", 1)[1] if "|" in v else v
        if spec.endswith("_cl"):
            assert ft.seg_known(spec[:-3]), f"unknown closure base: {spec[:-3]!r}"
            featmap[v] = [0] + panphon_onehot(spec[:-3]) + [1, 0, 0, 1]
        elif spec.endswith("_rl"):
            assert ft.seg_known(spec[:-3]), f"unknown release base: {spec[:-3]!r}"
            featmap[v] = [0] + panphon_onehot(spec[:-3]) + [0, 1, 1, 0]
        elif ft.seg_known(spec):
            featmap[v] = [0] + panphon_onehot(spec) + [0, 0, 0, 0]
        elif spec in _CR_DIPHTHONGS:
            continue
        else:
            raise ValueError(f"unexpected panphon-unknown label: {v!r}")
    return names, featmap
