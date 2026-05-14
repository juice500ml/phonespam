import warnings

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
from scipy.signal import find_peaks

warnings.filterwarnings(
    "ignore", message="Support for mismatched key_padding_mask"
)


class SilenceHandler:
    """Predicts silent frames from the ``speech+`` dimension of a fitted
    :class:`PhonologicalVectors`.

    ``speech+`` is the canonical phonological feature that is positive only
    for the silence token ``"_"`` — projecting per-frame SSL features onto
    ``pv_ipa`` therefore yields a per-frame silence probability at the
    ``speech+`` index that we threshold to a binary mask. This replaces the
    standalone scikit-learn classifier the original pipeline relied on.
    """

    def __init__(self, pv_ipa, *, threshold=0.5):
        if "speech+" not in pv_ipa.featnames:
            raise ValueError(
                "pv_ipa is missing the 'speech+' feature; the training "
                "vocab must include the silence token '_'."
            )
        self.pv_ipa = pv_ipa
        self.threshold = float(threshold)
        self.speech_plus_idx = pv_ipa.featnames.index("speech+")

    def _fill_gaps(self, raw_mask):
        neighbors_silent = np.logical_and(
            np.concatenate(([False], raw_mask[:-1])),
            np.concatenate((raw_mask[1:], [False])),
        )
        return np.logical_or(raw_mask, neighbors_silent)

    def predict_silence_mask(self, feats, threshold=None):
        thr = self.threshold if threshold is None else float(threshold)
        proj = self.pv_ipa.project(feats)
        raw_silence_mask = proj[:, self.speech_plus_idx] < thr
        return self._fill_gaps(raw_silence_mask)

    def handle_silence(self, preds, silence_mask, snap_tolerance=1):
        n_frames = len(silence_mask)

        spans = []
        start = None
        for i, v in enumerate(silence_mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                spans.append((start, i))
                start = None
        if start is not None:
            spans.append((start, len(silence_mask)))

        snapped = set()
        silence_boundaries = []

        for s, e in spans:
            if s > 0:
                nearby = preds[
                    (preds >= s - snap_tolerance)
                    & (preds <= s + snap_tolerance)
                ]
                if len(nearby) > 0:
                    outside = nearby[nearby <= s]
                    silence_boundaries.append(
                        outside.min() if len(outside) > 0 else nearby.min()
                    )
                    snapped.update(nearby.tolist())
                else:
                    silence_boundaries.append(s)

            if e < n_frames:
                nearby = preds[
                    (preds >= e - snap_tolerance)
                    & (preds <= e + snap_tolerance)
                ]
                if len(nearby) > 0:
                    outside = nearby[nearby >= e]
                    silence_boundaries.append(
                        outside.max() if len(outside) > 0 else nearby.max()
                    )
                    snapped.update(nearby.tolist())
                else:
                    silence_boundaries.append(e)

        keep = ~silence_mask[preds]
        for i, p in enumerate(preds):
            if p in snapped:
                keep[i] = False

        out = np.unique(
            np.concatenate(
                [preds[keep], np.array(silence_boundaries, dtype=int)]
            )
        )
        return out


class PhonologicalVectors:

    def __init__(
        self,
        df=None,
        vocab=None,
        group_col="ipa",
        filter_features=True,
        *,
        _from_state=None,
    ):
        if _from_state is not None:
            self.featnames = list(_from_state["featnames"])
            self.featmap = {
                k: list(v) for k, v in _from_state["featmap"].items()
            }
            self.pos_vecs = np.asarray(_from_state["pos_vecs"])
            self.zero_vecs = np.asarray(_from_state["zero_vecs"])
            self.scales = np.asarray(_from_state["scales"])
            self.biases = np.asarray(_from_state["biases"])
            self.in_dim = self.pos_vecs.shape[1]
            return
        assert (
            df is not None and vocab is not None
        ), "PhonologicalVectors requires (df, vocab) unless _from_state is given."
        import panphon

        ft = panphon.FeatureTable()
        self.featnames, self.featmap = self.prep_featmap(vocab, ft)
        self.calc_phnvectors(df, group_col)
        if filter_features:
            self._filter_features()

    @staticmethod
    def prep_featmap(vocab, ft):
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

    def split_phns(self, featname):
        index = self.featnames.index(featname)
        pos_phns = {p for p, v in self.featmap.items() if v[index] == 1}
        zero_phns = {p for p, v in self.featmap.items() if v[index] == 0}
        return pos_phns, zero_phns

    def calc_phnvectors(self, df, group_col):
        pos_vecs, zero_vecs, scales, biases = [], [], [], []
        self.in_dim = len(df[~df.feat.isna()].iloc[0].feat)

        for featname in self.featnames:
            pos_phns, zero_phns = self.split_phns(featname)
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
                pos_vec = np.zeros(self.in_dim)
                zero_vec = np.zeros(self.in_dim)
                bias = 0.0
                scale = 1.0

            pos_vecs.append(pos_vec)
            zero_vecs.append(zero_vec)
            scales.append(scale)
            biases.append(bias)

        self.pos_vecs = np.stack(pos_vecs)
        self.zero_vecs = np.stack(zero_vecs)
        self.scales = np.stack(scales)
        self.biases = np.stack(biases)

    def _filter_features(self):
        speech_pos, speech_zero = self.split_phns("speech+")
        keep = []
        for i, name in enumerate(self.featnames):
            pos_phns, zero_phns = self.split_phns(name)
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
        return cls(_from_state=state)

    def project_raw(self, feats):
        W = self.pos_vecs - self.zero_vecs
        raw = feats @ W.T + self.biases[None, :]
        raw = raw * self.scales[None, :]
        return raw

    def project(self, feats):
        raw = self.project_raw(feats)
        return 1.0 / (1.0 + np.exp(-raw))


def _melspec_kaldi(y, *, sr, frame_shift_ms, n_mels=40):
    waveform = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
    feats = kaldi.fbank(
        waveform,
        sample_frequency=float(sr),
        frame_length=25.0,
        frame_shift=frame_shift_ms,
        num_mel_bins=n_mels,
        use_power=True,
        use_energy=False,
        dither=0.0,
        snip_edges=False,
    )
    return feats.cpu().numpy()


def _mel_svf(mel_frames, left, right):
    mel_frames = np.asarray(mel_frames, dtype=float)
    n = mel_frames.shape[0]
    signal = np.full(n, np.nan)
    if n <= left + right:
        return signal
    a = mel_frames[: n - left - right]
    b = mel_frames[left + right :]
    dots = np.sum(a * b, axis=1)
    norms_a = np.linalg.norm(a, axis=1)
    norms_b = np.linalg.norm(b, axis=1)
    denom = norms_a * norms_b
    valid = denom > 0
    out = np.full(len(a), np.nan)
    out[valid] = 1.0 - dots[valid] / denom[valid]
    signal[left : n - right] = out
    finite = np.isfinite(signal)
    if finite.any():
        lo, hi = np.nanmin(signal), np.nanmax(signal)
        if hi > lo:
            signal[finite] = (signal[finite] - lo) / (hi - lo)
    return signal


def _mel_svf_signal(
    audio,
    left,
    right,
    target_len,
    *,
    sr,
    mel_frame_shift_ms,
):
    mel = _melspec_kaldi(audio, sr=sr, frame_shift_ms=mel_frame_shift_ms)
    sig = _mel_svf(mel, left=left, right=right)
    if len(sig) == 0 or target_len == 0:
        return np.full(target_len, np.nan, dtype=np.float32)
    indices = np.round(np.linspace(0, len(sig) - 1, target_len)).astype(int)
    return sig[indices].astype(np.float32)


def _cos_dist_pairs(a, b):
    dots = np.sum(a * b, axis=1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.ones(len(a))
    valid = norms > 0
    out[valid] = 1.0 - dots[valid] / norms[valid]
    return out


def _delta(proj, offset):
    T = proj.shape[0]
    delta = np.full(T, np.nan)
    if T <= offset:
        return delta
    delta[: T - offset] = _cos_dist_pairs(proj[:-offset], proj[offset:])
    return delta


def _fwd_contrast(proj_ipa, proj_r1, W_r1_to_ipa, lookahead):
    T = proj_ipa.shape[0]
    fwd_proj = proj_r1 @ W_r1_to_ipa
    contrast = np.full(T, np.nan)
    if T <= lookahead:
        return contrast
    n = T - lookahead
    fp = fwd_proj[:n]
    contrast[:n] = _cos_dist_pairs(fp, proj_ipa[:n]) - _cos_dist_pairs(
        fp, proj_ipa[lookahead:]
    )
    return contrast


def _bwd_contrast(proj_ipa, proj_l1, W_l1_to_ipa, lookbehind):
    T = proj_ipa.shape[0]
    bwd_proj = proj_l1 @ W_l1_to_ipa
    contrast = np.full(T, np.nan)
    if T <= lookbehind:
        return contrast
    bp = bwd_proj[lookbehind:]
    contrast[lookbehind:] = _cos_dist_pairs(
        bp, proj_ipa[lookbehind:]
    ) - _cos_dist_pairs(bp, proj_ipa[: T - lookbehind])
    return contrast


def _normalize_signal(signal, method):
    if method == "min":
        return signal - np.nanmin(signal)
    if method == "minmax":
        sig = np.asarray(signal, dtype=float).copy()
        finite = np.isfinite(sig)
        if finite.sum() < 2:
            return sig
        lo, hi = sig[finite].min(), sig[finite].max()
        if hi > lo:
            sig[finite] = (sig[finite] - lo) / (hi - lo)
        else:
            sig[finite] = 0.0
        return sig
    raise ValueError(f"Unknown norm_method: {method}")


def _combine_stacked(stacked, method):
    if method == "min":
        return np.prod(stacked, axis=0)
    with np.errstate(invalid="ignore"):
        return np.exp(np.mean(np.log(np.maximum(stacked, 1e-12)), axis=0))


def _shift_signal(signal, shift_frames):
    if shift_frames == 0:
        return signal.copy()

    shifted = np.full(signal.shape, np.nan)
    if abs(shift_frames) >= len(signal):
        return shifted

    if shift_frames > 0:
        shifted[shift_frames:] = signal[:-shift_frames]
    else:
        shifted[:shift_frames] = signal[-shift_frames:]
    return shifted


class Segmenter:
    COMBINED_SIGNALS = (
        "frame_delta",
        "fwd_delta",
        "bwd_delta",
        "fwd_contrast",
        "bwd_contrast",
        "mel_svf",
    )
    COMBINED_SIGNAL_KWARGS = {
        "frame_delta": {"offset": 2},
        "fwd_delta": {"offset": 2},
        "bwd_delta": {"offset": 1},
        "fwd_contrast": {"lookahead": 1},
        "bwd_contrast": {"lookbehind": 2},
        "mel_svf": {"left": 1, "right": 2},
    }
    COMBINED_SIGNAL_SHIFTS = {
        "frame_delta": 1,
        "fwd_delta": 1,
        "bwd_delta": 1,
        "fwd_contrast": 1,
        "bwd_contrast": -1,
        "mel_svf": 1,
    }
    COMBINED_DROP_K = 2
    COMBINED_PROMINENCE = 0.001

    @classmethod
    def default_hparams(cls):
        return {
            "use_combined": True,
            "combined_signals": list(cls.COMBINED_SIGNALS),
            "signal_kwargs": {
                k: dict(v) for k, v in cls.COMBINED_SIGNAL_KWARGS.items()
            },
            "signal_shifts": dict(cls.COMBINED_SIGNAL_SHIFTS),
            "drop_k": cls.COMBINED_DROP_K,
            "combined_prominence": cls.COMBINED_PROMINENCE,
            "norm_method": "min",
            "single_signal_name": "fwd_contrast",
            "single_signal_kwargs": {"lookahead": 1},
            "single_signal_shift": 1,
            "single_signal_prominence": 0.2,
            "snap_silence": True,
            "snap_tolerance": 2,
            "silence_threshold": 0.5,
        }

    def __init__(
        self,
        train_df=None,
        *,
        frame_shift,
        sr,
        mel_frame_shift_ms,
        _from_components=None,
        hparams=None,
    ):
        self.hparams = (
            dict(hparams) if hparams is not None else self.default_hparams()
        )
        self.frame_shift = int(frame_shift)
        self.sr = int(sr)
        self.mel_frame_shift_ms = int(mel_frame_shift_ms)

        if _from_components is not None:
            self.pv_ipa = _from_components["pv_ipa"]
            self.pv_l1 = _from_components["pv_l1"]
            self.pv_r1 = _from_components["pv_r1"]
            self.W_r1_to_ipa = np.asarray(_from_components["W_r1_to_ipa"])
            self.W_l1_to_ipa = np.asarray(_from_components["W_l1_to_ipa"])
            self.silence_handler = SilenceHandler(
                self.pv_ipa, threshold=self.hparams["silence_threshold"]
            )
            return

        assert (
            train_df is not None
        ), "Segmenter requires train_df unless _from_components is given."
        self._fit_from_df(train_df)
        self.silence_handler = SilenceHandler(
            self.pv_ipa, threshold=self.hparams["silence_threshold"]
        )

    def _fit_from_df(self, train_df):
        pv_train_df = train_df[~train_df.ipa.isna()]
        vocab = pv_train_df.ipa.unique().tolist()

        self.pv_ipa = PhonologicalVectors(
            pv_train_df, vocab, group_col="ipa"
        )
        self.pv_l1 = PhonologicalVectors(
            pv_train_df, vocab, group_col="l_1"
        )
        self.pv_r1 = PhonologicalVectors(
            pv_train_df, vocab, group_col="r_1"
        )

        df_sorted = train_df[~train_df.ipa.isna()].sort_values(
            ["audio_path", "min"]
        )
        same_utt = (
            df_sorted.audio_path.values[:-1]
            == df_sorted.audio_path.values[1:]
        )
        prev_feats = np.stack(df_sorted.feat.values[:-1][same_utt])
        curr_feats = np.stack(df_sorted.feat.values[1:][same_utt])

        proj_ipa_curr = self.pv_ipa.project_raw(curr_feats)
        proj_ipa_prev = self.pv_ipa.project_raw(prev_feats)
        proj_r1_prev = self.pv_r1.project_raw(prev_feats)
        proj_l1_next = self.pv_l1.project_raw(curr_feats)

        self.W_r1_to_ipa = np.linalg.lstsq(
            proj_r1_prev, proj_ipa_curr, rcond=None
        )[0]
        self.W_l1_to_ipa = np.linalg.lstsq(
            proj_l1_next, proj_ipa_prev, rcond=None
        )[0]

    @classmethod
    def fit(
        cls,
        train_df,
        *,
        frame_shift,
        sr,
        mel_frame_shift_ms,
        hparams=None,
    ):
        seg = cls.__new__(cls)
        seg.hparams = (
            dict(hparams) if hparams is not None else cls.default_hparams()
        )
        seg.frame_shift = int(frame_shift)
        seg.sr = int(sr)
        seg.mel_frame_shift_ms = int(mel_frame_shift_ms)
        seg._fit_from_df(train_df)
        seg.silence_handler = SilenceHandler(
            seg.pv_ipa, threshold=seg.hparams["silence_threshold"]
        )
        return seg

    @classmethod
    def from_artifact(cls, artifact):
        net_spec = artifact["net"]
        for key in ("frame_shift", "sr", "mel_frame_shift_ms"):
            assert (
                key in net_spec
            ), f"artifact['net'] missing required key '{key}'."
        components = {
            "pv_ipa": PhonologicalVectors.from_state(
                artifact["phonvecs"]["ipa"]
            ),
            "pv_l1": PhonologicalVectors.from_state(
                artifact["phonvecs"]["l_1"]
            ),
            "pv_r1": PhonologicalVectors.from_state(
                artifact["phonvecs"]["r_1"]
            ),
            "W_r1_to_ipa": artifact["regressors"]["W_r1_to_ipa"],
            "W_l1_to_ipa": artifact["regressors"]["W_l1_to_ipa"],
        }
        return cls(
            frame_shift=net_spec["frame_shift"],
            sr=net_spec["sr"],
            mel_frame_shift_ms=net_spec["mel_frame_shift_ms"],
            _from_components=components,
            hparams=artifact.get("hparams"),
        )

    def to_artifact(self, *, net_spec, tune_metrics=None):
        required = {"frame_shift", "sr", "mel_frame_shift_ms"}
        missing = required - set(net_spec)
        assert not missing, f"net_spec missing required keys: {missing}"
        return {
            "phonvecs": {
                "ipa": self.pv_ipa.to_state(),
                "l_1": self.pv_l1.to_state(),
                "r_1": self.pv_r1.to_state(),
            },
            "regressors": {
                "W_r1_to_ipa": self.W_r1_to_ipa,
                "W_l1_to_ipa": self.W_l1_to_ipa,
            },
            "hparams": dict(self.hparams),
            "net": dict(net_spec),
            "tune_metrics": tune_metrics,
        }

    def with_hparams(self, hparams_override):
        new = self.__class__.__new__(self.__class__)
        new.pv_ipa = self.pv_ipa
        new.pv_l1 = self.pv_l1
        new.pv_r1 = self.pv_r1
        new.W_r1_to_ipa = self.W_r1_to_ipa
        new.W_l1_to_ipa = self.W_l1_to_ipa
        new.silence_handler = self.silence_handler
        new.frame_shift = self.frame_shift
        new.sr = self.sr
        new.mel_frame_shift_ms = self.mel_frame_shift_ms
        new.hparams = {**self.hparams, **hparams_override}
        return new

    def _signal(self, name, proj_ipa, proj_r1, proj_l1, waveform_np, kwargs):
        if name == "frame_delta":
            return _delta(proj_ipa, kwargs["offset"])
        if name == "fwd_delta":
            return _delta(proj_r1, kwargs["offset"])
        if name == "bwd_delta":
            return _delta(proj_l1, kwargs["offset"])
        if name == "fwd_contrast":
            return _fwd_contrast(
                proj_ipa, proj_r1, self.W_r1_to_ipa, kwargs["lookahead"]
            )
        if name == "bwd_contrast":
            return _bwd_contrast(
                proj_ipa, proj_l1, self.W_l1_to_ipa, kwargs["lookbehind"]
            )
        if name == "mel_svf":
            return _mel_svf_signal(
                waveform_np,
                left=kwargs["left"],
                right=kwargs["right"],
                target_len=proj_ipa.shape[0],
                sr=self.sr,
                mel_frame_shift_ms=self.mel_frame_shift_ms,
            )
        raise ValueError(f"Unknown signal: {name}")

    def _combined_signal(self, proj_ipa, proj_r1, proj_l1, waveform_np):
        h = self.hparams
        norm = h["norm_method"]
        components = []
        for signal_name in h["combined_signals"]:
            sig = self._signal(
                signal_name,
                proj_ipa,
                proj_r1,
                proj_l1,
                waveform_np,
                kwargs=h["signal_kwargs"][signal_name],
            )
            shifted = _shift_signal(sig, h["signal_shifts"][signal_name])
            components.append(_normalize_signal(shifted, norm))

        stacked = np.stack(components, axis=0)
        if h["drop_k"] > 0:
            stacked = np.sort(stacked, axis=0)[h["drop_k"] :]
        return _combine_stacked(stacked, norm)

    def segment(
        self, net_feats, waveform_np, use_combined=None, snap_silence=None
    ):
        h = self.hparams
        if use_combined is None:
            use_combined = h["use_combined"]
        if snap_silence is None:
            snap_silence = h["snap_silence"]

        proj_ipa = self.pv_ipa.project_raw(net_feats)
        proj_r1 = self.pv_r1.project_raw(net_feats)
        proj_l1 = self.pv_l1.project_raw(net_feats)

        if use_combined:
            signal = self._combined_signal(
                proj_ipa, proj_r1, proj_l1, waveform_np
            )
            prominence = h["combined_prominence"]
        else:
            sig = self._signal(
                h["single_signal_name"],
                proj_ipa,
                proj_r1,
                proj_l1,
                waveform_np,
                kwargs=h["single_signal_kwargs"],
            )
            signal = _shift_signal(sig, h["single_signal_shift"])
            prominence = h["single_signal_prominence"]

        preds = find_peaks(signal, prominence=prominence)[0]
        if snap_silence:
            silence_mask = self.silence_handler.predict_silence_mask(
                net_feats, threshold=h["silence_threshold"]
            )
            preds = self.silence_handler.handle_silence(
                preds=preds,
                silence_mask=silence_mask,
                snap_tolerance=h["snap_tolerance"],
            )

        return preds

    def posteriogram(self, net_feats):
        """Phonological posteriogram: per-frame sigmoid projection onto the IPA vectors."""
        return self.pv_ipa.project(net_feats)
