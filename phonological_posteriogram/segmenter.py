"""Phone-boundary segmentation algorithm over a PhonologicalPosteriogram.

:class:`Segmenter` carries only hyperparameters — no trained weights — so a
new configuration is constructed in microseconds and different algorithm
settings can be compared without refitting the posteriogram. A future
``Recognizer`` will follow the same shape (constructed over the same
:class:`~phonological_posteriogram.posteriogram.PhonologicalPosteriogram`).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
from scipy.signal import find_peaks

warnings.filterwarnings(
    "ignore", message="Support for mismatched key_padding_mask"
)


# --------------------------------------------------------------------------- #
# Signal helpers                                                               #
# --------------------------------------------------------------------------- #


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


def _mel_svf(mel_frames, left, right, distance="cosine"):
    mel_frames = np.asarray(mel_frames, dtype=float)
    n = mel_frames.shape[0]
    signal = np.full(n, np.nan)
    if n <= left + right:
        return signal
    a = mel_frames[: n - left - right]
    b = mel_frames[left + right :]
    signal[left : n - right] = _pair_distance(a, b, distance)
    finite = np.isfinite(signal)
    if finite.any():
        lo, hi = np.nanmin(signal), np.nanmax(signal)
        if hi > lo:
            signal[finite] = (signal[finite] - lo) / (hi - lo)
    return signal


def _mel_svf_signal(
    audio, left, right, target_len, *, sr, mel_frame_shift_ms, distance="cosine"
):
    mel = _melspec_kaldi(audio, sr=sr, frame_shift_ms=mel_frame_shift_ms)
    sig = _mel_svf(mel, left=left, right=right, distance=distance)
    if len(sig) == 0 or target_len == 0:
        return np.full(target_len, np.nan, dtype=np.float32)
    indices = np.round(np.linspace(0, len(sig) - 1, target_len)).astype(int)
    return sig[indices].astype(np.float32)


DISTANCES = ("cosine", "l2")


def _pair_distance(a, b, kind="cosine"):
    """Per-row distance between corresponding rows of ``a`` and ``b``.

    ``"cosine"``: 1 - cos_sim; pairs with a zero-norm vector fall back to 1
    (max distance). ``"l2"``: Euclidean ``||a - b||``.
    """
    if kind == "cosine":
        dots = np.sum(a * b, axis=1)
        norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        out = np.ones(len(a))
        valid = norms > 0
        out[valid] = 1.0 - dots[valid] / norms[valid]
        return out
    if kind == "l2":
        return np.linalg.norm(a - b, axis=1)
    raise ValueError(
        f"Unknown distance {kind!r}; choose one of {DISTANCES}."
    )


def _delta(proj, offset, distance="cosine"):
    T = proj.shape[0]
    delta = np.full(T, np.nan)
    if T <= offset:
        return delta
    delta[: T - offset] = _pair_distance(
        proj[:-offset], proj[offset:], distance
    )
    return delta


def _fwd_contrast(proj_ipa, proj_r1, W_r1_to_ipa, lookahead, distance="cosine"):
    T = proj_ipa.shape[0]
    fwd_proj = proj_r1 @ W_r1_to_ipa
    contrast = np.full(T, np.nan)
    if T <= lookahead:
        return contrast
    n = T - lookahead
    fp = fwd_proj[:n]
    contrast[:n] = _pair_distance(fp, proj_ipa[:n], distance) - _pair_distance(
        fp, proj_ipa[lookahead:], distance
    )
    return contrast


def _bwd_contrast(proj_ipa, proj_l1, W_l1_to_ipa, lookbehind, distance="cosine"):
    T = proj_ipa.shape[0]
    bwd_proj = proj_l1 @ W_l1_to_ipa
    contrast = np.full(T, np.nan)
    if T <= lookbehind:
        return contrast
    bp = bwd_proj[lookbehind:]
    contrast[lookbehind:] = _pair_distance(
        bp, proj_ipa[lookbehind:], distance
    ) - _pair_distance(bp, proj_ipa[: T - lookbehind], distance)
    return contrast


NORM_METHODS = ("none", "min", "minmax")


def _normalize_signal(signal, method):
    """Normalize one per-frame signal before stacking.

    ``"none"``: pass through unchanged. ``"min"``: subtract the (nan-)min so
    the lowest finite value is 0. ``"minmax"``: rescale finite values to
    [0, 1].
    """
    if method == "none":
        return np.asarray(signal, dtype=float).copy()
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
    raise ValueError(
        f"Unknown norm_method {method!r}; choose one of {NORM_METHODS}."
    )


COMBINE_METHODS = ("min", "logmeanexp")


def _combine_stacked(stacked, method):
    """Combine the stacked per-frame signals into one signal.

    ``"min"``: product over signals — a fuzzy-AND that stays at or below the
    per-frame minimum. ``"logmeanexp"``: geometric mean, exp(mean(log(.))) —
    a softer consensus.
    """
    if method == "min":
        return np.prod(stacked, axis=0)
    if method == "logmeanexp":
        with np.errstate(invalid="ignore"):
            return np.exp(
                np.mean(np.log(np.maximum(stacked, 1e-12)), axis=0)
            )
    raise ValueError(
        f"Unknown combine_method {method!r}; choose one of {COMBINE_METHODS}."
    )


def _copy_spec(spec):
    """Normalize + deep-copy a signal spec dict {name, kwargs, shift}."""
    return {
        "name": spec["name"],
        "kwargs": dict(spec.get("kwargs", {})),
        "shift": spec.get("shift", 0),
    }


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


# --------------------------------------------------------------------------- #
# Segmenter                                                                    #
# --------------------------------------------------------------------------- #


class Segmenter:
    """Phone-boundary algorithm over a fitted ``PhonologicalPosteriogram``.

    Holds no trained weights — only ``hparams`` plus the encoder's ``sr`` and
    ``frame_shift`` (needed by the mel-SVF auxiliary signal and frame↔time
    conversion). Constructing one is cheap, so hparam sweeps don't refit.
    """

    # A "signal spec" is a dict {"name", "kwargs", "shift"}. ``combined_signals``
    # is a *list* of specs, so the same signal type may appear more than once
    # with different kwargs (e.g. two fwd_contrast at different lookaheads).
    DEFAULT_COMBINED_SIGNALS = (
        {"name": "frame_delta", "kwargs": {"offset": 2}, "shift": 1},
        {"name": "fwd_delta", "kwargs": {"offset": 2}, "shift": 1},
        {"name": "bwd_delta", "kwargs": {"offset": 1}, "shift": 1},
        {"name": "fwd_contrast", "kwargs": {"lookahead": 1}, "shift": 1},
        {"name": "bwd_contrast", "kwargs": {"lookbehind": 2}, "shift": -1},
        {"name": "mel_svf", "kwargs": {"left": 1, "right": 2}, "shift": 1},
    )
    DEFAULT_SINGLE_SIGNAL = {
        "name": "fwd_contrast",
        "kwargs": {"lookahead": 1},
        "shift": 1,
    }
    COMBINED_DROP_K = 2
    COMBINED_PROMINENCE = 0.001
    SINGLE_PROMINENCE = 0.2

    @classmethod
    def default_hparams(cls):
        return {
            "use_combined": True,
            "combined_signals": [
                _copy_spec(s) for s in cls.DEFAULT_COMBINED_SIGNALS
            ],
            "drop_k": cls.COMBINED_DROP_K,
            "combined_prominence": cls.COMBINED_PROMINENCE,
            # How each signal is normalized before stacking
            # ("none"/"min"/"minmax")...
            "norm_method": "min",
            # ...and how the stacked signals are combined ("min"/"logmeanexp").
            "combine_method": "min",
            # Activation applied to the phonological-vector projections that
            # feed the boundary signals: "none" (raw) or "sigmoid".
            "activation": "none",
            "single_signal": _copy_spec(cls.DEFAULT_SINGLE_SIGNAL),
            "single_signal_prominence": cls.SINGLE_PROMINENCE,
            # Pairwise distance for all distance-based signals
            # (frame_delta, fwd/bwd contrast, mel_svf): "cosine" or "l2".
            "distance": "cosine",
            "snap_silence": True,
            "snap_tolerance": 2,
            "silence_threshold": 0.5,
            "mel_frame_shift_ms": 10,
        }

    def __init__(self, posteriogram, *, sr, frame_shift, hparams=None):
        self.posteriogram = posteriogram
        self.sr = int(sr)
        self.frame_shift = int(frame_shift)
        # Merge onto defaults so a partial dict (or an artifact saved before
        # a new hparam was added) still yields a complete config.
        self.hparams = {**self.default_hparams(), **(hparams or {})}

    def with_hparams(self, hparams_override):
        """Return a copy with ``hparams_override`` merged on top.

        Cheap — the posteriogram weights are shared, not copied.
        """
        return self.__class__(
            self.posteriogram,
            sr=self.sr,
            frame_shift=self.frame_shift,
            hparams={**self.hparams, **hparams_override},
        )

    def _signal(self, name, proj_ipa, proj_r1, proj_l1, waveform_np, kwargs):
        distance = self.hparams["distance"]
        if name == "frame_delta":
            return _delta(proj_ipa, kwargs["offset"], distance=distance)
        if name == "fwd_delta":
            return _delta(proj_r1, kwargs["offset"], distance=distance)
        if name == "bwd_delta":
            return _delta(proj_l1, kwargs["offset"], distance=distance)
        if name == "fwd_contrast":
            return _fwd_contrast(
                proj_ipa,
                proj_r1,
                self.posteriogram.W_r1_to_ipa,
                kwargs["lookahead"],
                distance=distance,
            )
        if name == "bwd_contrast":
            return _bwd_contrast(
                proj_ipa,
                proj_l1,
                self.posteriogram.W_l1_to_ipa,
                kwargs["lookbehind"],
                distance=distance,
            )
        if name == "mel_svf":
            return _mel_svf_signal(
                waveform_np,
                left=kwargs["left"],
                right=kwargs["right"],
                target_len=proj_ipa.shape[0],
                sr=self.sr,
                mel_frame_shift_ms=self.hparams["mel_frame_shift_ms"],
                distance=distance,
            )
        raise ValueError(f"Unknown signal: {name}")

    def _signal_from_spec(self, spec, proj_ipa, proj_r1, proj_l1, waveform_np):
        """Compute one shifted signal from a {name, kwargs, shift} spec."""
        sig = self._signal(
            spec["name"],
            proj_ipa,
            proj_r1,
            proj_l1,
            waveform_np,
            kwargs=spec.get("kwargs", {}),
        )
        return _shift_signal(sig, spec.get("shift", 0))

    def _combined_signal(self, proj_ipa, proj_r1, proj_l1, waveform_np):
        h = self.hparams
        norm = h["norm_method"]
        components = [
            _normalize_signal(
                self._signal_from_spec(
                    spec, proj_ipa, proj_r1, proj_l1, waveform_np
                ),
                norm,
            )
            for spec in h["combined_signals"]
        ]

        stacked = np.stack(components, axis=0)
        if h["drop_k"] >= len(stacked):
            raise ValueError(
                f"drop_k={h['drop_k']} is >= the number of combined signals "
                f"({len(stacked)}); nothing would be left to combine."
            )
        if h["drop_k"] > 0:
            stacked = np.sort(stacked, axis=0)[h["drop_k"] :]
        return _combine_stacked(stacked, h["combine_method"])

    def segment(
        self, net_feats, waveform_np, use_combined=None, snap_silence=None
    ):
        h = self.hparams
        if use_combined is None:
            use_combined = h["use_combined"]
        if snap_silence is None:
            snap_silence = h["snap_silence"]

        act = h["activation"]
        proj_ipa = self.posteriogram.project(net_feats, view="ipa", act=act)
        proj_r1 = self.posteriogram.project(net_feats, view="r_1", act=act)
        proj_l1 = self.posteriogram.project(net_feats, view="l_1", act=act)

        if use_combined:
            signal = self._combined_signal(
                proj_ipa, proj_r1, proj_l1, waveform_np
            )
            prominence = h["combined_prominence"]
        else:
            signal = self._signal_from_spec(
                h["single_signal"], proj_ipa, proj_r1, proj_l1, waveform_np
            )
            prominence = h["single_signal_prominence"]

        preds = find_peaks(signal, prominence=prominence)[0]
        if snap_silence:
            silence_mask = self.posteriogram.predict_silence_mask(
                net_feats, threshold=h["silence_threshold"]
            )
            preds = self._handle_silence(
                preds, silence_mask, snap_tolerance=h["snap_tolerance"]
            )

        return preds

    def _handle_silence(self, preds, silence_mask, snap_tolerance=1):
        """Suppress peaks inside silence and snap nearby peaks to silence
        edges, injecting a boundary at each silence onset/offset."""
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

        return np.unique(
            np.concatenate(
                [preds[keep], np.array(silence_boundaries, dtype=int)]
            )
        )
