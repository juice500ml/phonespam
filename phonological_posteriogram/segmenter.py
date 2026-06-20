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

warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")


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


def _mel_svf_signal(audio, left, right, target_len, *, sr, mel_frame_shift_ms, distance="cosine"):
    mel = _melspec_kaldi(audio, sr=sr, frame_shift_ms=mel_frame_shift_ms)
    sig = _mel_svf(mel, left=left, right=right, distance=distance)
    out = np.full(target_len, np.nan, dtype=np.float32)
    if len(sig) == 0 or target_len == 0:
        return out
    # Mel frames run at half the S3M frame shift (10ms vs 20ms), so S3M frame t
    # is exactly mel frame 2t: take every other mel frame.
    sig = sig[::2]
    n = min(len(sig), target_len)
    out[:n] = sig[:n]
    return out


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
    raise ValueError(f"Unknown distance {kind!r}; choose one of {DISTANCES}.")


def _delta(proj, offset, distance="cosine"):
    T = proj.shape[0]
    delta = np.full(T, np.nan)
    if T <= offset:
        return delta
    delta[: T - offset] = _pair_distance(proj[:-offset], proj[offset:], distance)
    return delta


def _bwd_contrast(proj_ipa, bwd_proj, lookbehind, distance="cosine"):
    """dist(bwd_proj[u], ipa[u]) - dist(bwd_proj[u], ipa[u-lookbehind]).

    ``bwd_proj = feats @ W_bwd`` is the regressor's guess of the *preceding*
    phone's projection.
    """
    T = proj_ipa.shape[0]
    contrast = np.full(T, np.nan)
    if T <= lookbehind:
        return contrast
    bp = bwd_proj[lookbehind:]
    contrast[lookbehind:] = _pair_distance(bp, proj_ipa[lookbehind:], distance) - _pair_distance(
        bp, proj_ipa[: T - lookbehind], distance
    )
    return contrast


# Fixed (utterance-invariant) lower bound per signal family, subtracted so
# every component is >= 0 before the product. A cosine distance is in [0, 2];
# a contrast is a difference of two cosine distances in [-2, 2]; mel_svf is
# already min-max normalised to [0, 1].
SIGNAL_FAMILY_FLOOR = {
    "frame_delta": 0.0,
    "bwd_contrast": -2.0,
    "mel_svf": 0.0,
}


def _fixed_floor_norm(signal, family):
    """Shift a component to its theoretical floor so it is non-negative."""
    return signal - SIGNAL_FAMILY_FLOOR[family]


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

    Holds no trained weights — only ``hparams`` plus the encoder's ``sr``
    (needed by the mel-SVF auxiliary signal). Constructing one is cheap, so
    hparam sweeps don't refit.
    """

    # A "signal spec" is a dict {"name", "kwargs", "shift"}. ``combined_signals``
    # is a *list* of specs, so the same signal type may appear more than once
    # with different kwargs (e.g. frame_delta at three window widths). Each
    # component is shifted to its theoretical floor (SIGNAL_FAMILY_FLOOR) so it
    # is non-negative, and the components are multiplied — a fuzzy-AND boundary
    # signal. The default is the best-performing config:
    # frame_delta(3 widths) + bwd_contrast(3 widths) + mel_svf.
    DEFAULT_COMBINED_SIGNALS = (
        {"name": "frame_delta", "kwargs": {"offset": 3}, "shift": 2},
        {"name": "frame_delta", "kwargs": {"offset": 2}, "shift": 1},
        {"name": "frame_delta", "kwargs": {"offset": 1}, "shift": 1},
        {"name": "bwd_contrast", "kwargs": {"lookbehind": 2}, "shift": -1},
        {"name": "bwd_contrast", "kwargs": {"lookbehind": 3}, "shift": -1},
        {"name": "bwd_contrast", "kwargs": {"lookbehind": 1}, "shift": 0},
        {"name": "mel_svf", "kwargs": {"left": 2, "right": 1}, "shift": 0},
    )
    COMBINED_PROMINENCE = 0.001

    @classmethod
    def default_hparams(cls):
        return {
            "combined_signals": [_copy_spec(s) for s in cls.DEFAULT_COMBINED_SIGNALS],
            "combined_prominence": cls.COMBINED_PROMINENCE,
            # Activation applied to the phonological-vector projections that
            # feed the boundary signals: "none" (raw) or "sigmoid".
            "activation": "none",
            # Pairwise distance for all distance-based signals
            # (frame_delta, fwd/bwd contrast, mel_svf): "cosine" or "l2".
            "distance": "cosine",
            # Drop predicted boundaries at a stop/affricate closure->release
            # transition (closure+ before, release+ after, both > threshold).
            "drop_closure_release": True,
            "closure_release_threshold": 0.5,
            "snap_silence": True,
            "snap_tolerance": 2,
            "silence_threshold": 0.5,
            "mel_frame_shift_ms": 10,
        }

    def __init__(self, posteriogram, *, sr, hparams=None):
        self.posteriogram = posteriogram
        self.sr = int(sr)
        # Merge onto defaults so a partial dict (or an artifact saved before
        # a new hparam was added) still yields a complete config.
        self.hparams = {**self.default_hparams(), **(hparams or {})}
        if (
            hparams is not None
            and "drop_closure_release" not in hparams
            and not {"closure+", "release+"}.issubset(posteriogram.featnames)
        ):
            self.hparams["drop_closure_release"] = False

    def with_hparams(self, hparams_override):
        """Return a copy with ``hparams_override`` merged on top.

        Cheap — the posteriogram weights are shared, not copied.
        """
        return self.__class__(
            self.posteriogram,
            sr=self.sr,
            hparams={**self.hparams, **hparams_override},
        )

    def _signal(
        self,
        name,
        proj_ipa,
        bwd_proj,
        waveform_np,
        kwargs,
    ):
        distance = self.hparams["distance"]
        if name == "frame_delta":
            return _delta(proj_ipa, kwargs["offset"], distance=distance)
        if name == "bwd_contrast":
            return _bwd_contrast(proj_ipa, bwd_proj, kwargs["lookbehind"], distance=distance)
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

    def _signal_from_spec(
        self,
        spec,
        proj_ipa,
        bwd_proj,
        waveform_np,
    ):
        """Compute one shifted signal from a {name, kwargs, shift} spec."""
        sig = self._signal(
            spec["name"],
            proj_ipa,
            bwd_proj,
            waveform_np,
            kwargs=spec.get("kwargs", {}),
        )
        return _shift_signal(sig, spec.get("shift", 0))

    def _combined_signal(
        self,
        proj_ipa,
        bwd_proj,
        waveform_np,
    ):
        """Fixed-floor product of the per-spec signals (fuzzy-AND)."""
        h = self.hparams
        components = [
            _fixed_floor_norm(
                self._signal_from_spec(
                    spec,
                    proj_ipa,
                    bwd_proj,
                    waveform_np,
                ),
                spec["name"],
            )
            for spec in h["combined_signals"]
        ]
        stacked = np.stack(components, axis=0)
        with np.errstate(invalid="ignore"):
            return np.prod(stacked, axis=0)

    def segment(self, net_feats, waveform_np, snap_silence=None):
        h = self.hparams
        if snap_silence is None:
            snap_silence = h["snap_silence"]

        act = h["activation"]
        proj_ipa = self.posteriogram.project(net_feats, view="ipa", act=act)
        bwd_proj = net_feats @ self.posteriogram.W_bwd

        signal = self._combined_signal(proj_ipa, bwd_proj, waveform_np)
        preds = find_peaks(signal, prominence=h["combined_prominence"])[0]

        if h["drop_closure_release"]:
            preds = self._drop_closure_release_peaks(preds, proj_ipa)

        if snap_silence:
            silence_mask = self.posteriogram.predict_silence_mask(
                net_feats, threshold=h["silence_threshold"]
            )
            preds = self._handle_silence(preds, silence_mask, snap_tolerance=h["snap_tolerance"])

        return preds

    def _drop_closure_release_peaks(self, preds, proj_ipa):
        """Drop predicted boundaries that fall on a closure->release merge.

        TIMIT writes stops/affricates as closure + release; the projection's
        ``closure+`` / ``release+`` channels detect them. A peak whose
        preceding frame is closure-like and whose own frame is release-like is
        an internal stop boundary, not a phone boundary, so it is dropped.
        """
        thr = self.hparams["closure_release_threshold"]
        featnames = self.posteriogram.featnames
        closure_idx = featnames.index("closure+")
        release_idx = featnames.index("release+")
        keep = []
        for peak in preds:
            if peak <= 0 or peak >= len(proj_ipa):
                keep.append(peak)
                continue
            if proj_ipa[peak - 1, closure_idx] > thr and proj_ipa[peak, release_idx] > thr:
                continue
            keep.append(peak)
        return np.asarray(keep, dtype=preds.dtype)

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
                nearby = preds[(preds >= s - snap_tolerance) & (preds <= s + snap_tolerance)]
                if len(nearby) > 0:
                    outside = nearby[nearby <= s]
                    silence_boundaries.append(outside.min() if len(outside) > 0 else nearby.min())
                    snapped.update(nearby.tolist())
                else:
                    silence_boundaries.append(s)

            if e < n_frames:
                nearby = preds[(preds >= e - snap_tolerance) & (preds <= e + snap_tolerance)]
                if len(nearby) > 0:
                    outside = nearby[nearby >= e]
                    silence_boundaries.append(outside.max() if len(outside) > 0 else nearby.max())
                    snapped.update(nearby.tolist())
                else:
                    silence_boundaries.append(e)

        keep = ~silence_mask[preds]
        for i, p in enumerate(preds):
            if p in snapped:
                keep[i] = False

        return np.unique(np.concatenate([preds[keep], np.array(silence_boundaries, dtype=int)]))
