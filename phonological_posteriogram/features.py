"""SSL feature extraction for inference.

A thin wrapper around `transformers.AutoModel` that loads a self-supervised
speech encoder by HuggingFace repo id and returns per-frame features at a
chosen hidden layer.

SSL encoders are trained at 16 kHz, but their conv stack just consumes raw
float samples, so feeding a higher-rate signal (e.g. 32 kHz) gives a finer
frame hop (320 samples / sr seconds per frame). We always tell the HF
processor its native rate so its validation check passes, and let `sr`
carry the *real* input rate.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor


class SSLEncoder:
    """Loads an SSL speech encoder and runs it on raw audio.

    `sr` is the sample rate of the audio you'll feed into `__call__`. It can
    differ from the model's native 16 kHz; the conv stack is sr-agnostic.
    """

    def __init__(
        self,
        hf_repo: str,
        encoder_layer: int = -1,
        sr: int = 16000,
        device: str = "cpu",
    ):
        self.hf_repo = hf_repo
        self.encoder_layer = int(encoder_layer)
        self.sr = int(sr)
        self.device = device
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(hf_repo)
        self.model = AutoModel.from_pretrained(hf_repo).to(device).eval()
        # Conv-stack arithmetic, used by time_to_frame / frame_to_time. For
        # wav2vec2-family stacks, `output_lengths(L) = L/stride - k_off_frames`
        # exactly in the large-L regime, so probing once is enough.
        self.stride = int(np.prod(self.model.config.conv_stride))
        probe_samples = 1 * 16000
        probe_frames = int(
            self.model._get_feat_extract_output_lengths(
                torch.as_tensor(probe_samples, dtype=torch.long)
            ).item()
        )
        self.k_eff_samples = probe_samples - probe_frames * self.stride
        # Receptive field (window) of the conv feature encoder, in input
        # samples. For a no-padding strided conv stack, output frame ``idx``
        # depends on input ``[idx*stride, idx*stride + window)``, so the
        # frame's temporal *center* is ``idx*stride + window/2``. Computed from
        # the conv config (a function of architecture only, not the audio):
        # W = 1 + sum_i (kernel_i - 1) * prod_{j<i} stride_j.
        rf, pref = 1, 1
        for k, s in zip(
            self.model.config.conv_kernel, self.model.config.conv_stride
        ):
            rf += (int(k) - 1) * pref
            pref *= int(s)
        self.window_samples = int(rf)

    @torch.inference_mode()
    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Run the encoder on a 1D mono float waveform sampled at self.sr."""
        x = self.processor(
            raw_speech=[waveform],
            # Native processor sr (16 kHz for all SSL speech encoders); the
            # validation just checks this matches its config, so we pass the
            # config value rather than the real input sr.
            sampling_rate=self.processor.sampling_rate,
            padding=False,
            return_tensors="pt",
        )
        x = {k: t.to(self.device) for k, t in x.items()}

        if self.encoder_layer == -1:
            out = self.model(**x)
            feats = out.last_hidden_state
        else:
            out = self.model(output_hidden_states=True, **x)
            feats = out.hidden_states[self.encoder_layer]
        return feats[0].cpu().numpy()

    def time_to_frame(self, times_seconds: np.ndarray) -> np.ndarray:
        """Map times (seconds) to the *count* of feature frames at that
        time, using the model's own conv-stack accounting.

        ``time_to_frame(t)`` is the length of the per-frame feature array
        produced by the encoder for the first ``t`` seconds of input
        audio. Equivalently: ``last_valid_frame_index + 1``.
        """
        samples = np.asarray(times_seconds, dtype=np.float64) * self.sr
        samples = np.clip(samples, 0, None).astype(np.int64)
        out = self.model._get_feat_extract_output_lengths(
            torch.as_tensor(samples, dtype=torch.long)
        )
        if isinstance(out, torch.Tensor):
            out = out.cpu().numpy()
        return np.clip(out, 0, None).astype(np.int64)

    def frame_to_time(self, frame_indices: np.ndarray) -> np.ndarray:
        """Map feature-frame indices (0-indexed) to the time (seconds) at
        the *center* of each frame's receptive-field window.

        For a no-padding strided conv stack, frame ``idx`` is computed from
        input samples ``[idx*stride, idx*stride + window)``, so its temporal
        center is ``idx*stride + window/2`` (window == receptive field). This
        is the acoustically correct location for a boundary detected at frame
        ``idx`` — unlike the frame's right edge or a half-*hop* offset, which
        bias every boundary late by ``(window - stride)/2`` and ``stride/2``
        respectively.
        """
        idx = np.asarray(frame_indices, dtype=np.float64)
        samples = idx * self.stride + self.window_samples / 2.0
        return samples / self.sr
