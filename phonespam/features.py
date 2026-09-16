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

        self.stride_size = int(np.prod(self.model.config.conv_stride))
        rf, pref = 1, 1
        for k, s in zip(self.model.config.conv_kernel, self.model.config.conv_stride, strict=True):
            rf += (int(k) - 1) * pref
            pref *= int(s)
        self.window_size = int(rf)

    @torch.inference_mode()
    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Run the encoder on a 1D mono float waveform sampled at self.sr."""
        # Pre-pad by half the receptive-field overhang so each output frame is
        # *centered* on its stride window in the input signal. With this pad,
        # frame idx <-> input sample idx*stride exactly, so frame_to_time is a
        # plain idx*stride/sr (matching the convention used at fit time).
        pad = (self.window_size - self.stride_size) // 2
        waveform = np.pad(np.asarray(waveform, dtype=np.float32), (pad, pad))
        x = self.processor(
            raw_speech=[waveform],
            sampling_rate=self.processor.sampling_rate,
            padding=False,
            return_tensors="pt",
        )
        x = {k: t.to(self.device) for k, t in x.items()}

        if self.encoder_layer in (-1, self.model.config.num_hidden_layers):
            # Read the final layer from last_hidden_state: transformers 5 records
            # hidden_states[-1] before the encoder's final LayerNorm (4.x recorded
            # it after), so only last_hidden_state is consistent across versions.
            out = self.model(**x)
            feats = out.last_hidden_state
        else:
            out = self.model(output_hidden_states=True, **x)
            feats = out.hidden_states[self.encoder_layer]
        return feats[0].cpu().numpy()

    def time_to_frame(self, times_seconds: np.ndarray) -> np.ndarray:
        """Map times (seconds) to feature-frame indices.

        Returns the frame whose centered window ``[k*stride, (k+1)*stride)``
        contains the time, i.e. ``floor(t*sr/stride)``. This matches the
        centered padding applied in :meth:`__call__` (frame k is centered on
        input sample ``(k+0.5)*stride``), so the mapping is a plain divide by
        the stride with no receptive-field offset.
        """
        t = np.asarray(times_seconds, dtype=np.float64)
        idx = np.floor(t * self.sr / self.stride_size).astype(np.int64)
        return np.clip(idx, 0, None)

    def frame_to_time(self, frame_indices: np.ndarray) -> np.ndarray:
        """Map feature-frame indices (0-indexed) to times in seconds."""
        idx = np.asarray(frame_indices, dtype=np.float64)
        return idx * self.stride_size / self.sr
