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
        """Map times (seconds) to feature-frame indices using the model's
        own conv-stack accounting."""
        samples = np.asarray(times_seconds, dtype=np.float64) * self.sr
        samples = np.clip(samples, 0, None).astype(np.int64)
        out = self.model._get_feat_extract_output_lengths(
            torch.as_tensor(samples, dtype=torch.long)
        )
        if isinstance(out, torch.Tensor):
            out = out.cpu().numpy()
        return np.clip(out, 0, None).astype(np.int64)
