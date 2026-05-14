"""HuggingFace-style loader for trained phonological-posteriogram models.

A trained model is a single artifact file (created by `Segmenter.to_artifact`
and torch.save'd) bundled together with a reference to the SSL encoder it
was fit on. The artifact's `net` spec carries `hf_repo`, `encoder_layer`,
`frame_shift`, `sr`, and `mel_frame_shift_ms`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .features import SSLEncoder
from .model import Segmenter

DEFAULT_ARTIFACT_FILENAME = "model.pt"


class PhonologicalPosteriogram:
    """End-to-end phonological-posteriogram model.

    Use `.from_pretrained(repo_id_or_path)` to load a trained model from the
    HuggingFace Hub or from a local directory/file. The class lazily loads
    the underlying SSL encoder so a `from_pretrained` call is cheap if you
    only need the segmenter components.
    """

    def __init__(
        self,
        segmenter: Segmenter,
        net_spec: dict,
        device: str = "cpu",
    ):
        self.segmenter = segmenter
        self.net_spec = dict(net_spec)
        self.device = device
        self._encoder: Optional[SSLEncoder] = None

    @property
    def encoder(self) -> SSLEncoder:
        """Lazily-loaded SSL encoder described by `net_spec`."""
        if self._encoder is None:
            self._encoder = SSLEncoder(
                hf_repo=self.net_spec["hf_repo"],
                encoder_layer=self.net_spec.get("encoder_layer", -1),
                sr=self.net_spec["sr"],
                device=self.device,
            )
        return self._encoder

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike],
        *,
        filename: str = DEFAULT_ARTIFACT_FILENAME,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        device: str = "cpu",
    ) -> "PhonologicalPosteriogram":
        """Load a trained model from the HuggingFace Hub or a local path.

        `model_name_or_path` can be:
          * a HuggingFace repo id (e.g. "user/phonpost-wavlm-large")
          * a local directory containing `filename`
          * a path to the artifact file itself
        """
        artifact_path = _resolve_artifact_path(
            model_name_or_path,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        segmenter = Segmenter.from_artifact(artifact)
        return cls(segmenter=segmenter, net_spec=artifact["net"], device=device)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        *,
        filename: str = DEFAULT_ARTIFACT_FILENAME,
        tune_metrics: Optional[dict] = None,
    ) -> Path:
        """Save the segmenter artifact to a directory (HF Hub-compatible layout)."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        artifact = self.segmenter.to_artifact(
            net_spec=self.net_spec, tune_metrics=tune_metrics
        )
        out = save_directory / filename
        torch.save(artifact, out)
        return out

    def to(self, device: str) -> "PhonologicalPosteriogram":
        self.device = device
        if self._encoder is not None:
            self._encoder.model.to(device)
            self._encoder.device = device
        return self

    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """Run the SSL encoder and return per-frame features."""
        return self.encoder(np.asarray(waveform, dtype=np.float32))

    def posteriogram(self, waveform: np.ndarray) -> np.ndarray:
        """Compute the phonological posteriogram for a single mono waveform."""
        feats = self.extract_features(waveform)
        return self.segmenter.posteriogram(feats)

    def segment(
        self,
        waveform: np.ndarray,
        *,
        use_combined: Optional[bool] = None,
        snap_silence: Optional[bool] = None,
    ) -> np.ndarray:
        """Predict boundary frame indices for a single mono waveform."""
        waveform = np.asarray(waveform, dtype=np.float32)
        feats = self.extract_features(waveform)
        return self.segmenter.segment(
            feats,
            waveform,
            use_combined=use_combined,
            snap_silence=snap_silence,
        )

    def segment_seconds(self, waveform: np.ndarray, **kwargs) -> np.ndarray:
        """Like `segment`, but return boundary times in seconds."""
        frames = self.segment(waveform, **kwargs)
        return frames * (self.net_spec["frame_shift"] / self.net_spec["sr"])


def _resolve_artifact_path(
    model_name_or_path,
    *,
    filename: str,
    revision: Optional[str],
    cache_dir: Optional[Union[str, os.PathLike]],
) -> str:
    p = Path(str(model_name_or_path))
    if p.is_file():
        return str(p)
    if p.is_dir():
        candidate = p / filename
        if not candidate.is_file():
            raise FileNotFoundError(
                f"No '{filename}' in directory '{p}'."
            )
        return str(candidate)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download models from the Hub. "
            "Install it with `pip install huggingface_hub`."
        ) from e

    return hf_hub_download(
        repo_id=str(model_name_or_path),
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
    )
