"""HuggingFace-style end-to-end phone model.

A :class:`PhoneModel` bundles an SSL speech encoder with a fitted
:class:`~phonological_posteriogram.posteriogram.PhonologicalPosteriogram`
(the trained weights) and a default set of algorithm hyperparameters. From
it you build cheap, swappable algorithm objects — :meth:`PhoneModel.segmenter`
today, a ``Recognizer`` later — that all share the same posteriogram.

A saved model is a single artifact file (``torch.save``'d dict)::

    {
        "posteriogram": {...},  # PhonologicalPosteriogram.to_state()
        "hparams":      {...},  # the tuned/blessed algorithm config
        "net":          {"hf_repo", "encoder_layer", "frame_shift", "sr"},
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .features import SSLEncoder
from .posteriogram import PhonologicalPosteriogram
from .segmenter import Segmenter

DEFAULT_ARTIFACT_FILENAME = "model.pt"


class PhoneModel:
    """SSL encoder + fitted PhonologicalPosteriogram + default hparams.

    Use :meth:`from_pretrained` to load. The SSL encoder is loaded lazily, so
    constructing or loading a ``PhoneModel`` is cheap if you only need the
    posteriogram weights or want to inspect/save the artifact.
    """

    def __init__(
        self,
        posteriogram: PhonologicalPosteriogram,
        net_spec: dict,
        hparams: Optional[dict] = None,
        device: str = "cpu",
    ):
        self.posteriogram = posteriogram
        self.net_spec = dict(net_spec)
        self.hparams = (
            dict(hparams) if hparams is not None else Segmenter.default_hparams()
        )
        self.device = device
        self._encoder: Optional[SSLEncoder] = None

    @property
    def encoder(self) -> SSLEncoder:
        """Lazily-loaded SSL encoder described by ``net_spec``."""
        if self._encoder is None:
            self._encoder = SSLEncoder(
                hf_repo=self.net_spec["hf_repo"],
                encoder_layer=self.net_spec.get("encoder_layer", -1),
                sr=self.net_spec["sr"],
                device=self.device,
            )
        return self._encoder

    # -- loading / saving ------------------------------------------------- #

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike],
        *,
        filename: str = DEFAULT_ARTIFACT_FILENAME,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        device: str = "cpu",
    ) -> "PhoneModel":
        """Load a trained model from the HuggingFace Hub or a local path.

        ``model_name_or_path`` may be a HuggingFace repo id, a local directory
        containing ``filename``, or a path to the artifact file itself.
        """
        artifact_path = _resolve_artifact_path(
            model_name_or_path,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
        artifact = torch.load(
            artifact_path, map_location="cpu", weights_only=False
        )
        posteriogram = PhonologicalPosteriogram.from_state(
            artifact["posteriogram"]
        )
        return cls(
            posteriogram=posteriogram,
            net_spec=artifact["net"],
            hparams=artifact.get("hparams"),
            device=device,
        )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        *,
        filename: str = DEFAULT_ARTIFACT_FILENAME,
        tune_metrics: Optional[dict] = None,
    ) -> Path:
        """Save the model artifact to a directory (HF Hub-compatible layout)."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        artifact = {
            "posteriogram": self.posteriogram.to_state(),
            "hparams": dict(self.hparams),
            "net": dict(self.net_spec),
            "tune_metrics": tune_metrics,
        }
        out = save_directory / filename
        torch.save(artifact, out)
        return out

    def to(self, device: str) -> "PhoneModel":
        self.device = device
        if self._encoder is not None:
            self._encoder.model.to(device)
            self._encoder.device = device
        return self

    # -- algorithm factories --------------------------------------------- #

    def segmenter(self, hparam_overrides: Optional[dict] = None) -> Segmenter:
        """Build a :class:`Segmenter` over this model's posteriogram.

        ``hparam_overrides`` is merged on top of the model's default
        ``hparams``. Cheap to call repeatedly with different settings — the
        posteriogram weights are shared, never refit.
        """
        hparams = dict(self.hparams)
        if hparam_overrides:
            hparams.update(hparam_overrides)
        return Segmenter(
            self.posteriogram,
            sr=self.net_spec["sr"],
            frame_shift=self.net_spec["frame_shift"],
            hparams=hparams,
        )

    # -- end-to-end convenience ------------------------------------------ #

    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """Run the SSL encoder and return per-frame features."""
        return self.encoder(np.asarray(waveform, dtype=np.float32))

    def compute_posteriogram(self, waveform: np.ndarray) -> np.ndarray:
        """Per-frame phonological posteriogram for a single mono waveform."""
        feats = self.extract_features(waveform)
        return self.posteriogram.project(feats, view="ipa", act="sigmoid")

    def segment(
        self,
        waveform: np.ndarray,
        *,
        hparam_overrides: Optional[dict] = None,
        use_combined: Optional[bool] = None,
        snap_silence: Optional[bool] = None,
    ) -> np.ndarray:
        """Predict boundary frame indices for a single mono waveform."""
        waveform = np.asarray(waveform, dtype=np.float32)
        feats = self.extract_features(waveform)
        return self.segmenter(hparam_overrides).segment(
            feats,
            waveform,
            use_combined=use_combined,
            snap_silence=snap_silence,
        )

    def segment_seconds(self, waveform: np.ndarray, **kwargs) -> np.ndarray:
        """Like :meth:`segment`, but return boundary times in seconds."""
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
            raise FileNotFoundError(f"No '{filename}' in directory '{p}'.")
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
