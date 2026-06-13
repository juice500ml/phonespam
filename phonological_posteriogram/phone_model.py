"""HuggingFace-style end-to-end phone model.

A :class:`PhoneModel` bundles an SSL speech encoder with a fitted
:class:`~phonological_posteriogram.posteriogram.PhonologicalPosteriogram`
(the trained weights), a default set of algorithm hyperparameters, and an
embedded :class:`~phonological_posteriogram.recognizer.Recognizer`. Load a
model with :meth:`from_pretrained`, then drive the pieces directly:
:meth:`load_audio` / :meth:`extract_features`, the shared
:attr:`posteriogram`, and :meth:`segmenter` (a cheap factory that builds a
fresh :class:`~phonological_posteriogram.segmenter.Segmenter` over the shared
posteriogram, e.g. for hparam sweeps).

A saved model is a single artifact file (``torch.save``'d dict)::

    {
        "posteriogram": {...},  # PhonologicalPosteriogram.to_state()
        "hparams":      {...},  # the tuned/blessed algorithm config
        "net":          {"hf_repo", "encoder_layer", "sr"},
    }
"""

from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np
import torch

from .features import SSLEncoder
from .posteriogram import PhonologicalPosteriogram
from .recognizer import Recognizer
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
        hparams: dict | None = None,
        device: str = "cpu",
    ):
        self.posteriogram = posteriogram
        self.net_spec = dict(net_spec)
        self.hparams = dict(hparams) if hparams is not None else Segmenter.default_hparams()
        if hparams is None and not {"closure+", "release+"}.issubset(self.posteriogram.featnames):
            self.hparams["drop_closure_release"] = False
        self.device = device
        self._encoder: SSLEncoder | None = None
        self.recognizer = Recognizer(
            featnames=self.posteriogram.featnames,
            featmap=self.posteriogram.views["ipa"].featmap,
        )

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
        model_name_or_path: str | os.PathLike,
        *,
        filename: str = DEFAULT_ARTIFACT_FILENAME,
        revision: str | None = None,
        cache_dir: str | os.PathLike | None = None,
        device: str = "cpu",
    ) -> PhoneModel:
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
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        posteriogram = PhonologicalPosteriogram.from_state(artifact["posteriogram"])
        return cls(
            posteriogram=posteriogram,
            net_spec=artifact["net"],
            hparams=artifact.get("hparams"),
            device=device,
        )

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        *,
        filename: str = DEFAULT_ARTIFACT_FILENAME,
        tune_metrics: dict | None = None,
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

    def push_to_hub(
        self,
        repo_id: str,
        *,
        filename: str = DEFAULT_ARTIFACT_FILENAME,
        private: bool = False,
        commit_message: str | None = None,
        tune_metrics: dict | None = None,
        token: str | None = None,
    ) -> str:
        """Save the artifact and push it to a HuggingFace Hub model repo.

        Creates ``repo_id`` (public unless ``private=True``) if it doesn't
        exist, writes the artifact to a temp dir, and uploads it via
        :class:`huggingface_hub.HfApi`. Returns the repo URL.

        Args:
            repo_id: ``"org/name"`` — pushed as ``filename`` at the repo root.
            filename: Artifact filename inside the repo (default ``model.pt``).
            private: When the repo is created on this push, mark it private.
            commit_message: Override the auto commit message.
            tune_metrics: Optional sweep metrics dict to record in the artifact.
            token: Override the HF auth token (default uses the cached login).
        """
        import tempfile

        from huggingface_hub import HfApi, create_repo

        create_repo(
            repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
            token=token,
        )
        api = HfApi(token=token)
        with tempfile.TemporaryDirectory() as td:
            self.save_pretrained(td, filename=filename, tune_metrics=tune_metrics)
            api.upload_folder(
                folder_path=td,
                repo_id=repo_id,
                repo_type="model",
                commit_message=(commit_message or f"Upload PhoneModel artifact ({filename})"),
            )
        return f"https://huggingface.co/{repo_id}"

    def to(self, device: str) -> PhoneModel:
        self.device = device
        if self._encoder is not None:
            self._encoder.model.to(device)
            self._encoder.device = device
        return self

    # -- algorithm factories --------------------------------------------- #

    def segmenter(self, hparam_overrides: dict | None = None) -> Segmenter:
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
            hparams=hparams,
        )

    # -- end-to-end convenience ------------------------------------------ #

    def load_audio(self, path: str | os.PathLike) -> np.ndarray:
        """Load an audio file as a mono float32 waveform at the model's sr.

        Thin wrapper around ``librosa.load`` that locks in the conventions
        the model expects (mono, ``self.net_spec["sr"]``).
        """
        y, _ = librosa.load(str(path), sr=self.net_spec["sr"], mono=True)
        return y.astype(np.float32)

    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """Run the SSL encoder and return per-frame features."""
        return self.encoder(np.asarray(waveform, dtype=np.float32))


def _resolve_artifact_path(
    model_name_or_path,
    *,
    filename: str,
    revision: str | None,
    cache_dir: str | os.PathLike | None,
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
