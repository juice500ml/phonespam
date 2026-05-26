"""HuggingFace-style end-to-end phone model.

A :class:`PhoneModel` bundles an SSL speech encoder with a fitted
:class:`~phonological_posteriogram.posteriogram.PhonologicalPosteriogram`
(the trained weights), a default set of algorithm hyperparameters, and an
embedded :class:`~phonological_posteriogram.recognizer.Recognizer`. The
public end-to-end entry point is :meth:`PhoneModel.recognize`; for hparam
sweeps, :meth:`segmenter` exposes a cheap factory that builds a fresh
:class:`~phonological_posteriogram.segmenter.Segmenter` over the shared
posteriogram.

A saved model is a single artifact file (``torch.save``'d dict)::

    {
        "posteriogram": {...},  # PhonologicalPosteriogram.to_state()
        "hparams":      {...},  # the tuned/blessed algorithm config
        "net":          {"hf_repo", "encoder_layer", "sr"},
    }
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import librosa
import torch

from .evaluation import SegmentationUnit
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
        self.recognizer = Recognizer(featnames=self.posteriogram.featnames)

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
            hparams=hparams,
        )

    # -- end-to-end convenience ------------------------------------------ #

    def load_audio(self, path: Union[str, os.PathLike]) -> np.ndarray:
        """Load an audio file as a mono float32 waveform at the model's sr.

        Thin wrapper around ``librosa.load`` that locks in the conventions
        the model expects (mono, ``self.net_spec["sr"]``).
        """
        y, _ = librosa.load(str(path), sr=self.net_spec["sr"], mono=True)
        return y.astype(np.float32)

    def _coerce_waveform(
        self,
        audio: Union[str, os.PathLike, np.ndarray],
        sr: Optional[int],
    ) -> np.ndarray:
        """Normalize ``audio`` (path-or-array) to a mono float32 waveform
        at the model's sample rate, resampling (with a warning) if needed.
        """
        if isinstance(audio, (str, os.PathLike)):
            if sr is not None:
                warnings.warn(
                    "`sr=` is ignored when `audio` is a file path; the file "
                    "is always resampled to the model's expected sample rate.",
                    stacklevel=3,
                )
            return self.load_audio(audio)

        waveform = np.asarray(audio, dtype=np.float32)
        target_sr = self.net_spec["sr"]
        if sr is not None and int(sr) != int(target_sr):
            warnings.warn(
                f"Input sample rate ({sr} Hz) does not match the model's "
                f"expected rate ({target_sr} Hz); resampling.",
                stacklevel=3,
            )
            import librosa

            waveform = librosa.resample(
                waveform, orig_sr=int(sr), target_sr=int(target_sr)
            ).astype(np.float32)
        return waveform

    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """Run the SSL encoder and return per-frame features."""
        return self.encoder(np.asarray(waveform, dtype=np.float32))

    def compute_posteriogram(self, waveform: np.ndarray) -> np.ndarray:
        """Per-frame phonological posteriogram for a single mono waveform."""
        feats = self.extract_features(waveform)
        return self.posteriogram.project(feats, view="ipa", act="sigmoid")

    def recognize(
        self,
        audio: Union[str, os.PathLike, np.ndarray],
        *,
        sr: Optional[int] = None,
        lang: Optional[str] = None,
        phoible_id: Optional[int] = None,
        phoneme: bool = False,
        vocab: Optional[Sequence[str]] = None,
    ) -> List[SegmentationUnit]:
        """End-to-end: load → encode → segment → per-segment recognize.

        Args:
            audio: either a path to an audio file (any librosa-readable
                format) or a 1D mono waveform array.
            sr: sample rate of ``audio`` if it is an array. When omitted the
                array is assumed to already be at the model's expected sr;
                when supplied and different, the waveform is resampled with
                a warning. Ignored when ``audio`` is a file path.
            vocab: optional explicit phone vocabulary to constrain the
                recognizer output to (panphon-known phones only).
            lang / phoible_id / phoneme: optional Phoible-inventory vocab
                constraint (mutually exclusive with ``vocab``).

        Returns a list of :class:`SegmentationUnit` with start/end in
        seconds (via :meth:`SSLEncoder.frame_to_time`).
        """
        waveform = self._coerce_waveform(audio, sr)
        feats = self.extract_features(waveform)
        boundaries = self.segmenter().segment(feats, waveform)
        posteriogram = self.posteriogram.project(
            feats, view="ipa", act="sigmoid"
        )
        triples = self.recognizer.recognize(
            posteriogram,
            boundaries,
            lang=lang,
            phoible_id=phoible_id,
            phoneme=phoneme,
            vocab=vocab,
        )
        if not triples:
            return []
        starts = self.encoder.frame_to_time(np.array([t[0] for t in triples]))
        ends = self.encoder.frame_to_time(np.array([t[1] for t in triples]))
        return [
            SegmentationUnit(
                float(starts[i]), float(ends[i]), triples[i][2]
            )
            for i in range(len(triples))
        ]


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
