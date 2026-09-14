"""HuggingFace-style end-to-end phone model.

A :class:`PhoneModel` bundles an SSL speech encoder with a fitted
:class:`~phonespam.posteriogram.PhonologicalPosteriogram`
(the trained weights), a default set of algorithm hyperparameters, and an
embedded :class:`~phonespam.recognizer.Recognizer`. Load a
model with :meth:`from_pretrained`, then drive the pieces directly:
:meth:`load_audio` / :meth:`extract_features`, the shared
:attr:`posteriogram`, and :meth:`segmenter` (a cheap factory that builds a
fresh :class:`~phonespam.segmenter.Segmenter` over the shared
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
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import librosa
import numpy as np
import torch

from .features import SSLEncoder
from .posteriogram import PhonologicalPosteriogram
from .recognizer import Recognizer
from .segmenter import Segmenter

DEFAULT_ARTIFACT_FILENAME = "model.pt"


class Segment(NamedTuple):
    """One labelled phone segment. Times are in seconds."""

    start: float
    end: float
    label: str


class Spam:
    """An S3M-based Phonological Activation Map: the model's intermediate.

    One row per encoder frame, one column per phonological feature. This is
    what both the segmenter and the recognizer actually consume, so it is the
    thing to look at when you want to see *why* the model segmented or
    labelled the way it did.

    ``values`` are in ``[0, 1]`` when ``act="sigmoid"`` (the default) and are
    unbounded raw projections when ``act="none"``.

    Wraps the array rather than returning it bare so the feature names and
    frame times travel with it. ``np.asarray(spam)`` gives the plain matrix
    back, and the object can be handed directly to
    :meth:`Recognizer.recognize`.
    """

    __slots__ = ("values", "featnames", "times", "act")

    def __init__(self, values, featnames, times, act):
        self.values = np.asarray(values)
        self.featnames = tuple(featnames)
        self.times = np.asarray(times, dtype=float)
        self.act = act

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)

    def __len__(self):
        return len(self.values)

    @property
    def shape(self):
        return self.values.shape

    def __repr__(self):
        return (
            f"Spam(frames={self.shape[0]}, features={self.shape[1]}, "
            f"act={self.act!r}, duration={float(self.times[-1]) if len(self.times) else 0.0:.2f}s)"
        )

    def select(self, featnames) -> Spam:
        """Return a new :class:`Spam` with only ``featnames``, in that order.

        Saves looking up column positions by hand when plotting or comparing
        a handful of channels::

            spam.select(["silence+", "hi+", "strid+"])
        """
        index = {name: i for i, name in enumerate(self.featnames)}
        missing = [n for n in featnames if n not in index]
        if missing:
            raise KeyError(
                f"unknown feature name(s) {missing}; this model has "
                f"{len(self.featnames)}: {', '.join(sorted(self.featnames))}"
            )
        cols = [index[n] for n in featnames]
        return Spam(self.values[:, cols], tuple(featnames), self.times, self.act)

    def to_frame(self):
        """As a :class:`pandas.DataFrame` indexed by frame time in seconds."""
        import pandas as pd

        return pd.DataFrame(
            self.values,
            index=pd.Index(self.times, name="time"),
            columns=list(self.featnames),
        )


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
        self._recognizer: Recognizer | None = None
        self._segmenter: Segmenter | None = None

    # -- lazily-built components ------------------------------------------ #
    #
    # The encoder, recognizer and segmenter are all built on first use and
    # cached, so loading a model does no work you might not need -- the
    # encoder in particular pulls weights off the Hub. All three are
    # assignable if you want to swap in your own.

    @property
    def encoder(self) -> SSLEncoder:
        """SSL encoder described by ``net_spec``."""
        if self._encoder is None:
            self._encoder = SSLEncoder(
                hf_repo=self.net_spec["hf_repo"],
                encoder_layer=self.net_spec.get("encoder_layer", -1),
                sr=self.net_spec["sr"],
                device=self.device,
            )
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: SSLEncoder) -> None:
        self._encoder = encoder

    @property
    def recognizer(self) -> Recognizer:
        """Recognizer over this model's posteriogram.

        Built from the posteriogram's fitted feature map, which is quick; it
        is lazy for symmetry with :attr:`segmenter` and :attr:`encoder`, so
        that loading a model does no work you might not need.
        """
        if self._recognizer is None:
            self._recognizer = Recognizer(
                featnames=self.posteriogram.featnames,
                featmap=self.posteriogram.views["ipa"].featmap,
            )
        return self._recognizer

    @recognizer.setter
    def recognizer(self, recognizer: Recognizer) -> None:
        self._recognizer = recognizer

    @property
    def segmenter(self) -> Segmenter:
        """Segmenter over this model's posteriogram, using its ``hparams``.

        For a different configuration, copy it rather than mutating this one::

            seg = model.segmenter.with_hparams({"combined_prominence": 0.005})

        The copy shares the posteriogram weights, so sweeps never refit.
        """
        if self._segmenter is None:
            self._segmenter = Segmenter(
                self.posteriogram,
                sr=self.sr,
                hparams=self.hparams,
            )
        return self._segmenter

    @segmenter.setter
    def segmenter(self, segmenter: Segmenter) -> None:
        self._segmenter = segmenter

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

    # -- convenience accessors -------------------------------------------- #

    @property
    def sr(self) -> int:
        """Sample rate the model's features are produced at."""
        return int(self.net_spec["sr"])

    @property
    def frame_shift(self) -> int:
        """Encoder stride in input samples; one frame spans ``frame_shift/sr`` seconds.

        Reads the loaded encoder, so the first access pays for loading it.
        """
        return int(self.encoder.stride_size)

    # -- end-to-end convenience ------------------------------------------ #

    def load_audio(self, path: str | os.PathLike) -> np.ndarray:
        """Load an audio file as a mono float32 waveform at the model's sr.

        Thin wrapper around ``librosa.load`` that locks in the conventions
        the model expects (mono, ``self.sr``).
        """
        y, _ = librosa.load(str(path), sr=self.sr, mono=True)
        return y.astype(np.float32)

    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """Run the SSL encoder and return per-frame features."""
        return self.encoder(np.asarray(waveform, dtype=np.float32))

    def _waveform(self, audio) -> np.ndarray:
        """Accept a path or an in-memory waveform and return the waveform."""
        if isinstance(audio, (str, os.PathLike)):
            return self.load_audio(audio)
        return np.asarray(audio, dtype=np.float32)

    def boundaries(
        self,
        audio: str | os.PathLike | np.ndarray,
        *,
        hparam_overrides: dict | None = None,
        snap_silence: bool | None = None,
    ) -> np.ndarray:
        """Predicted phone boundaries for ``audio``, as **times in seconds**.

        ``audio`` is a path or an already-loaded waveform at :attr:`sr`.

        Note the units: :meth:`Segmenter.segment` is the lower-level call and
        returns *frame indices*. This returns seconds, which is what
        :meth:`Recognizer.recognize` and most evaluation code expect.
        """
        wav = self._waveform(audio)
        feats = self.extract_features(wav)
        segmenter = self.segmenter
        if hparam_overrides:
            segmenter = segmenter.with_hparams(hparam_overrides)
        frames = segmenter.segment(feats, wav, snap_silence=snap_silence)
        return self.encoder.frame_to_time(frames)

    def spam(
        self,
        audio: str | os.PathLike | np.ndarray,
        *,
        act: str = "sigmoid",
    ) -> Spam:
        """Phonological activation map for ``audio``.

        This is the model's intermediate representation, computed before any
        segmentation or recognition happens.

        Args:
            audio: Path to an audio file, or a mono waveform at :attr:`sr`.
            act: ``"sigmoid"`` for normalized activations in ``[0, 1]`` (what
                the recognizer scores), or ``"none"`` for the raw projections
                (what the segmenter's default configuration differences).

        Example:
            >>> spam = model.spam("utt.wav")
            >>> spam.select(["silence+", "hi+"]).to_frame().head()
        """
        return self.spam_from_features(self.extract_features(self._waveform(audio)), act=act)

    def spam_from_features(self, features: np.ndarray, *, act: str = "sigmoid") -> Spam:
        """:meth:`spam` for features you already extracted.

        Use this to avoid running the SSL encoder twice when you want both the
        activation map and a transcription of the same audio.
        """
        values = self.posteriogram.project(features, view="ipa", act=act)
        times = self.encoder.frame_to_time(np.arange(len(values)))
        return Spam(values, self.posteriogram.featnames, times, act)

    def transcribe(
        self,
        audio: str | os.PathLike | np.ndarray,
        *,
        vocab: Sequence[str] | None = None,
        hparam_overrides: dict | None = None,
        snap_silence: bool | None = None,
    ) -> list[Segment]:
        """Segment and label ``audio`` in one call.

        This is the whole pipeline: load audio, run the SSL encoder, project
        to the phonological posteriogram, find boundaries, and label each
        segment. It exists mainly so callers don't have to rediscover the
        conventions that tie the pieces together -- the sigmoid activation
        the recognizer expects, the frames-to-seconds conversion between
        segmenter and recognizer, and the model's own ``sr``/``frame_shift``.

        Args:
            audio: Path to an audio file, or a mono waveform at :attr:`sr`.
            vocab: Optional phone inventory to constrain the output to, e.g.
                ``phonespam.vocab_for_language("deu")``.
            hparam_overrides: Segmenter hyperparameters to override for this
                call only.
            snap_silence: Override the segmenter's silence-snapping setting.

        Returns:
            A list of :class:`Segment` ``(start, end, label)`` namedtuples,
            contiguous and in order, with times in seconds.

        Example:
            >>> model = PhoneModel.from_pretrained("juice500/wavlm-24-phonemodel")
            >>> for seg in model.transcribe("utt.wav"):
            ...     print(f"{seg.start:.2f}-{seg.end:.2f}  {seg.label}")
        """
        wav = self._waveform(audio)
        feats = self.extract_features(wav)
        # The recognizer scores a sigmoid posteriogram; raw projections would
        # silently shift a couple of percent of the labels.
        post = self.posteriogram.project(feats, view="ipa", act="sigmoid")

        segmenter = self.segmenter
        if hparam_overrides:
            segmenter = segmenter.with_hparams(hparam_overrides)
        frames = segmenter.segment(feats, wav, snap_silence=snap_silence)
        times = self.encoder.frame_to_time(frames)

        labels = self.recognizer.recognize(
            post, times, sr=self.sr, frame_shift=self.frame_shift, vocab=vocab
        )
        edges = np.concatenate(
            [[0.0], np.asarray(times, dtype=float), [len(feats) * self.frame_shift / self.sr]]
        )
        return [
            Segment(float(a), float(b), label)
            for a, b, label in zip(edges[:-1], edges[1:], labels, strict=True)
        ]


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
