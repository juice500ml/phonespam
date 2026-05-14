"""Smoke tests that don't require any model downloads or audio files."""

from pathlib import Path

import numpy as np
import torch

import phonological_posteriogram as pp
from phonological_posteriogram.model import (
    PhonologicalVectors,
    Segmenter,
    SilenceHandler,
)
from phonological_posteriogram.pretrained import PhonologicalPosteriogram


def test_public_api_exposed():
    assert pp.PhonologicalPosteriogram is PhonologicalPosteriogram
    assert pp.Segmenter is Segmenter
    assert hasattr(pp, "__version__")


def _make_logreg_state(in_dim=4):
    return {
        "_class_name": "LogisticRegression",
        "coef_": np.zeros((1, in_dim), dtype=np.float32),
        "intercept_": np.zeros((1,), dtype=np.float32),
        "classes_": np.array([0, 1]),
    }


def _make_phonvec_state(in_dim=4, n_feat=3):
    return {
        "featnames": [f"f{i}" for i in range(n_feat)],
        "featmap": {"a": [1, 0, 1], "b": [0, 1, 0]},
        "pos_vecs": np.zeros((n_feat, in_dim), dtype=np.float32),
        "zero_vecs": np.zeros((n_feat, in_dim), dtype=np.float32),
        "scales": np.ones((n_feat,), dtype=np.float32),
        "biases": np.zeros((n_feat,), dtype=np.float32),
    }


def _make_artifact(in_dim=4, n_feat=3):
    return {
        "phonvecs": {
            "ipa": _make_phonvec_state(in_dim, n_feat),
            "l_1": _make_phonvec_state(in_dim, n_feat),
            "r_1": _make_phonvec_state(in_dim, n_feat),
        },
        "regressors": {
            "W_r1_to_ipa": np.eye(n_feat, dtype=np.float32),
            "W_l1_to_ipa": np.eye(n_feat, dtype=np.float32),
        },
        "silence_detector": _make_logreg_state(in_dim),
        "hparams": Segmenter.default_hparams(),
        "net": {
            "hf_repo": "microsoft/wavlm-large",
            "encoder_layer": -1,
            "frame_shift": 320,
            "sr": 16000,
            "mel_frame_shift_ms": 10,
        },
    }


def test_silence_handler_state_roundtrip():
    state = _make_logreg_state()
    handler = SilenceHandler.from_state(state)
    out = handler.to_state()
    np.testing.assert_array_equal(out["coef_"], state["coef_"])
    np.testing.assert_array_equal(out["classes_"], state["classes_"])


def test_phonvectors_state_roundtrip():
    state = _make_phonvec_state()
    pv = PhonologicalVectors.from_state(state)
    assert pv.featnames == state["featnames"]
    assert pv.pos_vecs.shape == state["pos_vecs"].shape


def test_segmenter_from_artifact():
    artifact = _make_artifact()
    seg = Segmenter.from_artifact(artifact)
    assert seg.frame_shift == 320
    assert seg.sr == 16000


def test_pretrained_save_and_load_roundtrip(tmp_path: Path):
    artifact = _make_artifact()
    artifact_path = tmp_path / "model.pt"
    torch.save(artifact, artifact_path)

    model = PhonologicalPosteriogram.from_pretrained(tmp_path)
    assert model.net_spec["sr"] == 16000
    assert model.net_spec["hf_repo"] == "microsoft/wavlm-large"

    out_dir = tmp_path / "saved"
    out = model.save_pretrained(out_dir)
    assert out.exists()
    reloaded = PhonologicalPosteriogram.from_pretrained(out_dir)
    assert reloaded.net_spec == model.net_spec


def test_from_pretrained_accepts_file_path(tmp_path: Path):
    artifact = _make_artifact()
    artifact_path = tmp_path / "custom.pt"
    torch.save(artifact, artifact_path)
    model = PhonologicalPosteriogram.from_pretrained(artifact_path)
    assert model.segmenter.frame_shift == 320


def test_segmenter_default_hparams_self_consistent():
    h = Segmenter.default_hparams()
    for name in h["combined_signals"]:
        assert name in h["signal_kwargs"]
        assert name in h["signal_shifts"]
