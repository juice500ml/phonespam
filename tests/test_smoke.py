"""Smoke tests that don't require any model downloads or audio files."""

from pathlib import Path

import numpy as np
import pytest
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


def _make_phonvec_state(in_dim=4, n_feat=3, featnames=None):
    if featnames is None:
        featnames = ["speech+"] + [f"f{i}" for i in range(n_feat - 1)]
    return {
        "featnames": featnames,
        "featmap": {"_": [1, 0, 0], "a": [0, 1, 0], "b": [0, 0, 1]},
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
        "hparams": Segmenter.default_hparams(),
        "net": {
            "hf_repo": "microsoft/wavlm-large",
            "encoder_layer": -1,
            "frame_shift": 320,
            "sr": 16000,
            "mel_frame_shift_ms": 10,
        },
    }


def test_silence_handler_requires_speech_plus():
    pv_no_speech = PhonologicalVectors.from_state(
        _make_phonvec_state(featnames=["a+", "b+", "c+"])
    )
    with pytest.raises(ValueError, match="speech\\+"):
        SilenceHandler(pv_no_speech)


def test_silence_handler_uses_pv_ipa_speech_plus():
    """Silence handler thresholds the speech+ dim of pv_ipa.project(feats)."""
    in_dim = 4
    state = _make_phonvec_state(in_dim=in_dim)
    # Make speech+ unmistakable: pos_vec is a strong direction; everything
    # else is zero. Then a feat aligned with speech+ projects to ~1 there.
    state["pos_vecs"] = np.zeros((3, in_dim), dtype=np.float32)
    state["pos_vecs"][0] = np.array([10.0, 0, 0, 0], dtype=np.float32)
    pv = PhonologicalVectors.from_state(state)

    handler = SilenceHandler(pv, threshold=0.5)
    assert handler.speech_plus_idx == 0

    # 4 frames: silence | speech | speech | silence. The two speech frames
    # each have a non-silent neighbor, so _fill_gaps leaves them alone.
    feats = np.array(
        [
            [1.0, 0, 0, 0],
            [-1.0, 0, 0, 0],
            [-1.0, 0, 0, 0],
            [1.0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    mask = handler.predict_silence_mask(feats)
    assert mask.tolist() == [True, False, False, True]


def test_silence_handler_threshold_override():
    in_dim = 4
    state = _make_phonvec_state(in_dim=in_dim)
    state["pos_vecs"] = np.zeros((3, in_dim), dtype=np.float32)
    state["pos_vecs"][0] = np.array([0.1, 0, 0, 0], dtype=np.float32)
    pv = PhonologicalVectors.from_state(state)

    handler = SilenceHandler(pv, threshold=0.5)
    feats = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
    # Weak alignment -> sigmoid output is ~0.525, between thresholds.
    assert not handler.predict_silence_mask(feats, threshold=0.6)[0]
    assert handler.predict_silence_mask(feats, threshold=0.5)[0]


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


def test_add_phone_context_splits_diphthongs():
    """Diphthongs expand to their component phones for context lookup.

    For [p, eɪ, t]: l_1 of t == ɪ, l_2 == e, l_3 == p; mirrored on the right.
    The diphthong row's own ipa stays compound; its l_1/r_1 are the
    surrounding non-diphthong neighbors.
    """
    pd = pytest.importorskip("pandas")
    prep = pytest.importorskip("phonological_posteriogram.training.prepare_datasets")

    df = pd.DataFrame(
        {
            "audio_path": ["u.wav"] * 3,
            "min": [0.0, 0.1, 0.3],
            "max": [0.1, 0.3, 0.4],
            "ipa": ["p", "eɪ", "t"],
        }
    )
    out = prep._add_phone_context(df, n=3)
    t_row = out.iloc[2]
    p_row = out.iloc[0]
    d_row = out.iloc[1]

    assert t_row["l_1"] == "ɪ"
    assert t_row["l_2"] == "e"
    assert t_row["l_3"] == "p"
    assert pd.isna(t_row["r_1"])

    assert p_row["r_1"] == "e"
    assert p_row["r_2"] == "ɪ"
    assert p_row["r_3"] == "t"
    assert pd.isna(p_row["l_1"])

    # The diphthong row's own context treats it as one unit.
    assert d_row["l_1"] == "p"
    assert d_row["r_1"] == "t"
    assert pd.isna(d_row["l_2"]) and pd.isna(d_row["r_2"])


def test_evaluator_perfect_alignment():
    from phonological_posteriogram.evaluation import (
        SegmentationEvaluator,
        SegmentationUnit,
    )

    gt = [SegmentationUnit(0.0, 0.1), SegmentationUnit(0.1, 0.2)]
    pred = [SegmentationUnit(0.0, 0.1), SegmentationUnit(0.1, 0.2)]
    res = SegmentationEvaluator(tolerance_ms=20).evaluate_boundaries(pred, gt)
    assert res["precision"] == pytest.approx(1.0, abs=1e-5)
    assert res["recall"] == pytest.approx(1.0, abs=1e-5)
    assert res["f1"] == pytest.approx(1.0, abs=1e-5)


def test_evaluator_within_tolerance():
    """Predicted boundaries shifted by 10 ms with tolerance 20 ms → P=R=1."""
    from phonological_posteriogram.evaluation import (
        SegmentationEvaluator,
        SegmentationUnit,
    )

    gt = [SegmentationUnit(0.0, 0.1), SegmentationUnit(0.1, 0.2)]
    pred = [SegmentationUnit(0.01, 0.11), SegmentationUnit(0.11, 0.21)]
    res = SegmentationEvaluator(tolerance_ms=20).evaluate_boundaries(pred, gt)
    assert res["precision"] == pytest.approx(1.0, abs=1e-5)
    assert res["recall"] == pytest.approx(1.0, abs=1e-5)


def test_evaluator_outside_tolerance():
    """30 ms shift exceeds 20 ms tolerance → only the shared 0.0 boundary matches."""
    from phonological_posteriogram.evaluation import (
        SegmentationEvaluator,
        SegmentationUnit,
    )

    gt = [SegmentationUnit(0.0, 0.1)]
    pred = [SegmentationUnit(0.03, 0.13)]
    res = SegmentationEvaluator(tolerance_ms=20).evaluate_boundaries(pred, gt)
    # Boundaries in each: {0.0, 0.1} (gt), {0.03, 0.13} (pred). With 20ms
    # tolerance none align, so P=R=0.
    assert res["precision"] == pytest.approx(0.0, abs=1e-5)
    assert res["recall"] == pytest.approx(0.0, abs=1e-5)


def test_evaluator_strict_vs_lenient():
    """Two pred boundaries near one GT boundary: lenient counts both, strict at most one."""
    from phonological_posteriogram.evaluation import (
        SegmentationEvaluator,
        SegmentationUnit,
    )

    # GT has boundaries {0.0, 0.1}. Pred has boundaries {0.0, 0.105, 0.115}
    # — two predicted boundaries are both within 20 ms of GT's 0.1.
    gt = [SegmentationUnit(0.0, 0.1)]
    pred = [
        SegmentationUnit(0.0, 0.105),
        SegmentationUnit(0.105, 0.115),
    ]
    lenient = SegmentationEvaluator(
        tolerance_ms=20, match_mode="lenient"
    ).evaluate_boundaries(pred, gt)
    strict = SegmentationEvaluator(
        tolerance_ms=20, match_mode="strict"
    ).evaluate_boundaries(pred, gt)

    # 3 pred boundaries, 2 GT boundaries.
    # Lenient precision: all 3 pred are within tol of some GT → 3/3 = 1.0
    assert lenient["precision"] == pytest.approx(1.0, abs=1e-5)
    # Strict precision: at most min(3, 2) = 2 pairings → 2/3 ≈ 0.667
    assert strict["precision"] == pytest.approx(2.0 / 3.0, abs=1e-5)


def test_evaluator_forced_mode_includes_pbe_and_symbol_breakdown():
    from phonological_posteriogram.evaluation import (
        SegmentationEvaluator,
        SegmentationUnit,
    )

    gt = [
        SegmentationUnit(0.0, 0.1, "a"),
        SegmentationUnit(0.1, 0.2, "b"),
    ]
    pred = [
        SegmentationUnit(0.01, 0.11, "a"),
        SegmentationUnit(0.11, 0.21, "b"),
    ]
    res = SegmentationEvaluator(
        tolerance_ms=20, forced=True
    ).evaluate_boundaries(pred, gt, symbols=["a", "b"])

    assert "pbe_mean" in res
    assert res["pbe_mean"] == pytest.approx(10.0, abs=1e-3)  # 10 ms shift
    assert "symbol_errors" in res
    assert set(res["symbol_errors"]) == {"a", "b"}


def test_evaluator_batch_aggregates_micro_and_macro():
    from phonological_posteriogram.evaluation import (
        SegmentationEvaluator,
        SegmentationUnit,
    )

    batch_pred = {
        "u1": [SegmentationUnit(0.0, 0.1), SegmentationUnit(0.1, 0.2)],
        # Bad outlier: predicted segment far outside any GT boundary's tol.
        "u2": [SegmentationUnit(0.5, 0.6)],
    }
    batch_gt = {
        "u1": [SegmentationUnit(0.0, 0.1), SegmentationUnit(0.1, 0.2)],
        "u2": [SegmentationUnit(0.0, 0.1)],
    }
    agg = SegmentationEvaluator(tolerance_ms=20).evaluate_batch(
        batch_pred, batch_gt
    )

    # Unique boundary counts:
    #   u1 pred {0.0, 0.1, 0.2} = 3,  u1 gt {0.0, 0.1, 0.2} = 3 (all match)
    #   u2 pred {0.5, 0.6} = 2,       u2 gt  {0.0, 0.1}      = 2 (none match)
    assert agg["pred_counter"] == 3 + 2
    assert agg["gt_counter"] == 3 + 2
    assert agg["precision_counter"] == 3
    assert agg["recall_counter"] == 3
    assert agg["total_segments"] == 2


def test_evaluator_demo_runs():
    """The CLI demo at the bottom of evaluation.py runs end-to-end."""
    from phonological_posteriogram import evaluation

    # Free-mode demo — exercises pretty_print(); rich is optional.
    evaluation._demo([])
    evaluation._demo(["--forced"])


def test_training_evaluate_helpers():
    """_gt_units sorts by min; _pred_units pads with 0 and audio_duration."""
    pd = pytest.importorskip("pandas")
    ev = pytest.importorskip("phonological_posteriogram.training.evaluate")

    df = pd.DataFrame(
        {
            "audio_path": ["a.wav"] * 3,
            "min": [0.2, 0.0, 0.1],
            "max": [0.3, 0.1, 0.2],
            "ipa": ["c", "a", "b"],
        }
    )
    units = ev._gt_units(df)
    assert [u.label for u in units] == ["a", "b", "c"]
    assert [u.start for u in units] == [0.0, 0.1, 0.2]

    pred_units = ev._pred_units(np.array([0.1, 0.2]), audio_duration=0.5)
    assert len(pred_units) == 3
    assert pred_units[0].start == 0.0 and pred_units[0].end == pytest.approx(0.1)
    assert pred_units[-1].end == pytest.approx(0.5)


def test_training_evaluate_end_to_end(tmp_path, monkeypatch):
    """Run training/evaluate.py against a stubbed model + tiny CSV.

    Patches `PhonologicalPosteriogram.from_pretrained` and `librosa.load` so
    no HF download or real audio is needed. Verifies the script produces a
    sensible aggregated result.
    """
    pd = pytest.importorskip("pandas")
    ev = pytest.importorskip("phonological_posteriogram.training.evaluate")

    csv_path = tmp_path / "fake.csv"
    pd.DataFrame(
        {
            "audio_path": ["fake.wav"] * 3,
            "min": [0.0, 0.10, 0.20],
            "max": [0.10, 0.20, 0.30],
            "ipa": ["a", "b", "c"],
            "split": ["test"] * 3,
        }
    ).to_csv(csv_path, index=False)

    class FakeModel:
        net_spec = {"sr": 16000, "frame_shift": 320}

        def segment_seconds(self, waveform, **_):
            # Predict exactly the interior GT boundaries (perfect score).
            return np.array([0.10, 0.20], dtype=float)

    monkeypatch.setattr(
        ev.PhonologicalPosteriogram, "from_pretrained",
        classmethod(lambda cls, *a, **kw: FakeModel()),
    )
    monkeypatch.setattr(
        ev.librosa,
        "load",
        lambda path, sr=None, mono=True: (
            np.zeros(int(0.30 * sr), dtype=np.float32),
            sr,
        ),
    )

    args = ev._get_args(
        [
            "--model", "ignored",
            "--dataset_csv", str(csv_path),
            "--split", "test",
        ]
    )
    results = ev.run(args)
    assert results["f1"] == pytest.approx(1.0, abs=1e-5)
    assert results["total_segments"] == 1


def test_training_evaluate_handles_missing_audio(tmp_path, monkeypatch):
    """A failing librosa.load is reported and the utterance is skipped, but
    the script doesn't crash."""
    pd = pytest.importorskip("pandas")
    ev = pytest.importorskip("phonological_posteriogram.training.evaluate")

    csv_path = tmp_path / "fake.csv"
    pd.DataFrame(
        {
            "audio_path": ["good.wav", "bad.wav"],
            "min": [0.0, 0.0],
            "max": [0.1, 0.1],
            "ipa": ["a", "b"],
            "split": ["test", "test"],
        }
    ).to_csv(csv_path, index=False)

    class FakeModel:
        net_spec = {"sr": 16000, "frame_shift": 320}

        def segment_seconds(self, waveform, **_):
            return np.array([0.05], dtype=float)

    monkeypatch.setattr(
        ev.PhonologicalPosteriogram, "from_pretrained",
        classmethod(lambda cls, *a, **kw: FakeModel()),
    )

    def fake_load(path, sr=None, mono=True):
        if "bad.wav" in str(path):
            raise FileNotFoundError(path)
        return np.zeros(int(0.10 * sr), dtype=np.float32), sr

    monkeypatch.setattr(ev.librosa, "load", fake_load)

    args = ev._get_args(
        ["--model", "ignored", "--dataset_csv", str(csv_path)]
    )
    results = ev.run(args)
    # Only the good utterance was evaluated.
    assert results["total_segments"] == 1


def test_add_phone_context_adjacent_diphthongs():
    """Two diphthongs back-to-back: each splits independently."""
    pd = pytest.importorskip("pandas")
    prep = pytest.importorskip(
        "phonological_posteriogram.training.prepare_datasets"
    )

    df = pd.DataFrame(
        {
            "audio_path": ["u.wav"] * 2,
            "min": [0.0, 0.2],
            "max": [0.2, 0.4],
            "ipa": ["eɪ", "aɪ"],
        }
    )
    out = prep._add_phone_context(df, n=3)
    first, second = out.iloc[0], out.iloc[1]

    # [e, ɪ, a, ɪ] is the expanded sequence.
    assert first["r_1"] == "a"
    assert first["r_2"] == "ɪ"
    assert second["l_1"] == "ɪ"
    assert second["l_2"] == "e"
