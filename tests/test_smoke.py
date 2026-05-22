"""Smoke tests that don't require any model downloads or audio files."""

from pathlib import Path

import numpy as np
import pytest
import torch

import phonological_posteriogram as pp
from phonological_posteriogram.phone_model import PhoneModel
from phonological_posteriogram.posteriogram import PhonologicalPosteriogram
from phonological_posteriogram.recognizer import Recognizer
from phonological_posteriogram.segmenter import Segmenter

NET_SPEC = {
    "hf_repo": "microsoft/wavlm-large",
    "encoder_layer": -1,
    "frame_shift": 320,
    "sr": 16000,
}


def test_public_api_exposed():
    assert pp.PhoneModel is PhoneModel
    assert pp.PhonologicalPosteriogram is PhonologicalPosteriogram
    assert pp.Segmenter is Segmenter
    assert hasattr(pp, "__version__")


def _make_view_state(in_dim=4, n_feat=3, featnames=None, pos_vecs=None):
    """State dict for one PhonologicalPosteriogram view."""
    if featnames is None:
        featnames = ["speech+"] + [f"f{i}" for i in range(n_feat - 1)]
    if pos_vecs is None:
        pos_vecs = np.zeros((n_feat, in_dim), dtype=np.float32)
    return {
        "featnames": featnames,
        "featmap": {"_": [1, 0, 0], "a": [0, 1, 0], "b": [0, 0, 1]},
        "pos_vecs": np.asarray(pos_vecs, dtype=np.float32),
        "zero_vecs": np.zeros((n_feat, in_dim), dtype=np.float32),
        "scales": np.ones((n_feat,), dtype=np.float32),
        "biases": np.zeros((n_feat,), dtype=np.float32),
    }


def _make_posteriogram_state(in_dim=4, n_feat=3, ipa_featnames=None,
                             ipa_pos_vecs=None):
    return {
        "views": {
            "ipa": _make_view_state(in_dim, n_feat, ipa_featnames, ipa_pos_vecs),
            "l_1": _make_view_state(in_dim, n_feat),
            "r_1": _make_view_state(in_dim, n_feat),
        },
        "W_r1_to_ipa": np.eye(n_feat, dtype=np.float32),
        "W_l1_to_ipa": np.eye(n_feat, dtype=np.float32),
    }


def _wire_recognizer(rec, n_feat=3):
    """Hand-populate a Recognizer's attributes for tests (no panphon)."""
    rec.vocab = ["_"] + [f"phone_{i}" for i in range(n_feat - 1)]
    rec.predmat = np.eye(n_feat, dtype=np.float32)
    rec._vocab_to_idx = {p: i for i, p in enumerate(rec.vocab)}
    rec._mask_cache = {}
    rec.hparams = {}
    return rec


def _make_phone_model(monkeypatch, posteriogram, net_spec, *, n_feat=3):
    """Construct a PhoneModel without paying the panphon predmat-build cost.

    The dev env may not have panphon installed (it's a hard dep at runtime
    but optional for unit tests). We stub Recognizer.__init__ to a no-op
    and hand-wire the recognizer's attributes afterward.
    """
    monkeypatch.setattr(Recognizer, "__init__", lambda self, **kw: None)
    model = PhoneModel(posteriogram, net_spec=net_spec)
    _wire_recognizer(model.recognizer, n_feat=n_feat)
    return model


def _make_recognizer(n_feat=3):
    """Bypass Recognizer.__init__ to avoid actually building the panphon
    predmat — that's expensive (iterates the whole panphon segment table)
    and the recognize() logic only needs a hand-set vocab + predmat plus
    the per-instance mask cache."""
    return _wire_recognizer(Recognizer.__new__(Recognizer), n_feat=n_feat)


def _make_posteriogram(**kw):
    return PhonologicalPosteriogram.from_state(_make_posteriogram_state(**kw))


def _make_artifact(in_dim=4, n_feat=3):
    return {
        "posteriogram": _make_posteriogram_state(in_dim, n_feat),
        "hparams": Segmenter.default_hparams(),
        "net": dict(NET_SPEC),
    }


def test_predict_silence_mask_requires_speech_plus():
    post = _make_posteriogram(ipa_featnames=["a+", "b+", "c+"])
    feats = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="speech\\+"):
        post.predict_silence_mask(feats)


def test_predict_silence_mask_uses_speech_plus():
    """predict_silence_mask thresholds the speech+ posteriogram channel."""
    in_dim = 4
    # Make speech+ unmistakable: pos_vec is a strong direction; everything
    # else is zero. Then a feat aligned with speech+ projects to ~1 there.
    pos = np.zeros((3, in_dim), dtype=np.float32)
    pos[0] = np.array([10.0, 0, 0, 0], dtype=np.float32)
    post = _make_posteriogram(in_dim=in_dim, ipa_pos_vecs=pos)

    # 4 frames: silence | speech | speech | silence. The two speech frames
    # each have a non-silent neighbor, so the gap-fill leaves them alone.
    feats = np.array(
        [
            [1.0, 0, 0, 0],
            [-1.0, 0, 0, 0],
            [-1.0, 0, 0, 0],
            [1.0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    mask = post.predict_silence_mask(feats)
    assert mask.tolist() == [True, False, False, True]


def test_predict_silence_mask_threshold():
    in_dim = 4
    pos = np.zeros((3, in_dim), dtype=np.float32)
    pos[0] = np.array([0.1, 0, 0, 0], dtype=np.float32)
    post = _make_posteriogram(in_dim=in_dim, ipa_pos_vecs=pos)

    feats = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
    # Weak alignment -> speech+ projection sigmoid ≈ 0.525, between
    # thresholds. Silence is `speech+ > threshold`, so:
    #   threshold=0.5 → 0.525 > 0.5 → silence (True)
    #   threshold=0.6 → 0.525 > 0.6 → not silence (False)
    assert post.predict_silence_mask(feats, threshold=0.5)[0]
    assert not post.predict_silence_mask(feats, threshold=0.6)[0]


def test_posteriogram_state_roundtrip():
    post = _make_posteriogram()
    reloaded = PhonologicalPosteriogram.from_state(post.to_state())
    assert reloaded.featnames == post.featnames
    assert set(reloaded.views) == {"ipa", "l_1", "r_1"}
    np.testing.assert_array_equal(reloaded.W_r1_to_ipa, post.W_r1_to_ipa)
    np.testing.assert_array_equal(reloaded.W_l1_to_ipa, post.W_l1_to_ipa)


def test_posteriogram_project_activation():
    """project() unifies raw and sigmoid via the `act` argument."""
    in_dim = 4
    pos = np.zeros((3, in_dim), dtype=np.float32)
    pos[0] = np.array([2.0, 0, 0, 0], dtype=np.float32)
    post = _make_posteriogram(in_dim=in_dim, ipa_pos_vecs=pos)
    feats = np.array([[1.0, 0, 0, 0]], dtype=np.float32)

    raw = post.project(feats, view="ipa", act="none")
    sig = post.project(feats, view="ipa", act="sigmoid")
    np.testing.assert_allclose(sig, 1.0 / (1.0 + np.exp(-raw)), rtol=1e-6)
    # Default activation is "none" (raw projection).
    np.testing.assert_array_equal(post.project(feats, view="ipa"), raw)

    with pytest.raises(ValueError, match="activation"):
        post.project(feats, view="ipa", act="bogus")


def test_segmenter_construction():
    post = _make_posteriogram()
    seg = Segmenter(post, sr=16000, frame_shift=320)
    assert seg.frame_shift == 320
    assert seg.sr == 16000
    assert seg.posteriogram is post


def test_phone_model_save_and_load_roundtrip(tmp_path, monkeypatch):
    # Bypass Recognizer.__init__ (panphon predmat build) for from_pretrained.
    monkeypatch.setattr(Recognizer, "__init__", lambda self, **kw: None)
    artifact = _make_artifact()
    torch.save(artifact, tmp_path / "model.pt")

    model = PhoneModel.from_pretrained(tmp_path)
    assert model.net_spec["sr"] == 16000
    assert model.net_spec["hf_repo"] == "microsoft/wavlm-large"

    out_dir = tmp_path / "saved"
    out = model.save_pretrained(out_dir)
    assert out.exists()
    reloaded = PhoneModel.from_pretrained(out_dir)
    assert reloaded.net_spec == model.net_spec
    # Posteriogram weights survive the round-trip.
    np.testing.assert_array_equal(
        reloaded.posteriogram.W_r1_to_ipa, model.posteriogram.W_r1_to_ipa
    )


def test_from_pretrained_accepts_file_path(tmp_path, monkeypatch):
    monkeypatch.setattr(Recognizer, "__init__", lambda self, **kw: None)
    artifact = _make_artifact()
    torch.save(artifact, tmp_path / "custom.pt")
    model = PhoneModel.from_pretrained(tmp_path / "custom.pt")
    assert model.segmenter().frame_shift == 320


def test_phone_model_frame_to_time_roundtrip(monkeypatch):
    """PhoneModel.frame_to_time is the model-accurate inverse of
    SSLEncoder.time_to_frame: for the wav2vec2 conv stack,
    output_lengths(L) = L/stride - 1 in the typical regime, so
    k_eff_samples == stride and the round-trip is exact."""
    # Mirror what extract_features.py records.
    stride = 320
    k_eff = stride  # wav2vec2-family
    model = _make_phone_model(
        monkeypatch,
        _make_posteriogram(),
        net_spec={**NET_SPEC, "k_eff_samples": k_eff, "frame_shift": stride},
    )
    sr = NET_SPEC["sr"]

    # For wav2vec2 the relation output_lengths(L) = L/stride - 1 is exact
    # in the large-L regime: indices [0..L/stride - 2] are valid, so the
    # last-valid-index for time t is (t*sr)/stride - 2 and that index's
    # emergence time is exactly t.
    for t in (0.04, 0.10, 0.50, 1.00):
        count = int(t * sr) // stride - 1   # output_lengths
        last_idx = count - 1
        np.testing.assert_allclose(
            model.frame_to_time(np.array([last_idx])),
            [t],
            atol=1e-6,
        )

    # Backward compat: artifacts without k_eff_samples fall back to the
    # naive `idx * stride / sr` formula.
    legacy = _make_phone_model(
        monkeypatch,
        _make_posteriogram(),
        net_spec={**NET_SPEC, "frame_shift": stride},  # no k_eff_samples
    )
    np.testing.assert_allclose(
        legacy.frame_to_time(np.array([4])), [4 * stride / sr]
    )


def test_recognizer_phoneme_without_lang_raises():
    """phoneme=True (without a language) is a recognize()-time error: the
    construct-time predmat is language-agnostic now."""
    rec = _make_recognizer(n_feat=3)
    posteriogram = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="phoneme=True"):
        rec.recognize(posteriogram, [], phoneme=True)


def test_recognizer_phoneme_with_vocab_raises():
    """phoneme=True conflicts with an explicit ``vocab=`` list."""
    rec = _make_recognizer(n_feat=3)
    posteriogram = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="phoneme=True"):
        rec.recognize(posteriogram, [], vocab=["p", "t"], phoneme=True)


def test_recognizer_rejects_conflicting_vocab_args():
    """Pass at most one of vocab / lang / phoible_id."""
    rec = _make_recognizer(n_feat=3)
    posteriogram = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="at most one"):
        rec.recognize(posteriogram, [], lang="Korean", phoible_id=1)
    with pytest.raises(ValueError, match="at most one"):
        rec.recognize(posteriogram, [], vocab=["p"], lang="Korean")
    with pytest.raises(ValueError, match="at most one"):
        rec.recognize(posteriogram, [], vocab=["p"], phoible_id=1)


def test_recognizer_resolves_unknown_language():
    from phonological_posteriogram.recognizer import _resolve_lang

    with pytest.raises(ValueError, match="not found in Phoible"):
        _resolve_lang("zzz_no_such_language")


def test_recognizer_resolves_ambiguous_language_picks_smallest_id():
    """A language with multiple inventories picks the smallest InventoryID
    (with a warning pointing at phoible.org for context)."""
    from phonological_posteriogram.recognizer import _phoible, _resolve_lang

    df = _phoible()
    # Find a LanguageName that has >1 InventoryID (Phoible has many).
    counts = df.groupby("LanguageName")["InventoryID"].nunique()
    ambiguous = counts[counts > 1]
    if ambiguous.empty:
        pytest.skip("no ambiguous language names in this Phoible snapshot")
    name = ambiguous.index[0]
    expected = int(
        df[df["LanguageName"] == name]["InventoryID"].min()
    )
    with pytest.warns(UserWarning, match="https://phoible.org/languages/"):
        resolved = _resolve_lang(name)
    assert resolved == expected


def test_recognizer_load_inventory_helpers():
    """Phoible CSV is packaged and parses correctly; _load_inventory drops
    phones panphon doesn't recognize."""
    from phonological_posteriogram.recognizer import _phoible, _load_inventory

    df = _phoible()
    # Pick the first inventory and verify allophone parsing.
    inv_id = int(df["InventoryID"].iloc[0])
    phonemes = _load_inventory(inv_id, phoneme=True)
    allophones = _load_inventory(inv_id, phoneme=False)
    assert len(phonemes) > 0
    assert len(allophones) >= len(phonemes)  # surface forms ⊇ phonemes
    # The cache key is (id, phoneme); a second call returns the same tuple.
    assert _load_inventory(inv_id, phoneme=True) is phonemes


def test_recognizer_is_pure_posteriogram_plus_boundaries():
    """Recognizer.recognize is a pure mapping of (posteriogram, boundaries)
    -> per-segment labels; no segmenter or posteriogram object inside.
    Output is *frame-based* (start_frame, end_frame, label) triples — the
    caller handles frame→time conversion."""
    rec = _make_recognizer(n_feat=3)
    assert not hasattr(rec, "posteriogram")
    assert not hasattr(rec, "segmenter")
    assert not hasattr(rec, "_frame_to_time")  # frame→time isn't this class's job

    # Pre-computed per-frame posteriogram (30 frames, 3 feats). Centers of
    # [0,10), [10,20), [20,30) are 5, 15, 25 — arrange those to peak at
    # distinct features.
    posteriogram = np.full((30, 3), 0.1, dtype=np.float32)
    posteriogram[5] = [0.9, 0.1, 0.1]    # → "_"
    posteriogram[15] = [0.1, 0.9, 0.1]   # → "phone_0"
    posteriogram[25] = [0.1, 0.1, 0.9]   # → "phone_1"

    triples = rec.recognize(posteriogram, [10, 20])
    assert triples == [
        (0, 10, "_"),
        (10, 20, "phone_0"),
        (20, 30, "phone_1"),
    ]


def test_recognizer_handles_empty_boundaries():
    """No boundaries → one segment covering the whole feature span."""
    rec = _make_recognizer(n_feat=3)
    posteriogram = np.tile(np.array([0.9, 0.1, 0.1], dtype=np.float32), (8, 1))
    assert rec.recognize(posteriogram, []) == [(0, 8, "_")]


def test_recognizer_vocab_constrains_output(monkeypatch):
    """Passing ``vocab=`` constrains the argmax to those phones (+ silence)."""
    import phonological_posteriogram.recognizer as rec_mod

    # The fake vocab names ("phone_0", ...) aren't panphon-known, so stub
    # _validate_vocab to pass them through unchanged.
    monkeypatch.setattr(rec_mod, "_validate_vocab", lambda v: tuple(v))

    rec = _make_recognizer(n_feat=3)
    # Without a vocab constraint, the center of [0, 6) is frame 3 with
    # posteriogram aligned to "phone_1".
    posteriogram = np.full((6, 3), 0.1, dtype=np.float32)
    posteriogram[3] = [0.1, 0.1, 0.9]   # → "phone_1" unconstrained
    assert rec.recognize(posteriogram, []) == [(0, 6, "phone_1")]

    # With vocab=["phone_0"], "phone_1" is masked out. Silence "_" is
    # always allowed; with the asymmetric scores below, phone_0 wins.
    posteriogram[3] = [0.1, 0.5, 0.9]   # silence=0.1, phone_0=0.5, phone_1=0.9
    triples = rec.recognize(posteriogram, [], vocab=["phone_0"])
    assert triples == [(0, 6, "phone_0")]


def test_recognizer_vocab_unknown_phones_are_filtered(recwarn):
    """Unknown phones in user vocab are dropped (with a warning)."""
    from phonological_posteriogram.recognizer import _validate_vocab

    # Real panphon-known phones plus a junk one. _validate_vocab is cached;
    # use a unique junk token so this test isn't affected by prior runs.
    panphon = pytest.importorskip("panphon")  # noqa: F841 - needs panphon
    out = _validate_vocab(("p", "t", "zzzz_not_a_phone_xyz"))
    assert "zzzz_not_a_phone_xyz" not in out
    assert "p" in out and "t" in out
    assert any("not recognized by panphon" in str(w.message) for w in recwarn.list)


def test_recognizer_vocab_all_unknown_raises():
    """A vocab of only-unknown phones raises (nothing to constrain to)."""
    from phonological_posteriogram.recognizer import _validate_vocab

    pytest.importorskip("panphon")
    with pytest.raises(ValueError, match="no panphon-known phones"):
        _validate_vocab(("zzz1_unknown_a", "zzz2_unknown_b"))


def test_phone_model_embeds_recognizer(monkeypatch):
    """PhoneModel exposes the Recognizer as an attribute (no factory).

    Frame→time conversion is the model's job, not the recognizer's, so the
    embedded Recognizer carries no ``_frame_to_time`` attribute.
    """
    state = _make_posteriogram_state(in_dim=4)
    state["views"]["ipa"]["pos_vecs"][0] = [10, 0, 0, 0]
    post = PhonologicalPosteriogram.from_state(state)
    model = _make_phone_model(monkeypatch, post, dict(NET_SPEC))
    assert isinstance(model.recognizer, Recognizer)
    assert not hasattr(model.recognizer, "_frame_to_time")


def test_phone_model_recognize_returns_segmentation_units(monkeypatch):
    """model.recognize(waveform) chains extract_features + segmenter +
    embedded recognizer, returning a list of SegmentationUnit in seconds."""
    from phonological_posteriogram.evaluation import SegmentationUnit

    state = _make_posteriogram_state(in_dim=4)
    state["views"]["ipa"]["pos_vecs"][0] = [10, 0, 0, 0]
    post = PhonologicalPosteriogram.from_state(state)
    model = _make_phone_model(
        monkeypatch, post, {**NET_SPEC, "k_eff_samples": 320}
    )

    extract_calls = {"n": 0}

    def fake_extract(wav):
        extract_calls["n"] += 1
        return np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (30, 1))

    monkeypatch.setattr(model, "extract_features", fake_extract)

    # Stub the segmenter factory to return a tiny object whose .segment
    # returns fixed boundaries.
    monkeypatch.setattr(
        model, "segmenter",
        lambda overrides=None: type("S", (), {
            "segment": lambda self, f, w, **k: np.array([10, 20])
        })(),
    )

    units = model.recognize(np.zeros(16000, dtype=np.float32))

    assert extract_calls["n"] == 1  # encoder runs exactly once
    assert len(units) == 3
    assert all(isinstance(u, SegmentationUnit) for u in units)
    # Posteriogram is high on speech+ everywhere → every segment labels "_".
    assert [u.label for u in units] == ["_", "_", "_"]
    starts = [u.start for u in units]
    ends = [u.end for u in units]
    assert starts == sorted(starts)
    assert ends == sorted(ends)


def test_phone_model_recognize_accepts_filename(monkeypatch, tmp_path):
    """model.recognize accepts a file path: it dispatches through load_audio
    and runs the full pipeline end-to-end without a precomputed waveform."""
    from phonological_posteriogram.evaluation import SegmentationUnit

    state = _make_posteriogram_state(in_dim=4)
    state["views"]["ipa"]["pos_vecs"][0] = [10, 0, 0, 0]
    post = PhonologicalPosteriogram.from_state(state)
    model = _make_phone_model(
        monkeypatch, post, {**NET_SPEC, "k_eff_samples": 320}
    )

    load_calls = []

    def fake_load_audio(self, path):
        load_calls.append(str(path))
        return np.zeros(int(0.3 * NET_SPEC["sr"]), dtype=np.float32)

    monkeypatch.setattr(PhoneModel, "load_audio", fake_load_audio)
    monkeypatch.setattr(
        model, "extract_features",
        lambda w: np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (10, 1)),
    )
    monkeypatch.setattr(
        model, "segmenter",
        lambda overrides=None: type("S", (), {
            "segment": lambda self, f, w, **k: np.array([5])
        })(),
    )

    fake_path = tmp_path / "speech.wav"
    units = model.recognize(fake_path)
    assert load_calls == [str(fake_path)]
    assert len(units) == 2
    assert all(isinstance(u, SegmentationUnit) for u in units)


def test_phone_model_recognize_resamples_with_warning(monkeypatch):
    """Mismatched ``sr=`` triggers librosa.resample and a UserWarning."""
    state = _make_posteriogram_state(in_dim=4)
    post = PhonologicalPosteriogram.from_state(state)
    model = _make_phone_model(monkeypatch, post, dict(NET_SPEC))

    # Patch librosa.resample so we don't actually need librosa for the math.
    import phonological_posteriogram.phone_model as pm

    resample_calls = []

    class FakeLibrosa:
        @staticmethod
        def resample(y, *, orig_sr, target_sr):
            resample_calls.append((orig_sr, target_sr, len(y)))
            return np.zeros(int(len(y) * target_sr / orig_sr), dtype=np.float32)

    # The `import librosa` inside _coerce_waveform sees this module's
    # sys.modules entry.
    monkeypatch.setitem(__import__("sys").modules, "librosa", FakeLibrosa())
    monkeypatch.setattr(
        model, "extract_features",
        lambda w: np.zeros((5, 4), dtype=np.float32),
    )
    monkeypatch.setattr(
        model, "segmenter",
        lambda overrides=None: type("S", (), {
            "segment": lambda self, f, w, **k: np.array([])
        })(),
    )

    waveform = np.zeros(8000, dtype=np.float32)
    with pytest.warns(UserWarning, match="resampling"):
        model.recognize(waveform, sr=8000)
    assert resample_calls == [(8000, NET_SPEC["sr"], 8000)]

    # Matching sr does NOT resample and does NOT warn.
    resample_calls.clear()
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any warning would raise
        model.recognize(waveform, sr=NET_SPEC["sr"])
    assert resample_calls == []


def test_phone_model_recognize_ignores_sr_when_path_supplied(monkeypatch, tmp_path):
    """``sr=`` is meaningless when ``audio`` is a path; emit a warning."""
    state = _make_posteriogram_state(in_dim=4)
    post = PhonologicalPosteriogram.from_state(state)
    model = _make_phone_model(monkeypatch, post, dict(NET_SPEC))

    monkeypatch.setattr(
        PhoneModel, "load_audio",
        lambda self, p: np.zeros(NET_SPEC["sr"], dtype=np.float32),
    )
    monkeypatch.setattr(
        model, "extract_features",
        lambda w: np.zeros((5, 4), dtype=np.float32),
    )
    monkeypatch.setattr(
        model, "segmenter",
        lambda overrides=None: type("S", (), {
            "segment": lambda self, f, w, **k: np.array([])
        })(),
    )

    path = tmp_path / "f.wav"
    with pytest.warns(UserWarning, match="ignored when `audio` is a file path"):
        model.recognize(path, sr=8000)


def test_phone_model_segmenter_hparam_override(monkeypatch):
    """The point of the refactor: building a Segmenter with different
    hparams is cheap and shares the (expensive) posteriogram weights."""
    model = _make_phone_model(monkeypatch, _make_posteriogram(), dict(NET_SPEC))
    seg_a = model.segmenter()
    seg_b = model.segmenter({"drop_k": 0})

    assert seg_a.posteriogram is seg_b.posteriogram  # weights not copied
    assert seg_b.hparams["drop_k"] == 0
    assert seg_a.hparams["drop_k"] == Segmenter.default_hparams()["drop_k"]


def test_segmenter_default_hparams_self_consistent():
    h = Segmenter.default_hparams()
    for spec in h["combined_signals"]:
        assert set(spec) == {"name", "kwargs", "shift"}
    assert set(h["single_signal"]) == {"name", "kwargs", "shift"}
    assert "mel_frame_shift_ms" in h
    assert h["activation"] in ("none", "sigmoid")
    assert h["combine_method"] in ("min", "logmeanexp")
    assert h["distance"] in ("cosine", "l2")


def test_segmenter_hparams_merge_onto_defaults():
    """A partial hparams dict still yields a complete config (missing keys
    filled from default_hparams)."""
    post = _make_posteriogram()
    seg = Segmenter(post, sr=16000, frame_shift=320, hparams={"drop_k": 1})
    assert seg.hparams["drop_k"] == 1
    assert seg.hparams["activation"] == "none"  # filled from defaults
    assert "combined_signals" in seg.hparams


def test_segmenter_combines_duplicate_signal_specs():
    """combined_signals is a list of specs, so the same signal type may
    appear twice with different kwargs."""
    post = _make_posteriogram(in_dim=4, n_feat=3)
    seg = Segmenter(
        post,
        sr=16000,
        frame_shift=320,
        hparams={
            **Segmenter.default_hparams(),
            "combined_signals": [
                {"name": "frame_delta", "kwargs": {"offset": 1}, "shift": 0},
                {"name": "frame_delta", "kwargs": {"offset": 3}, "shift": 0},
            ],
            "drop_k": 0,
            "snap_silence": False,
        },
    )
    feats = np.random.default_rng(0).normal(size=(30, 4)).astype(np.float32)
    preds = seg.segment(feats, np.zeros(9600, dtype=np.float32))
    assert isinstance(preds, np.ndarray)


@pytest.mark.parametrize("combine_method", ["min", "logmeanexp"])
def test_segmenter_combine_method_hparam(combine_method):
    """Both combine methods run through segment()."""
    post = _make_posteriogram(in_dim=4, n_feat=3)
    seg = Segmenter(
        post,
        sr=16000,
        frame_shift=320,
        hparams={"combine_method": combine_method, "snap_silence": False},
    )
    feats = np.random.default_rng(1).normal(size=(30, 4)).astype(np.float32)
    preds = seg.segment(feats, np.zeros(9600, dtype=np.float32))
    assert isinstance(preds, np.ndarray)


def test_segmenter_rejects_unknown_combine_method():
    from phonological_posteriogram.segmenter import _combine_stacked

    with pytest.raises(ValueError, match="combine_method"):
        _combine_stacked(np.ones((2, 5)), "bogus")


def test_pair_distance_methods():
    from phonological_posteriogram.segmenter import _pair_distance

    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Cosine: identical rows -> 0; orthogonal rows -> 1.
    np.testing.assert_allclose(_pair_distance(a, b, "cosine"), [0.0, 1.0])
    # L2: ||(0,0)|| = 0; ||(1,-1)|| = sqrt(2).
    np.testing.assert_allclose(
        _pair_distance(a, b, "l2"), [0.0, np.sqrt(2)]
    )
    # Cosine zero-norm fallback: 1.0.
    zeros = np.zeros_like(a)
    np.testing.assert_array_equal(_pair_distance(a, zeros, "cosine"), [1.0, 1.0])

    with pytest.raises(ValueError, match="distance"):
        _pair_distance(a, b, "bogus")


@pytest.mark.parametrize("distance", ["cosine", "l2"])
def test_segmenter_distance_hparam(distance):
    """Both distances run through segment()."""
    post = _make_posteriogram(in_dim=4, n_feat=3)
    seg = Segmenter(
        post,
        sr=16000,
        frame_shift=320,
        hparams={"distance": distance, "snap_silence": False},
    )
    feats = np.random.default_rng(2).normal(size=(30, 4)).astype(np.float32)
    preds = seg.segment(feats, np.zeros(9600, dtype=np.float32))
    assert isinstance(preds, np.ndarray)


def test_normalize_signal_methods():
    from phonological_posteriogram.segmenter import _normalize_signal

    sig = np.array([2.0, 4.0, 6.0])
    # "none" passes the values through unchanged (but copies).
    out_none = _normalize_signal(sig, "none")
    np.testing.assert_array_equal(out_none, sig)
    assert out_none is not sig
    # "min" subtracts the minimum.
    np.testing.assert_array_equal(_normalize_signal(sig, "min"), [0, 2, 4])
    # "minmax" rescales to [0, 1].
    np.testing.assert_allclose(_normalize_signal(sig, "minmax"), [0, 0.5, 1])

    with pytest.raises(ValueError, match="norm_method"):
        _normalize_signal(sig, "bogus")


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


def test_per_token_level_edit_distance():
    """PER is Levenshtein over label TOKENS (compound labels like 'eɪ' are
    one token), divided by reference length."""
    from phonological_posteriogram.evaluation import PhoneRecognitionEvaluator

    e = PhoneRecognitionEvaluator(skip_labels=())  # don't skip anything
    # Identical → 0.
    assert e.per(["p", "eɪ", "t"], ["p", "eɪ", "t"]) == pytest.approx(0.0)
    # Single substitution among 3 ref tokens → 1/3.
    assert e.per(["b", "eɪ", "t"], ["p", "eɪ", "t"]) == pytest.approx(1 / 3)
    # Compound diphthong is ONE token: substituting eɪ→aɪ is 1 edit, not 2.
    assert e.per(["p", "aɪ", "t"], ["p", "eɪ", "t"]) == pytest.approx(1 / 3)
    # Empty reference → 0 (defined by convention).
    assert e.per(["p"], []) == 0.0


def test_per_pfer_skip_silence_by_default():
    """Silence "_" is dropped from both sides before scoring."""
    from phonological_posteriogram.evaluation import PhoneRecognitionEvaluator

    e = PhoneRecognitionEvaluator()  # default skip_labels = {"_"}
    # With "_" dropped, both sequences are just ["p", "t"] → PER = 0.
    assert e.per(["_", "p", "_", "t"], ["p", "t", "_"]) == pytest.approx(0.0)


def test_per_pfer_batch_micro_averaged():
    """Batch aggregation is micro-averaged: total distance / total ref len."""
    from phonological_posteriogram.evaluation import PhoneRecognitionEvaluator

    e = PhoneRecognitionEvaluator(skip_labels=())
    # u1: 1 sub in 3 → 1/3 PER (would give 0.333 if averaged).
    # u2: 0 errors in 5 → 0 PER.
    # Micro: total errors = 1, total ref = 8 → 1/8 = 0.125.
    pred = {"u1": ["p", "b", "t"], "u2": ["a", "b", "c", "d", "e"]}
    gt = {"u1": ["p", "eɪ", "t"], "u2": ["a", "b", "c", "d", "e"]}

    # Bypass PFER (panphon) by clearing the panphon-using path: just call
    # per() per utterance, since evaluate_batch also invokes pfer().
    assert e.per(pred["u1"], gt["u1"]) == pytest.approx(1 / 3)
    assert e.per(pred["u2"], gt["u2"]) == pytest.approx(0.0)


def test_pfer_uses_panphon(monkeypatch):
    """PFER calls panphon.feature_edit_distance and divides by ref length."""
    from phonological_posteriogram.evaluation import PhoneRecognitionEvaluator

    e = PhoneRecognitionEvaluator(skip_labels=())

    class FakeDist:
        def feature_edit_distance(self, s1, s2):
            # Return a fixed cost regardless of input — we only care that
            # PFER wires up to this function and normalizes by ref length.
            return 1.5

    e._dist = FakeDist()  # pre-populate the lazy panphon attribute
    # 3 ref tokens, cost 1.5 → PFER = 0.5.
    assert e.pfer(["p", "eɪ", "t"], ["p", "eɪ", "t"]) == pytest.approx(0.5)


def test_phone_recognition_evaluator_accepts_segmentation_units():
    """Accepts list[SegmentationUnit] in addition to list[str]."""
    from phonological_posteriogram.evaluation import (
        PhoneRecognitionEvaluator,
        SegmentationUnit,
    )

    e = PhoneRecognitionEvaluator(skip_labels=())
    pred_units = [
        SegmentationUnit(0.0, 0.1, "p"),
        SegmentationUnit(0.1, 0.2, "t"),
    ]
    ref_units = [
        SegmentationUnit(0.0, 0.1, "p"),
        SegmentationUnit(0.1, 0.2, "t"),
    ]
    assert e.per(pred_units, ref_units) == pytest.approx(0.0)


def test_phone_recognition_evaluator_batch(monkeypatch):
    """evaluate_batch micro-averages PER and PFER across utterances."""
    from phonological_posteriogram.evaluation import PhoneRecognitionEvaluator

    e = PhoneRecognitionEvaluator(skip_labels=())

    class FakeDist:
        def feature_edit_distance(self, s1, s2):
            return 2.0  # constant per-utterance feature cost

    e._dist = FakeDist()

    pred = {"u1": ["p", "b", "t"], "u2": ["a", "b"]}
    gt = {"u1": ["p", "eɪ", "t"], "u2": ["a", "b"]}
    result = e.evaluate_batch(pred, gt)

    # Total PER errors = 1 + 0 = 1; total ref = 3 + 2 = 5; PER = 0.2.
    assert result["per"] == pytest.approx(0.2)
    # Total PFER cost = 2 + 2 = 4; total ref = 5; PFER = 0.8.
    assert result["pfer"] == pytest.approx(0.8)
    assert result["n_utterances"] == 2
    assert result["total_ref_phones"] == 5


def test_training_evaluate_gt_units_sorted():
    """_gt_units sorts segments by min so evaluation is order-insensitive
    to the input CSV row order."""
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


class _FakePost:
    """Stub posteriogram object with the .project method evaluate.py uses."""

    featnames = ["speech+", "a+", "b+"]

    def project(self, feats, view="ipa", act="sigmoid"):
        return np.zeros((len(feats), 3), dtype=np.float32)


class _FakeRecognizerForEval:
    """Tiles whatever boundaries it's given, labeling segments a/b/c...

    Returns frame-based ``(start_frame, end_frame, label)`` triples — the
    same shape as the real ``Recognizer.recognize``. Frame→time conversion
    is the caller's job (typically via ``model.frame_to_time``).
    """

    LABELS = ["a", "b", "c", "d"]

    def recognize(self, posteriogram, boundaries, **kw):
        T = len(posteriogram)
        bs = np.unique(
            np.concatenate(
                [np.asarray([0]), np.asarray(boundaries, dtype=int), np.asarray([T])]
            )
        )
        return [
            (int(bs[i]), int(bs[i + 1]), self.LABELS[i % len(self.LABELS)])
            for i in range(len(bs) - 1)
        ]


class _FakeRecogEval:
    """Stub PhoneRecognitionEvaluator so tests don't pay the real
    panphon.distance.Distance() construction + feature_edit_distance cost."""

    def __init__(self, *_, **__):
        pass

    def evaluate_batch(self, predictions, ground_truth):
        return {
            "per": 0.0,
            "pfer": 0.0,
            "n_utterances": sum(1 for k in ground_truth if k in predictions),
            "total_ref_phones": sum(len(g) for g in ground_truth.values()),
        }


def _make_fake_eval_model(internal_boundaries, n_frames=30, bad_paths=()):
    """A model stub with all the methods evaluate.py / tune.py call.

    `n_frames` controls the feature/posteriogram length, which in turn
    determines the last predicted segment's end (via frame_to_time).
    `bad_paths` is a collection of substrings; ``load_audio`` on a path
    matching one raises ``FileNotFoundError`` (so we can exercise the
    error-handling branch).

    Mirrors the new API: ``recognize(audio, *, sr, lang, phoible_id,
    phoneme, vocab)`` returns ``List[SegmentationUnit]`` in seconds; the
    embedded recognizer is an attribute (not a factory) and itself returns
    frame-based triples.
    """
    from phonological_posteriogram.evaluation import SegmentationUnit

    def _frame_to_time(idxs):
        return np.asarray(idxs, dtype=float) * 0.02

    class FakeModel:
        net_spec = {"sr": 16000, "frame_shift": 320}
        posteriogram = _FakePost()
        hparams = {"snap_silence": True}
        recognizer = _FakeRecognizerForEval()

        def load_audio(self, path):
            if any(bad in str(path) for bad in bad_paths):
                raise FileNotFoundError(path)
            return np.zeros(int(0.30 * self.net_spec["sr"]), dtype=np.float32)

        def extract_features(self, x):
            return np.zeros((n_frames, 4), dtype=np.float32)

        def frame_to_time(self, idxs):
            return _frame_to_time(idxs)

        def recognize(
            self, audio, *, sr=None, lang=None, phoible_id=None,
            phoneme=False, vocab=None,
        ):
            if isinstance(audio, (str, Path)):
                waveform = self.load_audio(audio)
            else:
                waveform = np.asarray(audio, dtype=np.float32)
            posteriogram = self.posteriogram.project(
                self.extract_features(waveform)
            )
            triples = self.recognizer.recognize(
                posteriogram,
                np.asarray(internal_boundaries, dtype=int),
                lang=lang, phoible_id=phoible_id, phoneme=phoneme, vocab=vocab,
            )
            if not triples:
                return []
            starts = self.frame_to_time(np.array([t[0] for t in triples]))
            ends = self.frame_to_time(np.array([t[1] for t in triples]))
            return [
                SegmentationUnit(
                    float(starts[i]), float(ends[i]), triples[i][2]
                )
                for i in range(len(triples))
            ]

    return FakeModel()


def test_training_evaluate_end_to_end(tmp_path, monkeypatch):
    """Run training/evaluate.py against a stubbed model + tiny CSV.

    Verifies that boundary metrics (SegmentationEvaluator) and recognition
    metrics (PhoneRecognitionEvaluator) both flow through the script's
    pipeline and end up in the merged results dict.
    """
    pd = pytest.importorskip("pandas")
    ev = pytest.importorskip("phonological_posteriogram.training.evaluate")

    csv_path = tmp_path / "fake.csv"
    # GT segments at [0,0.10), [0.10,0.20), [0.20,0.30). With our fake
    # model's frame_to_time = idx * 0.02, that corresponds to internal
    # boundaries at frames 5 and 10 (segment edges 0, 5, 10, 15).
    pd.DataFrame(
        {
            "audio_path": ["fake.wav"] * 3,
            "min": [0.0, 0.10, 0.20],
            "max": [0.10, 0.20, 0.30],
            "ipa": ["a", "b", "c"],
            "split": ["test"] * 3,
        }
    ).to_csv(csv_path, index=False)

    # 15 frames at 0.02s/frame → last segment ends at 0.30s, matching GT.
    fake_model = _make_fake_eval_model(internal_boundaries=[5, 10], n_frames=15)
    monkeypatch.setattr(
        ev.PhoneModel, "from_pretrained",
        classmethod(lambda cls, *a, **kw: fake_model),
    )
    # Stub out the panphon-using PhoneRecognitionEvaluator for speed.
    monkeypatch.setattr(ev, "PhoneRecognitionEvaluator", _FakeRecogEval)

    args = ev._get_args(
        [
            "--model", "ignored",
            "--dataset_csv", str(csv_path),
            "--split", "test",
        ]
    )
    results = ev.run(args)
    # Boundary metrics
    assert results["f1"] == pytest.approx(1.0, abs=1e-5)
    assert results["total_segments"] == 1
    # Recognition metrics merged into the same dict
    assert results["per"] == 0.0
    assert results["pfer"] == 0.0


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

    fake_model = _make_fake_eval_model(
        internal_boundaries=[15], bad_paths=("bad.wav",)
    )
    monkeypatch.setattr(
        ev.PhoneModel, "from_pretrained",
        classmethod(lambda cls, *a, **kw: fake_model),
    )
    monkeypatch.setattr(ev, "PhoneRecognitionEvaluator", _FakeRecogEval)

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


def _make_synthetic_features_pkl(tmp_path):
    """Build a tiny pkl that mimics extract_features.py's output.

    Returns the path to the pkl. The dataset has two short utterances with
    a small phone vocab + silence, enough rows for Segmenter.fit's
    consecutive-pair regression to be non-empty.
    """
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    in_dim = 8

    rows = []
    for utt in ("u1.wav", "u2.wav"):
        # Phone sequence with silence padding so "_" appears in the vocab.
        seq = ["_", "p", "i", "t", "_"]
        for i, ipa in enumerate(seq):
            rows.append(
                {
                    "audio_path": utt,
                    "min": i * 0.10,
                    "max": (i + 1) * 0.10,
                    "ipa": ipa,
                    "l_1": seq[i - 1] if i > 0 else None,
                    "r_1": seq[i + 1] if i < len(seq) - 1 else None,
                    "feat": rng.normal(size=in_dim).astype(np.float32),
                }
            )

    df = pd.DataFrame(rows)
    df.attrs["hf_repo"] = "microsoft/wavlm-large"
    df.attrs["encoder_layer"] = -1
    df.attrs["pool"] = "center"
    df.attrs["sr"] = 16000
    df.attrs["frame_shift"] = 320

    pkl_path = tmp_path / "feats.pkl"
    df.to_pickle(pkl_path)
    return pkl_path


def test_training_train_requires_features_attrs(tmp_path):
    """Missing df.attrs keys produce a clear error, not an opaque crash."""
    pd = pytest.importorskip("pandas")
    train_mod = pytest.importorskip(
        "phonological_posteriogram.training.train"
    )

    pkl_path = tmp_path / "bad.pkl"
    pd.DataFrame({"ipa": ["a"], "feat": [np.zeros(4, dtype=np.float32)]}).to_pickle(
        pkl_path
    )

    args = train_mod._get_args(
        [
            "--features_pkl", str(pkl_path),
            "--output_dir", str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ValueError, match="df.attrs"):
        train_mod.run(args)


def test_training_train_end_to_end(tmp_path):
    """Fit on a synthetic features pkl, save, and reload via from_pretrained."""
    train_mod = pytest.importorskip(
        "phonological_posteriogram.training.train"
    )

    pkl_path = _make_synthetic_features_pkl(tmp_path)
    out_dir = tmp_path / "trained"

    args = train_mod._get_args(
        [
            "--features_pkl", str(pkl_path),
            "--output_dir", str(out_dir),
        ]
    )
    out = train_mod.run(args)
    assert out.exists()
    assert out.parent == out_dir
    assert out.name == "model.pt"

    # Round-trip through HF-style loader without needing the actual SSL
    # weights (encoder is lazy-loaded).
    reloaded = PhoneModel.from_pretrained(out_dir)
    assert reloaded.net_spec["hf_repo"] == "microsoft/wavlm-large"
    assert reloaded.net_spec["sr"] == 16000
    assert reloaded.net_spec["frame_shift"] == 320
    # mel_frame_shift_ms now lives in the algorithm hparams, not net_spec.
    assert reloaded.hparams["mel_frame_shift_ms"] == 10
    # Silence detection lives on the posteriogram (speech+ channel); the
    # fitted vocab includes "_", so the feature is present.
    assert "speech+" in reloaded.posteriogram.featnames


def test_training_tune_runs_encoder_once_and_sweeps(tmp_path, monkeypatch):
    """tune.py encodes each utterance once, then sweeps the hparam list over
    the cached features (no re-encoding per candidate)."""
    import json

    pd = pytest.importorskip("pandas")
    tune = pytest.importorskip("phonological_posteriogram.training.tune")

    csv_path = tmp_path / "d.csv"
    pd.DataFrame(
        {
            "audio_path": ["a.wav", "b.wav"],
            "min": [0.0, 0.0],
            "max": [0.1, 0.1],
            "ipa": ["a", "b"],
            "split": ["test", "test"],
            "language": ["eng", "eng"],
        }
    ).to_csv(csv_path, index=False)

    # Three candidate hparam overrides to sweep.
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(
        json.dumps(
            [
                {"drop_k": 0, "snap_silence": False},
                {"drop_k": 1, "snap_silence": False},
                {"drop_k": 2, "snap_silence": False},
            ]
        )
    )

    model = _make_phone_model(
        monkeypatch, _make_posteriogram(in_dim=4), dict(NET_SPEC)
    )
    monkeypatch.setattr(
        tune.PhoneModel, "from_pretrained",
        classmethod(lambda cls, *a, **kw: model),
    )
    monkeypatch.setattr(
        PhoneModel, "load_audio",
        lambda self, path: np.zeros(
            int(0.3 * self.net_spec["sr"]), dtype=np.float32
        ),
    )

    # Count encoder calls — must be once per utterance, not per candidate.
    calls = {"n": 0}
    rng = np.random.default_rng(0)

    def fake_extract(waveform):
        calls["n"] += 1
        return rng.normal(size=(30, 4)).astype(np.float32)

    monkeypatch.setattr(model, "extract_features", fake_extract)
    # Swap in the fake recognizer (attribute, not factory) so the sweep
    # doesn't pay the real Recognizer's panphon-aware logic.
    model.recognizer = _FakeRecognizerForEval()
    # Stub PhoneRecognitionEvaluator so the sweep doesn't pay the real
    # panphon.distance.Distance() construction cost.
    monkeypatch.setattr(tune, "PhoneRecognitionEvaluator", _FakeRecogEval)

    args = tune._get_args(
        [
            "--model", "ignored",
            "--dataset_csv", str(csv_path),
            "--hparams_json", str(grid_path),
        ]
    )
    results = tune.run(args)

    assert len(results) == 3  # one result per candidate
    assert calls["n"] == 2  # 2 utterances encoded once each, not 2 * 3
    # Results are ranked best-first.
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_tune_known_lang_kwargs_resolves_per_utterance(monkeypatch):
    """Each row's `language` (an ISO 639-3 code) maps to a phoible_id."""
    pd = pytest.importorskip("pandas")
    tune = pytest.importorskip("phonological_posteriogram.training.tune")

    df = pd.DataFrame(
        {
            "audio_path": ["a.wav", "b.wav", "c.wav"],
            "min": [0.0, 0.0, 0.0],
            "max": [0.1, 0.1, 0.1],
            "ipa": ["a", "b", "c"],
            "language": ["xxA", "xxB", "xxC"],
        }
    )
    fake_resolve = {"xxA": 11, "xxB": 22}  # xxC fails

    def _fake_resolve(lang):
        if lang in fake_resolve:
            return fake_resolve[lang]
        raise ValueError(f"unknown {lang}")

    monkeypatch.setattr(tune, "_resolve_lang", _fake_resolve)

    with pytest.warns(UserWarning, match="Could not resolve"):
        out = tune._resolve_known_lang_kwargs(
            df, ["a.wav", "b.wav", "c.wav"]
        )
    assert out["a.wav"]["phoible_id"] == 11
    assert out["b.wav"]["phoible_id"] == 22
    # Unresolved language → no constraint for that utterance.
    assert out["c.wav"]["phoible_id"] is None
    assert out["c.wav"]["lang"] is None


def test_tune_known_lang_kwargs_requires_language_column():
    """tune.py refuses to run on a CSV without a `language` column."""
    pd = pytest.importorskip("pandas")
    tune = pytest.importorskip("phonological_posteriogram.training.tune")

    df = pd.DataFrame(
        {
            "audio_path": ["a.wav"],
            "min": [0.0],
            "max": [0.1],
            "ipa": ["a"],
        }
    )
    with pytest.raises(ValueError, match="`language` column"):
        tune._resolve_known_lang_kwargs(df, ["a.wav"])


def test_training_tune_lower_is_better_for_known_per(tmp_path, monkeypatch):
    """--metric=known_per sorts ascending (lower is better) and both
    known_per and unknown_per appear in the per-candidate metrics dict."""
    import json

    pd = pytest.importorskip("pandas")
    tune = pytest.importorskip("phonological_posteriogram.training.tune")

    csv_path = tmp_path / "d.csv"
    pd.DataFrame(
        {
            "audio_path": ["a.wav"],
            "min": [0.0],
            "max": [0.1],
            "ipa": ["a"],
            "split": ["test"],
            "language": ["eng"],
        }
    ).to_csv(csv_path, index=False)
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(json.dumps([{"drop_k": 0}, {"drop_k": 1}]))

    model = _make_phone_model(
        monkeypatch, _make_posteriogram(in_dim=4), dict(NET_SPEC)
    )
    monkeypatch.setattr(
        tune.PhoneModel, "from_pretrained",
        classmethod(lambda cls, *a, **kw: model),
    )
    monkeypatch.setattr(
        PhoneModel, "load_audio",
        lambda self, path: np.zeros(
            int(0.3 * self.net_spec["sr"]), dtype=np.float32
        ),
    )
    rng = np.random.default_rng(0)
    monkeypatch.setattr(
        model, "extract_features",
        lambda w: rng.normal(size=(30, 4)).astype(np.float32),
    )
    model.recognizer = _FakeRecognizerForEval()

    # Each candidate triggers TWO evaluate_batch calls (known + unknown
    # recognizer pass). Give each candidate a distinct known PER and a
    # distinct unknown PER so the test can verify both end up in the dict.
    sequence = iter([
        (0.5, 0.7),  # cand 0: (known_per, unknown_per)
        (0.1, 0.3),  # cand 1: (known_per, unknown_per)
    ])
    current = {"known": None, "unknown": None, "toggle": 0}

    class _OrderedRecogEval:
        def __init__(self, *_, **__): pass
        def evaluate_batch(self, p, g):
            if current["toggle"] == 0:
                current["known"], current["unknown"] = next(sequence)
                v = current["known"]
            else:
                v = current["unknown"]
            current["toggle"] ^= 1
            return {
                "per": v, "pfer": 0.0,
                "n_utterances": len(g),
                "total_ref_phones": sum(len(vv) for vv in g.values()),
            }

    monkeypatch.setattr(tune, "PhoneRecognitionEvaluator", _OrderedRecogEval)

    args = tune._get_args(
        [
            "--model", "ignored",
            "--dataset_csv", str(csv_path),
            "--hparams_json", str(grid_path),
            "--metric", "known_per",
        ]
    )
    results = tune.run(args)
    # Lower known_per wins → candidate index 1 (known_per=0.1) ranks first.
    assert results[0]["index"] == 1
    assert results[0]["score"] == pytest.approx(0.1)
    # Both known_ and unknown_ metrics are present per candidate.
    for r in results:
        assert "known_per" in r["metrics"] and "unknown_per" in r["metrics"]
        assert "known_pfer" in r["metrics"] and "unknown_pfer" in r["metrics"]
    # And they're distinct on at least one candidate (sanity).
    assert any(
        r["metrics"]["known_per"] != r["metrics"]["unknown_per"]
        for r in results
    )


def test_tune_cli_rejects_removed_args():
    """The pre-refactor --lang/--phoible_id/--phoneme/--vocab args are gone."""
    tune = pytest.importorskip("phonological_posteriogram.training.tune")

    base = ["--model", "x", "--dataset_csv", "y", "--hparams_json", "z"]
    for removed in ("--lang", "--phoible_id", "--phoneme", "--vocab"):
        argv = base + (
            [removed, "English"] if removed != "--phoneme" else [removed]
        )
        with pytest.raises(SystemExit):
            tune._get_args(argv)
