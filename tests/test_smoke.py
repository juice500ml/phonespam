"""Smoke tests that don't require any model downloads or audio files."""

import numpy as np
import pytest
import torch

import phonological_posteriogram as pp
from phonological_posteriogram.phone_model import PhoneModel
from phonological_posteriogram.posteriogram import PhonologicalPosteriogram
from phonological_posteriogram.recognizer import Recognizer, panphon_featmap
from phonological_posteriogram.segmenter import Segmenter

NET_SPEC = {
    "hf_repo": "microsoft/wavlm-large",
    "encoder_layer": -1,
    "sr": 16000,
}


def test_public_api_exposed():
    assert pp.PhoneModel is PhoneModel
    assert pp.PhonologicalPosteriogram is PhonologicalPosteriogram
    assert pp.Recognizer is Recognizer
    assert pp.Segmenter is Segmenter
    assert pp.panphon_featmap is panphon_featmap
    assert hasattr(pp, "__version__")


def _make_view_state(in_dim=4, n_feat=3, featnames=None, pos_vecs=None):
    """State dict for one PhonologicalPosteriogram view."""
    if featnames is None:
        featnames = ["silence+"] + [f"f{i}" for i in range(n_feat - 1)]
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


def _make_posteriogram_state(in_dim=4, n_feat=3, ipa_featnames=None, ipa_pos_vecs=None):
    W = np.zeros((in_dim, n_feat), dtype=np.float32)
    W[: min(in_dim, n_feat), : min(in_dim, n_feat)] = np.eye(min(in_dim, n_feat), dtype=np.float32)
    return {
        "view": _make_view_state(in_dim, n_feat, ipa_featnames, ipa_pos_vecs),
        "W_bwd": W,
    }


def _wire_recognizer(rec, n_feat=3):
    """Hand-populate a Recognizer's attributes for tests (no panphon)."""
    rec.vocab = ["_"] + [f"phone_{i}" for i in range(n_feat - 1)]
    rec.predmat = np.eye(n_feat, dtype=np.float32)
    rec._featnames = [f"f{i}" for i in range(n_feat)]
    rec._feat_idxs = np.arange(n_feat, dtype=np.int64)
    rec._uses_fitted_featmap = False
    rec._output_labels = list(rec.vocab)
    rec._vocab_to_idx = {p: i for i, p in enumerate(rec.vocab)}
    rec._output_to_idxs = {p: [i] for i, p in enumerate(rec.vocab)}
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
    # Pre-populate the lazily-loaded SSL encoder with a stub (320-sample
    # stride at 16 kHz) so recognize() never downloads the model.
    model._encoder = type(
        "FakeEncoder",
        (),
        {
            "stride_size": 320,
            "frame_to_time": staticmethod(lambda idxs: np.asarray(idxs, dtype=float) * 0.02),
        },
    )()
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


def test_predict_silence_mask_requires_silence_plus():
    post = _make_posteriogram(ipa_featnames=["a+", "b+", "c+"])
    feats = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="silence\\+"):
        post.predict_silence_mask(feats)


def test_predict_silence_mask_uses_speech_plus():
    """predict_silence_mask thresholds the silence+ posteriogram channel."""
    in_dim = 4
    # Make silence+ unmistakable: pos_vec is a strong direction; everything
    # else is zero. Then a feat aligned with silence+ projects to ~1 there.
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
    # Weak alignment -> silence+ projection sigmoid ≈ 0.525, between
    # thresholds. Silence is `silence+ > threshold`, so:
    #   threshold=0.5 → 0.525 > 0.5 → silence (True)
    #   threshold=0.6 → 0.525 > 0.6 → not silence (False)
    assert post.predict_silence_mask(feats, threshold=0.5)[0]
    assert not post.predict_silence_mask(feats, threshold=0.6)[0]


def test_posteriogram_state_roundtrip():
    post = _make_posteriogram()
    reloaded = PhonologicalPosteriogram.from_state(post.to_state())
    assert reloaded.featnames == post.featnames
    assert set(reloaded.views) == {"ipa"}
    np.testing.assert_array_equal(reloaded.W_bwd, post.W_bwd)


def test_posteriogram_project_activation():
    """project() unifies raw and sigmoid via the `act` argument."""
    in_dim = 4
    pos = np.zeros((3, in_dim), dtype=np.float32)
    pos[0] = np.array([2.0, 0, 0, 0], dtype=np.float32)
    post = _make_posteriogram(in_dim=in_dim, ipa_pos_vecs=pos)
    feats = np.array([[1.0, 0, 0, 0]], dtype=np.float32)

    raw = post.project(feats, view="ipa", act="none")
    np.testing.assert_allclose(raw, [[2.0, 0.0, 0.0]])
    sig = post.project(feats, view="ipa", act="sigmoid")
    np.testing.assert_allclose(sig, [[0.8807971, 0.5, 0.5]], rtol=1e-6)
    # Default activation is "none" (raw projection).
    np.testing.assert_array_equal(post.project(feats, view="ipa"), raw)

    with pytest.raises(ValueError, match="activation"):
        post.project(feats, view="ipa", act="bogus")


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
    np.testing.assert_array_equal(reloaded.posteriogram.W_bwd, model.posteriogram.W_bwd)


def test_from_pretrained_accepts_file_path(tmp_path, monkeypatch):
    monkeypatch.setattr(Recognizer, "__init__", lambda self, **kw: None)
    artifact = _make_artifact()
    torch.save(artifact, tmp_path / "custom.pt")
    model = PhoneModel.from_pretrained(tmp_path / "custom.pt")
    assert model.segmenter().sr == 16000


def test_ssl_encoder_time_to_frame_inverts_frame_to_time():
    from phonological_posteriogram.features import SSLEncoder

    enc = SSLEncoder.__new__(SSLEncoder)  # bypass the model download
    enc.stride_size = 320
    enc.sr = 16000

    for idx in (0, 3, 17, 99):
        assert int(enc.time_to_frame(enc.frame_to_time(np.array([idx])))[0]) == idx


def test_phoible_resolves_unknown_language():
    from phonological_posteriogram.phoible import inventory_id_for_language

    with pytest.raises(ValueError, match="not found in Phoible"):
        inventory_id_for_language("zzz_no_such_language")


def test_phoible_ambiguous_language_falls_back_to_smallest_id():
    """When no candidate inventory is dialect-free, the smallest InventoryID
    is used (with a warning pointing at phoible.org for context)."""
    from phonological_posteriogram.phoible import inventory_id_for_language

    # Abkhaz has inventories 2468 and 2552, both with a SpecificDialect.
    with pytest.warns(UserWarning, match="https://phoible.org/languages/"):
        assert inventory_id_for_language("Abkhaz") == 2468


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("Apurinã", 1804),  # 1092 is "Japiim"; 1804's SpecificDialect is empty
        ("Friulian", 2334),  # 1231 is "Udine"; 2334's is the literal "NA"
    ],
)
def test_phoible_ambiguous_language_prefers_dialect_free(language, expected):
    """A dialect-free inventory is preferred over lower-numbered dialected
    ones, whichever way PHOIBLE spells the missing dialect."""
    from phonological_posteriogram.phoible import inventory_id_for_language

    with pytest.warns(UserWarning, match="https://phoible.org/languages/"):
        assert inventory_id_for_language(language) == expected


def test_phoible_vocab_helpers():
    """Phoible CSV is packaged and parses correctly; vocab_for_inventory drops
    phones panphon doesn't recognize."""
    from phonological_posteriogram.phoible import vocab_for_inventory

    # InventoryID 1 (Korean) lists unreleased stops like "k̚" as allophones.
    phonemes = vocab_for_inventory(1, phoneme=True)
    with pytest.warns(UserWarning, match="not recognized by panphon"):
        allophones = vocab_for_inventory(1, phoneme=False)
    assert len(phonemes) > 0
    assert set(phonemes) <= set(allophones)
    assert "k̚" not in allophones
    assert "k" in allophones


def test_recognizer_closure_release_internal_labels_merge_to_base():
    """Closure/release state channels add internal stop variants whose
    returned labels are merged back to the ordinary phone."""
    rec = Recognizer(
        featnames=["silence+", "cons+", "closure+", "closure-", "release+", "release-"]
    )
    p_cl = rec.vocab.index("p_cl")
    p_rl = rec.vocab.index("p_rl")
    assert rec._output_labels[p_cl] == "p"
    assert rec._output_labels[p_rl] == "p"
    # A plain vocab constraint for "p" should keep both internal variants.
    mask = rec._phones_to_mask(("p",))
    assert p_cl in mask
    assert p_rl in mask


def test_recognizer_fitted_featmap_labels_can_separate_output_from_features():
    """Fitted feature-map keys may encode ``output|feature-spec`` labels."""
    rec = Recognizer(
        featnames=["silence+", "cons+", "closure+"],
        featmap={
            "_": [1, 0, 0],
            "tcl|t͡ʃ_cl": [0, 1, 1],
            "ch|t͡ʃ_rl": [0, 1, 0],
        },
    )
    assert rec._output_labels == ["_", "tcl", "ch"]
    mask = rec._phones_to_mask(("tcl",))
    assert rec.vocab.index("tcl|t͡ʃ_cl") in mask
    post = np.asarray([[0, 1, 1]], dtype=np.float32)
    assert rec.recognize(post, [], sr=16000) == ["tcl"]


def test_panphon_featmap_sparse_rows_ignore_omitted_dimensions():
    from phonological_posteriogram.recognizer import panphon_featmap

    pytest.importorskip("panphon")

    featnames = ["silence+", "cons+", "son+", "closure+"]
    featmap = panphon_featmap(["p", "m"], featnames)
    assert set(featmap["p"]) == {"cons+", "son+"}
    assert "silence+" not in featmap["p"]
    assert "closure+" not in featmap["p"]

    rec = Recognizer(featnames=featnames, featmap=featmap)
    # `closure+` strongly favors "p" if included, but sparse scoring ignores it.
    post = np.asarray([[0.0, 0.25, 1.0, 100.0]], dtype=np.float32)
    assert rec.recognize(post, [], sr=16000) == ["m"]


def test_panphon_featmap_supports_silence_token():
    from phonological_posteriogram.recognizer import panphon_featmap

    pytest.importorskip("panphon")

    featnames = ["silence+", "cons+", "son+", "closure+"]
    featmap = panphon_featmap(["p", "_"], featnames)

    assert set(featmap["p"]) == {"silence+", "cons+", "son+"}
    assert set(featmap["_"]) == {"silence+", "cons+", "son+"}
    assert featmap["p"]["silence+"] == 0
    assert featmap["_"]["silence+"] == 1
    assert featmap["_"]["cons+"] == 0
    assert featmap["_"]["son+"] == 0

    rec = Recognizer(featnames=featnames, featmap=featmap)
    assert rec.vocab == ["p", "_"]
    assert rec._feat_idxs.tolist() == [0, 1, 2]

    assert panphon_featmap(["_"], ["silence+"]) == {"_": {"silence+": 1}}


def test_recognizer_labels_each_segment_at_its_center():
    """Recognizer.recognize maps (posteriogram, boundary times) to one label
    per segment."""
    rec = _make_recognizer(n_feat=3)

    # Pre-computed per-frame posteriogram (30 frames, 3 feats). Centers of
    # [0,10), [10,20), [20,30) are 5, 15, 25 — arrange those to peak at
    # distinct features.
    posteriogram = np.full((30, 3), 0.1, dtype=np.float32)
    posteriogram[5] = [0.9, 0.1, 0.1]  # → "_"
    posteriogram[15] = [0.1, 0.9, 0.1]  # → "phone_0"
    posteriogram[25] = [0.1, 0.1, 0.9]  # → "phone_1"

    # Boundaries are times; frames 10 and 20 at 16 kHz / stride 320 are 0.2 s and 0.4 s.
    labels = rec.recognize(posteriogram, [0.2, 0.4], sr=16000)
    assert labels == ["_", "phone_0", "phone_1"]


def test_recognizer_handles_empty_boundaries():
    """No boundaries → one segment covering the whole feature span."""
    rec = _make_recognizer(n_feat=3)
    posteriogram = np.tile(np.array([0.9, 0.1, 0.1], dtype=np.float32), (8, 1))
    assert rec.recognize(posteriogram, [], sr=16000) == ["_"]


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
    posteriogram[3] = [0.1, 0.1, 0.9]  # → "phone_1" unconstrained
    assert rec.recognize(posteriogram, [], sr=16000) == ["phone_1"]

    # With vocab=["phone_0"], "phone_1" is masked out. Silence "_" is
    # always allowed; with the asymmetric scores below, phone_0 wins.
    posteriogram[3] = [0.1, 0.5, 0.9]  # silence=0.1, phone_0=0.5, phone_1=0.9
    labels = rec.recognize(posteriogram, [], sr=16000, vocab=["phone_0"])
    assert labels == ["phone_0"]


def test_recognizer_vocab_all_unknown_raises():
    """A vocab whose phones decompose to *no* panphon segments raises."""
    from phonological_posteriogram.recognizer import _validate_vocab

    pytest.importorskip("panphon")
    # Pure punctuation/digits: ipa_segs() yields nothing for either token,
    # so there is no segment left to constrain to.
    with (
        pytest.warns(UserWarning, match="not recognized by panphon"),
        pytest.raises(ValueError, match="no panphon-known phones"),
    ):
        _validate_vocab(("123", "!!!"))


def test_filter_panphon_known_decomposes_multisegment_phones():
    """Multi-segment Phoible phones (ts, mb, ...) decompose into their
    component panphon segments instead of being dropped wholesale; the
    result is de-duplicated."""
    from phonological_posteriogram.recognizer import _filter_panphon_known

    pytest.importorskip("panphon")
    # "ts" -> t,s ; "mb" -> m,b ; "b" -> b (b is a dup, dropped).
    out = _filter_panphon_known(("ts", "mb", "b"), context="test")
    assert out == ("t", "s", "m", "b")  # order preserved, "b" not duplicated


def test_filter_panphon_known_keeps_recognized_parts_and_warns():
    """A phone that doesn't round-trip still contributes the segments
    panphon *did* recognize, but a warning flags the partial match."""
    from phonological_posteriogram.recognizer import _filter_panphon_known

    pytest.importorskip("panphon")
    # "k̚" (unreleased k) loses the ◌̚ diacritic in ipa_segs → "k"; the
    # phone is flagged (didn't round-trip) but its recognized "k" is kept.
    with pytest.warns(UserWarning, match="not.*recognized by panphon"):
        out = _filter_panphon_known(("p", "k̚"), context="test")
    assert "p" in out
    assert "k" in out  # the recognized portion survives


def test_phone_model_recognize_takes_frame_boundaries(monkeypatch):
    """PhoneModel.recognize takes frame-index boundaries (as returned by
    segment()) and converts them to times for the Recognizer."""
    model = _make_phone_model(monkeypatch, _make_posteriogram(), dict(NET_SPEC))
    posteriogram = np.full((30, 3), 0.1, dtype=np.float32)
    posteriogram[5] = [0.9, 0.1, 0.1]  # → "_"
    posteriogram[15] = [0.1, 0.9, 0.1]  # → "phone_0"
    posteriogram[25] = [0.1, 0.1, 0.9]  # → "phone_1"
    assert model.recognize(posteriogram, np.array([10, 20])) == ["_", "phone_0", "phone_1"]


def test_phone_model_segmenter_hparam_override(monkeypatch):
    """The point of the refactor: building a Segmenter with different
    hparams is cheap and shares the (expensive) posteriogram weights."""
    model = _make_phone_model(monkeypatch, _make_posteriogram(), dict(NET_SPEC))
    seg_a = model.segmenter()
    seg_b = model.segmenter({"combined_prominence": 0.25})

    assert seg_a.posteriogram is seg_b.posteriogram  # weights not copied
    assert seg_b.hparams["combined_prominence"] == 0.25
    assert (
        seg_a.hparams["combined_prominence"] == Segmenter.default_hparams()["combined_prominence"]
    )


def test_segmenter_hparams_merge_onto_defaults():
    """A partial hparams dict still yields a complete config (missing keys
    filled from default_hparams)."""
    post = _make_posteriogram()
    seg = Segmenter(post, sr=16000, hparams={"combined_prominence": 0.01})
    assert seg.hparams["combined_prominence"] == 0.01
    assert seg.hparams["activation"] == "none"  # filled from defaults
    assert "combined_signals" in seg.hparams


FRAME_DELTA_1 = {"name": "frame_delta", "kwargs": {"offset": 1}, "shift": 0}


def _segment_step(second_half, *, distance, signals):
    """Segment 30 frames: 15 of [1,0,0,0] then 15 of ``second_half``."""
    feats = np.array([[1, 0, 0, 0]] * 15 + [second_half] * 15, dtype=np.float32)
    post = _make_posteriogram(in_dim=4, n_feat=3, ipa_pos_vecs=np.eye(3, 4))
    seg = Segmenter(
        post,
        sr=16000,
        hparams={
            "distance": distance,
            "combined_signals": signals,
            "drop_closure_release": False,
            "snap_silence": False,
        },
    )
    return seg.segment(feats, np.zeros(len(feats) * 320, dtype=np.float32)).tolist()


@pytest.mark.parametrize("distance", ["cosine", "l2"])
def test_segmenter_finds_direction_change(distance):
    assert _segment_step([0, 1, 0, 0], distance=distance, signals=[FRAME_DELTA_1]) == [14]


def test_segmenter_distance_hparam():
    """A pure magnitude change is invisible to cosine distance but not to l2."""
    assert _segment_step([3, 0, 0, 0], distance="cosine", signals=[FRAME_DELTA_1]) == []
    assert _segment_step([3, 0, 0, 0], distance="l2", signals=[FRAME_DELTA_1]) == [14]


def test_segmenter_duplicate_signal_specs_all_contribute():
    """combined_signals may repeat a signal type; every copy enters the
    fuzzy-AND product, so two frame_delta copies shifted apart cancel out."""
    shifted = {**FRAME_DELTA_1, "shift": 2}
    step = [0, 1, 0, 0]
    assert _segment_step(step, distance="cosine", signals=[FRAME_DELTA_1]) == [14]
    assert _segment_step(step, distance="cosine", signals=[FRAME_DELTA_1, shifted]) == []


def test_pair_distance_methods():
    from phonological_posteriogram.segmenter import _pair_distance

    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Cosine: identical rows -> 0; orthogonal rows -> 1.
    np.testing.assert_allclose(_pair_distance(a, b, "cosine"), [0.0, 1.0])
    # L2: ||(0,0)|| = 0; ||(1,-1)|| = sqrt(2).
    np.testing.assert_allclose(_pair_distance(a, b, "l2"), [0.0, np.sqrt(2)])
    # Cosine zero-norm fallback: 1.0.
    zeros = np.zeros_like(a)
    np.testing.assert_array_equal(_pair_distance(a, zeros, "cosine"), [1.0, 1.0])

    with pytest.raises(ValueError, match="distance"):
        _pair_distance(a, b, "bogus")


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
        # Phone sequence with silence padding so "_" appears in the vocab. A
        # closure (tcl) precedes the t release so the TIMIT closure/release
        # fit sees both a _cl and a _rl label.
        seq = [
            ("h#", "_"),
            ("p", "p"),
            ("iy", "i"),
            ("tcl", np.nan),
            ("t", "t"),
            ("h#", "_"),
        ]
        for i, (timit_phn, ipa) in enumerate(seq):
            rows.append(
                {
                    "audio_path": utt,
                    "min": i * 0.10,
                    "max": (i + 1) * 0.10,
                    "timit_phn": timit_phn,
                    "ipa": ipa,
                    "l_1": seq[i - 1][1] if i > 0 else None,
                    "r_1": seq[i + 1][1] if i < len(seq) - 1 else None,
                    "feat": rng.normal(size=in_dim).astype(np.float32),
                    "split": "train",
                }
            )

    df = pd.DataFrame(rows)
    df.attrs["hf_repo"] = "microsoft/wavlm-large"
    df.attrs["encoder_layer"] = -1
    df.attrs["pool"] = "center"
    df.attrs["sr"] = 16000

    pkl_path = tmp_path / "feats.pkl"
    df.to_pickle(pkl_path)
    return pkl_path


def test_training_train_requires_features_attrs(tmp_path):
    """Missing df.attrs keys produce a clear error, not an opaque crash."""
    pd = pytest.importorskip("pandas")
    train_mod = pytest.importorskip("phonological_posteriogram.training.train")

    pkl_path = tmp_path / "bad.pkl"
    pd.DataFrame({"ipa": ["a"], "feat": [np.zeros(4, dtype=np.float32)]}).to_pickle(pkl_path)

    args = train_mod._get_args(
        [
            "--features_pkl",
            str(pkl_path),
            "--output_dir",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ValueError, match="df.attrs"):
        train_mod.run(args)


def test_mel_frames_per_s3m_frame():
    from phonological_posteriogram.segmenter import mel_frames_per_s3m_frame

    assert mel_frames_per_s3m_frame(16000, 10) == 2
    assert mel_frames_per_s3m_frame(32000, 10) == 1
    assert mel_frames_per_s3m_frame(8000, 10) == 4
    with pytest.raises(ValueError, match="whole multiple"):
        mel_frames_per_s3m_frame(48000, 10)
    with pytest.raises(ValueError, match="whole multiple"):
        mel_frames_per_s3m_frame(16000, 15)


def test_extract_features_rejects_non_integral_sr(tmp_path):
    ef_mod = pytest.importorskip("phonological_posteriogram.training.extract_features")
    base = ["--dataset_csv", str(tmp_path / "x.csv"), "--output_path", str(tmp_path / "x.pkl")]
    assert ef_mod._get_args(base).sr == 16000
    assert ef_mod._get_args([*base, "--sr", "32000"]).sr == 32000
    with pytest.raises(SystemExit):
        ef_mod._get_args([*base, "--sr", "48000"])


def test_training_train_mel_frame_shift_ms(tmp_path):
    train_mod = pytest.importorskip("phonological_posteriogram.training.train")

    pkl_path = _make_synthetic_features_pkl(tmp_path)
    base = ["--features_pkl", str(pkl_path), "--output_dir", str(tmp_path / "trained")]
    with pytest.raises(ValueError, match="whole multiple"):
        train_mod.run(train_mod._get_args([*base, "--mel_frame_shift_ms", "15"]))
    out = train_mod.run(train_mod._get_args([*base, "--mel_frame_shift_ms", "20"]))
    assert PhoneModel.from_pretrained(out).hparams["mel_frame_shift_ms"] == 20


def test_training_train_end_to_end(tmp_path):
    """Fit on a synthetic features pkl, save, and reload via from_pretrained."""
    train_mod = pytest.importorskip("phonological_posteriogram.training.train")

    pkl_path = _make_synthetic_features_pkl(tmp_path)
    out_dir = tmp_path / "trained"

    args = train_mod._get_args(
        [
            "--features_pkl",
            str(pkl_path),
            "--output_dir",
            str(out_dir),
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
    # mel_frame_shift_ms now lives in the algorithm hparams, not net_spec.
    assert reloaded.hparams["mel_frame_shift_ms"] == 10
    # Silence detection lives on the posteriogram (silence+ channel); the
    # fitted vocab includes "_", so the feature is present.
    assert "silence+" in reloaded.posteriogram.featnames
