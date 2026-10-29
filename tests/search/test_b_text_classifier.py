import numpy as np
import pandas as pd

from text.classifier import (TOKEN_PATTERN, build_models, load_artifact, predict_message, save_artifact)
from text.dataset import SAMPLE, import_form, load_messages, validate_messages


def _fitted():
    df = load_messages(SAMPLE)
    y = (df.label == "scam").astype(int)
    b = build_models()["nb_word_char"][0].fit(df.text, y)
    m = build_models()["nb_word"][0].fit(df.text, df.scam_type)
    return df, b, m


def test_sample_messages_valid_and_synthetic():
    df = load_messages(SAMPLE)
    assert validate_messages(df) == [] and df.attrs["synthetic"]
    assert df.msg_id.str.startswith("SYN").all() and len(df) >= 20


def test_every_pipeline_fits():
    df = load_messages(SAMPLE)
    y = (df.label == "scam").astype(int)
    for name, (pipe, _) in build_models().items():
        p = pipe.fit(df.text, y).predict_proba(df.text)[:, 1]
        assert p.shape == (len(df),), name


def test_save_load_round_trip(tmp_path):
    df, b, m = _fitted()
    path = tmp_path / "nb.joblib"
    save_artifact(b, m, sorted(df.scam_type.unique()), len(df), path)
    art = load_artifact(path)
    assert art["trained_on"] == len(df)
    np.testing.assert_array_equal(art["binary"].predict_proba(df.text), b.predict_proba(df.text))
    out = predict_message("Your KYC expired, update at <URL> now", art)
    assert set(out) == {"scam_prob", "scam_type", "top_tokens"}
    assert 0 <= out["scam_prob"] <= 1 and out["scam_type"] in art["labels"]


def test_placeholders_survive_tokenisation():
    vec = build_models()["nb_word"][0].named_steps["features"]
    tokens = vec.build_analyzer()("Pay at <URL> or call <PHONE> via <UPI>")
    assert {"<url>", "<phone>", "<upi>"} <= set(tokens)
    assert "<" in TOKEN_PATTERN


def test_import_form(tmp_path):
    form = pd.DataFrame({
        "Timestamp": ["t1", "t2", "t3", "t4"],
        "Paste the message": ["KYC expired call 9876543210", "kyc  expired call 9876543210!", "Dear Ravi pay Rs 500",
                              "no consent row"],
        "Where did you receive it?": ["SMS", "SMS", "WhatsApp", "SMS"],
        "Scam type": ["Fake KYC", "Fake KYC", "Not sure", "Loan app"],
        "Language": ["English", "English", "Hinglish", "English"],
        "I agree this anonymised text may be used": ["I agree", "I agree", "I agree", ""],
    })
    src = tmp_path / "form.csv"
    form.to_csv(src, index=False)
    out = tmp_path / "messages.csv"
    df = import_form(src, out)
    assert len(df) == 2  # one duplicate after normalisation, one without consent
    saved = load_messages(out)
    assert validate_messages(saved) == []
    assert "9876543210" not in " ".join(saved.text)
    assert set(saved.scam_type) == {"fake_kyc", "other_scam"} and set(saved.source) == {"form"}
    assert df.needs_review.tolist() == [False, True]
