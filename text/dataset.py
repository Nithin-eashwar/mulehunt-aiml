"""Scam-message dataset tools (Task 9).

    python -m text.dataset import <form_export.csv> [--out data/raw/scam_messages/messages.csv]

Maps a Google Form export to the schema, anonymises every message, drops rows without
consent, de-duplicates on normalised text, writes messages.csv and prints the rows that
need a human review. Keep the raw form export OUTSIDE the repo; review the flagged rows by
hand before committing messages.csv.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from mulehunt.b_params import SCAM_TYPES
from text.anonymise import anonymise

MESSAGES = Path("data/raw/scam_messages/messages.csv")
SAMPLE = Path("data/raw/scam_messages/sample_messages.csv")
COLUMNS = ["msg_id", "text", "label", "scam_type", "language", "channel", "source"]
LABELS = ("scam", "genuine")
LANGUAGES = ("english", "hindi", "hinglish", "other")
CHANNELS = ("sms", "whatsapp", "other")
SOURCES = ("form", "self", "public")
TYPE_WORDS = {"kyc": "fake_kyc", "arrest": "digital_arrest", "police": "digital_arrest", "job": "job_task",
              "task": "job_task", "loan": "loan_app", "invest": "fake_investment", "trading": "fake_investment",
              "genuine": "genuine", "real": "genuine", "not a scam": "genuine"}


def _find(columns, *words):
    for c in columns:
        if any(w in c.lower() for w in words):
            return c
    raise KeyError(f"no form column mentions any of {words}")


def _scam_type(value: str) -> str:
    v = str(value).strip().lower()
    if v in SCAM_TYPES:
        return v
    for word, t in TYPE_WORDS.items():
        if word in v:
            return t
    return "other_scam"  # includes "not sure"


def _pick(value: str, allowed, default="other") -> str:
    v = str(value).strip().lower()
    return next((a for a in allowed if a in v), default)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w<>]+", " ", str(text).lower())).strip()


def import_form(path, out=MESSAGES) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str).fillna("")
    cols = raw.columns
    consent = _find(cols, "agree", "consent")
    raw = raw[raw[consent].str.strip().str.lower().isin(["yes", "true", "i agree", "agree", "1", "checked"])
              | raw[consent].str.lower().str.contains("agree")]
    text_col = _find(cols, "message", "text")
    rows = []
    for _, r in raw.iterrows():
        clean, review = anonymise(r[text_col])
        t = _scam_type(r[_find(cols, "type")])
        rows.append({"text": clean.strip(), "label": "genuine" if t == "genuine" else "scam", "scam_type": t,
                     "language": _pick(r[_find(cols, "language")], LANGUAGES),
                     "channel": _pick(r[_find(cols, "where", "channel", "received")], CHANNELS),
                     "source": "form", "needs_review": review})
    df = pd.DataFrame(rows)
    df = df[df.text.str.len() > 0]
    df = df.loc[~df.text.map(normalise).duplicated()].reset_index(drop=True)
    df.insert(0, "msg_id", [f"MSG{i:05d}" for i in range(1, len(df) + 1)])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(out, index=False, lineterminator="\n")
    flagged = df[df.needs_review]
    print(f"wrote {out}: {len(df)} messages ({len(flagged)} need review)")
    for r in flagged.itertuples():
        print(f"  REVIEW {r.msg_id}: {r.text[:120]}")
    return df


def validate_messages(df: pd.DataFrame) -> list[str]:
    """An empty list means valid (same convention as simulator.schema)."""
    errors = []
    if list(df.columns) != COLUMNS:
        return [f"columns {list(df.columns)} != {COLUMNS}"]
    for col, allowed in (("label", LABELS), ("scam_type", SCAM_TYPES), ("language", LANGUAGES),
                         ("channel", CHANNELS), ("source", SOURCES)):
        bad = set(df[col]) - set(allowed)
        if bad:
            errors.append(f"{col}: unexpected values {sorted(bad)}")
    if df.msg_id.duplicated().any():
        errors.append("msg_id not unique")
    if ((df.label == "genuine") != (df.scam_type == "genuine")).any():
        errors.append("label and scam_type disagree")
    if df.text.str.strip().eq("").any():
        errors.append("empty text")
    return errors


def load_messages(path=None) -> pd.DataFrame:
    """messages.csv when present, else the synthetic fixtures; validated."""
    path = Path(path) if path else (MESSAGES if MESSAGES.exists() else SAMPLE)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    errors = validate_messages(df)
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    df.attrs["path"] = str(path)
    df.attrs["synthetic"] = Path(path).name == SAMPLE.name
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scam-message dataset tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="import a Google Form export")
    imp.add_argument("form_csv")
    imp.add_argument("--out", default=str(MESSAGES))
    a = ap.parse_args(argv)
    if Path(a.form_csv).resolve().is_relative_to(Path.cwd().resolve()):
        print("warning: keep the raw form export outside the repo", file=sys.stderr)
    import_form(a.form_csv, a.out)


if __name__ == "__main__":
    main()
