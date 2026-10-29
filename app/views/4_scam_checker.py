"""Page 4: score a message with the Naive Bayes scam classifier."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import html  # noqa: E402
import re  # noqa: E402

import streamlit as st  # noqa: E402

from app import data, style  # noqa: E402
from text.anonymise import anonymise  # noqa: E402
from text.classifier import predict_message  # noqa: E402
from text.dataset import SAMPLE, load_messages  # noqa: E402

style.header("Scam message checker",
             "Paste a message a customer received. It is scrubbed of numbers, links and IDs first, then scored by a "
             "Naive Bayes model trained on word and character n-grams, so Hinglish spellings still match.")
art = data.text_model()
if art is None:
    st.warning("No text model yet. Train it with `python -m text.classifier`.")
    data.footer()
    st.stop()
samples = load_messages(SAMPLE)
types = sorted(samples.scam_type.unique())
pick = st.pills("Try an example (written by us, never a contributor's message)", types,
                format_func=lambda t: t.replace("_", " "))
if pick and st.session_state.get("_last_pick") != pick:
    st.session_state["msg"] = samples[samples.scam_type == pick].text.iloc[0]
    st.session_state["_last_pick"] = pick
text = st.text_area("Message", key="msg", height=110, placeholder="Paste a message here")
if text.strip():
    clean, _ = anonymise(text)
    out = predict_message(clean, art)
    p = out["scam_prob"]
    flagged = p >= 0.5
    marked = html.escape(clean)
    for tok, w in out["top_tokens"]:
        if w > 0 and len(tok) > 1:
            pattern = rf"(?i)(?<![\w;]){re.escape(html.escape(tok))}(?!\w)"
            marked = re.sub(pattern, lambda m: f"<mark>{m.group(0)}</mark>", marked)
    st.markdown(
        f"<div class='mh-panel' style='margin:.6rem 0 1rem'><div class='mh-verdict'>"
        f"<span class='v' style='color:{style.SIGNAL if flagged else style.CALM}'>"
        f"{'Likely scam' if flagged else 'Looks genuine'}</span>"
        f"<span class='n'>P(scam) {p:.1%}</span>"
        + (style.tag(out['scam_type'].replace('_', ' '), True) if flagged else "")
        + "</div></div>", unsafe_allow_html=True)
    st.markdown("### What the model reacted to")
    st.markdown(f"<div class='mh-msg'>{marked}</div>", unsafe_allow_html=True)
    st.caption("Highlighted: the tokens that pushed hardest towards 'scam'. "
               + ", ".join(f"{t} ({w:+.2f})" for t, w in out["top_tokens"]))
st.caption(f"Model trained on {art['trained_on']:,} messages: real UCI SMS (CC BY 4.0), Indian scam messages written "
           "from cited news and bank advisories, and synthetic Indian messages. On cited scams it never saw in "
           "training it catches 54 of 56. No message was collected from a person.")
data.footer()
