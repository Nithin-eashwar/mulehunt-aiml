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

from app import data  # noqa: E402
from text.anonymise import anonymise  # noqa: E402
from text.classifier import predict_message  # noqa: E402
from text.dataset import SAMPLE, load_messages  # noqa: E402

st.title("Scam message checker")
art = data.text_model()
if art is None:
    st.warning("No text model yet: run `python -m text.classifier`.")
    data.footer()
    st.stop()
st.caption(f"Model trained on {art['trained_on']} messages. Examples below are synthetic (written by us), "
           "never contributors' messages.")
samples = load_messages(SAMPLE)
cols = st.columns(4)
for i, t in enumerate(sorted(samples.scam_type.unique())):
    if cols[i % 4].button(t.replace("_", " "), use_container_width=True):
        st.session_state["msg"] = samples[samples.scam_type == t].text.iloc[0]
text = st.text_area("Message", key="msg", height=120, placeholder="Paste a message...")
if text.strip():
    clean, _ = anonymise(text)
    out = predict_message(clean, art)
    c1, c2 = st.columns(2)
    c1.metric("Scam probability", f"{out['scam_prob']:.1%}")
    c2.metric("Likely type", out["scam_type"].replace("_", " "))
    marked = html.escape(clean)
    for tok, w in out["top_tokens"]:
        if w > 0 and len(tok) > 1:
            pattern = rf"(?i)(?<![\w;]){re.escape(html.escape(tok))}(?!\w)"
            marked = re.sub(pattern, lambda m: f"<mark>{m.group(0)}</mark>", marked)
    st.markdown(f"<div style='padding:8px;border:1px solid #ccc;border-radius:6px'>{marked}</div>",
                unsafe_allow_html=True)
    st.write("Top scam-indicating tokens:", ", ".join(f"`{t}` ({w:+.2f})" for t, w in out["top_tokens"]))
data.footer()
