"""Page 1: score a payment by the riskier of its two accounts (B-D19)."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import html  # noqa: E402

import streamlit as st  # noqa: E402

from app import data, style  # noqa: E402
from app.logic import score_account, transaction_check  # noqa: E402
from mulehunt.features import FEATURE_DESCRIPTIONS  # noqa: E402

style.header("Transaction check",
             "The models score accounts, not single payments. A payment takes the higher risk of its sender and "
             "receiver; the bars show where each sits against the model's flag threshold.")
data.data_banner()
g, kind = data.graph()
risk, risk_src = data.risk()
bundle = data.bundle()
feats = data.features()
threshold = float(bundle["threshold"]) if bundle and kind == "real" else 0.5
if feats is None:
    st.warning("Feature table missing, so only risk scores are shown. Rebuild it with `python -m mulehunt.features`.")
if kind == "toy":
    feats = None  # the saved models were trained on the real feature table

demo = data.demo_txns()
c1, c2 = st.columns([1.15, 2], gap="medium")
mode = c1.segmented_control("Look up", ["Demo payment", "Transaction id", "Account id"], default="Demo payment")
if mode == "Transaction id":
    txn, account = c2.text_input("Transaction id", value=str(demo.txn_id.iloc[0])).strip(), None
elif mode == "Account id":
    txn, account = None, c2.text_input("Account id", value=g.ids[0]).strip()
else:
    choice = c2.selectbox("Demo payment", demo.txn_id + "  ·  " + demo.kind)
    txn, account = choice.split()[0], None


def reasons(score, title):
    st.markdown(f"### {title}")
    if not score["explain"]:
        st.caption("No feature row for this account, so no reasons to show.")
        return
    st.markdown("<div class='mh-panel'>" + style.reasons(score["explain"], FEATURE_DESCRIPTIONS) + "</div>",
                unsafe_allow_html=True)


try:
    if txn:
        res = transaction_check(g, txn, risk, threshold, feats, bundle)
        e, d = res["edge"], res["decision"]
        p = d["risk_prob"]
        flagged = p is not None and p >= threshold
        verdict = "Send to review" if flagged else "Below threshold"
        st.markdown(
            f"<div class='mh-panel'><div class='mh-verdict'><span class='v' style='color:"
            f"{style.SIGNAL if flagged else style.CALM}'>{verdict}</span>"
            f"<span class='n'>payment risk {'n/a' if p is None else f'{p:.3f}'} · threshold {threshold:.3f}</span></div>"
            f"<p style='margin:.4rem 0 1rem;color:{style.MUTED}'><span class='mh-id'>{html.escape(e.txn_id)}</span> · "
            f"<span class='mh-id'>{e.sender}</span> → <span class='mh-id'>{e.receiver}</span> · "
            f"<span class='mh-id'>Rs {e.amount:,.2f}</span></p>"
            + style.gauge(f"Sender {e.sender}", res["sender"]["risk_prob"], threshold)
            + style.gauge(f"Receiver {e.receiver}", res["receiver"]["risk_prob"], threshold)
            + f"<p style='font-size:.78rem;color:{style.MUTED};margin:0'>Scores from {risk_src}. "
              "The dark tick marks the threshold.</p></div>", unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")
        with left:
            reasons(res["sender"], "Why the sender scores this way")
        with right:
            reasons(res["receiver"], "Why the receiver scores this way")
    elif account:
        if account not in g.index:
            st.error(f"No account {account} in this dataset. Ids look like ACC000123.")
        else:
            s = score_account(account, risk, threshold, feats, bundle)
            st.markdown("<div class='mh-panel'>" + style.gauge(f"Account {account}", s["risk_prob"], threshold)
                        + "</div>", unsafe_allow_html=True)
            reasons(s, "Top reasons")
except KeyError:
    st.error(f"No transaction {txn} in this dataset. Ids look like TXN00590803.")
data.footer()
