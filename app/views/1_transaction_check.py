"""Page 1: score a payment by the riskier of its two accounts (B-D19)."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import data  # noqa: E402
from app.logic import score_account, transaction_check  # noqa: E402
from mulehunt.features import FEATURE_DESCRIPTIONS  # noqa: E402

st.title("Transaction check")
st.write("The fraud models score **accounts**, not single payments, so a payment gets the higher risk of its "
         "sender and receiver, with the top reasons for that account.")
data.data_banner()
g, kind = data.graph()
risk, risk_src = data.risk()
bundle = data.bundle()
feats = data.features()
threshold = float(bundle["threshold"]) if bundle and kind == "real" else 0.5
if feats is None:
    st.warning("`data/processed/features.parquet` is missing, so only risk scores are shown. "
               "Regenerate it with `python -m mulehunt.features`.")
if kind == "toy":
    feats = None  # the saved models were trained on the real feature table

demo = data.demo_txns()
mode = st.radio("Look up", ["Demo transaction", "Transaction id", "Account id"], horizontal=True)
if mode == "Demo transaction":
    choice = st.selectbox("Demo transaction", demo.txn_id + "  -  " + demo.kind)
    txn, account = choice.split()[0], None
elif mode == "Transaction id":
    txn, account = st.text_input("txn_id", value=str(demo.txn_id.iloc[0])).strip(), None
else:
    txn, account = None, st.text_input("account_id", value=g.ids[0]).strip()


def show(score, heading):
    st.subheader(heading)
    p = score["risk_prob"]
    c1, c2 = st.columns(2)
    c1.metric("risk_prob", "n/a" if p is None else f"{p:.3f}")
    c2.metric(f"flagged at {threshold:.3f}", "YES" if score["flagged"] else "no")
    if score["explain"]:
        ex = pd.DataFrame(score["explain"], columns=["feature", "effect"])
        ex["meaning"] = ex.feature.map(FEATURE_DESCRIPTIONS)
        st.bar_chart(ex.set_index("feature").effect, horizontal=True)
        st.dataframe(ex, hide_index=True, use_container_width=True)


try:
    if txn:
        res = transaction_check(g, txn, risk, threshold, feats, bundle)
        e = res["edge"]
        st.write(f"**{e.txn_id}**: {e.sender} -> {e.receiver}, Rs {e.amount:,.2f}")
        d = res["decision"]
        st.metric("Payment risk (higher of the two accounts)",
                  "n/a" if d["risk_prob"] is None else f"{d['risk_prob']:.3f}", help=f"from {risk_src}")
        left, right = st.columns(2)
        with left:
            show(res["sender"], f"Sender {e.sender}")
        with right:
            show(res["receiver"], f"Receiver {e.receiver}")
    elif account:
        if account not in g.index:
            st.error(f"Unknown account {account}")
        else:
            show(score_account(account, risk, threshold, feats, bundle), f"Account {account}")
except KeyError:
    st.error(f"Unknown transaction {txn}")
data.footer()
