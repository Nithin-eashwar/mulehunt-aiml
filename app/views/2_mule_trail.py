"""Page 2: trace a flagged victim payment through the mule network, live."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from app import data  # noqa: E402
from app.logic import run_trail, trail_html, trail_metrics  # noqa: E402
from mulehunt import b_params as P  # noqa: E402

st.title("Mule trail explorer")
st.write("Start from a flagged victim payment and search forward along **time-respecting** hops (each hop "
         "strictly later than the money arrived, within the window) until a known cash-out account.")
data.data_banner()
g, kind = data.graph()
risk, _ = data.risk()
bundle = data.bundle()
threshold = float(bundle["threshold"]) if bundle and kind == "real" else 0.3
demo = data.demo_txns()
victims = demo[demo.ring_id.notna()]
c1, c2, c3 = st.columns([2, 1, 1])
choice = c1.selectbox("Flagged victim payment", victims.txn_id + "  -  " + victims.kind)
algorithm = c2.radio("Algorithm", ["BFS", "IDDFS", "A*"], horizontal=True)
window = c3.select_slider("Hop window (h)", [12, 24, 48], value=P.HOP_WINDOW_H)
reveal = st.toggle("Reveal ground truth (colour by true tier, mark mules the classifier missed)")
txn = choice.split()[0]

with st.spinner(f"Running {algorithm} (budget {P.MAX_EXPANSIONS:,} expansions)..."):
    res = run_trail(g, txn, algorithm, window, risk)
st.dataframe(trail_metrics(res), hide_index=True, use_container_width=True)
if res.truncated:
    st.warning("The expansion budget was hit: this is 'truncated', not 'no trail'.")
elif not res.found:
    st.info("No time-respecting trail to a known cash-out within this window.")
components.html(trail_html(g, txn, res.path, risk, data.rings(), reveal, threshold), height=500)
st.caption("Thick edges: the victim payment and the trail. Grey: one-hop context. Node colour: risk_prob "
           "(green low, red high), or true tier when revealed.")
data.footer()
