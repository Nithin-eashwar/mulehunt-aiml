"""Page 3: the fraudster-vs-detector game with minimax and alpha-beta."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st  # noqa: E402

from app import data  # noqa: E402
from app.logic import game_run  # noqa: E402
from mulehunt import b_params as P  # noqa: E402

st.title("Evasion game")
st.write("Each round the detector commits to a rule (threshold, velocity on/off); the fraudster sees it and "
         "chooses how much to move and into how many parts. Fixed thresholds get split under; minimax gives "
         "the detector its worst-case-safe rule.")
fa = data.false_alarms()
c1, c2, c3 = st.columns(3)
X = c1.slider("Money to move X (Rs)", 50_000, 500_000, P.GAME_X, step=10_000)
R = c2.slider("Rounds R", 1, 4, P.GAME_R)
lam = c3.select_slider("False-alarm price lambda (Rs per alert)", [0, 25, 100, 400], value=P.ALERT_COST)
options = ["minimax"] + [f"fixed T={T:,} velocity {'on' if v else 'off'}"
                         for T in P.GAME_THRESHOLDS for v in (False, True)]
pick = st.selectbox("Detector policy", options)
policy = "minimax" if pick == "minimax" else (int(pick.split("=")[1].split()[0].replace(",", "")), pick.endswith("on"))
with st.spinner("Solving the game..."):
    rounds, acc, counts = game_run(X, R, lam, policy, fa)
m = st.columns(4)
m[0].metric("Moved uncaught", f"Rs {acc['uncaught'] / 1000:,.1f}k")
m[1].metric("Caught (frozen)", f"Rs {acc['caught'] / 1000:,.1f}k")
m[2].metric("Mule fees + commission", f"Rs {acc['fees'] / 1000:,.1f}k")
m[3].metric("Game value (fraudster)", f"Rs {acc['payoff'] / 1000:,.1f}k")
st.subheader("Fraudster's best splitting strategy per round")
st.dataframe(rounds.round(2), hide_index=True, use_container_width=True)
if counts:
    st.subheader("Game-tree nodes visited")
    st.bar_chart(counts)
data.footer()
