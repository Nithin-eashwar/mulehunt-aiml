"""Page 3: the fraudster-vs-detector game with minimax and alpha-beta."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import data, style  # noqa: E402
from app.logic import game_run  # noqa: E402
from mulehunt import b_params as P  # noqa: E402

style.header("Evasion game",
             "Each day the bank commits to a rule, then the fraudster sees it and decides how much to move and in "
             "how many parts. Any fixed threshold gets split under; minimax finds the rule that is safest against "
             "the best possible reply.")
fa = data.false_alarms()
c1, c2, c3 = st.columns(3, gap="large")
X = c1.slider("Money to move", 50_000, 500_000, P.GAME_X, step=10_000, format="Rs %d")
R = c2.slider("Days", 1, 4, P.GAME_R)
lam = c3.select_slider("Cost of one false alarm", [0, 25, 100, 400], value=P.ALERT_COST,
                       format_func=lambda v: f"Rs {v}")
options = ["Minimax (worst-case safe)"] + [f"Always Rs {T:,} threshold{', velocity rule on' if v else ''}"
                                           for T in P.GAME_THRESHOLDS for v in (False, True)]
pick = st.selectbox("Bank's policy", options)
if pick.startswith("Minimax"):
    policy = "minimax"
else:
    T = int(pick.split("Rs ")[1].split()[0].replace(",", ""))
    policy = (T, pick.endswith("on"))
with st.spinner("Solving the game"):
    rounds, acc, counts = game_run(X, R, lam, policy, fa)

m = st.columns(4)
m[0].metric("Moved past the bank", f"Rs {acc['uncaught'] / 1000:,.1f}k")
m[1].metric("Frozen", f"Rs {acc['caught'] / 1000:,.1f}k")
m[2].metric("Spent on mules", f"Rs {acc['fees'] / 1000:,.1f}k")
m[3].metric("Fraudster's payoff", f"Rs {acc['payoff'] / 1000:,.1f}k")

st.markdown("### The fraudster's best reply, day by day")
days = rounds[rounds["round"] != "frozen"].copy()
view = pd.DataFrame({
    "Day": days["round"].astype(int),
    "Bank's rule": [f"Rs {t:,.0f}{' + velocity' if v else ''}" for t, v in zip(days.threshold, days.velocity)],
    "Moves": [f"{f:.0%} of Rs {r:,.0f}" for f, r in zip(days.fraction, days.remaining_before)],
    "Parts": days.parts.astype(int),
    "Part size": [f"Rs {x:,.0f}" for x in days.moved / days.parts],
    "Frozen": [f"Rs {x:,.0f}" for x in days.caught], "Got through": [f"Rs {x:,.0f}" for x in days.uncaught],
})
st.dataframe(view, hide_index=True, use_container_width=True)

if counts:
    st.markdown("### Game-tree nodes visited for the same answer")
    nodes = pd.DataFrame({"method": ["plain minimax", "alpha-beta", "alpha-beta, ordered"], "nodes": list(counts.values())})
    st.altair_chart(style.bars(nodes, "method", "nodes", height=200), use_container_width=True)
    saving = 1 - counts["alpha-beta + ordering"] / counts["minimax"]
    st.caption(f"Alpha-beta with move ordering skips {saving:.0%} of the tree and returns the same rule and value.")
data.footer()
