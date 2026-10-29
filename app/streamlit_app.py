"""MuleHunt dashboard: streamlit run app/streamlit_app.py (from the repo root)."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="MuleHunt", page_icon=":material/travel_explore:", layout="wide")

from app import style  # noqa: E402

style.setup()
pages = [
    st.Page("views/1_transaction_check.py", title="Transaction check", icon=":material/fact_check:", default=True),
    st.Page("views/2_mule_trail.py", title="Mule trail explorer", icon=":material/account_tree:"),
    st.Page("views/3_evasion_game.py", title="Evasion game", icon=":material/sports_esports:"),
    st.Page("views/4_scam_checker.py", title="Scam message checker", icon=":material/sms:"),
]
st.navigation(pages).run()
