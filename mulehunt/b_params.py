"""Person B's decision values (plan section 1), one constant per decision.

Values tagged PROPOSAL were chosen by the plan and not yet agreed with Person A;
changing one is a single edit here. Every report prints the values it used
(see `params_table`).
"""

import pandas as pd

HONEST_FRAMING = ("Real transaction data is legally unavailable; we built a simulator calibrated to NPCI/RBI "
                  "statistics and validated against the IBM AML benchmark.")

# B-D1 (PROPOSAL): search goal set = accounts with role == "cashout" (the only ground-truth read in search).
GOAL_ROLE = "cashout"
# B-D2 (agreed): next hop strictly later than the arrival and at most this many hours after it.
HOP_WINDOW_H = 48
WINDOW_SENSITIVITY_H = (12, 24, 48)
# B-D3 (PROPOSAL): a search starts right after a victim -> tier-1 mule payment lands.
# B-D4 (PROPOSAL): edge cost = -log(risk_prob of the receiver); unknown accounts get this risk.
DEFAULT_RISK = 0.05
# B-D6 (PROPOSAL): search budgets.
DLS_LIMIT = 6
MAX_DEPTH = 8
MAX_EXPANSIONS = 200_000

# B-D7 (PROPOSAL): game money and rounds.
GAME_X = 200_000
GAME_R = 3
DEEP_R = 6
DEEP_DEPTHS = (4, 6, 8)
# B-D8 (PROPOSAL, thresholds agreed): detector moves = threshold x velocity rule.
GAME_THRESHOLDS = (10_000, 25_000, 50_000)  # same as load_config()["fraud"]["thresholds"]
VELOCITY_K = 3
# B-D9 (PROPOSAL): fraudster moves = fraction of remaining x number of equal parts.
FRACTIONS = (0.5, 1.0)
PARTS = (1, 2, 4, 8)
# B-D10 (PROPOSAL): mule cost.
MULE_FEE = 1_500
COMMISSION = 0.05
# B-D11 (PROPOSAL): detector false-alarm cost per alert, and the sweep.
ALERT_COST = 100
ALERT_COST_SWEEP = (0, 25, 100, 400)
# B-D12 (PROPOSAL): depth-limited evaluation = remaining x (1 - COMMISSION) - MULE_FEE.

# B-D13 (fitness agreed, penalty PROPOSAL): feature-selection penalty per selected feature.
FEATURE_PENALTY = 0.002
# B-D14 (PROPOSAL): local-search budgets.
EVAL_BUDGET = 200
SEEDS = (0, 1, 2)
BEAM_K = 4
GA_POP = 20
GA_TOURNAMENT = 3
GA_CX_RATE = 0.9
GA_MUT_RATE = 1 / 20
SA_T0 = 0.05
SA_ALPHA = 0.97
# B-D15 (PROPOSAL): threshold-tuning costs; 50 is the headline ratio.
C_FP = 1
C_FN_SWEEP = (10, 50, 100)
C_FN_HEADLINE = 50
THRESHOLD_STEP_SD = 0.05

# B-D16 (PROPOSAL): scam-message labels.
SCAM_TYPES = ("fake_kyc", "digital_arrest", "job_task", "loan_app", "fake_investment", "other_scam", "genuine")
# B-D17 (PROPOSAL): public supplement only below this many own messages.
MIN_OWN_MESSAGES = 300
# B-D18 (PROPOSAL): B's text model lives outside A's models/ folder.
TEXT_MODEL_PATH = "text/artifacts/scam_text_nb.joblib"


def params_table(names: list[str]) -> pd.DataFrame:
    """The listed constants and their values, for a report's Parameters section."""
    g = globals()
    return pd.DataFrame({"parameter": names, "value": [str(g[n]) for n in names]})
