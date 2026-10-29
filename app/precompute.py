"""python -m app.precompute: demo lists for the dashboard, written to build/b_app/ (gitignored)."""

import json
from pathlib import Path

from app.logic import demo_transactions, run_trail
from mulehunt.graph import TxGraph, load_risk_scores, placeholder_risk_scores
from mulehunt.trace import load_rings

OUT = Path("build/b_app")


def main(out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    g, kind = TxGraph.load()
    rings = load_rings(g.data_dir)
    risk = load_risk_scores() if kind == "real" else placeholder_risk_scores(g)
    demo = demo_transactions(g, rings)
    demo.to_csv(out / "demo_transactions.csv", index=False)
    trails = {}
    for row in demo[demo.ring_id.notna()].itertuples():
        r = run_trail(g, row.txn_id, "BFS", 48, risk)
        trails[row.txn_id] = {"found": r.found, "goal": r.goal, "hops": r.hops,
                              "trail": [e.txn_id for e in r.path]}
    (out / "demo_trails.json").write_text(json.dumps(trails, indent=1))
    print(f"wrote {out} ({kind} data, {len(demo)} demo transactions)")


if __name__ == "__main__":
    main()
