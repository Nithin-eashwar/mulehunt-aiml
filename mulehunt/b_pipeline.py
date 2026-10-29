"""Person B's studies in one command (plan Task 12). Run from the repo root.

    python -m mulehunt.b_pipeline            trace -> adversarial -> local_search -> text -> app precompute
    python -m mulehunt.b_pipeline --small    the same on Person A's 5,000-account run, inside build/small/
    python -m mulehunt.b_pipeline --stages trace,adversarial

Each stage is the same CLI you can run on its own, run as its own Python process (like
Person A's pipeline), and writes its own report:
trace -> reports/b_02_search.md, adversarial -> b_03_adversarial.md,
local_search -> b_04_local_search.md, text -> b_07_scam_text.md.
--small runs every stage with build/small/ as its working directory, so it reads
build/small/data/... and writes build/small/reports/... only. Make that folder first with
`python -m mulehunt.pipeline --small`.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SMALL_ROOT = Path("build/small")
STAGE_ORDER = ("trace", "adversarial", "local_search", "text", "precompute")
MODULES = {"trace": "mulehunt.trace", "adversarial": "mulehunt.adversarial", "local_search": "mulehunt.local_search",
           "text": "text.classifier", "precompute": "app.precompute"}


def _run(cwd, module: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH", "")) if p)
    subprocess.run([sys.executable, "-m", module], check=True, env=env, cwd=cwd)


def _has_messages(root: Path) -> bool:
    d = root / "data/raw/scam_messages"
    return (d / "messages.csv").exists() or (d / "sample_messages.csv").exists()


def run_b_pipeline(small: bool = False, stages=None) -> dict[str, float]:
    chosen = list(STAGE_ORDER) if stages is None else list(stages)
    unknown = sorted(set(chosen) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; choose from {', '.join(STAGE_ORDER)}")
    cwd = None
    root = REPO
    if small:
        root = cwd = REPO / SMALL_ROOT
        if not (root / "data/processed/features.parquet").exists():
            raise SystemExit(f"{SMALL_ROOT}/ has no features; run `python -m mulehunt.pipeline --small` first")
        src = REPO / "data/raw/scam_messages"
        if not (root / "data/raw/scam_messages").exists():
            shutil.copytree(src, root / "data/raw/scam_messages")
        print(f"writing all outputs under {SMALL_ROOT}/", flush=True)
    ordered = [s for s in STAGE_ORDER if s in chosen]
    times = {}
    for i, stage in enumerate(ordered, 1):
        if stage == "text" and not _has_messages(root):
            print(f"[{i}/{len(ordered)}] text skipped: no scam-message dataset", flush=True)
            continue
        print(f"\n[{i}/{len(ordered)}] {stage} ...", flush=True)
        t0 = time.perf_counter()
        _run(cwd, MODULES[stage])
        times[stage] = time.perf_counter() - t0
        print(f"[{i}/{len(ordered)}] {stage} done in {times[stage]:.1f} s", flush=True)
    return times


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run Person B's studies end to end (from the repo root)")
    ap.add_argument("--small", action="store_true", help=f"run inside {SMALL_ROOT}/ (Person A's --small output)")
    ap.add_argument("--stages", default=",".join(STAGE_ORDER), help="comma-separated subset, run in order")
    a = ap.parse_args(argv)
    times = run_b_pipeline(a.small, [s.strip() for s in a.stages.split(",") if s.strip()])
    print("\nstage            seconds")
    for stage, seconds in times.items():
        print(f"{stage:16s} {seconds:7.1f}")


if __name__ == "__main__":
    main()
