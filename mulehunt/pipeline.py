"""Person A's whole pipeline in one command (plan Task 14).

    python -m mulehunt.pipeline                       generate -> validate -> features -> models -> clustering
    python -m mulehunt.pipeline --small               5,000-account development run (overwrites data/generated)
    python -m mulehunt.pipeline --stages models,clustering

Each stage is the same CLI you can run on its own, run as its own Python process, and
each writes its own report: validate -> reports/a_01_simulator.md, models ->
a_05_classifiers.md, clustering -> a_06_clustering.md.

Separate processes free each stage's memory before the next starts, and make the
pipeline identical to running the CLIs by hand. When all stages shared one process on
a 16 GB laptop, clustering once took 44 min instead of about 1.5 min; no pair of
stages reproduced it, so the likely cause is memory held across all earlier stages.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from simulator.config import DEFAULT_PATH

STAGE_ORDER = ("generate", "validate", "features", "models", "clustering")
REPO = Path(__file__).resolve().parents[1]


def _run(module: str, *args: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH", "")) if p)
    subprocess.run([sys.executable, "-m", module, *args], check=True, env=env)


def _generate(config_path, small):
    _run("simulator.generate", "--config", str(config_path), *(["--small"] if small else []))


def _validate(config_path, small):
    _run("simulator.validate", "--config", str(config_path))


def _features(config_path, small):
    _run("mulehunt.features", "--config", str(config_path))


def _models(config_path, small):
    _run("mulehunt.models", "--all")


def _clustering(config_path, small):
    _run("mulehunt.clustering")


STAGES = {"generate": _generate, "validate": _validate, "features": _features, "models": _models,
          "clustering": _clustering}


def run_pipeline(config_path=DEFAULT_PATH, small: bool = False, stages: list[str] | None = None) -> dict[str, float]:
    """Run the chosen stages in pipeline order. Returns stage -> seconds."""
    chosen = list(STAGE_ORDER) if stages is None else list(stages)
    unknown = sorted(set(chosen) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; choose from {', '.join(STAGE_ORDER)}")
    ordered = [s for s in STAGE_ORDER if s in chosen]
    times = {}
    for i, stage in enumerate(ordered, 1):
        print(f"\n[{i}/{len(ordered)}] {stage} ...", flush=True)
        t0 = time.perf_counter()
        STAGES[stage](config_path, small)
        times[stage] = time.perf_counter() - t0
        print(f"[{i}/{len(ordered)}] {stage} done in {times[stage]:.1f} s", flush=True)
    return times


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Person A's pipeline end to end")
    parser.add_argument("--config", default=str(DEFAULT_PATH))
    parser.add_argument("--small", action="store_true",
                        help="5,000-account development dataset (overwrites data/generated)")
    parser.add_argument("--stages", default=",".join(STAGE_ORDER), help="comma-separated subset, run in order")
    args = parser.parse_args(argv)
    times = run_pipeline(args.config, args.small, [s.strip() for s in args.stages.split(",") if s.strip()])
    width = max(len(s) for s in times)
    print("\nstage" + " " * (width - 4) + "seconds")
    for stage, seconds in times.items():
        print(f"{stage:<{width}}  {seconds:7.1f}")
    print(f"{'total':<{width}}  {sum(times.values()):7.1f}")


if __name__ == "__main__":
    main()
