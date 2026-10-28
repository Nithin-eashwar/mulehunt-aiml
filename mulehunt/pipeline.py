"""Person A's whole pipeline in one command (plan Task 14). Run from the repo root.

    python -m mulehunt.pipeline                       generate -> validate -> features -> models -> clustering
    python -m mulehunt.pipeline --small               5,000-account development run, written to build/small/
    python -m mulehunt.pipeline --stages models,clustering
    python -m mulehunt.pipeline --out-root build/try  any run, written to a separate folder

Each stage is the same CLI you can run on its own, run as its own Python process, and
each writes its own report: validate -> reports/a_01_simulator.md, models ->
a_05_classifiers.md, clustering -> a_06_clustering.md.

A default run writes the committed deliverables (data/generated/CHECKSUMS.txt,
models/, reports/). Any other run writes under --out-root instead: the stages run
there with their own copy of the inputs, so a development run never touches the
files Person B relies on. --small does this by default.

Separate processes free each stage's memory before the next starts, and make the
pipeline identical to running the CLIs by hand. When all stages shared one process on
a 16 GB laptop, clustering once took 44 min instead of about 1.5 min; no pair of
stages reproduced it, so the likely cause is memory held across all earlier stages.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from simulator.config import DEFAULT_PATH

STAGE_ORDER = ("generate", "validate", "features", "models", "clustering")
REPO = Path(__file__).resolve().parents[1]
SMALL_OUT_ROOT = Path("build/small")
# What the stages read, copied into an --out-root so they can run there unchanged.
INPUTS = ("data/raw", "data/generated/toy")


def _run(cwd, module: str, *args: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH", "")) if p)
    subprocess.run([sys.executable, "-m", module, *args], check=True, env=env, cwd=cwd)


def _generate(config_path, small, cwd=None):
    _run(cwd, "simulator.generate", "--config", str(config_path), *(["--small"] if small else []))


def _validate(config_path, small, cwd=None):
    _run(cwd, "simulator.validate", "--config", str(config_path))


def _features(config_path, small, cwd=None):
    _run(cwd, "mulehunt.features", "--config", str(config_path))


def _models(config_path, small, cwd=None):
    _run(cwd, "mulehunt.models", "--all")


def _clustering(config_path, small, cwd=None):
    _run(cwd, "mulehunt.clustering")


STAGES = {"generate": _generate, "validate": _validate, "features": _features, "models": _models,
          "clustering": _clustering}


def prepare_out_root(out_root, config_path=DEFAULT_PATH) -> Path:
    """Copy the stage inputs (raw sources, toy data, config) into out_root.
    Returns the config path to pass to the stages, relative to out_root."""
    out_root = Path(out_root)
    for rel in INPUTS:
        src, dst = REPO / rel, out_root / rel
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
    src_cfg = Path(config_path) if Path(config_path).exists() else REPO / config_path
    dst_cfg = out_root / DEFAULT_PATH
    dst_cfg.parent.mkdir(parents=True, exist_ok=True)
    if src_cfg.resolve() != dst_cfg.resolve():
        shutil.copy(src_cfg, dst_cfg)
    return DEFAULT_PATH


def run_pipeline(config_path=DEFAULT_PATH, small: bool = False, stages: list[str] | None = None,
                 out_root=None) -> dict[str, float]:
    """Run the chosen stages in pipeline order. Returns stage -> seconds.
    `small` writes to build/small unless `out_root` says otherwise."""
    chosen = list(STAGE_ORDER) if stages is None else list(stages)
    unknown = sorted(set(chosen) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; choose from {', '.join(STAGE_ORDER)}")
    if small and out_root is None:
        out_root = SMALL_OUT_ROOT
    cwd = None
    if out_root is not None:
        config_path = prepare_out_root(out_root, config_path)
        cwd = Path(out_root)
        print(f"writing all outputs under {cwd}/", flush=True)
    ordered = [s for s in STAGE_ORDER if s in chosen]
    times = {}
    for i, stage in enumerate(ordered, 1):
        print(f"\n[{i}/{len(ordered)}] {stage} ...", flush=True)
        t0 = time.perf_counter()
        STAGES[stage](config_path, small, cwd)
        times[stage] = time.perf_counter() - t0
        print(f"[{i}/{len(ordered)}] {stage} done in {times[stage]:.1f} s", flush=True)
    return times


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Person A's pipeline end to end (from the repo root)")
    parser.add_argument("--config", default=str(DEFAULT_PATH))
    parser.add_argument("--small", action="store_true",
                        help=f"5,000-account development dataset, written to {SMALL_OUT_ROOT}/ by default")
    parser.add_argument("--out-root", default=None,
                        help="write every output under this folder instead of the committed locations")
    parser.add_argument("--stages", default=",".join(STAGE_ORDER), help="comma-separated subset, run in order")
    args = parser.parse_args(argv)
    times = run_pipeline(args.config, args.small, [s.strip() for s in args.stages.split(",") if s.strip()],
                         args.out_root)
    width = max(len(s) for s in times)
    print("\nstage" + " " * (width - 4) + "seconds")
    for stage, seconds in times.items():
        print(f"{stage:<{width}}  {seconds:7.1f}")
    print(f"{'total':<{width}}  {sum(times.values()):7.1f}")


if __name__ == "__main__":
    main()
