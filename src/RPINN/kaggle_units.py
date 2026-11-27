"""The whole V1 paper campaign cut into 110 independent units, one per remote session.

108 gas units: every network and scenario (9 x 3) split into four parts of 80 of the 320
held-out samples IPOPT solves. Every part rebuilds the same seeded demands and split
(``RPINN.paper_runs.held_out_split``), so parts never wait for one another. Part 0 also
trains both model variants (Modelo1 ``s=0`` and Modelo2 ``s=1``) and then solves its 80
samples on the same machine, which is what makes its RPINN and IPOPT times comparable.
Two mixture units run repetitions 0-49 and 50-99 on the 2024 data, one repetition per
process so a crash loses one repetition only.

Every unit writes ``machine.json`` (processor, cores, package versions) beside its results.

Usage::

    python -m RPINN.kaggle_units list
    python -m RPINN.kaggle_units run --unit K --out DIR [--workers 4] [--epochs 600] [--samples 80]

On a remote worker the job calls ``run_unit_isolated``, which runs the same command under
Python 3.11 with the pinned packages of the original runs (``PINS``).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PARTS = 4
MIXTURE_SPLITS = ((0, 50), (50, 100))

# The exact environment the VivoBook runs used. Remote images now ship Python 3.13, for which
# TensorFlow 2.15 has no wheel, so ``run_unit_isolated`` builds this one beside the image's own.
PYTHON = "3.11"
PINS = ("tensorflow-cpu==2.15.0", "keras==2.15.0", "gekko==1.0.6", "numpy==1.26.4", "pandas==3.0.6",
        "scipy==1.17.1", "scikit-learn==1.9.1", "h5py==3.16.0", "cvxpy==1.7.5", "openpyxl==3.1.5",
        "xlsxwriter==3.3.4", "matplotlib==3.11.2", "seaborn==0.13.2", "ipython==9.19.1")


def units() -> list[dict]:
    from RPINN.paper_runs import IPOPT_SAMPLES, NETS, TIPS
    size = IPOPT_SAMPLES // PARTS
    gas = [{"kind": "gas", "tip": tip, "net": net, "part": k, "start": k * size, "stop": (k + 1) * size}
           for tip in TIPS for net in NETS for k in range(PARTS)]
    mix = [{"kind": "mixture", "first": a, "stop": b} for a, b in MIXTURE_SPLITS]
    return [dict(u, unit=i) for i, u in enumerate(gas + mix)]


def machine() -> dict:
    cpu = platform.processor()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                   if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    from importlib import metadata
    versions = {}
    for dist in ("numpy", "tensorflow", "tensorflow-cpu", "gekko", "pandas"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    return {"cpu": cpu, "cores": os.cpu_count(), "python": platform.python_version(),
            "platform": platform.platform(), "versions": versions}


def run_gas(u: dict, out: Path, workers: int, epochs: int, samples: int | None) -> dict:
    from RPINN import ipopt_runs, paper_runs
    t0 = time.perf_counter()
    paper_runs.write_split(u["net"], u["tip"], out)
    trained = []
    if u["part"] == 0:
        for s in (0, 1):
            rec = paper_runs.run_one(u["net"], u["tip"], s, out, epochs=epochs)
            trained.append({"s": s, "train_seconds": rec["train_seconds"]})
    stop = u["stop"] if samples is None else min(u["stop"], u["start"] + samples)
    rec = ipopt_runs.run_network(out, u["tip"], u["net"], workers=workers, start=u["start"], limit=stop,
                                 part=u["part"])
    return {"trained": trained, "ipopt_samples": len(rec["status"]),
            "converged": int((rec["status"] == "ok").sum()), "seconds": time.perf_counter() - t0}


def run_mixture(u: dict, out: Path, workers: int, reps: int | None) -> dict:
    stop = u["stop"] if reps is None else min(u["stop"], u["first"] + reps)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="1",
               TF_NUM_INTEROP_THREADS="1", TF_CPP_MIN_LOG_LEVEL="3")
    src = str(Path(__file__).resolve().parents[1])  # a runner may put src on sys.path only, not in the env
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log = Path(out) / "Sinteticos" / f"mixture_{u['first']:03d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    def one(r: int) -> int:
        for _ in range(2):  # one retry: a crashed repetition is rerun once, never dropped silently
            if (Path(out) / "Sinteticos" / f"mixture_rep{r:03d}.npz").exists():
                return r
            with open(log, "a") as fh:
                subprocess.run([sys.executable, "-m", "RPINN.mixture_runs", "run", "--out", str(out),
                                "--first", str(r), "--reps", "1"], env=env, stdout=fh, stderr=subprocess.STDOUT)
        return r

    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, range(u["first"], stop)))
    done = [r for r in range(u["first"], stop) if (Path(out) / "Sinteticos" / f"mixture_rep{r:03d}.npz").exists()]
    return {"reps": len(done), "missing": sorted(set(range(u["first"], stop)) - set(done)),
            "seconds": time.perf_counter() - t0}


def run_unit(k: int, out: Path, workers: int = 4, epochs: int | None = None, samples: int | None = None) -> dict:
    from RPINN.paper_runs import EPOCHS
    u = units()[k]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    info = {"unit": u, "machine": machine()}
    if u["kind"] == "gas":
        info["result"] = run_gas(u, out, workers, EPOCHS if epochs is None else epochs, samples)
    else:
        info["result"] = run_mixture(u, out, workers, samples)
    (out / f"machine_unit{k:03d}.json").write_text(json.dumps(info, indent=1))
    return info


def unit_command(python: str, k: int, out: Path, workers: int, epochs: int | None, samples: int | None) -> list[str]:
    cmd = [python, "-m", "RPINN.kaggle_units", "run", "--unit", str(k), "--out", str(out), "--workers", str(workers)]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if samples is not None:
        cmd += ["--samples", str(samples)]
    return cmd


def run_unit_isolated(k: int, out: str = "out", workers: int = 4, epochs: int | None = None,
                      samples: int | None = None, env_dir: str = "/tmp/rpinn-py311") -> dict:
    """``run_unit`` under Python 3.11 with ``PINS``, built with uv; raises if the unit fails."""
    py, env = pinned_python(env_dir)
    subprocess.run(unit_command(py, k, Path(out).resolve(), workers, epochs, samples), env=env, check=True)
    return json.loads(Path(out, f"machine_unit{k:03d}.json").read_text())


def pinned_python(env_dir: str = "/tmp/rpinn-py311") -> tuple[str, dict]:
    """The Python 3.11 interpreter with ``PINS`` (built once with uv) and the environment to run ``src`` under it."""
    src = str(Path(__file__).resolve().parents[1])
    py = Path(env_dir) / "bin" / "python"
    if not py.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        subprocess.run([sys.executable, "-m", "uv", "venv", "--python", PYTHON, env_dir], check=True)
        subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python", str(py), *PINS], check=True)
    return str(py), dict(os.environ, PYTHONPATH=src, TF_CPP_MIN_LOG_LEVEL="3")


def run_campaign_isolated(out: str = "out", workers: int = 4, epochs: int | None = None,
                          samples: int | None = None) -> dict:
    """Run every unit this session was sent (``FORGE_RUN_UNITS``, a JSON list of unit ids).

    A unit that fails does not stop the others; ``campaign_summary.json`` lists both, and the
    session fails at the end when any unit failed, so a failure is never read as a result.
    """
    ids = [int(u) for u in json.loads(os.environ.get("FORGE_RUN_UNITS") or "[]")]
    if not ids:
        raise SystemExit("FORGE_RUN_UNITS names no unit: nothing to run")
    done, failed = [], {}
    for k in ids:
        try:
            run_unit_isolated(k, out, workers, epochs, samples)
            done.append(k)
        except Exception as e:  # recorded, then raised below
            failed[k] = f"{type(e).__name__}: {e}"[:300]
    summary = {"units": ids, "done": done, "failed": failed}
    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, "campaign_summary.json").write_text(json.dumps(summary, indent=1))
    if failed:
        raise RuntimeError(f"units failed: {failed}")
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m RPINN.kaggle_units", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="print every unit")
    r = sub.add_parser("run", help="run one unit")
    r.add_argument("--unit", type=int, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    r.add_argument("--epochs", type=int, default=None, help="training epochs (default: the paper's 600)")
    r.add_argument("--samples", type=int, default=None,
                   help="cap IPOPT samples (gas) or repetitions (mixture), for a smoke run")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for u in units():
            print(json.dumps(u))
        return
    print(json.dumps(run_unit(a.unit, a.out, a.workers, a.epochs, a.samples)["result"]), flush=True)


if __name__ == "__main__":
    main()
