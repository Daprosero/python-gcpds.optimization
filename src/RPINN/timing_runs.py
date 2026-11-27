"""Per-sample times of RPINN prediction and IPOPT solving, measured in one session (paper Fig. 12).

The campaign measured RPINN training and IPOPT solves on the same machine but not RPINN
prediction. One timing unit per scenario and network loads the campaign's trained models
(``RPINN/Data/v1-paper-run``), predicts each of the 320 held-out samples one at a time (batch
1, after one warm-up call), and then solves the first ``IPOPT_TIMED`` of those samples with
IPOPT on the same machine, with the campaign's settings (60 s cap, 4 workers). It writes
``OUT/timing/{tip}_{net}.json``: ``predict_seconds`` (Modelo2, ``s=1``), ``predict_seconds_s0``
(Modelo1), ``ipopt_wall_seconds``, ``ipopt_status`` and ``machine``.

Usage::

    python -m RPINN.timing_runs list
    python -m RPINN.timing_runs run --unit K --out DIR [--ipopt 20] [--workers 4]

On a remote worker the job calls ``run_timing_isolated``, which runs the same command under
the pinned Python 3.11 environment of the campaign.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

IPOPT_TIMED = 20
WORKERS = 4


def units() -> list[dict]:
    from RPINN.paper_runs import NETS, TIPS
    return [{"unit": i, "tip": tip, "net": net} for i, (tip, net) in enumerate((t, n) for t in TIPS for n in NETS)]


def predict_seconds(run: Path, tip: str, net: str, s: int) -> list[float]:
    """Wall time of one batch-1 prediction per held-out sample, after one warm-up call."""
    import tensorflow as tf
    from RPINN.model_real_data_ import flow_model
    from RPINN.paper_runs import IPOPT_SAMPLES, net_path
    k = s + 1
    X = np.load(Path(run) / tip / "split" / f"{net}_test.npy")[:IPOPT_SAMPLES].astype("float32")
    fd = np.load(Path(run) / tip / f"Modelo{k}" / f"fd_{net}.npy")
    m = flow_model(str(net_path(net)), fd, s=s)
    m.load_weights(str(Path(run) / tip / f"Modelo{k}" / f"model{k}_{net}.h5"))
    sub = tf.keras.Model(m.inputs, [m.get_layer("F").output, m.get_layer("P").output])
    sub.predict_on_batch(X[:1])
    out = []
    for x in X:
        t0 = time.perf_counter()
        sub.predict_on_batch(x[None])
        out.append(time.perf_counter() - t0)
    return out


def ipopt_seconds(run: Path, tip: str, net: str, n: int, workers: int) -> tuple[list[float], list[str]]:
    from RPINN.ipopt_runs import _solve_args, MAX_TIME
    X = np.load(Path(run) / tip / "split" / f"{net}_test.npy")[:n].astype(float)
    jobs = [(net, x, MAX_TIME) for x in X]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_solve_args, jobs))
    return [r["wall_seconds"] for r in res], [r["status"] for r in res]


def run_timing(k: int, out: Path, run: Path | None = None, n_ipopt: int = IPOPT_TIMED,
               workers: int = WORKERS) -> dict:
    from RPINN.kaggle_units import machine
    from RPINN.paper_figures import RUN
    u = units()[k]
    run = RUN if run is None else Path(run)
    t0 = time.perf_counter()
    rec = {"unit": u, "machine": machine(),
           "predict_seconds": predict_seconds(run, u["tip"], u["net"], 1),
           "predict_seconds_s0": predict_seconds(run, u["tip"], u["net"], 0)}
    rec["ipopt_wall_seconds"], rec["ipopt_status"] = ipopt_seconds(run, u["tip"], u["net"], n_ipopt, workers)
    rec["ipopt_workers"], rec["seconds"] = workers, time.perf_counter() - t0
    d = Path(out) / "timing"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{u['tip']}_{u['net']}.json").write_text(json.dumps(rec, indent=1))
    return rec


def timing_command(python: str, k: int, out: Path, n_ipopt: int, workers: int) -> list[str]:
    return [python, "-m", "RPINN.timing_runs", "run", "--unit", str(k), "--out", str(out),
            "--ipopt", str(n_ipopt), "--workers", str(workers)]


def run_timing_isolated(k: int, out: str = "out", n_ipopt: int = IPOPT_TIMED, workers: int = WORKERS,
                        env_dir: str = "/tmp/rpinn-py311") -> dict:
    """``run_timing`` under the campaign's pinned Python 3.11 environment; raises if it fails."""
    from RPINN.kaggle_units import pinned_python
    py, env = pinned_python(env_dir)
    subprocess.run(timing_command(py, k, Path(out).resolve(), n_ipopt, workers), env=env, check=True)
    u = units()[k]
    return json.loads(Path(out, "timing", f"{u['tip']}_{u['net']}.json").read_text())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m RPINN.timing_runs", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="print every timing unit")
    r = sub.add_parser("run", help="run one timing unit")
    r.add_argument("--unit", type=int, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--run", type=Path, default=None, help="campaign folder with the models (default: the repo's)")
    r.add_argument("--ipopt", type=int, default=IPOPT_TIMED, help="held-out samples IPOPT solves")
    r.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for u in units():
            print(json.dumps(u))
        return
    rec = run_timing(a.unit, a.out, a.run, a.ipopt, a.workers)
    print(json.dumps({"unit": rec["unit"], "predict_median": float(np.median(rec["predict_seconds"])),
                      "ipopt_median": float(np.median(rec["ipopt_wall_seconds"])),
                      "seconds": round(rec["seconds"], 1)}), flush=True)


if __name__ == "__main__":
    main()
