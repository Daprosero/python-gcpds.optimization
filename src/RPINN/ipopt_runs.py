"""IPOPT reference runs on exactly the test samples RPINN is evaluated on.

For every network and scenario, IPOPT (GEKKO, MPCC Weymouth, the paper's solver) solves the
first 320 held-out samples written by ``RPINN.paper_runs``. Each solve has a wall-time cap
(60 s by default, not stated in the paper). A sample that does not converge is kept with
NaN values and its failure reason; nothing is dropped.

Usage::

    python -m RPINN.ipopt_runs solve --out DIR [--nets ...] [--tips S M D] [--workers 8] [--max-time 60]
    python -m RPINN.ipopt_runs merge --out DIR [--nets ...] [--tips S M D]

writes ``DIR/{tip}/ipopt/{net}.npz`` with ``X`` (demands), ``F`` (flows in the RPINN layout:
wells, pipes, compressors, unsupplied gas per user class, storage), ``P`` (pressures),
``objective``, ``solve_seconds`` (IPOPT's own time), ``wall_seconds`` and ``status``
(``ok`` or the failure text). A run split into parts (``RPINN.kaggle_units``) writes
``{net}_part{k}.npz`` files that ``merge`` joins into ``{net}.npz``.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from RPINN.ipopt_gas import Natural_gas_flow
from RPINN.paper_runs import IPOPT_SAMPLES, NETS, TIPS, net_path

MAX_TIME = 60


def _network_with_demand(net: str, demand: np.ndarray) -> pd.ExcelFile:
    """The network workbook with node.dem replaced: residential demand, other classes zero."""
    base = pd.ExcelFile(net_path(net))
    sheets = {s: pd.read_excel(base, sheet_name=s) for s in base.sheet_names}
    zero = np.zeros_like(demand)
    sheets["node.dem"] = pd.DataFrame({"Res": demand, "Ind": zero, "NGV": zero,
                                       "Com": zero, "Pet": zero, "Ref": zero})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    return pd.ExcelFile(io.BytesIO(buf.getvalue()))


def solve_sample(net: str, demand, max_time: float = MAX_TIME) -> dict:
    demand = np.asarray(demand, float)
    n = len(demand)
    t0 = time.perf_counter()
    try:
        solve_s, values = Natural_gas_flow(_network_with_demand(net, demand), disp=False,
                                           weymouth_type="MPCC", initial=None,
                                           max_time=max_time).get_values()
        values = np.asarray(values, float)
        flows = np.concatenate((values[1:-n], np.zeros(2)))  # storage in/out, as the original notebook
        return {"status": "ok", "objective": float(values[0]), "F": flows, "P": values[-n:],
                "solve_seconds": float(solve_s), "wall_seconds": time.perf_counter() - t0}
    except Exception as e:  # GEKKO raises on any non-converged solve
        return {"status": str(e).strip()[:200] or type(e).__name__, "objective": np.nan, "F": None,
                "P": None, "solve_seconds": np.nan, "wall_seconds": time.perf_counter() - t0}


def _solve_args(args):
    return solve_sample(*args)


def run_network(out: Path, tip: str, net: str, workers: int = 1, max_time: float = MAX_TIME,
                limit: int = IPOPT_SAMPLES, start: int = 0, part: int | None = None) -> dict:
    """Solve held-out samples ``start:limit``; ``part`` writes ``{net}_part{part}.npz`` instead of ``{net}.npz``."""
    out = Path(out)
    X = np.load(out / tip / "split" / f"{net}_test.npy")[start:limit].astype(float)
    jobs = [(net, x, max_time) for x in X]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_solve_args, jobs))
    else:
        res = [_solve_args(j) for j in jobs]
    nf = next((len(r["F"]) for r in res if r["F"] is not None), 0)
    nan_f, nan_p = np.full(nf, np.nan), np.full(X.shape[1], np.nan)
    rec = {
        "X": X,
        "F": np.array([r["F"] if r["F"] is not None else nan_f for r in res]),
        "P": np.array([r["P"] if r["P"] is not None else nan_p for r in res]),
        "objective": np.array([r["objective"] for r in res]),
        "solve_seconds": np.array([r["solve_seconds"] for r in res]),
        "wall_seconds": np.array([r["wall_seconds"] for r in res]),
        "status": np.array([r["status"] for r in res]),
        "max_time": max_time,
        "start": start,
    }
    d = out / tip / "ipopt"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / (f"{net}.npz" if part is None else f"{net}_part{part}.npz"), **rec)
    return rec


def merge_parts(out: Path, tip: str, net: str) -> dict:
    """Join ``{net}_part*.npz`` (ordered by their first sample) into ``{net}.npz``."""
    d = Path(out) / tip / "ipopt"
    parts = sorted((dict(np.load(f)) for f in d.glob(f"{net}_part*.npz")), key=lambda r: int(r["start"]))
    if not parts:
        raise FileNotFoundError(f"no IPOPT parts for {tip}/{net} in {d}")
    nf = max(r["F"].shape[1] for r in parts)  # a part where nothing converged stores no flow columns
    for r in parts:
        if r["F"].shape[1] != nf:
            if r["F"].shape[1]:
                raise ValueError(f"{tip}/{net}: parts disagree on the flow layout")
            r["F"] = np.full((len(r["status"]), nf), np.nan)
    rec = {k: np.concatenate([r[k] for r in parts]) for k in ("X", "F", "P", "objective", "solve_seconds",
                                                              "wall_seconds", "status")}
    rec.update(max_time=parts[0]["max_time"], start=0)
    np.savez(d / f"{net}.npz", **rec)
    return rec


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m RPINN.ipopt_runs", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    so = sub.add_parser("solve", help="solve the held-out samples with IPOPT")
    so.add_argument("--out", type=Path, required=True)
    so.add_argument("--nets", nargs="+", default=list(NETS), choices=NETS)
    so.add_argument("--tips", nargs="+", default=list(TIPS), choices=TIPS)
    so.add_argument("--workers", type=int, default=1)
    so.add_argument("--max-time", type=float, default=MAX_TIME)
    me = sub.add_parser("merge", help="join per-part results into one file per network")
    me.add_argument("--out", type=Path, required=True)
    me.add_argument("--nets", nargs="+", default=list(NETS), choices=NETS)
    me.add_argument("--tips", nargs="+", default=list(TIPS), choices=TIPS)
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        for tip in a.tips:
            for net in a.nets:
                rec = merge_parts(a.out, tip, net)
                print(json.dumps({"tip": tip, "net": net, "samples": len(rec["status"]),
                                  "converged": int((rec["status"] == "ok").sum())}), flush=True)
        return
    for tip in a.tips:
        for net in a.nets:
            rec = run_network(a.out, tip, net, workers=a.workers, max_time=a.max_time)
            ok = rec["status"] == "ok"
            print(json.dumps({"tip": tip, "net": net, "converged": int(ok.sum()), "samples": len(ok),
                              "wall_seconds": round(float(rec["wall_seconds"].sum()), 1)}), flush=True)


if __name__ == "__main__":
    main()
