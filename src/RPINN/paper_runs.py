"""Training entry point that reproduces the gas experiments of the RPINN paper (Computers 2024, 13, 176).

Per network and scenario: 20,000 seeded demand samples drawn as in the original training
notebook, 30% held out for test, 600 epochs, batch 256, validation split 0.2 and the learning
rate divided by ten at epochs 400, 500 and 580. Both model variants of the paper's data layout
are trained on the same samples: ``s=0`` (Modelo1, no unsupplied-gas switch) and ``s=1``
(Modelo2). The first 320 test samples of every network are the ones IPOPT solves.

Usage::

    python -m RPINN.paper_runs train --out DIR [--nets 8 9 ... Col] [--tips S M D] [--models 0 1]
    python -m RPINN.paper_runs collect --out DIR

``train`` writes ``DIR/{tip}/Modelo{1,2}/model{1,2}_{net}.h5`` with the input bound
``fd_{net}.npy`` and the record ``record_{net}.json`` (per-epoch losses and seconds), plus the
held-out samples ``DIR/{tip}/split/{net}_test.npy``. ``collect`` assembles
``DIR/{tip}/{MAPE,MAE}/inputs_P.mat`` (key ``inputs``, one slot per network) from those files.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf

from RPINN import model_real_data_ as real

NETS = ("8", "9", "10", "11", "12", "13", "14", "15", "Col")
TIPS = ("S", "M", "D")
N_SAMPLES = 20000
TEST_SIZE = 0.3
EPOCHS = 600
BATCH = 256
VAL_SPLIT = 0.2
LR_EPOCHS = (400, 500, 580)
IPOPT_SAMPLES = 320

# Offsets subtracted from the largest node demand (training notebook, one set per scenario).
OFFSETS = {
    "D": (-12.4, -13, -2.8, -1.7, 0, 4, 8, 10, 10),
    "M": (6.5, 8.7, 10.5, 11.9, 13.1, 14.1, 15, 15.8, 36),
    "S": (12.4, 13.7, 14.8, 15.7, 16.4, 17.2, 17.8, 18.3, 48),
}


def net_path(net: str) -> Path:
    name = "ng_caseCol_18.xlsx" if net == "Col" else f"ng_case{net}.xlsx"
    return real.DATA_DIR / "Redes" / name


def source_capacity(net: str) -> float:
    return float(pd.read_excel(net_path(net), sheet_name="well")["Imax"].sum())


def _seed(base: int, *parts: str) -> int:
    return (base * 1_000_003 + zlib.crc32("/".join(parts).encode())) % (2**31 - 1)


def scenario_demands(net: str, tip: str, n: int = N_SAMPLES, seed: int = 0) -> np.ndarray:
    """Uniform node demands in [min, max - offset], min and max over the network's demand rows."""
    dem = pd.read_excel(net_path(net), sheet_name="node.dem").values.sum(axis=1)
    lo, hi = float(dem.min()), float(dem.max()) - OFFSETS[tip][NETS.index(net)]
    rng = np.random.default_rng(_seed(seed, "demand", net, tip))
    return rng.uniform(lo, hi, (n, len(dem))).astype("float32")


def split_paper(X: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_seed(seed, "split"))
    idx = rng.permutation(len(X))
    cut = int(round(len(X) * (1 - TEST_SIZE)))
    return X[idx[:cut]], X[idx[cut:]]


class _EpochClock(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.seconds: list[float] = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.seconds.append(time.perf_counter() - self._t)


def run_one(net: str, tip: str, s: int, out: Path, n: int = N_SAMPLES, epochs: int = EPOCHS,
            seed: int = 0, verbose: int = 0) -> dict:
    """Train one model variant on one network and scenario and write its files under ``out``."""
    out = Path(out)
    X = scenario_demands(net, tip, n=n, seed=seed)
    X_train, X_test = split_paper(X, seed=_seed(seed, net, tip))
    split_dir = out / tip / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / f"{net}_test.npy", X_test)

    fd = X_train.max(axis=0)
    model = real.flow_model(str(net_path(net)), fd, s=s, seed=_seed(seed, "model", net, tip, str(s)))
    targets = [np.zeros((len(X_train),) + tuple(o.shape[1:]), "float32") for o in model.outputs]
    targets[0] = X_train  # node balance reproduces the demand; Weymouth and ratio losses ignore targets
    clock = _EpochClock()
    t0 = time.perf_counter()
    hist = model.fit(X_train, targets, epochs=epochs, batch_size=BATCH, validation_split=VAL_SPLIT,
                     verbose=verbose, callbacks=[real.LearningRateReducer(list(LR_EPOCHS)), clock])
    k = s + 1
    mdir = out / tip / f"Modelo{k}"
    mdir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(mdir / f"model{k}_{net}.h5"))
    np.save(mdir / f"fd_{net}.npy", fd)
    rec = {
        "net": net, "tip": tip, "s": s, "seed": seed, "n": n, "epochs": epochs,
        "train_seconds": time.perf_counter() - t0, "epoch_seconds": clock.seconds,
        "loss": {key: [float(v) for v in vals] for key, vals in hist.history.items()},
    }
    (mdir / f"record_{net}.json").write_text(json.dumps(rec))
    return rec


def collect(out: Path, tips=TIPS, nets=NETS) -> None:
    """Write ``{tip}/{MAPE,MAE}/inputs_P.mat``: the first 320 test samples of each network."""
    out = Path(out)
    for tip in tips:
        slots = np.empty((1, len(NETS)), dtype=object)
        for i, net in enumerate(NETS):
            f = out / tip / "split" / f"{net}_test.npy"
            slots[0, i] = np.load(f)[:IPOPT_SAMPLES].astype(float) if net in nets and f.exists() else np.zeros((0, 0))
        for metric in ("MAPE", "MAE"):
            d = out / tip / metric
            d.mkdir(parents=True, exist_ok=True)
            sio.savemat(d / "inputs_P.mat", {"inputs": slots})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m RPINN.paper_runs", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train", help="train the paper's gas models")
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--nets", nargs="+", default=list(NETS), choices=NETS)
    tr.add_argument("--tips", nargs="+", default=list(TIPS), choices=TIPS)
    tr.add_argument("--models", nargs="+", type=int, default=[0, 1], choices=[0, 1])
    tr.add_argument("--epochs", type=int, default=EPOCHS)
    tr.add_argument("--seed", type=int, default=0)
    co = sub.add_parser("collect", help="assemble inputs_P.mat from the held-out samples")
    co.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "train":
        for tip in a.tips:
            for net in a.nets:
                for s in a.models:
                    rec = run_one(net, tip, s, a.out, epochs=a.epochs, seed=a.seed)
                    print(json.dumps({"net": net, "tip": tip, "s": s, "seconds": round(rec["train_seconds"], 1),
                                      "final": {k: v[-1] for k, v in rec["loss"].items()}}), flush=True)
    else:
        collect(a.out)


if __name__ == "__main__":
    main()
