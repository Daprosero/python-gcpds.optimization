"""Uniform mixture model experiment of the RPINN paper (Computers 2024, 13, 176), Section 3.1.

Data are the ones the original code generated in 2024, stored in ``RPINN/Data/Sintéticos``:
``PIC.npy`` (100 x 5 true mixing coefficients), ``U.npy`` (100 x 500 x 5 inputs) and
``Y.npy`` (5 noise levels x 100 x 500 targets, SNR -1, 1, 3, 5 dB and noise-free). Those
inputs do not follow the paper's text (z_rp ~ U(p-1, p) per column): within each repetition
samples 1-100 have all five inputs in [0, 1), samples 101-200 in [1, 2), and so on, so the
100 test samples (the last ones) lie in [4, 5). RPINN trains on the first 400 samples (30% of
them for validation); IPOPT solves the paper's least-squares problem on the same 400 samples.
Both are scored against the noise-free target.

RPINN follows the original code (``RPINN.model_synthetic_data_.Evaluate.create_custom_model``),
not the paper's Table tab:arch-mix: one Dense output neuron (ReLU, no bias, 5 parameters)
applied straight to z, its kernel projected after every step onto the simplex (negative
entries set to zero, then divided by their sum) and initialised at 1/5. Its weights are the
mixing coefficients (``x_rpinn``). The table's extra hidden layer (Dense_1, 25 parameters) is
commented out in that code; with it the output weights are no longer the coefficients. The
coefficients the net applies to z (``x_rpinn_effective``, mean input gradient over the test
samples) are kept as a check. Huber loss, Adamax lr 1e-3, 100 epochs, batch 16, as the
original code; noise levels -1, 1, 3, 5 dB and noise-free, as the original data.

Usage::

    python -m RPINN.mixture_runs run --out DIR [--reps 100] [--workers 8]

writes ``DIR/Sinteticos/mixture_rep{r:03d}.npz`` per repetition.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

P = 5
N_SAMPLES = 500
N_TRAIN = 400
VAL_SPLIT = 0.3
EPOCHS = 100
BATCH = 16
LR = 1e-3
SNRS = (-1, 1, 3, 5, None)
REPS = 100


def _data_dir() -> Path:
    from RPINN.paper_runs import real
    return Path(real.DATA_DIR) / "Sintéticos"


DATA_DIR = _data_dir()


def make_problem(rep: int):
    """The 2024 repetition ``rep``: true coefficients, inputs, noise-free target, targets per SNR."""
    pi = np.load(DATA_DIR / "PIC.npy")[rep].ravel()
    Z = np.load(DATA_DIR / "U.npy")[rep]
    noisy = np.load(DATA_DIR / "Y.npy")[:, rep, :, 0]
    return pi, Z, noisy[-1], noisy


def mape(true, pred) -> float:
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    return float(np.mean(np.abs((true - pred) / true)) * 100)


def mixture_model(seed: int = 0):
    import tensorflow as tf

    class SimplexConstraint(tf.keras.constraints.Constraint):
        """Original CustomConstraint: negative entries to zero, then divided by their sum."""

        def __call__(self, w):
            w = tf.nn.relu(w)
            return w / tf.reduce_sum(w)

    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input(shape=(P,))
    out = tf.keras.layers.Dense(1, activation="relu", use_bias=False, name="Dense_out",
                                kernel_constraint=SimplexConstraint(),
                                kernel_initializer=tf.keras.initializers.Constant(1.0 / P))(inp)
    model = tf.keras.Model(inp, out)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adamax(LR))
    return model


def fit_rpinn(Z, y_noisy, seed: int = 0, epochs: int = EPOCHS):
    model = mixture_model(seed)
    t0 = time.perf_counter()
    hist = model.fit(Z[:N_TRAIN], y_noisy[:N_TRAIN], epochs=epochs, batch_size=BATCH,
                     validation_split=VAL_SPLIT, verbose=0, shuffle=True)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(Z[N_TRAIN:], verbose=0).ravel()
    pred_s = time.perf_counter() - t0
    x = model.get_layer("Dense_out").get_weights()[0].ravel()
    return x, effective_coefficients(model, Z[N_TRAIN:]), pred, train_s, pred_s, {k: np.asarray(v) for k, v in hist.history.items()}


def effective_coefficients(model, Z) -> np.ndarray:
    """Mean gradient of the output with respect to each input: the coefficients the net applies to z."""
    import tensorflow as tf
    z = tf.convert_to_tensor(Z, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(z)
        out = model(z)
    return tape.gradient(out, z).numpy().mean(axis=0)


def solve_ipopt(Z, y):
    """Paper Eq. mix: min ||y - Z x||_2^2, 0 <= x <= 1, x^T 1 = 1 (GEKKO, IPOPT)."""
    from gekko import GEKKO
    m = GEKKO(remote=False)
    m.options.SOLVER = 3
    x = [m.Var(value=1.0 / Z.shape[1], lb=0, ub=1) for _ in range(Z.shape[1])]
    m.Equation(sum(x) == 1)
    # Same objective expanded: x^T (Z^T Z) x - 2 (Z^T y)^T x + y^T y (keeps GEKKO's equation short).
    Q, b, c = Z.T @ Z, Z.T @ y, float(y @ y)
    p = Z.shape[1]
    m.Minimize(sum(float(Q[i, j]) * x[i] * x[j] for i in range(p) for j in range(p))
               - 2 * sum(float(b[i]) * x[i] for i in range(p)) + c)
    t0 = time.perf_counter()
    m.solve(disp=False)
    return np.array([v.value[0] for v in x]), time.perf_counter() - t0


def run_rep(rep: int, out: Path, seed: int = 0, epochs: int = EPOCHS) -> dict:
    pi, Z, y, noisy = make_problem(rep)
    rec = {"pi": pi, "Z": Z, "y": y, "snr": np.array([np.nan if s is None else s for s in SNRS], float)}
    rows = {k: [] for k in ("y_noisy", "x_rpinn", "x_rpinn_effective", "x_ipopt", "pred_rpinn", "pred_ipopt",
                            "train_s_rpinn", "pred_s_rpinn", "solve_s_ipopt", "loss", "val_loss")}
    for j, snr in enumerate(SNRS):
        yn = noisy[j]
        xr, xe, pr, tr, ps, hist = fit_rpinn(Z, yn, seed=int(zlib.crc32(f"{seed}/{rep}/{snr}".encode()) % 2**31), epochs=epochs)
        xi, ti = solve_ipopt(Z[:N_TRAIN], yn[:N_TRAIN])
        for k, v in (("y_noisy", yn), ("x_rpinn", xr), ("x_rpinn_effective", xe), ("x_ipopt", xi), ("pred_rpinn", pr),
                     ("pred_ipopt", Z[N_TRAIN:] @ xi), ("train_s_rpinn", tr), ("pred_s_rpinn", ps),
                     ("solve_s_ipopt", ti), ("loss", hist["loss"]), ("val_loss", hist["val_loss"])):
            rows[k].append(v)
    rec.update({k: np.asarray(v) for k, v in rows.items()})
    d = Path(out) / "Sinteticos"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"mixture_rep{rep:03d}.npz", **rec)
    return rec


def _run_args(args):
    rec = run_rep(*args)
    return args[0], [mape(rec["y"][N_TRAIN:], p) for p in rec["pred_rpinn"]], [mape(rec["y"][N_TRAIN:], p) for p in rec["pred_ipopt"]]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m RPINN.mixture_runs", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the mixture repetitions")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--reps", type=int, default=REPS)
    r.add_argument("--first", type=int, default=0)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    jobs = [(k, a.out, a.seed) for k in range(a.first, a.first + a.reps)]
    with ProcessPoolExecutor(a.workers) as ex:
        for rep, mr, mi in ex.map(_run_args, jobs):
            print(json.dumps({"rep": rep, "mape_y_rpinn": [round(v, 2) for v in mr],
                              "mape_y_ipopt": [round(v, 2) for v in mi]}), flush=True)


if __name__ == "__main__":
    main()
