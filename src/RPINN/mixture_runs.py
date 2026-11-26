"""Uniform mixture model experiment of the RPINN paper (Computers 2024, 13, 176), Section 3.1.

Each repetition draws true mixing coefficients ``pi`` (five values in [0.1, 1] summing to one),
500 samples ``z_rp ~ U(p-1, p)`` and the target ``y = z @ pi``. White Gaussian noise is added to
the target at SNR -1, 3 and 5 dB (plus the noise-free case). RPINN trains on the first 400
samples (30% of them for validation) and is tested on the last 100; IPOPT solves the paper's
least-squares problem on the same 400 samples. Both are scored against the noise-free target.

RPINN (Table tab:arch-mix, 30 parameters): Dense_1 5 SELU without bias (25), Dense_2 1 SELU
without bias (5) whose kernel holds the l1-based max constraint (entries in [0, 1], l1 norm at
most 1). The paper reads the mixing coefficients from the output-layer weights (``x_rpinn``);
since Dense_1 sits before them, the coefficients the net actually applies to z are also saved
(``x_rpinn_effective``, mean input gradient over the test samples). Loss: lambda_H Huber +
lambda_L |x^T 1 - 1|; Adam lr 1e-3, 600 epochs, batch 32.

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
EPOCHS = 600
BATCH = 32
LR = 1e-3
SNRS = (-1, 3, 5, None)
REPS = 100
# lambda_H + lambda_L = 1 (paper Eq. mix-rpinn); the values are not stated in the paper.
LAMBDA_H, LAMBDA_L = 0.9, 0.1


def _rng(seed: int, *parts) -> np.random.Generator:
    return np.random.default_rng((seed * 1_000_003 + zlib.crc32("/".join(map(str, parts)).encode())) % (2**31 - 1))


def mixing_coefficients(rng: np.random.Generator, p: int = P) -> np.ndarray:
    """Values in [0.1, 1] rounded to two decimals that sum to exactly one (original generator)."""
    while True:
        x = np.round(rng.uniform(0.1, 1, p), 2)
        if np.isclose(x.sum(), 1.0, atol=1e-9):
            return x


def uniform_samples(rng: np.random.Generator, n: int = N_SAMPLES, p: int = P) -> np.ndarray:
    return np.column_stack([rng.uniform(k - 1, k, n) for k in range(1, p + 1)])


def add_noise(y: np.ndarray, snr: float | None, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise at ``snr`` dB relative to the mean signal power (original formula)."""
    if snr is None:
        return y.copy()
    noise_db = 10 * np.log10(abs(np.mean(y))) - snr
    return y + rng.normal(0.0, np.sqrt(10 ** (noise_db / 10)), y.shape)


def make_problem(rep: int, seed: int = 0):
    rng = _rng(seed, "problem", rep)
    pi = mixing_coefficients(rng)
    Z = uniform_samples(rng)
    return pi, Z, Z @ pi


def mape(true, pred) -> float:
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    return float(np.mean(np.abs((true - pred) / true)) * 100)


def mixture_model(seed: int = 0):
    import tensorflow as tf

    class L1MaxConstraint(tf.keras.constraints.Constraint):
        """Entries in [0, 1] and l1 norm at most one."""

        def __call__(self, w):
            w = tf.clip_by_value(w, 0.0, 1.0)
            l1 = tf.reduce_sum(w)
            return tf.where(l1 > 1.0, w / l1, w)

    def sum_to_one(w):
        return LAMBDA_L * tf.abs(tf.reduce_sum(w) - 1.0)

    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input(shape=(P,))
    h = tf.keras.layers.Dense(P, activation="selu", use_bias=False, name="Dense_1")(inp)
    out = tf.keras.layers.Dense(1, activation="selu", use_bias=False, name="Dense_2",
                                kernel_constraint=L1MaxConstraint(), kernel_regularizer=sum_to_one,
                                kernel_initializer=tf.keras.initializers.Constant(1.0 / P))(h)
    model = tf.keras.Model(inp, out)
    model.compile(loss=tf.keras.losses.Huber(), loss_weights=[LAMBDA_H],
                  optimizer=tf.keras.optimizers.Adam(LR))
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
    x = model.get_layer("Dense_2").get_weights()[0].ravel()
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
    pi, Z, y = make_problem(rep, seed)
    rec = {"pi": pi, "Z": Z, "y": y, "snr": np.array([np.nan if s is None else s for s in SNRS], float)}
    rows = {k: [] for k in ("y_noisy", "x_rpinn", "x_rpinn_effective", "x_ipopt", "pred_rpinn", "pred_ipopt",
                            "train_s_rpinn", "pred_s_rpinn", "solve_s_ipopt", "loss", "val_loss")}
    for snr in SNRS:
        yn = add_noise(y, snr, _rng(seed, "noise", rep, snr))
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
