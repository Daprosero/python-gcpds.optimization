"""Figures 7-12 and the result tables of the RPINN paper, from the regenerated runs.

Reads ``RPINN/Data/v1-paper-run`` (``RPINN.kaggle_units`` output) and writes PDF/PNG figures
and CSV/LaTeX tables to ``RPINN/Results/v1-paper``. RPINN and IPOPT are scored with the same
formulas on the same samples:

- node balance: MAPE (%) of ``F @ w.T`` against the demand (truth), nodes with zero demand left out;
- Weymouth: MAPE (%) of each pipe flow against the flow its end pressures imply (truth);
- compression ratio: share (%) of compressors whose outlet/inlet pressure ratio leaves [1, beta_max];
- cost: ``F @ c``; the cost difference is RPINN minus IPOPT on the samples IPOPT solved.

IPOPT samples that did not converge are counted in the tables and left out of the error
statistics, never dropped silently.

Usage::

    python -m RPINN.paper_figures [--run DIR] [--out DIR] [--only fig7 fig8 ...]
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from RPINN.paper_runs import IPOPT_SAMPLES, NETS, TIPS, net_path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "RPINN" / "Data" / "v1-paper-run"
OUT = ROOT / "RPINN" / "Results" / "v1-paper"
SNR_LABELS = ("-1 dB", "1 dB", "3 dB", "5 dB", "noise-free")
COLORS = {"RPINN": "#2a9d8f", "IPOPT": "#264653", "true": "#e76f51"}


def mape(true, pred, axis=-1, tol=1e-9) -> np.ndarray:
    """Mean absolute percentage error (x100) with ``true`` as the reference; |true| <= tol is left out."""
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    keep = np.abs(true) > tol
    err = np.where(keep, np.abs(true - pred) / np.where(keep, np.abs(true), 1.0), np.nan)
    return np.nanmean(err, axis=axis) * 100


# --- networks ---------------------------------------------------------------------
@lru_cache(maxsize=None)
def network(net: str) -> dict:
    from RPINN.model_real_data_ import gen_w
    x = pd.ExcelFile(net_path(net))
    sh = {s: pd.read_excel(x, sheet_name=s) for s in ("node.info", "node.demcost", "well", "pipe", "comp", "sto")}
    inf, dc, W, T, C, sto = (sh[s] for s in ("node.info", "node.demcost", "well", "pipe", "comp", "sto"))
    cost = np.concatenate((W["Cg"].values, T["C_O"].values, C["costc"], dc["al_Res"].values, dc["al_Ind"].values,
                           dc["al_Com"].values, dc["al_NGV"].values, dc["al_Ref"].values, dc["al_Pet"].values,
                           sto["C_S+"].values - sto["C_V"].values, -1 * (sto["C_S-"] - sto["C_V"]).values,
                           sto["C_V"].values)).astype(float)
    return {"w": gen_w(inf, W, T, C, dc, sto), "cost": cost[:-1], "nodes": len(inf), "nW": len(W), "nT": len(T),
            "nC": len(C), "i": T["fnode"].values.astype(int) - 1, "j": T["tnode"].values.astype(int) - 1,
            "K": T["Kij"].values.astype(float), "ic": C["fnode"].values.astype(int) - 1,
            "jc": C["tnode"].values.astype(int) - 1, "beta": C["ratio"].values.astype(float),
            "W": W, "T": T, "C": C, "info": inf}


def gas_metrics(net: str, X, F, P) -> dict:
    n = network(net)
    X, F, P = (np.asarray(a, float) for a in (X, F, P))
    balance = mape(X, F @ n["w"].T)
    pi, pj = P[:, n["i"]], P[:, n["j"]]
    wey_flow = np.sign(pi**2 - pj**2) * np.sqrt(n["K"] * np.abs(pi**2 - pj**2))
    weymouth = mape(wey_flow, F[:, n["nW"]:n["nW"] + n["nT"]])
    ratio = np.round(P[:, n["jc"]] / P[:, n["ic"]], 3)
    outside = (ratio < 1.0) | (ratio > n["beta"])
    return {"balance": balance, "weymouth": weymouth, "ratio": outside.mean(axis=1) * 100,
            "cost": F @ n["cost"][:F.shape[1]]}


# --- runs -------------------------------------------------------------------------
def ipopt(run: Path, tip: str, net: str) -> dict:
    d = np.load(Path(run) / tip / "ipopt" / f"{net}.npz")
    return {k: d[k] for k in d.files}


@lru_cache(maxsize=None)
def rpinn(run: Path, tip: str, net: str, s: int = 1) -> dict:
    import tensorflow as tf
    from RPINN.model_real_data_ import flow_model
    k = s + 1
    X = np.load(Path(run) / tip / "split" / f"{net}_test.npy")[:IPOPT_SAMPLES].astype("float32")
    fd = np.load(Path(run) / tip / f"Modelo{k}" / f"fd_{net}.npy")
    m = flow_model(str(net_path(net)), fd, s=s)
    m.load_weights(str(Path(run) / tip / f"Modelo{k}" / f"model{k}_{net}.h5"))
    sub = tf.keras.Model(m.inputs, [m.get_layer("F").output, m.get_layer("P").output])
    F, P = sub.predict(X, verbose=0)
    return {"X": X, "F": F, "P": P}


def gas_frame(run: Path = RUN, tips=TIPS, nets=NETS, s: int = 1) -> pd.DataFrame:
    """One row per (scenario, network, method, sample)."""
    rows = []
    for tip in tips:
        for net in nets:
            ip, rp = ipopt(run, tip, net), rpinn(Path(run), tip, net, s)
            ok = ip["status"] == "ok"
            for method, X, F, P, mask in (("RPINN", rp["X"], rp["F"], rp["P"], np.ones(len(ok), bool)),
                                          ("IPOPT", ip["X"], ip["F"], ip["P"], ok)):
                m = gas_metrics(net, X[mask], F[mask], P[mask])
                df = pd.DataFrame(m)
                df["sample"] = np.flatnonzero(mask)
                df["tip"], df["net"], df["method"] = tip, net, method
                rows.append(df)
    return pd.concat(rows, ignore_index=True)


def gas_table(frame: pd.DataFrame, run: Path = RUN) -> pd.DataFrame:
    out = []
    for (tip, net), g in frame.groupby(["tip", "net"], sort=False):
        ok = ipopt(run, tip, net)["status"] == "ok"
        r, i = g[g.method == "RPINN"].set_index("sample"), g[g.method == "IPOPT"].set_index("sample")
        both = i.index
        diff = r.loc[both, "cost"] - i["cost"]
        row = {"scenario": tip, "network": net, "ipopt_converged": int(ok.sum()), "samples": len(ok)}
        for metric in ("balance", "weymouth", "ratio"):
            row[f"rpinn_{metric}_median"] = float(np.nanmedian(r[metric]))
            row[f"ipopt_{metric}_median"] = float(np.nanmedian(i[metric])) if len(i) else np.nan
        row["cost_diff_median"] = float(np.median(diff)) if len(diff) else np.nan
        row["cost_diff_rel_median_pct"] = float(np.median(diff / np.abs(i["cost"]) * 100)) if len(diff) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def mixture(run: Path = RUN) -> list[dict]:
    files = sorted((Path(run) / "Sinteticos").glob("mixture_rep*.npz"))
    return [{k: v for k, v in np.load(f).items()} for f in files]


def mixture_frame(reps: list[dict]) -> pd.DataFrame:
    rows = []
    for r, d in enumerate(reps):
        y = d["y"][400:]
        for j, snr in enumerate(SNR_LABELS):
            for method in ("RPINN", "IPOPT"):
                key = method.lower()
                rows.append({"rep": r, "snr": snr, "method": method,
                             "output_mape": float(mape(y, d[f"pred_{key}"][j])),
                             "weights_mape": float(mape(d["pi"], d[f"x_{key}"][j])),
                             "seconds": float(d["train_s_rpinn"][j] + d["pred_s_rpinn"][j]) if method == "RPINN"
                             else float(d["solve_s_ipopt"][j])})
    return pd.DataFrame(rows)


# --- figures ----------------------------------------------------------------------
def _save(fig, out: Path, name: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(str(p))
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def _paired_boxes(ax, groups: list[tuple[str, np.ndarray, np.ndarray]], log=False):
    pos, data, cols = [], [], []
    for k, (_, a, b) in enumerate(groups):
        for off, vals, c in ((-0.18, a, COLORS["RPINN"]), (0.18, b, COLORS["IPOPT"])):
            v = np.asarray(vals, float)
            v = v[np.isfinite(v)]
            pos.append(k + off)
            data.append(v if len(v) else [np.nan])
            cols.append(c)
    bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    if log:
        ax.set_yscale("symlog", linthresh=1e-3)
    ax.grid(axis="y", alpha=0.3)


def _legend(ax):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS["RPINN"], label="RPINN"), Patch(color=COLORS["IPOPT"], label="IPOPT")],
              loc="best", fontsize=8)


def fig7(reps, out: Path = OUT) -> list[str]:
    """Output prediction and weights for SNR -1 dB, 3 dB and noise-free, on the repetition whose RPINN
    weight error is the median one at that SNR (a representative case, not the best one)."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(10, 9))
    for row, j in enumerate((0, 2, 4)):
        err = [mape(d["pi"], d["x_rpinn"][j]) for d in reps]
        r = int(np.argsort(err)[len(err) // 2])
        d = reps[r]
        order = np.argsort(d["y"][400:])
        ax = axs[row, 0]
        ax.plot(d["y_noisy"][j][400:][order], ".", color="0.6", ms=3, label="noisy target")
        ax.plot(d["y"][400:][order], color=COLORS["true"], lw=2, label="noise-free target")
        ax.plot(d["pred_ipopt"][j][order], "--", color=COLORS["IPOPT"], label="IPOPT")
        ax.plot(d["pred_rpinn"][j][order], ":", color=COLORS["RPINN"], lw=2, label="RPINN")
        ax.set_ylabel(f"{SNR_LABELS[j]}\noutput")
        ax.set_title(f"Repetition {r}: test samples sorted by target", fontsize=9)
        ax = axs[row, 1]
        k = np.arange(len(d["pi"]))
        ax.bar(k - 0.27, d["pi"], 0.27, color=COLORS["true"], label=r"true $\pi$")
        ax.bar(k, d["x_ipopt"][j], 0.27, color=COLORS["IPOPT"], label="IPOPT")
        ax.bar(k + 0.27, d["x_rpinn"][j], 0.27, color=COLORS["RPINN"], label="RPINN")
        ax.set_xticks(k)
        ax.set_xticklabels([rf"$\pi_{i + 1}$" for i in k])
        ax.set_ylabel("weight")
    axs[0, 0].legend(fontsize=8)
    axs[0, 1].legend(fontsize=8)
    axs[-1, 0].set_xlabel("test sample")
    return _save(fig, out, "fig7_mixture_prediction_weights")


def fig8(mix: pd.DataFrame, out: Path = OUT) -> list[str]:
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in ((axs[0], "output_mape", "Output error"), (axs[1], "weights_mape", "Weights error")):
        groups = [(s, mix[(mix.snr == s) & (mix.method == "RPINN")][col].values,
                   mix[(mix.snr == s) & (mix.method == "IPOPT")][col].values) for s in SNR_LABELS]
        _paired_boxes(ax, groups)
        ax.set_title(title)
        ax.set_ylabel("MAPE (%)")
        ax.set_xlabel("SNR")
    _legend(axs[0])
    return _save(fig, out, "fig8_mixture_mape")


def fig9(run: Path = RUN, out: Path = OUT, tip: str = "S", net: str = "8") -> list[str]:
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    e = np.linspace(-3, 3, 400)
    huber = np.where(np.abs(e) <= 1, 0.5 * e**2, np.abs(e) - 0.5)
    axs[0].plot(e, huber, color=COLORS["RPINN"])
    axs[0].set_title("Node balance and Weymouth: Huber")
    axs[0].set_xlabel("residual")
    beta = 1.5
    ratio = np.linspace(0.5, 2.2, 400)
    mid, span = 0.5 * (1 + beta), beta - 1
    pen = np.where((ratio >= 1) & (ratio <= beta), 0.0, ((ratio - mid) / span) ** 2)
    axs[1].plot(ratio, pen, color=COLORS["RPINN"])
    axs[1].axvspan(1, beta, color="0.9")
    axs[1].set_title(r"Compression ratio penalty ($\beta_{max}=1.5$)")
    axs[1].set_xlabel(r"$p_j/p_i$")
    rec = json.loads((Path(run) / tip / "Modelo2" / f"record_{net}.json").read_text())["loss"]
    for key, label in (("Bal_loss", "node balance"), ("Wey_loss", "Weymouth"), ("Ratio_loss", "compression ratio")):
        axs[2].plot(np.arange(1, len(rec[key]) + 1), rec[key], label=label)
    axs[2].set_yscale("log")
    axs[2].set_xlabel("epoch")
    axs[2].set_title(f"Penalty evolution (network {net}, scenario {tip})")
    axs[2].legend(fontsize=8)
    return _save(fig, out, "fig9_gas_losses")


def fig10(frame: pd.DataFrame, out: Path = OUT, net: str = "8") -> list[str]:
    import matplotlib.pyplot as plt
    g = frame[frame.net == net]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col, title in ((axs[0, 0], "balance", "Node balance MAPE (%)"), (axs[0, 1], "weymouth", "Weymouth MAPE (%)"),
                           (axs[1, 0], "ratio", "Compressors outside the ratio limits (%)")):
        groups = [(t, g[(g.tip == t) & (g.method == "RPINN")][col].values,
                   g[(g.tip == t) & (g.method == "IPOPT")][col].values) for t in TIPS]
        _paired_boxes(ax, groups, log=col != "ratio")
        ax.set_title(title)
    ax = axs[1, 1]
    data = []
    for t in TIPS:
        r = g[(g.tip == t) & (g.method == "RPINN")].set_index("sample")
        i = g[(g.tip == t) & (g.method == "IPOPT")].set_index("sample")
        data.append((r.loc[i.index, "cost"] - i["cost"]).values)
    ax.boxplot(data, showfliers=False, patch_artist=True, boxprops={"facecolor": COLORS["RPINN"], "alpha": 0.75})
    ax.set_xticks(range(1, len(TIPS) + 1))
    ax.set_xticklabels(list(TIPS))
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_yscale("symlog", linthresh=1e4)
    ax.set_title("Cost difference RPINN - IPOPT (IPOPT-solved samples)")
    for a in axs.ravel():
        a.set_xlabel("scenario")
    _legend(axs[0, 0])
    return _save(fig, out, f"fig10_gas_compliance_net{net}")


def fig11(run: Path = RUN, out: Path = OUT, tip: str = "S", net: str = "Col") -> list[str]:
    import matplotlib.pyplot as plt
    n, rp, ip = network(net), rpinn(Path(run), tip, net), ipopt(run, tip, net)
    ok = ip["status"] == "ok"
    nW, nT, nC = n["nW"], n["nT"], n["nC"]
    panels = (("Injection (MMSCFD)", slice(0, nW), n["W"]["Imin"].values if "Imin" in n["W"] else np.zeros(nW), n["W"]["Imax"].values),
              ("Pipe flow (MMSCFD)", slice(nW, nW + nT), n["T"]["Fg_min"].values, n["T"]["Fg_max"].values),
              ("Compressor flow (MMSCFD)", slice(nW + nT, nW + nT + nC), np.zeros(nC), n["C"]["fmaxc"].values),
              ("Pressure (psia)", None, n["info"]["Pmin"].values, n["info"]["Pmax"].values))
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (title, sl, lo, hi) in zip(axs.ravel(), panels):
        R = rp["P"] if sl is None else rp["F"][:, sl]
        I = ip["P"][ok] if sl is None else ip["F"][ok][:, sl]
        groups = [(str(k + 1), R[:, k], I[:, k]) for k in range(R.shape[1])]
        _paired_boxes(ax, groups)
        ax.plot(range(len(hi)), hi, "k*", ms=9)
        ax.plot(range(len(lo)), lo, "k*", ms=9)
        ax.set_title(title)
        ax.set_xlabel("element" if sl is not None else "node")
    _legend(axs[0, 0])
    fig.suptitle(f"Network {net}, scenario {tip}: stars are the limits", fontsize=10)
    return _save(fig, out, f"fig11_gas_bounds_{net}_{tip}")


def timing_frame(run: Path = RUN) -> pd.DataFrame:
    """Training and IPOPT times measured in the same session (the part-0 unit of each network and scenario)."""
    rows = []
    for f in sorted(Path(run).glob("machine_unit*.json")):
        u = json.loads(f.read_text())
        if u["unit"]["kind"] != "gas" or u["unit"]["part"] != 0:
            continue
        tip, net = u["unit"]["tip"], u["unit"]["net"]
        ip = ipopt(run, tip, net)
        sec = ip["wall_seconds"][:80]
        timing = Path(run) / "timing" / f"{tip}_{net}.json"
        pred = json.loads(timing.read_text())["predict_seconds"] if timing.exists() else None
        for t in u["result"]["trained"]:
            rows.append({"tip": tip, "net": net, "s": t["s"], "cpu": u["machine"]["cpu"],
                         "rpinn_train_seconds": t["train_seconds"], "ipopt_seconds_total_80": float(np.sum(sec)),
                         "ipopt_seconds_median": float(np.median(sec)),
                         "rpinn_predict_seconds_median": float(np.median(pred)) if pred else np.nan})
    return pd.DataFrame(rows)


def fig12(times: pd.DataFrame, run: Path = RUN, out: Path = OUT, tip: str = "M") -> list[str]:
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    t = times[(times.tip == tip) & (times.s == 1)].set_index("net").loc[list(NETS)]
    x = np.arange(len(NETS))
    axs[0].bar(x - 0.2, t["rpinn_train_seconds"], 0.4, color=COLORS["RPINN"], label="RPINN training (one model)")
    axs[0].bar(x + 0.2, t["ipopt_seconds_total_80"], 0.4, color=COLORS["IPOPT"], label="IPOPT, 80 samples")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("seconds")
    axs[0].set_title(f"Training vs solving (scenario {tip}, same session)")
    groups = []
    for net in NETS:
        timing = Path(run) / "timing" / f"{tip}_{net}.json"
        if timing.exists():  # both measured in one session (RPINN.timing_runs)
            rec = json.loads(timing.read_text())
            pred, ip = rec["predict_seconds"], rec["ipopt_wall_seconds"]
        else:
            pred, ip = [np.nan], ipopt(run, tip, net)["wall_seconds"][:80]
        groups.append((net, pred, ip))
    _paired_boxes(axs[1], groups)
    axs[1].set_yscale("log")
    axs[1].set_title("Per-sample time: RPINN prediction vs IPOPT solve")
    for a in axs:
        a.set_xticks(x)
        a.set_xticklabels(NETS)
        a.set_xlabel("network")
    axs[0].legend(fontsize=8)
    return _save(fig, out, f"fig12_times_{tip}")


# --- entry point ------------------------------------------------------------------
def build(run: Path = RUN, out: Path = OUT, only=None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    run, out = Path(run), Path(out)
    want = set(only or ("fig7", "fig8", "fig9", "fig10", "fig11", "fig12", "tables"))
    done = {}
    if want & {"fig7", "fig8", "tables"}:
        reps = mixture(run)
        mix = mixture_frame(reps)
        if "fig7" in want:
            done["fig7"] = fig7(reps, out)
        if "fig8" in want:
            done["fig8"] = fig8(mix, out)
        if "tables" in want:
            out.mkdir(parents=True, exist_ok=True)
            summ = mix.groupby(["snr", "method"], sort=False)[["output_mape", "weights_mape", "seconds"]].median()
            summ.to_csv(out / "table_mixture.csv")
            summ.to_latex(out / "table_mixture.tex", float_format="%.3f")
            done["table_mixture"] = str(out / "table_mixture.csv")
    if "fig9" in want:
        done["fig9"] = fig9(run, out)
    if want & {"fig10", "tables"}:
        cached = out / "gas_per_sample.csv"  # rebuilding 27 models is slow; delete the file to recompute
        frame = pd.read_csv(cached, dtype={"net": str}) if cached.exists() else gas_frame(run)
        if "fig10" in want:
            done["fig10"] = fig10(frame, out)
        if "tables" in want:
            frame.to_csv(out / "gas_per_sample.csv", index=False)
            tab = gas_table(frame, run)
            tab.to_csv(out / "table_gas.csv", index=False)
            tab.to_latex(out / "table_gas.tex", index=False, float_format="%.3g")
            done["table_gas"] = str(out / "table_gas.csv")
    if "fig11" in want:
        done["fig11"] = fig11(run, out)
    if want & {"fig12", "tables"}:
        times = timing_frame(run)
        if "fig12" in want:
            done["fig12"] = fig12(times, run, out)
        if "tables" in want:
            times.to_csv(out / "table_times.csv", index=False)
            done["table_times"] = str(out / "table_times.csv")
    return done


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m RPINN.paper_figures", description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.run, a.out, a.only), indent=1))


if __name__ == "__main__":
    main()
