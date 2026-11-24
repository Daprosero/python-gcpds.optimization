"""Invariants of the CL-RPINN gas model: every flow bounded and every node balanced by construction."""
import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

tf = pytest.importorskip("tensorflow")

from RPINN import model_real_data_ as v1  # noqa: E402
from CLRPINN import model as v2  # noqa: E402
from keras import Model  # noqa: E402

REDES = v2.DATA_DIR / "Redes"
NETS = {"8": (REDES / "ng_case8.xlsx", 0), "18": (REDES / "ng_caseCol_18.xlsx", 8)}


def _real_demand(net, scenario="M"):
    path, k = NETS[net]
    z = sio.loadmat(v2.DATA_DIR / scenario / "MAPE" / "inputs_P.mat")["inputs"][0][k]
    return str(path), np.asarray(z, dtype="float32")


def _flows(path, z, sd, seed=0, classes=("Res",)):
    sh = v2.read_network(path)
    A, lo, hi = v2.augmented_network(sh, classes)
    raw = np.random.default_rng(seed).normal(0.0, sd, size=(len(z), A.shape[1])).astype("float32")
    f = v2.BalancedFlows(A, lo, hi)([tf.constant(raw), tf.constant(z)]).numpy().astype("float64")
    hi_s = np.concatenate([np.broadcast_to(hi[~np.isnan(hi)], (len(z), int((~np.isnan(hi)).sum()))), z], axis=1)
    return f, A, lo, hi_s


@pytest.mark.parametrize("net", ["8", "18"])
@pytest.mark.parametrize("sd", [0.5, 15.0])
def test_real_samples_are_bounded_and_balanced(net, sd):
    path, z = _real_demand(net)
    f, A, lo, hi = _flows(path, z, sd)
    scale = z.sum(1).max()
    assert (f >= lo - 1e-4).all() and (f <= hi + 1e-4).all()
    b = np.hstack([-z.sum(1, keepdims=True), z])
    assert np.abs(f @ A.T - b).max() / scale < 1e-4


def test_several_sources_share_the_supply():
    # Three nodes in a line, a source at each end: 1 -- 2 -- 3, demand at node 2.
    sheets = {
        "inf": pd.DataFrame({"node_id": [1, 2, 3]}),
        "well": pd.DataFrame({"node": [1, 3], "Imin": [5.0, 0.0], "Imax": [30.0, 20.0]}),
        "pipe": pd.DataFrame({"fnode": [1, 2], "tnode": [2, 3], "Fg_min": [-40.0, -40.0], "Fg_max": [40.0, 40.0]}),
        "comp": pd.DataFrame({"fnode": [], "tnode": [], "fmaxc": []}),
        "dc": pd.DataFrame({f"al_{k}": [1.0] * 3 for k in v2.DEMAND_CLASSES}),
        "sto": pd.DataFrame({"node": []}),
    }
    A, lo, hi = v2.augmented_network(sheets, ("Res",))
    z = np.array([[0.0, 35.0, 0.0], [0.0, 60.0, 0.0]], dtype="float32")
    raw = np.random.default_rng(1).normal(0, 2, size=(2, A.shape[1])).astype("float32")
    f = v2.BalancedFlows(A, lo, hi)([tf.constant(raw), tf.constant(z)]).numpy()
    s, u = f[:, :2], f[:, 4:]
    assert (s >= [5.0, 0.0]).all() and (s <= np.array([30.0, 20.0]) + 1e-4).all()
    np.testing.assert_allclose(s.sum(1) + u.sum(1), z.sum(1), rtol=1e-5, atol=1e-4)
    # The layer guarantees balance and bounds only; how much to ration is the cost loss's job.
    assert u[1].sum() >= 10.0 - 1e-3  # 60 demanded, 50 available: at least the deficit
    assert (s[0] > 0.5).all()  # both sources supply when one alone would do


def test_gradient_reaches_the_raw_outputs():
    path, z = _real_demand("8")
    z = tf.constant(z[:64])
    sh = v2.read_network(path)
    A, lo, hi = v2.augmented_network(sh, ("Res",))
    raw = tf.Variable(np.random.default_rng(2).normal(0, 1, size=(64, A.shape[1])).astype("float32"))
    with tf.GradientTape() as tape:
        f = v2.BalancedFlows(A, lo, hi)([raw, z])
        loss = tf.reduce_sum(f * tf.constant(v2.cost_vector(sh)[:A.shape[1]]))
    g = tape.gradient(loss, raw).numpy()
    assert np.isfinite(g).all() and np.abs(g).max() > 0.0


def _ipopt(net, scenario="M"):
    path, k = NETS[net]
    load = lambda name: np.asarray(sio.loadmat(v2.DATA_DIR / scenario / "MAPE" / name)["inputs"][0][k], float)
    return str(path), load("inputs_P.mat"), load("Flujos_P.mat")


@pytest.mark.parametrize("net", ["8", "18"])
def test_pipe_flows_reproduce_ipopt_given_its_injections(net):
    path, z, F = _ipopt(net)
    sh = v2.read_network(path)
    w = v1.gen_w(sh["inf"], sh["well"], sh["pipe"], sh["comp"], sh["dc"], sh["sto"])
    nw, npi = len(sh["well"]), len(sh["pipe"])
    Bq = w[:, nw:nw + npi]
    rhs = z - (F @ w.T - F[:, nw:nw + npi] @ Bq.T)
    ps = v2.pipe_structure(sh)
    f = v2.PipeFlows(ps["pinv"], ps["loops"], ps["k"])(tf.constant(rhs.astype("float32"))).numpy()
    assert np.abs(f - F[:, nw:nw + npi]).max() < 1e-3


def _predict(net, n=256, scenario="M", gain=1.0):
    path, z = _real_demand(net, scenario)
    z = z[:n]
    model = v2.flow_model_v2(path)
    if gain != 1.0:  # saturate every raw output the closures receive
        for name in ("raw_sources", "raw_comps", "raw_levels"):
            layer = model.get_layer(name)
            layer.set_weights([gain * v for v in layer.get_weights()])
        bn = model.get_layer("raw_unsupplied")
        gamma, beta, *stats = bn.get_weights()
        bn.set_weights([gain * gamma, gain * beta, *stats])
    return path, z, model, model.predict(z, verbose=False, batch_size=512)


def _assert_physically_exact(path, z, F, P):
    sh = v2.read_network(path)
    F, P = F.astype("float64"), P.astype("float64")
    scale = z.sum(1).max()
    assert v2.node_balance_residual(path, F, z).max() / scale < 1e-4
    nw, npi, nc, N = len(sh["well"]), len(sh["pipe"]), len(sh["comp"]), len(sh["inf"])
    s, f, c, u = F[:, :nw], F[:, nw:nw + npi], F[:, nw + npi:nw + npi + nc], F[:, nw + npi + nc:][:, :N]
    assert (s >= sh["well"]["Imin"].values - 1e-3).all() and (s <= sh["well"]["Imax"].values + 1e-3).all()
    assert (c >= -1e-3).all() and (c <= sh["comp"]["fmaxc"].values + 1e-3).all()
    assert (u >= -1e-3).all() and (u <= z + 1e-3).all()
    assert (f >= sh["pipe"]["Fg_min"].values - 1e-3).all() and (f <= sh["pipe"]["Fg_max"].values + 1e-3).all()
    i, j = sh["pipe"]["fnode"].values - 1, sh["pipe"]["tnode"].values - 1
    wey = (P[:, i] ** 2 - P[:, j] ** 2) - f * np.abs(f) / sh["pipe"]["Kij"].values
    assert np.abs(wey).max() / sh["inf"]["Pmax"].max() ** 2 < 1e-5
    assert (P >= sh["inf"]["Pmin"].values - 1e-3).all() and (P <= sh["inf"]["Pmax"].values + 1e-3).all()
    ci, cj = sh["comp"]["fnode"].values - 1, sh["comp"]["tnode"].values - 1
    ratio = P[:, cj] / P[:, ci]
    assert (ratio >= 1 - 1e-5).all() and (ratio <= sh["comp"]["ratio"].values + 1e-5).all()


@pytest.mark.parametrize("gain", [1.0, 30.0, 100.0])
@pytest.mark.parametrize("scenario", ["S", "M", "D"])
@pytest.mark.parametrize("net", ["8", "18"])
def test_model_state_is_physically_exact(net, scenario, gain):
    path, z, _, out = _predict(net, n=None, scenario=scenario, gain=gain)
    _assert_physically_exact(path, z, out["F"], out["P"])


def test_capacity_scale_rations_when_pipes_bind():
    path, z, model, _ = _predict("18", n=None)
    theta = Model(model.input, model.get_layer("capacity").output[2]).predict(z, verbose=False, batch_size=512)
    assert ((theta >= 0.0) & (theta <= 1.0)).all()
    assert (theta < 1.0 - 1e-6).any()  # network 18: pipes bind before supply runs out


@pytest.mark.parametrize("net", ["8", "18"])
def test_model_trains_on_cost_alone(net):
    path, z, model, _ = _predict(net)
    history = model.fit(z, {"Cost": np.zeros((len(z), 1), "float32")}, batch_size=32, epochs=1, verbose=0)
    assert np.isfinite(history.history["loss"]).all()


def test_gradients_cross_every_closure():
    path, z, model, _ = _predict("8", n=64)
    x = tf.constant(z)
    with tf.GradientTape(persistent=True) as tape:
        out = model(x, training=False)
        cost, pressure = tf.reduce_sum(out["Cost"]), tf.reduce_sum(out["P"])
    for layer, target in (("raw_unsupplied", cost), ("raw_comps", cost), ("raw_levels", pressure)):
        grads = [g for g in tape.gradient(target, model.get_layer(layer).trainable_variables) if g is not None]
        assert grads and all(np.isfinite(g.numpy()).all() for g in grads)
        assert max(np.abs(g.numpy()).max() for g in grads) > 0.0


def test_zone_fed_by_two_compressors_is_refused():
    sheets = v2.read_network(str(NETS["8"][0]))
    comp = sheets["comp"].copy()
    comp.loc[len(comp)] = comp.iloc[1]          # a second compressor into the same zone
    sheets["comp"] = comp
    with pytest.raises(ValueError, match="more than one compressor"):
        v2.zone_structure(sheets)


def test_two_classes_fill_their_own_user_blocks():
    path, z = _real_demand("8")
    x = np.concatenate([z[:64], 0.3 * z[64:128]], axis=1)
    model = v2.flow_model_v2(path, classes=("Res", "Ind"))
    F = model.predict(x, verbose=False)["F"]
    users = F[:, 9:9 + 6 * 8]
    assert (users[:, :8] <= x[:, :8] + 1e-4).all() and (users[:, 8:16] <= x[:, 8:] + 1e-4).all()
    assert np.abs(users[:, 16:]).max() == 0.0
    assert v2.node_balance_residual(path, F, x, ("Res", "Ind")).max() / x.sum(1).max() < 1e-4


def test_positive_source_minimum_is_refused(tmp_path):
    sheets = v2.read_network(str(NETS["8"][0]))
    out = tmp_path / "net.xlsx"
    with pd.ExcelWriter(out) as xl:
        for key, name in v2._SHEETS.items():
            df = sheets[key].copy()
            if key == "well":
                df["Imin"] = 5.0
            df.to_excel(xl, sheet_name=name, index=False)
    with pytest.raises(ValueError, match="Imin"):
        v2.flow_model_v2(str(out))


@pytest.mark.parametrize("classes", [(), ("Res", "Res"), ("Gas",)])
def test_invalid_classes_are_refused(classes):
    with pytest.raises(ValueError, match="classes"):
        v2.flow_model_v2(str(NETS["8"][0]), classes=classes)
