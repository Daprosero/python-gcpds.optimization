"""The v1 code reproduces the published RPINN paper (Computers 2024, 13, 176)."""
import numpy as np
import pytest
import tensorflow as tf

from RPINN import model_real_data_ as real

NET8 = str(real.DATA_DIR / "Redes" / "ng_case8.xlsx")


@pytest.fixture(scope="module")
def gas8():
    return real.flow_model(NET8, np.full(8, 20.0), s=1)


def test_gas_architecture_has_the_paper_parameter_count(gas8):
    # Table tab:arch-gas: 12,707 parameters, non-trainable and fixed weights included.
    assert gas8.count_params() == 12707


def test_gas_architecture_layers_match_the_paper_table(gas8):
    dense = {l.name: l for l in gas8.layers if type(l) is tf.keras.layers.Dense}
    assert dense["Dense_1"].units == 236 and dense["Dense_1"].input.shape[-1] == 8
    assert dense["Dense_2"].units == 8
    assert dense["Dense_3"].units == 236 and dense["Dense_3"].input.shape[-1] == 8
    bn = [l for l in gas8.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    assert sorted(l.count_params() for l in bn) == [32, 944, 944]
    # The pressure layer reads the output of BatchNormalization_3, never the bare dense.
    assert isinstance(gas8.get_layer("P").inbound_nodes[0].inbound_layers, tf.keras.layers.BatchNormalization)


def test_gas_loss_has_no_cost_and_paper_weights(gas8):
    # lambda_L = 0: the operating cost is evaluated, never trained on.
    assert gas8.output_names == ["Bal", "Wey", "Ratio"]
    assert list(gas8.compiled_loss._loss_weights) == [0.6, 0.1, 0.3]
    opt = gas8.optimizer
    assert type(opt).__name__ == "Adamax" and np.isclose(float(opt.learning_rate), 1e-2)


def test_unsupplied_model_keeps_its_previous_outputs():
    # s=0 (no unsupplied gas) still builds and predicts finite flows and pressures.
    model = real.flow_model(NET8, np.full(8, 20.0), s=0)
    x = np.random.default_rng(0).uniform(0, 20, (4, 8)).astype("float32")
    out = tf.keras.Model(model.inputs, [model.get_layer("F").output, model.get_layer("P").output]).predict(x, verbose=0)
    assert all(np.isfinite(o).all() for o in out)


# --- Training entry point (S5) -------------------------------------------------
from RPINN import paper_runs as runs


@pytest.mark.parametrize("tip,lo,hi", [("S", 0.0, 0.05), ("M", 0.40, 0.60), ("D", 0.95, 1.0)])
def test_scenario_demands_exceed_the_source_as_the_paper_states(tip, lo, hi):
    # S below the source capacity, M about half above it, D almost all above it.
    X = runs.scenario_demands("8", tip, n=20000, seed=0)
    assert X.shape == (20000, 8)
    cap = runs.source_capacity("8")
    share = float((X.sum(axis=1) > cap).mean())
    assert lo <= share <= hi


def test_scenario_demands_are_reproducible_from_the_seed():
    a = runs.scenario_demands("12", "M", n=50, seed=3)
    b = runs.scenario_demands("12", "M", n=50, seed=3)
    c = runs.scenario_demands("12", "M", n=50, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_split_keeps_thirty_percent_for_test():
    X = runs.scenario_demands("8", "S", n=1000, seed=0)
    tr, te = runs.split_paper(X, seed=0)
    assert len(tr) == 700 and len(te) == 300


def test_train_run_writes_weights_history_and_ipopt_inputs(tmp_path):
    rec = runs.run_one("8", "S", s=1, out=tmp_path, n=400, epochs=2, seed=0)
    w = tmp_path / "S" / "Modelo2" / "model2_8.h5"
    assert w.exists()
    assert len(rec["epoch_seconds"]) == 2 and set(rec["loss"]) >= {"Bal_loss", "Wey_loss", "Ratio_loss"}
    assert (tmp_path / "S" / "split" / "8_test.npy").exists()
    # The saved weights load into a fresh paper model built with the stored bound.
    fd = np.load(tmp_path / "S" / "Modelo2" / "fd_8.npy")
    m = real.flow_model(NET8, fd, s=1)
    m.load_weights(str(w))
    runs.collect(tmp_path, tips=["S"], nets=["8"])
    X = real.load_data(tmp_path / "S" / "MAPE" / "inputs_P.mat", "inputs")
    assert np.asarray(X[0]).shape == (120, 8)  # min(320, test size)


def test_lr_schedule_divides_by_ten_at_the_paper_epochs():
    assert runs.LR_EPOCHS == (400, 500, 580)


# --- IPOPT reference runs (S6) -------------------------------------------------
from RPINN import ipopt_runs


def test_ipopt_solves_the_same_held_out_samples_and_keeps_failures(tmp_path):
    split = tmp_path / "S" / "split"
    split.mkdir(parents=True)
    X = runs.scenario_demands("8", "S", n=3, seed=0)
    np.save(split / "8_test.npy", X)
    rec = ipopt_runs.run_network(tmp_path, "S", "8", max_time=5)
    saved = np.load(tmp_path / "S" / "ipopt" / "8.npz")
    # Same samples RPINN sees, one row per sample whatever the outcome.
    assert np.allclose(saved["X"], X) and len(saved["status"]) == 3
    assert saved["P"].shape == (3, 8) and float(saved["max_time"]) == 5
    ok = saved["status"] == "ok"
    assert np.isfinite(saved["objective"][ok]).all() and np.isnan(saved["objective"][~ok]).all()


def test_ipopt_failure_is_reported_not_raised():
    # A one-millisecond cap cannot converge: the sample comes back marked, not as an exception.
    r = ipopt_runs.solve_sample("8", np.full(8, 5.0), max_time=0.001)
    assert r["status"] != "ok" and np.isnan(r["objective"]) and r["F"] is None


# --- Uniform mixture model (S7) --------------------------------------------------
from RPINN import mixture_runs as mix


def test_mixture_architecture_matches_the_original_code():
    # One output neuron applied straight to z: its weights are the mixing coefficients.
    m = mix.mixture_model(0)
    assert m.count_params() == 5
    dense = [l for l in m.layers if isinstance(l, tf.keras.layers.Dense)]
    assert len(dense) == 1
    out = dense[0]
    assert out.units == 1 and not out.use_bias and out.activation.__name__ == "relu"
    assert np.allclose(out.get_weights()[0], 0.2)
    assert type(m.optimizer).__name__ == "Adamax" and np.isclose(float(m.optimizer.learning_rate), 1e-3)
    assert mix.EPOCHS == 100 and mix.BATCH == 16 and mix.SNRS == (-1, 1, 3, 5, None)


def test_output_weights_are_projected_to_sum_one():
    m = mix.mixture_model(0)
    c = m.get_layer("Dense_out").kernel_constraint
    w = c(tf.constant([[2.0], [-1.0], [0.5], [0.5], [1.0]])).numpy().ravel()
    assert np.allclose(w, [0.5, 0, 0.125, 0.125, 0.25])


def test_output_weights_recover_the_noise_free_coefficients():
    pi, Z, y, _ = mix.make_problem(0)
    x, _, _, _, _, _ = mix.fit_rpinn(Z, y, seed=0)
    assert np.allclose(x, pi, atol=0.02) and np.isclose(x.sum(), 1, atol=1e-5)


def test_mixture_problem_is_the_2024_data():
    d = mix.DATA_DIR
    U, Y, PIC = np.load(d / "U.npy"), np.load(d / "Y.npy"), np.load(d / "PIC.npy")
    pi, Z, y, noisy = mix.make_problem(7)
    assert np.array_equal(pi, PIC[7].ravel()) and np.array_equal(Z, U[7])
    assert np.array_equal(y, Y[-1, 7].ravel()) and noisy.shape == (5, 500)
    assert np.array_equal(noisy[0], Y[0, 7].ravel()) and np.array_equal(noisy[-1], y)


def test_ipopt_recovers_the_noise_free_coefficients():
    pi, Z, y, _ = mix.make_problem(1)
    x, _ = mix.solve_ipopt(Z[:400], y[:400])
    assert np.allclose(x, pi, atol=1e-3) and np.isclose(x.sum(), 1, atol=1e-6)


def test_one_repetition_writes_every_snr(tmp_path):
    rec = mix.run_rep(0, tmp_path, seed=0, epochs=2)
    assert np.array_equal(rec["y_noisy"], mix.make_problem(0)[3])
    saved = np.load(tmp_path / "Sinteticos" / "mixture_rep000.npz")
    assert saved["x_rpinn"].shape == (5, 5) and saved["pred_rpinn"].shape == (5, 100)
    assert saved["x_rpinn_effective"].shape == (5, 5)
    assert np.isnan(saved["snr"][-1]) and list(saved["snr"][:4]) == [-1, 1, 3, 5]


def test_effective_coefficients_of_a_linear_net_are_its_weights():
    inp = tf.keras.Input(shape=(5,))
    lin = tf.keras.Model(inp, tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.Constant(0.2))(inp))
    assert np.allclose(mix.effective_coefficients(lin, np.random.rand(10, 5)), 0.2)
