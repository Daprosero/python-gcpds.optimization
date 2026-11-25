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
