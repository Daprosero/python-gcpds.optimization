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
