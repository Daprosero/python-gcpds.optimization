"""Portability smoke: the paper code runs from a plain clone on any OS.

Rebuilds the gas-network RPINN of the article with its published weights and
writes a numeric fingerprint to RPINN/Results/portability/<platform>.json, so
runs on different operating systems can be compared value by value.
"""
import json
import platform
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from RPINN import model_real_data_ as real  # noqa: E402
from RPINN import model_synthetic_data_ as synth  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "RPINN" / "Results" / "portability"


def test_data_dir_resolves_inside_the_clone():
    assert real.DATA_DIR == REPO / "RPINN" / "Data"
    assert synth.DATA_DIR == REPO / "RPINN" / "Data"
    assert (real.DATA_DIR / "Redes" / "ng_case8.xlsx").is_file()


def test_synthetic_inputs_load():
    u = np.load(synth.DATA_DIR / "Sintéticos" / "U.npy")
    y = np.load(synth.DATA_DIR / "Sintéticos" / "Y.npy")
    assert u.size > 0 and y.size > 0
    assert np.isfinite(u).all() and np.isfinite(y).all()


@pytest.mark.parametrize("s, weights", [(0, "Modelo1/model1_8.h5"), (1, "Modelo2/model2_8.h5")])
def test_gas_model_reproduces_with_paper_weights(s, weights):
    rng = np.random.default_rng(1)
    x_test = rng.uniform(0.0, 20.0, size=(64, 8)).astype("float32")
    fd = x_test.max(axis=0)
    model = real.flow_model(str(real.DATA_DIR / "Redes" / "ng_case8.xlsx"), fd, s=s)
    model.load_weights(str(real.DATA_DIR / "M" / weights))
    flows = tf.keras.Model(model.inputs, model.get_layer("F").output).predict(x_test, verbose=False)
    pressures = tf.keras.Model(model.inputs, model.get_layer("P").output).predict(x_test, verbose=False)
    assert np.isfinite(flows).all() and np.isfinite(pressures).all()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{platform.system().lower()}-{platform.machine().lower()}.json"
    record = json.loads(out.read_text()) if out.is_file() else {}
    record[f"s{s}"] = {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "flows": flows[:4].round(6).tolist(),
        "pressures": pressures[:4].round(6).tolist(),
        "flowsSum": float(flows.sum()),
        "pressuresSum": float(pressures.sum()),
    }
    out.write_text(json.dumps(record, indent=2))
