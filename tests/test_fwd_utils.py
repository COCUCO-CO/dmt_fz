import ast
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest


scipy_signal = pytest.importorskip("scipy.signal")
from scipy.signal import hilbert as scipy_hilbert


FWD_PATH = Path("/home/tomas/PycharmProjects/dmt_fz/dmt/fwd.py")


def _load_fwd_functions():
    source = FWD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(FWD_PATH))
    target_names = {"diff_ang", "hilbert_transform", "calculate_syncro", "order_parameter"}
    selected_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in target_names
    ]

    module = ast.Module(body=selected_nodes, type_ignores=[])
    code = compile(module, filename=str(FWD_PATH), mode="exec")

    namespace = {
        "np": np,
        "hilbert": scipy_hilbert,
        "comb": combinations,
        "euler_notation": np.vectorize(lambda x: np.exp(1j * x)),
    }
    exec(code, namespace)
    return {name: namespace[name] for name in target_names}


@pytest.fixture(scope="module")
def fwd_funcs():
    return _load_fwd_functions()


def test_diff_ang_handles_wraparound(fwd_funcs):
    diff_ang = fwd_funcs["diff_ang"]
    theta1 = np.array([0, np.pi / 2, -np.pi])
    theta2 = np.array([np.pi, -np.pi / 2, np.pi])
    result = diff_ang(theta1, theta2)
    expected = np.array([np.pi, np.pi, 0.0])
    assert np.allclose(result, expected)


def test_calculate_syncro_returns_symmetric_matrix(fwd_funcs):
    calculate_syncro = fwd_funcs["calculate_syncro"]
    phase = np.vstack([
        np.linspace(0, np.pi, 200),
        np.linspace(0, np.pi, 200),
        np.linspace(0, np.pi, 200),
    ])
    syncro = calculate_syncro(phase)

    assert syncro.shape == (3, 3)
    assert np.allclose(syncro, syncro.T)
    off_diag = syncro[np.triu_indices(3, k=1)]
    assert np.allclose(off_diag, 1.0, atol=1e-6)
    assert np.allclose(np.diag(syncro), 0.0)


def test_hilbert_transform_produces_unit_envelope(fwd_funcs):
    hilbert_transform = fwd_funcs["hilbert_transform"]
    t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    signal = np.sin(t)
    envelope, phase = hilbert_transform([signal])

    assert envelope.shape == (1, 500)
    assert phase.shape == (1, 500)
    assert np.allclose(envelope[0][100:-100], 1.0, atol=1e-2)


def test_order_parameter_bounds(fwd_funcs):
    order_parameter = fwd_funcs["order_parameter"]

    identical_phase = np.zeros((5, 100))
    r_identical = order_parameter(identical_phase)
    assert np.allclose(r_identical, 1.0)

    rng = np.random.default_rng(seed=42)
    random_phase = rng.uniform(-np.pi, np.pi, size=(20, 200))
    r_random = order_parameter(random_phase)
    assert np.all(r_random <= 1.0)
    assert np.mean(r_random) < 0.3

