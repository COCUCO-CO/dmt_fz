import ast
from pathlib import Path

import numpy as np
import pandas as pd


MULTI2POOL_PATH = Path("/home/tomas/PycharmProjects/dmt_fz/dmt/multi2pool2.py")
PEARSON_PATH = Path("/home/tomas/PycharmProjects/dmt_fz/dmt/pearson.py")


def _load_functions(file_path, names, extra_globals=None):
    source = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        truncated = "\n".join(source.splitlines()[: max(exc.lineno - 1, 0)])
        tree = ast.parse(truncated, filename=str(file_path))
    selected_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]

    module = ast.Module(body=selected_nodes, type_ignores=[])
    code = compile(module, filename=str(file_path), mode="exec")
    namespace = {"np": np, "pd": pd}
    if extra_globals:
        namespace.update(extra_globals)
    exec(code, namespace)
    return {name: namespace[name] for name in names}


def test_network_filter_returns_expected_rows():
    funcs = _load_functions(MULTI2POOL_PATH, {"network_filter"})
    network_filter = funcs["network_filter"]

    phase = np.arange(12).reshape(4, 3)
    labels = pd.DataFrame(
        {
            "label": ["RH_FPN", "LH_FPN", "RH_DMN", "LH_DMN"],
            "hemi": ["RH", "LH", "RH", "LH"],
            "net": ["FPN", "FPN", "DMN", "DMN"],
        }
    )

    filtered = network_filter(phase, labels, hemi="RH", net="FPN")
    expected = pd.DataFrame(phase[[0]], columns=[0, 1, 2])
    assert filtered.reset_index(drop=True).equals(expected)


def test_order_parameter_filter_matches_manual_computation():
    funcs = _load_functions(
        PEARSON_PATH,
        {"order_parameter_filter", "order_parameter"},
        extra_globals={"np": np},
    )
    order_parameter_filter = funcs["order_parameter_filter"]
    order_parameter = funcs["order_parameter"]

    phase = np.vstack([
        np.zeros(50),
        np.zeros(50),
        np.zeros(50),
    ])
    labels = pd.DataFrame(
        {
            "label": ["RH_FPN", "LH_FPN", "RH_DMN"],
            "hemi": ["RH", "LH", "RH"],
            "net": ["FPN", "FPN", "DMN"],
        }
    )

    r_all = order_parameter(phase)
    r_filtered = order_parameter_filter(phase, labels, hemi="both", net="all")

    assert np.allclose(r_filtered, r_all)


def test_reject_outliers_removes_far_values():
    funcs = _load_functions(
        PEARSON_PATH,
        {"reject_outliers", "reject_outliers_per_subject"},
        extra_globals={"np": np},
    )
    reject_outliers = funcs["reject_outliers"]
    reject_outliers_per_subject = funcs["reject_outliers_per_subject"]

    data = np.array([1, 1, 1, 10, 25])
    mask = np.abs(data - data.mean()) < 2 * data.std()
    expected = data[mask]
    cleaned = reject_outliers(data, m=2)
    assert np.array_equal(cleaned, expected)

    dataset = np.array([1, 1, 1, 1, 1, 10, 25])
    subject_mask = np.abs(data - dataset.mean()) < 2 * dataset.std()
    expected_subject = data[subject_mask]
    cleaned_subject = reject_outliers_per_subject(data, dataset, m=2)
    assert np.array_equal(cleaned_subject, expected_subject)

