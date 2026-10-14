# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for splitting schedule-tree tasklets into a part that computes some values and a part that uses them."""

import numpy as np

import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.tasklet_splitting import (
    dependencies,
    parse_tasklet,
    split_saving,
    split_tasklet,
)


def _tasklet(code: str, inputs: dict, outputs: dict) -> tn.TaskletNode:
    """A tasklet with connectors -> memlet strings."""
    tasklet = dace.nodes.Tasklet("compute", set(inputs), set(outputs), code)
    return tn.TaskletNode(
        node=tasklet,
        in_memlets={c: dace.Memlet(m) for c, m in inputs.items()},
        out_memlets={c: dace.Memlet(m) for c, m in outputs.items()},
    )


def _containers() -> dict:
    return {
        "y": dace.data.Array(dace.float64, [2]),
        "z": dace.data.Array(dace.float64, [1]),
        "out": dace.data.Array(dace.float64, [1]),
    }


def _ratio_tasklet() -> tn.TaskletNode:
    """``r = b / a`` depends only on ``a`` and ``b``; ``o`` also on ``c``."""
    return _tasklet(
        "r = b / a\ns = r * (r + 0.5)\no = (c + r) / s", {"a": "y[0]", "b": "y[1]", "c": "z[0]"}, {"o": "out[0]"}
    )


def test_parse_single_assignments():
    code = parse_tasklet(_ratio_tasklet())
    assert code.targets == ["r", "s", "o"]
    assert code.reads[2] == {"c", "r", "s"}
    assert dependencies(code, {1}) == {0, 1}


def test_parse_rejects_other_forms():
    reassigned = _tasklet("t = a\nt = t + 1\no = t", {"a": "y[0]"}, {"o": "out[0]"})
    assert parse_tasklet(reassigned) is None
    branch = _tasklet("if a > 0:\n    o = a\nelse:\n    o = 0.0", {"a": "y[0]"}, {"o": "out[0]"})
    assert parse_tasklet(branch) is None
    writes_input = _tasklet("a = a + 1\no = a", {"a": "y[0]"}, {"o": "out[0]"})
    assert parse_tasklet(writes_input) is None
    effect = _ratio_tasklet()
    effect.node.side_effects = True
    assert parse_tasklet(effect) is None


def test_split_passes_values():
    node = _ratio_tasklet()
    code = parse_tasklet(node)
    stored = {}

    def store(name, dtype):
        stored[name] = dtype
        return dace.Memlet(f"tmp_{name}[0]")

    first, second = split_tasklet(node, code, {0, 1}, store, _containers())
    assert set(stored) == {"r", "s"}
    assert stored["r"] == dace.float64
    assert set(first.in_memlets) == {"a", "b"} and set(first.out_memlets) == {"r", "s"}
    assert set(second.in_memlets) == {"c", "r", "s"} and set(second.out_memlets) == {"o"}
    assert str(second.in_memlets["r"]) == "tmp_r[0]"
    assert first.node.code.as_string.count("=") == 2 and second.node.code.as_string.count("=") == 1


def test_split_repeats_copies():
    """A local that only copies an input is computed again instead of stored."""
    node = _tasklet("c = a\nd = c * c\no = c + d * e", {"a": "y[0]", "e": "z[0]"}, {"o": "out[0]"})
    code = parse_tasklet(node)
    stored = []
    store = lambda n, t: stored.append(n) or dace.Memlet(f"tmp_{n}[0]")
    first, second = split_tasklet(node, code, {0, 1}, store, _containers())
    assert stored == ["d"]
    assert set(second.in_memlets) == {"a", "e", "d"}


def test_split_runs():
    """Running the two parts one after the other computes what the tasklet computes."""
    sdfg = dace.SDFG("split_runs")
    sdfg.add_array("y", [2], dace.float64)
    sdfg.add_array("z", [1], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_state(is_start_block=True)
    stree = sdfg.as_schedule_tree()
    stree.children = []
    node = _ratio_tasklet()
    code = parse_tasklet(node)

    def store(name, dtype):
        stree.containers[f"tmp_{name}"] = dace.data.Array(dtype, [1], transient=True)
        return dace.Memlet(f"tmp_{name}[0]")

    stree.add_children(list(split_tasklet(node, code, {0, 1}, store, stree.containers)))
    y, z, out = np.array([2.0, 3.0]), np.array([5.0]), np.zeros(1)
    stree.as_sdfg()(y=y, z=z, out=out)
    r = 3.0 / 2.0
    assert np.allclose(out, (5.0 + r) / (r * (r + 0.5)))


def test_split_saving():
    """Divisions pay for storing their results; one cheap operation whose inputs the rest reads anyway does not."""
    code = parse_tasklet(_ratio_tasklet())
    assert split_saving(code, {0, 1}) > 0  # A division and two cheap operations for two loads
    cheap = parse_tasklet(_tasklet("t = a * 2.0\no = t + a * c", {"a": "y[0]", "c": "z[0]"}, {"o": "out[0]"}))
    assert split_saving(cheap, {0}) == 0


def test_split_keeps_declared_types():
    """A local declared as ``float32`` is stored as one, even if computed from doubles (the C++ local rounds)."""
    node = _tasklet("r: dace.float32\nr = b / a\no = c + r", {"a": "y[0]", "b": "y[1]", "c": "z[0]"}, {"o": "out[0]"})
    code = parse_tasklet(node)
    stored = {}
    first, second = split_tasklet(
        node, code, {0}, lambda n, t: stored.setdefault(n, t) and dace.Memlet("tmp[0]"), _containers()
    )
    assert stored == {"r": dace.float32}
    assert "r: dace.float32" in first.node.code.as_string and "dace.float32" not in second.node.code.as_string


if __name__ == "__main__":
    test_parse_single_assignments()
    test_parse_rejects_other_forms()
    test_split_passes_values()
    test_split_repeats_copies()
    test_split_runs()
    test_split_saving()
    test_split_keeps_declared_types()
