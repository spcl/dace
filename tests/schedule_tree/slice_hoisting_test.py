# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for hoisting the invariant slices of loop bodies out of loops."""

import copy

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes import hoist_invariant_slices
from dace.sdfg.state import LoopRegion

K = 8  # Layers
T = 4  # Fields remapped with the same layers


def _loop(var: str, start: int, end: int, body: list) -> tn.ForScope:
    loop = LoopRegion(f"loop_{var}", f"{var} < {end}", var, f"{var} = {start}", f"{var} = {var} + 1")
    return tn.ForScope(loop=loop, children=body)


def _tasklet(code: str, inputs: dict[str, str], outputs: dict[str, str]) -> tn.TaskletNode:
    """A tasklet with connectors -> memlet strings (connectors of ranges are pointers to ``float64``)."""
    connectors = {c: dace.pointer(dace.float64) if ":" in m else None for c, m in inputs.items()}
    tasklet = dace.nodes.Tasklet("compute", connectors, set(outputs), code)
    return tn.TaskletNode(
        node=tasklet,
        in_memlets={c: dace.Memlet(m) for c, m in inputs.items()},
        out_memlets={c: dace.Memlet(m) for c, m in outputs.items()},
    )


def _remap_tree(reset_inside: bool = True, variant_targets: bool = False, fields: int = T) -> tn.ScheduleTreeRoot:
    """
    Conservative remapping of ``fields`` fields from layers with interfaces ``edges`` to layers with interfaces
    ``targets``: for each field, a profile (``a``, different per field), then per target layer a search for the source
    layers it overlaps, which advances a position ``lev`` (relative to the layer index) under ``if``/``else`` and in a
    ``while`` loop. Only the profile depends on the field: the search and its fractions (divisions) are the same for all.

    :param reset_inside: Whether each field resets ``lev`` (else it is reset once before the loop and carried).
    :param variant_targets: Whether the target interfaces differ per field.
    :param fields: Number of fields.
    """
    sdfg = dace.SDFG("remap_slice")
    sdfg.add_array("edges", [K + 1], dace.float64)
    sdfg.add_array("widths", [K], dace.float64)
    sdfg.add_array("targets", [fields, K + 1] if variant_targets else [K + 1], dace.float64)
    sdfg.add_array("q", [fields, K], dace.float64)
    sdfg.add_array("out", [fields, K], dace.float64)
    sdfg.add_array("a", [K], dace.float64, transient=True)
    sdfg.add_scalar("lev", dace.int32, transient=True)
    sdfg.add_scalar("w", dace.float64, transient=True)
    sdfg.add_scalar("s", dace.float64, transient=True)
    sdfg.add_scalar("m", dace.bool, transient=True)
    sdfg.add_scalar("c", dace.bool, transient=True)
    sdfg.add_state(is_start_block=True)
    stree = sdfg.as_schedule_tree()
    stree.children = []

    y, y1 = ("targets[t, k]", "targets[t, k + 1]") if variant_targets else ("targets[k]", "targets[k + 1]")
    grid = {"e": f"edges[0:{K + 1}]", "d": f"widths[0:{K}]", "l": "lev[0]"}
    profile = _loop("k", 0, K, [_tasklet("o = 2.0 * v + 1.0", {"v": "q[t, k]"}, {"o": "a[k]"})])
    reset = _tasklet("o = 0", {}, {"o": "lev[0]"})
    fraction = _tasklet(
        "wl = (y - e[k + l]) / d[k + l]\nml = y1 <= e[k + l + 1]",
        {**grid, "y": y, "y1": y1},
        {"wl": "w[0]", "ml": "m[0]"},
    )
    inside = _tasklet(
        "r = (y1 - e[k + l]) / d[k + l]\no = av[k + l] * (r + wl)",
        {**grid, "y1": y1, "wl": "w[0]", "av": f"a[0:{K}]"},
        {"o": "out[t, k]"},
    )
    first = _tasklet(
        "sl = (e[k + l + 1] - y) * av[k + l] * (1.0 + wl)\nnl = l + 1",
        {**grid, "y": y, "wl": "w[0]", "av": f"a[0:{K}]"},
        {"sl": "s[0]", "nl": "lev[0]"},
    )
    condition = _tasklet("cl = e[k + l + 1] < y1", {**grid, "y1": y1}, {"cl": "c[0]"})
    whole = _tasklet(
        "sn = sl + d[k + l] * av[k + l]\nnl = l + 1\ncl = e[k + nl + 1] < y1",
        {**grid, "y1": y1, "sl": "s[0]", "av": f"a[0:{K}]"},
        {"sn": "s[0]", "nl": "lev[0]", "cl": "c[0]"},
    )
    last = _tasklet(
        "dp = y1 - e[k + l]\nes = dp / d[k + l]\nsn = sl + dp * av[k + l] * es\no = sn / (y1 - y)",
        {**grid, "y": y, "y1": y1, "sl": "s[0]", "av": f"a[0:{K}]"},
        {"o": "out[t, k]"},
    )
    search = tn.WhileScope(loop=LoopRegion("search", condition_expr=CodeBlock("c")), children=[whole])
    step = _tasklet("nl = l - 1", {"l": "lev[0]"}, {"nl": "lev[0]"})
    layers = _loop(
        "k",
        0,
        K,
        [
            fraction,
            tn.IfScope(condition=CodeBlock("m"), children=[inside]),
            tn.ElseScope(children=[first, condition, search, last]),
            step,
        ],
    )
    body = [profile, reset, layers] if reset_inside else [profile, layers]
    if not reset_inside:
        stree.add_child(copy.deepcopy(reset))
    stree.add_child(_loop("t", 0, fields, body))
    return stree


def _inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    widths = rng.random(K) + 0.5
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    new = rng.random(K) + 0.5
    targets = np.concatenate([[0.0], np.cumsum(new * edges[-1] / new.sum())])
    targets[-1] = edges[-1]  # The search stops at the top
    return {"edges": edges, "widths": widths, "targets": targets, "q": rng.random((T, K)), "out": np.zeros((T, K))}


def _run(stree: tn.ScheduleTreeRoot, arguments: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    arguments = {k: v.copy() for k, v in arguments.items()}
    stree.as_sdfg()(**arguments)
    return arguments


def _tasklets(nodes) -> list:
    return [n for root in nodes for n in root.preorder_traversal() if isinstance(n, tn.TaskletNode)]


def test_hoist_search():
    """The search runs once before the loop over fields, which loads its fractions instead of dividing."""
    stree = _remap_tree()
    assert hoist_invariant_slices(stree) == 1
    tn.validate_children_and_parents_align(stree, root=True)
    *copied, loop = stree.children
    assert type(loop) is tn.ForScope and loop.loop.loop_variable == "t"
    # The copy: the reset of the position and the layer loop with the search, without the profile
    assert len(copied) == 2 and type(copied[1]) is tn.ForScope
    assert any(type(n) is tn.WhileScope for n in copied[1].preorder_traversal())
    assert not any("av" in t.in_memlets for t in _tasklets(copied))
    # The loop keeps its control flow and position updates, but no division other than by the layer thickness
    assert sum(type(n) is tn.WhileScope for n in loop.preorder_traversal()) == 1
    divisions = [t.node.code.as_string for t in _tasklets([loop]) if "/" in t.node.code.as_string]
    assert len(divisions) == 1 and "sn" in divisions[0]

    expected = _run(_remap_tree(), _inputs())
    result = _run(stree, _inputs())
    assert np.array_equal(expected["out"], result["out"])


def test_hoist_search_not_if_carried_or_variant():
    # A position carried from one field to the next is not the same in every iteration
    assert hoist_invariant_slices(_remap_tree(reset_inside=False)) == 0
    # Neither is a search for targets that differ per field
    assert hoist_invariant_slices(_remap_tree(variant_targets=True)) == 0
    # A loop that runs once gains nothing
    assert hoist_invariant_slices(_remap_tree(fields=1)) == 0


def _guarded_tree(code: str = "r = b / c\no = v * r", inputs: dict[str, str] = None) -> tn.ScheduleTreeRoot:
    """``for t: for k: for i: if x[i] > 0: out[t, k, i] = q[t, k, i] * (y[k] / z[k])``: the quotient is the same for all
    ``t`` and ``i``, but runs under a guard that depends on ``i``."""
    sdfg = dace.SDFG("guarded_slice")
    for name, shape in (("x", [K]), ("y", [K]), ("z", [K]), ("q", [T, K, K]), ("out", [T, K, K])):
        sdfg.add_array(name, shape, dace.float64)
    sdfg.add_scalar("dt", dace.float64)
    sdfg.add_state(is_start_block=True)
    stree = sdfg.as_schedule_tree()
    stree.children = []
    statement = _tasklet(code, inputs or {"b": "y[k]", "c": "z[k]", "v": "q[t, k, i]"}, {"o": "out[t, k, i]"})
    guard = tn.IfScope(condition=CodeBlock("x[i] > 0.0"), children=[statement])
    stree.add_child(_loop("t", 0, T, [_loop("k", 0, K, [_loop("i", 0, K, [guard])])]))
    return stree


def _guarded_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    return {
        "x": rng.random(K) - 0.5,
        "y": rng.random(K),
        "z": rng.random(K) + 0.5,
        "dt": np.float64(0.3),
        "q": rng.random((T, K, K)),
        "out": np.zeros((T, K, K)),
    }


def test_hoist_guarded_value_per_dimension():
    """The quotient is stored per ``k`` only, the variable it depends on."""
    stree = _guarded_tree()
    assert hoist_invariant_slices(stree) == 1
    copied, loop = stree.children
    (part,) = _tasklets([copied])
    (memlet,) = part.out_memlets.values()
    assert stree.containers[memlet.data].shape == (K,) and str(memlet.subset) == "k"
    (rest,) = _tasklets([loop])
    assert "/" not in rest.node.code.as_string
    expected = _run(_guarded_tree(), _guarded_inputs())
    result = _run(stree, _guarded_inputs())
    assert np.array_equal(expected["out"], result["out"])


def test_hoist_guarded_value_not_if_cheap_or_constant():
    # One addition is not worth a store and a load
    assert hoist_invariant_slices(_guarded_tree("r = b + c\no = v * r")) == 0
    # A value the same in every iteration of every loop is the compiler's business
    assert hoist_invariant_slices(_guarded_tree("r = b / c\no = v * r", {"b": "dt", "c": "dt", "v": "q[t, k, i]"})) == 0


if __name__ == "__main__":
    test_hoist_search()
    test_hoist_search_not_if_carried_or_variant()
    test_hoist_guarded_value_per_dimension()
    test_hoist_guarded_value_not_if_cheap_or_constant()
