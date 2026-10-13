# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for moving statements and subtrees of schedule trees past their siblings."""

import copy
from collections.abc import Callable

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes import coarsen_guards, move_statements_to_readers, sink_into_guards
from dace.sdfg.analysis.schedule_tree.passes.code_motion import MotionAnalysis, Reordering
from dace.sdfg.state import LoopRegion


def _root(name: str, arrays: dict[str, tuple], transients: dict[str, tuple] = None) -> tn.ScheduleTreeRoot:
    """An empty tree with the given arrays (name -> (shape, dtype)) and transients (scalars if the shape is empty)."""
    sdfg = dace.SDFG(name)
    for array, (shape, dtype) in arrays.items():
        sdfg.add_array(array, shape, dtype)
    for array, (shape, dtype) in (transients or {}).items():
        if shape:
            sdfg.add_array(array, shape, dtype, transient=True)
        else:
            sdfg.add_scalar(array, dtype, transient=True)
    sdfg.add_state(is_start_block=True)
    stree = sdfg.as_schedule_tree()
    stree.children = []
    return stree


def _loop(var: str, start: int, end: int, body: list) -> tn.ForScope:
    loop = LoopRegion(f"loop_{var}", f"{var} < {end}", var, f"{var} = {start}", f"{var} = {var} + 1")
    return tn.ForScope(loop=loop, children=body)


def _descending_loop(var: str, start: int, last: int, body: list) -> tn.ForScope:
    loop = LoopRegion(f"loop_{var}", f"{var} >= {last}", var, f"{var} = {start}", f"{var} = {var} - 1")
    return tn.ForScope(loop=loop, children=body)


def _tasklet(code: str, inputs: dict[str, str], outputs: dict[str, str]) -> tn.TaskletNode:
    """A tasklet with connectors -> memlet strings."""
    tasklet = dace.nodes.Tasklet("compute", set(inputs), set(outputs), code)
    return tn.TaskletNode(
        node=tasklet,
        in_memlets={c: dace.Memlet(m) for c, m in inputs.items()},
        out_memlets={c: dace.Memlet(m) for c, m in outputs.items()},
    )


def _if(condition: str, body: list) -> tn.IfScope:
    return tn.IfScope(condition=CodeBlock(condition), children=body)


def _ancestors(node: tn.ScheduleTreeNode) -> list:
    result = []
    while node.parent is not None:
        node = node.parent
        result.append(node)
    return result


def _run(stree: tn.ScheduleTreeRoot, arguments: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    arguments = {k: v.copy() for k, v in arguments.items()}
    sdfg = stree.as_sdfg(simplify=dace.config.Config.get_bool("optimizer", "automatic_simplification"))
    sdfg(**arguments)
    return arguments


def _same_results(
    build: Callable[[], tn.ScheduleTreeRoot],
    transform: Callable[[tn.ScheduleTreeRoot], None],
    inputs: list[dict[str, np.ndarray]],
) -> tn.ScheduleTreeRoot:
    """Transforms a tree and checks that it computes what the original does for each set of inputs."""
    transformed = build()
    transform(transformed)
    tn.validate_children_and_parents_align(transformed, root=True)
    for arguments in inputs:
        expected = _run(build(), arguments)
        result = _run(copy.deepcopy(transformed), arguments)
        for name in expected:
            assert np.array_equal(expected[name], result[name]), name
    return transformed


# ----------------------------------------------------------------------------------------------------------------------
# Reordering siblings
# ----------------------------------------------------------------------------------------------------------------------


def _pair(unit: tn.ScheduleTreeNode, between: list, unit_first: bool = True, after: list = None) -> tn.ScheduleTreeRoot:
    """A tree with ``unit`` before (or after) the siblings ``between``, then ``after``."""
    stree = _root(
        "reorder",
        {"x": ([8], dace.float64), "y": ([8], dace.float64), "out": ([8], dace.float64)},
        {"t": ([8], dace.float64), "u": ([8], dace.float64)},
    )
    stree.add_children(([unit] + between if unit_first else between + [unit]) + (after or []))
    return stree


def _copy_loop(var: str = "i", source: str = "x", target: str = "t") -> tn.ForScope:
    """``for i: t[i] = x[i]``: independent iterations."""
    return _loop(var, 0, 8, [_tasklet("o = a", {"a": f"{source}[{var}]"}, {"o": f"{target}[{var}]"})])


def _scan_loop() -> tn.ForScope:
    """``for i in 1..7: t[i] = t[i - 1] + x[i]``: each iteration reads what the previous one wrote."""
    return _loop("i", 1, 8, [_tasklet("o = p + a", {"p": "t[i - 1]", "a": "x[i]"}, {"o": "t[i]"})])


def _read(element: str, target: str = "out[0]") -> tn.TaskletNode:
    return _tasklet("o = a", {"a": element}, {"o": target})


def _reordering(stree: tn.ScheduleTreeRoot, unit: tn.ScheduleTreeNode, between: list, downward: bool = True):
    return MotionAnalysis(stree).reordering(unit, stree, between, downward)


def test_reorder_unrelated_nests():
    unit, between = _copy_loop(), [_copy_loop("j", "y", "u")]
    stree = _pair(unit, between)
    analysis = MotionAnalysis(stree)
    assert analysis.reorder_conflicts(unit, stree, between) == []
    assert analysis.reordering(unit, stree, between, downward=True) == Reordering()


def test_reorder_reads_do_not_conflict():
    unit, between = _copy_loop(), [_copy_loop("j", "x", "u")]
    assert _reordering(_pair(unit, between), unit, between) == Reordering()


def test_reorder_not_past_reader_of_every_iteration():
    unit, between = _copy_loop(), [_copy_loop("j", "t", "u")]
    stree = _pair(unit, between)
    assert MotionAnalysis(stree).reorder_conflicts(unit, stree, between) == [(0, 7)]  # All iterations
    assert _reordering(stree, unit, between) is None


def test_reorder_not_past_writer_of_inputs():
    """Write after read: the sibling overwrites what the unit reads."""
    unit, between = _copy_loop(), [_copy_loop("j", "y", "x")]
    assert _reordering(_pair(unit, between), unit, between) is None


def test_reorder_not_past_writer_of_outputs():
    """Write after write: the final values would change."""
    unit, between = _copy_loop(), [_copy_loop("j", "y", "t")]
    assert _reordering(_pair(unit, between), unit, between) is None


def test_reorder_keeps_first_iterations():
    """The sibling reads what the first two iterations write: they stay; the rest moves past it in order, so even a
    loop whose iterations depend on each other may move."""
    unit, between = _scan_loop(), [_read("t[1:3]")]
    stree = _pair(unit, between)
    assert MotionAnalysis(stree).reorder_conflicts(unit, stree, between) == [(1, 2)]
    assert _reordering(stree, unit, between) == Reordering((1, 2))


def test_reorder_keeps_last_iterations_if_independent():
    """Moving down while the last iterations stay changes the order of iterations."""
    unit, between = _copy_loop(), [_read("t[7]")]
    assert _reordering(_pair(unit, between), unit, between) == Reordering((7, 7))
    unit, between = _scan_loop(), [_read("t[7]")]
    assert _reordering(_pair(unit, between), unit, between) is None


def test_reorder_up_keeps_last_iterations():
    """Moving up past a writer of the first input: the first iteration stays after it, which changes the order."""
    unit, between = _copy_loop(), [_tasklet("o = a", {"a": "y[0]"}, {"o": "x[0]"})]
    stree = _pair(unit, between, unit_first=False)
    assert _reordering(stree, unit, between, downward=False) == Reordering((0, 0))
    unit, between = _scan_loop(), [_tasklet("o = a", {"a": "y[0]"}, {"o": "x[7]"})]
    stree = _pair(unit, between, unit_first=False)
    assert _reordering(stree, unit, between, downward=False) == Reordering((7, 7))
    unit, between = _scan_loop(), [_tasklet("o = a", {"a": "y[0]"}, {"o": "x[1]"})]
    stree = _pair(unit, between, unit_first=False)
    assert _reordering(stree, unit, between, downward=False) is None


def test_reorder_descending_loop():
    """In a loop running from 7 down to 0, iteration 0 is the last one."""
    independent = _descending_loop("i", 7, 0, [_tasklet("o = a", {"a": "x[i]"}, {"o": "t[i]"})])
    stree = _pair(independent, [_read("t[0]")])
    assert _reordering(stree, independent, stree.children[1:]) == Reordering((0, 0))
    scan = _descending_loop("i", 6, 0, [_tasklet("o = p + a", {"p": "t[i + 1]", "a": "x[i]"}, {"o": "t[i]"})])
    stree = _pair(scan, [_read("t[0]")])
    assert _reordering(stree, scan, stree.children[1:]) is None
    scan = _descending_loop("i", 6, 0, [_tasklet("o = p + a", {"p": "t[i + 1]", "a": "x[i]"}, {"o": "t[i]"})])
    stree = _pair(scan, [_read("t[6]")])
    assert _reordering(stree, scan, stree.children[1:]) == Reordering((6, 6))


def test_reorder_not_if_middle_iterations_conflict():
    unit, between = _copy_loop(), [_read("t[4]")]
    assert _reordering(_pair(unit, between), unit, between) is None


def test_reorder_conflicts_in_conditions():
    unit, between = _copy_loop(), [_if("t[7] > 0", [_read("y[0]")])]
    assert _reordering(_pair(unit, between), unit, between) == Reordering((7, 7))
    unit, between = _copy_loop(), [_if("t[4] > 0", [_read("y[0]")])]
    assert _reordering(_pair(unit, between), unit, between) is None


def test_reorder_not_past_symbol_assignment():
    unit = _loop("i", 0, 8, [_tasklet("o = a", {"a": "x[n]"}, {"o": "t[i]"})])
    between = [tn.AssignNode(name="n", value=CodeBlock("3"), edge=dace.InterstateEdge())]
    stree = _pair(unit, between)
    stree.symbols["n"] = dace.int32
    assert _reordering(stree, unit, between) is None


def test_reorder_loop_variable_used_after():
    """Two loops over ``i`` commute unless the value of ``i`` after them is read."""
    unit, between = _copy_loop(), [_copy_loop("i", "y", "u")]
    assert _reordering(_pair(unit, between), unit, between) == Reordering()
    unit, between = _copy_loop(), [_copy_loop("i", "y", "u")]
    stree = _pair(unit, between, after=[_tasklet("o = i", {}, {"o": "out[0]"})])
    assert _reordering(stree, unit, between) is None


def test_reorder_not_past_side_effects():
    unit, effect = _copy_loop(), _read("y[0]")
    effect.node.side_effects = True
    assert _reordering(_pair(unit, [effect]), unit, [effect]) is None


def test_reorder_not_past_jump():
    """Inside a loop, a ``break`` after the unit decides whether later iterations run at all."""
    stree = _pair(_copy_loop(), [])
    stree.children = []
    unit = _tasklet("o = a", {"a": "x[k]"}, {"o": "t[k]"})
    exit_early = _if("y[k] > 0", [tn.BreakNode()])
    stree.add_child(_loop("k", 0, 8, [unit, exit_early]))
    analysis = MotionAnalysis(stree)
    assert analysis.reordering(unit, stree.children[0], [exit_early], downward=True) is None


def test_reorder_statement_of_nest():
    """A statement split off from the perfect nest around it moves with copies of the loops."""
    statement = _tasklet("o = a", {"a": "x[i]"}, {"o": "t[i]"})
    unit_nest = _loop("i", 0, 8, [statement])
    stree = _pair(unit_nest, [_read("t[7]")])
    assert MotionAnalysis(stree).reordering(statement, stree, stree.children[1:], True) == Reordering((7, 7))


# ----------------------------------------------------------------------------------------------------------------------
# Moving statements to their readers
# ----------------------------------------------------------------------------------------------------------------------


def _moving_tree(blocker_writes: str = None) -> tn.ScheduleTreeRoot:
    """``for k, j, i: if U[k + 1, i, j] != 0: q -= U; dm = q * dp`` (fixes, then a value only the correction uses), then
    a nest that counts columns to correct in ``Z`` (and optionally writes ``q``), then the correction under ``Z > 0``."""
    stree = _root(
        "move_to_readers",
        {
            "q": ([8, 4, 4], dace.float64),
            "dp": ([8, 4, 4], dace.float64),
            "U": ([9, 4, 4], dace.float64),
            "Z": ([4, 4], dace.int32),
            "out": ([8, 4, 4], dace.float64),
        },
        {"dm": ([8, 4, 4], dace.float64)},
    )
    fix = _if(
        "U[k + 1, i, j] != 0", [_tasklet("o = x - u", {"x": "q[k, i, j]", "u": "U[k + 1, i, j]"}, {"o": "q[k, i, j]"})]
    )
    value = _tasklet("m = x * d", {"x": "q[k, i, j]", "d": "dp[k, i, j]"}, {"m": "dm[k, i, j]"})
    count = [_if("q[7, i, j] < 0", [_tasklet("o = z + 1", {"z": "Z[i, j]"}, {"o": "Z[i, j]"})])]
    if blocker_writes:
        count.append(_tasklet("o = x * 2", {"x": f"q[{blocker_writes}, i, j]"}, {"o": f"q[{blocker_writes}, i, j]"}))
    correct = _if("Z[i, j] > 0", [_tasklet("o = m", {"m": "dm[k, i, j]"}, {"o": "out[k, i, j]"})])
    stree.add_children(
        [
            _loop("k", 0, 8, [_loop("j", 0, 4, [_loop("i", 0, 4, [fix, value])])]),
            _loop("j", 0, 4, [_loop("i", 0, 4, count)]),
            _loop("k", 0, 8, [_loop("j", 0, 4, [_loop("i", 0, 4, [correct])])]),
        ]
    )
    return stree


def _moving_inputs() -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(4)
    result = []
    for negative in (False, True):
        q = rng.random((8, 4, 4))
        if negative:
            q[7, 1, 2] = -1.0
        u = np.zeros((9, 4, 4))
        u[3, 0, 1] = 0.5
        result.append(
            {
                "q": q,
                "dp": rng.random((8, 4, 4)),
                "U": u,
                "Z": np.zeros((4, 4), dtype=np.int32),
                "out": np.zeros((8, 4, 4)),
            }
        )
    return result


def test_move_past_guard_assignment_then_sink():

    def transform(stree):
        assert move_statements_to_readers(stree) == 1
        assert sink_into_guards(stree) == 1
        # The fixes (now alone in their nest), the moved value and the correction
        assert coarsen_guards(stree, levels=("nest",)) == 3

    stree = _same_results(_moving_tree, transform, _moving_inputs())
    # The value is computed after the counting nest (which writes Z), under the guard
    order = [c for c in stree.children if type(c) in (tn.ForScope, tn.IfScope)]
    writes = lambda node, name: any(
        m.data == name for n in node.preorder_traversal() if type(n) is tn.TaskletNode for m in n.out_memlets.values()
    )
    counting = next(k for k, c in enumerate(order) if writes(c, "Z"))
    value = next(k for k, c in enumerate(order) if writes(c, "dm"))
    assert counting < value
    guarded = [n for n in stree.preorder_traversal() if type(n) is tn.TaskletNode and writes(n, "dm")]
    assert all(
        any(type(a) is tn.IfScope and a.condition.as_string.startswith("(Z") for a in _ancestors(n)) for n in guarded
    )


def test_move_keeps_conflicting_iterations():
    """The counting nest also writes ``q[7]``: the last iteration of ``k`` stays before it."""

    def transform(stree):
        assert move_statements_to_readers(stree) == 1
        headers = [c.loop.loop_condition.as_string for c in stree.children if type(c) is tn.ForScope]
        assert any("7" in h for h in headers[1:])

    _same_results(lambda: _moving_tree(blocker_writes="7"), transform, _moving_inputs())


def test_move_not_past_writes_of_every_iteration():
    stree = _moving_tree(blocker_writes="0:8")
    assert move_statements_to_readers(stree) == 0


if __name__ == "__main__":
    test_reorder_unrelated_nests()
    test_reorder_reads_do_not_conflict()
    test_reorder_not_past_reader_of_every_iteration()
    test_reorder_not_past_writer_of_inputs()
    test_reorder_not_past_writer_of_outputs()
    test_reorder_keeps_first_iterations()
    test_reorder_keeps_last_iterations_if_independent()
    test_reorder_up_keeps_last_iterations()
    test_reorder_descending_loop()
    test_reorder_not_if_middle_iterations_conflict()
    test_reorder_conflicts_in_conditions()
    test_reorder_not_past_symbol_assignment()
    test_reorder_loop_variable_used_after()
    test_reorder_not_past_side_effects()
    test_reorder_not_past_jump()
    test_reorder_statement_of_nest()
    test_move_past_guard_assignment_then_sink()
    test_move_keeps_conflicting_iterations()
    test_move_not_past_writes_of_every_iteration()
