# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests that leaf copy edges are given the correct direction when lowering an SDFG to a schedule tree.

A ``CopyNode`` names the destination container as its ``target`` and normalizes its memlet to the source container
(with ``other_subset`` holding the destination subset). This must hold regardless of the direction of the underlying
memlet path: inward (outer array into an in-scope node), outward (in-scope node out through one or more scope exits into
an outer array), same-scope, and write-conflict-resolution (WCR) copies. A previous implementation inverted outward
copies, naming the source as the target and normalizing to the destination.
"""

import numpy as np
import pytest

import dace
from dace.sdfg.analysis.schedule_tree import sdfg_to_tree, tree_to_sdfg
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol("N")
M = dace.symbol("M")


def _can_compile() -> bool:
    """Returns True if the DaCe CPU toolchain can compile a trivial SDFG in this environment."""
    try:
        sdfg = dace.SDFG("probe_copy_direction")
        sdfg.add_array("A", [1], dace.float64)
        state = sdfg.add_state()
        tasklet = state.add_tasklet("w", {}, {"out"}, "out = 1.0")
        state.add_edge(tasklet, "out", state.add_write("A"), None, dace.Memlet("A[0]"))
        sdfg.compile()
        return True
    except Exception:  # noqa: BLE001
        return False


def _assert_roundtrip_matches(sdfg: dace.SDFG, args: dict, tag: str) -> None:
    """Rebuilds ``sdfg`` from its schedule tree and asserts both compute identical outputs for ``args``.

    ``tag`` makes the program names unique per test so that compiled builds are not shared between tests that
    construct SDFGs of the same name.
    """
    tree = sdfg_to_tree.as_schedule_tree(sdfg)
    rebuilt = tree_to_sdfg.from_schedule_tree(tree)
    sdfg.name = f"{sdfg.name}_{tag}"
    rebuilt.name = f"{sdfg.name}_roundtrip"

    expected = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in args.items()}
    actual = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in args.items()}
    sdfg.compile()(**expected)
    rebuilt.compile()(**actual)

    for name, value in args.items():
        if isinstance(value, np.ndarray):
            assert np.allclose(actual[name], expected[name]), f'Output "{name}" differs after roundtrip'


@dace.program
def _outward_copy(A: dace.float64[N], B: dace.float64[N, 3]):
    for i in dace.map[0:N]:
        tmp = np.empty([3], dtype=np.float64)
        for j in dace.map[0:3]:
            tmp[j] = A[i] * (j + 1)
        B[i, :] = tmp


@dace.program
def _nested_outward_copy(A: dace.float64[N], B: dace.float64[N, M, 3]):
    for i in dace.map[0:N]:
        for k in dace.map[0:M]:
            tmp = np.empty([3], dtype=np.float64)
            for j in dace.map[0:3]:
                tmp[j] = A[i] * (j + 1) + k
            B[i, k, :] = tmp


def _inward_and_wcr_sdfg() -> dace.SDFG:
    """
    Builds an SDFG with one in-scope transient written by an inward copy and read out by an outward WCR copy.

    The map ``i`` copies ``A[i, 0:3]`` into an in-scope transient ``tmp`` (inward), then accumulates ``tmp[0:3]`` into
    the outer array ``B[0:3]`` through the map exit with a summation WCR (outward).
    """
    sdfg = dace.SDFG("inward_and_wcr")
    sdfg.add_array("A", [N, 3], dace.float64)
    sdfg.add_array("B", [3], dace.float64)
    sdfg.add_transient("tmp", [3], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("outer", dict(i="0:N"))
    read_a = state.add_read("A")
    tmp = state.add_access("tmp")
    write_b = state.add_write("B")
    state.add_memlet_path(read_a, entry, tmp,
                          memlet=dace.Memlet("A[i, 0:3]", other_subset=dace.subsets.Range([(0, 2, 1)])))
    state.add_memlet_path(tmp, exit_, write_b,
                          memlet=dace.Memlet("B[0:3]", wcr="lambda a, b: a + b",
                                             other_subset=dace.subsets.Range([(0, 2, 1)])))
    sdfg.validate()
    return sdfg


def _copy_nodes(tree: tn.ScheduleTreeScope) -> list[tn.CopyNode]:
    """Returns all ``CopyNode`` instances in a schedule tree, in preorder."""
    return [node for node in tree.preorder_traversal() if type(node) is tn.CopyNode]


def test_outward_copy_direction():
    """An outward copy through a map exit targets the outer destination array, reading the in-scope transient."""
    sdfg = _outward_copy.to_sdfg(simplify=True)
    tree = sdfg_to_tree.as_schedule_tree(sdfg)
    lines = tree.as_string()
    assert "B[i, 0:3] = copy tmp[0:3]" in lines
    assert "tmp = copy B[i, 0:3]" not in lines

    copies = _copy_nodes(tree)
    outward = next(c for c in copies if c.target == "B")
    assert outward.memlet.data == "tmp"

    if not _can_compile():
        pytest.skip("DaCe compilation is unavailable in this environment")
    _assert_roundtrip_matches(sdfg, dict(A=np.random.default_rng(0).random(5), B=np.zeros((5, 3)), N=5), "outward")


def test_inward_copy_direction():
    """An inward copy into an in-scope transient targets the transient, reading the outer source array."""
    sdfg = _inward_and_wcr_sdfg()
    tree = sdfg_to_tree.as_schedule_tree(sdfg)
    lines = tree.as_string()
    assert "tmp = copy A[i, 0:3]" in lines

    inward = next(c for c in _copy_nodes(tree) if c.target == "tmp")
    assert inward.memlet.data == "A"

    if not _can_compile():
        pytest.skip("DaCe compilation is unavailable in this environment")
    _assert_roundtrip_matches(sdfg, dict(A=np.random.default_rng(1).random((4, 3)), B=np.zeros(3), N=4), "inward")


def test_outward_wcr_copy_direction():
    """An outward WCR copy targets the outer destination array and carries its conflict resolution."""
    sdfg = _inward_and_wcr_sdfg()
    tree = sdfg_to_tree.as_schedule_tree(sdfg)
    lines = tree.as_string()
    assert "B = copy tmp[0:3] with lambda a, b: a + b" in lines

    wcr = next(c for c in _copy_nodes(tree) if c.target == "B")
    assert wcr.memlet.data == "tmp"
    assert wcr.memlet.wcr is not None

    if not _can_compile():
        pytest.skip("DaCe compilation is unavailable in this environment")
    _assert_roundtrip_matches(sdfg, dict(A=np.random.default_rng(2).random((4, 3)), B=np.zeros(3), N=4), "outward_wcr")


def test_nested_outward_copy_direction():
    """An outward copy through two nested map exits targets the outer destination array."""
    sdfg = _nested_outward_copy.to_sdfg(simplify=True)
    tree = sdfg_to_tree.as_schedule_tree(sdfg)
    lines = tree.as_string()
    assert "B[i, k, 0:3] = copy tmp[0:3]" in lines

    outward = next(c for c in _copy_nodes(tree) if c.target == "B")
    assert outward.memlet.data == "tmp"

    if not _can_compile():
        pytest.skip("DaCe compilation is unavailable in this environment")
    _assert_roundtrip_matches(
        sdfg, dict(A=np.random.default_rng(3).random(5), B=np.zeros((5, 2, 3)), N=5, M=2), "nested_outward"
    )


if __name__ == "__main__":
    test_outward_copy_direction()
    test_inward_copy_direction()
    test_outward_wcr_copy_direction()
    test_nested_outward_copy_direction()
