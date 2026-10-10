# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``pure`` expansions of ``ArgReduce``, ``Scan`` and ``ScatterConflictCheck`` are SDFG components -- loop
regions and Python tasklets, no C++ -- and compute exactly what the C++ ``CPU`` lowering computes.

The ``CPU`` lowering stays what every speed-driven choice picks: the ``Auto`` default on a parallel schedule
and CPF's rendering.
"""

import numpy as np
import pytest

import dace
from dace.codegen import cpf
from dace.libraries.sort.nodes.integer_sort import IntegerSort
from dace.libraries.sort.nodes.scatter_conflict_check import ScatterConflictCheck
from dace.libraries.standard.nodes.arg_reduce import ArgReduce
from dace.libraries.standard.nodes.find_first import FindFirst
from dace.libraries.standard.nodes.scan import (
    COEF_CONNECTOR_NAME,
    INIT_CONNECTOR_NAME,
    Scan,
    ScanOp,
    in_connector,
    out_connector,
)
from dace.sdfg.state import LoopRegion
from dace.sdfg.tasklet_utils import is_abort_guard
from dace.transformation.auto.auto_optimize import set_fast_implementations

N = dace.symbol("N")
M = dace.symbol("M")
LENGTH = 23
SEED = 7


def expanded(sdfg: dace.SDFG) -> dace.SDFG:
    sdfg.expand_library_nodes()
    return sdfg


def tasklet_languages(sdfg: dace.SDFG) -> set:
    return {n.code.language for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)}


def loop_count(sdfg: dace.SDFG) -> int:
    return sum(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions(recursive=True))


def arg_reduce_sdfg(implementation: str, op: str, stride: int) -> dace.SDFG:
    sdfg = dace.SDFG(f"argreduce_{implementation}_{op}_{stride}")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("val", [1], dace.float64)
    sdfg.add_array("idx", [1], dace.int64)
    state = sdfg.add_state()
    node = ArgReduce("argreduce", op=op, transform="abs")
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(state.add_read("a"), None, node, "_in", dace.Memlet(f"a[1:N:{stride}]"))
    state.add_edge(node, "_out_val", state.add_write("val"), None, dace.Memlet("val[0]"))
    state.add_edge(node, "_out_idx", state.add_write("idx"), None, dace.Memlet("idx[0]"))
    return expanded(sdfg)


def run_arg_reduce(sdfg: dace.SDFG, a: np.ndarray) -> tuple:
    val, idx = np.zeros(1), np.zeros(1, np.int64)
    sdfg(a=a.copy(), val=val, idx=idx, N=len(a))
    return val[0], idx[0]


@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("stride", [1, 3])
def test_the_pure_arg_reduce_is_a_python_loop_finding_the_first_extreme_the_cpu_one_finds(op, stride):
    ties = np.round(np.random.default_rng(SEED).standard_normal(3 * LENGTH))

    sut = arg_reduce_sdfg("pure", op, stride)

    assert tasklet_languages(sut) == {dace.Language.Python} and loop_count(sut) == 1
    assert run_arg_reduce(sut, ties) == run_arg_reduce(arg_reduce_sdfg("CPU", op, stride), ties)


def scan_sdfg(
    implementation: str,
    op: ScanOp,
    *,
    exclusive: bool = False,
    identity=None,
    stride: int = 1,
    chains: int = 1,
    init: bool = False,
) -> dace.SDFG:
    sdfg = dace.SDFG(f"scan_{implementation}_{op.value}_{exclusive}_{stride}_{chains}_{init}")
    state = sdfg.add_state()
    node = Scan("scan", op=op, exclusive=exclusive, identity=identity, chains=chains)
    node.stride = stride
    node.implementation = implementation
    state.add_node(node)
    for chain in range(chains):
        sdfg.add_array(f"x{chain}", [N], dace.float64)
        sdfg.add_array(f"y{chain}", [N], dace.float64)
        state.add_edge(state.add_read(f"x{chain}"), None, node, in_connector(chain), dace.Memlet(f"x{chain}[0:N]"))
        state.add_edge(node, out_connector(chain), state.add_write(f"y{chain}"), None, dace.Memlet(f"y{chain}[0:N]"))
    if op is ScanOp.AFFINE:
        sdfg.add_array("c", [N], dace.float64)
        state.add_edge(state.add_read("c"), None, node, COEF_CONNECTOR_NAME, dace.Memlet("c[0:N]"))
    if init:
        sdfg.add_array("seed", [stride], dace.float64)
        node.add_in_connector(INIT_CONNECTOR_NAME)
        state.add_edge(state.add_read("seed"), None, node, INIT_CONNECTOR_NAME, dace.Memlet(f"seed[0:{stride}]"))
    return expanded(sdfg)


def run_scan(sdfg: dace.SDFG) -> list:
    rng = np.random.default_rng(SEED)
    args = {name: rng.uniform(0.5, 1.5, LENGTH) for name in sdfg.arrays if name[0] in "xc"}
    if "seed" in sdfg.arrays:
        args["seed"] = rng.uniform(0.5, 1.5, sdfg.arrays["seed"].shape[0])
    outputs = {name: np.zeros(LENGTH) for name in sdfg.arrays if name[0] == "y"}
    sdfg(**args, **outputs, N=LENGTH)
    return [outputs[name] for name in sorted(outputs)]


SCANS = {
    "inclusive sum": dict(op=ScanOp.SUM),
    "inclusive min": dict(op=ScanOp.MIN),
    "exclusive max with identity": dict(op=ScanOp.MAX, exclusive=True, identity=-5),
    "inclusive product with init": dict(op=ScanOp.PRODUCT, init=True),
    "two chains": dict(op=ScanOp.SUM, chains=2),
    "strided sum": dict(op=ScanOp.SUM, stride=3),
    "strided max": dict(op=ScanOp.MAX, stride=4),
    "affine": dict(op=ScanOp.AFFINE, init=True),
    "strided affine": dict(op=ScanOp.AFFINE, stride=2, init=True),
}


@pytest.mark.parametrize("shape", sorted(SCANS))
def test_the_pure_scan_is_python_loops_computing_what_the_cpu_scan_computes(shape):
    sut = scan_sdfg("pure", **SCANS[shape])

    assert tasklet_languages(sut) == {dace.Language.Python} and loop_count(sut) >= 1
    for got, want in zip(run_scan(sut), run_scan(scan_sdfg("CPU", **SCANS[shape]))):
        np.testing.assert_allclose(got, want, rtol=1e-14)


def matrix_sdfg(scan_implementation: str, arg_implementation: str) -> dace.SDFG:
    """A Scan and an ArgReduce over a whole matrix, both of which walk it in row-major order."""
    sdfg = dace.SDFG(f"matrix_{arg_implementation}")
    sdfg.add_array("x", [M, M], dace.float64)
    sdfg.add_array("y", [M, M], dace.float64)
    sdfg.add_array("idx", [1], dace.int64)
    state = sdfg.add_state()
    scan = Scan("scan", op=ScanOp.MAX)
    arg = ArgReduce("argreduce", op="max")
    scan.implementation, arg.implementation = scan_implementation, arg_implementation
    state.add_node(scan)
    state.add_node(arg)
    read = state.add_read("x")
    state.add_edge(read, None, scan, in_connector(0), dace.Memlet("x[0:M, 0:M]"))
    state.add_edge(scan, out_connector(0), state.add_write("y"), None, dace.Memlet("y[0:M, 0:M]"))
    state.add_edge(read, None, arg, "_in", dace.Memlet("x[0:M, 0:M]"))
    state.add_edge(arg, "_out_idx", state.add_write("idx"), None, dace.Memlet("idx[0]"))
    return expanded(sdfg)


def run_matrix(sdfg: dace.SDFG) -> tuple:
    x = np.random.default_rng(SEED).standard_normal((5, 5))
    y, idx = np.zeros((5, 5)), np.zeros(1, np.int64)
    sdfg(x=x, y=y, idx=idx, M=5)
    return y, idx[0]


def test_a_matrix_operand_is_walked_in_row_major_order_like_the_cpu_lowering():
    got_y, got_idx = run_matrix(matrix_sdfg("pure", "pure"))
    want_y, want_idx = run_matrix(matrix_sdfg("CPU", "CPU"))

    np.testing.assert_array_equal(got_y, want_y)
    assert got_idx == want_idx


def test_the_pure_strided_scan_guards_its_stride_with_a_python_abort():
    sut = scan_sdfg("pure", ScanOp.SUM, stride=3)

    assert sum(is_abort_guard(n) for n, _ in sut.all_nodes_recursive()) == 1


def conflict_sdfg(implementation: str, owner: bool) -> dace.SDFG:
    sdfg = dace.SDFG(f"conflict_{implementation}_{owner}")
    sdfg.add_array("idx", [N], dace.int32)
    sdfg.add_array("count", [1], dace.int64)
    state = sdfg.add_state()
    node = ScatterConflictCheck("check")
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(
        state.add_read("idx"), None, node, ScatterConflictCheck.INPUT_CONNECTOR_NAME, dace.Memlet("idx[0:N]")
    )
    state.add_edge(
        node, ScatterConflictCheck.OUTPUT_CONNECTOR_NAME, state.add_write("count"), None, dace.Memlet("count[0]")
    )
    if owner:
        sdfg.add_array("owner", [M], dace.int64, transient=True)
        node.add_out_connector(ScatterConflictCheck.SCRATCH_CONNECTOR_NAME)
        state.add_edge(
            node, ScatterConflictCheck.SCRATCH_CONNECTOR_NAME, state.add_write("owner"), None, dace.Memlet("owner[0:M]")
        )
    return expanded(sdfg)


INDICES = {
    "a permutation": np.random.default_rng(SEED).permutation(LENGTH),
    "a duplicate": np.array([3, 1, 4, 1, 5]),
    "values outside the tag array": np.array([0, 30, 30, -1, 2]),
}


def run_conflict(sdfg: dace.SDFG, idx: np.ndarray) -> int:
    count = np.full(1, -1, np.int64)
    sdfg(idx=idx.astype(np.int32), count=count, N=len(idx), M=LENGTH + 1)
    return int(count[0])


@pytest.mark.parametrize("owner", [False, True])
@pytest.mark.parametrize("case", sorted(INDICES))
def test_the_pure_conflict_check_is_python_loops_flagging_what_the_cpu_check_flags(owner, case):
    sut = conflict_sdfg("pure", owner)

    assert tasklet_languages(sut) == {dace.Language.Python} and loop_count(sut) >= 2
    assert run_conflict(sut, INDICES[case]) == run_conflict(conflict_sdfg("CPU", owner), INDICES[case])


def test_a_duplicate_index_is_flagged_by_the_pure_check():
    assert run_conflict(conflict_sdfg("pure", True), INDICES["a duplicate"]) == 1


@pytest.mark.parametrize("node_type", [ArgReduce, Scan, ScatterConflictCheck])
def test_speed_driven_choices_keep_the_cpp_lowerings(node_type):
    node = node_type("node")
    renderable = cpf.renderable_implementations(node, dace.SDFGState())
    cpf_pick = next(impl for impl in renderable if impl in node_type.implementations)

    assert node_type.default_implementation == "Auto"
    assert cpf_pick == "CPU"


@pytest.mark.parametrize(
    "node_type, lowerings",
    [
        (ArgReduce, {"Auto", "CPU", "pure", "CUDA"}),
        (FindFirst, {"Auto", "CPU", "CUDA"}),
        (IntegerSort, {"CPU", "isocpp", "CUDA"}),
    ],
)
def test_cpu_names_the_parallel_cpp_lowering_and_pure_only_sdfg_components(node_type, lowerings):
    assert set(node_type.implementations) == lowerings
    assert node_type.default_implementation != "pure"


@pytest.mark.parametrize("node", [FindFirst("search", predicate="_a[__i] > 0", begin=0, end=M), IntegerSort("sort")])
def test_fast_implementations_prefer_the_cpu_lowering(node):
    sdfg = dace.SDFG(f"fast_{type(node).__name__}")
    state = sdfg.add_state()
    state.add_node(node)

    set_fast_implementations(sdfg, dace.DeviceType.CPU)

    assert node.implementation == "CPU"
