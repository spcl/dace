# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
import networkx as nx
import numpy as np
import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, ReturnBlock
from dace.transformation.dataflow import GPUTransformMap
from dace.transformation.optimizer import Optimizer
from dace.transformation.passes.offloading import offloading_helpers as helpers
from dace.transformation.passes.offloading.offload_to_accelerator import OffloadToAccelerator as OtA
from copy import deepcopy

# SDFGs for Tests


def scalar_to_gpu_sdfg():
    """
    in
    | x + 1 (CPU)
    v
    A
    | x * 2 (GPU)
    v
    out (CPU)

    computes out = 2*(in + 1) with an intermediate helper array A where +1 is on CPU, *2 is on GPU
    """
    sdfg = dace.SDFG("scalar_to_gpu_sdfg")
    # Arrays
    sdfg.add_array("in", [1], dace.float64)
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    # States + transition
    s1 = sdfg.add_state("s1", is_start_block=True)
    s2 = sdfg.add_state("s2")
    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    # s1: in -> A
    in_node = s1.add_access("in")
    a_s1 = s1.add_access("A")
    t1 = s1.add_tasklet("t1", {"x"}, {"y"}, "y = x + 1")
    s1.add_edge(in_node, None, t1, "x", dace.Memlet("in[0]"))
    s1.add_edge(t1, "y", a_s1, None, dace.Memlet("A[0]"))
    # s2: use A inside a map, write to out
    a_s2 = s2.add_access("A")
    out_s2 = s2.add_access("out")
    me, mx = s2.add_map("m", dict(i="0:1"))
    t2 = s2.add_tasklet("t2", {"a"}, {"y"}, "y = a * 2")
    s2.add_memlet_path(a_s2, me, t2, memlet=dace.Memlet("A[i]"), dst_conn="a")
    s2.add_memlet_path(t2, mx, out_s2, memlet=dace.Memlet("out[i]"), src_conn="y")
    sdfg.validate()
    return sdfg


def conditional_branch_map_sdfg():
    """
    Frontend-built SDFG with:
      - one symbolic conditional
      - a true branch that maps over the input array and writes back to it
      - a false branch that updates only the first element
      - a final map that copies input to output
    """

    @dace.program
    def conditional_branch_program(inp: dace.float64[5], out: dace.float64[5], flag: dace.int32):
        if flag > 0:
            for i in dace.map[0:5]:
                inp[i] = inp[i] + 1.0
        else:
            inp[0] = inp[0] - 1.0

        for i in dace.map[0:5]:
            out[i] = inp[i]

    sdfg = conditional_branch_program.to_sdfg()
    sdfg.validate()
    return sdfg


def scalar_to_gpu_within_loop_sdfg(num_iters: int = 4):
    """
    Build an SDFG where states `s1` and `s2` are executed in a sequential
    interstate for-loop (`i = 0 .. num_iters-1`).

    Body per iteration:
        s1: in -> A      (A = in + 1)
        s2: A  -> out    (out = A * 2) using a map inside s2

    The loop control is interstate (sequential), not a map.
    """
    if num_iters < 1:
        raise ValueError("num_iters must be >= 1")

    sdfg = dace.SDFG("forloop_with_map")

    sdfg.add_array("in", [1], dace.float64)
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("out", [1], dace.float64)

    init = sdfg.add_state("loop_init", is_start_block=True)
    s1 = sdfg.add_state("s1")
    s2 = sdfg.add_state("s2")
    after = sdfg.add_state("after_loop")

    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    sdfg.add_loop(init, s1, after, "i", "0", f"i < {num_iters}", "i + 1", loop_end_state=s2)

    in_node = s1.add_access("in")
    a_s1 = s1.add_access("A")
    t1 = s1.add_tasklet("t1", {"x"}, {"y"}, "y = x + 1")
    s1.add_edge(in_node, None, t1, "x", dace.Memlet("in[0]"))
    s1.add_edge(t1, "y", a_s1, None, dace.Memlet("A[0]"))

    a_s2 = s2.add_access("A")
    out_s2 = s2.add_access("out")
    me, mx = s2.add_map("m", dict(j="0:1"))
    t2 = s2.add_tasklet("t2", {"a"}, {"y"}, "y = a * 2")
    s2.add_memlet_path(a_s2, me, t2, memlet=dace.Memlet("A[j]"), dst_conn="a")
    s2.add_memlet_path(t2, mx, out_s2, memlet=dace.Memlet("out[j]"), src_conn="y")

    sdfg.validate()
    return sdfg


def scalar_to_gpu_within_loopregion_sdfg(num_iters: int = 4):
    """
    Build an SDFG where `s1` and `s2` are enclosed in a LoopRegion-based
    sequential for-loop (`i = 0 .. num_iters-1`).

    Loop body:
        s1: in -> A      (A = in + 1)
        s2: A  -> out    (out = A * 2) using a map in s2
    """
    if num_iters < 1:
        raise ValueError("num_iters must be >= 1")

    sdfg = dace.SDFG("scalar_to_gpu_loopregion")
    sdfg.using_explicit_control_flow = True

    sdfg.add_symbol("i", dace.int32)
    sdfg.add_array("in", [1], dace.float64)
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("out", [1], dace.float64)

    before = sdfg.add_state("before_loop", is_start_block=True)
    after = sdfg.add_state("after_loop")

    loop = LoopRegion(label="for_region",
                      condition_expr=f"i < {num_iters}",
                      loop_var="i",
                      initialize_expr="i = 0",
                      update_expr="i = i + 1",
                      inverted=False)
    sdfg.add_node(loop)
    sdfg.add_edge(before, loop, dace.InterstateEdge())
    sdfg.add_edge(loop, after, dace.InterstateEdge())

    s1 = loop.add_state("s1")
    s2 = loop.add_state("s2")
    loop.add_edge(s1, s2, dace.InterstateEdge())

    in_node = s1.add_access("in")
    a_s1 = s1.add_access("A")
    t1 = s1.add_tasklet("t1", {"x"}, {"y"}, "y = x + 1")
    s1.add_edge(in_node, None, t1, "x", dace.Memlet("in[0]"))
    s1.add_edge(t1, "y", a_s1, None, dace.Memlet("A[0]"))

    a_s2 = s2.add_access("A")
    out_s2 = s2.add_access("out")
    me, mx = s2.add_map("m", dict(j="0:1"))
    t2 = s2.add_tasklet("t2", {"a"}, {"y"}, "y = a * 2")
    s2.add_memlet_path(a_s2, me, t2, memlet=dace.Memlet("A[j]"), dst_conn="a")
    s2.add_memlet_path(t2, mx, out_s2, memlet=dace.Memlet("out[j]"), src_conn="y")

    sdfg.validate()
    return sdfg


def nested_sdfg():

    @dace.program
    def nested_kernel_program(inp: dace.float64[5], out: dace.float64[5]):
        for idx in dace.map[0:5]:
            tmp = dace.define_local([1], dace.float64)

            for phase in range(2):
                if phase == 0:
                    tmp[0] = inp[idx] + 1.0
                else:
                    out[idx] = tmp[0] * 2.0

    sdfg = nested_kernel_program.to_sdfg()
    sdfg.validate()
    return sdfg


def kernel_sdfg():
    TS = dace.symbol("TS")

    @dace.program
    def example(A: dace.float64[100, 100], B: dace.float64[100, 100], C: dace.float64[100, 100],
                D: dace.float64[100, 100], E: dace.float64[100]) -> dace.float64[100, 100]:
        for t1 in range(TS):
            for i, j in dace.map[0:100, 0:100]:
                C[i, j] = A[i, j] + B[i, j]
        for t2 in range(2):
            for j in range(100):
                for i in dace.map[0:100]:
                    E[i] = E[i] + C[i, j]
            for i in range(1, 100):
                E[i] = (E[i - 1] + E[i]) / 100.0
        for t3 in range(2):
            for i, j in dace.map[0:100, 0:100]:
                D[i, j] = E[i] * 2.0 + C[i, j]

    return example.to_sdfg()


def edge_assignment_sdfg():
    # State edge: edge.data.data = used array name
    #             edge.data.is_empty() -> no array
    # Interstate edge: edge.condition, edge.assignments as python code
    #                  edge.used_arrays(sdfg.arrays, True) -> all arrays used by edge
    # Note: Interstate assignment LHS must be a symbol name (not an array access).
    #       Array accesses are allowed on the RHS (e.g., "k = A[0]").

    sdfg = dace.SDFG("edge_condition_sdfg")

    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_symbol("k", dace.int32)

    s1 = sdfg.add_state("s1", is_start_block=True)
    s2 = sdfg.add_state("s2")

    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={"k": "A[0]"}))

    r1 = s1.add_read("A")
    w1 = s1.add_write("A")
    me1, mx1 = s1.add_map("m1", dict(i="0:4"))
    t1 = s1.add_tasklet("add_one", {"a"}, {"b"}, "b = a + 1")
    s1.add_memlet_path(r1, me1, t1, memlet=dace.Memlet("A[i]"), dst_conn="a")
    s1.add_memlet_path(t1, mx1, w1, memlet=dace.Memlet("A[i]"), src_conn="b")

    r2 = s2.add_read("A")
    w2 = s2.add_write("A")
    me2, mx2 = s2.add_map("m2", dict(j="0:4"))
    t2 = s2.add_tasklet("mul_two", {"a"}, {"b"}, "b = a * 2")
    s2.add_memlet_path(r2, me2, t2, memlet=dace.Memlet("A[j]"), dst_conn="a")
    s2.add_memlet_path(t2, mx2, w2, memlet=dace.Memlet("A[j]"), src_conn="b")

    sdfg.validate()
    return sdfg


def tasklet_map_wrapper_sdfg():

    @dace.program
    def tasklet_map_wrapper_program(A: dace.float64[4, 4], out: dace.float64[4, 4]):
        out = A @ A
        out[0, 0] += 1

    sdfg = tasklet_map_wrapper_program.to_sdfg()
    sdfg.validate()
    return sdfg


def tasklet_map_wrapper_larger_sdfg():

    @dace.program
    def tasklet_map_wrapper_program(A: dace.float64[4, 4], B: dace.float64[4, 4], out: dace.float64[4, 4]):
        B = A @ A  # parallel

        B[0, 0] += A[0, 0]  # sequential region
        A[1, 0] += 5

        B = B @ A  # parallel

        s = 5  # sequential region
        B[0, 3] += s
        out[1, 1] += s

    sdfg = tasklet_map_wrapper_program.to_sdfg()
    sdfg.validate()
    return sdfg


def scalar_init_sdfg():

    @dace.program
    def scalar_init_program(alpha: dace.float64, A: dace.float64[16], out: dace.float64[16]):
        A[0] = alpha  # - 1.0
        A[2] = alpha  # + 1.0

        for i in dace.map[0:16]:
            out[i] = A[i] * 2.0 + 1.0

    sdfg = scalar_init_program.to_sdfg()
    sdfg.validate()
    return sdfg


def len1_array_init_sdfg():

    @dace.program
    def len1_array_init(alpha: dace.float64[1], A: dace.float64[16], out: dace.float64[16]):
        alpha[0] = 3.0

        for i in dace.map[0:16]:
            out[i] = A[i] * 2.0 + alpha[0]

    sdfg = len1_array_init.to_sdfg()
    sdfg.validate()
    return sdfg


def reduce_to_scalar_sdfg(n: int = 16):
    sdfg = dace.SDFG("reduction_library_node")
    state = sdfg.add_state("state", is_start_block=True)

    sdfg.add_array("inp", [n], dace.float64)
    sdfg.add_scalar("red_scalar", dace.float64, transient=True)
    sdfg.add_array("out", [1], dace.float64)

    inp = state.add_access("inp")
    red_scalar = state.add_access("red_scalar")
    out = state.add_access("out")
    red = state.add_reduce("lambda a, b: a + b", axes=(0, ), identity=0)

    # Through the library node's own connectors: an ``add_nedge`` leaves ``_in`` / ``_out``
    # dangling, which is not a graph the reduction expansions can read.
    state.add_edge(inp, None, red, '_in', dace.Memlet(f"inp[0:{n}]"))
    state.add_edge(red, '_out', red_scalar, None, dace.Memlet("red_scalar[0]"))
    state.add_nedge(red_scalar, out, dace.Memlet("red_scalar[0]"))

    sdfg.validate()
    return sdfg


def reduce_to_array_sdfg(n: int = 16):
    sdfg = dace.SDFG("reduction_library_node")
    state = sdfg.add_state("state", is_start_block=True)

    sdfg.add_array("inp", [n], dace.float64)
    sdfg.add_transient("red_array", [1], dace.float64)
    sdfg.add_array("out", [1], dace.float64)

    inp = state.add_access("inp")
    red_array = state.add_access("red_array")
    out = state.add_access("out")
    red = state.add_reduce("lambda a, b: a + b", axes=(0, ), identity=0)

    # Through the library node's own connectors: an ``add_nedge`` leaves ``_in`` / ``_out``
    # dangling, which is not a graph the reduction expansions can read.
    state.add_edge(inp, None, red, '_in', dace.Memlet(f"inp[0:{n}]"))
    state.add_edge(red, '_out', red_array, None, dace.Memlet("red_array[0]"))
    state.add_nedge(red_array, out, dace.Memlet("red_array[0]"))

    sdfg.validate()
    return sdfg


def single_element_copy_sdfg():

    @dace.program
    def single_elements_map(A: dace.float64[16], B: dace.float64[16]):
        b = B[0]
        for i in dace.map[0:16]:
            A[i] = b * A[i]

    sdfg = single_elements_map.to_sdfg()
    sdfg.validate()
    return sdfg


# OFFLOADING TESTS


# helper
def run_numerical_offloading_test(sdfg, param_dict: dict, result_array1, result_array2, result_name="out"):
    # note: all parameters can be modified by this function
    # deepcopy before passing if previous state needs to be retained
    sdfg.validate()

    # compile and run sdfg without offloading (all on CPU)
    input1 = deepcopy(param_dict)
    input1[result_name] = result_array1
    sdfg(**input1)

    # offload sdfg (in place)
    OtA().apply_pass(sdfg, {})
    sdfg.validate()

    # compile and run offloaded sdfg (part may be on GPU, necessary copies were added)
    sdfg._recompile = True
    input2 = param_dict
    input2[result_name] = result_array2

    #print("PARAMS:", sdfg.arglist())
    sdfg(**input2)

    # assert the results are equal
    assert np.allclose(result_array1, result_array2), f"{result_array1} != {result_array2}"


@pytest.mark.gpu
def test_cpu_scalars_no_copies():

    def create_sdfg():
        sdfg = dace.SDFG("test_all_cpu_no_copy_needed")
        state = sdfg.add_state()

        sdfg.add_array("in", [1], dace.float64)  # input
        sdfg.add_array("out", [1], dace.float64)  # output
        sdfg.add_transient("A", [1], dace.float64)  # intermediate access node

        In = state.add_access("in")
        A = state.add_access("A")
        out = state.add_access("out")

        t1 = state.add_tasklet("comp1", {"x"}, {"a"}, "a = x + 1")
        t2 = state.add_tasklet("comp2", {"a"}, {"y"}, "y = a * 2")

        state.add_edge(In, None, t1, "x", dace.Memlet("in[0]"))
        state.add_edge(t1, "a", A, None, dace.Memlet("A[0]"))
        state.add_edge(A, None, t2, "a", dace.Memlet("A[0]"))
        state.add_edge(t2, "y", out, None, dace.Memlet("out[0]"))

        sdfg.validate()
        return sdfg

    sdfg = create_sdfg()
    input = 3490.2378
    orig_output = np.array([0.0])
    new_output = np.array([0.0])
    run_numerical_offloading_test(sdfg, {"in": np.array([input]), "A": np.array([0.0])}, orig_output, new_output)


@pytest.mark.gpu
def test_copy_scalar_to_gpu_and_back():
    sdfg = scalar_to_gpu_sdfg()
    """
    must copy out & A to GPU before the 2nd state
    must copy out and & A back to CPU after the last state
    NOTE: possible optimization: first copy of A and out not necessary: write only
    """

    input = -5678.0
    orig_output = np.array([0.0])
    new_output = np.array([0.0])
    run_numerical_offloading_test(sdfg, {"in": np.array([input]), "A": np.array([0.0])}, orig_output, new_output)


@pytest.mark.gpu
def test_loopregion_offload():
    sdfg = scalar_to_gpu_within_loopregion_sdfg()

    input = 4321.1234
    orig_output = np.array([0.0])
    new_output = np.array([0.0])
    run_numerical_offloading_test(sdfg, {"in": np.array([input]), "A": np.array([0.0])}, orig_output, new_output)


@pytest.mark.gpu
def test_conditional_offload_if():
    sdfg = conditional_branch_map_sdfg()

    orig_output = np.zeros(5, dtype=np.float64)
    new_output = np.zeros(5, dtype=np.float64)
    run_numerical_offloading_test(
        sdfg,
        {
            "inp": np.arange(5, dtype=np.float64),
            "flag": np.int32(0)
        },  # run with flag == 0
        orig_output,
        new_output,
    )


@pytest.mark.gpu
def test_conditional_offload_else():
    sdfg = conditional_branch_map_sdfg()

    orig_output = np.zeros(5, dtype=np.float64)
    new_output = np.zeros(5, dtype=np.float64)
    run_numerical_offloading_test(
        sdfg,
        {
            "inp": np.arange(5, dtype=np.float64),
            "flag": np.int32(1)
        },  # run with flag == 1
        orig_output,
        new_output,
    )


@pytest.mark.gpu
def test_nested_sdfg():
    sdfg = nested_sdfg()

    orig_output = np.zeros(5, dtype=np.float64)
    new_output = np.zeros(5, dtype=np.float64)
    run_numerical_offloading_test(
        sdfg,
        {"inp": np.arange(5, dtype=np.float64)},
        orig_output,
        new_output,
    )


@pytest.mark.gpu
def test_kernel_sdfg():
    sdfg = kernel_sdfg()
    orig_output = np.zeros((100, 100), dtype=np.float64)
    new_output = np.zeros((100, 100), dtype=np.float64)

    A = np.arange(10000, dtype=np.float64).reshape(100, 100) / 1000.0
    B = (np.arange(10000, dtype=np.float64).reshape(100, 100) % 97) / 97.0
    C = np.zeros((100, 100), dtype=np.float64)
    E = np.arange(100, 200, dtype=np.float64)

    run_numerical_offloading_test(
        sdfg,
        {
            "A": A,
            "B": B,
            "C": C,
            "E": E,
            "TS": np.int32(3),
        },
        orig_output,
        new_output,
        result_name="D",
    )


@pytest.mark.gpu
def test_edge_assignment_sdfg():
    sdfg = edge_assignment_sdfg()
    orig_A = np.array([1.0, -2.0, 3.5, 0.25], dtype=np.float64)
    new_A = orig_A.copy()

    run_numerical_offloading_test(
        sdfg,
        {},
        orig_A,
        new_A,
        result_name="A",
    )


@pytest.mark.gpu
def test_tasklet_map_wrapper():
    sdfg = tasklet_map_wrapper_sdfg()

    A = np.arange(16, dtype=np.float64).reshape(4, 4) / 10.0
    orig_out = np.zeros((4, 4), dtype=np.float64)
    new_out = np.zeros((4, 4), dtype=np.float64)

    run_numerical_offloading_test(
        sdfg,
        {"A": A},
        orig_out,
        new_out,
    )


@pytest.mark.gpu
def test_tasklet_map_wrapper_larger():
    sdfg = tasklet_map_wrapper_larger_sdfg()

    A = np.arange(16, dtype=np.float64).reshape(4, 4) / 10.0
    B = np.zeros((4, 4), dtype=np.float64)
    orig_out = np.zeros((4, 4), dtype=np.float64)
    new_out = np.zeros((4, 4), dtype=np.float64)

    run_numerical_offloading_test(
        sdfg,
        {
            "A": A,
            "B": B
        },
        orig_out,
        new_out,
    )


@pytest.mark.gpu
def test_scalar_init():
    sdfg = scalar_init_sdfg()

    alpha = np.float64(3.5)
    A = np.zeros(16, dtype=np.float64)
    orig_out = np.zeros(16, dtype=np.float64)
    new_out = np.zeros(16, dtype=np.float64)

    run_numerical_offloading_test(
        sdfg,
        {
            "alpha": alpha,
            "A": A
        },
        orig_out,
        new_out,
    )


@pytest.mark.gpu
def test_len1_array_init():
    sdfg = len1_array_init_sdfg()

    alpha = np.ones(1, dtype=np.float64)
    A = np.zeros(16, dtype=np.float64)
    orig_out = np.zeros(16, dtype=np.float64)
    new_out = np.zeros(16, dtype=np.float64)

    run_numerical_offloading_test(
        sdfg,
        {
            "alpha": alpha,
            "A": A
        },
        orig_out,
        new_out,
    )


@pytest.mark.gpu
def test_reduce_to_array():
    sdfg = reduce_to_array_sdfg()
    inp = np.arange(16, dtype=np.float64) + 1.0
    orig_out = np.zeros(1, dtype=np.float64)
    new_out = np.zeros(1, dtype=np.float64)

    run_numerical_offloading_test(
        sdfg,
        {"inp": inp},
        orig_out,
        new_out,
    )


@pytest.mark.gpu
def test_reduce_to_scalar():
    sdfg = reduce_to_scalar_sdfg()
    inp = np.arange(16, dtype=np.float64) + 1.0
    orig_out = np.zeros(1, dtype=np.float64)
    new_out = np.zeros(1, dtype=np.float64)

    run_numerical_offloading_test(
        sdfg,
        {"inp": inp},
        orig_out,
        new_out,
    )


@pytest.mark.gpu
def test_single_element_copy():
    sdfg = single_element_copy_sdfg()
    A = np.arange(16, dtype=np.float64)
    B = np.arange(16, dtype=np.float64) + 5.0
    orig_out = np.zeros(1, dtype=np.float64)
    new_out = np.zeros(1, dtype=np.float64)

    run_numerical_offloading_test(
        sdfg,
        {
            "A": A,
            "B": B
        },
        orig_out,
        new_out,
    )


def device_map_state_sdfg(host_writer_between: bool) -> dace.SDFG:
    """Two device states, and ``B`` first touched on the device in the second of them.

    With ``host_writer_between`` a host state writing ``B`` sits between the two, which is what
    pins ``B``'s copy to that point instead of letting it move up.
    """
    sdfg = dace.SDFG("device_map_states_" + ("pinned" if host_writer_between else "free"))
    sdfg.add_array("A", [20], dace.float64)
    sdfg.add_array("B", [20], dace.float64)

    first = sdfg.add_state("scale_a", is_start_block=True)
    entry, exit_ = first.add_map("scale", dict(i="0:20"))
    scale = first.add_tasklet("scale", {"x"}, {"y"}, "y = x * 2.0")
    first.add_memlet_path(first.add_read("A"), entry, scale, dst_conn="x", memlet=dace.Memlet("A[i]"))
    first.add_memlet_path(scale, exit_, first.add_write("A"), src_conn="y", memlet=dace.Memlet("A[i]"))

    previous = first
    if host_writer_between:
        seed = sdfg.add_state("seed_b")
        sdfg.add_edge(previous, seed, dace.InterstateEdge())
        one = seed.add_tasklet("one", {}, {"y"}, "y = 1.0")
        seed.add_edge(one, "y", seed.add_write("B"), None, dace.Memlet("B[0]"))
        previous = seed

    second = sdfg.add_state("fill_b")
    sdfg.add_edge(previous, second, dace.InterstateEdge())
    entry, exit_ = second.add_map("fill", dict(i="0:20"))
    fill = second.add_tasklet("fill", {"x"}, {"y"}, "y = x + 1.0")
    second.add_memlet_path(second.add_read("A"), entry, fill, dst_conn="x", memlet=dace.Memlet("A[i]"))
    second.add_memlet_path(fill, exit_, second.add_write("B"), src_conn="y", memlet=dace.Memlet("B[i]"))

    sdfg.fill_scope_connectors()
    sdfg.validate()
    return sdfg


def states_in_execution_order(sdfg: dace.SDFG) -> list:
    return list(nx.topological_sort(sdfg.nx))


def holds_device_map(state: dace.SDFGState) -> bool:
    return any(
        isinstance(node, nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device
        for node in state.nodes())


def writes_container(state: dace.SDFGState, name: str) -> bool:
    return any(node.data == name and state.in_degree(node) > 0 for node in state.data_nodes())


def test_a_device_copy_is_hoisted_above_the_states_that_do_not_touch_the_array():
    """Staging ``B`` must not split the device states.

    ``B`` is only touched in the second of two device states, so a copy that puts it on the device
    naively lands between them -- a host state in the middle of a run of kernels, which is exactly
    what a caller fusing that run into one persistent kernel cannot swallow. Here the second map
    writes ALL of ``B`` and nothing reads it first, so there is no stage-down to place at all; the
    companion test covers the case where a host writer forces one to exist.
    """
    sdfg = device_map_state_sdfg(host_writer_between=False)
    OtA().apply_pass(sdfg, {})

    order = states_in_execution_order(sdfg)
    device_at = [index for index, state in enumerate(order) if holds_device_map(state)]
    assert len(device_at) == 2, f"expected both maps on the device, got {[s.label for s in order]}"
    assert device_at == list(range(device_at[0], device_at[-1] +
                                   1)), (f"a host state sits between two device states: {[s.label for s in order]}")
    staged = [index for index, state in enumerate(order) if writes_container(state, "B_gpu")]
    assert all(
        index in device_at
        for index in staged), (f"B was staged down although the device writes all of it: {[s.label for s in order]}")


def test_a_device_copy_stays_below_a_host_state_that_writes_the_array():
    """The hoist is only free where the array is untouched -- a host writer keeps the copy below it."""
    sdfg = device_map_state_sdfg(host_writer_between=True)
    OtA().apply_pass(sdfg, {})

    order = states_in_execution_order(sdfg)
    labels = [state.label for state in order]
    seeded_at = next(index for index, state in enumerate(order) if state.label == "seed_b")
    copied_at = next(index for index, state in enumerate(order) if writes_container(state, "B_gpu"))
    assert copied_at > seeded_at, f"B was copied to the device before the host wrote it: {labels}"


def read_only_input_sdfg() -> dace.SDFG:
    """``A`` is read and never written, ``B`` is written -- the two halves of the copy-back rule.

    ``C`` is touched by nothing at all, the case that must not be placed anywhere.
    """
    sdfg = dace.SDFG("read_only_input")
    sdfg.add_array("A", [20], dace.float64)
    sdfg.add_array("B", [20], dace.float64)
    sdfg.add_array("C", [20], dace.float64)

    state = sdfg.add_state("scale", is_start_block=True)
    entry, exit_ = state.add_map("scale", dict(i="0:20"))
    scale = state.add_tasklet("scale", {"x"}, {"y"}, "y = x * 2.0")
    state.add_memlet_path(state.add_read("A"), entry, scale, dst_conn="x", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(scale, exit_, state.add_write("B"), src_conn="y", memlet=dace.Memlet("B[i]"))

    sdfg.fill_scope_connectors()
    sdfg.validate()
    return sdfg


N = dace.symbol("N")


@dace.program
def laplace_program(A: dace.float64[N], T: dace.int64):
    tmp = np.zeros_like(A)
    for _ in range(T):
        for i in dace.map[1:N - 1]:
            tmp[i] = A[i - 1] - 2 * A[i] + A[i + 1]
        for i in dace.map[1:N - 1]:
            A[i] = tmp[i - 1] - 2 * tmp[i] + tmp[i + 1]


def test_a_never_written_input_is_not_copied_back_to_the_host():
    """Only a container something wrote is restored; the twin that makes it readable still stands.

    A container never moves -- a twin is staged on the other side and the accesses are pointed at
    it -- so its home copy goes stale only once something writes the twin. Restoring a read-only
    one copies bytes already in place, and that write is what a nested SDFG's input-only connector
    refuses.
    """
    sdfg = read_only_input_sdfg()
    OtA().apply_pass(sdfg, {})

    states = list(sdfg.states())
    labels = [state.label for state in states]
    assert any(writes_container(state, "A_gpu") for state in states), \
        f"the read-only array was not staged onto the device at all: {labels}"
    assert any(writes_container(state, "B") for state in states), \
        f"the written array was not copied back: {labels}"
    assert not any(writes_container(state, "A") for state in states), \
        f"the read-only array was copied back: {labels}"


def test_a_container_nothing_touches_is_left_where_it_started():
    """An array no state reads or writes is not staged, not copied, and grows no twin."""
    sdfg = read_only_input_sdfg()
    OtA().apply_pass(sdfg, {})

    assert "C_gpu" not in sdfg.arrays, f"an untouched array was given a device twin: {sorted(sdfg.arrays)}"
    assert sdfg.arrays["C"].storage == dace.StorageType.Default, \
        f"an untouched array was moved off the host: {sdfg.arrays['C'].storage}"
    assert not any(writes_container(state, "C") for state in sdfg.states()), \
        "an untouched array was copied"


def test_a_map_over_a_read_only_container_survives_being_nested():
    """``GPUTransformMap`` nests one map and offloads it, so each container reaches it one-way.

    laplace is the shape that exposes it: one map reads ``A`` and writes ``tmp``, the next reads
    ``tmp`` and writes ``A``, so whichever map is nested holds one container it never writes.
    Copying that one back writes through a connector the nested SDFG only has as an input, which
    validation refuses: "Data descriptor A is written to, but only given to nested SDFG as an
    input connector".
    """
    sdfg = laplace_program.to_sdfg()
    matches = list(Optimizer(sdfg).get_pattern_matches(patterns=[GPUTransformMap]))
    assert matches, "no map to offload -- the fixture no longer exercises the transformation"

    for match in matches:
        candidate = deepcopy(sdfg)
        cfg = candidate.cfg_list[match.cfg_id]
        target = cfg.sdfg if not isinstance(cfg, dace.SDFG) else cfg
        graph = cfg.node(match.state_id) if match.state_id >= 0 else cfg
        match._sdfg = target
        match.apply(graph, target)
        candidate.validate()


def view_on_both_sides_sdfg() -> dace.SDFG:
    """``C_view`` aliases ``C``, and the two are read on different sides of the machine.

    npbench mandelbrot2 has this shape: one state reads the view inside a kernel, another reads it
    from host code, and a view carries one storage.
    """
    sdfg = dace.SDFG("view_on_both_sides")
    sdfg.add_array("C", [4, 5], dace.float64)
    sdfg.add_array("out", [20], dace.float64)
    sdfg.add_array("total", [1], dace.float64)
    sdfg.add_view("C_view", [20], dace.float64)

    device = sdfg.add_state("on_the_device", is_start_block=True)
    flat = device.add_access("C_view")
    device.add_edge(device.add_read("C"), None, flat, "views", dace.Memlet("C[0:4, 0:5]"))
    entry, exit_ = device.add_map("scale", dict(i="0:20"))
    scale = device.add_tasklet("scale", {"x"}, {"y"}, "y = x * 2.0")
    device.add_memlet_path(flat, entry, scale, dst_conn="x", memlet=dace.Memlet("C_view[i]"))
    device.add_memlet_path(scale, exit_, device.add_write("out"), src_conn="y", memlet=dace.Memlet("out[i]"))

    host = sdfg.add_state("on_the_host")
    sdfg.add_edge(device, host, dace.InterstateEdge())
    host_flat = host.add_access("C_view")
    host.add_edge(host.add_read("C"), None, host_flat, "views", dace.Memlet("C[0:4, 0:5]"))
    pick = host.add_tasklet("pick", {"x"}, {"y"}, "y = x")
    host.add_edge(host_flat, None, pick, "x", dace.Memlet("C_view[0]"))
    host.add_edge(pick, "y", host.add_write("total"), None, dace.Memlet("total[0]"))

    sdfg.fill_scope_connectors()
    sdfg.validate()
    return sdfg


def test_a_view_of_a_staged_container_is_staged_with_it():
    """A view follows the container it aliases onto whichever side that container was staged to.

    Left behind, the alias names a buffer on the other side and the dispatcher refuses the access it
    cannot make ("Illegal copy!"), because one descriptor carries one storage and the view is read
    from a kernel in one state and from host code in another.
    """
    sdfg = view_on_both_sides_sdfg()
    OtA().apply_pass(sdfg, {})
    sdfg.validate()

    views = {name: desc for name, desc in sdfg.arrays.items() if isinstance(desc, dace.data.View)}
    assert len(views) > 1, f"the view was not staged with its container: {sorted(views)}"

    for state in sdfg.states():
        for node in state.data_nodes():
            if node.data not in views:
                continue
            origin = helpers.view_origin(state, node)
            assert origin is not None, f"{node.data} in {state.label} aliases nothing"
            assert views[node.data].storage == sdfg.arrays[origin].storage, (
                f"{node.data} ({views[node.data].storage}) does not live where "
                f"{origin} ({sdfg.arrays[origin].storage}) does")


def two_arm_branch_sdfg(arms_meet: bool, big_arm_padding: int = 0) -> dace.SDFG:
    """A device map doubles ``A`` into ``t``, and a condition on ``t[0]`` picks one of two device maps writing ``A``.

    With ``arms_meet`` both arms flow into an empty ``end`` state; without it each arm is a sink of
    the control flow, so the program has two exits. ``big_arm_padding`` empty states follow ``big``.
    """
    sdfg = dace.SDFG("two_arm_branch_" + ("meeting" if arms_meet else "sinks") + f"_{big_arm_padding}")
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("t", [8], dace.float64, transient=True)
    fill = sdfg.add_state("fill", is_start_block=True)
    fill.add_mapped_tasklet("double",
                            dict(i="0:8"), {"inp": dace.Memlet("A[i]")},
                            "out = inp * 2.0", {"out": dace.Memlet("t[i]")},
                            external_edges=True)
    arms = []
    for label, condition, offset in (("big", "t[0] > 5.0", "1.0"), ("small", "not (t[0] > 5.0)", "-1.0")):
        arm = sdfg.add_state(label)
        arm.add_mapped_tasklet(label,
                               dict(i="0:8"), {"inp": dace.Memlet("t[i]")},
                               f"out = inp + {offset}", {"out": dace.Memlet("A[i]")},
                               external_edges=True)
        sdfg.add_edge(fill, arm, dace.InterstateEdge(condition=condition))
        arms.append(arm)
    for index in range(big_arm_padding):
        padding = sdfg.add_state(f"big_padding_{index}")
        sdfg.add_edge(arms[0], padding, dace.InterstateEdge())
        arms[0] = padding
    if arms_meet:
        end = sdfg.add_state("end")
        for arm in arms:
            sdfg.add_edge(arm, end, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("arms_meet", [False, True], ids=["arms_are_sinks", "arms_meet_in_an_end_state"])
def test_every_exit_of_a_device_branch_copies_the_written_array_back(arms_meet: bool) -> None:
    """Each way out of the program restores ``A``, not only the exit the IR visited last.

    With one sink per arm, the copy-back landed after ``small`` alone and the ``big`` arm returned
    the caller's array untouched.
    """
    sdfg = two_arm_branch_sdfg(arms_meet)
    OtA().apply_pass(sdfg, {})

    labels = [state.label for state in sdfg.states()]
    arms = [state for state in sdfg.states() if state.label in ("big", "small")]
    assert len(arms) == 2 and all(holds_device_map(arm) for arm in arms), f"an arm was not offloaded: {labels}"
    paths = [path for sink in sdfg.sink_nodes() for path in nx.all_simple_paths(sdfg.nx, sdfg.start_block, sink)]
    assert len(paths) == 2, f"expected one way out per arm, got {[[s.label for s in p] for p in paths]}"
    stale_exits = [[state.label for state in path] for path in paths
                   if not any(writes_container(state, "A") for state in path)]
    assert not stale_exits, f"these exits never copy A back to the host: {stale_exits}"


@pytest.mark.gpu
@pytest.mark.parametrize("arms_meet", [False, True], ids=["arms_are_sinks", "arms_meet_in_an_end_state"])
def test_a_device_branch_returns_the_result_of_the_arm_it_took(arms_meet: bool) -> None:
    """3.0 doubles to 6.0 and takes ``big`` (+1 -> 7.0); 0.5 doubles to 1.0 and takes ``small`` (-1 -> 0.0)."""
    sdfg = two_arm_branch_sdfg(arms_meet)
    OtA().apply_pass(sdfg, {})
    sdfg.validate()
    compiled = sdfg.compile()

    took_big = np.full(8, 3.0)
    compiled(A=took_big)
    took_small = np.full(8, 0.5)
    compiled(A=took_small)

    np.testing.assert_array_equal(took_big, np.full(8, 7.0))
    np.testing.assert_array_equal(took_small, np.full(8, 0.0))


def test_the_exit_joining_two_sink_arms_does_not_outlive_the_pass() -> None:
    """The copy-back follows the join, so the join is spliced out and no empty state is left behind."""
    sdfg = two_arm_branch_sdfg(arms_meet=False)
    OtA().apply_pass(sdfg, {})

    empty = [state.label for state in sdfg.states() if state.number_of_nodes() == 0]
    assert not empty, f"the pass left empty states behind: {empty}"


@pytest.mark.parametrize("arms_meet", [False, True], ids=["arms_are_sinks", "arms_meet_in_an_end_state"])
def test_the_short_arm_of_an_uneven_device_branch_copies_the_written_array_back(arms_meet: bool) -> None:
    """BFS reaches the exit through ``small`` before the end of the long ``big`` arm; both ways out still restore ``A``."""
    sdfg = two_arm_branch_sdfg(arms_meet, big_arm_padding=2)
    OtA().apply_pass(sdfg, {})

    paths = [path for sink in sdfg.sink_nodes() for path in nx.all_simple_paths(sdfg.nx, sdfg.start_block, sink)]
    assert len(paths) == 2, f"expected one way out per arm, got {[[s.label for s in p] for p in paths]}"
    stale_exits = [[state.label for state in path] for path in paths
                   if not any(writes_container(state, "A") for state in path)]
    assert not stale_exits, f"these exits never copy A back to the host: {stale_exits}"


def host_only_two_arm_branch_sdfg() -> dace.SDFG:
    """A condition on ``A[0]`` picks one of two host tasklets writing ``A[0]``; each arm is a sink."""
    sdfg = dace.SDFG("host_only_two_arm_branch")
    sdfg.add_array("A", [4], dace.float64)
    fill = sdfg.add_state("fill", is_start_block=True)
    for label, condition, value in (("big", "A[0] > 5.0", "1.0"), ("small", "not (A[0] > 5.0)", "-1.0")):
        arm = sdfg.add_state(label)
        write = arm.add_tasklet(label, {}, {"y"}, f"y = {value}")
        arm.add_edge(write, "y", arm.add_write("A"), None, dace.Memlet("A[0]"))
        sdfg.add_edge(fill, arm, dace.InterstateEdge(condition=condition))
    sdfg.validate()
    return sdfg


def test_a_host_only_branch_keeps_its_arms_as_the_exits() -> None:
    """Nothing is copied, so the pass takes the join it added back out and each arm still ends the program."""
    sdfg = host_only_two_arm_branch_sdfg()
    OtA().apply_pass(sdfg, {})

    assert sorted(state.label for state in sdfg.sink_nodes()) == ["big", "small"]
    assert sorted(state.label for state in sdfg.states()) == ["big", "fill", "small"]


def early_return_sdfg(return_after_arm: bool) -> dace.SDFG:
    """A device map doubles ``A`` into ``A`` and ``t``; if ``t[0] > 5`` the program returns early, else ``small`` runs.

    With ``return_after_arm`` a device map ``big`` adds 1 to ``A`` before the return; without it the
    condition jumps straight to the return. ``small`` subtracts 1 from ``A`` and falls through to the end.
    """
    sdfg = dace.SDFG("early_return_" + ("after_arm" if return_after_arm else "direct"))
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("t", [8], dace.float64, transient=True)
    fill = sdfg.add_state("fill", is_start_block=True)
    fill.add_mapped_tasklet("double",
                            dict(i="0:8"), {"inp": dace.Memlet("A[i]")},
                            "out_a = inp * 2.0\nout_t = inp * 2.0", {
                                "out_a": dace.Memlet("A[i]"),
                                "out_t": dace.Memlet("t[i]")
                            },
                            external_edges=True)
    early = sdfg.add_return("early")
    big_condition = dace.InterstateEdge(condition="t[0] > 5.0")
    if return_after_arm:
        big = sdfg.add_state("big")
        big.add_mapped_tasklet("big",
                               dict(i="0:8"), {"inp": dace.Memlet("t[i]")},
                               "out = inp + 1.0", {"out": dace.Memlet("A[i]")},
                               external_edges=True)
        sdfg.add_edge(fill, big, big_condition)
        sdfg.add_edge(big, early, dace.InterstateEdge())
    else:
        sdfg.add_edge(fill, early, big_condition)
    small = sdfg.add_state("small")
    small.add_mapped_tasklet("small",
                             dict(i="0:8"), {"inp": dace.Memlet("t[i]")},
                             "out = inp - 1.0", {"out": dace.Memlet("A[i]")},
                             external_edges=True)
    sdfg.add_edge(fill, small, dace.InterstateEdge(condition="not (t[0] > 5.0)"))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("return_after_arm", [True, False], ids=["return_after_a_device_arm", "return_on_the_branch"])
def test_an_early_return_copies_the_written_array_back_first(return_after_arm: bool) -> None:
    """The return leaves the program, so ``A`` must reach the host on the way to it, not only at the end."""
    sdfg = early_return_sdfg(return_after_arm)
    OtA().apply_pass(sdfg, {})

    early = next(block for block in sdfg.nodes() if isinstance(block, ReturnBlock))
    paths = list(nx.all_simple_paths(sdfg.nx, sdfg.start_block, early))
    assert paths, "the return is no longer reachable"
    stale = [[block.label for block in path] for path in paths
             if not any(writes_container(block, "A") for block in path if isinstance(block, dace.SDFGState))]
    assert not stale, f"these paths return without copying A back to the host: {stale}"


@pytest.mark.gpu
@pytest.mark.parametrize("return_after_arm", [True, False], ids=["return_after_a_device_arm", "return_on_the_branch"])
def test_an_early_return_hands_back_the_device_result(return_after_arm: bool) -> None:
    """3.0 doubles to 6.0 and returns (+1 -> 7.0 after ``big``); 0.5 doubles to 1.0 and takes ``small`` (-1 -> 0.0)."""
    sdfg = early_return_sdfg(return_after_arm)
    OtA().apply_pass(sdfg, {})
    sdfg.validate()
    compiled = sdfg.compile()

    returned = np.full(8, 3.0)
    compiled(A=returned)
    fell_through = np.full(8, 0.5)
    compiled(A=fell_through)

    np.testing.assert_array_equal(returned, np.full(8, 7.0 if return_after_arm else 6.0))
    np.testing.assert_array_equal(fell_through, np.full(8, 0.0))


def test_the_state_before_an_early_return_does_not_outlive_the_pass() -> None:
    """The copy-back follows the state the pass puts before the return; nothing empty is left behind."""
    sdfg = early_return_sdfg(return_after_arm=True)
    OtA().apply_pass(sdfg, {})

    empty = [state.label for state in sdfg.states() if state.number_of_nodes() == 0]
    assert not empty, f"the pass left empty states behind: {empty}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
