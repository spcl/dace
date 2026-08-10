"""
Automatic test suite for DaCe GPU offloading pass using pytest.

Usage:
    pytest testsuite_offloading.py                   # Run all tests
    pytest testsuite_offloading.py -v                # Verbose output
    pytest testsuite_offloading.py -m basic          # Run only 'basic' group
    pytest testsuite_offloading.py -m copy_insertion # Run only 'copy_insertion' group
    pytest testsuite_offloading.py::test_my_sdfg     # Run specific test
"""

import pytest
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.offloading.OffloadToAccelerator import OffloadToAccelerator as OtA
from dace.transformation import pass_pipeline as ppl
from copy import deepcopy

# ============================================================================
# Test Groups (pytest markers)
# ============================================================================
# Run tests in a group: pytest -m basic, pytest -m copy_insertion, etc.

pytest.mark.basic = pytest.mark.basic
pytest.mark.copy_analysis = pytest.mark.copy_analysis
pytest.mark.gpu_offload = pytest.mark.gpu_offload
pytest.mark.current = pytest.mark.current

# ============================================================================
# SDFGS (reused ones)
# ============================================================================

def simple_no_copy_sdfg():
    sdfg = dace.SDFG("simple_sdfg")
    state = sdfg.add_state()
    
    sdfg.add_array("X", [1], dace.float64)
    sdfg.add_array("Y", [1], dace.float64)
    
    X = state.add_access("X")
    Y = state.add_access("Y")
    
    t = state.add_tasklet("add_one", {"x"}, {"y"}, "y = x + 1")
    state.add_edge(X, None, t, "x", dace.Memlet("X[0]"))
    state.add_edge(t, "y", Y, None, dace.Memlet("Y[0]"))
    
    sdfg.validate()
    return sdfg

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
    def example(A: dace.float64[100, 100], B: dace.float64[100, 100], C: dace.float64[100, 100], D: dace.float64[100, 100], E: dace.float64[100]) -> dace.float64[100, 100]:
        for t1 in range(TS):
            for i, j in dace.map[0:100, 0:100]:
                C[i, j] = A[i, j] + B[i, j]
        for t2 in range(2):
            for j in range(100):
                for i in dace.map[0:100]:
                    E[i] = E[i] + C[i, j]
            for i in range(1, 100):
                E[i] = (E[i-1] + E[i]) / 100.0
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


def edge_condition_sdfg():
    pass

# ============================================================================
# BASIC TESTS (single-state SDFGs, no offloading)
# ============================================================================



@pytest.mark.basic
def test_simple_sdfg_basic():
    """Test that a basic single-state SDFG runs without offloading."""
    sdfg = simple_no_copy_sdfg()
    output = np.array([0.0])
    sdfg(X=np.array([5.0]), Y=output)
    
    assert np.allclose(output, np.array([6.0])), f"Expected 6.0, got {output[0]}"


@pytest.mark.basic
def test_simple_map_cpu():
    """Test that a simple CPU map executes correctly."""
    sdfg = dace.SDFG("simple_map_cpu")
    state = sdfg.add_state()
    
    n = 5
    sdfg.add_array("X", [n], dace.float64)
    sdfg.add_array("Y", [n], dace.float64)
    
    X = state.add_access("X")
    Y = state.add_access("Y")
    
    me, mx = state.add_map("m", dict(i=f"0:{n}"))
    t = state.add_tasklet("mul_two", {"x"}, {"y"}, "y = x * 2")
    
    state.add_memlet_path(X, me, t, memlet=dace.Memlet("X[i]"), dst_conn="x")
    state.add_memlet_path(t, mx, Y, memlet=dace.Memlet("Y[i]"), src_conn="y")
    
    sdfg.validate()
    
    input_arr = np.arange(n, dtype=np.float64)
    output = np.zeros(n, dtype=np.float64)
    sdfg(X=input_arr, Y=output)
    
    expected = input_arr * 2
    assert np.allclose(output, expected), f"Expected {expected}, got {output}"

# ============================================================================
# COPY ANALYSIS TESTS
# create the IR of the sdfgs
# ensure it has found all arrays and has categorized them correctly
# ============================================================================

# helper
def test_IR(IR, expected:dict):
    for node in IR.next:
        if node.block:
            name = node.block.label
            if name in expected:
                cpu = expected[name]["cpu"]
                gpu = expected[name]["gpu"]
                assert node.cpu_set == cpu, f"{name} failed on cpu set:\nexpected = {expected}, actual = {node}"
                assert node.gpu_set == gpu, f"{name} failed on gpu set:\nexpected = {expected}, actual = {node}"

        test_IR(node, expected)


@pytest.mark.copy_analysis
def test_all_arrays_on_cpu():
    def create_sdfg():
        sdfg = dace.SDFG("minimal_chain_multiuse_A")
        state = sdfg.add_state()
        # I/O
        sdfg.add_array("X", [1], dace.float64)
        sdfg.add_array("Y", [1], dace.float64)
        # intermediates
        sdfg.add_transient("A", [1], dace.float64)
        sdfg.add_transient("B", [1], dace.float64)
        sdfg.add_transient("C", [1], dace.float64)
        sdfg.add_transient("D", [1], dace.float64)
        X = state.add_access("X")
        A = state.add_access("A")
        B = state.add_access("B")
        C = state.add_access("C")
        D = state.add_access("D")
        Y = state.add_access("Y")
        # original two computations
        t1 = state.add_tasklet("comp1", {"x"}, {"a"}, "a = x + 1")
        t2 = state.add_tasklet("comp2", {"a"}, {"b"}, "b = a * 2")
        # new computations, all reading A
        t3 = state.add_tasklet("comp3", {"a"}, {"c"}, "c = a + 3")
        t4 = state.add_tasklet("comp4", {"a"}, {"d"}, "d = a - 1")
        # final merge into Y
        t5 = state.add_tasklet("merge", {"b", "c", "d"}, {"y"}, "y = (b + c + d) / 3.0")
        # X -> t1 -> A
        state.add_edge(X, None, t1, "x", dace.Memlet("X[0]"))
        state.add_edge(t1, "a", A, None, dace.Memlet("A[0]"))
        # A -> (t2, t3, t4)
        state.add_edge(A, None, t2, "a", dace.Memlet("A[0]"))
        state.add_edge(A, None, t3, "a", dace.Memlet("A[0]"))
        state.add_edge(A, None, t4, "a", dace.Memlet("A[0]"))
        # (t2, t3, t4) -> (B, C, D)
        state.add_edge(t2, "b", B, None, dace.Memlet("B[0]"))
        state.add_edge(t3, "c", C, None, dace.Memlet("C[0]"))
        state.add_edge(t4, "d", D, None, dace.Memlet("D[0]"))
        # (B, C, D) -> t5 -> Y
        state.add_edge(B, None, t5, "b", dace.Memlet("B[0]"))
        state.add_edge(C, None, t5, "c", dace.Memlet("C[0]"))
        state.add_edge(D, None, t5, "d", dace.Memlet("D[0]"))
        state.add_edge(t5, "y", Y, None, dace.Memlet("Y[0]"))
        return sdfg
    
    IR = OtA().get_IR(create_sdfg())
    expected = {
        "block": {
            "cpu" : set(["X", "A", "B", "C", "D", "Y"]),
            "gpu" : set()
        }
    }

    test_IR(IR, expected)

@pytest.mark.copy_analysis
def test_cpu_then_gpu():
    sdfg = scalar_to_gpu_sdfg()
    IR = OtA().get_IR(sdfg)

    expected = {
        "s1": {
            "cpu" : set(["A", "in", "out"]),
            "gpu" : set()
        },
        "s2": {
            "cpu" : set(["in"]),
            "gpu" : set(["A", "out"])
        },
    }

    test_IR(IR, expected)

    rep = "\
_SDFG_head:                       cpu = ['A', 'in', 'out'], gpu = []\n\
_SDFG_head => s1:                 cpu = ['A', 'in', 'out'], gpu = []\n\
s1 => s2:                         cpu = ['in'], gpu = ['A', 'out']\n\
s2 => _SDFG_tail:                 cpu = ['A', 'in', 'out'], gpu = []\n"
    assert rep == f"{IR}"

        
@pytest.mark.copy_analysis
def test_manual_loop():
    # Q: not a line graph -> support or not?
    sdfg = scalar_to_gpu_within_loop_sdfg()
    #sdfg.view()

    IR = OtA().get_IR(sdfg)
    print(IR)

    assert False

@pytest.mark.copy_analysis
def test_loopregion():
    sdfg = scalar_to_gpu_within_loopregion_sdfg()
    IR = OtA().get_IR(sdfg)
    
    rep = "\
_SDFG_head:                       cpu = ['A', 'in', 'out'], gpu = []\n\
_SDFG_head => before_loop:        cpu = ['A', 'in', 'out'], gpu = []\n\
before_loop => _loop_head:        cpu = ['A', 'in', 'out'], gpu = []\n\
_loop_head => s1:                 cpu = ['A', 'in', 'out'], gpu = []\n\
s1 => s2:                         cpu = ['in'], gpu = ['A', 'out']\n\
s2 => _loop_head:                 cpu = ['A', 'in', 'out'], gpu = []\n\
s2 => _loop_tail:                 cpu = ['in'], gpu = ['A', 'out']\n\
_loop_tail => after_loop:         cpu = ['in'], gpu = ['A', 'out']\n\
after_loop => _SDFG_tail:         cpu = ['A', 'in', 'out'], gpu = []\n"

    ir = f"{IR}"
    for i in range(len(ir)):
        assert ir[i] == rep[i], f"{i}: {ir[i]} != {rep[i]}\n{ir}"
    assert f"{IR}" == rep

@pytest.mark.copy_analysis
def test_branch_with_cpu_else_gpu():
    sdfg = conditional_branch_map_sdfg()
    IR = OtA().get_IR(sdfg)
    print(IR)
    #sdfg.view()

def test_nested_IR():
    sdfg = nested_sdfg()
    sdfg.view()
    IR = OtA().get_IR(sdfg)
    print(IR)

# ============================================================================
# OFFLOADING TESTS
# ============================================================================

# helper
def run_numerical_offloading_test(sdfg, param_dict:dict, result_array1, result_array2, result_name="out"): 
    # note: all parameters can be modified by this function
    # deepcopy before passing if previous state needs to be retained
    sdfg.validate()
    #sdfg.view()

    # compile and run sdfg without offloading (all on CPU)
    input1 = deepcopy(param_dict)
    input1[result_name] = result_array1
    sdfg(**input1) 
    
    # offload sdfg (in place)
    OtA().apply_pass(sdfg, {})
    sdfg.validate()
    sdfg.view()

    # compile and run offloaded sdfg (part may be on GPU, necessary copies were added)
    sdfg._recompile = True
    input2 = param_dict
    input2[result_name] = result_array2

    #print("PARAMS:", sdfg.arglist())
    sdfg(**input2)
    
    # assert the results are equal
    assert np.allclose(result_array1, result_array2),f"{result_array1} != {result_array2}"


@pytest.mark.gpu_offload
def test_cpu_scalars_no_copies():
    def create_sdfg():
        sdfg = dace.SDFG("test_all_cpu_no_copy_needed")
        state = sdfg.add_state()

        sdfg.add_array("in", [1], dace.float64)      # input
        sdfg.add_array("out", [1], dace.float64)      # output
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
    run_numerical_offloading_test(sdfg, {"in": np.array([input]), "A":np.array([0.0])}, orig_output, new_output)

@pytest.mark.gpu_offload
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
    run_numerical_offloading_test(sdfg, {"in": np.array([input]), "A":np.array([0.0])}, orig_output, new_output)

@pytest.mark.gpu_offload
def test_loopregion_offload():
    # not pretty but it passes
    # structural issue to be solved later
    # I want to see whether it shows up elsewhere too -> refactor or not -> patch
    sdfg = scalar_to_gpu_within_loopregion_sdfg()
    
    input = 4321.1234
    orig_output = np.array([0.0])
    new_output = np.array([0.0])
    run_numerical_offloading_test(sdfg, {"in": np.array([input]), "A":np.array([0.0])}, orig_output, new_output)

@pytest.mark.gpu_offload
def test_conditional_offload_if():
    sdfg = conditional_branch_map_sdfg()

    orig_output = np.zeros(5, dtype=np.float64)
    new_output = np.zeros(5, dtype=np.float64)
    run_numerical_offloading_test(
        sdfg,
        {"inp": np.arange(5, dtype=np.float64), "flag": np.int32(0)}, # run with flag == 0
        orig_output,
        new_output,
    )

@pytest.mark.gpu_offload
def test_conditional_offload_else():
    sdfg = conditional_branch_map_sdfg()
    
    orig_output = np.zeros(5, dtype=np.float64)
    new_output = np.zeros(5, dtype=np.float64)
    run_numerical_offloading_test(
        sdfg,
        {"inp": np.arange(5, dtype=np.float64), "flag": np.int32(1)}, # run with flag == 1
        orig_output,
        new_output,
    )

@pytest.mark.gpu_offload
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

@pytest.mark.gpu_offload
def test_kernel_sdfg():
    sdfg = kernel_sdfg()
    #sdfg.view()
    orig_output = np.zeros((100, 100), dtype=np.float64)
    new_output = np.zeros((100, 100), dtype=np.float64)

    A = np.arange(10000, dtype=np.float64).reshape(100, 100) / 1000.0
    B = (np.arange(10000, dtype=np.float64).reshape(100, 100) % 97) / 97.0
    C = np.zeros((100, 100), dtype=np.float64)
    E = np.arange(100,200, dtype=np.float64)

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

@pytest.mark.gpu_offload
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

def tasklet_map_wrapper_sdfg():
    @dace.program
    def tasklet_map_wrapper_program(A: dace.float64[4, 4], out: dace.float64[4, 4]):
        out = A @ A
        out[0, 0] += 1

    sdfg = tasklet_map_wrapper_program.to_sdfg()
    sdfg.validate()
    return sdfg

@pytest.mark.gpu_offload
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

def tasklet_map_wrapper_larger_sdfg():
    @dace.program
    def tasklet_map_wrapper_program(A: dace.float64[4, 4], B: dace.float64[4, 4], out: dace.float64[4, 4]):
        B = A @ A # parallel

        B[0,0] += A[0,0] # sequential region
        A[1,0] += 5

        B = B @ A # parallel

        s = 5 # sequential region
        B[0,3] += s
        out[1,1] += s

    sdfg = tasklet_map_wrapper_program.to_sdfg()
    sdfg.validate()
    return sdfg

@pytest.mark.gpu_offload
def test_tasklet_map_wrapper_larger():
    sdfg = tasklet_map_wrapper_larger_sdfg()

    A = np.arange(16, dtype=np.float64).reshape(4, 4) / 10.0
    B = np.zeros((4, 4), dtype=np.float64)
    orig_out = np.zeros((4, 4), dtype=np.float64)
    new_out = np.zeros((4, 4), dtype=np.float64)

    run_numerical_offloading_test(
        sdfg,
        {"A": A, "B": B},
        orig_out,
        new_out,
    )


def scalar_init_sdfg():
    @dace.program
    def scalar_init_program(alpha: dace.float64, A: dace.float64[16], out: dace.float64[16]):
        A[0] = alpha# - 1.0
        A[2] = alpha# + 1.0

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

    state.add_nedge(inp, red, dace.Memlet(f"inp[0:{n}]"))
    state.add_nedge(red, red_scalar, dace.Memlet("red_scalar[0]"))
    state.add_nedge(red_scalar, out, dace.Memlet("red_scalar[0]"))

    #sdfg.validate()
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

    state.add_nedge(inp, red, dace.Memlet(f"inp[0:{n}]"))
    state.add_nedge(red, red_array, dace.Memlet("red_array[0]"))
    state.add_nedge(red_array, out, dace.Memlet("red_array[0]"))

    sdfg.validate()
    return sdfg

@pytest.mark.gpu_offload
def test_scalar_init():
    sdfg = scalar_init_sdfg()

    alpha = np.float64(3.5)
    A = np.zeros(16, dtype=np.float64)
    orig_out = np.zeros(16, dtype=np.float64)
    new_out = np.zeros(16, dtype=np.float64)

    run_numerical_offloading_test(
        sdfg,
        {"alpha": alpha, "A": A},
        orig_out,
        new_out,
    )

@pytest.mark.gpu_offload
def test_len1_array_init():
    sdfg = len1_array_init_sdfg()

    alpha = np.ones(1, dtype=np.float64)
    A = np.zeros(16, dtype=np.float64)
    orig_out = np.zeros(16, dtype=np.float64)
    new_out = np.zeros(16, dtype=np.float64)

    run_numerical_offloading_test(
        sdfg,
        {"alpha": alpha, "A": A},
        orig_out,
        new_out,
    )


@pytest.mark.current
@pytest.mark.gpu_offload
def test_reduce_to_array():
    sdfg = reduce_to_array_sdfg()
    sdfg.view()
    inp = np.arange(16, dtype=np.float64) + 1.0
    orig_out = np.zeros(1, dtype=np.float64)
    new_out = np.zeros(1, dtype=np.float64)

    run_numerical_offloading_test(
        sdfg,
        {"inp": inp},
        orig_out,
        new_out,
    )

@pytest.mark.current
@pytest.mark.gpu_offload
def test_reduce_to_scalar():
    sdfg = reduce_to_scalar_sdfg()
    sdfg.view()
    inp = np.arange(16, dtype=np.float64) + 1.0
    orig_out = np.zeros(1, dtype=np.float64)
    new_out = np.zeros(1, dtype=np.float64)

    run_numerical_offloading_test(
        sdfg,
        {"inp": inp},
        orig_out,
        new_out,
    )
    
# ============================================================================
# Fixtures and Helpers
# ============================================================================

@pytest.fixture
def cleanup_sdfgs():
    """Fixture to ensure SDFG cleanup between tests."""
    yield
    # Any cleanup logic here if needed
    pass


# ============================================================================
# Conftest-style marker registration (if running standalone)
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "basic: mark test as part of basic SDFG suite")
    config.addinivalue_line("markers", "copy_analysis: mark test as copy analysis suite")
    config.addinivalue_line("markers", "gpu_offload: mark test as GPU offload suite")
    config.addinivalue_line("markers", "current: tests of current interest")


if __name__ == "__main__":
    
    # Run with: python testsuite_offloading.py
    #pytest.main([__file__, "-s", "-v", "--tb=short", "-m", "current"])
    pytest.main([__file__, "-v", "--tb=short", "-m", "gpu_offload"])

    """
    import importlib
    import sys
    from pathlib import Path

    workspace_root = Path(__file__).resolve().parents[6]
    npbench_repo_root = workspace_root / "npbench"
    if str(npbench_repo_root) not in sys.path:
        sys.path.insert(0, str(npbench_repo_root))

    module = importlib.import_module("npbench.benchmarks.azimint_hist.azimint_hist_dace")
    sdfg = module.azimint_hist.to_sdfg()
    sdfg.view()
    OtA().apply_pass(sdfg, {})
    sdfg.validate()
    sdfg.view()
    """
    
    
