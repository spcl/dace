import dace
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState, ConditionalBlock, ControlFlowRegion, LoopRegion, ReturnBlock, ContinueBlock, BreakBlock, ControlFlowBlock
from dace import dtypes
from dace.transformation.passes.offloading.OffloadToAccelerator import OffloadToAccelerator as OtA



"""


sdfg = example.to_sdfg()
ota = OtA()
ota.set_toplevel_to_GPU(sdfg, nodes.MapEntry)
ota.set_toplevel_to_GPU(sdfg, nodes.LibraryNode)
ota.insert_copies_cfregion(sdfg, sdfg)

"""


"""
thesis
introduction: rephrase thesis globals
list contributions
related work: how do others lower things to GPU
implementation: document code and ask claude to summarize
evaluation: test against nbench, test sdfgs which failed previously
"""


# copy insertion tests (copy inserted)

# A -> add -> tmp(A) -> mul -> C
def simple_sdfg():
    sdfg = dace.SDFG("minimal_chain")
    state = sdfg.add_state()

    sdfg.add_array("X", [1], dace.float64)      # input
    sdfg.add_array("Y", [1], dace.float64)      # output
    sdfg.add_transient("A", [1], dace.float64)  # intermediate access node

    X = state.add_access("X")
    A = state.add_access("A")
    Y = state.add_access("Y")

    t1 = state.add_tasklet("comp1", {"x"}, {"a"}, "a = x + 1")
    t2 = state.add_tasklet("comp2", {"a"}, {"y"}, "y = a * 2")

    state.add_edge(X, None, t1, "x", dace.Memlet("X[0]"))
    state.add_edge(t1, "a", A, None, dace.Memlet("A[0]"))
    state.add_edge(A, None, t2, "a", dace.Memlet("A[0]"))
    state.add_edge(t2, "y", Y, None, dace.Memlet("Y[0]"))

    return sdfg, state, A

def simple_sdfg_with_copy():
    # Build SDFG
    sdfg = dace.SDFG("minimal_chain_with_copy")
    state = sdfg.add_state()

    sdfg.add_array("X", [1], dace.float64)            # input to computation 1
    sdfg.add_transient("A", [1], dace.float64)        # output of computation 1
    sdfg.add_transient("A_copy", [1], dace.float64)   # explicit copy of A
    sdfg.add_array("Y", [1], dace.float64)            # output of computation 2

    X = state.add_access("X")
    A = state.add_access("A")
    A_copy = state.add_access("A_copy")
    Y = state.add_access("Y")

    t1 = state.add_tasklet("comp1", {"x"}, {"a"}, "a = x + 1")
    t2 = state.add_tasklet("comp2", {"a"}, {"y"}, "y = a * 2")
    
    # computation 1: X -> A
    state.add_edge(X, None, t1, "x", dace.Memlet("X[0]"))
    state.add_edge(t1, "a", A, None, dace.Memlet("A[0]"))

    # explicit copy: A -> A_copy
    state.add_edge(A, None, A_copy, None, dace.Memlet("A[0] -> A_copy[0]"))

    # computation 2: A_copy -> Y
    state.add_edge(A_copy, None, t2, "a", dace.Memlet("A_copy[0]"))
    state.add_edge(t2, "y", Y, None, dace.Memlet("Y[0]"))
    return sdfg

def medium_sdfg():
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

    return sdfg, state, A

def complex_sdfg():
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


def two_state_sdfg():
    sdfg = dace.SDFG("two_states")

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

    return sdfg

def pretty_print(gpu_set, cpu_set):
    print("both GPU and CPU:")
    for name in gpu_set & cpu_set:
        print("\t", name)

    print("\nGPU only:")
    for name in gpu_set - cpu_set:
        print("\t", name)
    
    print("\nCPU only:")
    for name in cpu_set - gpu_set:
        print("\t", name)



### tests ###


import numpy as np

def test_two_state_sdfg(input:float):
    sdfg = two_state_sdfg()
    sdfg.validate()

    orig_output = np.array([0.0])
    sdfg(**{"in": np.array([input]), "A":np.array([0.0]), "out": orig_output})
    
    ota = OtA()
    """ota.set_toplevel_to_GPU(sdfg, nodes.MapEntry)
    ota.set_toplevel_to_GPU(sdfg, nodes.LibraryNode)
    IR_two_state_sdfg = ota.sdfg_to_IR(sdfg)
    ota.eval_IR(sdfg, IR_two_state_sdfg)"""
    ota.apply_pass(sdfg, {})
    sdfg.view()
    sdfg.validate()   

    sdfg._recompile = True
    new_output = np.array([0.0])
    sdfg(**{"in": np.array([input]), "A":np.array([0.0]), "out": new_output})

    assert np.allclose(orig_output, new_output),f"{orig_output} != {new_output}"

test_two_state_sdfg(17.0)


# TODO:
# look through previous testcases, see what still applies
# collect in automatic test suite
# add new test cases to suite
#   simple for all 4 scenarios
#   more involved for all 4 scenarios
# before you fix the bugs:
#   add yakups test cases to the suite
#   add big sdfgs to the suite
#   add heat3d & npbench to the suite
# goal of tomorrow: have a fully functional suite, even if some test don't pass yet
# goal for weekend: get all test cases to run


# TODO: scalars have pass by copy right now -> if map writes to single scalar, detect and raise error or convert to array of length one and properly offload
# curently not offloaded -> incorrect -> run "replace_all_length1_arrays_with_scalars" at start, then replace back iff written to by GPU / map

# todo(?): make interstate edge copy method also work with controlflowblocks 

# TODO: dace rep -> tests -> npbench -> polybench -> copy2d, copy1d, heat3d, ... -> lib nodes, wcr edges etc. -> use as test cases
# views, subset and wcr edges might create problems -> if they do, discuss with Yakup, might not need to handle this
# use npbench -> polybench and s-cases

# TODO: clean up -> PR branch -> write thesis!



