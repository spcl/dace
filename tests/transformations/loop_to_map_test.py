# Copyright 2020-2020 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import dace
import numpy as np
import os
import tempfile
from dace.sdfg import nodes
from dace.transformation.interstate import LoopToMap


def make_sdfg(with_wcr, map_in_guard, reverse_loop, use_variable, assign_after, log_path):

    sdfg = dace.SDFG(f"loop_to_map_test_{with_wcr}_{map_in_guard}_"
                     f"{reverse_loop}_{use_variable}_{assign_after}")
    sdfg.set_global_code("#include <fstream>\n#include <mutex>")

    init = sdfg.add_state("init")
    guard = sdfg.add_state("guard")
    body = sdfg.add_state("body")
    after = sdfg.add_state("after")
    post = sdfg.add_state("post")

    N = dace.symbol("N", dace.int32)

    if not reverse_loop:
        sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={"i": "0"}))
        sdfg.add_edge(guard, body, dace.InterstateEdge(condition="i < N"))
        sdfg.add_edge(guard, after, dace.InterstateEdge(condition="i >= N"))
        sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={"i": "i + 1"}))
    else:
        sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={"i": "N - 1"}))
        sdfg.add_edge(guard, body, dace.InterstateEdge(condition="i >= 0"))
        sdfg.add_edge(guard, after, dace.InterstateEdge(condition="i < 0"))
        sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={"i": "i - 1"}))
    sdfg.add_edge(after, post, dace.InterstateEdge(assignments={"i": "N"} if assign_after else None))

    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    sdfg.add_array("C", [N], dace.float64)
    sdfg.add_array("D", [N], dace.float64)
    sdfg.add_array("E", [1], dace.uint16)

    a = body.add_read("A")
    b = body.add_read("B")
    c = body.add_write("C")
    d = body.add_write("D")

    if map_in_guard:
        guard_read = guard.add_read("C")
        guard_write = guard.add_write("C")
        guard.add_mapped_tasklet("write_self", {"i": "0:N"}, {"c_in": dace.Memlet("C[i]")},
                                 "c_out = c_in", {"c_out": dace.Memlet("C[i]")},
                                 external_edges=True,
                                 input_nodes={"C": guard_read},
                                 output_nodes={"C": guard_write})

    tasklet0 = body.add_tasklet("tasklet0", {"a"}, {"c"}, "c = 1/a")
    tasklet1 = body.add_tasklet("tasklet1", {"a", "b"}, {"d"}, "d = sqrt(a**2 + b**2)")

    tasklet2 = body.add_tasklet("tasklet2", {}, {},
                                f"""\
static std::mutex mutex;
std::unique_lock<std::mutex> lock(mutex);
std::ofstream of("{log_path}", std::ofstream::app);
of << i << "\\n";""",
                                language=dace.Language.CPP)

    body.add_memlet_path(a, tasklet0, dst_conn="a", memlet=dace.Memlet("A[i]"))
    body.add_memlet_path(tasklet0,
                         c,
                         src_conn="c",
                         memlet=dace.Memlet("C[i]", wcr="lambda a, b: a + b" if with_wcr else None))

    body.add_memlet_path(a, tasklet1, dst_conn="a", memlet=dace.Memlet("A[i]"))
    body.add_memlet_path(b, tasklet1, dst_conn="b", memlet=dace.Memlet("B[i]"))
    body.add_memlet_path(tasklet1,
                         d,
                         src_conn="d",
                         memlet=dace.Memlet("D[i]", wcr="lambda a, b: a + b" if with_wcr else None))

    e = post.add_write("E")
    post_tasklet = post.add_tasklet("post", {}, {"e"}, "e = i" if use_variable else "e = N")
    post.add_memlet_path(post_tasklet, e, src_conn="e", memlet=dace.Memlet("E[0]"))

    return sdfg


def run_loop_to_map(n, *args):

    # Use mangled temporary file to avoid name clashes when run in parallel
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    sdfg = make_sdfg(*args, temp_path)

    if n is None:
        n = dace.int32(16)

    a = 4 * np.ones((n, ), dtype=np.float64)
    b = 3 * np.ones((n, ), dtype=np.float64)
    c = np.zeros((n, ), dtype=np.float64)
    d = np.zeros((n, ), dtype=np.float64)
    e = np.empty((1, ), dtype=np.uint16)

    num_transformations = sdfg.apply_transformations(LoopToMap)

    sdfg(A=a, B=b, C=c, D=d, E=e, N=n)

    if not all(c[:] == 0.25) or not all(d[:] == 5):
        raise ValueError("Validation failed.")

    if e[0] != n:
        raise ValueError("Validation failed.")

    numbers_written = []
    with open(temp_path, "r") as f:
        for line in f:
            numbers_written.append(int(line.strip()))
    if not all(sorted(numbers_written) == np.arange(n)):
        raise ValueError("Validation failed.")

    os.remove(temp_path)  # Clean up

    return num_transformations


def test_loop_to_map(n=None):
    # Case 0: no wcr, no dataflow in guard. Transformation should apply
    if run_loop_to_map(n, False, False, False, False, False) != 1:
        raise RuntimeError("LoopToMap was not applied.")


def test_loop_to_map_wcr(n=None):
    # Case 1: WCR on the edge. Transformation should apply
    if run_loop_to_map(n, True, False, False, False, False) != 1:
        raise RuntimeError("LoopToMap was not applied.")


def test_loop_to_map_dataflow_on_guard(n=None):
    # Case 2: there is dataflow on the guard state. Should not apply
    if run_loop_to_map(n, False, True, False, False, False) != 0:
        raise RuntimeError("LoopToMap should not have been applied.")


def test_loop_to_map_negative_step(n=None):
    # Case 3: loop order reversed. Transformation should still apply
    if run_loop_to_map(n, False, False, True, False, False) != 1:
        raise RuntimeError("LoopToMap was not applied.")


def test_loop_to_map_variable_used(n=None):
    # Case 4: the loop variable is used in a later state: should not apply
    if run_loop_to_map(n, False, False, False, True, False) != 0:
        raise RuntimeError("LoopToMap should not have been applied.")


def test_loop_to_map_variable_reassigned(n=None):
    # Case 5: the loop variable is used in a later state, but reassigned first:
    # should apply
    if run_loop_to_map(n, False, False, False, True, True) != 1:
        raise RuntimeError("LoopToMap was not applied.")


def test_output_copy():

    @dace.program
    def l2mtest_copy(A: dace.float64[20, 20]):
        for i in range(1, 20):
            A[i, :] = A[i - 1] + 5

    A = np.random.rand(20, 20)
    regression = np.copy(A)
    for i in range(1, 20):
        regression[i, :] = regression[i - 1] + 5

    sdfg = l2mtest_copy.to_sdfg()

    assert sdfg.apply_transformations(LoopToMap) == 0
    sdfg(A=A)

    assert np.allclose(A, regression)


def test_output_accumulate():

    @dace.program
    def l2mtest_accumulate(A: dace.float64[20, 20]):
        for i in range(1, 20):
            A[i, :] += A[i - 1] + 5

    A = np.random.rand(20, 20)
    regression = np.copy(A)
    for i in range(1, 20):
        regression[i, :] += regression[i - 1] + 5

    sdfg = l2mtest_accumulate.to_sdfg()

    assert sdfg.apply_transformations(LoopToMap) == 0
    sdfg(A=A)

    assert np.allclose(A, regression)


def test_specialize():
    # Test inspired by issue #909

    size = dace.symbol("size")

    @dace.program
    def is_greater(in_data: dace.float64[size], out_data: dace.bool[size]):
        tmp = np.empty(size, dtype=dace.bool)

        @dace.map
        def detect_greater(i: _[0:size]):
            inp << in_data[i]
            is_greater >> tmp[i]

            if (inp > 0.5):
                is_greater = True
            else:
                is_greater = False

        # Write to memory
        for nb in range(size):
            out_data[nb] = tmp[nb]

    x = np.random.rand(8)
    y = np.empty(8, dtype=bool)
    regression = np.empty(8, dtype=bool)
    for i in range(8):
        regression[i] = x[i] > 0.5

    sdfg = is_greater.to_sdfg()
    sdfg.specialize(dict(size=8))
    assert sdfg.apply_transformations_repeated(LoopToMap) == 1

    sdfg(x, y)

    assert np.allclose(y, regression)


def test_empty_loop():

    @dace.program
    def empty_loop():
        for i in range(10):
            pass

    sdfg = empty_loop.to_sdfg(simplify=False)
    assert sdfg.apply_transformations(LoopToMap) == 1

    sdfg.validate()

    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            return

    assert False


def test_interstate_dep():

    sdfg = dace.SDFG('intestate_dep')
    sdfg.add_array('A', (10, ), dtype=np.int32)
    init = sdfg.add_state('init', is_start_state=True)
    guard = sdfg.add_state('guard')
    body0 = sdfg.add_state('body0')
    body1 = sdfg.add_state('body1')
    pexit = sdfg.add_state('exit')

    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={'i': '1'}))
    sdfg.add_edge(guard, body0, dace.InterstateEdge(condition='i < 9'))
    sdfg.add_edge(body0, body1, dace.InterstateEdge(assignments={'s': 'A[i-1] + A[i+1]'}))
    sdfg.add_edge(body1, guard, dace.InterstateEdge(assignments={'i': 'i+1'}))
    sdfg.add_edge(guard, pexit, dace.InterstateEdge(condition='i >= 9'))

    t = body1.add_tasklet('tasklet', {}, {'__out'}, '__out = s')
    a = body1.add_access('A')
    body1.add_edge(t, '__out', a, None, dace.Memlet('A[i]'))

    ref = np.random.randint(0, 10, size=(10, ), dtype=np.int32)
    val = np.copy(ref)
    sdfg(A=ref)

    assert sdfg.apply_transformations(LoopToMap) == 0
    sdfg(A=val)

    assert np.array_equal(val, ref)


def test_need_for_tasklet():

    sdfg = dace.SDFG('needs_tasklet')
    aname, _ = sdfg.add_array('A', (10,), dace.int32)
    bname, _ = sdfg.add_array('B', (10,), dace.int32)
    body = sdfg.add_state('body')
    _, _, _ = sdfg.add_loop(None, body, None, 'i', '0', 'i < 10', 'i + 1', None)
    anode = body.add_access(aname)
    bnode = body.add_access(bname)
    body.add_nedge(anode, bnode, dace.Memlet(data=aname, subset='i', other_subset='9 - i'))

    sdfg.apply_transformations_repeated(LoopToMap)
    found = False
    for n, s in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.Tasklet):
            found = True
            break
    
    assert found

    A = np.arange(10, dtype=np.int32)
    B = np.empty((10,), dtype=np.int32)
    sdfg(A=A, B=B)

    assert np.array_equal(B, np.arange(9, -1, -1, dtype=np.int32))


def test_need_for_transient():

    sdfg = dace.SDFG('needs_transient')
    aname, _ = sdfg.add_array('A', (10, 10), dace.int32)
    bname, _ = sdfg.add_array('B', (10, 10), dace.int32)
    body = sdfg.add_state('body')
    _, _, _ = sdfg.add_loop(None, body, None, 'i', '0', 'i < 10', 'i + 1', None)
    anode = body.add_access(aname)
    bnode = body.add_access(bname)
    body.add_nedge(anode, bnode, dace.Memlet(data=aname, subset='0:10, i', other_subset='0:10, 9 - i'))

    sdfg.apply_transformations_repeated(LoopToMap)
    found = False
    for n, s in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.AccessNode) and n.data not in (aname, bname):
            found = True
            break
    
    assert found

    A = np.arange(100, dtype=np.int32).reshape(10, 10).copy()
    B = np.empty((10, 10), dtype=np.int32)
    sdfg(A=A, B=B)

    for i in range(10):
        start = i * 10
        assert np.array_equal(B[i], np.arange(start + 9, start -1, -1, dtype=np.int32))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=16, type=int)
    args = parser.parse_args()

    n = np.int32(args.N)

    test_loop_to_map(n)
    test_loop_to_map_wcr(n)
    test_loop_to_map_dataflow_on_guard(n)
    test_loop_to_map_negative_step(n)
    test_loop_to_map_variable_used(n)
    test_loop_to_map_variable_reassigned(n)
    test_output_copy()
    test_output_accumulate()
    test_empty_loop()
    test_interstate_dep()
    test_need_for_tasklet()
    test_need_for_transient()
