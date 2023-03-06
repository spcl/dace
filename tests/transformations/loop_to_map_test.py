# Copyright 2020-2020 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import os
import tempfile

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.interstate import LoopToMap


def make_sdfg(with_wcr, map_in_guard, reverse_loop, use_variable, assign_after, log_path):

    sdfg = dace.SDFG(f"loop_to_map_test_{with_wcr}_{map_in_guard}_{reverse_loop}_{use_variable}_{assign_after}")
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
    aname, _ = sdfg.add_array('A', (10, ), dace.int32)
    bname, _ = sdfg.add_array('B', (10, ), dace.int32)
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
    B = np.empty((10, ), dtype=np.int32)
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
        assert np.array_equal(B[i], np.arange(start + 9, start - 1, -1, dtype=np.int32))

def test_iteration_variable_used_outside():
    N = dace.symbol("N", dace.int32)

    @dace.program
    def tester(A: dace.float64[N], output: dace.float64[1]):
        i = -1

        for i in range(N):
            A[i] += 1

        if i > 10:
            output[0] = 1.0

    sdfg = tester.to_sdfg(simplify=True)
    assert sdfg.apply_transformations(LoopToMap) == 0


def test_symbol_race():

    # Adapted from npbench's crc16 test
    # https://github.com/spcl/npbench/blob/main/npbench/benchmarks/crc16/crc16_dace.py
    poly: dace.uint16 = 0x8408

    @dace.program
    def tester(data: dace.int32[20]):
        crc: dace.uint16 = 0xFFFF
        for i in range(20):
            b = data[i]
            cur_byte = 0xFF & b
            for _ in range(0, 8):
                if (crc & 0x0001) ^ (cur_byte & 0x0001):
                    crc = (crc >> 1) ^ poly
                else:
                    crc >>= 1
                cur_byte >>= 1
        crc = (~crc & 0xFFFF)
        crc = (crc << 8) | ((crc >> 8) & 0xFF)

    sdfg = tester.to_sdfg(simplify=True)
    assert sdfg.apply_transformations(LoopToMap) == 0


def test_symbol_write_before_read():
    sdfg = dace.SDFG('tester')
    init = sdfg.add_state(is_start_state=True)
    body_start = sdfg.add_state()
    body = sdfg.add_state()
    body_end = sdfg.add_state()
    sdfg.add_loop(init, body_start, None, 'i', '0', 'i < 20', 'i + 1', loop_end_state=body_end)

    # Internal loop structure
    sdfg.add_edge(body_start, body, dace.InterstateEdge(assignments=dict(j='0')))
    sdfg.add_edge(body, body_end, dace.InterstateEdge(assignments=dict(j='j + 1')))

    assert sdfg.apply_transformations(LoopToMap) == 1


@pytest.mark.parametrize('overwrite', (False, True))
def test_symbol_array_mix(overwrite):
    sdfg = dace.SDFG('tester')
    sdfg.add_transient('tmp', [1], dace.float64)
    sdfg.add_symbol('sym', dace.float64)
    init = sdfg.add_state(is_start_state=True)
    body_start = sdfg.add_state()
    body = sdfg.add_state()
    body_end = sdfg.add_state()
    after = sdfg.add_state()
    sdfg.add_loop(init, body_start, after, 'i', '0', 'i < 20', 'i + 1', loop_end_state=body_end)

    sdfg.out_edges(init)[0].data.assignments['sym'] = '0.0'

    # Internal loop structure
    t = body_start.add_tasklet('def', {}, {'o'}, 'o = i')
    body_start.add_edge(t, 'o', body_start.add_write('tmp'), None, dace.Memlet('tmp'))

    if overwrite:
        sdfg.add_edge(body_start, body, dace.InterstateEdge(assignments=dict(sym='tmp')))
    else:
        sdfg.add_edge(body_start, body, dace.InterstateEdge(assignments=dict(sym='sym + tmp')))
    sdfg.add_edge(body, body_end, dace.InterstateEdge(assignments=dict(sym='sym + 1.0')))

    assert sdfg.apply_transformations(LoopToMap) == (1 if overwrite else 0)

@pytest.mark.parametrize('parallel', (False, True))
def test_symbol_array_mix_2(parallel):
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_symbol('sym', dace.float64)
    init = sdfg.add_state(is_start_state=True)
    body_start = sdfg.add_state()
    body_end = sdfg.add_state()
    after = sdfg.add_state()
    sdfg.add_loop(init, body_start, after, 'i', '1', 'i < 20', 'i + 1', loop_end_state=body_end)

    sdfg.out_edges(init)[0].data.assignments['sym'] = '0.0'

    # Internal loop structure
    if not parallel:
        t = body_start.add_tasklet('def', {}, {'o'}, 'o = i')
        body_start.add_edge(t, 'o', body_start.add_write('A'), None, dace.Memlet('A[i]'))

    sdfg.add_edge(body_start, body_end, dace.InterstateEdge(assignments=dict(sym='A[i - 1]')))
    t = body_start.add_tasklet('use', {}, {'o'}, 'o = sym')
    body_start.add_edge(t, 'o', body_start.add_write('B'), None, dace.Memlet('B[i]'))

    assert sdfg.apply_transformations(LoopToMap) == (1 if parallel else 0)


@pytest.mark.parametrize('overwrite', (False, True))
def test_internal_symbol_used_outside(overwrite):
    sdfg = dace.SDFG('tester')
    init = sdfg.add_state(is_start_state=True)
    body_start = sdfg.add_state()
    body = sdfg.add_state()
    body_end = sdfg.add_state()
    after = sdfg.add_state()
    sdfg.add_loop(init, body_start, after, 'i', '0', 'i < 20', 'i + 1', loop_end_state=body_end)

    # Internal loop structure
    sdfg.add_edge(body_start, body, dace.InterstateEdge(assignments=dict(j='0')))
    sdfg.add_edge(body, body_end, dace.InterstateEdge(assignments=dict(j='j + 1')))

    # Use after
    after_1 = sdfg.add_state()
    after_1.add_tasklet('use', {}, {}, 'printf("%d\\n", j)')

    if overwrite:
        sdfg.add_edge(after, after_1, dace.InterstateEdge(assignments=dict(j='5')))
    else:
        sdfg.add_edge(after, after_1, dace.InterstateEdge())

    assert sdfg.apply_transformations(LoopToMap) == (1 if overwrite else 0)


def test_shared_local_transient_single_state():
    """
    Array A has one element per iteration and can be allocated outside the for-loop/Map.
    """

    sdfg = dace.SDFG('shared_local_transient_single_state')
    begin = sdfg.add_state('begin', is_start_state=True)
    guard = sdfg.add_state('guard')
    body = sdfg.add_state('body')
    end = sdfg.add_state('end')

    sdfg.add_edge(begin, guard, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard, body, dace.InterstateEdge(condition='i < 10', assignments={'j': 'i+1'}))
    sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition='i >= 10'))

    sdfg.add_array('A', (10,), dace.int32, transient=True)
    sdfg.add_array('__return', (10,), dace.int32)

    t1 = body.add_tasklet('t1', {}, {'__out'}, '__out = 5 + j')
    anode = body.add_access('A')
    t2 = body.add_tasklet('t2', {'__inp'}, {'__out'}, '__out = __inp * 2')
    bnode = body.add_access('__return')
    body.add_edge(t1, '__out', anode, None, dace.Memlet(data='A', subset='i'))
    body.add_edge(anode, None, t2, '__inp', dace.Memlet(data='A', subset='i'))
    body.add_edge(t2, '__out', bnode, None, dace.Memlet(data='__return', subset='i'))

    sdfg.apply_transformations_repeated(LoopToMap)
    assert 'A' in sdfg.arrays

    ref = (np.arange(10, dtype=np.int32) + 6) * 2
    val = sdfg()
    assert np.allclose(ref, val)


def test_thread_local_transient_single_state():
    """
    The shape of array A depends on the iteration variable and, therefore, it is thread-local.
    """

    sdfg = dace.SDFG('thread_local_transient_single_state')
    begin = sdfg.add_state('begin', is_start_state=True)
    guard = sdfg.add_state('guard')
    body = sdfg.add_state('body')
    end = sdfg.add_state('end')

    sdfg.add_symbol('i', dace.int32)
    i = dace.symbol('i', dace.int32)

    sdfg.add_edge(begin, guard, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard, body, dace.InterstateEdge(condition='i < 10', assignments={'j': 'i+1'}))
    sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition='i >= 10'))

    sdfg.add_array('A', (i+1,), dace.int32, transient=True)
    sdfg.add_array('__return', (10,), dace.int32)

    t1 = body.add_tasklet('t1', {}, {'__out'}, '__out = 5 + j')
    anode = body.add_access('A')
    t2 = body.add_tasklet('t2', {'__inp'}, {'__out'}, '__out = __inp * 2')
    bnode = body.add_access('__return')
    body.add_edge(t1, '__out', anode, None, dace.Memlet(data='A', subset='i'))
    body.add_edge(anode, None, t2, '__inp', dace.Memlet(data='A', subset='i'))
    body.add_edge(t2, '__out', bnode, None, dace.Memlet(data='__return', subset='i'))

    sdfg.apply_transformations_repeated(LoopToMap)
    assert not ('A' in sdfg.arrays)

    ref = (np.arange(10, dtype=np.int32) + 6) * 2
    val = sdfg()
    assert np.allclose(ref, val)


def test_shared_local_transient_multi_state():
    """
    Array A has one element per iteration and can be allocated outside the for-loop/Map.
    """

    sdfg = dace.SDFG('shared_local_transient_multi_state')
    begin = sdfg.add_state('begin', is_start_state=True)
    guard = sdfg.add_state('guard')
    body0 = sdfg.add_state('body0')
    body1 = sdfg.add_state('body1')
    end = sdfg.add_state('end')

    sdfg.add_edge(begin, guard, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard, body0, dace.InterstateEdge(condition='i < 10'))
    sdfg.add_edge(body0, body1, dace.InterstateEdge())
    sdfg.add_edge(body1, guard, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition='i >= 10'))

    sdfg.add_array('A', (10,), dace.int32, transient=True)
    sdfg.add_array('__return', (10,), dace.int32)

    t1 = body0.add_tasklet('t1', {}, {'__out'}, '__out = 5 + i + 1')
    anode0 = body0.add_access('A')
    anode1 = body1.add_access('A')
    t2 = body1.add_tasklet('t2', {'__inp'}, {'__out'}, '__out = __inp * 2')
    bnode = body1.add_access('__return')
    body0.add_edge(t1, '__out', anode0, None, dace.Memlet(data='A', subset='i'))
    body1.add_edge(anode1, None, t2, '__inp', dace.Memlet(data='A', subset='i'))
    body1.add_edge(t2, '__out', bnode, None, dace.Memlet(data='__return', subset='i'))

    sdfg.apply_transformations_repeated(LoopToMap)
    assert 'A' in sdfg.arrays

    ref = (np.arange(10, dtype=np.int32) + 6) * 2
    val = sdfg()
    assert np.allclose(ref, val)


def test_thread_local_transient_multi_state():
    """
    The shape of array A depends on the iteration variable and, therefore, it is thread-local.
    """

    sdfg = dace.SDFG('thread_local_transient_multi_state')
    begin = sdfg.add_state('begin', is_start_state=True)
    guard = sdfg.add_state('guard')
    body0 = sdfg.add_state('body0')
    body1 = sdfg.add_state('body1')
    end = sdfg.add_state('end')

    sdfg.add_symbol('i', dace.int32)
    i = dace.symbol('i', dace.int32)

    sdfg.add_edge(begin, guard, dace.InterstateEdge(assignments={'i': 0}))
    sdfg.add_edge(guard, body0, dace.InterstateEdge(condition='i < 10'))
    sdfg.add_edge(body0, body1, dace.InterstateEdge())
    sdfg.add_edge(body1, guard, dace.InterstateEdge(assignments={'i': 'i + 1'}))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition='i >= 10'))

    sdfg.add_array('A', (i+1,), dace.int32, transient=True)
    sdfg.add_array('__return', (10,), dace.int32)

    t1 = body0.add_tasklet('t1', {}, {'__out'}, '__out = 5 + i + 1')
    anode0 = body0.add_access('A')
    anode1 = body1.add_access('A')
    t2 = body1.add_tasklet('t2', {'__inp'}, {'__out'}, '__out = __inp * 2')
    bnode = body1.add_access('__return')
    body0.add_edge(t1, '__out', anode0, None, dace.Memlet(data='A', subset='i'))
    body1.add_edge(anode1, None, t2, '__inp', dace.Memlet(data='A', subset='i'))
    body1.add_edge(t2, '__out', bnode, None, dace.Memlet(data='__return', subset='i'))

    sdfg.apply_transformations_repeated(LoopToMap)
    assert not ('A' in sdfg.arrays)

    ref = (np.arange(10, dtype=np.int32) + 6) * 2
    val = sdfg()
    assert np.allclose(ref, val)



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
    test_iteration_variable_used_outside()
    test_symbol_race()
    test_symbol_write_before_read()
    test_symbol_array_mix(False)
    test_symbol_array_mix(True)
    test_symbol_array_mix_2(False)
    test_symbol_array_mix_2(True)
    test_internal_symbol_used_outside(False)
    test_internal_symbol_used_outside(True)
    test_shared_local_transient_single_state()
    test_thread_local_transient_single_state()
    test_shared_local_transient_multi_state()
    test_thread_local_transient_multi_state()
