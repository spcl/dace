# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import dtypes, nodes
from typing import Any, Dict, List, Union
import numpy as np

N = dace.symbol("N")


@dace.program
def arrayop(inp: dace.float32[N], out: dace.float32[N]):
    for i in dace.map[0:N]:
        out[i] = 2 * inp[i]


def key_exists(d: Union[List[Any], Dict[str, Any]], key: str):
    if isinstance(d, list):
        for item in d:
            if key_exists(item, key):
                return True
    elif isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                return True
            if key_exists(v, key):
                return True

    return False


def test_lack_of_omp_props():

    sdfg = arrayop.to_sdfg(simplify=True)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.EntryNode):
            assert (isinstance(node, nodes.MapEntry))
            node.map.schedule = dtypes.ScheduleType.Sequential
            break

    json = sdfg.to_json()
    assert (not key_exists(json, 'omp_num_threads'))
    assert (not key_exists(json, 'omp_schedule'))
    assert (not key_exists(json, 'omp_chunk_size'))


def test_omp_props():

    sdfg = arrayop.to_sdfg(simplify=True)
    mapnode = None
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.EntryNode):
            assert (isinstance(node, nodes.MapEntry))
            mapnode = node.map
            break

    mapnode.schedule = dtypes.ScheduleType.CPU_Multicore
    json = sdfg.to_json()
    assert (key_exists(json, 'omp_num_threads'))
    assert (key_exists(json, 'omp_schedule'))
    assert (key_exists(json, 'omp_chunk_size'))
    code = sdfg.generate_code()[0].clean_code
    assert ("#pragma omp parallel for" in code)

    mapnode.omp_num_threads = 10
    code = sdfg.generate_code()[0].clean_code
    assert ("#pragma omp parallel for num_threads(10)" in code)

    mapnode.omp_schedule = dtypes.OMPScheduleType.Guided
    code = sdfg.generate_code()[0].clean_code
    assert ("#pragma omp parallel for schedule(guided) num_threads(10)" in code)

    mapnode.omp_chunk_size = 5
    code = sdfg.generate_code()[0].clean_code
    assert ("#pragma omp parallel for schedule(guided, 5) num_threads(10)" in code)


def test_omp_parallel():

    @dace.program
    def tester(A: dace.float64[1]):
        for t in dace.map[0:1] @ dace.ScheduleType.CPU_Persistent:
            A[0] += 1

    sdfg = tester.to_sdfg()
    me = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))
    me.map.omp_num_threads = 2

    code = sdfg.generate_code()[0].clean_code
    assert ("#pragma omp parallel num_threads(2)" in code)

    a = np.random.rand(1)
    ref = a + 2
    sdfg(a)
    assert np.allclose(a, ref)


def test_omp_parallel_for_in_parallel():
    """
    Tests that an OpenMP map inside a parallel section ends up without an
    extra (semantically-incorrect) ``parallel`` statement.
    """

    @dace.program
    def tester(A: dace.float64[20]):
        for t in dace.map[0:1] @ dace.ScheduleType.CPU_Persistent:
            for i in dace.map[0:20] @ dace.ScheduleType.CPU_Multicore:
                A[i] += 1

    sdfg = tester.to_sdfg()
    code = sdfg.generate_code()[0].clean_code
    assert "#pragma omp parallel" in code
    assert "#pragma omp for" in code

    a = np.random.rand(20)
    ref = a + 1
    sdfg(a)
    assert np.allclose(a, ref)


def test_omp_get_tid():

    @dace.program
    def tester(A: dace.float64[20]):
        for t in dace.map[0:1] @ dace.ScheduleType.CPU_Persistent:
            A[t] += 1

    sdfg = tester.to_sdfg()
    me = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))
    me.map.omp_num_threads = 2

    code = sdfg.generate_code()[0].clean_code
    assert "#pragma omp parallel num_threads(2)" in code
    assert "omp_get_thread_num()" in code

    a = np.random.rand(20)
    ref = np.copy(a)
    ref[:2] += 1

    sdfg(a)
    assert np.allclose(a, ref)


def test_omp_get_tid_elision():

    @dace.program
    def tester(A: dace.float64[20]):
        for t in dace.map[0:1] @ dace.ScheduleType.CPU_Persistent:
            A[0] += 1

    sdfg = tester.to_sdfg()
    code = sdfg.generate_code()[0].clean_code
    assert "omp_get_thread_num()" not in code


def test_omp_get_ntid():
    __omp_num_threads = dace.symbol('__omp_num_threads')

    @dace.program
    def tester(A: dace.int64[1]):
        for _ in dace.map[0:__omp_num_threads] @ dace.ScheduleType.CPU_Persistent:
            A[0] = __omp_num_threads

    sdfg = tester.to_sdfg()
    code = sdfg.generate_code()[0].clean_code
    assert "omp_get_num_threads()" in code

    me = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))
    me.map.omp_num_threads = 3

    a = np.zeros([1], dtype=np.int64)
    sdfg(a, __omp_num_threads=1)  # Feed in some other value
    assert np.allclose(a, 3)


if __name__ == "__main__":
    test_lack_of_omp_props()
    test_omp_props()
    test_omp_parallel()
    test_omp_parallel_for_in_parallel()
    test_omp_get_tid()
    test_omp_get_tid_elision()
    test_omp_get_ntid()
