# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from functools import reduce
from operator import mul
from typing import Dict, Collection
import warnings

from dace import SDFG, Memlet, dtypes
from dace.codegen import codegen
from dace.codegen.targets import cpp
from dace.sdfg.state import SDFGState
from dace.subsets import Range
from dace.transformation.dataflow import RedundantArray


def _add_map_with_connectors(st: SDFGState, name: str, ndrange: Dict[str, str],
                             en_conn_bases: Collection[str] = None, ex_conn_bases: Collection[str] = None):
    en, ex = st.add_map(name, ndrange)
    if en_conn_bases:
        for c in en_conn_bases:
            en.add_in_connector(f"IN_{c}")
            en.add_out_connector(f"OUT_{c}")
    if ex_conn_bases:
        for c in ex_conn_bases:
            ex.add_in_connector(f"IN_{c}")
            ex.add_out_connector(f"OUT_{c}")
    return en, ex


def test_reshape_strides_multidim_array_all_dims_unit():
    r = Range([(0, 0, 1), (0, 0, 1)])

    # To smaller-sized shape
    target_dims = [1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == [1]
    assert strides == [1]

    # To equal-sized shape
    target_dims = [1, 1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == [1, 1]
    assert strides == [1, 1]

    # To larger-sized shape
    target_dims = [1, 1, 1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == [1, 1, 1]
    assert strides == [1, 1, 1]


def test_reshape_strides_multidim_array_some_dims_unit():
    r = Range([(0, 1, 1), (0, 0, 1)])

    # To smaller-sized shape
    target_dims = [2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1]

    # To equal-sized shape
    target_dims = [2, 1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1, 1]
    # To equal-sized shape, but units first.
    target_dims = [1, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [2, 1]

    # To larger-sized shape.
    target_dims = [2, 1, 1]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1, 1, 1]
    # To larger-sized shape, but units first.
    target_dims = [1, 1, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [2, 2, 1]


def test_reshape_strides_multidim_array_different_shape():
    r = Range([(0, 4, 1), (0, 5, 1)])

    # To smaller-sized shape
    target_dims = [30]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1]

    # To equal-sized shape
    target_dims = [15, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [2, 1]

    # To larger-sized shape
    target_dims = [3, 5, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [10, 2, 1]


def test_reshape_strides_from_strided_range():
    r = Range([(0, 4, 2), (0, 6, 2)])

    # To smaller-sized shape
    target_dims = [12]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1]

    # To equal-sized shape
    target_dims = [4, 3]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [3, 1]

    # To larger-sized shape
    target_dims = [2, 3, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [6, 2, 1]


def test_reshape_strides_from_strided_and_offset_range():
    r = Range([(10, 14, 2), (10, 16, 2)])

    # To smaller-sized shape
    target_dims = [12]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [1]

    # To equal-sized shape
    target_dims = [4, 3]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [3, 1]

    # To larger-sized shape
    target_dims = [2, 3, 2]
    assert reduce(mul, r.size_exact()) == reduce(mul, target_dims)
    reshaped, strides = cpp.reshape_strides(r, None, None, target_dims)
    assert reshaped == target_dims
    assert strides == [6, 2, 1]


def redundant_array_crashes_codegen_test_original_graph():
    g = SDFG('prog')
    g.add_array('A', (5, 5), dtypes.float32)
    g.add_array('b', (1,), dtypes.float32, transient=True)
    g.add_array('c', (5, 5), dtypes.float32, transient=True)

    st0 = g.add_state('st0', is_start_block=True)
    st = st0

    # Make a single map that copies A[i, j] to a transient "scalar" b, then copies that out to a transient array
    # c[i, j], then finally back to A[i, j] again.
    A = st.add_access('A')
    en, ex = _add_map_with_connectors(st, 'm0', {'i': '0:1', 'j': '0:1'}, ['A'], ['A'])
    st.add_edge(A, None, en, 'IN_A', Memlet(expr='A[0:1, 0:1]'))
    b = st.add_access('b')
    st.add_edge(en, 'OUT_A', b, None, Memlet(expr='A[i, j] -> b[0]'))
    c = st.add_access('c')
    st.add_nedge(b, c, Memlet(expr='b[0] -> c[i, j]'))
    st.add_edge(c, None, ex, 'IN_A', Memlet(expr='c[i, j] -> A[i, j]'))
    A = st.add_access('A')
    st.add_edge(ex, 'OUT_A', A, None, Memlet(expr='A[0:1, 0:1]'))
    st0.fill_scope_connectors()

    g.validate()
    g.compile()
    return g


def test_redundant_array_does_not_crash_codegen_but_produces_bad_graph_now():
    """
    This test demonstrates the bug in CPP Codegen that the [PR](https://github.com/spcl/dace/pull/1692) fixes.
    """
    g = redundant_array_crashes_codegen_test_original_graph()
    g.apply_transformations(RedundantArray)
    g.validate()
    g.compile()

    # NOTE: The produced graph still has bug. So, let's test for its existence.
    assert len(g.states()) == 1
    st = g.states()[0]
    assert len(st.source_nodes()) == 1
    src = st.source_nodes()[0]
    assert len(st.out_edges(src)) == 1
    e = st.out_edges(src)[0]
    # This is the wrong part. These symbols are not available in this scope.
    assert e.data.free_symbols == {'i', 'j'}


def test_arrays_bigger_than_max_stack_size_get_deallocated():
    # Setup SDFG with array A that is too big to be allocated on the stack.
    sdfg = SDFG("test")
    sdfg.add_array(name="A", shape=(10000,), dtype=dtypes.float64, storage=dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state("state", is_start_block=True)
    read = state.add_access("A")
    tasklet = state.add_tasklet("dummy", {"a"}, {}, "a = 1")
    state.add_memlet_path(read, tasklet, dst_conn="a", memlet=Memlet("A[0]"))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Generate code for the program by traversing the SDFG state by state
        program_objects = codegen.generate_code(sdfg)

        # Assert that we get the expected warning message
        assert w
        assert any("was allocated on the heap instead of" in str(warn.message) for warn in w)

        # In code, assert that we allocate _and_ deallocate on the heap
        code = program_objects[0].clean_code
        assert code.find("A = new double") > 0, "A is allocated on the heap."
        assert code.find("delete[] A") > 0, "A is deallocated from the heap."


if __name__ == '__main__':
    test_reshape_strides_multidim_array_all_dims_unit()
    test_reshape_strides_multidim_array_some_dims_unit()
    test_reshape_strides_multidim_array_different_shape()
    test_reshape_strides_from_strided_range()
    test_reshape_strides_from_strided_and_offset_range()

    test_redundant_array_does_not_crash_codegen_but_produces_bad_graph_now()
    test_arrays_bigger_than_max_stack_size_get_deallocated()
