# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from functools import reduce
from operator import mul
from typing import Dict, Collection

import dace
from dace import SDFG, Memlet
from dace.codegen.targets import cpp
from dace.sdfg.state import SDFGState
from dace.subsets import Range
from dace.transformation.dataflow import RedundantArray


def _add_map_with_connectors(st: SDFGState,
                             name: str,
                             ndrange: Dict[str, str],
                             en_conn_bases: Collection[str] = None,
                             ex_conn_bases: Collection[str] = None):
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
    g.add_array('A', (5, 5), dace.float32)
    g.add_array('b', (1, ), dace.float32, transient=True)
    g.add_array('c', (5, 5), dace.float32, transient=True)

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


if __name__ == '__main__':
    test_reshape_strides_multidim_array_all_dims_unit()
    test_reshape_strides_multidim_array_some_dims_unit()
    test_reshape_strides_multidim_array_different_shape()
    test_reshape_strides_from_strided_range()
    test_reshape_strides_from_strided_and_offset_range()
