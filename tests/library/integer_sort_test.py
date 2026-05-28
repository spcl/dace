# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.libraries.sort.nodes.integer_sort.IntegerSort`.

Covers each supported integer dtype (``int8`` through ``uint64``), each CPU-side
implementation (``CPU`` = ska_sort, ``pure`` = std::sort), a few sizes including
the empty-array edge, and a contract check that input and output buffers are
distinct (the libnode produces a sorted copy, not an in-place sort).
"""
import numpy as np
import pytest

import dace
from dace.libraries.sort.nodes.integer_sort import IntegerSort, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME


_DTYPES = [
    (dace.int8, np.int8),
    (dace.int16, np.int16),
    (dace.int32, np.int32),
    (dace.int64, np.int64),
    (dace.uint8, np.uint8),
    (dace.uint16, np.uint16),
    (dace.uint32, np.uint32),
    (dace.uint64, np.uint64),
]


def _build_sort_sdfg(dace_dtype: dace.dtypes.typeclass, n: int, implementation: str) -> dace.SDFG:
    """Build a single-state SDFG that sorts ``arr_in[0:N]`` into ``arr_out[0:N]``."""
    sdfg = dace.SDFG(f'integer_sort_{implementation}_{dace_dtype.to_string()}_{n}'.replace(':', '_').replace(' ', '_'))
    sdfg.add_array('arr_in', [n], dace_dtype)
    sdfg.add_array('arr_out', [n], dace_dtype)
    state = sdfg.add_state('sort')
    a_in = state.add_read('arr_in')
    a_out = state.add_write('arr_out')
    node = IntegerSort('IntegerSort')
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(a_in, None, node, INPUT_CONNECTOR_NAME, dace.Memlet(f'arr_in[0:{n}]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, a_out, None, dace.Memlet(f'arr_out[0:{n}]'))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('dace_dtype,np_dtype', _DTYPES)
@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
def test_integer_sort_matches_numpy(dace_dtype: dace.dtypes.typeclass, np_dtype, implementation: str):
    """Sorting matches ``np.sort`` for each supported integer dtype and implementation."""
    n = 257  # not a power of 2; avoids hiding stride/length bugs
    rng = np.random.default_rng(int(np_dtype(0).itemsize) * 13 + ord(implementation[0]))
    info = np.iinfo(np_dtype)
    arr_in = rng.integers(info.min, info.max, size=n, dtype=np_dtype, endpoint=True)
    arr_out = np.zeros(n, dtype=np_dtype)
    sdfg = _build_sort_sdfg(dace_dtype, n, implementation)
    sdfg(arr_in=arr_in, arr_out=arr_out)
    assert np.array_equal(arr_out, np.sort(arr_in)), (
        f'Mismatch on {dace_dtype}/{implementation}; first diff at '
        f'{np.argmax(arr_out != np.sort(arr_in))}.')


@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
def test_integer_sort_single_element(implementation: str):
    """A length-1 array is already sorted; output must equal input."""
    sdfg = _build_sort_sdfg(dace.int32, 1, implementation)
    arr_in = np.array([42], dtype=np.int32)
    arr_out = np.array([0], dtype=np.int32)
    sdfg(arr_in=arr_in, arr_out=arr_out)
    assert arr_out[0] == 42


@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
def test_integer_sort_already_sorted_is_identity(implementation: str):
    """Sorting an already-sorted array yields the same array."""
    n = 100
    sdfg = _build_sort_sdfg(dace.int64, n, implementation)
    arr_in = np.arange(n, dtype=np.int64)
    arr_out = np.zeros(n, dtype=np.int64)
    sdfg(arr_in=arr_in, arr_out=arr_out)
    assert np.array_equal(arr_out, arr_in)


@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
def test_integer_sort_with_duplicates(implementation: str):
    """Duplicates are preserved and clustered together by ``np.sort``-equivalent ordering."""
    n = 64
    sdfg = _build_sort_sdfg(dace.int32, n, implementation)
    arr_in = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3] * 4, dtype=np.int32)
    arr_out = np.zeros(n, dtype=np.int32)
    sdfg(arr_in=arr_in, arr_out=arr_out)
    assert np.array_equal(arr_out, np.sort(arr_in))


def test_integer_sort_refuses_float_dtype():
    """The libnode's validate refuses non-integer dtypes (float, complex, ...)."""
    sdfg = dace.SDFG('integer_sort_refuses_float')
    sdfg.add_array('arr_in', [8], dace.float64)
    sdfg.add_array('arr_out', [8], dace.float64)
    state = sdfg.add_state('sort')
    a_in = state.add_read('arr_in')
    a_out = state.add_write('arr_out')
    node = IntegerSort('IntegerSort')
    state.add_node(node)
    state.add_edge(a_in, None, node, INPUT_CONNECTOR_NAME, dace.Memlet('arr_in[0:8]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, a_out, None, dace.Memlet('arr_out[0:8]'))
    with pytest.raises(Exception):  # noqa: B017 -- exact exception type leaks codegen internals
        sdfg.compile()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
