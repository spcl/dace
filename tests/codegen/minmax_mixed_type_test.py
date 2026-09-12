# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that the C++ ``Min``/``Max`` templates reject mixing floating-point and
integer arguments.

Background: a symbolic ``Min(1, x)`` (where ``x`` is a double) used to compile to
a template whose return type was deduced from the first argument, silently
truncating the double result to an integer. We now make that a hard compile-time
error instead of producing wrong numbers, while still allowing all-integer and
all-floating-point ``Min``/``Max``.
"""

import numpy as np
import pytest

import dace
from dace import subsets, symbolic
from dace.codegen.exceptions import CompilationError
from dace.sdfg import SDFG, InterstateEdge

# Substring of the static_assert message emitted by the C++ templates; used to
# confirm the compile error came from our guard and not some unrelated failure.
_MIXED_TYPE_ERROR_FRAGMENT = 'mixing floating-point and integer arguments'


def _build_minmax_interstate_sdfg(rhs: str) -> SDFG:
    """Build an SDFG whose only work is a ``zanew = <rhs>`` interstate assignment.

    The double symbol ``zanew`` is seeded to ``0.375`` (a cloud fraction in
    ``[0, 1]``), the assignment ``rhs`` is applied on the next edge, and the
    result is written to the single-element output array.

    :param rhs: the right-hand side of the ``zanew`` interstate assignment.
    :return: the constructed SDFG.
    """
    sdfg = SDFG('minmax_interstate')
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_symbol('zanew', dace.float64)
    init = sdfg.add_state('init', is_start_block=True)
    middle = sdfg.add_state('middle')
    final = sdfg.add_state('final')
    sdfg.add_edge(init, middle, InterstateEdge(assignments={'zanew': '0.375'}))
    sdfg.add_edge(middle, final, InterstateEdge(assignments={'zanew': rhs}))
    write = final.add_write('out')
    tasklet = final.add_tasklet('write_zanew', {}, {'o'}, 'o = zanew')
    final.add_edge(tasklet, 'o', write, None, dace.Memlet('out[0]'))
    return sdfg


def test_min_mixed_int_double_is_compile_error():
    """A ``Min`` mixing an integer literal with a double must fail to compile."""
    sdfg = _build_minmax_interstate_sdfg('Min(1, zanew)')
    out = np.zeros(1, dtype=np.float64)
    with pytest.raises(CompilationError) as excinfo:
        sdfg(out=out)
    assert _MIXED_TYPE_ERROR_FRAGMENT in str(excinfo.value)


def test_max_mixed_int_double_is_compile_error():
    """A ``Max`` mixing an integer literal with a double must fail to compile."""
    sdfg = _build_minmax_interstate_sdfg('Max(0, zanew)')
    out = np.zeros(1, dtype=np.float64)
    with pytest.raises(CompilationError) as excinfo:
        sdfg(out=out)
    assert _MIXED_TYPE_ERROR_FRAGMENT in str(excinfo.value)


def test_min_all_double_compiles_and_runs():
    """``Min`` over doubles compiles and keeps full precision (no truncation)."""
    sdfg = _build_minmax_interstate_sdfg('Min(1.0, zanew)')
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    assert out[0] == pytest.approx(0.375)


def test_max_all_double_compiles_and_runs():
    """``Max`` over doubles compiles and keeps full precision (no truncation)."""
    sdfg = _build_minmax_interstate_sdfg('Max(0.0, zanew)')
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    assert out[0] == pytest.approx(0.375)


def test_min_all_integer_range_compiles_and_runs():
    """``Min`` over integer arguments (the common tiling case) is allowed.

    The map range ``0:Min(N, 4)`` mixes an ``int`` literal with an ``int64``
    symbol; this is legitimate integral math and must keep compiling.
    """
    n_symbol = dace.symbol('N', dace.int64)
    sdfg = SDFG('int_range_min')
    sdfg.add_array('A', [n_symbol], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', {'i': subsets.Range([(0, symbolic.pystr_to_symbolic('Min(N, 4) - 1'), 1)])})
    read = state.add_read('A')
    write = state.add_write('A')
    tasklet = state.add_tasklet('double_it', {'a'}, {'b'}, 'b = a * 2.0')
    state.add_memlet_path(read, entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, exit_node, write, src_conn='b', memlet=dace.Memlet('A[i]'))

    data = np.ones(10, dtype=np.float64)
    sdfg(A=data, N=10)
    # Only the first 4 elements (range 0:Min(10, 4)) are doubled.
    assert np.allclose(data[:4], 2.0)
    assert np.allclose(data[4:], 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
