# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``prune_unused_nsdfg_connectors``"""
import numpy as np

import dace
from dace.sdfg.construction_utils import (
    prune_unused_nsdfg_connectors,
    prune_unused_nsdfg_connectors_recursive,
)


def _make_inner_sdfg(name: str, used_arrays, unused_arrays):
    """Builds a one-state nested SDFG that computes B[i] = A[i] + 1 over an
    inner element-wise map, and declares (but never touches)
    ``unused_arrays``. Shape of every array is [N]."""
    inner = dace.SDFG(name)
    inner.add_symbol('N', dace.int64)
    for a in list(used_arrays) + list(unused_arrays):
        inner.add_array(a, [dace.symbol('N')], dace.float64)

    state = inner.add_state('body', is_start_block=True)
    if 'A' in used_arrays and 'B' in used_arrays:
        rA = state.add_read('A')
        wB = state.add_write('B')
        ime, imx = state.add_map('inner_m', {'k': '0:N'})
        t = state.add_tasklet('add_one', {'i'}, {'o'}, 'o = i + 1.0')
        state.add_memlet_path(rA, ime, t, dst_conn='i', memlet=dace.Memlet('A[k]'))
        state.add_memlet_path(t, imx, wB, src_conn='o', memlet=dace.Memlet('B[k]'))

    return inner


def _wrap_with_map(outer: dace.SDFG, inner: dace.SDFG, in_arrays, out_arrays, extra_connectors=()):
    """Puts ``inner`` as a NestedSDFG inside a map in ``outer``. Adds the
    connectors listed in ``in_arrays + out_arrays + extra_connectors`` so
    there can be "unused" ones."""
    state = outer.add_state('driver', is_start_block=True)
    me, mx = state.add_map('m', {'i': '0:1'})

    in_conns = set(in_arrays) | set(extra_connectors)
    out_conns = set(out_arrays) | set(extra_connectors)

    nsdfg = state.add_nested_sdfg(inner, in_conns, out_conns, symbol_mapping={'N': dace.symbol('N')})

    for name in in_conns:
        rd = state.add_read(name)
        state.add_memlet_path(rd, me, nsdfg, dst_conn=name, memlet=dace.Memlet(f'{name}[0:N]'))
    for name in out_conns:
        wr = state.add_write(name)
        state.add_memlet_path(nsdfg, mx, wr, src_conn=name, memlet=dace.Memlet(f'{name}[0:N]'))

    return state, nsdfg


def test_prune_unused_input_connector():
    """An array declared as an input connector but never read inside the
    nested SDFG is pruned; the corresponding access-node tap and map
    connectors are cleaned up as well."""

    N = 8
    outer = dace.SDFG('outer')
    outer.add_symbol('N', dace.int64)
    for name in ('A', 'B', 'C'):
        outer.add_array(name, [dace.symbol('N')], dace.float64)

    inner = _make_inner_sdfg('inner', used_arrays=['A', 'B'], unused_arrays=['C'])
    state, nsdfg = _wrap_with_map(outer, inner, in_arrays=['A', 'C'], out_arrays=['B'])

    outer.validate()
    assert 'C' in nsdfg.in_connectors

    removed = prune_unused_nsdfg_connectors(state, nsdfg)
    assert removed == 1
    assert 'C' not in nsdfg.in_connectors
    assert 'C' not in inner.arrays

    # No orphan access node for C should remain.
    c_accesses = [n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == 'C']
    assert c_accesses == []

    # No map connectors associated with the pruned array.
    map_entry = [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)][0]
    assert 'IN_C' not in map_entry.in_connectors
    assert 'OUT_C' not in map_entry.out_connectors

    outer.validate()

    # Sanity: SDFG still runs and produces expected result.
    A = np.arange(N, dtype=np.float64)
    B = np.zeros(N, dtype=np.float64)
    C = np.empty(N, dtype=np.float64)
    outer(A=A, B=B, C=C, N=np.int64(N))
    np.testing.assert_allclose(B, A + 1.0)


def test_prune_unused_output_connector():
    """Output connector whose array is never written inside the nested SDFG
    is pruned."""

    N = 6
    outer = dace.SDFG('outer_out')
    outer.add_symbol('N', dace.int64)
    for name in ('A', 'B', 'D'):
        outer.add_array(name, [dace.symbol('N')], dace.float64)

    inner = _make_inner_sdfg('inner_out', used_arrays=['A', 'B'], unused_arrays=['D'])
    state, nsdfg = _wrap_with_map(outer, inner, in_arrays=['A'], out_arrays=['B', 'D'])

    outer.validate()
    assert 'D' in nsdfg.out_connectors

    removed = prune_unused_nsdfg_connectors(state, nsdfg)
    assert removed == 1
    assert 'D' not in nsdfg.out_connectors
    assert 'D' not in inner.arrays

    map_exit = [n for n in state.nodes() if isinstance(n, dace.nodes.MapExit)][0]
    assert 'IN_D' not in map_exit.in_connectors
    assert 'OUT_D' not in map_exit.out_connectors

    outer.validate()


def test_prune_recursive_preserves_used_connectors():
    """Running the recursive variant on an SDFG with nested layers prunes
    the dead connectors but keeps the used ones intact."""

    N = 4
    outer = dace.SDFG('outer_rec')
    outer.add_symbol('N', dace.int64)
    for name in ('A', 'B', 'X', 'Y'):
        outer.add_array(name, [dace.symbol('N')], dace.float64)

    inner = _make_inner_sdfg('inner_rec', used_arrays=['A', 'B'], unused_arrays=['X', 'Y'])
    state, nsdfg = _wrap_with_map(outer, inner, in_arrays=['A', 'X'], out_arrays=['B', 'Y'])

    outer.validate()

    removed = prune_unused_nsdfg_connectors_recursive(outer)
    assert removed == 2  # X (input) + Y (output)

    assert 'A' in nsdfg.in_connectors
    assert 'B' in nsdfg.out_connectors
    assert 'X' not in nsdfg.in_connectors
    assert 'Y' not in nsdfg.out_connectors

    outer.validate()

    A = np.random.rand(N).astype(np.float64)
    B = np.zeros(N, dtype=np.float64)
    X = np.zeros(N, dtype=np.float64)
    Y = np.zeros(N, dtype=np.float64)
    outer(A=A, B=B, X=X, Y=Y, N=np.int64(N))
    np.testing.assert_allclose(B, A + 1.0)


if __name__ == '__main__':
    test_prune_unused_input_connector()
    test_prune_unused_output_connector()
    test_prune_recursive_preserves_used_connectors()