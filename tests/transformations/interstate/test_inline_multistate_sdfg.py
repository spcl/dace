"""Unit tests for :class:`InlineMultistateSDFG`.

Covers shapes that ``MapToForLoop(inline_after=True)`` and the
canonicalize pipeline rely on:

* a pre/post tasklet pair that share an array name with the NSDFG but
  use SEPARATE outer ``AccessNode`` objects (the data-name ordering case
  that drives ``isolate_nested_sdfg``);
* non-identity ``symbol_mapping`` entries that must be lowered to
  interstate-edge assignments on the predecessor edge (and the matching
  new outer symbol declarations);
* connectors whose inner array name differs from the outer array name
  must be renamed before the inline so the inlined dataflow references
  the outer name;
* map-scoped NSDFGs must be refused (inherits from
  ``isolate_nested_sdfg``).
"""
import copy

import dace
import numpy as np
import pytest

from dace import Memlet, dtypes
from dace.sdfg import SDFG, nodes
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG

N = dace.symbol('N')


def _find_nsdfg(sdfg: SDFG):
    for state in sdfg.states():
        for nd in state.nodes():
            if isinstance(nd, nodes.NestedSDFG):
                return state, nd
    return None, None


def _apply_inline(sdfg: SDFG):
    state, nsdfg_node = _find_nsdfg(sdfg)
    assert nsdfg_node is not None
    xf = InlineMultistateSDFG()
    xf.nested_sdfg = nsdfg_node
    xf.expr_index = 0
    assert xf.can_be_applied(state, 0, sdfg, permissive=False)
    xf.apply(state, sdfg)
    sdfg.validate()


def _build_pre_nsdfg_post() -> SDFG:
    """Outer state holds ``pre_map(inp -> mid1)``, an NSDFG reading
    ``mid1`` and writing ``mid2``, then ``post_map(mid2 -> out)``. The
    outer state uses SEPARATE ``AccessNode`` objects for each side of
    the chain (the harder shape ``isolate_nested_sdfg`` must classify
    by data-name ordering).
    """
    outer = SDFG('pre_inner_post')
    outer.add_array('inp', [N], dtypes.float64)
    outer.add_array('out', [N], dtypes.float64)
    outer.add_array('mid1', [N], dtypes.float64, transient=True)
    outer.add_array('mid2', [N], dtypes.float64, transient=True)
    s = outer.add_state('main', is_start_block=True)

    inp_r = s.add_read('inp')
    mid1_w = s.add_write('mid1')
    me, mx = s.add_map('pre_map', dict(i='0:N'))
    pre_t = s.add_tasklet('pre', {'x'}, {'y'}, 'y = x * 2.0')
    s.add_memlet_path(inp_r, me, pre_t, dst_conn='x', memlet=Memlet('inp[i]'))
    s.add_memlet_path(pre_t, mx, mid1_w, src_conn='y', memlet=Memlet('mid1[i]'))

    inner = SDFG('inner')
    inner.add_array('mid1', [N], dtypes.float64)
    inner.add_array('mid2', [N], dtypes.float64)
    ist = inner.add_state('inner_main', is_start_block=True)
    me2, mx2 = ist.add_map('inner_map', dict(j='0:N'))
    t = ist.add_tasklet('inner_t', {'x'}, {'y'}, 'y = x + 1.0')
    ist.add_memlet_path(ist.add_read('mid1'), me2, t, dst_conn='x', memlet=Memlet('mid1[j]'))
    ist.add_memlet_path(t, mx2, ist.add_write('mid2'), src_conn='y', memlet=Memlet('mid2[j]'))
    inner.validate()

    nsdfg_node = s.add_nested_sdfg(inner, {'mid1'}, {'mid2'}, symbol_mapping={'N': 'N'})
    s.add_edge(s.add_read('mid1'), None, nsdfg_node, 'mid1', Memlet('mid1[0:N]'))
    s.add_edge(nsdfg_node, 'mid2', s.add_write('mid2'), None, Memlet('mid2[0:N]'))

    mid2_r = s.add_read('mid2')
    out_w = s.add_write('out')
    me3, mx3 = s.add_map('post_map', dict(k='0:N'))
    post_t = s.add_tasklet('post', {'x'}, {'y'}, 'y = x * x')
    s.add_memlet_path(mid2_r, me3, post_t, dst_conn='x', memlet=Memlet('mid2[k]'))
    s.add_memlet_path(post_t, mx3, out_w, src_conn='y', memlet=Memlet('out[k]'))
    outer.validate()
    return outer


def test_inline_preserves_pre_and_post_numerics():
    sdfg = _build_pre_nsdfg_post()
    n = 16
    rng = np.random.default_rng(0xCAFE)
    inp_a = rng.standard_normal(n).astype(np.float64)
    expected = ((inp_a * 2.0) + 1.0)**2

    out_baseline = np.zeros(n, dtype=np.float64)
    copy.deepcopy(sdfg)(inp=inp_a.copy(), out=out_baseline, N=n)
    assert np.allclose(out_baseline, expected, atol=1e-12)

    _apply_inline(sdfg)
    out_inlined = np.zeros(n, dtype=np.float64)
    sdfg(inp=inp_a.copy(), out=out_inlined, N=n)
    assert np.allclose(out_inlined, expected, atol=1e-12)
    assert sum(1 for _ in (n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))) == 0


def _build_non_identity_symbol_mapping_sdfg() -> SDFG:
    """Outer SDFG passes ``M = N // 2`` into the NSDFG's symbol ``M``."""
    outer = SDFG('non_identity_mapping')
    outer.add_array('a', [N], dtypes.float64)
    outer.add_array('b', [N], dtypes.float64)
    s = outer.add_state('main', is_start_block=True)

    inner = SDFG('inner')
    M = dace.symbol('M', dtype=dtypes.int64)
    inner.add_symbol('M', dtypes.int64)
    inner.add_array('a', [N], dtypes.float64)
    inner.add_array('b', [N], dtypes.float64)
    ist = inner.add_state('inner_main', is_start_block=True)
    me, mx = ist.add_map('m', dict(j='0:N'))
    t = ist.add_tasklet('t', {'x'}, {'y'}, 'y = x + M')
    ist.add_memlet_path(ist.add_read('a'), me, t, dst_conn='x', memlet=Memlet('a[j]'))
    ist.add_memlet_path(t, mx, ist.add_write('b'), src_conn='y', memlet=Memlet('b[j]'))
    inner.validate()

    nsdfg_node = s.add_nested_sdfg(inner, {'a'}, {'b'}, symbol_mapping={'N': 'N', 'M': 'N // 2'})
    s.add_edge(s.add_read('a'), None, nsdfg_node, 'a', Memlet('a[0:N]'))
    s.add_edge(nsdfg_node, 'b', s.add_write('b'), None, Memlet('b[0:N]'))
    outer.validate()
    return outer


def test_inline_lowers_non_identity_symbol_mapping_to_iedge_assignment():
    sdfg = _build_non_identity_symbol_mapping_sdfg()
    _apply_inline(sdfg)

    # M must now be a declared outer symbol (picked up from the inner table).
    assert 'M' in sdfg.symbols
    # Some interstate edge must carry an assignment for ``M`` derived from
    # ``N // 2`` (the inline lowers it; codegen may parenthesize it).
    found = False
    for e in sdfg.all_interstate_edges():
        if 'M' in e.data.assignments:
            cleaned = str(e.data.assignments['M']).replace(' ', '').replace('(', '').replace(')', '')
            if 'N//2' in cleaned or 'int_floor(N,2)' in cleaned:
                found = True
                break
    assert found, 'non-identity symbol_mapping entry must surface as iedge assignment'


def _build_renamed_inner_connector_sdfg() -> SDFG:
    """NSDFG has its input array named ``buf`` internally, but the
    outer edge wires array ``data_outer`` to that connector. The inline
    must rename the inner connector to the outer array name."""
    outer = SDFG('renamed_conn')
    outer.add_array('data_outer', [N], dtypes.float64)
    outer.add_array('out_outer', [N], dtypes.float64)
    s = outer.add_state('main', is_start_block=True)

    inner = SDFG('inner')
    inner.add_array('buf', [N], dtypes.float64)
    inner.add_array('result', [N], dtypes.float64)
    ist = inner.add_state('inner_main', is_start_block=True)
    me, mx = ist.add_map('m', dict(j='0:N'))
    t = ist.add_tasklet('t', {'x'}, {'y'}, 'y = x * 3.0')
    ist.add_memlet_path(ist.add_read('buf'), me, t, dst_conn='x', memlet=Memlet('buf[j]'))
    ist.add_memlet_path(t, mx, ist.add_write('result'), src_conn='y', memlet=Memlet('result[j]'))
    inner.validate()

    nsdfg_node = s.add_nested_sdfg(inner, {'buf'}, {'result'}, symbol_mapping={'N': 'N'})
    s.add_edge(s.add_read('data_outer'), None, nsdfg_node, 'buf', Memlet('data_outer[0:N]'))
    s.add_edge(nsdfg_node, 'result', s.add_write('out_outer'), None, Memlet('out_outer[0:N]'))
    outer.validate()
    return outer


def test_inline_renames_inner_connector_to_outer_array_name():
    sdfg = _build_renamed_inner_connector_sdfg()
    n = 12
    rng = np.random.default_rng(0xC0DE)
    a = rng.standard_normal(n).astype(np.float64)
    expected = a * 3.0

    out_baseline = np.zeros(n, dtype=np.float64)
    copy.deepcopy(sdfg)(data_outer=a.copy(), out_outer=out_baseline, N=n)
    assert np.allclose(out_baseline, expected, atol=1e-12)

    _apply_inline(sdfg)
    # After inline no AccessNode in the SDFG should reference the inner
    # connector names anymore.
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode):
            assert node.data != 'buf', 'inner connector buf must have been renamed to data_outer'
            assert node.data != 'result', 'inner connector result must have been renamed to out_outer'

    out_inlined = np.zeros(n, dtype=np.float64)
    sdfg(data_outer=a.copy(), out_outer=out_inlined, N=n)
    assert np.allclose(out_inlined, expected, atol=1e-12)


def test_inline_refuses_inside_map_scope():
    """An NSDFG nested inside a Map scope must not be inlined."""

    @dace.program
    def kernel(a: dace.float64[N, N], b: dace.float64[N, N]):
        for i in dace.map[0:N]:
            for j in range(N):
                b[i, j] = a[i, j] + 1.0

    sdfg = kernel.to_sdfg(simplify=True)
    # Find any NSDFG inside a Map scope and assert can_be_applied is False.
    refused = 0
    for state in sdfg.states():
        for nd in state.nodes():
            if isinstance(nd, nodes.NestedSDFG) and state.scope_dict()[nd] is not None:
                xf = InlineMultistateSDFG()
                xf.nested_sdfg = nd
                xf.expr_index = 0
                assert not xf.can_be_applied(state, 0, sdfg, permissive=False)
                refused += 1
    # If the frontend does not surface any Map-scoped NSDFG here, the test
    # is vacuous - skip rather than silently pass.
    if refused == 0:
        pytest.skip('frontend did not produce a Map-scoped NSDFG for this kernel')


if __name__ == '__main__':
    test_inline_preserves_pre_and_post_numerics()
    test_inline_lowers_non_identity_symbol_mapping_to_iedge_assignment()
    test_inline_renames_inner_connector_to_outer_array_name()
    test_inline_refuses_inside_map_scope()
    print('all ok')
