# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for `DataflowGraphView.unordered_arglist`'s
`AccessNode/CodeNode -> ExitNode` branch.

The branch collects the external arrays a map writes through its MapExit.
It iterates the matching out-edges of the MapExit and, for each one, has to
name the actual destination array. Pre-fix it trusted `oedge.data.data`; an
outgoing memlet that still names an inner transient (instead of the outer
destination) then dropped the real array -- and its shape/stride symbols --
from the arglist, and codegen later emitted a kernel signature that
referenced an undeclared identifier.

The fix resolves the destination from the memlet path's terminal AccessNode
(equivalently, the memlet tree's root in this branch).
"""
import dace
from dace.sdfg import nodes as nd


def _build_src_relative_exit_sdfg() -> dace.SDFG:
    """Construct the pathological shape: outgoing memlet from MapExit to
    AccessNode('D') has `data='tmp'` (the inner transient), not `data='D'`.
    """
    sdfg = dace.SDFG('exit_write_src_rel')
    sdfg.add_array('D', [10], dace.float64)
    sdfg.add_scalar('tmp', dace.float64, transient=True)

    state = sdfg.add_state('s', is_start_block=True)
    me, mx = state.add_map('m', dict(i='0:10'))
    t = state.add_tasklet('w', set(), {'_o'}, '_o = 1.0')
    tmp = state.add_access('tmp')
    d_an = state.add_write('D')

    state.add_edge(me, None, t, None, dace.Memlet())
    state.add_edge(t, '_o', tmp, None, dace.Memlet('tmp[0]'))
    # Source-relative memlet INSIDE the map: names the inner transient.
    state.add_edge(tmp, None, mx, 'IN_D', dace.Memlet(data='tmp', subset='0', other_subset='i'))
    # Source-relative memlet OUTSIDE the map: still names the inner transient
    # (this is the pathological field -- the fix tolerates it instead of
    # propagating 'tmp' into the arglist).
    state.add_edge(mx, 'OUT_D', d_an, None, dace.Memlet(data='tmp', subset='0'))
    mx.add_in_connector('IN_D', force=True)
    mx.add_out_connector('OUT_D', force=True)
    sdfg.validate()
    return sdfg


def test_arglist_resolves_outer_destination_from_source_relative_outgoing_memlet():
    """`unordered_arglist` must surface the OUTER destination array (``D``)
    even when the outgoing memlet from MapExit names an inner transient
    (``tmp``). Pre-fix it returned 'tmp' in the arglist and dropped 'D' --
    a downstream codegen would then emit a kernel signature that references
    'D' without declaring it (`identifier "D" is undefined`).
    """
    sdfg = _build_src_relative_exit_sdfg()
    state = next(iter(sdfg.states()))
    me = next(n for n in state.nodes() if isinstance(n, nd.MapEntry))

    arglist = state.scope_subgraph(me).arglist()
    assert 'D' in arglist, f"outer destination 'D' missing from arglist: {sorted(arglist.keys())}"
    # The inner transient must NOT appear in the kernel arglist (it lives
    # inside the scope; arglist exposes only externally-visible arrays).
    assert 'tmp' not in arglist, f"inner transient 'tmp' leaked into arglist: {sorted(arglist.keys())}"


if __name__ == '__main__':
    test_arglist_resolves_outer_destination_from_source_relative_outgoing_memlet()
