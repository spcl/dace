# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SubgraphFusion`` must not leave two unordered writes to the same outer slot.

Reduced from ``tests/npbench/weather_stencils/vadv_test.py``.
"""
import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.transformation.subgraph import SubgraphFusion

N = dace.symbol('N')
K = dace.symbol('K')
k = dace.symbol('k')


def _vadv_dcol_sdfg(copy_back: bool) -> dace.SDFG:
    """``dcol[:, k] = A[:, k] + 1`` then ``dcol[:, k] = dcol[:, k] * s`` via a scratch transient.

    :param copy_back: Write the scaled transient back onto ``dcol[:, k]``. Without it nothing else
                      writes the slot, so the first store is the live one.
    """
    sdfg = dace.SDFG('vadv_dcol')
    for sym in ('N', 'K', 'k'):
        sdfg.add_symbol(sym, dace.int64)
    sdfg.add_array('A', [N, K], dace.float64)
    sdfg.add_array('out', [N, K], dace.float64)
    sdfg.add_transient('dcol', [N, K], dace.float64)
    sdfg.add_transient('dcol_scaled', [N], dace.float64)
    sdfg.add_scalar('s', dace.float64)

    state = sdfg.add_state('sweep')
    a, sweep_out = state.add_read('A'), state.add_access('dcol')
    scalar, scaled = state.add_read('s'), state.add_access('dcol_scaled')

    state.add_mapped_tasklet('sum', {'i': '0:N'}, {'_a': dace.Memlet('A[i, k]')},
                             '_b = _a + 1.0', {'_b': dace.Memlet('dcol[i, k]')},
                             external_edges=True,
                             input_nodes={'A': a},
                             output_nodes={'dcol': sweep_out})
    state.add_mapped_tasklet('mult', {'i': '0:N'}, {
        '_b': dace.Memlet('dcol[i, k]'),
        '_s': dace.Memlet('s[0]')
    },
                             '_o = _b * _s', {'_o': dace.Memlet('dcol_scaled[i]')},
                             external_edges=True,
                             input_nodes={
                                 'dcol': sweep_out,
                                 's': scalar
                             },
                             output_nodes={'dcol_scaled': scaled})
    if copy_back:
        state.add_edge(scaled, None, state.add_access('dcol'), None, dace.Memlet('dcol_scaled[0:N] -> [0:N, k]'))

    reader = sdfg.add_state_after(state, 'consume')
    reader.add_mapped_tasklet('use', {'i': '0:N'}, {'_d': dace.Memlet('dcol[i, k]')},
                              '_o = _d', {'_o': dace.Memlet('out[i, k]')},
                              external_edges=True)
    sdfg.validate()
    return sdfg


def _fuse_first_state(sdfg: dace.SDFG) -> dace.SDFGState:
    state = sdfg.states()[0]
    subgraph = SubgraphView(state, state.nodes())
    fusion = SubgraphFusion()
    fusion.setup_match(subgraph, sdfg.cfg_id, sdfg.node_id(state))
    assert fusion.can_be_applied(sdfg, subgraph)
    fusion.apply(sdfg)
    sdfg.validate()
    return state


def _writers_of(state: dace.SDFGState, name: str):
    return [(e.src, e.data.get_dst_subset(e, state)) for node in state.data_nodes() if node.data == name
            for e in state.in_edges(node) if not e.data.is_empty()]


def test_dead_intermediate_store_is_not_materialized():
    state = _fuse_first_state(_vadv_dcol_sdfg(copy_back=True))
    writers = _writers_of(state, 'dcol')
    assert len(writers) == 1, f'dcol has {len(writers)} unordered writers: {writers}'
    assert not isinstance(writers[0][0], nodes.MapExit)


def test_live_intermediate_store_survives():
    state = _fuse_first_state(_vadv_dcol_sdfg(copy_back=False))
    assert any(isinstance(src, nodes.MapExit) for src, _ in _writers_of(state, 'dcol'))


def test_fused_result_is_numerically_correct():
    sdfg = _vadv_dcol_sdfg(copy_back=True)
    _fuse_first_state(sdfg)

    a = np.copy(np.arange(16 * 3, dtype=np.float64).reshape(16, 3))
    out = np.zeros((16, 3), dtype=np.float64)
    sdfg(A=a, out=out, s=2.0, N=16, K=3, k=1)
    np.testing.assert_allclose(out[:, 1], (a[:, 1] + 1.0) * 2.0)


if __name__ == '__main__':
    test_dead_intermediate_store_is_not_materialized()
    test_live_intermediate_store_survives()
    test_fused_result_is_numerically_correct()
