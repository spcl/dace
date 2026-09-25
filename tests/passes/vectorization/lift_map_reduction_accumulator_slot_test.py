# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.transformation.passes.vectorization.lift_map_reduction import LiftMapReductionToReduce
from dace.transformation.passes.vectorization.utils.reductions import recognize_map_reduction


def rmw_map(seed: str, slot: str, accs: tuple[str, ...] = ('x', )) -> dace.SDFG:
    """``acc[seed] = 0; for k in map[0:16]: acc[slot] = acc[slot] + a[k]`` per acc, body in a nested SDFG."""
    body = dace.SDFG('body')
    bs = body.add_state()
    sdfg = dace.SDFG(f'rmw_{seed}_{slot}_{"_".join(accs)}')
    sdfg.add_array('a', [16], dace.float64)
    st = sdfg.add_state()
    me, mx = st.add_map('m', {'k': '0:16'})
    for name in ['v'] + [f'{c}_{io}' for c in accs for io in 'io']:
        body.add_scalar(name, dace.float64)
    ns = st.add_nested_sdfg(body, {'v': None, **{f'{c}_i': None for c in accs}}, {f'{c}_o': None for c in accs})
    st.add_memlet_path(st.add_read('a'), me, ns, dst_conn='v', memlet=dace.Memlet('a[k]'))
    for c in accs:
        sdfg.add_array(c, [16], dace.float64)
        t = bs.add_tasklet(c, {'_i': None, '_v': None}, {'_o': None}, '_o = _i + _v')
        bs.add_edge(bs.add_read(f'{c}_i'), None, t, '_i', dace.Memlet(f'{c}_i[0]'))
        bs.add_edge(bs.add_read('v'), None, t, '_v', dace.Memlet('v[0]'))
        bs.add_edge(t, '_o', bs.add_write(f'{c}_o'), None, dace.Memlet(f'{c}_o[0]'))
        acc_in = st.add_access(c)
        st.add_edge(st.add_tasklet('init', {}, {'o': None}, 'o = 0.0'), 'o', acc_in, None, dace.Memlet(f'{c}[{seed}]'))
        st.add_memlet_path(acc_in, me, ns, dst_conn=f'{c}_i', memlet=dace.Memlet(f'{c}[{slot}]'))
        st.add_memlet_path(ns, mx, st.add_write(c), src_conn=f'{c}_o', memlet=dace.Memlet(f'{c}[{slot}]'))
    return sdfg


def test_elementwise_in_place_update_is_not_lifted() -> None:
    sdfg = rmw_map(seed='0', slot='k')
    assert LiftMapReductionToReduce(rmw_only=True).apply_pass(sdfg, {}) is None
    a, x = np.arange(16.0), np.ones(16)
    sdfg(a=a, x=x)
    assert np.allclose(x, np.r_[0.0, np.ones(15)] + a)


def test_reduction_into_element_3_is_written_back_to_element_3() -> None:
    sdfg = rmw_map(seed='3', slot='3')
    assert LiftMapReductionToReduce(rmw_only=True).apply_pass(sdfg, {}) == 1
    a, x = np.arange(16.0), np.ones(16)
    sdfg(a=a, x=x)
    assert np.allclose(x, np.r_[1.0, 1.0, 1.0, 120.0, np.ones(12)])


def test_reduction_seeded_at_another_element_is_not_lifted() -> None:
    sdfg = rmw_map(seed='0', slot='3')
    assert LiftMapReductionToReduce(rmw_only=True).apply_pass(sdfg, {}) is None
    a, x = np.arange(16.0), np.ones(16)
    sdfg(a=a, x=x)
    assert np.allclose(x, np.r_[0.0, 1.0, 1.0, 121.0, np.ones(12)])


def test_first_written_accumulator_is_recognised_whatever_the_names_hash_to() -> None:
    states = [rmw_map('0', '0', accs).start_block for accs in (('y', 'x'), ('x', 'y'))]
    picks = [
        recognize_map_reduction(s, next(n for n in s.nodes() if isinstance(n, dace.nodes.MapEntry))) for s in states
    ]
    assert [p.accumulator for p in picks] == ['y', 'x']
