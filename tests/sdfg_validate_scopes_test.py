# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import dace


def test_connector_mismatch():
    """Try to detect invalid scopes in SDFG"""
    sdfg = dace.SDFG('a')
    sdfg.add_array('A', [1], dace.float32)

    state = sdfg.add_state()
    me, mx = state.add_map('b', dict(i="0:1"))
    A = state.add_access('A')
    T = state.add_tasklet('T', {'a'}, {}, 'printf("%f", a)')

    me.add_in_connector('IN_a')
    me.add_out_connector('OUT_b')
    state.add_edge(A, None, me, 'IN_a', dace.Memlet.from_array(A.data, A.desc(sdfg)))
    state.add_edge(me, 'OUT_b', T, 'a', dace.Memlet.simple(A, '0'))
    state.add_edge(T, None, mx, None, dace.Memlet())

    with pytest.raises(dace.sdfg.InvalidSDFGError):
        sdfg.validate()


if __name__ == '__main__':
    test_connector_mismatch()
