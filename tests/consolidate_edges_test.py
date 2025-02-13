# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg.utils import consolidate_edges

sdfg = dace.SDFG('cetest')
sdfg.add_array('A', [50], dace.float32)
sdfg.add_array('B', [48], dace.float32)
state = sdfg.add_state()

r = state.add_read('A')
me, mx = state.add_map('map', dict(i='1:49'))
t = state.add_tasklet('op', {'a', 'b', 'c'}, {'out'}, 'out = a + b + c')
w = state.add_write('B')

state.add_memlet_path(r, me, t, dst_conn='a', memlet=dace.Memlet.simple('A', 'i-1'))
state.add_memlet_path(r, me, t, dst_conn='b', memlet=dace.Memlet.simple('A', 'i'))
state.add_memlet_path(r, me, t, dst_conn='c', memlet=dace.Memlet.simple('A', 'i+1'))
state.add_memlet_path(t, mx, w, src_conn='out', memlet=dace.Memlet.simple('B', 'i-1'))


def test_consolidate_edges():
    assert len(state.edges()) == 8
    consolidate_edges(sdfg)
    assert len(state.edges()) == 6


if __name__ == '__main__':
    test_consolidate_edges()
