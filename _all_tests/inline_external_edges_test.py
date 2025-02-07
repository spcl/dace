# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

sdfg = dace.SDFG('inline_external_edges_test')
sdfg.add_array('L', [2], dace.float32)
sdfg.add_array('M', [2], dace.float32)

state = sdfg.add_state()
L_in = state.add_read('L')
L_out = state.add_write('L')
M_out = state.add_write('M')

me, mx = state.add_map('Nested_map', dict(i='0:2'))

# Nested SDFG
nsdfg = dace.SDFG('nested_iee_test')
nsdfg.add_array('local', [1], dace.float32)
nsdfg.add_array('m', [1], dace.float32)
nstate = nsdfg.add_state()
t = nstate.add_tasklet('init_local', {}, {'l'}, 'l = 2')
L_inout = nstate.add_access('local')
M_localout = nstate.add_access('m')
nstate.add_edge(t, 'l', L_inout, None, dace.Memlet.simple('local', '0'))
t2 = nstate.add_tasklet('set_m', {'l'}, {'mm'}, 'mm = l + 5')
nstate.add_edge(L_inout, None, t2, 'l', dace.Memlet.simple('local', '0'))
nstate.add_edge(t2, 'mm', M_localout, None, dace.Memlet.simple('m', '0'))
###############

nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'local'}, {'local', 'm'})
state.add_memlet_path(L_in, me, nsdfg_node, memlet=dace.Memlet.simple('L', 'i'), dst_conn='local')
state.add_memlet_path(nsdfg_node, mx, L_out, memlet=dace.Memlet.simple('L', 'i'), src_conn='local')
state.add_memlet_path(nsdfg_node, mx, M_out, memlet=dace.Memlet.simple('M', 'i'), src_conn='m')


def test():
    L = np.random.rand(2).astype(np.float32)
    M = np.random.rand(2).astype(np.float32)

    sdfg.simplify()
    sdfg(L=L, M=M)

    expected = np.array([2.0, 2.0, 7.0, 7.0])
    result = np.array([L[0], L[1], M[0], M[1]])
    diff = np.linalg.norm(expected - result)
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
