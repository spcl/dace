# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet


# Constructs an SDFG with two consecutive tasklets
def test_nested_map():
    # Externals (parameters, symbols)
    N = dp.symbol('N')
    n = 20
    input = dp.ndarray([n], dp.int32)
    output = dp.ndarray([n], dp.int32)
    input[:] = dp.int32(5)
    output[:] = dp.int32(0)

    # Construct SDFG
    mysdfg = SDFG('ctasklet_nested_map')
    mysdfg.add_array('A', [N], dp.int32)
    mysdfg.add_array('B', [N], dp.int32)
    state = mysdfg.add_state()
    A_ = state.add_access('A')
    B_ = state.add_access('B')

    omap_entry, omap_exit = state.add_map('omap', dict(k='0:2'))
    map_entry, map_exit = state.add_map('mymap', dict(i='0:N/2'))
    tasklet = state.add_tasklet('mytasklet', {'a'}, {'b'}, 'b = 5*a')
    state.add_edge(map_entry, None, tasklet, 'a', Memlet.simple(A_, 'k*N/2+i'))
    tasklet2 = state.add_tasklet('mytasklet2', {'c'}, {'d'}, 'd = 2*c')
    state.add_edge(tasklet, 'b', tasklet2, 'c', Memlet())
    state.add_edge(tasklet2, 'd', map_exit, None, Memlet.simple(B_, 'k*N/2+i'))

    # Add outer edges
    state.add_edge(A_, None, omap_entry, None, Memlet.simple(A_, '0:N'))
    state.add_edge(omap_entry, None, map_entry, None, Memlet.simple(A_, 'k*N/2:(k+1)*N/2'))
    state.add_edge(map_exit, None, omap_exit, None, Memlet.simple(B_, 'k*N/2:(k+1)*N/2'))
    state.add_edge(omap_exit, None, B_, None, Memlet.simple(B_, '0:N'))

    # Fill missing connectors
    mysdfg.fill_scope_connectors()
    mysdfg.validate()

    mysdfg(A=input, B=output, N=n)

    diff = np.linalg.norm(10 * input - output) / n
    assert diff <= 1e-5


def test_nested_sdfg():
    # Externals (parameters, symbols)
    N = dp.symbol('N')
    n = 20
    input = dp.ndarray([n], dp.int32)
    output = dp.ndarray([n], dp.int32)
    input[:] = dp.int32(5)
    output[:] = dp.int32(0)

    # Construct outer SDFG
    mysdfg = SDFG('ctasklet_nested_sdfg')
    mysdfg.add_array('A', [N], dp.int32)
    mysdfg.add_array('B', [N], dp.int32)
    state = mysdfg.add_state()
    A_ = state.add_access('A')
    B_ = state.add_access('B')

    # Construct inner SDFG
    nsdfg = dp.SDFG('ctasklet_nested_sdfg_inner')
    nsdfg.add_array('a', [N], dp.int32)
    nsdfg.add_array('b', [N], dp.int32)
    nstate = nsdfg.add_state()
    a = nstate.add_access('a')
    b = nstate.add_access('b')
    map_entry, map_exit = nstate.add_map('mymap', dict(i='0:N/2'))
    tasklet = nstate.add_tasklet('mytasklet', {'aa'}, {'bb'}, 'bb = 5*aa')
    nstate.add_memlet_path(a, map_entry, tasklet, dst_conn='aa', memlet=Memlet('a[k*N/2+i]'))
    tasklet2 = nstate.add_tasklet('mytasklet2', {'cc'}, {'dd'}, 'dd = 2*cc')
    nstate.add_edge(tasklet, 'bb', tasklet2, 'cc', Memlet())
    nstate.add_memlet_path(tasklet2, map_exit, b, src_conn='dd', memlet=Memlet('b[k*N/2+i]'))

    # Add outer edges
    omap_entry, omap_exit = state.add_map('omap', dict(k='0:2'))
    nsdfg_node = state.add_nested_sdfg(nsdfg, {'a'}, {'b'})
    state.add_memlet_path(A_, omap_entry, nsdfg_node, dst_conn='a', memlet=Memlet('A[0:N]'))
    state.add_memlet_path(nsdfg_node, omap_exit, B_, src_conn='b', memlet=Memlet('B[0:N]'))

    mysdfg.validate()
    mysdfg(A=input, B=output, N=n)

    diff = np.linalg.norm(10 * input - output) / n
    assert diff <= 1e-5

    mysdfg.simplify()

    mysdfg(A=input, B=output, N=n)

    diff = np.linalg.norm(10 * input - output) / n
    assert diff <= 1e-5


if __name__ == '__main__':
    test_nested_map()
    test_nested_sdfg()
