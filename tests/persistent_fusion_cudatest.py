# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import networkx as nx
import dace
from dace.sdfg.graph import SubgraphView
from dace.transformation.subgraph import GPUPersistentKernel
import pytest

N = dace.symbol('N')
nnz = dace.symbol('nnz')


def _make_sdfg():
    bfs = dace.SDFG('bfs')

    # Inputs to the BFS SDFG
    bfs.add_array('col_index', shape=[nnz], dtype=dace.int32)
    bfs.add_array('row_index', shape=[N + 1], dtype=dace.int32)
    bfs.add_scalar('root', dtype=dace.int32)
    bfs.add_array('result', shape=[N], dtype=dace.int32)

    # Transients fot interstate data transfers
    # TODO: Replace may_alias with better code generation
    bfs.add_transient('count1', shape=[1], dtype=dace.int32, may_alias=True)
    bfs.add_transient('frontier1', shape=[N], dtype=dace.int32, may_alias=True)

    bfs.add_transient('count2', shape=[1], dtype=dace.int32, may_alias=True)
    bfs.add_transient('frontier2', shape=[N], dtype=dace.int32, may_alias=True)

    # Transient streams to accommodate dynamic size of frontier arrays
    bfs.add_stream('stream1', dtype=dace.int32, transient=True, buffer_size=N)
    bfs.add_stream('stream2', dtype=dace.int32, transient=True, buffer_size=N)

    # Transients needed for update states
    bfs.add_transient('temp_ids1', shape=[1], dtype=dace.int32, storage=dace.StorageType.Register)
    bfs.add_transient('temp_ide1', shape=[1], dtype=dace.int32, storage=dace.StorageType.Register)

    bfs.add_transient('temp_ids2', shape=[1], dtype=dace.int32, storage=dace.StorageType.Register)
    bfs.add_transient('temp_ide2', shape=[1], dtype=dace.int32, storage=dace.StorageType.Register)

    # Adding states
    # init data
    s_init = bfs.add_state('init')

    # copy of the states because we don't want to copy the data
    s_reset1 = bfs.add_state('reset1')
    s_update1 = bfs.add_state('update1')

    s_reset2 = bfs.add_state('reset2')
    s_update2 = bfs.add_state('update2')

    # end state to make transformation work
    s_end = bfs.add_state('end')

    # Connecting states with appropriate conditions and depth updates
    bfs.add_edge(s_init, s_reset1, dace.InterstateEdge(None, {'depth': '1'}))
    bfs.add_edge(s_reset1, s_update1, dace.InterstateEdge(None))
    bfs.add_edge(s_update1, s_reset2, dace.InterstateEdge('count2[0] > 0', {'depth': 'depth + 1'}))
    bfs.add_edge(s_update1, s_end, dace.InterstateEdge('count2[0] <= 0'))
    bfs.add_edge(s_reset2, s_update2, dace.InterstateEdge(None))
    bfs.add_edge(s_update2, s_reset1, dace.InterstateEdge('count1[0] > 0', {'depth': 'depth + 1'}))
    bfs.add_edge(s_update2, s_end, dace.InterstateEdge('count1[0] <= 0'))

    # =============================================================
    # State: init
    # Filling init state with init of result, frontier1, and count1

    root_in = s_init.add_read('root')

    count1_out = s_init.add_write('count1')
    result_out = s_init.add_write('result')
    frontier_out = s_init.add_write('frontier1')

    s_init.add_memlet_path(root_in, frontier_out, memlet=dace.Memlet.simple(root_in.data, '0', other_subset_str='0'))

    tasklet = s_init.add_tasklet(
        'set_count1',
        {},
        {'out'},
        'out = 1',
    )

    s_init.add_memlet_path(tasklet, count1_out, src_conn='out', memlet=dace.Memlet.simple(count1_out.data, '0'))

    map_entry, map_exit = s_init.add_map(
        'set_result_map',
        dict(i='0:N'),
    )

    tasklet = s_init.add_tasklet('set_result', {'root_idx'}, {'result_out'}, 'result_out = 0 if i == root_idx else -1')

    s_init.add_memlet_path(root_in, map_entry, tasklet, dst_conn='root_idx', memlet=dace.Memlet.simple(root_in.data, '0'))

    s_init.add_memlet_path(tasklet,
                        map_exit,
                        result_out,
                        src_conn='result_out',
                        memlet=dace.Memlet.simple(result_out.data, 'i'))

    # -------------------------------------------------------------

    # =============================================================
    # State: reset
    # Filling reset states, respective count is reset to 0

    count2_out = s_reset1.add_write('count2')
    init_scalar(s_reset1, count2_out, 0)

    count1_out = s_reset2.add_write('count1')
    init_scalar(s_reset2, count1_out, 0)

    # -------------------------------------------------------------

    # Filling update states, only difference is which frontier/count they read/write from/to

    front_in = s_update1.add_read('frontier1')
    count_in = s_update1.add_read('count1')

    front_out = s_update1.add_write('frontier2')
    count_out = s_update1.add_write('count2')

    stream2_io = s_update1.add_access('stream2')

    temp_ids1_io = s_update1.add_access('temp_ids1')
    temp_ide1_io = s_update1.add_access('temp_ide1')

    fill_update_state(s_update1, front_in, count_in, front_out, count_out, stream2_io, temp_ids1_io, temp_ide1_io)

    front_in = s_update2.add_read('frontier2')
    count_in = s_update2.add_read('count2')

    front_out = s_update2.add_write('frontier1')
    count_out = s_update2.add_write('count1')

    stream1_io = s_update2.add_access('stream1')

    temp_ids2_io = s_update2.add_access('temp_ids2')
    temp_ide2_io = s_update2.add_access('temp_ide2')

    fill_update_state(s_update2, front_in, count_in, front_out, count_out, stream1_io, temp_ids2_io, temp_ide2_io)

    # validate and generate sdfg
    bfs.fill_scope_connectors()
    bfs.validate()
    return bfs, s_init

# -----------------------------
# Helper functions to init data
# -----------------------------


def init_scalar(state, node, value):
    tasklet = state.add_tasklet('set_%s' % node.data, {}, {'out'}, '''
out = %d
        ''' % value)

    state.add_memlet_path(tasklet, node, src_conn='out', memlet=dace.Memlet.simple(node.data, '0'))


# Here the state is duplicated so the memory doesn't have to be copied from one to another
# array.
def fill_update_state(state, front_in, front_in_count, front_out, front_out_count, s_frontier_io, temp_ids_io,
                      temp_ide_io):
    row_index_in = state.add_read('row_index')
    col_index_in = state.add_read('col_index')
    result_in = state.add_read('result')

    result_out = state.add_write('result')

    # Map iterates over all nodes in frontier
    front_enter, front_exit = state.add_map('frontier_map', dict(x='0:count_val'))

    state.add_memlet_path(front_in_count,
                          front_enter,
                          dst_conn='count_val',
                          memlet=dace.Memlet.simple(front_in_count.data, '0'))

    # Find number of neighbors of current node
    t_find_range = state.add_tasklet('find_range', ['f_x', 'row'], ['index_start', 'index_end'], '''
index_start = row[f_x]
index_end = row[f_x + 1]
        ''')

    # iterate over all neighbors of current node
    neigh_enter, neigh_exit = state.add_map('neighbor_map', dict(i='map_start:map_end'))

    state.add_memlet_path(t_find_range,
                          temp_ids_io,
                          src_conn='index_start',
                          memlet=dace.Memlet.simple(temp_ids_io.data, '0'))

    state.add_memlet_path(t_find_range,
                          temp_ide_io,
                          src_conn='index_end',
                          memlet=dace.Memlet.simple(temp_ide_io.data, '0'))

    state.add_memlet_path(temp_ids_io,
                          neigh_enter,
                          dst_conn='map_start',
                          memlet=dace.Memlet.simple(temp_ids_io.data, '0'))

    state.add_memlet_path(temp_ide_io,
                          neigh_enter,
                          dst_conn='map_end',
                          memlet=dace.Memlet.simple(temp_ide_io.data, '0'))

    state.add_memlet_path(row_index_in,
                          front_enter,
                          t_find_range,
                          dst_conn='row',
                          memlet=dace.Memlet.simple(row_index_in.data, '0:N', num_accesses=2))

    state.add_memlet_path(front_in,
                          front_enter,
                          t_find_range,
                          dst_conn='f_x',
                          memlet=dace.Memlet.simple(front_in.data, 'x'))

    # update tasklet (this is where the magic happens)
    t_add_neighbor = state.add_tasklet(
        'add_neighbor', ['neighbor', 'res'], ['new_res', 'add_to_count', 'add_to_front'], '''
if res[neighbor] == -1:
  new_res[neighbor] = depth
  add_to_front = neighbor
  add_to_count = 1
        ''')

    state.add_memlet_path(col_index_in,
                          front_enter,
                          neigh_enter,
                          t_add_neighbor,
                          dst_conn='neighbor',
                          memlet=dace.Memlet.simple(col_index_in.data, 'i'))

    state.add_memlet_path(result_in,
                          front_enter,
                          neigh_enter,
                          t_add_neighbor,
                          dst_conn='res',
                          memlet=dace.Memlet.simple(result_in.data, '0:N', num_accesses=1))

    state.add_memlet_path(t_add_neighbor,
                          neigh_exit,
                          front_exit,
                          front_out_count,
                          src_conn='add_to_count',
                          memlet=dace.Memlet.simple(front_out_count.data,
                                                    '0',
                                                    num_accesses=-1,
                                                    wcr_str='lambda a, b: a + b'))

    state.add_memlet_path(t_add_neighbor,
                          neigh_exit,
                          front_exit,
                          s_frontier_io,
                          src_conn='add_to_front',
                          memlet=dace.Memlet.simple(s_frontier_io.data, '0', num_accesses=-1))

    state.add_memlet_path(t_add_neighbor,
                          neigh_exit,
                          front_exit,
                          result_out,
                          src_conn='new_res',
                          memlet=dace.Memlet.simple(result_out.data, '0:N', num_accesses=-1))

    state.add_memlet_path(s_frontier_io, front_out, memlet=dace.Memlet.simple(front_out.data, '0'))



@pytest.mark.gpu
def test_persistent_fusion():
    sdfg, s_init = _make_sdfg()

    sdfg.apply_gpu_transformations(validate=False, simplify=False)  # Only validate after fusion

    # All nodes but copy-in, copy-out, and init
    content_nodes = set(sdfg.nodes()) - {sdfg.start_state, sdfg.sink_nodes()[0], s_init}

    subgraph = SubgraphView(sdfg, content_nodes)
    transform = GPUPersistentKernel()
    transform.setup_match(subgraph)
    transform.kernel_prefix = 'bfs'
    transform.apply(sdfg)

    subgraph = SubgraphView(sdfg, [s_init])
    transform = GPUPersistentKernel()
    transform.setup_match(subgraph)
    transform.kernel_prefix = 'init'
    transform.apply(sdfg)

    sdfg.validate()

    V = 1024
    E = 2048
    srcnode = 0
    vtype = np.uint32

    # Generate a random graph
    graph = nx.gnm_random_graph(V, E, seed=42)
    E = E * 2

    # Extract adjacency matrix
    M = nx.to_scipy_sparse_matrix(graph, dtype=vtype).tocsr()
    assert M.nnz == E

    G_row = np.ndarray([V + 1], dtype=vtype)
    G_col = np.ndarray([E], dtype=vtype)
    G_row[:] = M.indptr
    G_col[:] = M.indices

    # Regression
    reference = nx.shortest_path(graph, source=srcnode)
    reference = np.array([len(reference[v]) - 1 if v in reference else np.iinfo(vtype).max for v in range(V)])

    print('Breadth-First Search (E = {}, V = {})'.format(E, V))

    # Allocate output arrays
    depth = np.ndarray([V], vtype)

    sdfg(row_index=G_row, col_index=G_col, result=depth, root=srcnode, N=V, nnz=E)

    assert np.allclose(depth, reference), "Result doesn't match!"

def test_persistent_fusion_interstate():
    N = dace.symbol('N', dtype=dace.int64)


    @dace.program(auto_optimize=False, device=dace.DeviceType.GPU)
    def func(A: dace.float64[N], B: dace.float64[N]):
        a = 10.2

        for t in range(1, 10):
            if t < N:
                A[:] = (A + B + a) / 2
                a += 1

    # Initialization
    N = 100
    A = np.random.rand(N)
    B = np.random.rand(N)

    sdfg = func.to_sdfg()
    sdfg.apply_gpu_transformations()
    content_nodes = set(sdfg.nodes()) - {sdfg.start_state, sdfg.sink_nodes()[0]}
    subgraph = SubgraphView(sdfg, content_nodes)

    transform = GPUPersistentKernel()
    transform.setup_match(subgraph)
    transform.kernel_prefix = 'stuff'
    transform.apply(sdfg)

    aref = np.copy(A)
    func.f(aref, B)

    sdfg(A=A, B=B, N=N)
    
    assert np.allclose(A, aref)


# Actual execution
if __name__ == "__main__":
    test_persistent_fusion()
    test_persistent_fusion_interstate()
