# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.dataflow import MapFusion

def test_hierarchical_sdfg_with_loops_and_map_fusion():
    sdfg = dace.SDFG('hierarchical_sdfg_with_loops_and_map_fusion')

    sdfg.add_array('A', [32, 32, 32], dace.float32)
    sdfg.add_array('B', [32, 32, 32], dace.float32)
    sdfg.add_array('C', [32, 32, 32], dace.float32)
    sdfg.add_transient('tmp', [32, 32], dace.float32)

    init_state = sdfg.add_state('init_state', is_start_block=True)
    loop = LoopRegion('loop', 'k < 32', 'k', 'k = 1', 'k = k + 1')
    sdfg.add_node(loop)
    loop_state = loop.add_state('loop_state', is_start_block=True)
    exit_state = sdfg.add_state('exit_state')
    sdfg.add_edge(init_state, loop, dace.InterstateEdge())
    sdfg.add_edge(loop, exit_state, dace.InterstateEdge())

    acc_a = loop_state.add_access('A')
    acc_b = loop_state.add_access('B')
    acc_rc = loop_state.add_access('C')
    acc_wc = loop_state.add_access('C')
    acc_tmp = loop_state.add_access('tmp')
    loop_state.add_mapped_tasklet('t1', {'i': '0:32', 'j': '0:32'},
                                  {
                                      'i1': dace.Memlet('A[i,j,k]'),
                                      'i2': dace.Memlet('B[i,j,k]'),
                                      'i3': dace.Memlet('C[i,j,k-1]'),
                                  },
                                  'o1 = i1 + i2 + i3',
                                  {'o1': dace.Memlet('tmp[i,j]')},
                                  input_nodes={'A': acc_a, 'B': acc_b, 'C': acc_rc},
                                  output_nodes={'tmp': acc_tmp},
                                  external_edges=True)
    loop_state.add_mapped_tasklet('t2', {'i': '0:32', 'j': '0:32'},
                                  {
                                      'i1': dace.Memlet('tmp[i,j]'),
                                  },
                                  'o1 = i1 * i1',
                                  {'o1': dace.Memlet('C[i,j,k]')},
                                  input_nodes={'tmp': acc_tmp},
                                  output_nodes={'C': acc_wc},
                                  external_edges=True)

    sdfg.apply_transformations_repeated(MapFusion)

    print(sdfg)


if __name__ == '__main__':
    test_hierarchical_sdfg_with_loops_and_map_fusion()
