# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.memlet import Memlet
from dace.dtypes import Language, StorageType
from dace.properties import CodeProperty
from dace.sdfg import propagation
from dace.sdfg.sdfg import InterstateEdge
from dace.symbolic import symbol
import dace

N = dace.symbol('N')
M = dace.symbol('M')


def memlet_check_parameters(memlet, volume, dynamic, subsets):
    if memlet.volume != volume:
        raise RuntimeError('Expected volume of {}, got {}'.format(volume, memlet.volume))
    elif dynamic and not memlet.dynamic:
        raise RuntimeError('Expected dynamic volume, got static')
    elif memlet.dynamic and not dynamic:
        raise RuntimeError('Expected static volume, got dynamic')

    if len(subsets) != memlet.subset.dims():
        raise RuntimeError('Expected subset of dim {}, got {}'.format(len(subsets), memlet.subset.dims()))

    for i in range(len(subsets)):
        if subsets[i] != memlet.subset.ranges[i]:
            raise RuntimeError('Expected subset {} at dim {}, got {}'.format(subsets[i], i, memlet.subset.ranges[i]))


def state_check_executions(state, expected, expected_dynamic=False):
    if state.executions != expected:
        raise RuntimeError('Expected {} executions, got {}'.format(expected, state.executions))
    elif expected_dynamic and not state.dynamic_executions:
        raise RuntimeError('Expected dynamic executions, got static')
    elif state.dynamic_executions and not expected_dynamic:
        raise RuntimeError('Expected static executions, got dynamic')


def make_nested_sdfg():
    sdfg = dace.SDFG('vol_propagation_nested')

    assign_loop_bound = sdfg.add_state('assign')
    guard_state = sdfg.add_state('guard')
    loop_state = sdfg.add_state('for')
    end_state = sdfg.add_state('endfor')

    sdfg.add_edge(assign_loop_bound, guard_state, InterstateEdge(assignments={'i': '0'}))
    sdfg.add_edge(guard_state, loop_state,
                  InterstateEdge(condition=CodeProperty.from_string('i < loop_bound', language=Language.Python)))
    sdfg.add_edge(loop_state, guard_state, InterstateEdge(assignments={'i': 'i+1'}))
    sdfg.add_edge(guard_state, end_state,
                  InterstateEdge(condition=CodeProperty.from_string('not (i < loop_bound)', language=Language.Python)))

    in_bound = assign_loop_bound.add_stream('IN_bound', dace.int32, storage=StorageType.FPGA_Local)
    loop_bound = assign_loop_bound.add_scalar('loop_bound',
                                              dace.int32,
                                              transient=True,
                                              storage=StorageType.FPGA_Registers)
    assign_loop_bound.add_memlet_path(in_bound, loop_bound, memlet=Memlet.simple(loop_bound, '0'))

    in_a = loop_state.add_array('IN_a', [N], dace.int32, storage=StorageType.FPGA_Global)
    out_stream = loop_state.add_stream('OUT_stream', dace.int32, storage=StorageType.FPGA_Local)
    tasklet2 = loop_state.add_tasklet('compute', {'_IN_a'}, {'_OUT_stream'}, '_OUT_stream = _IN_a[0]')
    loop_state.add_memlet_path(in_a, tasklet2, dst_conn='_IN_a', memlet=Memlet.simple(in_a, '0:N'))
    loop_state.add_memlet_path(tasklet2, out_stream, src_conn='_OUT_stream', memlet=Memlet.simple(out_stream, '0'))

    return sdfg


def make_sdfg():
    sdfg = dace.SDFG('vol_propagation')

    sdfg.add_symbol('N', dace.int32)
    sdfg.add_symbol('M', dace.int32)

    state = sdfg.add_state('main')

    a_in = state.add_array('A_in', [N], dace.int32, storage=StorageType.FPGA_Global)
    bound_pipe = state.add_stream('bound_in', dace.int32, transient=True, storage=StorageType.FPGA_Local)
    out_stream = state.add_stream('out_stream', dace.int32, transient=True, storage=StorageType.FPGA_Local)

    nest = state.add_nested_sdfg(make_nested_sdfg(), sdfg, {
        'IN_a',
        'IN_bound',
    }, {
        'OUT_stream',
    })

    state.add_memlet_path(a_in, nest, dst_conn='IN_a', memlet=Memlet.simple(a_in, '0:N'))
    state.add_memlet_path(bound_pipe, nest, dst_conn='IN_bound', memlet=Memlet.simple(bound_pipe, '0', num_accesses=-1))
    state.add_memlet_path(nest,
                          out_stream,
                          src_conn='OUT_stream',
                          memlet=Memlet.simple(out_stream, '0', num_accesses=-1))

    return sdfg


def test_memlet_volume_propagation_nsdfg():
    sdfg = make_sdfg()
    propagation.propagate_memlets_sdfg(sdfg)

    main_state = sdfg.nodes()[0]
    data_in_memlet = main_state.edges()[0].data
    bound_stream_in_memlet = main_state.edges()[1].data
    out_stream_memlet = main_state.edges()[2].data

    memlet_check_parameters(data_in_memlet, 0, True, [(0, N - 1, 1)])
    memlet_check_parameters(bound_stream_in_memlet, 1, False, [(0, 0, 1)])
    memlet_check_parameters(out_stream_memlet, 0, True, [(0, 0, 1)])

    nested_sdfg = main_state.nodes()[3].sdfg

    loop_state = nested_sdfg.nodes()[2]

    state_check_executions(loop_state, symbol('loop_bound'))


def test_memlet_volume_constants():
    sdfg = dace.SDFG('cmprop')
    sdfg.add_constant('N', 32)
    sdfg.add_array('A', [32], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet('doit', dict(i='0:N'), {}, 'a = i', dict(a=dace.Memlet('A[i]')), external_edges=True)

    sdfg.validate()
    sink_node = next(iter(state.sink_nodes()))
    edge = state.in_edges(sink_node)[0]

    assert not edge.data.dynamic
    assert edge.data.volume == dace.symbol('N')


if __name__ == '__main__':
    test_memlet_volume_propagation_nsdfg()
    test_memlet_volume_constants()
