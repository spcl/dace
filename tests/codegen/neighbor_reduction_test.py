""" Tests code generation for reduction on GPU target. """
import dace
from dace.transformation.passes import constant_propagation, fusion_inline, scalar_to_symbol
from dace.sdfg.state import SDFGState
from dace.transformation import transformation as xf, interstate as isxf
from dace.transformation.interstate import loop_detection as ld
from dace import registry
from dace.transformation import helpers as xfh
from dace.transformation.auto import auto_optimize

import collections
import numpy as np
import pytest


N_E2V_NEIGHBORS = 2

def build_sdfg_neighbor_reduction(sdfg_name: str):
    dtype = dace.float64

    sdfg = dace.SDFG(sdfg_name)
    N_EDGES, N_VERTICES = (dace.symbol(s) for s in ['N_EDGES', 'N_VERTICES'])
    sdfg.add_array('EDGES', (N_EDGES,), dtype)
    sdfg.add_array('VERTICES', (N_VERTICES,), dtype)
    sdfg.add_array('EDGE_NEIGHBOR_WEIGHTS', (N_E2V_NEIGHBORS, N_EDGES,), dtype)
    sdfg.add_array('E2V_TABLE', (N_EDGES, N_E2V_NEIGHBORS), dace.int32)
    sdfg.add_scalar('edge_idx_value', dace.int32, transient=True)

    nsdfg = dace.SDFG('nsdfg')
    nsdfg.add_array('_weights', sdfg.arrays['EDGE_NEIGHBOR_WEIGHTS'].shape, dtype)
    nsdfg.add_array('_field', sdfg.arrays['VERTICES'].shape, dtype)
    nsdfg.add_array('_table', sdfg.arrays['E2V_TABLE'].shape, dace.int32)
    nsdfg.add_scalar('_input_field_idx', dace.int32)
    nsdfg.add_scalar('_input_weight_idx', dace.int32)
    nsdfg.add_scalar('_acc', dtype, transient=True)
    nsdfg.add_scalar('_result', dtype)
    istate = nsdfg.add_state('init', is_start_block=True)
    istate.add_nedge(
        istate.add_access('_result'),
        istate.add_access('_acc'),
        dace.Memlet.simple('_result', '0')
    )
    nstate = nsdfg.add_state_after(istate, 'compute')
    me_red, mx_red = nstate.add_map('neighbors', dict(_neighbor_idx=f"0:{N_E2V_NEIGHBORS}"))
    nsdfg.add_scalar('_field_idx', dace.int32, transient=True)
    field_idx_node = nstate.add_access('_field_idx')
    nsdfg.add_scalar('_field_value', dtype, transient=True)
    field_value_node = nstate.add_access('_field_value')
    nsdfg.add_scalar('_neighbor_weight', dtype, transient=True)
    neighbor_weight_node = nstate.add_access('_neighbor_weight')
    acc_node = nstate.add_access('_acc')
    shift_node = nstate.add_tasklet('shift', {'_inp', '_idx'}, {'_out'}, '_out = _inp[_idx, _neighbor_idx]')
    deref_field = nstate.add_tasklet('deref', {'_inp', '_idx'}, {'_out'}, '_out = _inp[_idx]')
    deref_weight = nstate.add_tasklet('deref', {'_inp', '_idx'}, {'_out'}, '_out = _inp[_neighbor_idx, _idx]')
    nstate.add_memlet_path(
        nstate.add_access('_weights'),
        me_red,
        deref_weight,
        dst_conn='_inp',
        memlet=dace.Memlet.from_array('_weights', nsdfg.arrays['_weights'])
    )
    nstate.add_memlet_path(
        nstate.add_access('_input_weight_idx'),
        me_red,
        deref_weight,
        dst_conn='_idx',
        memlet=dace.Memlet.simple('_input_weight_idx', '0')
    )
    nstate.add_edge(
        deref_weight,
        '_out',
        neighbor_weight_node,
        None,
        memlet=dace.Memlet.simple('_neighbor_weight', '0')
    )
    nstate.add_memlet_path(
        nstate.add_access('_table'),
        me_red,
        shift_node,
        dst_conn='_inp',
        memlet=dace.Memlet.from_array('_table', nsdfg.arrays['_table'])
    )
    nstate.add_memlet_path(
        nstate.add_access('_input_field_idx'),
        me_red,
        shift_node,
        dst_conn='_idx',
        memlet=dace.Memlet.simple('_input_field_idx', '0')
    )
    nstate.add_edge(
        shift_node,
        '_out',
        field_idx_node,
        None,
        dace.Memlet.simple('_field_idx', '0')
    )
    nstate.add_memlet_path(
        nstate.add_access('_field'),
        me_red,
        deref_field,
        dst_conn='_inp',
        memlet=dace.Memlet.from_array('_field', nsdfg.arrays['_field'])
    )
    nstate.add_edge(
        field_idx_node,
        None,
        deref_field,
        '_idx',
        dace.Memlet.simple('_field_idx', '0')
    )
    nstate.add_edge(
        deref_field,
        '_out',
        field_value_node,
        None,
        dace.Memlet.simple('_field_value', '0')
    )
    compute_node = nstate.add_tasklet('numeric', {'a', 'b'}, {'result'}, "result = a * b")
    nstate.add_edge(
        neighbor_weight_node,
        None,
        compute_node,
        'a',
        dace.Memlet.simple('_neighbor_weight', '0')
    )
    nstate.add_edge(
        field_value_node,
        None,
        compute_node,
        'b',
        dace.Memlet.simple('_field_value', '0')
    )
    nstate.add_memlet_path(
        compute_node,
        mx_red,
        acc_node,
        src_conn='result',
        memlet=dace.Memlet.simple('_acc', '0', "lambda x, y: x + y")
    )
    nstate.add_edge(
        acc_node,
        None,
        nstate.add_access('_result'),
        None,
        memlet=dace.Memlet.simple('_result', '0')
    )

    state = sdfg.add_state('main')
    me, mx = state.add_map('closure', dict(_edge_idx="0:N_EDGES"))
    nsdfg_node = state.add_nested_sdfg(
        nsdfg,
        sdfg,
        inputs={'_weights', '_field', '_table', '_input_field_idx', '_input_weight_idx'},
        outputs={'_result'},
        symbol_mapping={}
    )
    edge_idx_tasklet = state.add_tasklet('get_edge_idx', {}, {'_out'}, '_out = _edge_idx')
    state.add_nedge(me, edge_idx_tasklet, dace.Memlet())
    edge_idx_node = state.add_access('edge_idx_value')
    state.add_edge(
        edge_idx_tasklet,
        '_out',
        edge_idx_node,
        None,
        dace.Memlet.simple('edge_idx_value', '0')
    )
    state.add_edge(
        edge_idx_node,
        None,
        nsdfg_node,
        '_input_field_idx',
        dace.Memlet.simple('edge_idx_value', '0')
    )
    state.add_edge(
        edge_idx_node,
        None,
        nsdfg_node,
        '_input_weight_idx',
        dace.Memlet.simple('edge_idx_value', '0')
    )
    state.add_memlet_path(
        state.add_access('EDGE_NEIGHBOR_WEIGHTS'),
        me,
        nsdfg_node,
        dst_conn='_weights',
        memlet=dace.Memlet.from_array('EDGE_NEIGHBOR_WEIGHTS', sdfg.arrays['EDGE_NEIGHBOR_WEIGHTS'])
    )
    state.add_memlet_path(
        state.add_access('VERTICES'),
        me,
        nsdfg_node,
        dst_conn='_field',
        memlet=dace.Memlet.from_array('VERTICES', sdfg.arrays['VERTICES'])
    )
    state.add_memlet_path(
        state.add_access('E2V_TABLE'),
        me,
        nsdfg_node,
        dst_conn='_table',
        memlet=dace.Memlet.from_array('E2V_TABLE', sdfg.arrays['E2V_TABLE'])
    )
    state.add_memlet_path(
        nsdfg_node,
        mx,
        state.add_access('EDGES'),
        src_conn='_result',
        memlet=dace.Memlet.simple('EDGES', '_edge_idx')
    )

    return sdfg


def test_neighbor_reduction_cpu():
    sdfg = build_sdfg_neighbor_reduction("neighbor_reduction_cpu")

    N_EDGES = np.int32(50)
    N_VERTICES = np.int32(40)

    rng = np.random.default_rng(42)
    e = rng.random((N_EDGES,))
    v = rng.random((N_VERTICES,))
    w = rng.random((N_E2V_NEIGHBORS, N_EDGES,))
    e2v_table = np.random.randint(0, N_VERTICES, (N_EDGES, N_E2V_NEIGHBORS), np.int32)

    e_ref = np.asarray([np.sum(v[e2v_table[idx, :]] * w[:,idx], initial=e[idx]) for idx in range(N_EDGES)])

    sdfg(
        EDGES=e,
        VERTICES=v,
        EDGE_NEIGHBOR_WEIGHTS=w,
        E2V_TABLE=e2v_table,
        N_EDGES=N_EDGES,
        N_VERTICES=N_VERTICES,
    )

    assert np.allclose(e_ref, e)


@pytest.mark.gpu
def test_neighbor_reduction_gpu():


    sdfg = build_sdfg_neighbor_reduction("neighbor_reduction_gpu")
    auto_optimize.apply_gpu_storage(sdfg)

    N_EDGES = np.int32(50)
    N_VERTICES = np.int32(40)

    rng = np.random.default_rng(42)
    e = rng.random((N_EDGES,))
    v = rng.random((N_VERTICES,))
    w = rng.random((N_E2V_NEIGHBORS, N_EDGES,))
    e2v_table = np.random.randint(0, N_VERTICES, (N_EDGES, N_E2V_NEIGHBORS), np.int32)

    e_ref = np.asarray([np.sum(v[e2v_table[idx, :]] * w[:,idx], initial=e[idx]) for idx in range(N_EDGES)])

    sdfg.compile()

    import cupy as cp

    e_dev = cp.asarray(e)
    v_dev = cp.asarray(v)
    w_dev = cp.asarray(w)
    e2v_table_dev = cp.asarray(e2v_table)

    sdfg(
        EDGES=e_dev,
        VERTICES=v_dev,
        EDGE_NEIGHBOR_WEIGHTS=w_dev,
        E2V_TABLE=e2v_table_dev,
        N_EDGES=N_EDGES,
        N_VERTICES=N_VERTICES,
    )

    assert np.allclose(e_ref, e_dev.get())


if __name__ == '__main__':
    test_neighbor_reduction_cpu()
    test_neighbor_reduction_gpu()


