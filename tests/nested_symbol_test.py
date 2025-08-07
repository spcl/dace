# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import warnings
import pytest

from dace.sdfg import dealias

N = dace.symbol('N')


@dace.program
def nested(A: dace.float64[N], B: dace.float64[N], factor: dace.float64):
    B[:] = A * factor


@dace.program
def nested_symbol(A: dace.float64[N], B: dace.float64[N]):
    nested(A[0:5], B[0:5], 0.5)
    nested(A=A[5:N], B=B[5:N], factor=2.0)


@dace.program
def nested_symbol_dynamic(A: dace.float64[N]):
    for i in range(5):
        nested(A[0:i], A[0:i], i)


def test_nested_symbol():
    A = np.random.rand(20)
    B = np.random.rand(20)
    nested_symbol(A, B)
    assert np.allclose(B[0:5], A[0:5] / 2) and np.allclose(B[5:20], A[5:20] * 2)


def test_nested_symbol_dynamic():
    if not dace.Config.get_bool('optimizer', 'automatic_simplification'):
        warnings.warn("Test disabled (missing allocation lifetime support)")
        return

    A = np.random.rand(5)
    expected = A.copy()
    for i in range(5):
        expected[0:i] *= i
    nested_symbol_dynamic(A)
    assert np.allclose(A, expected)


def test_scal2sym():
    N = dace.symbol('N', dace.float64)

    @dace.program
    def symarg(A: dace.float64[20]):
        A[:] = N

    @dace.program
    def scalarg(A: dace.float64[20], scal: dace.float64):
        s2 = scal + 1
        symarg(A, N=s2)

    sdfg = scalarg.to_sdfg(simplify=False)
    A = np.random.rand(20)
    sc = 5.0

    sdfg(A, sc)
    assert np.allclose(A, sc + 1)


def test_arr2sym():
    N = dace.symbol('N', dace.float64)

    @dace.program
    def symarg(A: dace.float64[20]):
        A[:] = N

    @dace.program
    def scalarg(A: dace.float64[20], arr: dace.float64[2]):
        symarg(A, N=arr[1])

    sdfg = scalarg.to_sdfg(simplify=False)
    A = np.random.rand(20)
    sc = np.array([2.0, 3.0])

    sdfg(A, sc)
    assert np.allclose(A, sc[1])


def test_nested_symbol_in_args():
    inner = dace.SDFG('inner')
    state = inner.add_state('inner_state')
    inner.add_symbol('rdt', stype=float)
    inner.add_datadesc('field', dace.float64[10])
    state.add_mapped_tasklet('tasklet',
                             map_ranges={'i': "0:10"},
                             inputs={},
                             outputs={'field_out': dace.Memlet.simple('field', subset_str="i")},
                             code="field_out = rdt",
                             external_edges=True)
    inner.arg_names = ['field', 'rdt']

    @dace.program
    def funct(field, dt):
        rdt = 1.0 / dt
        inner(field, rdt)

    sdfg = funct.to_sdfg(np.random.randn(10, ), 1.0, simplify=False)
    sdfg(np.random.randn(10, ), 1.0)


def test_nested_symbol_as_constant():
    inner = dace.SDFG('inner')
    state = inner.add_state('inner_state')
    inner.add_symbol('rdt', stype=float)
    inner.add_datadesc('field', dace.float64[10])
    tasklet, map_entry, map_exit = state.add_mapped_tasklet(
        'tasklet',
        map_ranges={'i': "0:10"},
        inputs={},
        outputs={'field_out': dace.Memlet.simple('field', subset_str="i")},
        code="field_out = rdt",
        external_edges=True)
    inner.arg_names = ['field', 'rdt']
    rdt = 1e30

    @dace.program
    def funct(field):
        inner(field, rdt)

    funct(np.random.randn(10, ))


@pytest.mark.parametrize("in_symbol_mapping", [True, False])
@pytest.mark.parametrize("global_symbol", [True, False])
def test_nested_symbol_collision(in_symbol_mapping, global_symbol):
    """
    Test for symbol name collision between outer map and nested SDFG.

    This test checks for a potential bug where a map uses symbol 'i' as its iterator,
    and a nested SDFG inside that map also defines its own symbol 'i'. The nested
    SDFG's symbol should be independent, but there's a risk that the outer 'i'
    could be incorrectly substituted into the nested SDFG, causing incorrect behavior
    when accessing array elements like B[i] where 'i' should refer to the outer symbol.
    """
    # Create outer SDFG
    sdfg = dace.SDFG('test_symbol_collision')
    sdfg.add_array('B', [43], dace.float64)

    state = sdfg.add_state('outer_state')

    # Create map with Sequential schedule using 'i' as iterator
    map_entry, map_exit = state.add_map('outer_map', dict(i='0:10'), schedule=dace.ScheduleType.Sequential)

    # Create nested SDFG
    nsdfg = dace.SDFG('nested')
    if global_symbol:
        nsdfg.add_symbol('i', stype=dace.int32)  # Different 'i' symbol
    nsdfg.add_scalar('b', dace.float64, transient=False)

    nstate = nsdfg.add_state('nested_state')

    # Add tasklet that uses the nested 'i' symbol
    tasklet = nstate.add_tasklet('set_i', {}, {'out'}, 'out = i')
    b_access = nstate.add_access('b')
    nstate.add_edge(tasklet, 'out', b_access, None, dace.Memlet.simple('b', '0'))

    # Add nested SDFG to outer state
    if in_symbol_mapping:
        # Set the nested 'i' symbol to a fixed value
        nested_node = state.add_nested_sdfg(nsdfg, {}, {'b'}, symbol_mapping={'i': 42})
    else:
        # Set i internally via inter-state edge
        init_state = nsdfg.add_state(is_start_block=True)
        nsdfg.add_edge(init_state, nstate, dace.InterstateEdge(assignments={'i': 42}))
        nested_node = state.add_nested_sdfg(nsdfg, {}, {'b'})

    # Connect map to nested SDFG
    B = state.add_write('B')

    state.add_memlet_path(map_entry, nested_node, dst_conn=None, memlet=dace.Memlet())
    state.add_memlet_path(nested_node, map_exit, B, src_conn='b', memlet=dace.Memlet.simple('B', 'i'))

    # Integrate the nested SDFG into the outer SDFG
    dealias.integrate_nested_sdfg(nsdfg)

    # Execute and verify
    B_val = np.zeros(43, dtype=np.float64)

    sdfg.validate()

    sdfg(B=B_val)

    # All elements of B should be 42 (the nested 'i' value)
    # not 0,1,2,...,9 (the outer 'i' values)
    assert np.allclose(B_val[0:10], 42), f"Expected all 42s, got {B_val}"


def test_nested_symbol_collision_map():
    """
    Test for symbol name collision between outer map and nested SDFG's inner map.

    This test checks for a potential bug where a map uses symbol 'i' as its iterator,
    and a nested SDFG inside that map also defines its own symbol 'i' as a map iterate. The nested
    SDFG's symbol should be independent, but there's a risk that the outer 'i'
    could be incorrectly substituted into the nested SDFG, causing incorrect behavior
    when accessing array elements like B[i] where 'i' should refer to the outer symbol.
    """
    # Create outer SDFG
    sdfg = dace.SDFG('test_symbol_collision')
    sdfg.add_array('B', [43], dace.float64)

    state = sdfg.add_state('outer_state')

    # Create map with Sequential schedule using 'i' as iterator
    map_entry, map_exit = state.add_map('outer_map', dict(i='0:10'), schedule=dace.ScheduleType.Sequential)

    # Create nested SDFG
    nsdfg = dace.SDFG('nested')
    nsdfg.add_scalar('b', dace.float64, transient=False)

    nstate = nsdfg.add_state('nested_state')

    # Add tasklet that uses the nested 'i' symbol
    imap_entry, imap_exit = nstate.add_map('inner_map', dict(i='42:43'), schedule=dace.ScheduleType.Sequential)
    tasklet = nstate.add_tasklet('set_i', {}, {'out'}, 'out = i')
    b_access = nstate.add_access('b')
    nstate.add_nedge(imap_entry, tasklet, dace.Memlet())
    nstate.add_memlet_path(tasklet, imap_exit, b_access, src_conn='out', memlet=dace.Memlet.simple('b', '0'))

    nested_node = state.add_nested_sdfg(nsdfg, {}, {'b'})

    # Connect map to nested SDFG
    B = state.add_write('B')

    state.add_memlet_path(map_entry, nested_node, dst_conn=None, memlet=dace.Memlet())
    state.add_memlet_path(nested_node, map_exit, B, src_conn='b', memlet=dace.Memlet.simple('B', 'i'))

    # Integrate the nested SDFG into the outer SDFG
    dealias.integrate_nested_sdfg(nsdfg)

    # Execute and verify
    B_val = np.zeros(43, dtype=np.float64)

    sdfg.validate()

    sdfg(B=B_val)

    # All elements of B should be 42 (the nested 'i' value)
    # not 0,1,2,...,9 (the outer 'i' values)
    assert np.allclose(B_val[0:10], 42), f"Expected all 42s, got {B_val}"


if __name__ == '__main__':
    test_nested_symbol()
    test_nested_symbol_dynamic()
    test_scal2sym()
    test_arr2sym()
    test_nested_symbol_in_args()
    test_nested_symbol_as_constant()
    test_nested_symbol_collision(False, False)
    test_nested_symbol_collision(False, True)
    test_nested_symbol_collision(True, False)
    test_nested_symbol_collision(True, True)
    test_nested_symbol_collision_map()
