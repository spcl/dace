# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for cuTile ``generate_scope`` / kernel-parameter codegen.

Covers the code-review findings on the scope generator:

1. **Bound map params never become launch args**: a kernel map parameter that
   collides with a host loop variable or an interstate-assignment key must not
   pass the ``runtime_defined`` filter in ``_collect_free_symbols`` (it would
   become an undefined-at-callsite launch argument -> ``NameError``).
2. **Scalar in kernel input AND output**: the launch site passes
   kernel-written scalars raw while in-kernel reads bind numeric scalars as
   0-d tiles (and a raw bool 0-d host buffer is rejected by ``ct.launch``)
   -- no in/out convention works, so codegen must raise loudly.
3. **Numeric SYMBOLS**: by-value float kernel arguments are typed float32 by
   the cuda.tile frontend (and by-value ints int32); numeric symbols must
   ride the same device-memory path as numeric Scalars (1-element device
   array + 0-d tile load). Runtime-defined names (interstate-assigned, absent
   from ``sdfg.symbols``) ride it too, staged with the runtime value's own
   dtype (``cupy.asarray(s).reshape(1)``, no ``dtype=``).
4. **Launch guard**: non-positive grid dims (zero- or negative-trip maps) skip
   the launch (``min((...)) > 0``), not just zero dims.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.dtypes import Language, ScheduleType, StorageType
from dace.memlet import Memlet
from dace.sdfg import nodes

# ---------------------------------------------------------------------------
# 1: _collect_free_symbols excludes scope-bound map parameters
# ---------------------------------------------------------------------------


def _scope_sdfg_with_colliding_param(param: str):
    """A CuTile map whose parameter collides with an interstate-assigned name.

    :param param: The map-parameter name (also used as an interstate
        assignment key on a later edge).
    :returns: ``(sdfg, state, map_entry)``.
    """
    N = dace.symbol('N', dtype=dace.int64)
    sdfg = dace.SDFG(f'collide_{param}')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [N], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('B', [N], dace.float64, storage=StorageType.GPU_Global)
    state = sdfg.add_state('main')
    state.add_mapped_tasklet('t', {param: '0:N:8'}, {'_a': Memlet(f'A[{param}]')},
                             '_b = _a + 1.0', {'_b': Memlet(f'B[{param}]')},
                             external_edges=True,
                             schedule=ScheduleType.CuTile)
    # A later interstate edge assigns the same name, making it
    # "runtime-defined" from _collect_free_symbols' point of view.
    tail = sdfg.add_state('tail')
    sdfg.add_edge(state, tail, dace.InterstateEdge(assignments={param: '0'}))
    entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    return sdfg, state, entry


def test_map_param_colliding_with_interstate_key_not_a_kernel_symbol():
    """The map param is scope-bound and must not appear in the free symbols."""
    from dace.codegen.py.cutile_target import _collect_free_symbols
    sdfg, state, entry = _scope_sdfg_with_colliding_param('ii')
    syms = _collect_free_symbols(entry, state.scope_subgraph(entry), sdfg)
    assert 'ii' not in syms
    assert 'N' in syms


def test_map_param_colliding_with_loop_variable_not_a_kernel_symbol():
    """Same collision through a LoopRegion loop variable."""
    from dace.codegen.py.cutile_target import _collect_free_symbols
    from dace.sdfg.state import LoopRegion
    sdfg, state, entry = _scope_sdfg_with_colliding_param('jj')
    loop = LoopRegion('loop', 'jj < 4', 'jj', 'jj = 0', 'jj = jj + 1')
    sdfg.add_node(loop)
    loop.add_state('body')
    syms = _collect_free_symbols(entry, state.scope_subgraph(entry), sdfg)
    assert 'jj' not in syms


# ---------------------------------------------------------------------------
# 2: numeric Scalar in input AND output -> NotImplementedError
# ---------------------------------------------------------------------------


def _inout_scalar_sdfg(dtype) -> dace.SDFG:
    """A CuTile map reading and writing the same non-transient Scalar."""
    sdfg = dace.SDFG(f'inout_scalar_{dtype.to_string()}')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_scalar('s', dtype, storage=StorageType.GPU_Global)
    state = sdfg.add_state('main')
    me, mx = state.add_map('cutile_map', {'tile_i': '0:1'}, schedule=ScheduleType.CuTile)
    r = state.add_read('s')
    w = state.add_write('s')
    tk = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp + 1', language=Language.Python)
    state.add_memlet_path(r, me, tk, dst_conn='inp', memlet=Memlet('s[0]'))
    state.add_memlet_path(tk, mx, w, src_conn='out', memlet=Memlet('s[0]'))
    sdfg.fill_scope_connectors()
    return sdfg


def test_float_scalar_inout_raises_not_implemented():
    sdfg = _inout_scalar_sdfg(dace.float64)
    with pytest.raises(NotImplementedError, match='both a kernel input and a kernel output'):
        sdfg.generate_code()


def test_int_scalar_inout_raises_not_implemented():
    """Integer scalars ride the device-memory convention too, so in/out
    integer scalars are rejected the same way as floats."""
    sdfg = _inout_scalar_sdfg(dace.int64)
    with pytest.raises(NotImplementedError, match='both a kernel input and a kernel output'):
        sdfg.generate_code()


def test_bool_scalar_inout_raises_not_implemented():
    """Bool in/out scalars are rejected too: a raw bool in/out Scalar is a 0-d
    host buffer that ``ct.launch`` rejects at runtime (probed: ``RuntimeError:
    NumPy only supports stream=None``), with no writeback path either."""
    sdfg = _inout_scalar_sdfg(dace.bool)
    with pytest.raises(NotImplementedError, match='both a kernel input and a kernel output'):
        sdfg.generate_code()


# ---------------------------------------------------------------------------
# 3 + 4: float64 symbols + launch guard
# ---------------------------------------------------------------------------


def _float_symbol_sdfg(name: str, n: int = 64, tile_w: int = 32) -> dace.SDFG:
    """``y = x * alpha_v`` with ``alpha_v`` a float64 SYMBOL, tiles built
    manually (the AccessNode-centric pattern used across these tests)."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol('alpha_v', dace.float64)
    sdfg.add_array('x', [n], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('y', [n], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('_tx', [tile_w], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array('_ty', [tile_w], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    state = sdfg.add_state('main')
    me, mx = state.add_map('cutile_map', {'tile_i': f'0:{n}:{tile_w}'}, schedule=ScheduleType.CuTile)
    xr = state.add_read('x')
    yw = state.add_write('y')
    tx = state.add_access('_tx')
    ty = state.add_access('_ty')
    tk = state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * alpha_v', language=Language.Python)
    state.add_memlet_path(xr, me, tx, memlet=Memlet(data='x', subset=f'0:{n}'))
    state.add_edge(tx, None, tk, 'inp', Memlet(data='_tx', subset=f'0:{tile_w}'))
    state.add_edge(tk, 'out', ty, None, Memlet(data='_ty', subset=f'0:{tile_w}'))
    state.add_memlet_path(ty, mx, yw, memlet=Memlet(data='y', subset=f'0:{n}'))
    sdfg.fill_scope_connectors()
    return sdfg


def test_float64_symbol_staged_through_device_memory():
    """The launch site wraps the float symbol in a 1-element device array and
    the kernel rebinds it as a 0-d tile (by-value floats are typed float32
    by cuda.tile, silently losing f64 precision)."""
    sdfg = _float_symbol_sdfg('float_sym_structural')
    code = sdfg.generate_code()[0].code.replace(' ', '')
    assert 'cupy.asarray(alpha_v,dtype=numpy.float64).reshape(1)' in code
    assert 'alpha_v=ct.load(alpha_v,(0,),shape=()).item()' in code


def test_launch_guard_skips_nonpositive_grid():
    """The launch is guarded by ``min((...)) > 0`` (not merely ``0 not in``),
    so negative-trip grids (e.g. ``1:N-1`` at ``N == 1``) are skipped too."""
    sdfg = _float_symbol_sdfg('launch_guard_structural')
    code = sdfg.generate_code()[0].code.replace(' ', '').replace('\n', '')
    assert '>0:ct.launch' in code
    assert 'ifmin((' in code
    assert '0notin' not in code


def _int_symbol_sdfg(name: str, n: int = 64, tile_w: int = 32) -> dace.SDFG:
    """``y = x + k_off`` with ``k_off`` an int64 SYMBOL (same manual-tile
    pattern as :func:`_float_symbol_sdfg`)."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol('k_off', dace.int64)
    sdfg.add_array('x', [n], dace.int64, storage=StorageType.GPU_Global)
    sdfg.add_array('y', [n], dace.int64, storage=StorageType.GPU_Global)
    sdfg.add_array('_tx', [tile_w], dace.int64, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array('_ty', [tile_w], dace.int64, storage=StorageType.CuTile_Tile, transient=True)
    state = sdfg.add_state('main')
    me, mx = state.add_map('cutile_map', {'tile_i': f'0:{n}:{tile_w}'}, schedule=ScheduleType.CuTile)
    xr = state.add_read('x')
    yw = state.add_write('y')
    tx = state.add_access('_tx')
    ty = state.add_access('_ty')
    tk = state.add_tasklet('shift', {'inp'}, {'out'}, 'out = inp + k_off', language=Language.Python)
    state.add_memlet_path(xr, me, tx, memlet=Memlet(data='x', subset=f'0:{n}'))
    state.add_edge(tx, None, tk, 'inp', Memlet(data='_tx', subset=f'0:{tile_w}'))
    state.add_edge(tk, 'out', ty, None, Memlet(data='_ty', subset=f'0:{tile_w}'))
    state.add_memlet_path(ty, mx, yw, memlet=Memlet(data='y', subset=f'0:{n}'))
    sdfg.fill_scope_connectors()
    return sdfg


def test_int64_symbol_staged_through_device_memory():
    """Int symbols ride the same device-memory path as float symbols
    (by-value ints are typed int32 by cuda.tile: OverflowError >= 2**31)."""
    sdfg = _int_symbol_sdfg('int_sym_structural')
    code = sdfg.generate_code()[0].code.replace(' ', '')
    assert 'cupy.asarray(k_off,dtype=numpy.int64).reshape(1)' in code
    assert 'k_off=ct.load(k_off,(0,),shape=()).item()' in code


@pytest.mark.gpu
def test_int64_symbol_large_value_runtime():
    """End-to-end: an int64 symbol >= 2**31 (by-value would OverflowError)."""
    cupy = pytest.importorskip('cupy')
    n = 64
    sdfg = _int_symbol_sdfg('int_sym_runtime', n=n)
    k_off = 2**40 + 12345
    x_host = np.arange(n, dtype=np.int64)
    x = cupy.asarray(x_host)
    y = cupy.zeros(n, dtype=cupy.int64)
    sdfg.compile()(x=x, y=y, k_off=k_off)
    np.testing.assert_array_equal(cupy.asnumpy(y), x_host + k_off)


@pytest.mark.gpu
def test_float64_symbol_full_precision_runtime():
    """End-to-end: an f32-typed symbol would lose the 2**-40 component."""
    cupy = pytest.importorskip('cupy')
    n = 64
    sdfg = _float_symbol_sdfg('float_sym_runtime', n=n)
    alpha = 1.0 + 2.0**-40
    x_host = np.random.default_rng(11).random(n)
    x = cupy.asarray(x_host)
    y = cupy.zeros(n, dtype=cupy.float64)
    sdfg.compile()(x=x, y=y, alpha_v=alpha)
    ref = x_host * alpha
    # Tight tolerance: a float32-typed symbol argument shows ~1e-8 error.
    assert np.abs(cupy.asnumpy(y) - ref).max() < 1e-14


def _runtime_defined_symbol_sdfg(name: str,
                                 assign_expr: str,
                                 dtype,
                                 op: str,
                                 n: int = 64,
                                 tile_w: int = 32) -> dace.SDFG:
    """``y = x <op> c_rt`` with ``c_rt`` a RUNTIME-DEFINED name: assigned on an
    interstate edge and deliberately NOT declared in ``sdfg.symbols``."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array('x', [n], dtype, storage=StorageType.GPU_Global)
    sdfg.add_array('y', [n], dtype, storage=StorageType.GPU_Global)
    sdfg.add_array('_tx', [tile_w], dtype, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array('_ty', [tile_w], dtype, storage=StorageType.CuTile_Tile, transient=True)
    init = sdfg.add_state('init')
    state = sdfg.add_state('main')
    edge = sdfg.add_edge(init, state, dace.InterstateEdge(assignments={'c_rt': f'numpy.{dtype.type.__name__}({assign_expr})'}))
    sdfg.symbol_repo.add('c_rt', dtype, at=edge.data)
    me, mx = state.add_map('cutile_map', {'tile_i': f'0:{n}:{tile_w}'}, schedule=ScheduleType.CuTile)
    xr = state.add_read('x')
    yw = state.add_write('y')
    tx = state.add_access('_tx')
    ty = state.add_access('_ty')
    tk = state.add_tasklet('t', {'inp'}, {'out'}, f'out = inp {op} c_rt', language=Language.Python)
    state.add_memlet_path(xr, me, tx, memlet=Memlet(data='x', subset=f'0:{n}'))
    state.add_edge(tx, None, tk, 'inp', Memlet(data='_tx', subset=f'0:{tile_w}'))
    state.add_edge(tk, 'out', ty, None, Memlet(data='_ty', subset=f'0:{tile_w}'))
    state.add_memlet_path(ty, mx, yw, memlet=Memlet(data='y', subset=f'0:{n}'))
    sdfg.fill_scope_connectors()
    assert 'c_rt' not in sdfg.symbols
    return sdfg


def test_runtime_defined_symbol_staged_with_scoped_dtype():
    """Resolve interstate bindings at the kernel launch site."""
    sdfg = _runtime_defined_symbol_sdfg('rt_sym_structural', '1.0 + 2.0**(-40)', dace.float64, '*')
    code = sdfg.generate_code()[0].code.replace(' ', '')
    assert 'cupy.asarray(c_rt,dtype=numpy.float64).reshape(1)' in code
    assert 'c_rt=ct.load(c_rt,(0,),shape=()).item()' in code


@pytest.mark.gpu
def test_runtime_defined_float_symbol_runtime():
    """End-to-end: a runtime-defined float (``1 + 2**-40``, not f32-exact)
    survives the kernel exactly (by value it would round to float32)."""
    cupy = pytest.importorskip('cupy')
    n = 64
    sdfg = _runtime_defined_symbol_sdfg('rt_float_runtime', '1.0 + 2.0**(-40)', dace.float64, '*', n=n)
    x_host = np.random.default_rng(13).random(n)
    x = cupy.asarray(x_host)
    y = cupy.zeros(n, dtype=cupy.float64)
    sdfg.compile()(x=x, y=y)
    ref = x_host * (1.0 + 2.0**-40)
    np.testing.assert_array_equal(cupy.asnumpy(y), ref)


@pytest.mark.gpu
def test_runtime_defined_int_symbol_large_value_runtime():
    """End-to-end: a runtime-defined int >= 2**31 (by value cuda.tile types
    Python ints int32 -> OverflowError)."""
    cupy = pytest.importorskip('cupy')
    n = 64
    big = 2**40 + 12345
    sdfg = _runtime_defined_symbol_sdfg('rt_int_runtime', str(big), dace.int64, '+', n=n)
    x_host = np.arange(n, dtype=np.int64)
    x = cupy.asarray(x_host)
    y = cupy.zeros(n, dtype=cupy.int64)
    sdfg.compile()(x=x, y=y)
    np.testing.assert_array_equal(cupy.asnumpy(y), x_host + big)


# ---------------------------------------------------------------------------
# 5: py_mod / int_floor in rendered element-index expressions
# ---------------------------------------------------------------------------



if __name__ == '__main__':
    pytest.main([__file__, '-v'])


@pytest.mark.gpu
@pytest.mark.parametrize('dtype,value', [
    (dace.int32, 123), (dace.int64, 2**40 + 17),
    (dace.uint64, 2**63 + 17), (dace.float64, 1.0 + 2.0**-40),
])
def test_interstate_symbol_repository_dtype_runtime(dtype: dace.typeclass, value: object):
    """The type resolved at an interstate edge determines the device buffer."""
    cupy = pytest.importorskip('cupy')
    sdfg = _runtime_defined_symbol_sdfg(f'scoped_{dtype.type.__name__}', repr(value), dtype, '+')
    edge = sdfg.edges()[0]
    sdfg.symbol_repo.add('c_rt', dtype, at=edge.data)
    compiled = sdfg.compile()
    assert f'cupy.asarray(c_rt, dtype=numpy.{dtype.type.__name__})' in compiled.code
    host = np.arange(64, dtype=dtype.as_numpy_dtype())
    output = cupy.zeros(64, dtype=dtype.as_numpy_dtype())
    compiled(x=cupy.asarray(host), y=output)
    np.testing.assert_array_equal(cupy.asnumpy(output), host + value)
