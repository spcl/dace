# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU-offloading bug reproducers.

Each test pre-validates the SDFG after ``auto_optimize(device=GPU)``; numerical
correctness is asserted where the kernel is small and tractable.
"""
import numpy as np
import pytest

import dace
from dace.transformation.auto.auto_optimize import auto_optimize

N = dace.symbol('N')


@dace.program
def _reduce_view_kernel(x: dace.float64[N, 4, 4, 8], out: dace.float64[N, 2, 2, 8]):
    # WHY: Python ``for`` (not ``dace.map``) keeps each Reduce at SDFG
    # top-level so the View input keeps ``GPU_Global`` storage and selects
    # the ``GPUAuto`` expansion path.
    for i in range(2):
        for j in range(2):
            out[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2, :], axis=(1, 2))


@pytest.mark.gpu
def test_reduce_on_view():
    sdfg = _reduce_view_kernel.to_sdfg()
    sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    sdfg.validate()


@dace.program
def _host_iedge_read_kernel(data: dace.float64[N], out: dace.int32[1]):
    I = np.greater(data, 0.5)
    count: dace.int32 = 0
    for j in range(N):
        if I[j]:
            count = count + 1
    out[0] = count


@pytest.mark.gpu
def test_host_iedge_read():
    sdfg = _host_iedge_read_kernel.to_sdfg()
    sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    sdfg.validate()

    rng = np.random.default_rng(1)
    data = rng.uniform(0, 1, size=20)
    out = np.zeros(1, dtype=np.int32)
    sdfg(data=data, out=out, N=20)
    assert int(out[0]) == int(np.sum(data > 0.5))


@dace.program
def _map_straddle_kernel(data: dace.float64[N], out: dace.bool_[N]):
    I = np.greater(data, 0.5)

    keep_alive: dace.int32 = 0
    for j in range(N):
        if I[j]:
            keep_alive = keep_alive + 1

    not_I = np.logical_not(I)
    for k in dace.map[0:N]:
        out[k] = not_I[k]
    # WHY: prevents the iedge read from being dead-code eliminated.
    if keep_alive < 0:
        out[0] = False


@pytest.mark.gpu
def test_map_straddle():
    sdfg = _map_straddle_kernel.to_sdfg()
    sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    sdfg.validate()

    rng = np.random.default_rng(2)
    data = rng.uniform(0, 1, size=15)
    out = np.zeros(15, dtype=np.bool_)
    sdfg(data=data, out=out, N=15)
    ref = np.logical_not(data > 0.5)
    assert np.array_equal(out, ref)


# WHY: ``K`` is declared unsigned so ``pow(R, K)`` lands on the integer pow
# overload; a signed exponent would route through ``std::pow`` and yield a
# ``double`` that can't be used as an array subscript or launch-dim.
R = dace.symbol('R', dace.int64)
K = dace.symbol('K', dace.uint32)


@dace.program
def _int_pow_index_kernel(out: dace.float64[R**K]):
    for i in dace.map[0:R**K]:
        out[i] = out[i] + 1.0


@pytest.mark.gpu
def test_int_pow_used_as_index():
    sdfg = _int_pow_index_kernel.to_sdfg()
    sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    sdfg.validate()

    R_v, K_v = 2, 6
    out = np.zeros(R_v**K_v, dtype=np.float64)
    sdfg(out=out, R=R_v, K=K_v)
    assert np.array_equal(out, np.ones(R_v**K_v, dtype=np.float64))


@pytest.mark.gpu
def test_int_pow_signed_base_unsigned_exp():
    test_int_pow_used_as_index()


@dace.program
def _reduce_view_complex_kernel(x: dace.complex128[N, 4, 4, 8], out: dace.complex128[N, 2, 2, 8]):
    for i in range(2):
        for j in range(2):
            out[:, i, j, :] = np.sum(x[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2, :], axis=(1, 2))


@pytest.mark.gpu
def test_reduce_on_view_complex():
    sdfg = _reduce_view_complex_kernel.to_sdfg()
    sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    sdfg.validate()


# WHY: ``L`` is an SDFG symbol with an interstate assignment inside the SDFG, so a
# temporary shaped ``(L,)`` is "non-free-symbol dependent" at the SDFG scope. The
# framecode then splits the DECLARE (at SDFG scope) from the ALLOCATE/DEALLOCATE
# (at the first/last producing states). Hand-built so the two access sites of
# ``tmp`` land in DIFFERENT states under one outer ControlFlowRegion, exercising
# the same split-declare/allocate codegen path the ``declare_array`` regression
# hit (previously omitted the host pointer from ``defined_vars`` so kernel-scope
# variable definition failed with KeyError).
L = dace.symbol('L', dace.int64)


def _build_split_scope_transient_sdfg():
    """Build a GPU-resident SDFG where ``tmp`` is Scope-lifetime, multi-state,
    and its shape depends on a non-free symbol. The two producer/consumer states
    live inside a ``LoopRegion`` so they are reachable from each other across
    iterations -- which is the structural shape that mandelbrot2 lands on after
    ``auto_optimize`` and what makes the framecode split DECLARE (at SDFG scope)
    from ALLOCATE (at the first producing state). The previous
    ``_declare_pointer_if_needed`` skipped ``defined_vars.add`` on the second
    state's allocate, leaving the host pointer absent from the kernel's
    variable-definition lookup.
    """
    from dace.sdfg.state import LoopRegion
    GPU = dace.dtypes.StorageType.GPU_Global
    sdfg = dace.SDFG('split_scope_lifetime_transient')
    sdfg.add_symbol('L', dace.int64)
    sdfg.add_symbol('length', dace.int64)
    sdfg.add_array('Z', (L, ), dace.float64, storage=GPU)
    sdfg.add_array('C', (L, ), dace.float64, storage=GPU)
    sdfg.add_array('out', (L, ), dace.float64, storage=GPU)
    length_sym = dace.symbol('length', dace.int64)
    sdfg.add_transient('tmp', (length_sym, ), dace.float64, storage=GPU, lifetime=dace.dtypes.AllocationLifetime.Scope)

    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion(label='lr',
                      condition_expr='length > 0',
                      loop_var='length',
                      initialize_expr='length = L',
                      update_expr='length = length - 1')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())

    write_tmp = loop.add_state('write_tmp', is_start_block=True)
    z_in = write_tmp.add_read('Z')
    tmp_w = write_tmp.add_write('tmp')
    me, mx = write_tmp.add_map('mul_map', dict(i='0:length'), schedule=dace.ScheduleType.GPU_Device)
    t = write_tmp.add_tasklet('mul', {'a'}, {'b'}, 'b = a * a')
    write_tmp.add_memlet_path(z_in, me, t, dst_conn='a', memlet=dace.Memlet('Z[i]'))
    write_tmp.add_memlet_path(t, mx, tmp_w, src_conn='b', memlet=dace.Memlet('tmp[i]'))

    read_tmp = loop.add_state('read_tmp')
    tmp_r = read_tmp.add_read('tmp')
    c_in = read_tmp.add_read('C')
    o_w = read_tmp.add_write('out')
    me2, mx2 = read_tmp.add_map('add_map', dict(i='0:length'), schedule=dace.ScheduleType.GPU_Device)
    t2 = read_tmp.add_tasklet('add', {'a', 'c'}, {'b'}, 'b = a + c')
    read_tmp.add_memlet_path(tmp_r, me2, t2, dst_conn='a', memlet=dace.Memlet('tmp[i]'))
    read_tmp.add_memlet_path(c_in, me2, t2, dst_conn='c', memlet=dace.Memlet('C[i]'))
    read_tmp.add_memlet_path(t2, mx2, o_w, src_conn='b', memlet=dace.Memlet('out[i]'))

    loop.add_edge(write_tmp, read_tmp, dace.InterstateEdge())
    return sdfg


@pytest.mark.gpu
def test_split_scope_lifetime_transient_across_states():
    # This bug only surfaces on the experimental CUDA backend; force-select it.
    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        sdfg = _build_split_scope_transient_sdfg()
        sdfg.validate()
        # The compile path is the actual regression site (codegen-time KeyError).
        sdfg.compile()


if __name__ == "__main__":
    test_reduce_on_view()
    test_host_iedge_read()
    test_map_straddle()
    test_int_pow_used_as_index()
    test_int_pow_signed_base_unsigned_exp()
    test_reduce_on_view_complex()
    test_split_scope_lifetime_transient_across_states()
