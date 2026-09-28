# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MoveArrayOutOfKernel`` never demotes an accumulator (a WCR target) to per-thread ``Register``."""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.transformation.passes.move_array_out_of_kernel import MoveArrayOutOfKernel

GLOBAL = dtypes.StorageType.GPU_Global
ROWS, COLS = 8, 8


def set_block_sizes(sdfg: dace.SDFG) -> None:
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device:
            node.map.gpu_block_size = [32, 1, 1]


def wcr_targets(sdfg: dace.SDFG) -> list:
    return [(sub, e.data.data) for sub in sdfg.all_sdfgs_recursive() for state in sub.states() for e in state.edges()
            if e.data.wcr is not None]


def assert_no_wcr_target_in_registers(sdfg: dace.SDFG) -> None:
    targets = wcr_targets(sdfg)
    assert targets, 'no accumulation left, so this covers nothing'
    for owner, name in targets:
        assert owner.arrays[name].storage != dtypes.StorageType.Register, (owner.name, name)


def kernel_with_accumulator(wcr: bool) -> dace.SDFG:
    """``out[i] = sum_j A[i, j]`` through a kernel-local ``acc[1]``, accumulated with a WCR or overwritten."""
    sdfg = dace.SDFG(f'kernel_accumulator_{"wcr" if wcr else "plain"}')
    sdfg.add_array('A', [ROWS, COLS], dace.float64, storage=GLOBAL)
    sdfg.add_array('out', [ROWS], dace.float64, storage=GLOBAL)
    sdfg.add_transient('acc', [1], dace.float64, storage=GLOBAL)
    state = sdfg.add_state('s')
    kernel_entry, kernel_exit = state.add_map('kernel', dict(i=f'0:{ROWS}'), schedule=dtypes.ScheduleType.GPU_Device)
    kernel_entry.map.gpu_block_size = [32, 1, 1]
    row_entry, row_exit = state.add_map('row', dict(j=f'0:{COLS}'), schedule=dtypes.ScheduleType.Sequential)

    init = state.add_tasklet('init', {}, {'o': None}, 'o = 0.0')
    state.add_nedge(kernel_entry, init, dace.Memlet())
    zeroed = state.add_access('acc')
    state.add_edge(init, 'o', zeroed, None, dace.Memlet('acc[0]'))

    add = state.add_tasklet('add', {'a': None}, {'o': None}, 'o = a')
    state.add_memlet_path(state.add_read('A'),
                          kernel_entry,
                          row_entry,
                          add,
                          dst_conn='a',
                          memlet=dace.Memlet('A[i, j]'))
    state.add_nedge(zeroed, row_entry, dace.Memlet())
    total = state.add_access('acc')
    state.add_memlet_path(add,
                          row_exit,
                          total,
                          src_conn='o',
                          memlet=dace.Memlet('acc[0]', wcr='lambda x, y: x + y' if wcr else None))

    copy = state.add_tasklet('copy', {'a': None}, {'o': None}, 'o = a')
    state.add_edge(total, None, copy, 'a', dace.Memlet('acc[0]'))
    state.add_memlet_path(copy, kernel_exit, state.add_write('out'), src_conn='o', memlet=dace.Memlet('out[i]'))
    sdfg.validate()
    return sdfg


def test_a_small_accumulator_is_lifted_not_demoted():
    """One element is under the demotion threshold, yet the WCR keeps it in memory: it is lifted instead."""
    sdfg = kernel_with_accumulator(wcr=True)

    with pytest.warns(UserWarning, match='will be lifted outside the kernel'):
        assert MoveArrayOutOfKernel().apply_pass(sdfg, {}) == 1

    desc = sdfg.arrays['acc']
    assert desc.storage == GLOBAL and desc.transient
    assert tuple(desc.shape) == (ROWS, 1), desc.shape
    assert None in [sdfg.start_state.entry_node(n) for n in sdfg.start_state.data_nodes() if n.data == 'acc']
    assert_no_wcr_target_in_registers(sdfg)
    sdfg.validate()


def test_a_small_plain_buffer_is_demoted():
    """Negative control: without the WCR the same buffer goes to registers and keeps its shape."""
    sdfg = kernel_with_accumulator(wcr=False)

    assert MoveArrayOutOfKernel().apply_pass(sdfg, {}) == 1

    assert sdfg.arrays['acc'].storage == dtypes.StorageType.Register
    assert tuple(sdfg.arrays['acc'].shape) == (1, )


@pytest.mark.gpu
def test_the_lifted_accumulator_sums_each_row():
    import cupy  # Only present on GPU runners.
    sdfg = kernel_with_accumulator(wcr=True)
    with pytest.warns(UserWarning, match='will be lifted outside the kernel'):
        MoveArrayOutOfKernel().apply_pass(sdfg, {})
    A = cupy.arange(ROWS * COLS, dtype=cupy.float64).reshape(ROWS, COLS)
    out = cupy.zeros(ROWS, dtype=cupy.float64)

    sdfg(A=A, out=out)

    assert np.array_equal(cupy.asnumpy(out), cupy.asnumpy(A).sum(axis=1))


@pytest.mark.gpu
def test_wcr_via_augmented_assign():
    """``acc[0] += A[i]`` in a GPU_Device map accumulates atomically; the accumulator is not demoted."""

    @dace.program
    def aug_assign(A: dace.float64[64] @ dace.StorageType.GPU_Global,
                   acc: dace.float64[1] @ dace.StorageType.GPU_Global):
        for i in dace.map[0:64] @ dace.ScheduleType.GPU_Device:
            acc[0] += A[i]

    import cupy as cp
    sdfg = aug_assign.to_sdfg()
    set_block_sizes(sdfg)
    MoveArrayOutOfKernel().apply_pass(sdfg, {})
    assert_no_wcr_target_in_registers(sdfg)
    assert sdfg.arrays['acc'].storage == GLOBAL and tuple(sdfg.arrays['acc'].shape) == (1, )

    A = cp.arange(64, dtype=cp.float64)
    acc = cp.zeros(1, dtype=cp.float64)
    sdfg(A=A, acc=acc)
    assert float(acc[0]) == float(cp.sum(A))


@pytest.mark.gpu
def test_wcr_via_reduction_kernel():
    """Row-reduction kernel: a 2D map atomically accumulates each row of ``A`` into ``row_sums[i]``."""

    @dace.program
    def row_reduce(A: dace.float64[8, 8] @ dace.StorageType.GPU_Global,
                   row_sums: dace.float64[8] @ dace.StorageType.GPU_Global):
        for i, j in dace.map[0:8, 0:8] @ dace.ScheduleType.GPU_Device:
            row_sums[i] += A[i, j]

    import cupy as cp
    sdfg = row_reduce.to_sdfg()
    set_block_sizes(sdfg)
    MoveArrayOutOfKernel().apply_pass(sdfg, {})
    assert_no_wcr_target_in_registers(sdfg)
    assert sdfg.arrays['row_sums'].storage == GLOBAL and tuple(sdfg.arrays['row_sums'].shape) == (8, )

    A = cp.arange(64, dtype=cp.float64).reshape(8, 8)
    row_sums = cp.zeros(8, dtype=cp.float64)
    sdfg(A=A, row_sums=row_sums)
    cp.testing.assert_array_equal(row_sums, A.sum(axis=1))


@pytest.mark.gpu
def test_wcr_np_sum_small_n_auto_staging():
    """``total[0] = np.sum(A)`` with no storage annotations reduces correctly after
    ``auto_optimize`` for GPU."""
    from dace.dtypes import DeviceType
    from dace.transformation.auto.auto_optimize import auto_optimize

    @dace.program
    def reduce_sum(A: dace.float64[64], total: dace.float64[1]):
        total[0] = np.sum(A)

    sdfg = reduce_sum.to_sdfg()
    auto_optimize(sdfg, DeviceType.GPU)
    set_block_sizes(sdfg)
    before = {(owner.name, name): desc.storage for owner, name in wcr_targets(sdfg) for desc in [owner.arrays[name]]}
    MoveArrayOutOfKernel().apply_pass(sdfg, {})
    after = {(owner.name, name): owner.arrays[name].storage for owner, name in wcr_targets(sdfg)}
    assert before and after == before, (before, after)

    A = np.arange(64, dtype=np.float64)
    total = np.zeros(1, dtype=np.float64)
    sdfg(A=A, total=total)
    assert total[0] == np.sum(A)
