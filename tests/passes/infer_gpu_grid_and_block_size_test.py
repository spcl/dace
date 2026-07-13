# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`InferGPUGridAndBlockSize`: reconciling an explicit ``gpu_block_size`` with the
sizes of nested ``GPU_ThreadBlock`` maps."""
import dace
import pytest

from dace.transformation.passes.analysis.infer_gpu_grid_and_block_size import InferGPUGridAndBlockSize


def _kernel_with_nested_threadblock(user_block_size, tb_extent: int) -> tuple:
    """A ``GPU_Device`` kernel with ``gpu_block_size = user_block_size`` wrapping a single nested
    ``GPU_ThreadBlock`` map of range ``0:tb_extent``. Returns ``(sdfg, state, dev_entry)``."""
    sdfg = dace.SDFG('infer_block')
    sdfg.add_array('A', [tb_extent], dace.float64, storage=dace.dtypes.StorageType.GPU_Global, transient=False)
    state = sdfg.add_state('s')

    dev_me, dev_mx = state.add_map('kernel', dict(i='0:1'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    dev_me.map.gpu_block_size = list(user_block_size)
    tb_me, tb_mx = state.add_map('tb', dict(j=f'0:{tb_extent}'), schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock)
    t = state.add_tasklet('w', {}, {'o'}, 'o = 1.0')
    a = state.add_write('A')

    tb_mx.add_scope_connectors('A')
    dev_mx.add_scope_connectors('A')
    state.add_nedge(dev_me, tb_me, dace.Memlet())
    state.add_nedge(tb_me, t, dace.Memlet())
    state.add_edge(t, 'o', tb_mx, 'IN_A', dace.Memlet('A[j]'))
    state.add_edge(tb_mx, 'OUT_A', dev_mx, 'IN_A', dace.Memlet(f'A[0:{tb_extent}]'))
    state.add_edge(dev_mx, 'OUT_A', a, None, dace.Memlet(f'A[0:{tb_extent}]'))
    return sdfg, state, dev_me


def test_infer_block_size_conflict_with_larger_threadblock_map():
    """A user ``gpu_block_size`` smaller than a nested ``GPU_ThreadBlock`` map is a conflict and must
    raise. Regression: the conflict check compared the running elementwise max against the current
    size, so a monotonically larger thread-block size ([64,1,1] vs the declared [32,1,1]) was
    silently accepted and the user's block size overridden."""
    sdfg, _state, _dev = _kernel_with_nested_threadblock([32, 1, 1], tb_extent=64)
    with pytest.raises(ValueError):
        InferGPUGridAndBlockSize().apply_pass(sdfg, set())


def test_infer_block_size_matching_user_and_threadblock_no_conflict():
    """A user ``gpu_block_size`` equal to the nested thread-block size is not a conflict."""
    sdfg, _state, dev = _kernel_with_nested_threadblock([64, 1, 1], tb_extent=64)
    dims = InferGPUGridAndBlockSize().apply_pass(sdfg, set())
    _grid, block = dims[dev]
    assert [int(b) for b in block] == [64, 1, 1], block


if __name__ == '__main__':
    test_infer_block_size_conflict_with_larger_threadblock_map()
    test_infer_block_size_matching_user_and_threadblock_no_conflict()
