# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.libraries.linalg import TensorDot
from dace.sdfg.infer_types import set_default_schedule_and_storage_types


def test_notbmap():
    sdfg = dace.SDFG('default_storage_test_1')
    sdfg.add_array('A', [20], dace.float64, dace.StorageType.GPU_Global)
    sdfg.add_transient('tmp', [1], dace.float64)
    state = sdfg.add_state()

    r = state.add_read('A')
    me, mx = state.add_map('kernel', dict(i='0:20'), dace.ScheduleType.GPU_Device)
    tmp = state.add_access('tmp')
    t = state.add_tasklet('add', {'a'}, {'b'}, 'b = a + 1')
    w = state.add_write('A')

    state.add_memlet_path(r, me, tmp, memlet=dace.Memlet.simple('A', 'i'))
    state.add_memlet_path(tmp, t, dst_conn='a', memlet=dace.Memlet.simple('tmp', '0'))
    state.add_memlet_path(t, mx, w, src_conn='b', memlet=dace.Memlet.simple('A', 'i'))

    set_default_schedule_and_storage_types(sdfg, None)
    assert sdfg.arrays['tmp'].storage == dace.StorageType.Register


def test_tbmap_sequential():
    sdfg = dace.SDFG('default_storage_test_2')
    sdfg.add_array('A', [20, 32], dace.float64, dace.StorageType.GPU_Global)
    sdfg.add_transient('tmp', [1], dace.float64)
    state = sdfg.add_state()

    r = state.add_read('A')
    ome, omx = state.add_map('kernel', dict(i='0:20'), dace.ScheduleType.GPU_Device)
    sme, smx = state.add_map('seq', dict(j='0:1'), dace.ScheduleType.Sequential)
    ime, imx = state.add_map('block', dict(ti='0:32'), dace.ScheduleType.GPU_ThreadBlock)
    tmp = state.add_access('tmp')
    t = state.add_tasklet('add', {'a'}, {'b'}, 'b = a + 1')
    w = state.add_write('A')

    state.add_memlet_path(r, ome, sme, tmp, memlet=dace.Memlet.simple('A', 'i+j, 0:32'))
    state.add_memlet_path(tmp, ime, t, dst_conn='a', memlet=dace.Memlet.simple('tmp', '0, ti'))
    state.add_memlet_path(t, imx, smx, omx, w, src_conn='b', memlet=dace.Memlet.simple('A', 'i+j, ti'))
    set_default_schedule_and_storage_types(sdfg, None)
    assert sdfg.arrays['tmp'].storage == dace.StorageType.GPU_Shared


def ttgt_contraction_sdfg() -> dace.SDFG:
    """A TTGT tensor contraction whose operands both need a transpose, storage not yet inferred."""

    @dace.program
    def contraction(A: dace.float32[3, 3, 3, 3], B: dace.float32[3, 3, 3, 3], C: dace.float32[3, 3, 3, 3]):
        C[:] = np.tensordot(A, B, axes=([0, 3], [3, 1]))

    sdfg = contraction.to_sdfg(simplify=True)
    tensordot = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TensorDot))
    tensordot.implementation = 'TTGT'
    return sdfg


def transposed_operand_storages(sdfg: dace.SDFG) -> list[dace.StorageType]:
    expansion = next(nsdfg for nsdfg in sdfg.all_sdfgs_recursive() if nsdfg.parent_sdfg is sdfg)
    return [expansion.arrays[name].storage for name in ('ttgt_left_transposed', 'ttgt_right_transposed')]


def test_expansion_before_inference_leaves_accessed_transients_undecided():
    sdfg = ttgt_contraction_sdfg()

    sdfg.expand_library_nodes(recursive=False)

    assert transposed_operand_storages(sdfg) == [dace.StorageType.Default] * 2
    sdfg.validate()


def test_transposed_operands_of_an_early_expansion_live_on_the_heap():
    sdfg = ttgt_contraction_sdfg()
    sdfg.expand_library_nodes(recursive=False)

    set_default_schedule_and_storage_types(sdfg, None)

    assert transposed_operand_storages(sdfg) == [dace.StorageType.CPU_Heap] * 2
    sdfg.validate()


if __name__ == '__main__':
    test_notbmap()
    test_tbmap_sequential()
    test_expansion_before_inference_leaves_accessed_transients_undecided()
    test_transposed_operands_of_an_early_expansion_live_on_the_heap()
