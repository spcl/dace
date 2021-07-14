# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg import nodes
from dace import dtypes
import numpy as np
from numba import cuda
import pytest

# Define data type to use
dtype = dace.float64
np_dtype = np.float64


def create_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('aCPU_aGPU')

    # add arrays and transients
    sdfg.add_array('a', [1], dtype)
    sdfg.add_array('b', [1], dtype)
    sdfg.add_transient('gpu_a', [1], dtype, dtypes.StorageType.GPU_Global)

    a = nodes.AccessNode('a')
    b = nodes.AccessNode('b')
    gpu_a = nodes.AccessNode('gpu_a')

    # copyin state
    copyin_state = sdfg.add_state(sdfg.label + '_copyin')
    copyin_state.add_node(a)
    copyin_state.add_node(gpu_a)
    copyin_memlet = dace.Memlet.from_array(a, a.desc(copyin_state))
    copyin_state.add_edge(a, None, gpu_a, None, copyin_memlet)

    # wcr state
    wcr_state = sdfg.add_state(sdfg.label + '_wcr')
    sdfg.add_edge(copyin_state, wcr_state, dace.InterstateEdge())
    wcr_state.add_node(gpu_a)
    wcr_state.add_node(b)
    wcr_memlet = dace.Memlet.from_array(gpu_a, gpu_a.desc(wcr_state))
    wcr_memlet.wcr = 'lambda old, new: new'
    wcr_state.add_edge(b, None, gpu_a, None, wcr_memlet)

    # copyout state
    copyout_state = sdfg.add_state(sdfg.label + '_copyout')
    sdfg.add_edge(wcr_state, copyout_state, dace.InterstateEdge())
    copyout_state.add_node(a)
    copyout_state.add_node(gpu_a)
    copyout_memlet = dace.Memlet.from_array(gpu_a, gpu_a.desc(copyout_state))
    copyout_state.add_edge(gpu_a, None, a, None, copyout_memlet)

    return sdfg


@pytest.mark.multigpu
def test_aCPU_aGPU():
    sdfg = create_sdfg()

    np.random.seed(0)
    a = cuda.pinned_array(shape=1, dtype=np_dtype)
    a.fill(1)
    b = cuda.pinned_array(shape=1, dtype=np_dtype)
    b.fill(10)

    sdfg(a=a, b=b)
    assert (b[0] == 1 and a[0] == 10)

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/reductions/'+sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_aCPU_aGPU()