import dace
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode

import pytest
import numpy as np

def _get_sdfg(implementation, gpu=True) -> dace.SDFG:
    sdfg = dace.SDFG("memset_sdfg")
    name = "gpuB" if gpu else "B"
    sdfg.add_array(name=name,
                   shape=[200,],
                   dtype=dace.dtypes.float64,
                   storage=dace.dtypes.StorageType.GPU_Global if gpu else dace.dtypes.StorageType.CPU_Heap,
                   transient=False)

    state = sdfg.add_state("main")

    b1 = state.add_access(name)

    libnode = MemsetLibraryNode(name="memset1",
                                inputs={},
                                outputs={name})
    if implementation is not None:
        libnode.implementation = implementation

    # Only set a slice
    state.add_edge(libnode, name, b1, None, dace.memlet.Memlet(f"{name}[50:100]"))

    return sdfg


def test_memset_pure_cpu():
    sdfg = _get_sdfg("pure", gpu=False)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = np.zeros((200,), dtype=np.float64)
    exe(B=B)

    # Check that the slice is set to 0 (default memset value)
    np.testing.assert_array_equal(B[50:100], 0)
    # Other parts should remain untouched
    assert np.all(B[:50] == 0)
    assert np.all(B[100:] == 0)


@pytest.mark.gpu
def test_memset_pure_gpu():
    import cupy as cp

    sdfg = _get_sdfg("pure", gpu=True)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = cp.zeros((200,), dtype=cp.float64)
    exe(B=B)

    cp.testing.assert_array_equal(B[50:100], 0)
    assert cp.all(B[:50] == 0)
    assert cp.all(B[100:] == 0)


@pytest.mark.gpu
def test_memset_cuda_gpu():
    import cupy as cp

    sdfg = _get_sdfg("CUDA", gpu=True)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = cp.zeros((200,), dtype=cp.float64)
    exe(B=B)

    cp.testing.assert_array_equal(B[50:100], 0)
    assert cp.all(B[:50] == 0)
    assert cp.all(B[100:] == 0)


@pytest.mark.gpu
def test_memset_cuda_cpu():
    import numpy as np

    # Test CUDA implementation on CPU arrays
    sdfg = _get_sdfg("CUDA", gpu=False)
    sdfg.validate()
    sdfg.expand_library_nodes()
    with pytest.raises(Exception):
        sdfg.validate()
        sdfg.compile()


if __name__ == "__main__":
    test_memset_pure_cpu()
    test_memset_pure_gpu()
    test_memset_cuda_gpu()
    test_memset_cuda_cpu()
