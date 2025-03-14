""" Tests code generation for array copy on GPU target. """
import dace
from dace.transformation.auto import auto_optimize
from dace.sdfg import nodes as dace_nodes

import pytest
import copy
import re

# this test requires cupy module
cp = pytest.importorskip("cupy")

# initialize random number generator
rng = cp.random.default_rng(42)


def count_node(sdfg: dace.SDFG, node_type):
    nb_nodes = 0
    for rsdfg in sdfg.all_sdfgs_recursive():
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, node_type):
                    nb_nodes += 1
    return nb_nodes


def _make_2d_gpu_copy_sdfg(c_order: bool, ) -> dace.SDFG:
    """The SDFG performs a copy from the input of the output, that is continuous.

    Essentially the function will generate am SDFG that performs the following
    operation:
    ```python
        B[2:7, 3:9] = A[1:6, 2:8]
    ```
    However, two arrays have a shape of `(20, 30)`. This means that this copy
    can not be expressed as a continuous copy. Regardless which memory order
    that is used, which can be selected by `c_order`.
    """
    sdfg = dace.SDFG(f'gpu_2d_copy_{"corder" if c_order else "forder"}_copy_sdfg')
    state = sdfg.add_state(is_start_block=True)

    for aname in 'AB':
        sdfg.add_array(
            name=aname,
            shape=(20, 30),
            dtype=dace.float64,
            storage=dace.StorageType.GPU_Global,
            transient=False,
            strides=((30, 1) if c_order else (1, 20)),
        )

    state.add_nedge(
        state.add_access("A"),
        state.add_access("B"),
        dace.Memlet("A[1:6, 2:8] -> [2:7, 3:9]"),
    )
    sdfg.validate()

    return sdfg


def _perform_2d_gpu_copy_test(c_order: bool, ):
    """Check 2D strided copies are handled by the `Memcpy2D` family.
    """
    sdfg = _make_2d_gpu_copy_sdfg(c_order=c_order)
    assert count_node(sdfg, dace_nodes.AccessNode) == 2
    assert count_node(sdfg, dace_nodes.MapEntry) == 0

    # Now generate the code.
    csdfg = sdfg.compile()

    # Ensure that the copy was not turned into a Map
    assert count_node(csdfg.sdfg, dace_nodes.AccessNode) == 2
    assert count_node(csdfg.sdfg, dace_nodes.MapEntry) == 0

    # Ensure that the correct call was issued.
    #  We have to look at the CPU code and not at the GPU.
    code = sdfg.generate_code()[0].clean_code
    m = re.search(r'(cuda|hip)Memcpy2DAsync\b', code)
    assert m is not None

    # Now run the sdfg.
    ref = {
        "A": cp.array(cp.zeros((20, 30)), dtype=cp.float64, order="C" if c_order else "F"),
        "B": cp.array(cp.ones((20, 30)), dtype=cp.float64, order="C" if c_order else "F"),
    }

    # We can not use `deepcopy` or `.copy()` because this would set the strides to `C` order.
    res = {}
    for name in ref.keys():
        res[name] = cp.empty_like(ref[name])
        res[name][:] = ref[name][:]

    exp_strides = (240, 8) if c_order else (8, 160)
    assert all(v.strides == exp_strides for v in ref.values())
    assert all(v.strides == exp_strides for v in res.values())

    # Now apply the operation on the reference
    ref["B"][2:7, 3:9] = ref["A"][1:6, 2:8]

    # Now run the SDFG
    csdfg(**res)

    assert all(cp.all(ref[k] == res[k]) for k in ref.keys())


@pytest.mark.gpu
def test_gpu_shared_to_global_1D():
    M = 32
    N = dace.symbol('N')

    @dace.program
    def transpose_shared_to_global(A: dace.float64[M, N], B: dace.float64[N, M]):
        for i in dace.map[0:N]:
            local_gather = dace.define_local([M], A.dtype, storage=dace.StorageType.GPU_Shared)
            for j in dace.map[0:M]:
                local_gather[j] = A[j, i]
            B[i, :] = local_gather

    sdfg = transpose_shared_to_global.to_sdfg()
    auto_optimize.apply_gpu_storage(sdfg)

    size_M = M
    size_N = 128

    A = rng.random((
        size_M,
        size_N,
    ))
    B = rng.random((
        size_N,
        size_M,
    ))

    ref = A.transpose()

    sdfg(A, B, N=size_N)
    cp.allclose(ref, B)

    code = sdfg.generate_code()[1].clean_code  # Get GPU code (second file)
    m = re.search('dace::SharedToGlobal1D<.+>::Copy', code)
    assert m is not None


@pytest.mark.gpu
def test_gpu_shared_to_global_1D_accumulate():
    M = 32
    N = dace.symbol('N')

    @dace.program
    def transpose_and_add_shared_to_global(A: dace.float64[M, N], B: dace.float64[N, M]):
        for i in dace.map[0:N]:
            local_gather = dace.define_local([M], A.dtype, storage=dace.StorageType.GPU_Shared)
            for j in dace.map[0:M]:
                local_gather[j] = A[j, i]
            local_gather[:] >> B(M, lambda x, y: x + y)[i, :]

    sdfg = transpose_and_add_shared_to_global.to_sdfg()
    auto_optimize.apply_gpu_storage(sdfg)

    size_M = M
    size_N = 128

    A = rng.random((
        size_M,
        size_N,
    ))
    B = rng.random((
        size_N,
        size_M,
    ))

    ref = A.transpose() + B

    sdfg(A, B, N=size_N)
    cp.allclose(ref, B)

    code = sdfg.generate_code()[1].clean_code  # Get GPU code (second file)
    m = re.search('dace::SharedToGlobal1D<.+>::template Accum', code)
    assert m is not None


def test_2d_c_order_gpu_copy():
    _perform_2d_gpu_copy_test(c_order=True)


def test_2d_f_order_gpu_copy():
    _perform_2d_gpu_copy_test(c_order=False)


if __name__ == '__main__':
    test_gpu_shared_to_global_1D()
    test_gpu_shared_to_global_1D_accumulate()
    test_2d_c_order_copy()
    test_2d_f_order_copy()
