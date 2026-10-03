# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Unit tests for the GPU to-device transformation. """

import re

import dace
import numpy as np
import pytest
from dace.transformation.interstate import GPUTransformSDFG


def test_toplevel_transient_lifetime():
    N = dace.symbol('N')

    @dace.program
    def program(A: dace.float64[20, 20]):
        for i in range(20):
            tmp = A[:i, :i]
            tmp2 = A[:5, :N]
            tmp *= 5
            tmp2 *= 10

    sdfg = program.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, options=dict(toplevel_trans=True))

    for name, desc in sdfg.arrays.items():
        if name == 'tmp2' and type(desc) is dace.data.Array:
            assert desc.lifetime is dace.AllocationLifetime.SDFG
        else:
            assert desc.lifetime is not dace.AllocationLifetime.SDFG


@pytest.mark.gpu
def test_scalar_to_symbol_in_nested_sdfg():
    """
    GPUTransformSDFG will automatically create copy-out states for GPU scalars that are used in host-side interstate
    edges. However, this process may only be applied in top-level SDFGs and not in NestedSDFGs that have GPU-device
    schedule but are not part of a single GPU kernel, leading to illegal memory accesses.
    """

    @dace.program
    def nested_program(a: dace.int32, out: dace.int32[10]):
        for i in range(10):
            if a < 5:
                out[i] = 0
                a *= 2
            else:
                out[i] = 10
                a /= 2

    @dace.program
    def main_program(a: dace.int32):
        out = np.ndarray((10, ), dtype=np.int32)
        nested_program(a, out)
        return out

    sdfg = main_program.to_sdfg(simplify=False)
    sdfg.apply_transformations(GPUTransformSDFG)
    out = sdfg(a=4)
    assert np.array_equal(out, np.array([0, 10] * 5, dtype=np.int32))


@pytest.mark.gpu
def test_write_subset():

    @dace.program
    def write_subset(A: dace.int32[20, 20]):
        for i, j in dace.map[2:18, 2:18]:
            A[i, j] = i + j

    sdfg = write_subset.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)

    ref = np.ones((20, 20), dtype=np.int32)
    val = np.copy(ref)

    write_subset.f(ref)
    sdfg(A=val)

    assert np.array_equal(ref, val)


def test_write_full():

    M, N = dace.symbol('M'), dace.symbol('N')

    @dace.program
    def write_full(A: dace.int32[M, N]):
        for i, j in dace.map[0:M, 0:N]:
            A[i, j] = i + j

    sdfg = write_full.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)

    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode) and node.data == 'A':
                assert state.out_degree(node) == 0


@pytest.mark.gpu
def test_write_subset_dynamic():

    @dace.program
    def write_subset_dynamic(A: dace.int32[20, 20], x: dace.int32[20], y: dace.int32[20]):
        for i, j in dace.map[2:18, 2:18]:
            A[x[i], y[j]] = i + j

    sdfg = write_subset_dynamic.to_sdfg(simplify=True)
    sdfg.apply_transformations(GPUTransformSDFG)

    ref = np.ones((20, 20), dtype=np.int32)
    val = np.copy(ref)

    x = np.random.permutation(20).astype(np.int32)
    y = np.random.permutation(20).astype(np.int32)

    write_subset_dynamic.f(ref, x, y)
    sdfg(A=val, x=x, y=y)

    assert np.array_equal(ref, val)


@pytest.mark.parametrize(["transient", "scalar"], [[False, False], [False, True], [True, False], [True, True]])
def test_free_tasklet(transient, scalar):
    sdfg = dace.SDFG("assign")

    state = sdfg.add_state("main")
    if scalar:
        arr_name, arr = sdfg.add_scalar("A", dace.float32, transient=transient)
    else:
        arr_name, arr = sdfg.add_array("A", (4, ), dace.float32, transient=transient)

    an = state.add_access(arr_name)

    t = state.add_tasklet("assign", {}, {"_out"}, "_out = 2.0")
    state.add_edge(t, "_out", an, None, dace.memlet.Memlet("A" if scalar else "A[0]"))

    sdfg.validate()

    sdfg.apply_gpu_transformations(validate=True,
                                   validate_all=True,
                                   permissive=True,
                                   sequential_innermaps=True,
                                   register_transients=False,
                                   simplify=False)

    sdfg.validate()


if __name__ == '__main__':
    test_toplevel_transient_lifetime()
    test_scalar_to_symbol_in_nested_sdfg()
    test_write_subset()
    test_write_full()
    test_write_subset_dynamic()
    for scalar in [False, True]:
        for transient in [False, True]:
            test_free_tasklet(transient, scalar)


@pytest.mark.gpu
def test_an_interstate_condition_reading_a_gpu_map_output_reads_a_host_copy():
    """A GPU map writes ``flag`` and the next interstate condition reads ``flag[0]`` on the host. The
    transformation must register the moved array so a host copy-out precedes the reading edge;
    without it the condition names device memory and validation rejects the SDFG."""
    sdfg = dace.SDFG("gpu_output_gate")
    sdfg.add_array("flag", [4], dace.int32, transient=True)
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("out", [8], dace.float64)
    set_flag = sdfg.add_state("set_flag", is_start_block=True)
    set_flag.add_mapped_tasklet("write_flag", {"i": "0:4"}, {},
                                "o = 1", {"o": dace.Memlet("flag[i]")},
                                external_edges=True)
    compute = sdfg.add_state("compute")
    compute.add_mapped_tasklet("double", {"j": "0:8"}, {"a": dace.Memlet("A[j]")},
                               "o = 2 * a", {"o": dace.Memlet("out[j]")},
                               external_edges=True)
    end = sdfg.add_state("end")
    sdfg.add_edge(set_flag, compute, dace.InterstateEdge("flag[0] > 0"))
    sdfg.add_edge(set_flag, end, dace.InterstateEdge("not (flag[0] > 0)"))
    sdfg.add_edge(compute, end, dace.InterstateEdge())

    sdfg.apply_transformations(GPUTransformSDFG, options=dict(simplify=False))

    assert sdfg.arrays["flag"].storage == dace.StorageType.GPU_Global
    host_copies = [
        name for name, desc in sdfg.arrays.items()
        if name.startswith("host_flag") and desc.storage == dace.StorageType.CPU_Heap
    ]
    assert len(host_copies) == 1, sorted(sdfg.arrays)
    conditions = [e.data.condition.as_string for e in sdfg.all_interstate_edges() if not e.data.is_unconditional()]
    assert len(conditions) == 2, conditions
    assert all(host_copies[0] in c and not re.search(r"\bflag\[", c) for c in conditions), conditions

    A = np.random.rand(8)
    out = np.zeros(8)
    sdfg(A=A, out=out)
    assert np.allclose(out, 2 * A)
