# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests used-symbols in code generation."""
import dace
import numpy
import pytest

n0i, n0j, n0k = (dace.symbol(s, dtype=dace.int32) for s in ('n0i', 'n0j', 'n0k'))
n1i, n1j, n1k = (dace.symbol(s, dtype=dace.int64) for s in ('n1i', 'n1j', 'n1k'))


@dace.program
def rprj3(r: dace.float64[n0i, n0j, n0k], s: dace.float64[n1i, n1j, n1k]):

    for i, j, k in dace.map[1:s.shape[0] - 1, 1:s.shape[1] - 1, 1:s.shape[2] - 1]:

        s[i, j,
          k] = (0.5000 * r[2 * i, 2 * j, 2 * k] + 0.2500 *
                (r[2 * i - 1, 2 * j, 2 * k] + r[2 * i + 1, 2 * j, 2 * k] + r[2 * i, 2 * j - 1, 2 * k] +
                 r[2 * i, 2 * j + 1, 2 * k] + r[2 * i, 2 * j, 2 * k - 1] + r[2 * i, 2 * j, 2 * k + 1]) + 0.1250 *
                (r[2 * i - 1, 2 * j - 1, 2 * k] + r[2 * i - 1, 2 * j + 1, 2 * k] + r[2 * i + 1, 2 * j - 1, 2 * k] +
                 r[2 * i + 1, 2 * j + 1, 2 * k] + r[2 * i - 1, 2 * j, 2 * k - 1] + r[2 * i - 1, 2 * j, 2 * k + 1] +
                 r[2 * i + 1, 2 * j, 2 * k - 1] + r[2 * i + 1, 2 * j, 2 * k + 1] + r[2 * i, 2 * j - 1, 2 * k - 1] +
                 r[2 * i, 2 * j - 1, 2 * k + 1] + r[2 * i, 2 * j + 1, 2 * k - 1] + r[2 * i, 2 * j + 1, 2 * k + 1]) +
                0.0625 * (r[2 * i - 1, 2 * j - 1, 2 * k - 1] + r[2 * i - 1, 2 * j - 1, 2 * k + 1] +
                          r[2 * i - 1, 2 * j + 1, 2 * k - 1] + r[2 * i - 1, 2 * j + 1, 2 * k + 1] +
                          r[2 * i + 1, 2 * j - 1, 2 * k - 1] + r[2 * i + 1, 2 * j - 1, 2 * k + 1] +
                          r[2 * i + 1, 2 * j + 1, 2 * k - 1] + r[2 * i + 1, 2 * j + 1, 2 * k + 1]))


def test_codegen_used_symbols_cpu():

    rng = numpy.random.default_rng(42)
    r = rng.random((10, 10, 10))
    s_ref = numpy.zeros((4, 4, 4))
    s_val = numpy.zeros((4, 4, 4))

    rprj3.f(r, s_ref)
    rprj3(r, s_val)

    assert numpy.allclose(s_ref, s_val)


def test_codegen_used_symbols_cpu_2():

    @dace.program
    def rprj3_nested(r: dace.float64[n0i, n0j, n0k], s: dace.float64[n1i, n1j, n1k]):
        rprj3(r, s)

    rng = numpy.random.default_rng(42)
    r = rng.random((10, 10, 10))
    s_ref = numpy.zeros((4, 4, 4))
    s_val = numpy.zeros((4, 4, 4))

    rprj3.f(r, s_ref)
    rprj3_nested(r, s_val)

    assert numpy.allclose(s_ref, s_val)


@pytest.mark.gpu
def test_codegen_used_symbols_gpu():

    sdfg = rprj3.to_sdfg()
    for _, desc in sdfg.arrays.items():
        if not desc.transient and isinstance(desc, dace.data.Array):
            desc.storage = dace.StorageType.GPU_Global
    sdfg.apply_gpu_transformations()
    func = sdfg.compile()

    try:
        import cupy

        rng = numpy.random.default_rng(42)
        r = rng.random((10, 10, 10))
        r_dev = cupy.asarray(r)
        s_ref = numpy.zeros((4, 4, 4))
        s_val = cupy.zeros((4, 4, 4))

        rprj3.f(r, s_ref)
        func(r=r_dev, s=s_val, n0i=10, n0j=10, n0k=10, n1i=4, n1j=4, n1k=4)

        assert numpy.allclose(s_ref, s_val)

    except (ImportError, ModuleNotFoundError):
        pass


def test_codegen_edge_assignment_with_indirection():
    rng = numpy.random.default_rng(42)
    (M, N, K) = (dace.symbol(x, dace.int32) for x in ['M', 'N', 'K'])

    sdfg = dace.SDFG('edge_assignment_with_indirection')
    [sdfg.add_symbol(x, dace.int32) for x in {'__indirect_idx', '__neighbor_idx'}]
    sdfg.add_array('_field', (M, ), dace.float64)
    sdfg.add_array('_table', (N, K), dace.int32)
    sdfg.add_array('_out', (N, ), dace.float64)

    state0 = sdfg.add_state(is_start_block=True)
    state1 = sdfg.add_state()
    sdfg.add_edge(state0, state1,
                  dace.InterstateEdge(assignments={'_field_idx': '_table[__indirect_idx, __neighbor_idx]'}))
    state1.add_memlet_path(state1.add_access('_field'),
                           state1.add_access('_out'),
                           memlet=dace.Memlet(data='_out',
                                              subset='__indirect_idx',
                                              other_subset='_field_idx',
                                              wcr='lambda x, y: x + y'))

    M, N, K = (5, 4, 2)
    field = rng.random((M, ))
    out = rng.random((N, ))
    table = numpy.random.randint(0, M, (N, K), numpy.int32)

    TEST_INDIRECT_IDX = numpy.random.randint(0, N)
    TEST_NEIGHBOR_IDX = numpy.random.randint(0, K)

    reference = numpy.asarray(
        [out[i] + field[table[i, TEST_NEIGHBOR_IDX]] if i == TEST_INDIRECT_IDX else out[i] for i in range(N)])

    sdfg(_field=field,
         _table=table,
         _out=out,
         M=M,
         N=N,
         K=K,
         __indirect_idx=TEST_INDIRECT_IDX,
         __neighbor_idx=TEST_NEIGHBOR_IDX)

    assert numpy.allclose(out, reference)


if __name__ == "__main__":

    test_codegen_used_symbols_cpu()
    test_codegen_used_symbols_cpu_2()
    test_codegen_used_symbols_gpu()
    test_codegen_edge_assignment_with_indirection()
