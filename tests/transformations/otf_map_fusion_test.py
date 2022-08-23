# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
import sympy

from scipy.signal import convolve2d

from dace.transformation.dataflow import OTFMapFusion

N = dace.symbol("N")
M = dace.symbol("M")


def count_maps(sdfg):
    maps = 0
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                maps += 1

    return maps


@dace.program
def fusion_chain(A: dace.float64[10, 20], B: dace.float64[10, 20]):
    tmp = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            b >> tmp[i, j]

            b = a * a

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << tmp[i, j]
            b >> B[i, j]

            b = a + 2


@dace.program
def fusion_chain_renamed(A: dace.float64[10, 20], B: dace.float64[10, 20]):
    tmp = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            b >> tmp[i, j]

            b = a * a

    for k, l in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << tmp[k, l]
            b >> B[k, l]

            b = a + 2


@dace.program
def fusion_flip(A: dace.float64[N, M], B: dace.float64[N, M]):
    tmp = dace.define_local([N, M], dtype=A.dtype)
    for i, j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            a << A[i, j]
            b >> tmp[i, j]

            b = a * a

    for k, l in dace.map[0:N, 0:M]:
        with dace.tasklet:
            a << tmp[N - k - 1, M - l - 1]
            b >> B[k, l]

            b = a + 2


@dace.program
def fusion_transposed(A: dace.float64[10, 20], B: dace.float64[20, 10]):
    tmp = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            b >> tmp[i, j]

            b = a * a

    for i, j in dace.map[0:20, 0:10]:
        with dace.tasklet:
            a << tmp[j, i]
            b >> B[i, j]

            b = a + 2


@dace.program
def fusion_convolve(A: dace.float64[20, 20], B: dace.float64[16, 16]):
    tmp = dace.define_local([18, 18], dtype=A.dtype)
    for i, j in dace.map[1:19, 1:19]:
        with dace.tasklet:
            a0 << A[i - 1, j - 1]
            a1 << A[i - 1, j]
            a2 << A[i - 1, j + 1]
            a3 << A[i, j - 1]
            a4 << A[i, j]
            a5 << A[i, j + 1]
            a6 << A[i + 1, j - 1]
            a7 << A[i + 1, j]
            a8 << A[i + 1, j + 1]
            b >> tmp[i - 1, j - 1]

            b = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) / 9.0

    for i, j in dace.map[1:17, 1:17]:
        with dace.tasklet:
            a0 << tmp[i - 1, j - 1]
            a1 << tmp[i - 1, j]
            a2 << tmp[i - 1, j + 1]
            a3 << tmp[i, j - 1]
            a4 << tmp[i, j]
            a5 << tmp[i, j + 1]
            a6 << tmp[i + 1, j - 1]
            a7 << tmp[i + 1, j]
            a8 << tmp[i + 1, j + 1]
            b >> B[i - 1, j - 1]

            b = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) / 9.0


@dace.program
def fusion_convolve_overlap(A: dace.float64[20, 20], B: dace.float64[16, 16]):
    tmp = dace.define_local([18, 18], dtype=A.dtype)
    for i, j in dace.map[1:19, 1:19]:
        with dace.tasklet:
            a0 << A[i - 1, j]
            a1 << A[i - 1, j]
            a2 << A[i - 1, j + 1]
            a3 << A[i, j]
            a4 << A[i, j]
            a5 << A[i, j + 1]
            a6 << A[i + 1, j - 1]
            a7 << A[i + 1, j]
            a8 << A[i + 1, j + 1]
            b >> tmp[i - 1, j - 1]

            b = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) / 9.0

    for i, j in dace.map[1:17, 1:17]:
        with dace.tasklet:
            a0 << tmp[i - 1, j]
            a1 << tmp[i - 1, j]
            a2 << tmp[i - 1, j + 1]
            a3 << tmp[i, j]
            a4 << tmp[i, j]
            a5 << tmp[i, j + 1]
            a6 << tmp[i + 1, j - 1]
            a7 << tmp[i + 1, j]
            a8 << tmp[i + 1, j + 1]
            b >> B[i - 1, j - 1]

            b = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) / 9.0


@dace.program
def fusion_convolve_transposed(A: dace.float64[20, 20], B: dace.float64[16, 16]):
    tmp = dace.define_local([18, 18], dtype=A.dtype)
    for i, j in dace.map[1:19, 1:19]:
        with dace.tasklet:
            a0 << A[i - 1, j - 1]
            a1 << A[i - 1, j]
            a2 << A[i - 1, j + 1]
            a3 << A[i, j - 1]
            a4 << A[i, j]
            a5 << A[i, j + 1]
            a6 << A[i + 1, j - 1]
            a7 << A[i + 1, j]
            a8 << A[i + 1, j + 1]
            b >> tmp[i - 1, j - 1]

            b = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) / 9.0

    for i, j in dace.map[1:17, 1:17]:
        with dace.tasklet:
            a0 << tmp[j - 1, i - 1]
            a1 << tmp[j, i - 1]
            a2 << tmp[j + 1, i - 1]
            a3 << tmp[j - 1, i]
            a4 << tmp[j, i]
            a5 << tmp[j + 1, i]
            a6 << tmp[j - 1, i + 1]
            a7 << tmp[j, i + 1]
            a8 << tmp[j + 1, i + 1]
            b >> B[i - 1, j - 1]

            b = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) / 9.0


@dace.program
def fusion_tree(A: dace.float64[10, 20], B: dace.float64[10, 20], C: dace.float64[10, 20]):
    tmp = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            b >> tmp[i, j]

            b = a * a

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << tmp[i, j]
            b >> B[i, j]

            b = a + 2

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << tmp[i, j]
            c >> C[i, j]

            c = a + 4


def test_memlet_equation():
    i = dace.symbolic.symbol('i')
    j = dace.symbolic.symbol('j')
    write_params = [i, j]
    write_accesses = ((i, i, 1), (j - 1, j - 1, 1))

    k = dace.symbolic.symbol('k')
    l = dace.symbolic.symbol('l')
    read_params = [k, l]
    read_accesses = ((k, k, 1), (l + 2, l + 2, 1))

    sol = OTFMapFusion.solve(write_params, write_accesses, read_params, read_accesses)
    assert sol[i] == k
    assert sol[j] == l + 3


def test_memlet_equation_transposed():
    i = dace.symbolic.symbol('i')
    j = dace.symbolic.symbol('j')
    write_params = [i, j]
    write_accesses = ((i, i, 1), (j - 1, j - 1, 1))

    read_params = [j, i]
    read_accesses = ((j + 2, j + 2, 1), (i, i, 1))

    sol = OTFMapFusion.solve(write_params, write_accesses, read_params, read_accesses)
    assert sol[i] == j + 2
    assert sol[j] == i + 1


def test_memlet_equation_constant_read():
    i = dace.symbolic.symbol('i')
    write_params = [i]
    write_accesses = ((i, i, 1), )

    read_params = [i]
    read_accesses = ((0, 0, 1), )

    sol = OTFMapFusion.solve(write_params, write_accesses, read_params, read_accesses)
    assert sol[i] == 0


def test_memlet_equation_constant_read_and_write_fail():
    write_params = []
    write_accesses = ((0, 0, 1), )

    read_params = []
    read_accesses = ((1, 1, 1), )

    sol = OTFMapFusion.solve(write_params, write_accesses, read_params, read_accesses)
    assert sol is None


def test_memlet_equation_constant_write():
    i = dace.symbolic.symbol('i')
    write_params = [i]
    write_accesses = ((0, 0, 1), )

    read_params = [i]
    read_accesses = ((i, i, 1), )

    sol = OTFMapFusion.solve(write_params, write_accesses, read_params, read_accesses)
    assert sol is None


def test_fusion_chain():
    sdfg = fusion_chain.to_sdfg()
    sdfg.simplify()

    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    # Validate output

    A = np.random.rand(10, 20).astype(np.float64)
    B = np.zeros_like(A)

    target = A * A + 2
    sdfg(A=A, B=B)

    diff = np.linalg.norm(target - B)
    assert diff <= 1e-12


def test_fusion_chain_renamed():
    sdfg = fusion_chain_renamed.to_sdfg()
    sdfg.simplify()

    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    # Validate output

    A = np.random.rand(10, 20).astype(np.float64)
    B = np.zeros_like(A)

    target = A * A + 2
    sdfg(A=A, B=B)

    diff = np.linalg.norm(target - B)
    assert diff <= 1e-12


def test_fusion_flip():
    sdfg = fusion_flip.to_sdfg()
    sdfg.simplify()

    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    # Validate output
    A = np.random.rand(10, 20).astype(np.float64)
    target = np.flip(A * A + 2)

    B = np.zeros_like(A)
    sdfg(A=A, B=B, M=20, N=10)

    diff = np.linalg.norm(target - B)
    assert diff <= 1e-12


def test_fusion_transposed():
    sdfg = fusion_transposed.to_sdfg()
    sdfg.simplify()

    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    # Validate output

    A = np.random.rand(10, 20).astype(np.float64)
    B = np.zeros((20, 10), dtype=np.float64)

    target = A * A
    target = target.T + 2
    sdfg(A=A, B=B)

    diff = np.linalg.norm(target - B)
    assert diff <= 1e-12


def test_fusion_convolve():
    sdfg = fusion_convolve.to_sdfg()
    sdfg.simplify()

    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion, validate_all=False)
    assert count_maps(sdfg) == 1

    # Validate output

    A = np.random.rand(20, 20).astype(np.float64)
    B = np.zeros((16, 16), dtype=np.float64)

    mask = np.ones((3, 3), dtype=A.dtype) / 9.0
    tmp = np.zeros((18, 18), dtype=A.dtype)
    target = np.zeros((16, 16), dtype=A.dtype)

    tmp = convolve2d(A, mask, mode="valid")
    target = convolve2d(tmp, mask, mode="valid")

    sdfg(A=A, B=B)

    diff = np.linalg.norm(target - B)
    assert diff <= 1e-12


def test_fusion_convolve_overlap():
    # Overlapping accesses should be connected correctly
    sdfg = fusion_convolve_overlap.to_sdfg()
    sdfg.simplify()

    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion, validate_all=False)
    assert count_maps(sdfg) == 1

    # Validate SDFG
    sdfg.validate()


def test_fusion_convolve_transposed():
    sdfg = fusion_convolve_transposed.to_sdfg()
    sdfg.simplify()

    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion, validate_all=False)
    assert count_maps(sdfg) == 1

    # Validate output

    A = np.random.rand(20, 20).astype(np.float64)
    B = np.zeros((16, 16), dtype=np.float64)

    mask = np.ones((3, 3), dtype=A.dtype) / 9.0
    tmp = np.zeros((18, 18), dtype=A.dtype)
    target = np.zeros((16, 16), dtype=A.dtype)

    tmp = convolve2d(A, mask, mode="valid")
    target = convolve2d(tmp.T, mask, mode="valid")

    sdfg(A=A, B=B)

    diff = np.linalg.norm(target - B)
    assert diff <= 1e-12


def test_fusion_tree():
    sdfg = fusion_tree.to_sdfg()
    sdfg.simplify()

    assert count_maps(sdfg) == 3

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 3

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 2

    # Validate output

    A = np.random.rand(10, 20).astype(np.float64)
    B = np.zeros_like(A)
    C = np.zeros_like(A)

    b_target = A * A + 2
    c_target = A * A + 4
    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(b_target - B)
    assert diff <= 1e-12

    diff = np.linalg.norm(c_target - C)
    assert diff <= 1e-12


if __name__ == '__main__':
    test_fusion_chain()
    test_fusion_chain_renamed()
    test_fusion_flip()
    test_fusion_transposed()
    test_fusion_convolve()
    test_fusion_convolve_overlap()
    test_fusion_convolve_transposed()
    test_fusion_tree()
