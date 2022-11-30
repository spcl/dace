# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
import math

from scipy.signal import convolve2d

from dace.transformation.dataflow import OTFMapFusion, MapExpansion, MapCollapse, RedundantArray

N = dace.symbol("N")
M = dace.symbol("M")


def count_maps(sdfg):
    maps = 0
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                maps += 1

    return maps


def test_solve():
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


def test_solve_permute():
    i = dace.symbolic.symbol('i')
    j = dace.symbolic.symbol('j')
    write_params = [i, j]
    write_accesses = ((i, i, 1), (j - 1, j - 1, 1))

    read_params = [j, i]
    read_accesses = ((j + 2, j + 2, 1), (i, i, 1))

    sol = OTFMapFusion.solve(write_params, write_accesses, read_params, read_accesses)
    assert sol[i] == j + 2
    assert sol[j] == i + 1


def test_solve_constant():
    i = dace.symbolic.symbol('i')
    write_params = [i]
    write_accesses = ((i, i, 1), )

    read_params = [i]
    read_accesses = ((0, 0, 1), )

    sol = OTFMapFusion.solve(write_params, write_accesses, read_params, read_accesses)
    assert sol[i] == 0


def test_solve_constant2():
    write_params = []
    write_accesses = ((0, 0, 1), )

    read_params = []
    read_accesses = ((1, 1, 1), )

    sol = OTFMapFusion.solve(write_params, write_accesses, read_params, read_accesses)
    assert (0, 0) in sol and sol[(0, 0)] == (1, 1)
    assert len(sol) == 1


def test_solve_unsolvable():
    i = dace.symbolic.symbol('i')
    write_params = [i]
    write_accesses = ((0, 0, 1), )

    read_params = [i]
    read_accesses = ((i, i, 1), )

    sol = OTFMapFusion.solve(write_params, write_accesses, read_params, read_accesses)
    assert sol is None


@dace.program
def trivial_fusion(A: dace.float64[10, 20], B: dace.float64[10, 20]):
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


def test_trivial_fusion():
    sdfg = trivial_fusion.to_sdfg()
    sdfg.simplify()
    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    A = np.random.rand(10, 20).astype(np.float64)
    B = np.zeros_like(A)

    sdfg(A=A, B=B)

    ref = A * A + 2
    assert np.allclose(B, ref)


@dace.program
def trivial_fusion_rename(A: dace.float64[10, 20], B: dace.float64[10, 20]):
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


def test_trivial_fusion_rename():
    sdfg = trivial_fusion_rename.to_sdfg()
    sdfg.simplify()
    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    A = np.random.rand(10, 20).astype(np.float64)
    B = np.zeros_like(A)

    sdfg(A=A, B=B)

    ref = A * A + 2
    assert np.allclose(B, ref)


@dace.program
def trivial_fusion_flip(A: dace.float64[10, 20], B: dace.float64[10, 20]):
    tmp = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            b >> tmp[i, j]

            b = a * a

    for k, l in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << tmp[10 - k - 1, 20 - l - 1]
            b >> B[k, l]

            b = a + 2


def test_trivial_fusion_flip():
    sdfg = trivial_fusion_flip.to_sdfg()
    sdfg.simplify()
    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    A = np.random.rand(10, 20).astype(np.float64)
    B = np.zeros_like(A)

    sdfg(A=A, B=B)

    ref = np.flip(A * A + 2)
    assert np.allclose(B, ref)


@dace.program
def trivial_fusion_permute(A: dace.float64[10, 20], B: dace.float64[20, 10]):
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


def test_trivial_fusion_permute():
    sdfg = trivial_fusion_permute.to_sdfg()
    sdfg.simplify()
    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    A = np.random.rand(10, 20).astype(np.float64)
    B = np.zeros((20, 10), dtype=A.dtype)

    sdfg(A=A, B=B)

    ref = A * A + 2
    assert np.allclose(B, ref.T)


@dace.program
def trivial_fusion_not_remove_map(A: dace.float64[10, 20], B: dace.float64[10, 20], C: dace.float64[10, 20]):
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


def test_trivial_fusion_not_remove_map():
    sdfg = trivial_fusion_not_remove_map.to_sdfg()
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

    sdfg(A=A, B=B, C=C)

    b_ref = A * A + 2
    assert np.allclose(B, b_ref)

    c_ref = A * A + 4
    assert np.allclose(C, c_ref)


@dace.program
def trivial_fusion_nested_sdfg(A: dace.int64[128], B: dace.int64[128]):
    tmp = dace.define_local([128], dtype=dace.bool)
    for i in dace.map[0:128]:
        num = A[i]

        stop = int(math.ceil(math.sqrt(num)))
        is_prime = True

        j = 2
        while j <= stop:
            if num % j == 0:
                is_prime = False
                break

            j = j + 1

        tmp[i] = is_prime

    for j in dace.map[0:128]:
        num = A[j]
        is_prime = tmp[j]

        if not is_prime:
            B[j] = 0
        else:
            i = 1
            while i < 3:
                num = num + i
                i = i + 1

            B[j] = num


def test_trivial_fusion_nested_sdfg():
    sdfg = trivial_fusion_nested_sdfg.to_sdfg()
    sdfg.simplify()

    assert count_maps(sdfg) == 2

    nums = np.arange(2, 130, 1, dtype=np.int64)
    res = np.zeros((128, ), dtype=np.int64)
    sdfg(A=nums, B=res)

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    res_fused = np.zeros((128, ), dtype=np.int64)
    sdfg(A=nums, B=res_fused)
    assert (res == res_fused).all()


@dace.program
def undefined_subset(A: dace.float64[10], B: dace.float64[10]):
    tmp = dace.define_local([10], dtype=A.dtype)
    for i in dace.map[5:10]:
        with dace.tasklet:
            a << A[i]
            b >> tmp[i]

            b = a + 1

    for i in dace.map[5:10]:
        with dace.tasklet:
            a << tmp[i - 5]
            b >> B[i]

            b = a


def test_undefined_subset():
    sdfg = undefined_subset.to_sdfg()
    sdfg.simplify()
    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 2


@dace.program
def defined_subset(A: dace.float64[10], B: dace.float64[10]):
    tmp = dace.define_local([10], dtype=A.dtype)
    for i in dace.map[5:10]:
        with dace.tasklet:
            a << A[i]
            b >> tmp[i]

            b = a + 1

    for i in dace.map[0:5]:
        with dace.tasklet:
            a << tmp[i + 5]
            b >> B[i]

            b = a


def test_defined_subset():
    sdfg = defined_subset.to_sdfg()
    sdfg.simplify()
    assert count_maps(sdfg) == 2

    A = np.random.rand(10).astype(np.float64)
    B = np.zeros_like(A)
    sdfg(A=A, B=B)

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    B_ = np.zeros_like(A)
    sdfg(A=A, B=B_)

    assert np.allclose(B_, B)


@dace.program
def undefined_subset_step(A: dace.float64[10], B: dace.float64[10]):
    tmp = dace.define_local([10], dtype=A.dtype)
    for i in dace.map[0:10:2]:
        with dace.tasklet:
            a << A[i]
            b >> tmp[i]

            b = a + 1

    for i in dace.map[0:10]:
        with dace.tasklet:
            a << tmp[i]
            b >> B[i]

            b = a


def test_undefined_subset_step():
    sdfg = undefined_subset_step.to_sdfg()
    sdfg.simplify()
    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 2


@dace.program
def defined_subset_step(A: dace.float64[10], B: dace.float64[10]):
    tmp = dace.define_local([10], dtype=A.dtype)
    for i in dace.map[0:10:2]:
        with dace.tasklet:
            a << A[i]
            b >> tmp[i]

            b = a + 1

    for i in dace.map[0:10:2]:
        with dace.tasklet:
            a << tmp[i]
            b >> B[i]

            b = a


def test_defined_subset_step():
    sdfg = defined_subset_step.to_sdfg()
    sdfg.simplify()
    assert count_maps(sdfg) == 2

    A = np.random.rand(10).astype(np.float64)
    B = np.zeros_like(A)
    sdfg(A=A, B=B)

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    B_ = np.zeros_like(A)
    sdfg(A=A, B=B_)

    assert np.allclose(B_, B)


@dace.program
def recomputation_fusion(A: dace.float64[20, 20], B: dace.float64[16, 16]):
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


def test_recomputation_fusion():
    sdfg = recomputation_fusion.to_sdfg()
    sdfg.simplify()
    assert count_maps(sdfg) == 2

    A = np.random.rand(20, 20).astype(np.float64)
    B = np.zeros((16, 16), dtype=np.float64)

    sdfg(A=A, B=B)

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    B_ = np.zeros_like(B)
    sdfg(A=A, B=B_)

    mask = np.ones((3, 3), dtype=A.dtype) / 9.0
    tmp = convolve2d(A, mask, mode="valid")
    ref = convolve2d(tmp, mask, mode="valid")

    assert np.allclose(B, B_)
    assert np.allclose(B_, ref)


@dace.program
def local_storage_fusion(A: dace.float64[10, 10], B: dace.float64[10, 10]):
    tmp = dace.define_local([10, 10], dtype=A.dtype)
    for i in dace.map[0:10]:
        with dace.tasklet:
            a << A[i, :]
            b >> tmp[i, :]

            for j in range(10):
                b[j] = a[j] + 1

    for i in dace.map[0:10]:
        with dace.tasklet:
            a << tmp[i, :]
            b >> B[i, :]

            for j in range(10):
                b[j] = a[j] + 2


def test_local_storage_fusion():
    sdfg = local_storage_fusion.to_sdfg()
    sdfg.simplify()
    assert count_maps(sdfg) == 2

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 1

    A = np.random.rand(10, 10).astype(np.float64)
    B = np.zeros_like(A)
    sdfg(A=A, B=B)

    ref = A + 1 + 2
    assert np.allclose(B, ref)


@dace.program
def local_storage_fusion_nested_map(A: dace.float64[10, 10], B: dace.float64[10, 10]):
    tmp = dace.define_local([10, 10], dtype=A.dtype)
    for i in dace.map[0:10]:
        for j in dace.map[0:10]:
            with dace.tasklet:
                a << A[i, j]
                b >> tmp[i, j]

                b = a + 1

    for i in dace.map[0:10]:
        for j in dace.map[0:10]:
            with dace.tasklet:
                a << tmp[i, j]
                b >> B[i, j]

                b = a + 2


def test_local_storage_fusion_nested_map():
    sdfg = local_storage_fusion_nested_map.to_sdfg()
    sdfg.simplify()
    assert count_maps(sdfg) == 4

    sdfg.apply_transformations(OTFMapFusion)
    assert count_maps(sdfg) == 3

    A = np.random.rand(10, 10).astype(np.float64)
    B = np.zeros_like(A)
    sdfg(A=A, B=B)

    ref = A + 1 + 2
    assert np.allclose(B, ref)


@dace.program
def matmuls(A: dace.float32[64, 32], B: dace.float32[32, 16], C: dace.float32[16, 64], o1: dace.float32[64, 16],
            o2: dace.float32[64, 64]):
    for i, j, k in dace.map[0:64, 0:16, 0:32]:
        with dace.tasklet:
            in_A << A[i, k]
            in_B << B[k, j]
            o >> o1(0, lambda x, y: x + y)[i, j]

            o = in_A * in_B

    for i, j, k in dace.map[0:64, 0:64, 0:16]:
        with dace.tasklet:
            in_t << o1[i, k]
            in_c << C[k, j]
            o >> o2(0, lambda x, y: x + y)[i, j]

            o = in_t * in_c


def test_matmuls():
    sdfg = matmuls.to_sdfg()
    sdfg.simplify()
    assert count_maps(sdfg) == 2

    A = np.random.random((64, 32)).astype(np.float32)
    B = np.random.random((32, 16)).astype(np.float32)
    C = np.random.random((16, 64)).astype(np.float32)

    ref = (A @ B) @ C

    # Fuse
    sdfg.arrays["o1"].transient = True

    first_map = None
    for node in sdfg.start_state.nodes():
        if not isinstance(node, dace.nodes.MapEntry):
            continue

        if "IN_A" in node.in_connectors:
            first_map = node
            break

    new_maps = MapExpansion.apply_to(sdfg=sdfg, map_entry=first_map)
    new_maps[0].map.schedule = dace.ScheduleType.Default
    new_maps[1].map.schedule = dace.ScheduleType.Default

    sdfg.apply_transformations_repeated(MapCollapse)

    applied = sdfg.apply_transformations(OTFMapFusion, options={"identity": 0})
    assert applied == 1

    o2 = np.zeros((64, 64), dtype=A.dtype)
    sdfg(A=A, B=B, C=C, o2=o2)

    assert np.allclose(o2, ref)


@dace.program
def hdiff(in_field: dace.float32[128 + 4, 128 + 4, 64], out_field: dace.float32[128, 128, 64],
          coeff: dace.float32[128, 128, 64]):
    lap_field = 4.0 * in_field[1:128 + 3, 1:128 + 3, :] - (
        in_field[2:128 + 4, 1:128 + 3, :] + in_field[0:128 + 2, 1:128 + 3, :] + in_field[1:128 + 3, 2:128 + 4, :] +
        in_field[1:128 + 3, 0:128 + 2, :])

    res1 = lap_field[1:, 1:128 + 1, :] - lap_field[:128 + 1, 1:128 + 1, :]
    flx_field = np.where(
        (res1 * (in_field[2:128 + 3, 2:128 + 2, :] - in_field[1:128 + 2, 2:128 + 2, :])) > 0,
        0,
        res1,
    )
    res2 = lap_field[1:128 + 1, 1:, :] - lap_field[1:128 + 1, :128 + 1, :]
    fly_field = np.where(
        (res2 * (in_field[2:128 + 2, 2:128 + 3, :] - in_field[2:128 + 2, 1:128 + 2, :])) > 0,
        0,
        res2,
    )
    out_field[:, :, :] = in_field[2:128 + 2, 2:128 + 2, :] - coeff[:, :, :] * (
        flx_field[1:, :, :] - flx_field[:-1, :, :] + fly_field[:, 1:, :] - fly_field[:, :-1, :])


def test_hdiff():
    sdfg = hdiff.to_sdfg()
    sdfg.simplify()
    assert count_maps(sdfg) == 20

    in_field = np.random.random((132, 132, 64)).astype(np.float32)
    coeff = np.random.random((128, 128, 64)).astype(np.float32)

    out_field = np.zeros_like(coeff)
    sdfg(in_field=in_field, coeff=coeff, out_field=out_field)

    sdfg.apply_transformations_repeated(OTFMapFusion)
    assert count_maps(sdfg) == 1
    
    out_field_ = np.zeros_like(coeff)
    sdfg(in_field=in_field, coeff=coeff, out_field=out_field_)

    # TODO: Numerically instable?
    # assert np.allclose(out_field, out_field_)


if __name__ == '__main__':
    # Solver
    test_solve()
    test_solve_permute()
    test_solve_constant()
    test_solve_constant2()
    test_solve_unsolvable()

    # Trivial fusion
    test_trivial_fusion()
    test_trivial_fusion_rename()
    test_trivial_fusion_flip()
    test_trivial_fusion_permute()
    test_trivial_fusion_not_remove_map()
    test_trivial_fusion_nested_sdfg()

    # Defined subsets
    test_undefined_subset()
    test_defined_subset()
    test_undefined_subset_step()
    test_defined_subset_step()

    # Recomputation
    test_recomputation_fusion()

    # Local buffer
    test_local_storage_fusion()
    test_local_storage_fusion_nested_map()

    # Applications
    test_matmuls()

    # TODO: Numerically instable?
    # test_hdiff()
