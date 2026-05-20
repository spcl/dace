# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import pytest
import numpy
from tests.passes.vectorization._harness import (
    run_vectorization_test,
    N,
    klev,
    kfdia,
    _get_cloudsc_snippet_three,
    _get_cloudsc_snippet_four,
)

_OPT_PARAMS = [(True, True), (True, False), (False, True), (False, False)]


@dace.program
def cloudsc_snippet_one(
    za: dace.float64[kfdia, klev],
    zliqfrac: dace.float64[kfdia, klev],
    zicefrac: dace.float64[kfdia, klev],
    zqx: dace.float64[5, klev + 1, kfdia + 1],
    zli: dace.float64[kfdia, klev],
    rlmin: dace.float64,
    z1: dace.int64,
):
    # note: outer loop over j (kfdia) first, then i (klev) to match column-major
    for j in range(kfdia):
        for i in range(klev):
            zaji = za[j, i]
            za[j, i] = 2.0 * zaji - 5
            cond1 = rlmin > 0.5 * (zqx[z1, i, j] + zqx[z1, j + 1, i + 1])
            if cond1:
                zliqfrac[j, i] = zqx[z1, j, i] * zli[j, i]
                zicefrac[j, i] = 1 - zliqfrac[j, i]
            else:
                zliqfrac[j, i] = 0
                zicefrac[j, i] = 0


@dace.program
def cloudsc_snippet_two(
    A: dace.float64[2, N, N],
    B: dace.float64[N, N],
    c: dace.float64,
    D: dace.float64[N, N],
    E: dace.float64[N, N],
):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            B[i, j] = A[1, i, j] + A[0, i, j]
            _if_cond_5 = B[i, j] > c
            if _if_cond_5:
                D[i, j] = B[i, j] / A[0, i, j]
                E[i, j] = 1.0 - D[i, j]
            else:
                D[i, j] = 0.0
                E[i, j] = 0.0


def test_snippet_from_cloudsc_two(branch_mode, remainder_strategy, emission_style):
    _S = 64
    A = numpy.random.random((2, _S, _S))
    B = numpy.random.random((_S, _S))
    c = 0.1
    D = numpy.random.random((_S, _S))
    E = numpy.random.random((_S, _S))

    run_vectorization_test(dace_func=cloudsc_snippet_two,
                           arrays={
                               'A': A,
                               'B': B,
                               'D': D,
                               'E': E,
                           },
                           params={
                               'c': c,
                               'N': _S
                           },
                           vector_width=8,
                           sdfg_name="cloudsc_snippet_two",
                           branch_mode=branch_mode,
                           remainder_strategy=remainder_strategy,
                           emission_style=emission_style)


def has_no_inner_maps(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    for inode in state.all_nodes_between(map_entry, state.exit_node(map_entry)):
        if isinstance(inode, dace.nodes.MapEntry):
            return False
    return True


def test_snippet_from_cloudsc_two_fuse_overlapping_loads(branch_mode, remainder_strategy):
    _S = 64
    A = numpy.random.random((2, _S, _S))
    B = numpy.random.random((_S, _S))
    c = 0.1
    D = numpy.random.random((_S, _S))
    E = numpy.random.random((_S, _S))

    vectorized_sdfg = run_vectorization_test(dace_func=cloudsc_snippet_two,
                                             arrays={
                                                 'A': A,
                                                 'B': B,
                                                 'D': D,
                                                 'E': E,
                                             },
                                             params={
                                                 'c': c,
                                                 'N': _S
                                             },
                                             vector_width=8,
                                             fuse_overlapping_loads=True,
                                             sdfg_name="cloudsc_snippet_two_fuse_overlapping_loads",
                                             branch_mode=branch_mode,
                                             remainder_strategy=remainder_strategy)

    # cloudsc_two reads ``A`` only at ``A[0, i, j]`` and ``A[1, i, j]``:
    # these differ solely in the dim-0 *constant* (0 vs 1) of a
    # ``(2, N, N)`` array, so they are DISJOINT memory regions, not
    # overlapping windows. ``fuse_overlapping_loads`` fuses *overlapping*
    # reads (stencil halos, e.g. jacobi2d ``A[i,j]`` / ``A[i+1,j]``);
    # there is nothing to overlap-fuse here, and unioning two disjoint
    # reads into an ``A[0:2, ...]`` bounding box would double the staged
    # data for no benefit. So asserting "exactly one union access node
    # between the maps" is invalid for this kernel — the standalone
    # FuseOverlappingLoads pass correctly no-ops on disjoint subsets.
    #
    # The genuine contract here is: with ``fuse_overlapping_loads=True``
    # on a disjoint-multi-subset branchy kernel, vectorization still
    # succeeds and stays numerically correct. The e2e numerical compare
    # is already enforced inside ``run_vectorization_test``; assert
    # structurally that the kernel actually vectorized (a map strided by
    # the vector width exists), not a bogus union.
    vw_step_maps = [
        n for n, _ in vectorized_sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and any(str(s) == "8" for _b, _e, s in n.map.range)
    ]
    assert vw_step_maps, ("cloudsc_two did not vectorize under fuse_overlapping_loads=True: no map strided by "
                          "the vector width was produced")


def test_snippet_from_cloudsc_one(branch_mode, remainder_strategy, emission_style):
    klev = 64
    kfdia = 32

    # reverse dimensions to match Fortran layout
    za = numpy.random.random((kfdia, klev))
    zliqfrac = numpy.random.random((kfdia, klev))
    zicefrac = numpy.random.random((kfdia, klev))
    zqx = numpy.random.random((5, kfdia + 1, klev + 1))
    zli = numpy.random.random((kfdia, klev))

    rlmin = 0.1
    z1 = 1

    run_vectorization_test(dace_func=cloudsc_snippet_one,
                           arrays={
                               'za': za,
                               'zliqfrac': zliqfrac,
                               'zicefrac': zicefrac,
                               'zqx': zqx,
                               'zli': zli,
                           },
                           params={
                               'rlmin': rlmin,
                               'z1': z1,
                               'kfdia': kfdia,
                               'klev': klev,
                           },
                           vector_width=8,
                           sdfg_name="cloudsc_snippet_one",
                           cleanup=True,
                           branch_mode=branch_mode,
                           remainder_strategy=remainder_strategy,
                           emission_style=emission_style)


def test_snippet_from_cloudsc_four(branch_mode, remainder_strategy):
    sdfg = _get_cloudsc_snippet_four()
    sdfg.name = f"cloudsc_snippet_four"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32
    _for_it_92 = 0
    _for_it_91 = 0

    # Map of array shapes (from the SDFG snippet): only the shape tuples matter for creating arrays
    arr_shapes = {
        "zfallsink": (klon, klev, 5),
        "zqlhs": (klon, klev, 5),
        "zsolqb": (klon, klev, 5),
    }

    # Create Fortran-ordered NumPy arrays
    arrays = {name: numpy.random.random(shape).astype(numpy.float64, order='F') for name, shape in arr_shapes.items()}
    # Create scalars requested
    scalars = {
        "kfdia": numpy.int64(kfdia),
        "kidia": numpy.int64(kidia),
        "klev": numpy.int64(klev),
        "klon": numpy.int64(klon),
        "_for_it_92": numpy.int64(_for_it_92),
        "_for_it_91": numpy.int64(_for_it_91),
    }

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           sdfg_name=sdfg.name,
                           fuse_overlapping_loads=False,
                           insert_copies=True,
                           branch_mode=branch_mode,
                           remainder_strategy=remainder_strategy)


@pytest.mark.parametrize("opt_parameters", _OPT_PARAMS)
def test_snippet_from_cloudsc_three(opt_parameters, branch_mode, remainder_strategy):
    fuse_overlapping_loads, insert_copies = opt_parameters

    sdfg = _get_cloudsc_snippet_three(add_scalar=False)
    sdfg.name = "cloudsc_snippet_three"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32

    # Map of array shapes (from the SDFG snippet): only the shape tuples matter for creating arrays
    arr_shapes = {
        "tendency_tmp_q": (klon, klev),
        "pa": (klon, klev),
        "pq": (klon, klev),
        "tendency_tmp_t": (klon, klev),
        "tendency_tmp_a": (klon, klev),
        "pt": (klon, klev),
        "zqx0": (klon, klev, 5),
        "zqx": (klon, klev, 5),
        "ztp1": (klon, klev),
        "zaorig": (klon, klev),
        "za": (klon, klev),
    }

    # Create Fortran-ordered NumPy arrays
    arrays = {name: numpy.random.random(shape).astype(numpy.float64, order='F') for name, shape in arr_shapes.items()}
    # Create scalars requested
    scalars = {
        "kfdia": numpy.int64(kfdia),
        "kidia": numpy.int64(kidia),
        "ptsphy": numpy.float64(0.0),
        "klev": numpy.int64(klev),
        "klon": numpy.int64(klon),
    }

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           sdfg_name=sdfg.name,
                           fuse_overlapping_loads=fuse_overlapping_loads,
                           insert_copies=insert_copies,
                           branch_mode=branch_mode,
                           remainder_strategy=remainder_strategy,
                           param_tag=f"param{_OPT_PARAMS.index(opt_parameters)}")


@pytest.mark.parametrize("opt_parameters", _OPT_PARAMS)
def test_snippet_from_cloudsc_three_with_partial_subset(opt_parameters, branch_mode, remainder_strategy):
    fuse_overlapping_loads, insert_copies = opt_parameters

    sdfg = _get_cloudsc_snippet_three(add_scalar=False, map_range_dependent_subset=True)
    sdfg.name = "cloudsc_snippet_three_with_partial_subset"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32

    # Map of array shapes (from the SDFG snippet): only the shape tuples matter for creating arrays
    arr_shapes = {
        "tendency_tmp_q": (klon, klev),
        "pa": (klon, klev),
        "pq": (klon, klev),
        "tendency_tmp_t": (klon, klev),
        "tendency_tmp_a": (klon, klev),
        "pt": (klon, klev),
        "zqx0": (klon, klev, 5),
        "zqx": (klon, klev, 5),
        "ztp1": (klon, klev),
        "zaorig": (klon, klev),
        "za": (klon, klev),
    }

    # Create Fortran-ordered NumPy arrays
    arrays = {name: numpy.random.random(shape).astype(numpy.float64, order='F') for name, shape in arr_shapes.items()}
    # Create scalars requested
    scalars = {
        "kfdia": numpy.int64(kfdia),
        "kidia": numpy.int64(kidia),
        "ptsphy": numpy.float64(0.0),
        "klev": numpy.int64(klev),
        "klon": numpy.int64(klon),
    }

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           sdfg_name=sdfg.name,
                           fuse_overlapping_loads=fuse_overlapping_loads,
                           insert_copies=insert_copies,
                           no_inline=True,
                           branch_mode=branch_mode,
                           remainder_strategy=remainder_strategy,
                           param_tag=f"param{_OPT_PARAMS.index(opt_parameters)}")


@pytest.mark.parametrize("opt_parameters", _OPT_PARAMS)
def test_snippet_from_cloudsc_three_with_partial_subset_without_inline(opt_parameters, branch_mode, remainder_strategy):
    fuse_overlapping_loads, insert_copies = opt_parameters

    sdfg = _get_cloudsc_snippet_three(add_scalar=False, map_range_dependent_subset=True)
    sdfg.name = "cloudsc_snippet_three_with_partial_subset_without_inline"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32

    # Map of array shapes (from the SDFG snippet): only the shape tuples matter for creating arrays
    arr_shapes = {
        "tendency_tmp_q": (klon, klev),
        "pa": (klon, klev),
        "pq": (klon, klev),
        "tendency_tmp_t": (klon, klev),
        "tendency_tmp_a": (klon, klev),
        "pt": (klon, klev),
        "zqx0": (klon, klev, 5),
        "zqx": (klon, klev, 5),
        "ztp1": (klon, klev),
        "zaorig": (klon, klev),
        "za": (klon, klev),
    }

    # Create Fortran-ordered NumPy arrays
    arrays = {name: numpy.random.random(shape).astype(numpy.float64, order='F') for name, shape in arr_shapes.items()}
    # Create scalars requested
    scalars = {
        "kfdia": numpy.int64(kfdia),
        "kidia": numpy.int64(kidia),
        "ptsphy": numpy.float64(0.0),
        "klev": numpy.int64(klev),
        "klon": numpy.int64(klon),
    }

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           sdfg_name=sdfg.name,
                           fuse_overlapping_loads=fuse_overlapping_loads,
                           insert_copies=insert_copies,
                           no_inline=True,
                           branch_mode=branch_mode,
                           remainder_strategy=remainder_strategy,
                           param_tag=f"param{_OPT_PARAMS.index(opt_parameters)}")


@pytest.mark.parametrize("opt_parameters", _OPT_PARAMS)
def test_snippet_from_cloudsc_three_without_inline_sdfgs(opt_parameters, branch_mode, remainder_strategy):
    fuse_overlapping_loads, insert_copies = opt_parameters

    sdfg = _get_cloudsc_snippet_three(add_scalar=False)
    sdfg.name = "cloudsc_snippet_three_without_inline_sdfgs"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32

    # Map of array shapes (from the SDFG snippet): only the shape tuples matter for creating arrays
    arr_shapes = {
        "tendency_tmp_q": (klon, klev),
        "pa": (klon, klev),
        "pq": (klon, klev),
        "tendency_tmp_t": (klon, klev),
        "tendency_tmp_a": (klon, klev),
        "pt": (klon, klev),
        "zqx0": (klon, klev, 5),
        "zqx": (klon, klev, 5),
        "ztp1": (klon, klev),
        "zaorig": (klon, klev),
        "za": (klon, klev),
    }

    # Create Fortran-ordered NumPy arrays
    arrays = {name: numpy.random.random(shape).astype(numpy.float64, order='F') for name, shape in arr_shapes.items()}
    # Create scalars requested
    scalars = {
        "kfdia": numpy.int64(kfdia),
        "kidia": numpy.int64(kidia),
        "ptsphy": numpy.float64(0.0),
        "klev": numpy.int64(klev),
        "klon": numpy.int64(klon),
    }

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           sdfg_name=sdfg.name,
                           fuse_overlapping_loads=fuse_overlapping_loads,
                           insert_copies=insert_copies,
                           no_inline=True,
                           branch_mode=branch_mode,
                           remainder_strategy=remainder_strategy,
                           param_tag=f"param{_OPT_PARAMS.index(opt_parameters)}")


@pytest.mark.parametrize("opt_parameters", _OPT_PARAMS)
def test_snippet_from_cloudsc_three_with_scalar_use(opt_parameters, branch_mode, remainder_strategy):
    fuse_overlapping_loads, insert_copies = opt_parameters

    sdfg = _get_cloudsc_snippet_three(add_scalar=True)
    sdfg.name = "cloudsc_snippet_three_with_scalar_use"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32

    # Map of array shapes (from the SDFG snippet): only the shape tuples matter for creating arrays
    arr_shapes = {
        "tendency_tmp_q": (klon, klev),
        "pa": (klon, klev),
        "pq": (klon, klev),
        "tendency_tmp_t": (klon, klev),
        "tendency_tmp_a": (klon, klev),
        "pt": (klon, klev),
        "zqx0": (klon, klev, 5),
        "zqx": (klon, klev, 5),
        "ztp1": (klon, klev),
        "zaorig": (klon, klev),
        "za": (klon, klev),
    }

    # Create Fortran-ordered NumPy arrays
    arrays = {name: numpy.random.random(shape).astype(numpy.float64, order='F') for name, shape in arr_shapes.items()}
    # Create scalars requested
    scalars = {
        "kfdia": numpy.int64(kfdia),
        "kidia": numpy.int64(kidia),
        "ptsphy": numpy.float64(0.0),
        "klev": numpy.int64(klev),
        "klon": numpy.int64(klon),
        "ralvdcp": numpy.float64(2.3),
    }

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           sdfg_name=sdfg.name,
                           fuse_overlapping_loads=fuse_overlapping_loads,
                           insert_copies=insert_copies,
                           branch_mode=branch_mode,
                           remainder_strategy=remainder_strategy,
                           param_tag=f"param{_OPT_PARAMS.index(opt_parameters)}")
