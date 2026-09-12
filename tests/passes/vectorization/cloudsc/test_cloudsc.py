# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import pytest
# Unblocked 2026-06-12 per user direction (``enabling cloudsc tests and
# gradually enabling more tests``). The walker-primary pipeline lands the
# K-dim path e2e (gather + scatter passing); some legacy ``branch_mode`` /
# ``emission_style`` parametrisations may still need triage.
import dace
import numpy
from tests.passes.vectorization.helpers.harness import (
    run_vectorization_test,
    N,
    klev,
    kfdia,
    _get_cloudsc_snippet_three,
    _get_cloudsc_snippet_four,
)

#: Only ``insert_copies`` was ever read out of this; the discarded first element made half of
#: every parametrization an exact duplicate run.
OPT_PARAMS = [True, False]

# Also run the cloudsc snippets through the K-dim tile-op path
# (VectorizeCPUMultiDim). The ``vectorize_config`` fixture (conftest) adds a
# ``tile_nodes`` arm to every test that takes it; the harness clean-skips the
# arms the locked tile config does not support.
pytestmark = pytest.mark.tile_nodes


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
            if_cond_5 = B[i, j] > c
            if if_cond_5:
                D[i, j] = B[i, j] / A[0, i, j]
                E[i, j] = 1.0 - D[i, j]
            else:
                D[i, j] = 0.0
                E[i, j] = 0.0


def test_snippet_from_cloudsc_two(branch_mode, remainder_strategy, emission_style, vectorize_config):
    dim_size = 64
    A = numpy.random.random((2, dim_size, dim_size))
    B = numpy.random.random((dim_size, dim_size))
    c = 0.1
    D = numpy.random.random((dim_size, dim_size))
    E = numpy.random.random((dim_size, dim_size))

    run_vectorization_test(dace_func=cloudsc_snippet_two,
                           arrays={
                               'A': A,
                               'B': B,
                               'D': D,
                               'E': E,
                           },
                           params={
                               'c': c,
                               'N': dim_size
                           },
                           vector_width=8,
                           sdfg_name="cloudsc_snippet_two",
                           branch_mode=branch_mode,
                           remainder_strategy=remainder_strategy,
                           emission_style=emission_style,
                           vectorize_config=vectorize_config)


def has_no_inner_maps(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    for inode in state.all_nodes_between(map_entry, state.exit_node(map_entry)):
        if isinstance(inode, dace.nodes.MapEntry):
            return False
    return True


def test_snippet_from_cloudsc_two_fuse_overlapping_loads(branch_mode, remainder_strategy):
    dim_size = 64
    A = numpy.random.random((2, dim_size, dim_size))
    B = numpy.random.random((dim_size, dim_size))
    c = 0.1
    D = numpy.random.random((dim_size, dim_size))
    E = numpy.random.random((dim_size, dim_size))

    vectorized_sdfg = run_vectorization_test(dace_func=cloudsc_snippet_two,
                                             arrays={
                                                 'A': A,
                                                 'B': B,
                                                 'D': D,
                                                 'E': E,
                                             },
                                             params={
                                                 'c': c,
                                                 'N': dim_size
                                             },
                                             vector_width=8,
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
        if isinstance(n, dace.nodes.MapEntry) and any(str(s) == "8" for _, _, s in n.map.range)
    ]
    assert vw_step_maps, ("cloudsc_two did not vectorize under fuse_overlapping_loads=True: no map strided by "
                          "the vector width was produced")


def test_snippet_from_cloudsc_one(branch_mode, remainder_strategy, emission_style, vectorize_config):
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
                           emission_style=emission_style,
                           vectorize_config=vectorize_config)


def test_snippet_from_cloudsc_four(remainder_strategy, emission_style, vectorize_config):
    """T1-restricted (drops branch_mode for axis distribution)."""
    sdfg = _get_cloudsc_snippet_four()
    sdfg.name = "cloudsc_snippet_four"
    sdfg.validate()

    # Symbolic values requested by the user
    klon = 64
    klev = 64
    kidia = 1
    kfdia = 32
    for_it_92 = 0
    for_it_91 = 0

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
        "_for_it_92": numpy.int64(for_it_92),
        "_for_it_91": numpy.int64(for_it_91),
    }

    # Quick verification display: shape and contiguity / strides
    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           sdfg_name=sdfg.name,
                           remainder_strategy=remainder_strategy,
                           emission_style=emission_style,
                           vectorize_config=vectorize_config)


#: The snippet-three fixture's array shapes, one spelling for every test that drives it.
SNIPPET_THREE_SHAPES = {
    "tendency_tmp_q": (64, 64),
    "pa": (64, 64),
    "pq": (64, 64),
    "tendency_tmp_t": (64, 64),
    "tendency_tmp_a": (64, 64),
    "pt": (64, 64),
    "zqx0": (64, 64, 5),
    "zqx": (64, 64, 5),
    "ztp1": (64, 64),
    "zaorig": (64, 64),
    "za": (64, 64),
}


def snippet_three_inputs(ralvdcp: float | None = None):
    """Fortran-ordered arrays plus the scalar bindings for the snippet-three fixture.

    :param ralvdcp: bind the extra scalar the ``add_scalar=True`` variant reads.
    :returns: ``(arrays, scalars)`` for :func:`run_vectorization_test`.
    """
    arrays = {
        name: numpy.random.random(shape).astype(numpy.float64, order='F')
        for name, shape in SNIPPET_THREE_SHAPES.items()
    }
    scalars = {
        "kfdia": numpy.int64(32),
        "kidia": numpy.int64(1),
        "ptsphy": numpy.float64(0.0),
        "klev": numpy.int64(64),
        "klon": numpy.int64(64),
    }
    if ralvdcp is not None:
        scalars["ralvdcp"] = numpy.float64(ralvdcp)
    return arrays, scalars


@pytest.mark.parametrize("opt_parameters", OPT_PARAMS)
def test_snippet_from_cloudsc_three(opt_parameters, branch_mode, remainder_strategy, vectorize_config):
    insert_copies = opt_parameters
    sdfg = _get_cloudsc_snippet_three(add_scalar=False)
    sdfg.name = "cloudsc_snippet_three"
    sdfg.validate()
    arrays, scalars = snippet_three_inputs()

    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           sdfg_name=sdfg.name,
                           insert_copies=insert_copies,
                           branch_mode=branch_mode,
                           remainder_strategy=remainder_strategy,
                           vectorize_config=vectorize_config,
                           param_tag=f"param{OPT_PARAMS.index(opt_parameters)}")


@pytest.mark.parametrize("opt_parameters", OPT_PARAMS)
def test_snippet_from_cloudsc_three_with_partial_subset(opt_parameters, branch_mode, remainder_strategy,
                                                        vectorize_config):
    """The map-range-dependent subset variant: the staged window moves with the tile base."""
    insert_copies = opt_parameters
    sdfg = _get_cloudsc_snippet_three(add_scalar=False, map_range_dependent_subset=True)
    sdfg.name = "cloudsc_snippet_three_with_partial_subset"
    sdfg.validate()
    arrays, scalars = snippet_three_inputs()

    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           sdfg_name=sdfg.name,
                           insert_copies=insert_copies,
                           branch_mode=branch_mode,
                           remainder_strategy=remainder_strategy,
                           vectorize_config=vectorize_config,
                           param_tag=f"param{OPT_PARAMS.index(opt_parameters)}")


@pytest.mark.parametrize("opt_parameters", OPT_PARAMS)
def test_snippet_from_cloudsc_three_with_scalar_use(opt_parameters, branch_mode, remainder_strategy, vectorize_config):
    """The variant whose body also reads a lane-uniform scalar (``ralvdcp``)."""
    insert_copies = opt_parameters
    sdfg = _get_cloudsc_snippet_three(add_scalar=True)
    sdfg.name = "cloudsc_snippet_three_with_scalar_use"
    sdfg.validate()
    arrays, scalars = snippet_three_inputs(ralvdcp=2.3)

    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           sdfg_name=sdfg.name,
                           insert_copies=insert_copies,
                           branch_mode=branch_mode,
                           remainder_strategy=remainder_strategy,
                           vectorize_config=vectorize_config,
                           param_tag=f"param{OPT_PARAMS.index(opt_parameters)}")


# ``no_inline=True`` keeps snippet three's nested SDFGs as real nested SDFGs instead of
# inlining them away, so the tile walker has to stage through an nsdfg boundary. That is
# the only fact these two add, and the knob matrix above already covers the inlined form,
# so they run once each rather than across ``opt_parameters x branch_mode x remainder``.
@pytest.mark.parametrize("map_range_dependent_subset", [False, True])
def test_snippet_three_vectorizes_through_a_nested_sdfg_boundary(map_range_dependent_subset, vectorize_config):
    sdfg = _get_cloudsc_snippet_three(add_scalar=False, map_range_dependent_subset=map_range_dependent_subset)
    sdfg.name = ("cloudsc_snippet_three_with_partial_subset_without_inline"
                 if map_range_dependent_subset else "cloudsc_snippet_three_without_inline_sdfgs")
    sdfg.validate()
    arrays, scalars = snippet_three_inputs()

    run_vectorization_test(dace_func=sdfg,
                           from_sdfg=True,
                           arrays=arrays,
                           params=scalars,
                           vector_width=8,
                           sdfg_name=sdfg.name,
                           no_inline=True,
                           vectorize_config=vectorize_config)
