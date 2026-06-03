# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import pytest
import numpy
from tests.passes.vectorization.helpers.harness import (
    run_vectorization_test,
    assert_fused_nsdfg_structure,
    S,
)

# Core stencil kernels — also exercise the K-dim tile-op config.
pytestmark = pytest.mark.tile_nodes


@dace.program
def jacobi2d(A: dace.float64[S, S], B: dace.float64[S, S], tsteps: dace.int64):  #, N, tsteps):
    for t in range(tsteps):
        for i, j in dace.map[0:S - 2, 0:S - 2]:
            B[i + 1, j + 1] = 0.2 * (A[i + 1, j + 1] + A[i, j + 1] + A[i + 2, j + 1] + A[i + 1, j] + A[i + 1, j + 2])

        for i, j in dace.map[0:S - 2, 0:S - 2]:
            A[i + 1, j + 1] = 0.2 * (B[i + 1, j + 1] + B[i, j + 1] + B[i + 2, j + 1] + B[i + 1, j] + B[i + 1, j + 2])


def test_jacobi2d(emission_style, vectorize_config):
    _S = 66
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    run_vectorization_test(dace_func=jacobi2d,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={
                               'S': _S,
                               'tsteps': 5,
                           },
                           vector_width=8,
                           sdfg_name="jacobi2d",
                           emission_style=emission_style,
                           vectorize_config=vectorize_config)


def test_jacobi2d_with_filter_map():
    _S = 66
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    sdfg = jacobi2d.to_sdfg()

    run_vectorization_test(dace_func=jacobi2d,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={
                               'S': _S,
                               'tsteps': 5,
                           },
                           vector_width=8,
                           sdfg_name="jacobi2d_with_filter_map",
                           filter_map=1)


def test_jacobi2d_with_fuse_overlapping_loads():
    _S = 66
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    vectorized_sdfg: dace.SDFG = run_vectorization_test(dace_func=jacobi2d,
                                                        arrays={
                                                            'A': A,
                                                            'B': B
                                                        },
                                                        params={
                                                            'S': _S,
                                                            'tsteps': 5,
                                                        },
                                                        vector_width=8,
                                                        sdfg_name="jacobi2d_with_fuse_overlapping_loads",
                                                        fuse_overlapping_loads=True,
                                                        insert_copies=True)

    # Should have 1 access node between two maps
    inner_map_entries = {(n, g)
                         for n, g in vectorized_sdfg.all_nodes_recursive()
                         if isinstance(n, dace.nodes.MapEntry) and g.scope_dict()[n] is not None}
    for inner_map_entry, state in inner_map_entries:
        src_access_nodes = {
            ie.src
            for ie in state.in_edges(inner_map_entry) if isinstance(ie.src, dace.nodes.AccessNode)
        }

        src_src_access_nodes = set()
        for src_acc_node in src_access_nodes:
            src_src_access_nodes = src_src_access_nodes.union(
                {ie.src
                 for ie in state.in_edges(src_acc_node) if isinstance(ie.src, dace.nodes.AccessNode)})

        assert len(src_src_access_nodes
                   ) == 1, f"Excepted one access node got {len(src_src_access_nodes)}, ({src_src_access_nodes})"

    # Fused-window structural contract: in each body NSDFG the stencil
    # read of A (resp. B) must collapse to ONE widened union-window
    # buffer ``A_vec`` — not N per-subset copies — whose contiguous dim
    # exceeds the vector width (the ``+2`` halo of the 5-point stencil),
    # and that single buffer must be read at multiple distinct offsets
    # inside the NSDFG (the 5 stencil taps reading the one window).
    vw = 8
    found_window = False
    for nsdfg, _ in vectorized_sdfg.all_nodes_recursive():
        if not isinstance(nsdfg, dace.nodes.NestedSDFG):
            continue
        inner = nsdfg.sdfg
        for base in ("A", "B"):
            win = f"{base}_vec"
            if win not in inner.arrays:
                continue
            desc = inner.arrays[win]
            # The plain W-vector output buffer is also named ``<base>_vec``
            # (size == vw); the fused *input* window is the wider one.
            sizes = [int(str(s)) for s in desc.shape if str(s).isdigit()]
            if not sizes or max(sizes) <= vw:
                continue
            found_window = True
            # 1. collapse: no per-subset ``<base>_vec_<n>`` copies.
            copies = [a for a in inner.arrays if a.startswith(f"{win}_") and a[len(win) + 1:].isdigit()]
            assert not copies, f"{win} did not collapse: per-subset copies {copies} survived in {inner.label}"
            # 2. widened union window: a dim wider than the vector width
            #    (the stencil ``+2`` halo).
            assert max(sizes) > vw, (f"{win} in {inner.label} has shape {tuple(str(s) for s in desc.shape)}; "
                                     f"expected a dim > vector_width={vw} (the union ``+halo`` window)")
            # 3. read at multiple distinct offsets from the one window.
            read_subsets = {str(e.data.subset) for ist in inner.all_states() for e in ist.edges() if e.data.data == win}
            assert len(read_subsets) >= 2, (f"{win} in {inner.label} read at only {read_subsets}; expected the "
                                            f"fused stencil to read the one window at multiple offsets")
    assert found_window, "no fused union-window buffer (>vector_width) found in any body NSDFG"


@pytest.mark.parametrize("fuse_overlapping_loads", [False, True])
def test_jacobi2d_with_parameters(fuse_overlapping_loads, tile_emit_mode, emission_style, branch_mode,
                                  remainder_strategy, vectorize_config):
    """T1 stencil — full knob matrix.

    ``tile_emit_mode`` covers ``nest_map_bodies`` × ``insert_copies``;
    ``fuse_overlapping_loads`` is the stencil-specific fixture marker.
    The structural ``assert_fused_nsdfg_structure`` assertion fires only
    in the canonical config (``fuse=True``, copies via
    ``tile_emit_mode="nested_copies"`` on the default emission tile arm)
    — that combo guarantees the union-window buffer the assertion
    expects.
    """
    nest_map_bodies, insert_copies = tile_emit_mode
    _S = 66
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))

    vectorized_sdfg = run_vectorization_test(
        dace_func=jacobi2d,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'S': _S,
            'tsteps': 5,
        },
        vector_width=8,
        sdfg_name="jacobi2d_with_parameters",
        fuse_overlapping_loads=fuse_overlapping_loads,
        insert_copies=insert_copies,
        nest_map_bodies=nest_map_bodies,
        branch_mode=branch_mode,
        remainder_strategy=remainder_strategy,
        emission_style=emission_style,
        vectorize_config=vectorize_config,
    )
    if (fuse_overlapping_loads and insert_copies and emission_style == "default"
            and vectorize_config == "tile_nodes"):
        assert_fused_nsdfg_structure(vectorized_sdfg, ("A", ))


@dace.program
def jacobi1d(A: dace.float64[S], B: dace.float64[S], tsteps: dace.int64):
    for t in range(tsteps):
        for i in dace.map[0:S - 2]:
            B[i + 1] = 0.33333 * (A[i] + A[i + 1] + A[i + 2])

        for i in dace.map[0:S - 2]:
            A[i + 1] = 0.33333 * (B[i] + B[i + 1] + B[i + 2])


@pytest.mark.parametrize("fuse_overlapping_loads", [False, True])
def test_jacobi1d_with_parameters(fuse_overlapping_loads, tile_emit_mode, branch_mode, remainder_strategy,
                                  vectorize_config):
    """T1-restricted sibling of canonical ``test_jacobi2d_with_parameters``.

    Distributed-axis restriction: ``jacobi1d`` drops ``emission_style``;
    ``heat3d`` drops ``branch_mode``. Between the three stencil tests,
    every fixture axis is exercised by at least two — so we get full
    cross-coverage in the aggregate without paying the full combo cost
    on every kernel.
    """
    nest_map_bodies, insert_copies = tile_emit_mode
    _S = 130
    A = numpy.random.random((_S, ))
    B = numpy.random.random((_S, ))

    vectorized_sdfg = run_vectorization_test(
        dace_func=jacobi1d,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'S': _S,
            'tsteps': 5,
        },
        vector_width=8,
        sdfg_name="jacobi1d_with_parameters",
        fuse_overlapping_loads=fuse_overlapping_loads,
        insert_copies=insert_copies,
        nest_map_bodies=nest_map_bodies,
        branch_mode=branch_mode,
        remainder_strategy=remainder_strategy,
        vectorize_config=vectorize_config,
    )
    if fuse_overlapping_loads and insert_copies and vectorize_config == "tile_nodes":
        assert_fused_nsdfg_structure(vectorized_sdfg, ("A", ))


@dace.program
def heat3d(A: dace.float64[S, S, S], B: dace.float64[S, S, S], tsteps: dace.int64):
    for t in range(tsteps):
        for i, j, k in dace.map[0:S - 2, 0:S - 2, 0:S - 2]:
            B[i + 1, j + 1,
              k + 1] = (0.125 * (A[i + 2, j + 1, k + 1] - 2.0 * A[i + 1, j + 1, k + 1] + A[i, j + 1, k + 1]) + 0.125 *
                        (A[i + 1, j + 2, k + 1] - 2.0 * A[i + 1, j + 1, k + 1] + A[i + 1, j, k + 1]) + 0.125 *
                        (A[i + 1, j + 1, k + 2] - 2.0 * A[i + 1, j + 1, k + 1] + A[i + 1, j + 1, k]) +
                        A[i + 1, j + 1, k + 1])

        for i, j, k in dace.map[0:S - 2, 0:S - 2, 0:S - 2]:
            A[i + 1, j + 1,
              k + 1] = (0.125 * (B[i + 2, j + 1, k + 1] - 2.0 * B[i + 1, j + 1, k + 1] + B[i, j + 1, k + 1]) + 0.125 *
                        (B[i + 1, j + 2, k + 1] - 2.0 * B[i + 1, j + 1, k + 1] + B[i + 1, j, k + 1]) + 0.125 *
                        (B[i + 1, j + 1, k + 2] - 2.0 * B[i + 1, j + 1, k + 1] + B[i + 1, j + 1, k]) +
                        B[i + 1, j + 1, k + 1])


@pytest.mark.parametrize("fuse_overlapping_loads", [False, True])
def test_heat3d_with_parameters(fuse_overlapping_loads, tile_emit_mode, emission_style, remainder_strategy,
                                vectorize_config):
    """T1-restricted sibling of canonical ``test_jacobi2d_with_parameters``.

    Distributed-axis restriction: drops ``branch_mode`` (jacobi1d drops
    ``emission_style``). Between jacobi2d / jacobi1d / heat3d every
    fixture axis is exercised by at least two siblings.
    """
    nest_map_bodies, insert_copies = tile_emit_mode
    _S = 18
    A = numpy.random.random((_S, _S, _S))
    B = numpy.random.random((_S, _S, _S))

    vectorized_sdfg = run_vectorization_test(
        dace_func=heat3d,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'S': _S,
            'tsteps': 3,
        },
        vector_width=8,
        sdfg_name="heat3d_with_parameters",
        fuse_overlapping_loads=fuse_overlapping_loads,
        insert_copies=insert_copies,
        nest_map_bodies=nest_map_bodies,
        remainder_strategy=remainder_strategy,
        emission_style=emission_style,
        vectorize_config=vectorize_config,
    )
    if (fuse_overlapping_loads and insert_copies and emission_style == "default"
            and vectorize_config == "tile_nodes"):
        assert_fused_nsdfg_structure(vectorized_sdfg, ("A", ))
