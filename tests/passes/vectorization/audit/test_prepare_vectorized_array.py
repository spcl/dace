# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tripwire: ``prepare_vectorized_array`` calls ``drop_dims`` with swapped argument
order, and the helper hardcodes ``keep_mask[-1] = 1`` — wrong for Fortran-layout.
Together they leave inner-SDFG accesses unchanged after the connector array has
been reshaped from ``(D0, D1)`` to ``(vector_width,)``; downstream this produces
validation errors or wrong code.

The existing cloudsc tests don't surface either issue because cloudsc is C-layout
(so the ``keep_mask[-1]=1`` heuristic happens to pick the contiguous dim) and the
cloudsc array names are explicitly skipped via ``Vectorize.user_skip_nsdfg_arrays``
so ``prepare_vectorized_array`` is never called for them.

Expected to xfail until the planned pass-through-subsets redesign deletes
``prepare_vectorized_array`` entirely. After the redesign there is no per-connector
reshape and this test should xpass.
"""
import dace
from dace.transformation.passes.vectorization.utils.nsdfg_reshape import prepare_vectorized_array


def test_prepare_vectorized_array_updates_inner_2d_access():
    """A 2D connector with an inner 2D access should become a 1D connector with a 1D access."""
    # Inner SDFG: a single state with a 2D access node copying to a sink scalar.
    inner_sdfg = dace.SDFG("inner")
    inner_sdfg.add_array("arr", shape=(8, 16), dtype=dace.float64, transient=False)
    inner_sdfg.add_array("sink", shape=(1, ), dtype=dace.float64, transient=False)
    inner_state = inner_sdfg.add_state("s", is_start_block=True)
    an_in = inner_state.add_access("arr")
    an_out = inner_state.add_access("sink")
    inner_state.add_edge(an_in, None, an_out, None, dace.memlet.Memlet("arr[5, 7]"))

    # Parent state holding a 2D C-layout original array. (C-layout: strides=(16, 1),
    # so the contiguous dim is dim 1; the buggy keep_mask[-1]=1 happens to keep dim 1
    # in C-layout, which is why this test is most informative on the *call-site* bug
    # — drop_dims is invoked with swapped args, so it never updates the inner memlet
    # regardless of which dim we'd want to keep.)
    parent_sdfg = dace.SDFG("parent")
    parent_sdfg.add_array("arr_p", shape=(8, 16), dtype=dace.float64, strides=(16, 1))
    parent_state = parent_sdfg.add_state("p", is_start_block=True)

    prepare_vectorized_array(
        state=parent_state,
        inner_sdfg=inner_sdfg,
        inner_arr_name="arr",
        orig_dataname="arr_p",
        orig_arr=parent_sdfg.arrays["arr_p"],
        subset=dace.subsets.Range([(5, 5, 1), (7, 14, 1)]),
        vector_width=8,
        vector_storage=dace.dtypes.StorageType.Register,
    )

    # The connector array descriptor itself should be reshaped to (vector_width,)
    assert inner_sdfg.arrays["arr"].shape == (8, ), f"connector shape: {inner_sdfg.arrays['arr'].shape}"

    # The inner memlet that previously addressed arr[5, 7] (2 dims) must now be 1-D
    # to be compatible with the (8,)-shaped connector.
    arr_edges = [e for e in inner_state.edges() if e.data.data == "arr"]
    assert len(arr_edges) == 1, f"expected 1 edge to arr, got {len(arr_edges)}"
    inner_subset_dims = len(list(arr_edges[0].data.subset))
    assert inner_subset_dims == 1, (
        f"drop_dims arg-swap left inner memlet at {inner_subset_dims} dims, connector is now (8,) — "
        f"mismatch will fail SDFG validation downstream")
