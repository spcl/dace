# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unit tests for the ``dace.data.add_mask`` helper.

Lane-masks for the upcoming branch-normalization and map-preparation passes
are *plain* boolean transients with shape ``(vector_width,)``. There is
intentionally no typeclass tag or descriptor flag — passes detect masks from
the graph topology around them, not from a property on the descriptor. These
tests pin that representation.
"""
import dace
import pytest


def test_add_mask_creates_bool_register_transient_with_correct_shape():
    sdfg = dace.SDFG("m")
    name = dace.data.add_mask(sdfg, "_mask", vector_width=8)
    assert name in sdfg.arrays
    arr = sdfg.arrays[name]
    assert isinstance(arr, dace.data.Array)
    assert tuple(arr.shape) == (8, )
    assert arr.dtype == dace.bool_
    assert arr.storage == dace.dtypes.StorageType.Register
    assert arr.transient is True


def test_add_mask_avoids_collisions_via_find_new_name():
    sdfg = dace.SDFG("m")
    sdfg.add_array("_mask", shape=(1, ), dtype=dace.float64)
    name = dace.data.add_mask(sdfg, "_mask", vector_width=4)
    assert name != "_mask"
    assert sdfg.arrays[name].dtype == dace.bool_
    assert tuple(sdfg.arrays[name].shape) == (4, )
    assert sdfg.arrays["_mask"].dtype == dace.float64


def test_add_mask_rejects_non_positive_width():
    sdfg = dace.SDFG("m")
    with pytest.raises(ValueError):
        dace.data.add_mask(sdfg, "_mask", vector_width=0)
    with pytest.raises(ValueError):
        dace.data.add_mask(sdfg, "_mask", vector_width=-1)


def test_add_mask_descriptor_has_no_typeclass_flag():
    """Masks are identified by topology, not by a flag on the descriptor.
    Anyone checking ``isinstance(arr, dace.data.Array) and arr.dtype == dace.bool_``
    should be enough; explicit ``is_mask``-style properties are out of scope."""
    sdfg = dace.SDFG("m")
    name = dace.data.add_mask(sdfg, "_mask", vector_width=8)
    arr = sdfg.arrays[name]
    # No new attributes invented; just the standard descriptor surface.
    assert not hasattr(arr, "is_mask")


def test_add_mask_can_be_allocated_multiple_times_independently():
    sdfg = dace.SDFG("m")
    a = dace.data.add_mask(sdfg, "iter_mask", vector_width=8)
    b = dace.data.add_mask(sdfg, "iter_mask", vector_width=8)
    c = dace.data.add_mask(sdfg, "cond_mask", vector_width=8)
    assert len({a, b, c}) == 3
    for n in (a, b, c):
        assert sdfg.arrays[n].dtype == dace.bool_
        assert tuple(sdfg.arrays[n].shape) == (8, )
