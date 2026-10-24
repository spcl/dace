# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import AllNode, AnyNode, CountLibraryNode

NODES = {"all": AllNode, "any": AnyNode, "count": CountLibraryNode}


def mask_reduction_sdfg(name, op, mask_shape, mask_dtype, dim, out_shape, out_dtype, *, mask_subset=None, impl=None):
    node_class = NODES[op]
    sdfg = dace.SDFG(f"mask_reduction_{name}")
    sdfg.add_array("mask", mask_shape, mask_dtype)
    sdfg.add_array("out", out_shape, out_dtype)
    state = sdfg.add_state()
    node = node_class(op, dim=dim)
    if impl is not None:
        node.implementation = impl
    subset = mask_subset or ", ".join(f"0:{s}" for s in mask_shape)
    state.add_edge(state.add_read("mask"), None, node, node_class.INPUT_CONNECTOR_NAME, dace.Memlet(f"mask[{subset}]"))
    state.add_edge(
        node,
        node_class.OUTPUT_CONNECTOR_NAME,
        state.add_write("out"),
        None,
        dace.Memlet.from_array("out", sdfg.arrays["out"]),
    )
    return sdfg


@pytest.mark.parametrize("op", ["all", "any", "count"])
@pytest.mark.parametrize("dim", [1, 2])
def test_reduction_along_a_dim(op, dim):
    mask = (np.random.default_rng(2).random((4, 5)) > 0.4).astype(np.int32)
    expected = {"all": mask.all, "any": mask.any, "count": mask.sum}[op](axis=dim - 1).astype(np.int32)
    out = np.zeros(expected.shape, np.int32)
    mask_reduction_sdfg(f"{op}_dim{dim}", op, [4, 5], dace.int32, dim, list(expected.shape), dace.int32)(
        mask=mask, out=out
    )
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "op,impl",
    [("all", "reduction"), ("all", "sequential"), ("any", "reduction"), ("any", "sequential"), ("count", None)],
)
def test_every_nonzero_value_is_true(op, impl):
    """A Fortran LOGICAL may hold any non-zero value for .TRUE., such as -1."""
    mask = np.array([1, 0xFFFFFFFF, 0x7FFFFFFF, 2], dtype=np.uint32)
    out = np.full(1, 7, np.uint32)
    mask_reduction_sdfg(f"{op}_{impl}_true", op, [4], dace.uint32, -1, [1], dace.uint32, impl=impl)(mask=mask, out=out)
    assert out[0] == (4 if op == "count" else 1)


def test_count_of_a_strided_section():
    mask = (np.random.default_rng(3).random((6, 5)) > 0.5).astype(np.int32)
    out = np.zeros(1, np.int32)
    sdfg = mask_reduction_sdfg(
        "count_section", "count", [6, 5], dace.int32, -1, [1], dace.int32, mask_subset="1:6:2, 3"
    )
    sdfg(mask=mask, out=out)
    assert out[0] == mask[1:6:2, 3].sum()


if __name__ == "__main__":
    for op in ("all", "any", "count"):
        for dim in (1, 2):
            test_reduction_along_a_dim(op, dim)
    for op, impl in [
        ("all", "reduction"),
        ("all", "sequential"),
        ("any", "reduction"),
        ("any", "sequential"),
        ("count", None),
    ]:
        test_every_nonzero_value_is_true(op, impl)
    test_count_of_a_strided_section()
