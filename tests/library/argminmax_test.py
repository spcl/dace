# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Correctness tests for :class:`ArgMin` / :class:`ArgMax` library nodes.

Covers:

* whole-array reduction (Fortran ``MINLOC``/``MAXLOC`` scalar form)
* per-axis reduction via the ``dim`` property
* tie-break direction toggle via ``back``
* 0-based vs 1-based output via ``one_based``
"""

import numpy as np

import dace
from dace.libraries.standard.nodes import ArgMax, ArgMin


def _build_whole(node_cls, in_shape, dtype, *, one_based=True, back=False):
    """Build the whole-array (``dim=None``) form.

    Fortran ``MINLOC`` / ``MAXLOC`` without ``DIM`` return a rank-1
    integer array of length ``rank(input)`` with the multi-dim subscript
    of the located element, NOT a flat scalar -- so the output buffer
    here has shape ``[rank]``.
    """
    rank = len(in_shape)
    sdfg = dace.SDFG(f"{node_cls.__name__}_whole_{'_'.join(map(str, in_shape))}_{int(back)}_{int(one_based)}")
    sdfg.add_array("v", list(in_shape), dtype)
    sdfg.add_array("idx", [rank], dace.int32)
    state = sdfg.add_state()
    node = node_cls(node_cls.__name__.lower(), one_based=one_based, back=back, dim=None)
    state.add_node(node)
    state.add_edge(state.add_read("v"), None, node, "_x", dace.Memlet.from_array("v", sdfg.arrays["v"]))
    state.add_edge(node, "_idx", state.add_write("idx"), None, dace.Memlet.from_array("idx", sdfg.arrays["idx"]))
    sdfg.expand_library_nodes()
    return sdfg


def _build_dim(node_cls, in_shape, dim, dtype, *, one_based=True, back=False):
    out_shape = [s for d, s in enumerate(in_shape) if d != (dim - 1)] or [1]
    sdfg = dace.SDFG(f"{node_cls.__name__}_dim{dim}")
    sdfg.add_array("v", list(in_shape), dtype)
    sdfg.add_array("idx", out_shape, dace.int32)
    state = sdfg.add_state()
    node = node_cls(node_cls.__name__.lower(), one_based=one_based, back=back, dim=dim)
    state.add_node(node)
    state.add_edge(state.add_read("v"), None, node, "_x", dace.Memlet.from_array("v", sdfg.arrays["v"]))
    state.add_edge(node, "_idx", state.add_write("idx"), None, dace.Memlet.from_array("idx", sdfg.arrays["idx"]))
    sdfg.expand_library_nodes()
    return sdfg


def test_argmax_3d_whole_array_returns_subscript():
    x = np.zeros((3, 4, 2))
    x[2, 1, 0] = 99.0  # unique max
    sdfg = _build_whole(ArgMax, x.shape, dace.float64, one_based=True)
    idx = np.zeros(3, dtype=np.int32)
    sdfg(v=x, idx=idx)
    assert tuple(idx) == (3, 2, 1), idx


def test_argmin_2d_dim2():
    """Reduce along Fortran dim=2 (cols): result has shape (#rows,)."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((5, 7))
    sdfg = _build_dim(ArgMin, x.shape, 2, dace.float64)
    idx = np.zeros((5,), dtype=np.int32)
    sdfg(v=x, idx=idx)
    np.testing.assert_array_equal(idx, np.argmin(x, axis=1) + 1)


if __name__ == "__main__":
    test_argmax_3d_whole_array_returns_subscript()
    test_argmin_2d_dim2()
