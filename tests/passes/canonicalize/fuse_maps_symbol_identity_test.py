# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""npbench azimint_naive's mask map fuses into its masked reduction.

The two maps range over ``0:N`` with ``N`` spelled as two sympy symbols (``int64`` from one frontend
path, ``int`` from the other), so the parameter remapping used to see different ranges and refuse.
"""

import importlib

import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize import canonicalize

azimint = importlib.import_module("tests.corpus.npbench.map_reduce.azimint_naive")


@pytest.mark.parametrize("target", ["cpu", "gpu"])
def test_the_mask_is_fused_into_the_masked_reduction(target):
    sdfg = azimint.kernel.to_sdfg(simplify=True)
    canonicalize(sdfg, target=target)
    masks = [
        (state.label, node.data)
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.AccessNode) and node.data.startswith("mask")
    ]
    assert not masks, f"the N-wide mask survived canonicalization: {masks}"


def test_the_fused_kernel_computes_the_bin_means():
    sdfg = azimint.kernel.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    rng = np.random.default_rng(0)
    n, npt = 300, 5
    data, radius = rng.random(n), rng.random(n)
    want = np.zeros(npt)
    azimint.reference(data, radius, npt, want)
    np.testing.assert_allclose(sdfg(data=data, radius=radius, N=n, npt=npt), want, rtol=1e-12)
