# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stage-sliced canonicalize: split, resume, and compare against a full run."""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize, stage_labels

N = dace.symbol("N")


@dace.program
def producer_consumer(a: dace.float64[N], b: dace.float64[N]):
    """Elementwise producer (t) feeding a consumer (b) over the same range."""
    t = dace.define_local([N], dace.float64)
    t[:] = a * 2.0
    b[:] = t + 1.0


def count_top_level_maps(sdfg):
    """Number of MapEntry nodes with no enclosing scope."""
    return sum(
        1
        for state in sdfg.states()
        for node in state.nodes()
        if isinstance(node, nodes.MapEntry) and state.entry_node(node) is None
    )


def run(sdfg, n=64):
    """Compile and run over a fixed input; returns the input and the output."""
    rng = np.random.default_rng(7)
    a = rng.random(n)
    b = np.zeros(n)
    sdfg.compile()(a=a.copy(), b=b, N=n)
    return a, b


def test_stage_labels_are_duplicate_free_and_recompose_has_a_predecessor():
    """stage_labels() names every phase once, 'recompose' among them, after the others."""
    labels = stage_labels()
    assert len(labels) == len(set(labels))
    assert "recompose" in labels
    assert labels.index("recompose") > 0


def test_a_caller_owns_fusion_by_splitting_around_the_fuse_stage():
    """Everything before 'fuse' leaves both maps standing, 'fuse' alone fuses them, and the stages after it finish
    the run: the three pieces reach the full run's map count and values."""
    labels = stage_labels()
    idx = labels.index("fuse")

    sdfg_full = producer_consumer.to_sdfg(simplify=False)
    canonicalize(sdfg_full, validate=True)
    full_maps = count_top_level_maps(sdfg_full)
    b_full = run(sdfg_full)[1]

    sdfg_split = producer_consumer.to_sdfg(simplify=False)
    canonicalize(sdfg_split, validate=True, stages=labels[:idx])
    assert count_top_level_maps(sdfg_split) > full_maps
    canonicalize(sdfg_split, validate=True, stages=["fuse"])
    assert count_top_level_maps(sdfg_split) == full_maps
    canonicalize(sdfg_split, validate=True, stages=labels[idx + 1 :])
    assert count_top_level_maps(sdfg_split) == full_maps
    a_split, b_split = run(sdfg_split)
    assert np.allclose(b_split, b_full)
    assert np.allclose(b_split, a_split * 2.0 + 1.0)


def test_unknown_stage_label_raises():
    """A typo in stages must not silently run nothing."""
    sdfg = producer_consumer.to_sdfg(simplify=False)
    with pytest.raises(ValueError):
        canonicalize(sdfg, stages=["not_a_real_stage"])


if __name__ == "__main__":
    test_stage_labels_are_duplicate_free_and_recompose_has_a_predecessor()
    test_a_caller_owns_fusion_by_splitting_around_the_fuse_stage()
    test_unknown_stage_label_raises()
