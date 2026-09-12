# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap`` on a level-indexed multi-axis write -- also a memlet-propagation regression.

The cloudsc ``for_430`` shape: an outer ``LoopRegion`` over ``jk`` whose body writes a
3-D array at ``[species, jk-1, jl-1]`` for several species, with an inner map over ``jl``.
The write IS uniquely ``jk``-indexed, so the loop should parallelize. Without the
``symbols_defined_at`` fix in ``dace/sdfg/state.py`` (the one that folds enclosing
``LoopRegion`` loop variables into the defined-symbol set seen by
``propagate_memlets_nested_sdfg``), the nested-SDFG connector memlet would replace the
``jk-1`` index with the whole array extent (``Range.from_array``) and ``LoopToMap`` would
refuse with "dynamic write ... not indexed by the iteration variable". This test
exercises both: it would regress if ``LoopToMap``'s analysis tightens beyond what is
sound, *and* if memlet propagation's enclosing-loop-symbol handling regresses.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.interstate.loop_to_map import LoopToMap

NS, NL, NH = (dace.symbol(s) for s in ('NS', 'NL', 'NH'))


@dace.program
def level_indexed_write(out: dace.float64[NS, NL, NH], src: dace.float64[NS, NL, NH]):
    for jk in range(NL):
        for s in range(NS):
            for jl in range(NH):
                out[s, jk, jl] = src[s, jk, jl] * 2.0


def test_level_indexed_write_maps_and_propagates():
    """The outer ``jk`` loop maps; the rewritten SDFG matches the numpy reference."""
    sdfg = level_indexed_write.to_sdfg(simplify=True)
    n_maps_before = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))

    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    n_maps_after = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    assert n_maps_after > n_maps_before  # at least the outer jk-loop became a Map

    ns, nl, nh = 3, 4, 5
    src = np.random.default_rng(0).random((ns, nl, nh))
    out = np.zeros_like(src)
    sdfg(out=out, src=src, NS=ns, NL=nl, NH=nh)
    assert np.allclose(out, src * 2.0)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__]))
