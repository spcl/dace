# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import os

from dace.symbolic import simplify

import dace
from dace.transformation.interstate import IfExtraction
from dace.sdfg.nodes import NestedSDFG

N = dace.symbol('N', dtype=dace.int32)


@dace.program
def simple_application(flag: dace.bool, arr: dace.float32[N]):
    for i in dace.map[0:N]:
        if flag:
            outval = 1
        else:
            outval = 2
        arr[i] = outval


@dace.program
def dependant_application(flag: dace.bool, arr: dace.float32[N]):
    for i in dace.map[0:N]:
        if i == 0:
            outval = 1
        else:
            outval = 2
        arr[i] = outval


def test_simple_application():
    g = simple_application.to_sdfg(simplify=False, validate=False, use_cache=False)
    g.simplify(verbose=True)  # Simplify (for convenience) to get the actual test graph.
    g.save(os.path.join('_dacegraphs', 'simple-0.sdfg'))
    g.validate()
    g.compile()

    # Before, the outer graph had only one nested SDFG.
    assert len(g.nodes()) == 1

    assert g.apply_transformations_repeated([IfExtraction]) == 1
    g.save(os.path.join('_dacegraphs', 'simple-1.sdfg'))
    g.validate()
    g.compile()

    # But now, the outer graph have four: two copies of the original nested SDFGs and two for branch management.
    assert len(g.nodes()) == 4
    assert g.start_state.is_empty()

    g.simplify()

    for s in g.nodes():
        for n in s.nodes():
            assert not isinstance(n, NestedSDFG)


def test_fails_due_to_dependency():
    sdfg = dependant_application.to_sdfg(simplify=True)

    assert sdfg.apply_transformations_repeated([IfExtraction]) == 0


if __name__ == '__main__':
    test_simple_application()
    test_fails_due_to_dependency()
