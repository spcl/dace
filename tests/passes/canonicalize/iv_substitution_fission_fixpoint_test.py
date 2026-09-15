# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``IvSubstitutionFissionFixpoint`` settles its ``SimplifyPass`` to the graph the full fixpoint reaches."""
import copy

import dace
from dace.transformation.passes.canonicalize.pipeline import IvSubstitutionFissionFixpoint
from dace.transformation.passes.simplify import SimplifyPass

N = dace.symbol('N')


@dace.program
def constant_dead_branch(a: dace.float64[N], b: dace.float64[N]):
    # Promoting k kills a branch in one sweep; fusing what is left takes a second one.
    k = 0
    for i in range(N):
        if k > 0:
            b[i] = a[i]
        else:
            b[i] = 2.0 * a[i]


def strip_debuginfo(node: object) -> object:
    if isinstance(node, dict):
        return {key: strip_debuginfo(value) for key, value in node.items() if key != 'debuginfo'}
    if isinstance(node, list):
        return [strip_debuginfo(value) for value in node]
    return node


def structure(sdfg: dace.SDFG) -> str:
    return sdfg.hash_sdfg(strip_debuginfo(sdfg.to_json()))


def test_settling_early_reaches_the_graph_of_the_full_simplify_fixpoint():
    sdfg = constant_dead_branch.to_sdfg(simplify=False)
    states_before = len(list(sdfg.all_states()))
    reference = copy.deepcopy(sdfg)

    reference_changed = SimplifyPass().apply_pass(reference, {}) is not None
    changed = IvSubstitutionFissionFixpoint.simplify_until_settled(SimplifyPass(), sdfg)

    assert changed is True and reference_changed is True
    assert len(list(sdfg.all_states())) < states_before
    assert structure(sdfg) == structure(reference)


def test_a_settled_graph_reports_no_change_and_stays_untouched():
    sdfg = constant_dead_branch.to_sdfg(simplify=False)
    SimplifyPass().apply_pass(sdfg, {})
    settled = structure(sdfg)

    assert IvSubstitutionFissionFixpoint.simplify_until_settled(SimplifyPass(), sdfg) is False
    assert structure(sdfg) == settled
