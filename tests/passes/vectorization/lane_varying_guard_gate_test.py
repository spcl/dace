# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A guard over widened data must not decide an interstate symbol assignment.

``npbench``'s ``azimint_naive`` counts the points falling in each radial bin::

    for j in dace.map[0:N]:
        if mask_r12[j]:
            tmp += data[j]
            on_values += 1

Branch lowering if-converts the float accumulator (a dataflow write) but not the counter, whose
write is an interstate symbol assignment inside a ``ConditionalBlock``. Widening then turns the
guard's operand into a ``bool[W]`` buffer while the guard stays scalar control flow, so codegen
emits ``if (<bool[8]>)`` -- an array decaying to a never-null pointer. Every lane took the branch,
``on_values`` reached N instead of the masked count, and every bin came out scaled by exactly that
factor while the SDFG validated and the numerator stayed correct.
"""
import copy

import pytest

import dace
from dace import nodes
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.utils.pass_invariants import (
    no_conditional_interstate_assign_on_widened_data)
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from tests.corpus.npbench import npbench

WIDTHS = (8, )


def guarded_assign_sdfg(guard_shape) -> dace.SDFG:
    """``if guard: k = 1`` where ``guard`` has ``guard_shape`` -- the shape decides lane-varying."""
    sdfg = dace.SDFG(f'guarded_assign_{len(guard_shape)}d')
    sdfg.add_array('guard', guard_shape, dace.bool_, transient=True)
    sdfg.add_symbol('k', dace.int64)
    entry = sdfg.add_state('entry', is_start_block=True)

    cond_block = ConditionalBlock('if_guard')
    sdfg.add_node(cond_block)
    sdfg.add_edge(entry, cond_block, dace.InterstateEdge())
    branch = dace.sdfg.state.ControlFlowRegion('if_body', sdfg=sdfg)
    cond_block.add_branch(CodeBlock('guard'), branch)
    branch.add_edge(branch.add_state('b0', is_start_block=True), branch.add_state('b1'),
                    dace.InterstateEdge(assignments={'k': '1'}))
    return sdfg


def test_invariant_flags_a_guard_over_a_widened_buffer():
    violation = no_conditional_interstate_assign_on_widened_data(guarded_assign_sdfg(WIDTHS), WIDTHS)
    assert violation is not None, "a bool[8] guard deciding an interstate assignment was not flagged"
    assert 'guard' in violation and 'k' in violation, violation


def test_invariant_accepts_a_scalar_guard():
    """The control: a guard that is NOT a lane buffer decides once for the tile, which is legal."""
    assert no_conditional_interstate_assign_on_widened_data(guarded_assign_sdfg((1, )), WIDTHS) is None


@pytest.mark.skipif(not any(c['name'] == 'azimint_naive' for c in npbench.collect()),
                    reason='azimint_naive is not in the npbench corpus')
def test_azimint_naive_masked_counter_is_not_vectorized_per_tile():
    """End-to-end: the kernel that produced the miscompile must come back numerically correct.

    Without the gate the counter is widened away and every bin is scaled by N; the assertion below
    is on the values, so it fails on numbers, not on structure.
    """
    corpus = {c['name']: c for c in npbench.collect()}['azimint_naive']
    arrays, params = npbench.make_inputs(corpus)
    reference = npbench.reference_outputs(corpus, arrays, params)

    sdfg = npbench.fresh_sdfg(corpus)
    canonicalize(sdfg, validate=True)
    canon = copy.deepcopy(sdfg)
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=WIDTHS, target_isa='AVX512', remainder_strategy='full_mask',
                        branch_mode='merge')).apply_pass(sdfg, {})

    assert no_conditional_interstate_assign_on_widened_data(sdfg, WIDTHS) is None
    # Refused, so the pass handed back the canonicalized input: no tile lib node was introduced.
    tile_nodes = [
        n for sd in sdfg.all_sdfgs_recursive() for state in sd.states() for n in state.nodes()
        if isinstance(n, nodes.LibraryNode) and type(n).__name__.startswith('Tile')
    ]
    assert not tile_nodes, f"the refused kernel still carries tile nodes: {tile_nodes}"
    assert sum(len(sd.states()) for sd in sdfg.all_sdfgs_recursive()) == sum(
        len(sd.states()) for sd in canon.all_sdfgs_recursive()), 'the refusal did not restore the input'

    got = npbench.run_outputs(corpus, sdfg, arrays, params)
    assert npbench.outputs_match(reference, got), "the masked counter lost its predicate"
