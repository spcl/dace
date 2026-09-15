# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A guard over widened data must not decide an interstate symbol assignment.

``npbench``'s ``azimint_naive`` counts the points falling in each radial bin::

    for j in dace.map[0:N]:
        if mask_r12[j]:
            tmp += data[j]
            on_values += 1

Branch lowering if-converted the float accumulator (a dataflow write) but not the counter, whose
write is an interstate symbol assignment inside a ``ConditionalBlock``. Widening then turned the
guard's operand into a ``bool[W]`` buffer while the guard stayed scalar control flow, so codegen
emitted ``if (<bool[8]>)`` -- an array decaying to a never-null pointer. Every lane took the branch,
``on_values`` reached N instead of the masked count, and every bin came out scaled by exactly that
factor while the SDFG validated and the numerator stayed correct.

``LowerInterstateConditionalAssignmentsToTasklets`` now demotes such a binding to a scalar before
the two ITE passes look at the arm, so the counter becomes a dataflow write that gets its own
per-lane select. Both accumulators are predicated and the kernel vectorizes correctly, which is
what the end-to-end test below pins -- on the emitted selects AND on the numbers.
"""
import pytest

import dace
from dace import nodes
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock
from dace.libraries.tileops._dispatch import detect_host_isa
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


def test_azimint_naive_masked_counter_is_predicated_per_lane():
    """End-to-end: the kernel that produced the miscompile must vectorize, and come back correct.

    Two assertions, because either alone is passable for the wrong reason. The NUMBERS catch a lost
    predicate, but they also pass if the vectorizer simply refused the kernel and handed the input
    back untouched. The STRUCTURE catches that: the kernel really is tiled, with a per-lane select
    for EACH of the two masked accumulators. Before the demotion only the float accumulator got
    one -- the counter reached its reduction buffer through an unmasked constant broadcast.
    """
    by_name = {c['name']: c for c in npbench.collect()}
    assert 'azimint_naive' in by_name, ('azimint_naive left the npbench corpus; it is the kernel this '
                                        'whole file exists for, so a rename must fail loudly here')
    corpus = by_name['azimint_naive']
    arrays, params = npbench.make_inputs(corpus)
    reference = npbench.reference_outputs(corpus, arrays, params)

    sdfg = npbench.fresh_sdfg(corpus)
    canonicalize(sdfg, validate=True)
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=WIDTHS,
                        target_isa=detect_host_isa(),
                        remainder_strategy='full_mask',
                        branch_mode='merge')).apply_pass(sdfg, {})

    assert no_conditional_interstate_assign_on_widened_data(sdfg, WIDTHS) is None

    # One reduction buffer per accumulator, and one per-lane select per accumulator. The counter
    # used to reach its buffer through an unmasked constant broadcast and no select at all, so it
    # is the SECOND select that this pins -- a count of one is the old miscompile.
    assert len(_tile_nodes(sdfg, 'TileStore', into='_red_buf')) == 2, 'expected two reduction buffers'
    selects = _tile_nodes(sdfg, 'TileITE')
    assert len(selects) == 2, (f"expected a per-lane select for the float accumulator AND the counter, "
                               f"got {[n.label for n in selects]}")

    got = npbench.run_outputs(corpus, sdfg, arrays, params)
    assert npbench.outputs_match(reference, got), "the masked counter lost its predicate"


def _tile_nodes(sdfg: dace.SDFG, kind: str, into: str = None) -> list:
    """Every tile library node of type ``kind``; with ``into``, only those writing that prefix.

    Matched on the class name rather than by importing each tile node type: the test is about how
    many selects and stores the pipeline emitted, not about their classes.
    """
    found = []
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for n in state.nodes():
                if not (isinstance(n, nodes.LibraryNode) and type(n).__name__ == kind):
                    continue
                if into is not None and not any(e.data.data is not None and e.data.data.startswith(into)
                                                for e in state.out_edges(n)):
                    continue
                found.append(n)
    return found


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
