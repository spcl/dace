# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MPR states the device trade a canonicalizing pass declined to make.

Canonicalization always picks the most parallel form. That form is not the fastest one on every
device -- a parallel scan does more work than the sequential loop it replaces, and snapshotting a
read window costs a copy a CPU would rather not pay -- so the pass records what it could have
traded for instead of silently discarding the choice. The record is a ``specialization_hint`` on
the node or loop it applies to, and the standalone (MPR) rendering is the one place it is written
out: MPR output is read by a specializing pass or by a person, and both need to know which parts
of the shape were a decision.

A second kind of hint names what a loop IS. The rendering flattens a parallel Map and a sequential
LoopRegion to the same ``for``, so ``AnnotateLoopKinds`` writes the distinction back down before it
is lost -- and splits the sequential case in two, because "a carried dependence was proven" and
"the dependence test declined to answer" are different facts and a reader specializing this code
has to tell them apart.

These tests hold the ends of both contracts: a hint reaches the rendering, on a map, on a loop and
through a library node that the expansion consumes; every loop kind is named and named correctly;
the naming adds comments and nothing else; and nothing reaches ordinary code generation, which
already has a target and would only be given noise.
"""
import pytest

import dace
from dace import mpr_lowering
from dace.codegen.mpr import render
from dace.libraries.standard.nodes.scan import Scan, ScanOp
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.annotate_loop_kinds import AnnotateLoopKinds
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.wavefront_skew import WavefrontSkew

N = dace.symbol('N')

#: Two lines, because a hint names a trade per device and the emitter has to keep them apart.
MAP_HINT = 'traded a tiled map for a flat one.\nCPU: the tiles are worth trying.'


def scaling_sdfg(name: str) -> dace.SDFG:
    """A single parallel map, the smallest graph with a ``MapEntry`` to hang a hint on."""

    @dace.program
    def scaler(a: dace.float64[N], b: dace.float64[N]):
        for i in dace.map[0:N]:
            b[i] = a[i] * 2.0

    sdfg = scaler.to_sdfg(simplify=True)
    sdfg.name = name
    return sdfg


def map_entries(sdfg: dace.SDFG):
    return [node for state in sdfg.states() for node in state.nodes() if isinstance(node, nodes.MapEntry)]


def comment_lines(code: str):
    return [line.strip() for line in code.splitlines() if line.strip().startswith('//')]


def test_a_map_hint_is_rendered_one_comment_line_per_line():
    """Each line of a hint gets its own ``//``.

    The alternatives a hint describes are per-device (``CPU: ... / GPU: ...``) and do not read as
    one sentence, so joining them onto one line would be a worse comment than none.
    """
    sdfg = scaling_sdfg('mpr_hint_map')
    entries = map_entries(sdfg)
    assert entries, 'the program lost its map, so this test would pass without emitting anything'
    for entry in entries:
        entry.specialization_hint = MAP_HINT

    rendered = comment_lines(render(sdfg).code)
    assert '// traded a tiled map for a flat one.' in rendered
    assert '// CPU: the tiles are worth trying.' in rendered


def test_a_loop_hint_is_rendered():
    """A hint on a ``LoopRegion``, which is where ``BreakAntiDependence`` records its trade."""

    @dace.program
    def prefix(a: dace.float64[N], b: dace.float64[N]):
        for i in range(1, N):
            b[i] = b[i - 1] + a[i]

    sdfg = prefix.to_sdfg(simplify=True)
    sdfg.name = 'mpr_hint_loop'
    loops = [region for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion)]
    assert loops, 'the sequential loop was rewritten away, so there is nothing to hang a hint on'
    for loop in loops:
        loop.specialization_hint = 'kept the loop sequential.\nGPU: a scan is worth trying.'

    rendered = comment_lines(render(sdfg).code)
    assert '// kept the loop sequential.' in rendered
    assert '// GPU: a scan is worth trying.' in rendered


def test_a_scan_states_its_trade_although_the_expansion_consumes_the_node():
    """The hint has to be captured before expansion, for the same reason the description is.

    A ``Scan`` renders as several maps and a loop, and none of them remembers what chose their
    shape -- by the time code is emitted the library node is gone.
    """
    sdfg = dace.SDFG('mpr_hint_scan')
    sdfg.add_array('arr_in', [16], dace.float64)
    sdfg.add_array('arr_out', [16], dace.float64)
    state = sdfg.add_state('scan')
    node = Scan('Scan', op=ScanOp.SUM)
    state.add_node(node)
    state.add_edge(state.add_read('arr_in'), None, node, '_scan_in', dace.Memlet('arr_in[0:16]'))
    state.add_edge(node, '_scan_out', state.add_write('arr_out'), None, dace.Memlet('arr_out[0:16]'))
    sdfg.validate()

    rendered = comment_lines(render(sdfg).code)
    assert '// parallel scan; canonicalization takes the parallel form.' in rendered
    assert '// Alternative: a sequential loop over parallel maps.' in rendered
    # Folded into the node's description so one expansion is one trade, not one per map it leaves.
    assert rendered.count('// Alternative: a sequential loop over parallel maps.') == 1


def test_no_hint_and_an_empty_hint_render_identically():
    """An empty hint is not a hint. A pass that clears one must not leave a bare ``//`` behind."""
    without = render(scaling_sdfg('mpr_hint_absent')).code

    sdfg = scaling_sdfg('mpr_hint_empty')
    for entry in map_entries(sdfg):
        entry.specialization_hint = ''
    empty = render(sdfg).code

    blank = scaling_sdfg('mpr_hint_blank')
    for entry in map_entries(blank):
        entry.specialization_hint = '   \n\t'
    whitespace = render(blank).code

    assert comment_lines(empty) == comment_lines(without)
    assert comment_lines(whitespace) == comment_lines(without)
    assert '//' not in [line.strip() for line in empty.splitlines()]


@pytest.mark.parametrize('hint', [None, '', '   \n\t'])
def test_hint_comment_is_empty_for_an_empty_hint(hint):
    """The renderer answers for itself, so a caller cannot produce a dangling comment marker."""
    with mpr_lowering.dialect_scope(mpr_lowering.Dialect.STANDALONE):
        assert mpr_lowering.hint_comment(hint) == ''


def test_hint_comment_is_empty_outside_a_standalone_rendering():
    """An ordinary build already has a target; the alternatives are noise in code nobody reads."""
    with mpr_lowering.dialect_scope(mpr_lowering.Dialect.RUNTIME):
        assert mpr_lowering.hint_comment(MAP_HINT) == ''
    with mpr_lowering.dialect_scope(mpr_lowering.Dialect.STANDALONE_C):
        assert mpr_lowering.hint_comment(MAP_HINT) != ''


def test_ordinary_codegen_carries_no_hint():
    """The whole feature is scoped to the standalone rendering."""
    sdfg = scaling_sdfg('mpr_hint_leak')
    for entry in map_entries(sdfg):
        entry.specialization_hint = MAP_HINT

    with dace.config.set_temporary('compiler', 'cpu', 'implementation', value='experimental_readable'):
        ordinary = '\n'.join(obj.clean_code for obj in sdfg.generate_code())
    assert 'traded a tiled map for a flat one.' not in ordinary


# ---------------------------------------------------------------------------------------------
# Naming what every loop is: parallel, sequential-and-proven, sequential-and-undecided, wavefront.
# ---------------------------------------------------------------------------------------------


def loops_of(sdfg: dace.SDFG):
    return [region for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion)]


def named(sdfg: dace.SDFG) -> str:
    """``sdfg`` rendered after :class:`AnnotateLoopKinds` has named its loops."""
    AnnotateLoopKinds().apply_pass(sdfg, {})
    return render(sdfg).code


def carried_sdfg(name: str) -> dace.SDFG:
    """A row recurrence: the ``i`` loop carries a dependence, the ``j`` axis does not."""

    @dace.program
    def rows(a: dace.float64[N, N], b: dace.float64[N, N]):
        for i in range(1, N):
            for j in range(N):
                b[i, j] = b[i - 1, j] + a[i, j]

    sdfg = rows.to_sdfg(simplify=True)
    sdfg.name = name
    return sdfg


def breaking_sdfg(name: str) -> dace.SDFG:
    """A loop with an early exit. ``LoopToMap`` stops at the ``break`` -- before any dependence
    question is asked -- which is the case that must not be reported as a proof."""

    @dace.program
    def until_negative(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            if a[i] < 0.0:
                break
            b[i] = a[i]

    sdfg = until_negative.to_sdfg(simplify=True)
    sdfg.name = name
    return sdfg


def unexamined_sdfg(name: str) -> dace.SDFG:
    """A while loop: no iteration variable, so ``LoopToMap`` has nothing to match on and the
    dependence question is never put to it at all.

    Built by hand rather than written as a ``dace.program``, so the test states the one property it
    is about -- a ``LoopRegion`` with no ``loop_variable`` -- and does not ride on the Python
    frontend's while-loop support.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [16], dace.float64)
    sdfg.add_array('b', [1], dace.float64)
    sdfg.add_symbol('it', dace.int64)
    entry = sdfg.add_state('entry', is_start_block=True)
    loop = LoopRegion('spin', 'it < 16')
    sdfg.add_node(loop)
    sdfg.add_edge(entry, loop, dace.InterstateEdge(assignments={'it': '0'}))
    body = loop.add_state('body', is_start_block=True)
    loop.add_edge(body, loop.add_state('step'), dace.InterstateEdge(assignments={'it': 'it + 1'}))
    tasklet = body.add_tasklet('accumulate', {'inp'}, {'out'}, 'out = inp')
    body.add_edge(body.add_read('a'), None, tasklet, 'inp', dace.Memlet('a[it]'))
    body.add_edge(tasklet, 'out', body.add_write('b'), None, dace.Memlet('b[0]'))
    sdfg.validate()
    assert not loop.loop_variable, 'the loop grew an iteration variable, so it is no longer unexamined'
    return sdfg


def skewed_sdfg(name: str, tile: int) -> dace.SDFG:
    """``wf_north_west`` after an isolated skew. ``tile=0`` disables tiling."""

    @dace.program
    def wf_north_west(a: dace.float64[N, N]):
        for i in range(1, N):
            for j in range(1, N):
                a[i, j] = a[i, j] + a[i - 1, j] + a[i, j - 1]

    sdfg = wf_north_west.to_sdfg(simplify=True)
    sdfg.name = name
    skew = WavefrontSkew(target='cpu')
    skew.tile_i, skew.tile_j = tile, tile
    assert skew.apply_pass(sdfg, {}), 'the nest was not skewed, so there is no wavefront to name'
    sdfg.validate()
    return sdfg


def test_a_parallel_map_says_it_is_parallel():
    """A Map renders as a plain ``for``. Without the comment nothing in the text says the order
    is free, which is the first thing a reader specializing the code needs to know."""
    rendered = comment_lines(named(scaling_sdfg('mpr_kind_parallel')))
    assert any(line.startswith('// parallel -- the iterations are independent') for line in rendered)
    assert any('schedule decision' in line for line in rendered)


def test_a_proven_carried_dependence_is_rendered_as_a_proof_and_names_it():
    """The strong half of the sequential case: ``LoopToMap`` proved the dependence, so the order
    is required, and the hint carries the analysis' own words rather than restating the loop."""
    rendered = comment_lines(named(carried_sdfg('mpr_kind_carried')))
    proofs = [line for line in rendered if line.startswith('// sequential -- a loop-carried dependence was PROVEN')]
    assert proofs, rendered
    reasons = [line for line in rendered if line.startswith('// Proof:')]
    assert reasons and 'b' in reasons[0], reasons
    # The row axis is proven; nothing here is merely undecided.
    assert not [line for line in rendered if line.startswith('// potentially sequential')]


def test_a_declined_dependence_test_is_not_rendered_as_a_proof():
    """The weak half, and the reason the two are separate hints at all.

    ``LoopToMap`` refuses a loop with a ``break`` at the control-flow check, long before it asks
    anything about data. Calling that a carried dependence would be a claim the pipeline never
    made, and would send a reader looking for a recurrence that is not there.
    """
    rendered = comment_lines(named(breaking_sdfg('mpr_kind_undecided')))
    assert any(line.startswith('// potentially sequential -- nothing was proven either way') for line in rendered)
    assert any(line.startswith('// Declined at:') and 'Break' in line for line in rendered)
    assert any('not a proof of a dependence' in line for line in rendered)
    assert 'PROVEN' not in ' '.join(rendered)


def test_a_loop_that_was_never_examined_says_exactly_that():
    """The third sequential wording, and the most easily conflated with the other two.

    A while loop has no iteration variable, so nothing ever asked whether it carries a dependence.
    "Nothing is known" and "the test declined" are both weaker than a proof, but they are weaker
    in different places, and only the loop itself can tell a reader where to look next.
    """
    rendered = comment_lines(named(unexamined_sdfg('mpr_kind_unexamined')))
    assert any(line.startswith('// potentially sequential -- this loop was never examined') for line in rendered)
    assert any('not a proof of a dependence' in line for line in rendered)
    # Nothing was asked, so there is nothing to quote as a reason either way.
    assert not any(line.startswith('// Declined at:') or line.startswith('// Proof:') for line in rendered)


def test_a_wavefront_names_its_diagonal_and_its_front():
    """The skew is the only thing that knows an axis is a wavefront: afterwards it is an ordinary
    sequential loop over an ordinary map, and the generic classifier would say only that."""
    rendered = comment_lines(named(skewed_sdfg('mpr_kind_wavefront', tile=0)))
    assert any(line.startswith('// wavefront diagonal -- sequential') for line in rendered)
    assert any(line.startswith('// wavefront front -- parallel') for line in rendered)
    # A wavefront is a trade like any other, so it states the alternative and the device it pays on.
    assert any(line.startswith('// Alternative: the original unskewed nest') for line in rendered)


def test_a_tiled_wavefront_names_all_three_of_its_axes():
    """The tiled lowering is four loops, and a reader has to be able to tell which of them is the
    parallel one -- the tile column, not the diagonal above it or the interior below."""
    rendered = comment_lines(named(skewed_sdfg('mpr_kind_wavefront_tiled', tile=32)))
    assert any(line.startswith('// wavefront tile diagonal -- sequential') for line in rendered)
    assert any(line.startswith('// wavefront tile column -- parallel') for line in rendered)
    assert any(line.startswith('// wavefront tile interior -- sequential') for line in rendered)
    # The tiled form is a different trade from the untiled one and must not borrow its wording.
    assert not any(line.startswith('// wavefront diagonal') for line in rendered)


def test_naming_the_loops_does_not_move_a_line_of_code():
    """The whole feature is comments. Anything else -- an ``#pragma omp`` gained or lost, a bound
    rewritten -- would mean a note had turned into a decision.
    """

    def without_comments(code: str):
        return [line for line in code.splitlines() if not line.strip().startswith('//')]

    # The same SDFG rendered twice, so the entry point keeps its name and the only variable left
    # between the two texts is the annotation.
    for build in (scaling_sdfg, carried_sdfg, breaking_sdfg):
        sdfg = build(f'mpr_kind_stable_{build.__name__}')
        before = render(sdfg).code
        after = named(sdfg)
        assert without_comments(before) == without_comments(after), build.__name__
        assert len(comment_lines(after)) > len(comment_lines(before)), build.__name__


def test_an_already_recorded_hint_is_left_alone():
    """A pass that set a hint knew which alternative it declined. The generic classifier cannot
    re-derive that, so it must never overwrite one."""
    sdfg = carried_sdfg('mpr_kind_no_clobber')
    for loop in loops_of(sdfg):
        loop.specialization_hint = MAP_HINT
    rendered = comment_lines(named(sdfg))
    assert '// traded a tiled map for a flat one.' in rendered
    assert not any(line.startswith('// sequential --') for line in rendered)


def test_the_canonicalize_pipeline_names_the_loops_it_leaves_behind():
    """The pass is only worth anything if the recipe runs it: an unwired pass leaves every test
    above green while no real rendering ever gains a comment.

    Also the end-to-end shape of the feature -- one graph, both answers. Canonicalization maps the
    column axis and cannot map the row axis, and the rendering now says which is which.
    """
    sdfg = carried_sdfg('mpr_kind_pipeline')
    canonicalize(sdfg)
    rendered = comment_lines(render(sdfg).code)
    assert any(line.startswith('// parallel -- the iterations are independent') for line in rendered)
    assert any(line.startswith('// sequential -- a loop-carried dependence was PROVEN') for line in rendered)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
