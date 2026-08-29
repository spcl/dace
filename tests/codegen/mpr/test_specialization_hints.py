# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MPR states the device trade a canonicalizing pass declined to make.

Canonicalization always picks the most parallel form. That form is not the fastest one on every
device -- a parallel scan does more work than the sequential loop it replaces, and snapshotting a
read window costs a copy a CPU would rather not pay -- so the pass records what it could have
traded for instead of silently discarding the choice. The record is a ``specialization_hint`` on
the node or loop it applies to, and the standalone (MPR) rendering is the one place it is written
out: MPR output is read by a specializing pass or by a person, and both need to know which parts
of the shape were a decision.

These tests hold the two ends of that contract: a hint reaches the rendering, on a map, on a loop
and through a library node that the expansion consumes; and nothing reaches ordinary code
generation, which already has a target and would only be given noise.
"""
import pytest

import dace
from dace import mpr_lowering
from dace.codegen.mpr import render
from dace.libraries.standard.nodes.scan import Scan, ScanOp
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
