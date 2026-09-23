# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``state_local`` matching applies what the plain matcher applies, in its order, without re-walking."""
import copy

import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.passes.pattern_matching import PatternApplyOnceEverywhere

N = dace.symbol('N')


@dace.program
def branchy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N, N], s: dace.float64[1], flag: dace.int64):
    c[:, :] = 1.0
    b[:] = a + 1.0
    a[:] = b * 2.0
    if flag > 0:
        b[:] = a - 1.0
        a[:] = b * b
    else:
        a[:] = b + 3.0
    for i in dace.map[0:N]:
        s[0] += a[i]
    b[:] = a * 0.5


def layout(sdfg: dace.SDFG) -> list:
    rows = []
    for region in sdfg.all_control_flow_regions(recursive=True):
        rows.append((type(region).__name__, region.label))
        for block in region.nodes():
            rows.append(('block', type(block).__name__, block.label))
            if isinstance(block, dace.SDFGState):
                rows += [(str(e.src), e.src_conn, str(e.dst), e.dst_conn, str(e.data)) for e in block.edges()]
        rows += [(e.src.label, e.dst.label, e.data.condition.as_string, str(e.data.assignments))
                 for e in region.edges()]
    return rows


def lowering(state_local: bool) -> PatternApplyOnceEverywhere:
    """The canonicalization ``lower`` stage, with or without ``state_local``."""
    xform = MapToForLoop()
    xform.keep_reductions_parallel = True
    return PatternApplyOnceEverywhere([xform], validate=False, state_local=state_local)


def test_state_local_lowering_is_the_plain_matchers_lowering():
    """Every name and position must be the plain matcher's, or everything downstream of the
    canonicalization ``lower`` stage shifts."""
    sdfg = branchy.to_sdfg(simplify=True)
    reference = copy.deepcopy(sdfg)
    lowering(False).apply_pass(reference, {})
    lowering(True).apply_pass(sdfg, {})
    sdfg.validate()
    assert layout(sdfg) == layout(reference)
    assert any(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions(recursive=True))
    assert [r.label for r in sdfg.cfg_list] == [r.label for r in reference.cfg_list]


def chain_with_refused_head(count: int) -> dace.SDFG:
    """A 2-D map (refused: one parameter only) first, then ``count`` 1-D maps in states of their own."""
    sdfg = dace.SDFG('refused_head')
    sdfg.add_array('A', [N, N], dace.float64)
    for k in range(count):
        sdfg.add_array(f'x{k}', [N], dace.float64)
    head = sdfg.add_state('head')
    head.add_mapped_tasklet('twod', {
        'i': '0:N',
        'j': '0:N'
    }, {},
                            'o = 1.0', {'o': dace.Memlet('A[i, j]')},
                            external_edges=True)
    last = head
    for k in range(count):
        state = sdfg.add_state_after(last, f's{k}')
        state.add_mapped_tasklet(f'm{k}', {'i': '0:N'}, {},
                                 'o = 1.0', {'o': dace.Memlet(f'x{k}[i]')},
                                 external_edges=True)
        last = state
    return sdfg


def test_a_refused_map_is_probed_once_not_once_per_lowering(monkeypatch):
    """The plain matcher re-walks from the root after every lowering and probes the refused map again
    each time, which is what made the ``lower`` stage quadratic in the map count (warpx: 3300 maps)."""
    count = 12
    probes = []
    original = MapToForLoop.can_be_applied

    def counted(self, graph, expr_index, sdfg, permissive=False):
        probes.append(self.map_entry.map.label)
        return original(self, graph, expr_index, sdfg, permissive)

    monkeypatch.setattr(MapToForLoop, 'can_be_applied', counted)
    sdfg = chain_with_refused_head(count)
    assert len(lowering(True).apply_pass(sdfg, {})['MapToForLoop']) == count
    assert probes.count('twod_map') == 1, probes
    assert len(probes) == count + 1, probes
    remaining = [n for n, owner in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert [n.map.label for n in remaining] == ['twod_map']


def test_an_sdfg_without_maps_is_left_alone():
    sdfg = dace.SDFG('empty_of_maps')
    sdfg.add_state('only')
    cfg_list = sdfg.cfg_list
    assert lowering(True).apply_pass(sdfg, {}) is None
    assert sdfg.cfg_list is cfg_list


def test_state_local_refuses_an_interstate_transformation():
    """The resumed walk only holds for a transformation decided by its own state."""
    from dace.transformation.interstate import ConditionFusion
    with pytest.raises(ValueError, match='single-state'):
        PatternApplyOnceEverywhere([ConditionFusion()], state_local=True).apply_pass(dace.SDFG('x'), {})
