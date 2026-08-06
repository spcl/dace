"""The CFG list must contain every region attached to a tree, whatever route attached it.

A region parented by ``add_node``/``add_branch`` used to keep the private ``[self]`` list its
constructor made, so ``cfg_id`` reported 0 and every ``cfg_list[cfg_id]`` lookup resolved to the
ROOT.  Transformations swallow the resulting lookup error and decline the match, so the only
visible symptom was fusions quietly not happening -- hence the explicit checks here.
"""
import numpy as np
import pytest

import dace
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical


def misresolved(sdfg: dace.SDFG) -> list[str]:
    """Labels of regions whose ``cfg_id`` does not round-trip through the root's CFG list."""
    return [cfg.label for cfg in sdfg.all_control_flow_regions(recursive=True) if sdfg.cfg_list[cfg.cfg_id] is not cfg]


def loop_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('cfg_list_loop')
    sdfg.add_array('A', [20], dace.float64)
    loop = LoopRegion('outer', 'i < 20', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    tasklet = body.add_tasklet('t', {'inp'}, {'out'}, 'out = inp + 1.0')
    body.add_edge(body.add_access('A'), None, tasklet, 'inp', dace.Memlet('A[i]'))
    body.add_edge(tasklet, 'out', body.add_access('A'), None, dace.Memlet('A[i]'))
    return sdfg


def nested_region_sdfg() -> dace.SDFG:
    """Subtree built out-of-tree and attached only afterwards -- attach order must not matter."""
    sdfg = dace.SDFG('cfg_list_nested')
    sdfg.add_array('A', [20], dace.float64)
    outer = ControlFlowRegion('outer', sdfg=sdfg)
    inner = ControlFlowRegion('inner', sdfg=sdfg)
    inner.add_state('s', is_start_block=True)
    outer.add_node(inner, is_start_block=True)
    sdfg.add_node(outer, is_start_block=True)
    return sdfg


def conditional_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('cfg_list_cond')
    sdfg.add_array('A', [20], dace.float64)
    cond = ConditionalBlock('cond', sdfg=sdfg)
    sdfg.add_node(cond, is_start_block=True)
    branch = ControlFlowRegion('then', sdfg=sdfg)
    cond.add_branch(dace.properties.CodeBlock('A[0] > 0'), branch)
    branch.add_state('s', is_start_block=True)
    return sdfg


def add_loop_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG('cfg_list_add_loop')
    sdfg.add_array('A', [20], dace.float64)
    before = sdfg.add_state('before', is_start_block=True)
    body = sdfg.add_state('body')
    after = sdfg.add_state('after')
    sdfg.add_loop(before, body, after, 'i', '0', 'i < 20', 'i + 1')
    return sdfg


def nested_sdfg_sdfg() -> dace.SDFG:
    """Nested SDFG fully built before its ``NestedSDFG`` node joins the outer tree."""
    inner = dace.SDFG('cfg_list_inner')
    inner.add_array('A', [20], dace.float64)
    iloop = LoopRegion('iloop', 'i < 20', 'i', 'i = 0', 'i = i + 1')
    inner.add_node(iloop, is_start_block=True)
    ibody = iloop.add_state('ibody', is_start_block=True)
    tasklet = ibody.add_tasklet('t', {'inp'}, {'out'}, 'out = inp + 1.0')
    ibody.add_edge(ibody.add_access('A'), None, tasklet, 'inp', dace.Memlet('A[i]'))
    ibody.add_edge(tasklet, 'out', ibody.add_access('A'), None, dace.Memlet('A[i]'))

    outer = dace.SDFG('cfg_list_outer')
    outer.add_array('A', [20], dace.float64)
    state = outer.add_state('main', is_start_block=True)
    nsdfg = state.add_nested_sdfg(inner, {'A'}, {'A'})
    state.add_edge(state.add_access('A'), None, nsdfg, 'A', dace.Memlet('A[0:20]'))
    state.add_edge(nsdfg, 'A', state.add_access('A'), None, dace.Memlet('A[0:20]'))
    return outer


BUILDERS = [loop_sdfg, nested_region_sdfg, conditional_sdfg, add_loop_sdfg, nested_sdfg_sdfg]


@pytest.mark.parametrize('build', BUILDERS, ids=[b.__name__ for b in BUILDERS])
def test_fresh_sdfg_registers_every_region(build):
    """Every construction route must leave the tree's CFG list complete, with no explicit reset."""
    sdfg = build()
    assert misresolved(sdfg) == []


@pytest.mark.parametrize('build', BUILDERS, ids=[b.__name__ for b in BUILDERS])
def test_deserialized_sdfg_registers_every_region(build):
    """``from_json`` attaches regions with ``add_node`` too, so it must register them as well."""
    sdfg = dace.SDFG.from_json(build().to_json())
    assert misresolved(sdfg) == []


def test_deep_region_read_first_resolves():
    """Reading an inner region's ``cfg_id`` first must repair the tree, not just a root read."""
    sdfg = nested_sdfg_sdfg()
    innermost = next(c for c in sdfg.all_control_flow_regions(recursive=True) if c.label == 'iloop')
    assert innermost.root_sdfg.cfg_list[innermost.cfg_id] is innermost


def test_rebuild_is_deferred_and_happens_once(monkeypatch):
    """The rebuild is O(tree); attaching must only MARK, and repeated reads must not re-run it."""
    calls = []
    original = AbstractControlFlowRegion.reset_cfg_list
    monkeypatch.setattr(AbstractControlFlowRegion, 'reset_cfg_list', lambda self:
                        (calls.append(self.label), original(self))[1])

    sdfg = dace.SDFG('cfg_list_cost')
    sdfg.add_array('A', [20], dace.float64)
    loop = LoopRegion('outer', 'i < 20', 'i', 'i = 0', 'i = i + 1')
    loop.add_state('body', is_start_block=True)
    sdfg.add_node(loop, is_start_block=True)
    assert calls == [], 'attaching a region must not rebuild the CFG list'

    for _ in range(10):
        assert sdfg.cfg_list[loop.cfg_id] is loop
    assert len(calls) == 1, f'CFG list rebuilt {len(calls)} times for one structural change'


def test_map_fusion_applies_inside_a_loop_region():
    """The behavioural end of it: a stale CFG list makes this fuse 0 times instead of 1."""
    sdfg = dace.SDFG('cfg_list_fusion')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('C', [20], dace.float64)
    sdfg.add_transient('B', [20], dace.float64)
    loop = LoopRegion('outer', 'i < 3', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state('body', is_start_block=True)

    access_a, access_b, access_c = state.add_access('A'), state.add_access('B'), state.add_access('C')
    state.add_mapped_tasklet('first', {'j': '0:20'}, {'inp': dace.Memlet('A[j]')},
                             'out = inp + 1.0', {'out': dace.Memlet('B[j]')},
                             input_nodes={'A': access_a},
                             output_nodes={'B': access_b},
                             external_edges=True)
    state.add_mapped_tasklet('second', {'j': '0:20'}, {'inp': dace.Memlet('B[j]')},
                             'out = inp * 2.0', {'out': dace.Memlet('C[j]')},
                             input_nodes={'B': access_b},
                             output_nodes={'C': access_c},
                             external_edges=True)
    sdfg.validate()

    assert sdfg.apply_transformations_repeated(MapFusionVertical, validate=True) == 1

    a = np.random.rand(20)
    c = np.zeros(20)
    sdfg(A=a, C=c)
    assert np.allclose(c, (a + 1.0) * 2.0)


if __name__ == '__main__':
    for builder in BUILDERS:
        test_fresh_sdfg_registers_every_region(builder)
        test_deserialized_sdfg_registers_every_region(builder)
    test_deep_region_read_first_resolves()
    test_map_fusion_applies_inside_a_loop_region()
