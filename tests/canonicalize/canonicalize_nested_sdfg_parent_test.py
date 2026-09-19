# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A control-flow region carries its nested SDFGs' parent references with it.

``SDFGState.add_node`` re-homes the three back-references of a nested SDFG node it accepts
(``parent``, ``parent_sdfg``, ``parent_nsdfg_node``). ``ControlFlowRegion.add_node`` re-homes the
blocks of a region it accepts, and has to re-home the nested SDFGs inside those blocks for the same
reason: a region built by deepcopying a detached one -- what loop fission, loop specialization and
loop unrolling all do -- comes back with ``parent_sdfg`` unset, because the owning SDFG was absent
from the deepcopy memo. Validation rejects that with "Parent SDFG not properly set for nested SDFG
node", which is how the wavefront kernels below failed to canonicalize.
"""
import copy

import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation.passes.canonicalize import pipeline as canon
from dace.transformation.passes.loop_fission import LoopFission

N = dace.symbol('N')


@dace.program
def wf_triangular(a: dace.float64[N, N]):
    """The tagged kernel: the wavefront recurrence over the upper triangle ``j >= i``."""
    for i in range(1, N):
        for j in range(i, N):
            a[i, j] = a[i, j] + a[i - 1, j] + a[i, j - 1]


@dace.program
def wavefront2d(a: dace.float64[N, N]):
    """North + west + the ``(1, 1)`` corner dependence."""
    for i in range(1, N):
        for j in range(1, N):
            a[i, j] = 0.25 * (a[i, j] + a[i - 1, j] + a[i, j - 1] + a[i - 1, j - 1])


@dace.program
def wf_north_west(a: dace.float64[N, N]):
    """The rectangular north-west wavefront."""
    for i in range(1, N):
        for j in range(1, N):
            a[i, j] = a[i - 1, j] + a[i, j - 1]


def writer_sdfg(name: str) -> dace.SDFG:
    """A one-state SDFG writing a single element, for use as a nested SDFG."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('x', [10], dace.float64)
    state = sdfg.add_state('compute')
    tasklet = state.add_tasklet('t', {}, {'o'}, 'o = 1.0')
    state.add_edge(tasklet, 'o', state.add_access('x'), None, dace.Memlet('x[0]'))
    return sdfg


def loop_with_nested_sdfgs() -> tuple[dace.SDFG, LoopRegion]:
    """An SDFG holding one loop whose two body states each hold a nested SDFG."""
    sdfg = dace.SDFG('outer')
    sdfg.add_array('a', [10], dace.float64)
    sdfg.add_array('b', [10], dace.float64)
    loop = LoopRegion('loop', 'i < 10', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    for label, array in (('s0', 'a'), ('s1', 'b')):
        state = loop.add_state(label, is_start_block=(label == 's0'))
        node = state.add_nested_sdfg(writer_sdfg(f'in_{label}'), {}, {'x'})
        state.add_edge(node, 'x', state.add_access(array), None, dace.Memlet(f'{array}[i]'))
    loop.add_edge(loop.nodes()[0], loop.nodes()[1], dace.InterstateEdge())
    sdfg.validate()
    return sdfg, loop


def nested_sdfg_nodes(sdfg: dace.SDFG) -> list[nodes.NestedSDFG]:
    """Every nested SDFG node in ``sdfg``'s own namespace, in traversal order."""
    return [n for state in sdfg.all_states() for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]


def test_add_node_rehomes_nested_sdfgs_of_a_detached_region():
    """Accepting a detached region claims the nested SDFGs in its blocks, not just the blocks."""
    _, loop = loop_with_nested_sdfgs()
    detached = copy.deepcopy(loop)

    host = dace.SDFG('host')
    host.add_array('a', [10], dace.float64)
    host.add_array('b', [10], dace.float64)
    region = ControlFlowRegion('region')
    host.add_node(region, is_start_block=True)
    region.add_node(detached, is_start_block=True)

    found = nested_sdfg_nodes(host)
    assert len(found) == 2
    for node in found:
        assert node.sdfg.parent_sdfg is host
        assert node.sdfg.parent_nsdfg_node is node
        assert node.sdfg.parent.sdfg is host
    host.validate()


def test_loop_fission_clone_keeps_nested_sdfg_parents():
    """The per-group clones ``LoopFission`` makes stay attached to the SDFG they land in."""
    sdfg, loop = loop_with_nested_sdfgs()
    LoopFission._fission_blocks(loop, [[loop.nodes()[0]], [loop.nodes()[1]]])

    found = nested_sdfg_nodes(sdfg)
    assert len(found) == 2
    for node in found:
        assert node.sdfg.parent_sdfg is sdfg
    sdfg.validate()


@pytest.mark.parametrize('program', [wf_triangular, wavefront2d, wf_north_west],
                         ids=['wf_triangular', 'wavefront2d', 'wf_north_west'])
def test_wavefront_nest_canonicalizes_to_a_valid_sdfg(program):
    """A skewed or triangular nest survives the whole pipeline with its parent references intact."""
    sdfg = program.to_sdfg(simplify=False)
    canon.canonicalize(sdfg, target='cpu')
    sdfg.validate()


if __name__ == '__main__':
    test_add_node_rehomes_nested_sdfgs_of_a_detached_region()
    test_loop_fission_clone_keeps_nested_sdfg_parents()
    for prog in (wf_triangular, wavefront2d, wf_north_west):
        test_wavefront_nest_canonicalizes_to_a_valid_sdfg(prog)
