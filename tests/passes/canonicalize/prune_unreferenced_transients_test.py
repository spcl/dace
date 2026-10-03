# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A transient nothing names any more is removed; one that is still named survives.

The stages at the tail of the canonicalization recipe delete a temporary's last reader without
deleting its descriptor, and the reclaimer that would collect it (``ArrayElimination``) both runs
earlier and skips ``Scalar`` outright. What is left is invisible to codegen but not to a re-run,
so the recipe ends with :class:`PruneUnreferencedTransients`.

The dangerous direction is deleting a descriptor something still names, so the cases below pin
the ways a name survives without an access node: a C++ tasklet body, whose free symbols the AST
walker cannot see at all, and an interstate-edge assignment.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.transformation.passes.canonicalize.prune_unreferenced_transients import PruneUnreferencedTransients

N = dace.symbol('N')


def prune(sdfg: dace.SDFG) -> set:
    return PruneUnreferencedTransients().apply_pass(sdfg, {}) or set()


def one_map_sdfg(name: str) -> dace.SDFG:
    """``out[i] = x[i] + 1`` plus an unreferenced transient scalar."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('x', [N], dace.float64)
    sdfg.add_array('out', [N], dace.float64)
    sdfg.add_scalar('orphan', dace.float64, transient=True)
    state = sdfg.add_state()
    state.add_mapped_tasklet('m', {'i': '0:N'}, {'__x': dace.Memlet('x[i]')},
                             '__o = __x + 1.0', {'__o': dace.Memlet('out[i]')},
                             external_edges=True)
    return sdfg


def test_an_unreferenced_transient_is_removed():
    sdfg = one_map_sdfg('prune_orphan')
    sdfg.validate()
    assert prune(sdfg) == {'prune_orphan.orphan'}
    assert 'orphan' not in sdfg.arrays
    sdfg.validate()

    n = 64
    x = np.random.default_rng(0).random(n)
    out = np.zeros(n)
    sdfg(x=x, out=out, N=n)
    assert np.allclose(out, x + 1.0)


def test_a_live_transient_is_kept():
    """An access node is the ordinary reference, and it must hold the descriptor."""
    sdfg = dace.SDFG('prune_live')
    sdfg.add_array('x', [N], dace.float64)
    sdfg.add_array('out', [N], dace.float64)
    sdfg.add_transient('buf', [N], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet('m', {'i': '0:N'}, {'__x': dace.Memlet('x[i]')},
                             '__o = __x * 2.0', {'__o': dace.Memlet('buf[i]')},
                             external_edges=True)
    second = sdfg.add_state_after(state)
    second.add_mapped_tasklet('m2', {'i': '0:N'}, {'__b': dace.Memlet('buf[i]')},
                              '__o = __b + 1.0', {'__o': dace.Memlet('out[i]')},
                              external_edges=True)
    sdfg.validate()
    assert prune(sdfg) == set()
    assert 'buf' in sdfg.arrays


def test_a_name_only_a_cpp_tasklet_uses_is_kept():
    """``CodeBlock.get_free_symbols`` walks a PYTHON ast and returns nothing for any other
    language, so a C++ tasklet naming a transient is invisible to symbol analysis. Pruning on
    that answer would delete a buffer the emitted C++ still dereferences."""
    sdfg = dace.SDFG('prune_cpp')
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_scalar('seen_only_by_cpp', dace.float64, transient=True)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('t', {}, {'__o'},
                                'seen_only_by_cpp = 3.0;\n__o = seen_only_by_cpp;',
                                language=dtypes.Language.CPP)
    state.add_edge(tasklet, '__o', state.add_write('out'), None, dace.Memlet('out[0]'))
    sdfg.validate()
    assert prune(sdfg) == set()
    assert 'seen_only_by_cpp' in sdfg.arrays


def test_a_name_an_interstate_assignment_uses_is_kept():
    sdfg = dace.SDFG('prune_iedge')
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_scalar('carrier', dace.int32, transient=True)
    sdfg.add_symbol('picked', dace.int32)
    first = sdfg.add_state()
    second = sdfg.add_state()
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={'picked': 'carrier + 1'}))
    assert prune(sdfg) == set()
    assert 'carrier' in sdfg.arrays


def test_pruning_is_recursive_through_nested_sdfgs():
    sdfg = dace.SDFG('prune_outer')
    sdfg.add_array('out', [N], dace.float64)
    inner = dace.SDFG('prune_inner')
    inner.add_array('o', [N], dace.float64)
    inner.add_scalar('inner_orphan', dace.float64, transient=True)
    istate = inner.add_state()
    istate.add_mapped_tasklet('m', {'i': '0:N'}, {}, '__o = 1.0', {'__o': dace.Memlet('o[i]')}, external_edges=True)
    state = sdfg.add_state()
    nsdfg = state.add_nested_sdfg(inner, {}, {'o'}, symbol_mapping={'N': N})
    state.add_edge(nsdfg, 'o', state.add_write('out'), None, dace.Memlet('out[0:N]'))
    sdfg.validate()
    assert prune(sdfg) == {'prune_inner.inner_orphan'}
    assert 'inner_orphan' not in inner.arrays


def test_the_pipeline_leaves_no_unreferenced_transient():
    """The recipe's own output must already be pruned -- that is what the stage is for."""
    from dace.sdfg import nodes
    from dace.transformation.passes.canonicalize import canonicalize

    @dace.program
    def chained(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
        for i in dace.map[0:N]:
            out[i] = a[i] * b[i] + a[i]

    sdfg = chained.to_sdfg(simplify=True)
    canonicalize(sdfg, target='cpu')
    for sd in sdfg.all_sdfgs_recursive():
        named = {n.data for st in sd.states() for n in st.nodes() if isinstance(n, nodes.AccessNode)}
        named |= {e.data.data for st in sd.states() for e in st.edges() if e.data is not None and e.data.data}
        text = ' '.join(str(t.code) for st in sd.states() for t in st.nodes() if isinstance(t, nodes.Tasklet))
        stale = [
            n for n, d in sd.arrays.items() if d.transient and n not in named and n not in sd.symbols and n not in text
        ]
        assert not stale, f'{sd.label} still carries unreferenced transients: {stale}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
