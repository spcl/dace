# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests ``add_mapped_tasklet``'s precomputed scope symbol table.

The table is the caller's way out of walking every descriptor's free symbols once per mapped
tasklet. It has to be a pure saving -- the same graph as without it -- and it has to be the table
propagation actually reads, which the last case shows by handing over one that is missing a name.
"""

import copy
from typing import Dict, List, Optional

import dace
from dace import dtypes
from dace.memlet import Memlet
from dace.sdfg.state import sdfg_scope_symbols

N = dace.symbol('N', dace.int64)
M = dace.symbol('M', dace.int64)


def build(name: str, scope_symbols: Optional[Dict[str, dtypes.typeclass]], precompute: bool = False) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [N, M], dace.float64)
    sdfg.add_array('b', [N, M], dace.float64)
    if precompute:
        scope_symbols = sdfg_scope_symbols(sdfg)
    sdfg.add_state().add_mapped_tasklet('scale', {
        'i': '0:N',
        'j': '2:M - 2'
    }, {'__in': Memlet('a[i, j]')},
                                        '__out = __in * 2.0', {'__out': Memlet('b[i, j]')},
                                        external_edges=True,
                                        scope_symbols=scope_symbols)
    return sdfg


def outer_memlets(sdfg: dace.SDFG) -> List[str]:
    state = sdfg.nodes()[0]
    return sorted(
        str(e.data) for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) for e in state.all_edges(n))


def test_a_precomputed_table_builds_the_same_graph():
    fresh = build('fresh', None)
    given = dace.SDFG('given')
    given.add_array('a', [N, M], dace.float64)
    given.add_array('b', [N, M], dace.float64)
    table = sdfg_scope_symbols(given)
    handed_over = copy.copy(table)
    given.add_state().add_mapped_tasklet('scale', {
        'i': '0:N',
        'j': '2:M - 2'
    }, {'__in': Memlet('a[i, j]')},
                                         '__out = __in * 2.0', {'__out': Memlet('b[i, j]')},
                                         external_edges=True,
                                         scope_symbols=table)

    assert outer_memlets(given) == outer_memlets(fresh)
    # The helper documents the table as read-only to it; a reader that edited it would poison every
    # later call the owner answers out of the same object.
    assert table == handed_over


def test_a_table_missing_a_symbol_is_the_table_propagation_reads():
    """A name the table cannot see is not a defined variable, so its extent is propagated out."""
    exact = build('exact', None)
    blind = build('blind', {})

    assert '2:M - 2' in outer_memlets(exact)[0]
    assert outer_memlets(blind) != outer_memlets(exact)


if __name__ == '__main__':
    test_a_precomputed_table_builds_the_same_graph()
    test_a_table_missing_a_symbol_is_the_table_propagation_reads()
