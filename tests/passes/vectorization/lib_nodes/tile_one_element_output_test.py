# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A lane-invariant tile op may write ONE element of a larger array, so its output kind reads the memlet.

CloudSC's ``imelt[4] = -99`` lowers to a Symbol-operand ``TileUnop`` writing ``imelt[4]`` of an
``int[5]``. The memlet moves one element, so codegen binds ``_c`` by value (``int _c;``), but the
expansions judged the output by its DESCRIPTOR, saw an array, and walked the lane loop
``_c[__l0] = ...``: "subscripted value is not an array, pointer, or vector".
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop, TileUnop
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')


def element_write_sdfg(name: str, node, implementation: str) -> dace.SDFG:
    """``node`` (lane-invariant operands, 2-lane tile) writes ``F[4]`` of an ``int32[5]``."""
    sdfg = dace.SDFG(f'{name}_{implementation}')
    sdfg.add_array('F', [5], dace.int32)
    state = sdfg.add_state('main', is_start_block=True)
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(node, '_c', state.add_write('F'), None, dace.Memlet('F[4]'))
    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


def negate_99():
    return TileUnop(name='neg', widths=(2, ), op='neg', kind_a='Symbol', expr_a='99')


def add_40_2():
    return TileBinop(name='add', widths=(2, ), op='+', kind_a='Symbol', kind_b='Symbol', expr_a='40', expr_b='2')


@pytest.mark.parametrize('implementation', ['pure', 'scalar'])
@pytest.mark.parametrize('name,make_node,want', [('one_element_unop', negate_99, -99),
                                                 ('one_element_binop', add_40_2, 42)])
def test_a_lane_invariant_op_writes_exactly_the_one_element_its_memlet_names(name, make_node, want, implementation):
    sdfg = element_write_sdfg(name, make_node(), implementation)
    got = np.full(5, 7, dtype=np.int32)
    sdfg(F=got)
    np.testing.assert_array_equal(got, [7, 7, 7, 7, want])


@dace.program
def flag_every_column(a: dace.float64[N], out: dace.float64[N], flag: dace.int32[5]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            f >> flag[4]
            f = -99
        out[i] = a[i] * 2.0


def test_a_vectorized_map_storing_a_negated_literal_into_one_element_stores_it():
    """CloudSC's ``imelt[4] = -99`` in miniature, main tile and masked tail both."""
    sdfg = flag_every_column.to_sdfg(simplify=True)
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=(2, ), target_isa='SCALAR', remainder_strategy='masked_tail',
                        validate_all=True)).apply_pass(sdfg, {})
    stores = [(n.has_mask, str(e.data)) for n, state in sdfg.all_nodes_recursive() if isinstance(n, TileUnop)
              for e in state.out_edges(n)]
    assert sorted(stores) == [(False, 'flag[4]'), (True, 'flag[4]')], stores
    a = np.random.rand(7)
    out = np.zeros(7)
    flag = np.zeros(5, dtype=np.int32)
    sdfg(a=a, out=out, flag=flag, N=7)
    np.testing.assert_array_equal(flag, [0, 0, 0, 0, -99])
    np.testing.assert_array_equal(out, 2.0 * a)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
