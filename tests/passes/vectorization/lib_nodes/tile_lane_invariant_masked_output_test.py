# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A lane-invariant tile op with a one-element output still reads its tile mask per lane.

The converter wires the full ``(W,)`` iteration mask to every op in a masked body, including one
whose operands are all Symbol / Scalar and whose output stays one element. The expansions wrote
``_c = _mask ? v : 0`` for it, reading the ``bool*`` mask as one bool: always true, so an
all-inactive tile kept ``v`` ("address of array '_tile_iter_mask' will always evaluate to 'true'",
97 sites in the vectorized CloudSC).
"""
import re

import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop, TileUnop


def masked_sdfg(name: str, node, widths, implementation: str) -> dace.SDFG:
    """``node`` gated by the tile mask ``M`` writes the one element ``F[0]``."""
    sdfg = dace.SDFG(f'{name}_{implementation}')
    sdfg.add_array('F', [1], dace.float64)
    sdfg.add_array('M', widths, dace.bool_)
    state = sdfg.add_state('main', is_start_block=True)
    node.implementation = implementation
    state.add_node(node)
    full = ', '.join(f'0:{w}' for w in widths)
    state.add_edge(state.add_read('M'), None, node, '_mask', dace.Memlet(f'M[{full}]'))
    state.add_edge(node, '_c', state.add_write('F'), None, dace.Memlet('F[0]'))
    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


def negate(widths):
    return TileUnop(name='neg', widths=widths, op='neg', kind_a='Symbol', expr_a='2.5', has_mask=True)


def add(widths):
    return TileBinop(name='add',
                     widths=widths,
                     op='+',
                     kind_a='Symbol',
                     kind_b='Symbol',
                     expr_a='2.0',
                     expr_b='0.5',
                     has_mask=True)


OPS = [('masked_unop', negate, -2.5), ('masked_binop', add, 2.5)]


@pytest.mark.parametrize('implementation', ['pure', 'scalar'])
@pytest.mark.parametrize('name,make_node,value', OPS)
def test_the_generated_code_reads_a_tile_mask_lane_by_lane(name, make_node, value, implementation):
    """The guard against reading the mask pointer as one bool, which compiles with only a warning."""
    code = masked_sdfg(f'{name}_code', make_node((2, )), (2, ), implementation).generate_code()[0].clean_code
    assert '_mask[__l0]' in code, code
    assert not re.search(r'\b(_mask|M)\s*\?', code), code


@pytest.mark.parametrize('lanes,active', [
    pytest.param((True, True), True, id='TT'),
    pytest.param((True, False), True, id='TF'),
    pytest.param((False, True), True, id='FT'),
    pytest.param((False, False), False, id='FF'),
    pytest.param((True, ), True, id='T'),
    pytest.param((False, ), False, id='F'),
])
@pytest.mark.parametrize('implementation', ['pure', 'scalar'])
@pytest.mark.parametrize('name,make_node,value', OPS)
def test_the_one_element_output_keeps_its_value_iff_some_lane_is_active(name, make_node, value, implementation, lanes,
                                                                        active):
    widths = (len(lanes), )
    tag = ''.join('T' if lane else 'F' for lane in lanes)
    sdfg = masked_sdfg(f'{name}_{tag}', make_node(widths), widths, implementation)
    got = np.full(1, 7.0)
    sdfg(F=got, M=np.array(lanes, dtype=bool))
    assert got[0] == (value if active else 0.0), (lanes, got[0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
