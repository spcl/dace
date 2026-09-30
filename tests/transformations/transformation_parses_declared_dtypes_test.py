# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" A transformation parses its SDFG's symbols at their declared dtypes. """
import dace
from dace import symbolic
from dace.transformation import transformation as xf


class ParseBoundInApply(xf.SingleStateTransformation):
    access = xf.PatternNode(dace.nodes.AccessNode)
    parsed = []

    @classmethod
    def expressions(cls):
        return [dace.sdfg.utils.node_path_graph(cls.access)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return not ParseBoundInApply.parsed

    def apply(self, graph, sdfg):
        ParseBoundInApply.parsed.append(symbolic.pystr_to_symbolic('N - 1'))


def test_a_transformation_parses_its_sdfg_symbols_at_their_declared_dtype():
    sdfg = dace.SDFG('typed_bound')
    sdfg.add_symbol('N', dace.int64)
    for name in ('a', 'b'):
        sdfg.add_array(name, [dace.symbol('N', dace.int64)], dace.float64)
    state = sdfg.add_state()
    state.add_nedge(state.add_access('a'), state.add_access('b'), dace.Memlet('a[0:N]'))
    ParseBoundInApply.parsed.clear()
    assert sdfg.apply_transformations(ParseBoundInApply) == 1
    assert {s.dtype for s in ParseBoundInApply.parsed[0].free_symbols} == {dace.int64}


if __name__ == '__main__':
    test_a_transformation_parses_its_sdfg_symbols_at_their_declared_dtype()
