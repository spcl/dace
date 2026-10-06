# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A symbol re-parsed from a string property must carry the dtype its SDFG declares.

Loop bounds live in string-backed properties, so recovering one re-parses it. Minting the leaf at
``DEFAULT_SYMBOL_TYPE`` puts a SECOND symbol of that name in a graph whose descriptors carry the
declared one -- and a symbol's dtype is part of its identity, so ``N - N`` stops cancelling. Every
proof that compares a bound against a shape then fails on a well-formed program: the 2-D arg-reduce
match asks whether the nest covers the whole array and gets ``-LEN_2D + LEN_2D``.
"""
import numpy as np

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import LoopRegion
from dace.transformation import transformation
from dace.transformation.passes.pattern_matching import match_patterns
from dace.transformation.passes.canonicalize import pipeline as canon

N = dace.symbol('N', dtype=dace.int64)


def test_a_two_dimensional_argmax_lifts():
    """TSVC s3110: the nest covers the whole array, so a flat arg-reduce is equivalent."""

    @dace.program
    def argmax_2d(aa: dace.float64[N, N], bb: dace.float64[1]):
        maxv = aa[0, 0]
        xindex = 0
        yindex = 0
        for i in range(N):
            for j in range(N):
                if aa[i, j] > maxv:
                    maxv = aa[i, j]
                    xindex = i
                    yindex = j
        bb[0] = maxv + float(xindex) + float(yindex)

    sdfg = argmax_2d.to_sdfg(simplify=False)
    canon.canonicalize(sdfg)
    libs = [
        type(n).__name__ for g in sdfg.all_sdfgs_recursive() for st in g.states() for n in st.nodes()
        if isinstance(n, nodes.LibraryNode)
    ]
    loops = sum(1 for g in sdfg.all_sdfgs_recursive() for b in g.all_control_flow_regions(recursive=True)
                if isinstance(b, LoopRegion))
    assert 'ArgReduce' in libs, f'the 2-D argmax stayed sequential (libs={libs}, loops={loops})'
    assert loops == 0, f'a sequential loop survived (loops={loops})'

    n = 48
    rng = np.random.default_rng(7)
    aa = rng.random((n, n))
    flat = int(np.argmax(aa))
    want = aa.max() + float(flat // n) + float(flat % n)
    got = np.zeros(1)
    sdfg(aa=aa, bb=got, N=n)
    assert np.isclose(got[0], want), f'{got[0]} != {want}'


class ReadsNAtTheDeclaredDtype(transformation.SingleStateTransformation):
    """Applicable only when ``N``, read from text, comes out at the dtype the SDFG declares."""

    access = transformation.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.access)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return symbolic.pystr_to_symbolic('N').dtype == sdfg.symbols['N']

    def apply(self, graph, sdfg):
        return None


def test_a_predicate_reads_names_at_the_dtype_the_sdfg_declares():
    """A predicate that decides on a re-parsed name decides as the application will: a match the application
    cannot reproduce is applied forever by a repeated pass."""
    sdfg = dace.SDFG('predicate_declared_dtype')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_state().add_read('A')

    assert len(list(match_patterns(sdfg, [ReadsNAtTheDeclaredDtype]))) == 1


if __name__ == '__main__':
    test_a_two_dimensional_argmax_lifts()
    test_a_predicate_reads_names_at_the_dtype_the_sdfg_declares()
