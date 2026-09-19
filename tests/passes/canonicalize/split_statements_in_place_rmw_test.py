# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SplitStatements`` must refuse to replicate a body whose output is an in-place RMW.

Mechanism (1) of the pass (``_replicate_components``) clones a MapFission-blocking
``NestedSDFG`` once per independent output group. It never fires on a FIRST canonicalize of
a guarded kernel -- at ``prep`` the body is still a raw control-flow tree with no
``NestedSDFG`` at all. Canonicalization then MANUFACTURES the blocking shape itself:
``LoopToMap`` rewrites the loop into a Map whose ``ConditionalBlock`` body becomes a
``NestedSDFG``, and for an in-place kernel (``a[i] = a[i] + ...``) the same arrays land on
BOTH sides -- ``in=[a, b, d] out=[a, b]``. So a SECOND canonicalize walks straight into
mechanism (1) with an input/output-aliased body.

Two things were wrong there, and they compound:

* ``_output_dependency`` is blind to in-place RMW. An RMW output's own writer ``AccessNode``
  carries a name that is already an input, so the traversal stops before reading its
  in-edges and reports NO dependency -- ``a[i] = a[i] + d[i]`` and ``b[i] = b[i] * 2`` are
  declared independent even when the second reads what the first wrote, and the guard
  ``a[i] > b[i]`` that selects between them is not modelled at all.
* ``_split`` then built clones that could not be repaired. Each clone keeps ALL input
  connectors but only its own group's outputs, and the transient-flip that is supposed to
  neutralize the other groups' writes explicitly skips any array that is also an input
  connector -- so the write stayed, on a non-transient, with no output connector, and
  ``validate`` raised *"Data descriptor b is written to, but only given to nested SDFG as an
  input connector"*. ``SimplifyPass`` cannot clear it either: dead-dataflow elimination
  refuses to delete a write to a global.

Making it merely validate is the WRONG fix. Cloning re-evaluates every branch guard in
every clone while a sibling clone writes the array that guard reads, so the arms stop being
mutually exclusive: measured on :func:`guard_flip`, the if-arm's update to ``a`` flips
``a[i] > b[i]`` on 23 of 64 lanes and the else-arm's store to ``b`` then fires on if-arm
lanes. The pass must REFUSE -- the same in-place read-modify-write refusal
``_split_one_map`` already carries for the map path.
"""
import os

# ``to_sdfg`` lazily imports mpi4py; left to auto-init it can wedge the corpus import.
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.split_statements import SplitStatements

N = dace.symbol('N')


@dace.program
def rmw_pair(a: dace.float64[N], b: dace.float64[N], d: dace.float64[N]):
    """Two independent in-place updates under one guard -- the minimal blocking shape."""
    for i in range(N):
        if d[i] > 0.0:
            a[i] = a[i] + d[i]
            b[i] = b[i] * 2.0


@dace.program
def guard_flip(a: dace.float64[N], b: dace.float64[N], d: dace.float64[N]):
    """s2710's essential shape: each arm updates an operand of the guard that selects it."""
    for i in range(N):
        if a[i] > b[i]:
            a[i] = a[i] + d[i]
        else:
            b[i] = b[i] * 2.0


def guard_flip_reference(a, b, d):
    a, b = a.copy(), b.copy()
    for i in range(len(a)):
        if a[i] > b[i]:
            a[i] = a[i] + d[i]
        else:
            b[i] = b[i] * 2.0
    return a, b


def _inputs(n=64):
    """Inputs with a NEGATIVE ``d``, so the if-arm's update really can flip the guard."""
    rng = np.random.default_rng(0)
    a, b, d = rng.random(n), rng.random(n), -2.0 * rng.random(n)
    assert sum(1 for i in range(n) if a[i] > b[i] and not a[i] + d[i] > b[i]) > 0, 'no lane flips; test is vacuous'
    return a, b, d


def _canonicalize_twice(program):
    """First canonicalize (manufactures the aliased ``NestedSDFG``), then a second one."""
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, peel_limit=4, break_anti_dependence=True)
    sdfg.validate()
    canonicalize(sdfg, semantic_lifting=False, target='cpu')
    sdfg.validate()
    return sdfg


def test_double_canonicalize_of_a_guarded_rmw_pair_validates():
    """Without the refusal the second canonicalize raises ``InvalidSDFGNodeError``."""
    _canonicalize_twice(rmw_pair)


def test_double_canonicalize_preserves_values():
    """The real detector: a repair that only stops the crash gets WRONG numbers here."""
    a, b, d = _inputs()
    want_a, want_b = guard_flip_reference(a, b, d)
    sdfg = _canonicalize_twice(guard_flip)
    got_a, got_b = a.copy(), b.copy()
    sdfg.compile()(a=got_a, b=got_b, d=d.copy(), N=len(a))
    assert np.allclose(got_a, want_a)
    assert np.allclose(got_b, want_b)


def _shape(sdfg):
    """The graph, ignoring names.

    Not ``hash_sdfg``: that is name-sensitive, and canonicalization renames on EVERY run
    regardless of this bug -- a plain ``for i: a[i] = b[i] + d[i]`` relabels its state
    ``single_state_body_0 -> _1 -> _2``, and ``WCRToAugAssign`` mints a fresh private
    accumulator each time and abandons the previous descriptor (an orphan no node
    references). Both are pre-existing name churn in other passes. What must hold here is
    that the GRAPH stops moving: no further clone, no further global.
    """
    return (sorted(type(n).__name__ for n, _ in sdfg.all_nodes_recursive()),
            sorted(k for s in sdfg.all_sdfgs_recursive() for k, v in s.arrays.items() if not v.transient),
            sum(len(s.edges()) for s in sdfg.all_states()), sum(1 for _ in sdfg.all_states()))


def test_canonicalize_reaches_a_fixed_point():
    """Runs 3 and 4 must leave the graph exactly as run 2 left it."""
    sdfg = _canonicalize_twice(guard_flip)
    after_two = _shape(sdfg)
    for _ in range(2):
        canonicalize(sdfg, semantic_lifting=False, target='cpu')
        sdfg.validate()
        assert _shape(sdfg) == after_two


def test_split_statements_refuses_the_in_place_body():
    """The refusal is at the decision, not the rewrite: no groups, hence no mutation."""
    sdfg = guard_flip.to_sdfg(simplify=True)
    canonicalize(sdfg, peel_limit=4, break_anti_dependence=True)
    aliased = [(n, p) for n, p in sdfg.all_nodes_recursive()
               if isinstance(n, nodes.NestedSDFG) and any(c in n.in_connectors for c in n.out_connectors)]
    assert aliased, 'canonicalization no longer manufactures an input/output-aliased NestedSDFG'
    for node, state in aliased:
        assert SplitStatements._independent_output_groups(state, node) is None
    before = sdfg.hash_sdfg()
    SplitStatements().apply_pass(sdfg, {})
    assert sdfg.hash_sdfg() == before


@pytest.mark.parametrize('name', ['s2710_d_single'])
def test_tsvc_double_canonicalize_is_valid_and_correct(name):
    """The reported kernel, end to end against the numpy reference."""
    from tests.corpus.tsvc import tsvc
    from tests.corpus.tsvc.tsvc_numpy import REFERENCES

    kernel = tsvc.collect(name=name)[0]
    sdfg = tsvc.to_sdfg(kernel, tag='ss_rmw', simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4, break_anti_dependence=True)
    canonicalize(sdfg, validate=True, semantic_lifting=False, target='cpu')
    sdfg.validate()

    arrays, consts = tsvc.make_inputs(kernel, seed=1234)
    want = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[name](**want, **consts)
    got = {n: a.copy() for n, a in arrays.items()}
    sdfg.compile()(**got, **consts)
    for n, a in got.items():
        if isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.floating) and a.size:
            assert np.allclose(a, want[n], rtol=1e-9, atol=1e-9, equal_nan=True), f'{name}/{n} diverges'


if __name__ == '__main__':
    test_double_canonicalize_of_a_guarded_rmw_pair_validates()
    test_double_canonicalize_preserves_values()
    test_canonicalize_reaches_a_fixed_point()
    test_split_statements_refuses_the_in_place_body()
    test_tsvc_double_canonicalize_is_valid_and_correct('s2710_d_single')
