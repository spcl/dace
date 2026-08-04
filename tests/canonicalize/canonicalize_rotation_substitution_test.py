# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop-carried ROTATION becomes a shifted array read, and the loop becomes parallel.

A carried scalar overwritten every iteration with a loop-varying array element is a one-element
delay line: at iteration ``i`` it EQUALS ``b[i - stride]``. ``LoopToMap`` refuses such a loop
because a bare scalar has no ``a*i + b`` write subset, so the whole loop stays sequential over a
carry that is not a real dependence. Substituting the shifted read, deleting the update and peeling
the first iteration (the one the shifted read does not cover) leaves a DOALL body.

The assertions here are on the GENERATED CODE, not on a ``MapEntry`` count: a Map the backend never
turns into a parallel loop is not parallelism, and only the emitted ``#pragma omp parallel for``
proves the difference. Numerics are checked against a sequential reference in the same breath --
this rewrite's failure mode is a silent miscompile, so "parallel" and "correct" have to be asserted
together or neither means anything.

The shapes this rewrite must REFUSE live in ``canonicalize_rotation_false_positive_test.py``.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.induction_variable_substitution import LoopCarriedRotationSubstitution
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.dataflow import TrivialTaskletElimination

N = dace.symbol('N')
_LEN = 64


@dace.program
def rotate_one_deep(a: dace.float64[N], b: dace.float64[N]):
    x = b[N - 1]
    for i in range(N):
        a[i] = (b[i] + x) * 0.5
        x = b[i]


@dace.program
def rotate_two_deep(a: dace.float64[N], b: dace.float64[N]):
    x = b[N - 1]
    y = b[N - 2]
    for i in range(N):
        a[i] = (b[i] + x + y) * 0.333
        y = x
        x = b[i]


@dace.program
def accumulate(a: dace.float64[N], b: dace.float64[N]):
    x = 0.0
    for i in range(N):
        a[i] = (b[i] + x) * 0.5
        x = x + b[i]  # RHS names x -> an accumulation, no shifted-read closed form


def reference_one_deep(b):
    n = len(b)
    out, x = np.empty(n), b[n - 1]
    for i in range(n):
        out[i] = (b[i] + x) * 0.5
        x = b[i]
    return out


def reference_two_deep(b):
    n = len(b)
    out, x, y = np.empty(n), b[n - 1], b[n - 2]
    for i in range(n):
        out[i] = (b[i] + x + y) * 0.333
        y = x
        x = b[i]
    return out


def _canonicalized(program, label):
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = label
    canonicalize(sdfg, validate=True, peel_limit=4)
    return sdfg


def _generated_code(sdfg) -> str:
    return '\n'.join(obj.clean_code for obj in sdfg.generate_code())


def _structure(sdfg) -> str:
    """A structural summary of ``sdfg``: any node, edge, memlet or loop-bound change is a diff.

    Compared as text rather than hashed, so a failure shows WHAT moved instead of that something
    did. Built in traversal order, never from a set, so re-running it is not a coin flip.
    """
    lines = []
    for state in sdfg.all_states():
        lines.append(f'STATE {state.label}')
        ids = {n: i for i, n in enumerate(state.nodes())}
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                lines.append(f'  N{ids[node]} access {node.data}')
            elif isinstance(node, nodes.Tasklet):
                lines.append(f'  N{ids[node]} tasklet {node.code.as_string}')
            else:
                lines.append(f'  N{ids[node]} {type(node).__name__}')
        for e in state.edges():
            lines.append(f'  E N{ids[e.src]}:{e.src_conn} -> N{ids[e.dst]}:{e.dst_conn} {e.data}')
    for cfg in sdfg.all_control_flow_regions():
        lines.append(f'CFG {cfg.label} {[b.label for b in cfg.nodes()]}')
        for e in cfg.edges():
            lines.append(f'  IE {e.src.label} -> {e.dst.label} {e.data.condition.as_string} {e.data.assignments}')
    return '\n'.join(lines)


def _apply_rotation_alone(program, label):
    """``program`` with only the copy-tasklet cleanup the rotation pass expects, then the pass.

    ``TrivialTaskletElimination`` is what turns the frontend's ``x = <copy tasklet>(b[i])`` into the
    bare ``b[i] -> x[0]`` copy edge the rewrite matches; the full pipeline runs it several stages
    earlier. Applying just those two isolates the pass from everything else the pipeline does.
    """
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = label
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    before = _structure(sdfg)
    applied = LoopCarriedRotationSubstitution(peel_limit=4).apply_pass(sdfg, {})
    return sdfg, applied, before


# -- the rewrite lands: parallel code, identical numbers -----------------------------------


def test_one_deep_rotation_parallelizes():
    sdfg = _canonicalized(rotate_one_deep, 'rotate_one_deep_par')
    assert '#pragma omp parallel for' in _generated_code(sdfg), 'the delay line still blocks parallelization'

    rng = np.random.default_rng(2718)
    b = rng.random(_LEN)
    got = np.zeros(_LEN)
    sdfg.compile()(a=got, b=b.copy(), N=_LEN)
    assert np.allclose(got, reference_one_deep(b), rtol=0, atol=0), 'the shifted read changed the values'


def test_two_deep_rotation_parallelizes():
    """``x == b[i-1]`` and ``y == b[i-2]``: the outer stage only becomes a rotation after the inner
    one is substituted, and each stage costs one peeled iteration."""
    sdfg = _canonicalized(rotate_two_deep, 'rotate_two_deep_par')
    assert '#pragma omp parallel for' in _generated_code(sdfg), 'the two-stage delay line still blocks it'

    rng = np.random.default_rng(2718)
    b = rng.random(_LEN)
    got = np.zeros(_LEN)
    sdfg.compile()(a=got, b=b.copy(), N=_LEN)
    assert np.allclose(got, reference_two_deep(b), rtol=0, atol=0), 'a carried stage was shifted by the wrong amount'


# -- the pass itself: fires once, refuses an accumulation, and is a fixed point -------------


def test_pass_applies_once_per_stage():
    _sdfg, applied, _before = _apply_rotation_alone(rotate_one_deep, 'rotate_one_deep_unit')
    assert applied == 1, f'expected exactly one rotation substitution, got {applied}'
    _sdfg2, applied2, _b2 = _apply_rotation_alone(rotate_two_deep, 'rotate_two_deep_unit')
    assert applied2 == 2, f'a two-stage delay line needs two substitutions, got {applied2}'


def test_accumulation_is_refused_without_mutating():
    """``x = x + b[i]`` reads the scalar it writes, so no shifted read equals it. Refusing has to
    leave the SDFG untouched -- a pass that half-applies before declining is worse than one that
    applies wrongly, because nothing downstream is looking for the damage."""
    sdfg, applied, before = _apply_rotation_alone(accumulate, 'accumulate_unit')
    assert applied is None, 'an accumulation was rewritten as a rotation'
    assert _structure(sdfg) == before, 'the pass mutated the SDFG on a shape it refused'


def test_rotation_substitution_is_idempotent():
    """Re-running the pass on its own output must find nothing: the carry it removes is gone, and
    the shifted read it leaves behind is not itself a delay line."""
    sdfg = _canonicalized(rotate_two_deep, 'rotate_two_deep_idem')
    before = _structure(sdfg)
    assert LoopCarriedRotationSubstitution(peel_limit=4).apply_pass(sdfg, {}) is None, 'the pass re-fired on its output'
    assert _structure(sdfg) == before, 'a re-run mutated an already-substituted SDFG'


def test_peel_limit_zero_disables_the_rewrite():
    """The shifted read is wrong on the first iteration, so the rewrite cannot happen without the
    peel that removes that iteration. With no peel budget it must decline, not substitute anyway."""
    sdfg = rotate_one_deep.to_sdfg(simplify=True)
    sdfg.name = 'rotate_one_deep_peel0'
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    before = _structure(sdfg)
    assert LoopCarriedRotationSubstitution(peel_limit=0).apply_pass(sdfg, {}) is None
    assert _structure(sdfg) == before, 'the pass mutated the SDFG with no peel budget'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
