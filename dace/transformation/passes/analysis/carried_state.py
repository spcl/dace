# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Is a loop-carried scalar a GF(2)-affine recurrence whose linear part is loop-invariant?

A loop whose iterations chain through one scalar is sequential by construction::

    state = f(state, data[i])

No subset test can break that: :mod:`~dace.transformation.passes.analysis.smt_dependence` reasons
about WHICH ELEMENTS an iteration touches, and here the dependence is on the VALUE. The loop is
still parallelizable when ``f`` is a monoid action -- function composition is associative, so a
chain of transitions is a scan -- but only when composing two transitions is cheaper than running
the iterations they stand for.

This module answers the one case where that holds, and it is the case a static test decides::

    state' = A * state (+) c(data)      over GF(2): (+) is XOR, * is GF(2) matrix-vector

``A`` LOOP-INVARIANT (it does not read the loop's data) is what makes the lift free: ``A^len`` is a
compile-time constant, so a block of ``len`` iterations needs only its own ``c``, which the block
gets by running from the ZERO state at exactly the sequential per-element cost. Total work
``N + P * compose`` for a ``P``-way split. When ``A`` reads the data instead, building the block
matrix costs a full matrix per element and the lift loses before it starts -- which is why this
module refuses it rather than describing it. CRC and LFSR are the shape that wins: the feedback
network is fixed and only the injected word varies.

ONE ENCODER, USED TWICE. Extraction is not a solver search. With the body straight-line
(:class:`~dace.transformation.passes.parallelization_prep.ShortLoopUnroll` flattens the bit loop
first), ``c = f(0, d)`` and column ``j`` of ``A`` is ``f(e_j, d) (+) c`` -- ``width + 1``
evaluations of :mod:`~dace.transformation.passes.analysis.smt_body`, no solver at all. Evaluation
alone would be INADMISSIBLE: it fits a linear map through some points and says nothing about the
rest. So the same encoder then discharges, under a quantifier, that the candidate is exact for
every state and every datum. Milliseconds, and it is what makes the answer a proof.

PROOF OBLIGATION LEFT TO THE CALLER. ``state_bits`` narrows the claim to states below ``2 **
state_bits``, because that is the range a masking body keeps its state in and quantifying over the
full encoding width would refuse valid CRCs. The recurrence's INITIAL value comes from outside the
loop, so a caller that lifts on this verdict must also know the seed fits.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from dace.transformation.passes.analysis import smt_body


@dataclass(frozen=True, slots=True)
class Gf2Linear:
    """The loop-invariant linear part of a carried-state recurrence, over GF(2).

    Column-mask representation: ``columns[j]`` is the image of the basis vector ``e_j``, so the map
    is ``x -> XOR of columns[j] over the set bits j of x``. Composition is then a handful of XORs
    per column rather than a nested loop over a bit matrix.

    :ivar columns: image of each basis vector, ``state_bits`` of them.
    :ivar state_bits: width of the state this map acts on.
    """
    columns: Tuple[int, ...]
    state_bits: int

    def apply(self, x: int) -> int:
        """Image of ``x`` under this map."""
        out = 0
        for j, col in enumerate(self.columns):
            if (x >> j) & 1:
                out ^= col
        return out

    def then(self, outer: 'Gf2Linear') -> 'Gf2Linear':
        """This map followed by ``outer`` -- ``outer(self(x))``.

        Composition order matters and reads backwards from the iteration order: applying the
        recurrence twice is ``step.then(step)``, and for a non-symmetric ``A`` the other order is a
        different matrix.
        """
        return Gf2Linear(tuple(outer.apply(col) for col in self.columns), self.state_bits)

    def power(self, n: int) -> 'Gf2Linear':
        """This map applied ``n`` times, by repeated squaring: the ``A^len`` a block needs.

        ``n == 0`` is the identity, which is what an empty block should contribute.
        """
        if n < 0:
            raise ValueError(f'Gf2Linear.power needs a non-negative exponent, got {n}')
        result = identity(self.state_bits)
        base = self
        while n:
            if n & 1:
                result = result.then(base)
            base = base.then(base)
            n >>= 1
        return result


def identity(state_bits: int) -> Gf2Linear:
    """The identity map on ``state_bits`` bits."""
    return Gf2Linear(tuple(1 << j for j in range(state_bits)), state_bits)


def is_constant(term: Any, data: Sequence[Any]) -> Optional[int]:
    """The constant value of ``term``, or ``None`` when it depends on ``data``.

    Two steps, and the second is the one that counts: read a CANDIDATE off ``term`` with every
    datum pinned to zero, then prove the term equals that candidate for every datum. Pinning alone
    would call ``d & 0`` a constant and ``d`` a constant too.
    """
    candidate = z3_simplify(z3_substitute(term, [(d, smt_body.constant(0, d.size())) for d in data]))
    if not smt_body.z3.is_bv_value(candidate):
        return None
    if prove_equal(term, candidate) is not True:
        return None
    return candidate.as_long()


def z3_substitute(term: Any, pairs) -> Any:
    return smt_body.z3.substitute(term, *pairs) if pairs else term


def z3_simplify(term: Any) -> Any:
    return smt_body.z3.simplify(term)


def prove_equal(lhs: Any, rhs: Any, assumptions: Sequence[Any] = (), timeout_ms: int = 5000) -> Optional[bool]:
    """Whether ``lhs == rhs`` holds for every assignment satisfying ``assumptions``.

    ``True`` on a proof, ``False`` on a counterexample, ``None`` when the solver gives up -- an
    ``unknown`` is not a proof and must not be read as one.
    """
    solver = smt_body.z3.Solver()
    solver.set('timeout', timeout_ms)
    for a in assumptions:
        solver.add(a)
    solver.add(lhs != rhs)
    verdict = solver.check()
    if verdict == smt_body.z3.unsat:
        return True
    if verdict == smt_body.z3.sat:
        return False
    return None


def run_body(code: str, state_name: str, state_term: Any, data: Dict[str, Any], width: int) -> Optional[Any]:
    """The body's value for ``state_name``, starting from ``state_term``, or ``None``."""
    env = dict(data)
    env[state_name] = state_term
    after = smt_body.encode_body(code, env, width)
    if after is None:
        return None
    return after.get(state_name)


def extract_gf2_linear(code: str,
                       state_name: str,
                       data_names: Sequence[str],
                       state_bits: int,
                       width: int = smt_body.DEFAULT_WIDTH) -> Optional[Gf2Linear]:
    """The loop-invariant GF(2) linear part of ``code``'s update to ``state_name``.

    Returns ``None`` -- the only safe answer, and the answer for most bodies -- when the body is
    outside :mod:`smt_body`'s fragment, when the update is not GF(2)-affine in the state, or when
    the linear part reads the data. A caller cannot tell those apart, and does not need to: all
    three mean the lift is unavailable, and only the first is even a near miss.

    :param code: the tasklet body, straight-line Python after unrolling.
    :param state_name: the carried scalar, read and written by the body.
    :param data_names: the per-iteration inputs; the linear part must not depend on these.
    :param state_bits: the state's real width; the claim is quantified over states below
                       ``2 ** state_bits``, not over the whole encoding width.
    :param width: bitvector width to encode in. Must exceed anything the body's own masks produce.
    :returns: the proven linear part, or ``None``.
    """
    if not smt_body.HAS_Z3 or state_bits <= 0 or state_bits > width:
        return None

    data = {name: smt_body.bitvec(f'__d_{name}', width) for name in data_names}
    data_terms = tuple(data.values())

    # c = f(0, d): the part of the update the state does not reach.
    const_term = run_body(code, state_name, smt_body.constant(0, width), data, width)
    if const_term is None:
        return None

    # Column j = f(e_j, d) XOR c. Data-free is a PROOF here, not an inspection of the term.
    columns = []
    for j in range(state_bits):
        image = run_body(code, state_name, smt_body.constant(1 << j, width), data, width)
        if image is None:
            return None
        column = is_constant(z3_simplify(image ^ const_term), data_terms)
        if column is None or column >> state_bits:
            return None
        columns.append(column)
    candidate = Gf2Linear(tuple(columns), state_bits)

    # Admissibility. The candidate was fitted through state_bits + 1 points; nothing so far rules
    # out a body that agrees there and diverges everywhere else. Quantify over every state in range
    # and every datum.
    state = smt_body.bitvec('__carried_state', width)
    actual = run_body(code, state_name, state, data, width)
    if actual is None:
        return None
    predicted = const_term
    for j, column in enumerate(columns):
        bit = smt_body.z3.Extract(j, j, state) == smt_body.z3.BitVecVal(1, 1)
        predicted = predicted ^ smt_body.z3.If(bit, smt_body.constant(column, width), smt_body.constant(0, width))
    in_range = smt_body.z3.ULT(state, smt_body.constant(1 << state_bits, width)) if state_bits < width \
        else smt_body.z3.BoolVal(True)
    if prove_equal(actual, predicted, assumptions=(in_range, )) is not True:
        return None
    return candidate
