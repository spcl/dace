# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Is a loop-carried symbol a GF(2)-affine recurrence whose linear part does not read the loop's data?

A CRC chains its word through every byte::

    for b in range(N):
        cur = data[b]
        for bit in range(8):
            if (c ^ cur) & 1:
                c = (c >> 1) ^ poly
            else:
                c >>= 1
            cur >>= 1

The state is a full machine word, so :mod:`~dace.transformation.passes.analysis.finite_state` cannot enumerate it.
It composes cheaply anyway when the step is affine over GF(2)::

    f(s, d) = A s (+) c(d)            (+) is XOR, A s the GF(2) matrix-vector product

with ``A`` independent of the data ``d``. A block of ``l`` iterations is then ``x -> A^l x (+) c_block``: ``A^l``
follows from ``A`` by repeated squaring, and ``c_block`` is the block run from the zero state. CRC and LFSR
feedback networks are this shape: the network is fixed and only the injected word varies.

WHAT IS PROVEN. The step region runs symbolically
(:mod:`~dace.transformation.passes.analysis.smt_region`) with C semantics on the state's own type, its
per-iteration reads as free variables ``d`` and its loop-invariant reads as free parameters ``p``. Running it from
the zero state and from each basis vector gives a candidate ``c(d, p)`` and columns ``A_j(p)``, the columns taken
with every datum set to zero. z3 then proves, for every state of the type, every datum and every parameter::

    f(s, d, p) = c(d, p) (+) XOR over the set bits j of s of A_j(p)

The identity makes ``f`` affine in ``s`` and its linear part independent of ``d``, so the evaluations that found
the candidate cannot have fitted a step that only agrees with it on the basis.
"""

from dataclasses import dataclass
from typing import Any

from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import smt_body, smt_dependence, smt_region
from dace.transformation.passes.analysis.finite_state import StepSlice, step_sdfg, step_slices


@dataclass(frozen=True, slots=True)
class Gf2State:
    """A loop's GF(2)-affine carried state.

    :ivar step: the state and the slice that computes it.
    :ivar bits: the width of the state's type, the size of the matrix ``A``.
    """

    step: StepSlice
    bits: int


def proves_equal(lhs: Any, rhs: Any) -> bool:
    """Whether z3 proves ``lhs == rhs`` for every value of their free variables."""
    solver = smt_dependence.bounded_solver()
    solver.add(lhs != rhs)
    return solver.check() == smt_body.z3.unsat


def proves_gf2_affine(loop: LoopRegion, step: StepSlice) -> int | None:
    """The state's width when ``step`` is GF(2)-affine with a data-independent linear part, else ``None``."""
    if not smt_body.HAS_Z3:
        return None
    sdfg = step_sdfg(loop, step)
    kind = smt_region.value_type(sdfg, step.symbol)
    if kind is None:
        return None
    bits, signed = kind
    encoder = smt_region.RegionEncoder(sdfg, loop.loop_variable)
    width = encoder.width
    raw = smt_body.bitvec("carried_state", width)
    iterate = smt_body.bitvec("iterate", width)
    after = encoder.region(sdfg, {step.symbol: smt_body.fit(raw, bits, signed, width), loop.loop_variable: iterate})
    if after is None or step.symbol not in after:
        return None
    actual = after[step.symbol]
    data = [(variable, smt_body.constant(0, width)) for variable in encoder.data_variables()]
    data.append((iterate, smt_body.constant(0, width)))

    def from_state(value: int) -> Any:
        return smt_body.z3.substitute(actual, (raw, smt_body.constant(value, width)))

    constant = from_state(0)
    predicted = constant
    for bit in range(bits):
        column = smt_body.z3.simplify(smt_body.z3.substitute(from_state(1 << bit) ^ constant, *data))
        is_set = smt_body.z3.Extract(bit, bit, raw) == smt_body.z3.BitVecVal(1, 1)
        predicted = predicted ^ smt_body.z3.If(is_set, column, smt_body.constant(0, width))
    return bits if proves_equal(actual, predicted) else None


def gf2_state(loop: LoopRegion) -> Gf2State | None:
    """The GF(2)-affine carried state of ``loop``, or ``None``."""
    for step in step_slices(loop):
        bits = proves_gf2_affine(loop, step)
        if bits is not None:
            return Gf2State(step, bits)
    return None
