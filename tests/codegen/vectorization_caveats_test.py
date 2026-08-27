# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Codegen shapes that decide whether a host compiler will vectorize the loop DaCe emits.

Each test here pins one finding from the vectorization analysis of TSVC-2 and CloudSC ("How Well Do
Compilers Vectorize?", Bonsall and Budanaz). Every finding was a case where the C++ written by hand
vectorized and the C++ DaCe emitted for the SAME kernel did not, and in each case the blocker was a
property of the emitted text rather than of the algorithm:

* a reduction accumulator that DaCe reloaded from, and stored back to, its global array on EVERY
  iteration. GCC then sees a complex access pattern and gives up (s313, vdotr);
* a single-element copy emitted as a ``dace::CopyND`` call, which lowers to a memcpy and optimizes
  differently from the assignment it actually is (s314);
* a loop-carried induction variable left as a chain of copies, which GCC can only fold by
  REASSOCIATING floating-point additions -- so it vectorizes only under
  ``-funsafe-math-optimizations`` (s453);
* an index expression putting a runtime product on the INNERMOST loop variable, which defeats both
  vectorizers on a stride they cannot prove constant (CloudSC lu_solver).

The last two are why these are assertions and not compiler flags. The analysis reached s453 by
enabling ``-fassociative-math``; that is not a fix, it is permission to compute a different sum.
Canonicalization closes the induction variable instead -- the emitted body carries no accumulator at
all, so there is nothing left to reassociate -- and the loop vectorizes with FP semantics intact.
These tests state the shape, so a change that reintroduces the copies fails here rather than being
paid for later in an unsafe-math flag.

Both renderers are checked. MPR emits a self-contained unit through the same readable generator, and
a caveat that reappears in only one of them is still a caveat.
"""
import re
from typing import List, Tuple

import pytest

import dace
from dace.codegen.mpr import mpr
from dace.config import set_temporary
from dace.transformation.passes.canonicalize import canonicalize

from tests.corpus.tsvc import tsvc

#: The two renderings under test: DaCe's readable CPU generator, and the standalone MPR unit.
RENDERERS = ('readable', 'mpr')

NCLV = dace.symbol('NCLV', dtype=dace.int64)
KLON = dace.symbol('KLON', dtype=dace.int64)


@dace.program
def lu_solver_nest(zqlhs: dace.float64[NCLV, NCLV, KLON]):
    """CloudSC's LU-solver elimination nest, in the layout that fixed it.

    The porting bug the analysis found was this array declared ``[KLON, NCLV, NCLV]`` while ``jl``
    -- the innermost loop -- indexed its FIRST axis, so the emitted subscript multiplied ``jl`` by
    ``NCLV * NCLV`` and neither vectorizer could prove the stride. With ``jl`` on the last axis the
    innermost access is unit-stride and the kernel ran 8.4x faster.
    """
    for jn in range(NCLV - 1):
        for jm in range(jn + 1, NCLV):
            for jl in range(KLON):
                zqlhs[jn, jm, jl] = zqlhs[jn, jm, jl] / zqlhs[jn, jn, jl]


def canonicalized(name: str) -> dace.SDFG:
    """One TSVC-2 kernel, parsed and canonicalized for the CPU."""
    kernel = next(k for k in tsvc.collect() if k.name == name)
    sdfg = tsvc.to_sdfg(kernel, 'caveat', simplify=True)
    canonicalize(sdfg, validate=False, validate_all=False)
    return sdfg


def emitted(sdfg: dace.SDFG, renderer: str) -> str:
    """The generated C++ for ``sdfg`` from one of the two renderers.

    The generator is NAMED rather than inherited: these assertions are about text only
    ``experimental_readable`` produces -- the legacy generator keeps connector locals and emits the
    ``dace::CopyND`` calls one of the tests is here to forbid.
    """
    if renderer == 'mpr':
        return mpr(sdfg)
    with set_temporary('compiler', 'cpu', 'implementation', value='experimental_readable'):
        return '\n'.join(obj.clean_code for obj in sdfg.generate_code())


def loop_bodies(code: str) -> List[str]:
    """The body of every ``for`` loop in ``code``, brace-matched, outermost first.

    A regex cannot do this: the bodies nest, and the assertions below are about what appears
    ANYWHERE under a loop header, including inside an inner one.
    """
    bodies: List[str] = []
    for match in re.finditer(r'\bfor\s*\(', code):
        opening = code.find('{', match.end())
        if opening < 0:
            continue
        depth = 0
        for index in range(opening, len(code)):
            if code[index] == '{':
                depth += 1
            elif code[index] == '}':
                depth -= 1
                if depth == 0:
                    bodies.append(code[opening + 1:index])
                    break
    return bodies


def index_function(code: str, array: str) -> Tuple[List[str], str]:
    """``<array>_idx``'s data dimensions and its subscript arithmetic.

    Returns the ``__dN`` parameters in declaration order -- so the LAST one is the fastest-varying
    axis -- and the expression they are combined into.
    """
    match = re.search(r'\b%s_idx\s*\(([^)]*)\)\s*\{\s*return\s+(.+?);\s*\}' % re.escape(array), code)
    assert match is not None, f'no index function emitted for {array}'
    return re.findall(r'\b__d\d+\b', match.group(1)), match.group(2)


@pytest.mark.parametrize('renderer', RENDERERS)
@pytest.mark.parametrize('kernel,accumulator', [('s313_d_single', 'dot'), ('vdotr_d_single', 'dot_out')])
def test_a_reduction_accumulator_never_travels_through_memory_inside_the_loop(kernel, accumulator, renderer):
    """The accumulator is staged in a local; its array is touched only outside the loop.

    This is the analysis' "DaCe improvement 1". The emitted form it measured reloaded
    ``dot[dot_idx(0)]`` at the top of every iteration and stored it back at the bottom, which is
    what GCC reported as a complex access pattern. Canonicalization privatizes the accumulator, so
    the array name must not appear under a loop header at all.
    """
    code = emitted(canonicalized(kernel), renderer)
    for body in loop_bodies(code):
        assert f'{accumulator}[' not in body, \
            f'{kernel}: {accumulator} is accessed through memory inside a loop:\n{body}'
    assert f'{accumulator}[' in code, f'{kernel}: {accumulator} is never written at all'
    assert 'reduction(+:' in code.replace(' ', ''), f'{kernel}: the accumulator did not become an OpenMP reduction'


@pytest.mark.parametrize('renderer', RENDERERS)
def test_a_single_element_read_is_an_assignment_and_not_a_copy_call(renderer):
    """s314's ``x = a[0]`` stays an assignment.

    DaCe used to emit its copy helper for every copy including a one-element one, and that lowers to
    a memcpy which the vectorizer treats differently from a load. The analysis got the loop to
    vectorize by writing the assignment by hand.
    """
    code = emitted(canonicalized('s314_d_single'), renderer)
    for call in ('CopyND', 'dace::copy', 'memcpy'):
        assert call not in code, f's314: a single-element access still goes through {call}:\n{code}'


@pytest.mark.parametrize('renderer', RENDERERS)
def test_an_induction_variable_is_closed_rather_than_carried(renderer):
    """s453's ``s += 2.0`` is folded into the iterator, so the loop needs no FP reassociation.

    The analysis could only vectorize this under ``-funsafe-math-optimizations``, whose
    ``-fassociative-math`` lets GCC fold the copies DaCe left around the accumulator. Closing the
    induction variable removes the accumulator instead: the body must contain no self-update, and
    the loop must be marked parallel -- which is only legal because the carried dependence is gone.
    """
    code = emitted(canonicalized('s453_d_single'), renderer)
    bodies = loop_bodies(code)
    assert bodies, 's453: no loop was emitted at all'
    for body in bodies:
        assert not re.search(r'\bs\s*=\s*(?:\(\s*)?s\b', body), f's453: the accumulator is still carried:\n{body}'
    assert '#pragma omp parallel for' in code, 's453: the loop is not parallel, so the dependence survived'


@pytest.mark.parametrize('renderer', RENDERERS)
def test_the_innermost_loop_of_a_solver_nest_indexes_with_unit_stride(renderer):
    """The innermost loop variable lands on the last array axis, with no runtime factor on it.

    CloudSC's LU solver was 2x slower than Fortran because a row-major/column-major mix-up put
    ``jl`` on the outermost axis, so its subscript carried ``NCLV * NCLV`` -- a stride neither
    vectorizer could prove constant inside the hottest loop. The check is on the emitted index
    function: the last data dimension must appear as a bare term.
    """
    sdfg = lu_solver_nest.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=False, validate_all=False)
    code = emitted(sdfg, renderer)
    dimensions, expression = index_function(code, 'zqlhs')
    assert len(dimensions) == 3, f'zqlhs should be indexed by three axes, got {dimensions}'
    last = dimensions[-1]
    assert not re.search(r'\*\s*%s\b|\b%s\s*\*' % (last, last), expression), \
        f'zqlhs: the fastest-varying index {last} carries a stride factor: {expression}'
    assert re.search(r'\b%s\b' % last, expression), f'zqlhs: {last} does not reach the subscript: {expression}'
    # And the innermost loop is the one that supplies it. The call passes the data indices first and
    # the shape symbols after, so only the leading len(dimensions) arguments are axes.
    innermost = loop_bodies(code)[-1]
    call = re.search(r'zqlhs_idx\(([^)]*)\)', innermost)
    assert call is not None, f'zqlhs is not indexed in the innermost loop body:\n{innermost}'
    axes = [argument.strip() for argument in call.group(1).split(',')][:len(dimensions)]
    iterators = re.findall(r'\b_loop_it_\d+\b', innermost) or re.findall(r'\b_loop_it_\d+\b', code)
    assert axes[-1] == iterators[-1], \
        f'the innermost loop variable {iterators[-1]} is not on the last axis: zqlhs_idx({call.group(1)})'
