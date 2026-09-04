# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``sqrt``/``exp``/``log`` must reach ``dace::math``, whichever printer writes them.

A bare call binds to ``std::``, whose ``float``/``double``/``long double`` overloads are all equally
good for a 16-bit float -- ``dace::float16`` IS CUDA's ``half`` and converts non-explicitly to each
of them -- so nvcc rejects the translation unit as ambiguous. ``dace/math.h`` carries an exact
``dace::float16`` overload for exactly these names; qualifying the call is what reaches it.

The point of testing it here is that the same expression reaches C++ through TWO printers -- a
tasklet body through ``cppunparse``, a memlet subset or interstate assignment through
``dace.symbolic`` -- and the tasklet side was qualified while the symbolic side still wrote the
bare sympy name. One shared table (``mpr_lowering.RUNTIME_QUALIFIED_MATH``) now feeds both, and
these tests fail if either printer drifts off it.
"""
import numpy as np
import pytest
import sympy

import dace
from dace import mpr_lowering, symbolic
from dace.codegen import cppunparse

#: The unary names in the shared table -- ``fma`` is ternary and is exercised by its own tests.
UNARY = sorted(set(mpr_lowering.RUNTIME_QUALIFIED_MATH) - {'fma'})


@pytest.mark.parametrize('name', UNARY)
def test_both_printers_qualify_the_same_math_names(name):
    """The symbolic printer and the tasklet printer agree, name for name."""
    x = symbolic.symbol('x')
    expected = mpr_lowering.RUNTIME_QUALIFIED_MATH[name]
    assert expected in symbolic.symstr(sympy.Function(name)(x), cpp_mode=True, dialect=mpr_lowering.Dialect.RUNTIME)
    assert expected in cppunparse.py2cpp(f'b = {name}(a)')


def test_a_sympy_square_root_is_qualified_however_it_is_spelled():
    """``sqrt`` arrives two ways -- as ``Pow(x, 1/2)`` and as a call -- and both must qualify.

    The Pow path was already handled; the call path fell through to sympy's own string printer,
    which writes the bare name.
    """
    x = symbolic.symbol('x')
    runtime = mpr_lowering.Dialect.RUNTIME
    assert 'dace::math::sqrt' in symbolic.symstr(sympy.sqrt(x), cpp_mode=True, dialect=runtime)
    assert 'dace::math::sqrt' in symbolic.symstr(symbolic.pystr_to_symbolic('math.sqrt(x)'),
                                                 cpp_mode=True,
                                                 dialect=runtime)


def test_a_name_with_no_low_precision_overload_stays_bare():
    """Negative control. ``sin`` has no ``dace::float16`` overload in ``dace/math.h``, so qualifying
    it would move the same ambiguity one frame down into ``dace::math``'s template body rather than
    resolve it. A table that grew to cover every math name would pass every other test here."""
    x = symbolic.symbol('x')
    assert 'sin' not in mpr_lowering.RUNTIME_QUALIFIED_MATH
    assert 'dace::math::sin' not in symbolic.symstr(sympy.sin(x), cpp_mode=True, dialect=mpr_lowering.Dialect.RUNTIME)


@pytest.mark.parametrize('dialect', [mpr_lowering.Dialect.STANDALONE, mpr_lowering.Dialect.STANDALONE_C])
def test_the_standalone_dialects_still_get_the_standard_library(dialect):
    """MPR emits a translation unit with no DaCe headers at all, so a ``dace::`` name there does not
    compile. The standalone lowering runs first and must keep winning.

    The dialect is passed rather than left ambient because ``symstr`` is memoized on its arguments:
    an omitted dialect is resolved INSIDE the cached call, so every ambient-dialect caller shares
    one cache entry regardless of which dialect was active.
    """
    x = symbolic.symbol('x')
    for name in UNARY:
        rendered = symbolic.symstr(sympy.Function(name)(x), cpp_mode=True, dialect=dialect)
        assert 'dace::' not in rendered, rendered


def test_an_interstate_assignment_emits_the_qualified_call():
    """End to end through the symbolic printer: an interstate edge whose assignment calls ``sqrt``
    lands in the generated program as ``dace::math::sqrt``, and the program builds and runs."""
    sdfg = dace.SDFG('interstate_sqrt')
    sdfg.add_array('A', [2], dace.float64)
    sdfg.add_symbol('s', dace.float64)
    start = sdfg.add_state('start')
    body = sdfg.add_state('body')
    sdfg.add_edge(start, body, dace.InterstateEdge(assignments={'s': 'sqrt(9.0)'}))
    tasklet = body.add_tasklet('store', {}, {'out'}, 'out = s')
    write = body.add_write('A')
    body.add_edge(tasklet, 'out', write, None, dace.Memlet('A[0]'))

    code = sdfg.generate_code()[0].clean_code
    assert 'dace::math::sqrt' in code, code

    A = np.zeros(2, dtype=np.float64)
    sdfg(A=A)
    assert np.isclose(A[0], 3.0)
