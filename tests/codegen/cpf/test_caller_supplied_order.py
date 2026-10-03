# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A caller supplies the CPF entry point's parameter ORDER.

CPF's own order is ``SDFG.arglist()``: every array sorted by name, then every scalar sorted by
name. That is a fine order and it is not every caller's. A calling convention fixed somewhere else
-- an ABI that reserves a trailing scratch pair, so a POINTER sits behind the scalars -- names an
order no name sort reaches, and the consumer then has two choices: rewrite the rendered signature
after the fact, or ask for the order up front.

The first is what a consumer did, and a post-hoc rewrite of a signature is a second implementation
of the splitting rules :func:`~dace.codegen.cpf.qualify_readonly_pointers` already owns -- two
copies that drift, with a symbol the caller links and calls with its arguments shifted at the end
of the drift. Nothing in the body depends on the parameter order, and the unit is self-contained
(no prototype, no header), so the entry is declared once and the renderer is the right place to
place it.

What is asserted here is the whole contract: the order lands, the qualifiers survive it, the result
BUILDS and RUNS and reproduces the numbers when called that way, ``Rendering.arguments`` reports
what the text actually takes, and an order that is not the entry's parameter set is refused rather
than resolved by dropping or inventing a parameter.
"""
import numpy as np
import pytest

import dace
from dace.codegen.cpf import entry_parameter_name, render

from tests.codegen.cpf.conftest import assert_standalone, build_standalone, call_standalone

N = dace.symbol('N')

#: The kernel: ``dst = src * factor``. Its arglist is ``dst, src`` (arrays by name) then
#: ``N, factor`` (scalars by name).
ARGLIST_ORDER = ('dst', 'src', 'N', 'factor')

#: The caller's order. ``src`` is a POINTER placed AFTER both scalars, which is the shape a
#: reserved trailing argument forces and which no sort of the names can produce -- so a test that
#: only permuted within the two groups would pass against a renderer that still name-sorted.
CALLER_ORDER = ('dst', 'N', 'factor', 'src')


@dace.program
def cpf_order_scale(src: dace.float64[N], dst: dace.float64[N], factor: dace.float64):
    dst[:] = src * factor


def rendered(name: str, language: str, order=None):
    sdfg = cpf_order_scale.to_sdfg(simplify=True)
    sdfg.name = name
    return render(sdfg, language=language, order=order)


def entry_parameters(code: str, name: str):
    """The entry point's declared parameters, in the order the text takes them."""
    opened = code.index(f'void {name}(') + len(f'void {name}(')
    closed = code.index(')', opened)
    return [p.strip() for p in code[opened:closed].split(',')]


def test_no_order_keeps_cpfs_own_and_reports_it():
    """The default is unchanged, and ``arguments`` says so rather than leaving a consumer to
    re-derive the order from the arglist and hope the two agree."""
    result = rendered('cpf_order_default', 'c++')
    assert result.arguments == ARGLIST_ORDER
    assert tuple(result.arguments) == tuple(result.sdfg.arglist())
    names = [entry_parameter_name(p) for p in entry_parameters(result.code, 'cpf_order_default')]
    assert tuple(names) == ARGLIST_ORDER


@pytest.mark.parametrize('language', ['c++', 'c'])
def test_the_rendered_signature_takes_the_caller_s_order(language):
    """The order lands in the text, including the pointer placed behind the scalars."""
    result = rendered('cpf_order_asked', language, order=CALLER_ORDER)
    assert result.arguments == CALLER_ORDER
    names = [entry_parameter_name(p) for p in entry_parameters(result.code, 'cpf_order_asked')]
    assert tuple(names) == CALLER_ORDER
    assert_standalone(result.code, 'cpf_order_asked', language=language)


def test_the_qualifiers_survive_the_move():
    """Only the ORDER changes. ``src`` is read-only, so it keeps the ``const`` the qualifier pass
    gave it even though it moved to the end -- a reorder that re-spelled declarations instead of
    moving them would be free to lose it, and a binding published from the written-set would then
    describe a signature saying the opposite."""
    result = rendered('cpf_order_qual', 'c++', order=CALLER_ORDER)
    declarations = {entry_parameter_name(p): p for p in entry_parameters(result.code, 'cpf_order_qual')}
    assert declarations['src'].startswith('const '), declarations['src']
    assert not declarations['dst'].startswith('const '), declarations['dst']


@pytest.mark.parametrize('language', ['c++', 'c'])
def test_a_reordered_unit_builds_and_reproduces_the_numbers(language):
    """The point of the whole feature: the unit still compiles in an empty directory and computes
    the same result when CALLED in the order it was asked for. A rendering whose text says one
    order and whose body assumed another links fine and returns nonsense."""
    result = rendered('cpf_order_run', language, order=CALLER_ORDER)
    library = build_standalone(result.code, 'cpf_order_run', language=language)

    n = 64
    src = np.arange(n, dtype=np.float64) + 1.0
    dst = np.zeros(n, dtype=np.float64)
    call_standalone(library, result.sdfg, {'src': src, 'dst': dst, 'N': n, 'factor': 2.5}, order=result.arguments)
    np.testing.assert_allclose(dst, src * 2.5, rtol=0.0, atol=0.0)


def test_an_order_that_is_not_the_parameter_set_is_refused():
    """Refused, not resolved. Dropping a name the entry declares or inventing one it does not
    produces a signature the caller links by name and calls with its arguments shifted, which no
    compiler catches across a rename -- so both directions raise and the message names both sets."""
    with pytest.raises(ValueError, match='cpf_order_short'):
        rendered('cpf_order_short', 'c++', order=('dst', 'src', 'N'))
    with pytest.raises(ValueError, match='cpf_order_extra'):
        rendered('cpf_order_extra', 'c++', order=(*ARGLIST_ORDER, 'nonesuch'))
    with pytest.raises(ValueError, match='cpf_order_wrong'):
        rendered('cpf_order_wrong', 'c++', order=('dst', 'src', 'N', 'FACTOR'))
