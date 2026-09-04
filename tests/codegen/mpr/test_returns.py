# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""How a program's return value reaches an MPR caller.

DaCe carries a return value in a container named ``__return`` (a single value) or ``__return_0``,
``__return_1``, ... (a returned tuple). MPR emits one plain entry point, so the only way a result
can leave it is through a parameter, and whether that works turns entirely on the DESCRIPTOR:

* an ``Array`` return is spelled ``T * __restrict__`` -- an out-parameter, which works,
* a ``Scalar`` return is spelled ``T`` -- a BY-VALUE parameter, so the rendering would compute the
  result into the callee's own copy and the caller would read whatever it passed in.

The second case is a wrong answer rather than a compile error, which is why the tests below assert
the refusal and not merely that something raised. The Python frontend widens a scalar return to a
one-element array before it gets here, so the refusal is reachable only from a hand-built SDFG --
which is exactly the case a compile error would never catch.
"""
import re

import numpy as np
import pytest

import dace
from dace.codegen.mpr import render as render_sdfg, return_containers

from tests.codegen.mpr.conftest import assert_standalone, build_standalone, call_standalone


def render(program, name: str):
    """``(sdfg, code)`` for a ``@dace.program``, rendered under its own entry-point name."""
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    rendering = render_sdfg(sdfg)
    return rendering.sdfg, rendering.code


def test_returned_array_is_an_out_parameter():
    """A returned array round-trips: the caller allocates, the entry point writes through."""

    @dace.program
    def returns_array(a: dace.float64[20]):
        return a + 1.0

    sdfg, code = render(returns_array, 'mpr_ret_array')
    assert '__return' in sdfg.arglist(), 'the return container left the entry signature'
    assert 'double * __restrict__ __return' in code, ('the return container is not a pointer parameter, so the '
                                                      'caller could not read the result')
    assert_standalone(code, 'mpr_ret_array')

    a = np.random.rand(20)
    result = np.zeros(20)
    call_standalone(build_standalone(code, 'mpr_ret_array'), sdfg, {'a': a, '__return': result})
    assert np.allclose(result, a + 1.0)


def test_returned_tuple_becomes_one_out_parameter_each():
    """A returned tuple is ``__return_0``, ``__return_1``: both must be present and both written."""

    @dace.program
    def returns_tuple(a: dace.float64[20]):
        return a + 1.0, a * 2.0

    sdfg, code = render(returns_tuple, 'mpr_ret_tuple')
    arglist = sdfg.arglist()
    assert '__return_0' in arglist and '__return_1' in arglist, f'a return container was dropped: {list(arglist)}'
    assert '__return' not in arglist, 'the single-value name must not appear alongside the tuple names'

    a = np.random.rand(20)
    first, second = np.zeros(20), np.zeros(20)
    call_standalone(build_standalone(code, 'mpr_ret_tuple'), sdfg, {'a': a, '__return_0': first, '__return_1': second})
    assert np.allclose(first, a + 1.0)
    assert np.allclose(second, a * 2.0)


def test_scalar_return_from_the_frontend_is_widened_to_an_array():
    """``np.sum(a)`` returns a scalar in Python but a one-element ARRAY in the SDFG.

    This is the reason the refusal below is not reachable from ordinary code, and asserting it here
    is what makes that claim checkable rather than remembered.
    """

    @dace.program
    def returns_scalar(a: dace.float64[20]):
        return np.sum(a)

    sdfg, code = render(returns_scalar, 'mpr_ret_scalar')
    descriptor = sdfg.arrays['__return']
    assert isinstance(descriptor, dace.data.Array), f'__return is a {type(descriptor).__name__}, not an Array'
    assert descriptor.shape == (1, ), descriptor.shape

    a = np.random.rand(20)
    result = np.zeros(1)
    call_standalone(build_standalone(code, 'mpr_ret_scalar'), sdfg, {'a': a, '__return': result})
    assert np.allclose(result[0], a.sum())


def written_scalar_return_sdfg(name: str) -> dace.SDFG:
    """A hand-built SDFG whose ``__return`` is a written non-transient ``Scalar``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [20], dace.float64)
    sdfg.add_scalar('__return', dace.float64, transient=False)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('scale', {'i'}, {'o'}, 'o = i * 3.0')
    state.add_edge(state.add_access('a'), None, tasklet, 'i', dace.Memlet('a[0]'))
    state.add_edge(tasklet, 'o', state.add_access('__return'), None, dace.Memlet('__return[0]'))
    return sdfg


def test_written_scalar_return_is_promoted_to_an_out_parameter():
    """A written ``Scalar`` return is widened to a length-1 array, so the caller can read it.

    This is the same rewrite ``PromoteGPUScalarsToArrays`` performs for device memory; MPR reaches
    it through ``PromoteScalarOutputsToArrays``, which the GPU pass now wraps.
    """
    sdfg = written_scalar_return_sdfg('mpr_ret_promoted')
    assert 'double __return' in sdfg.signature(), ('the premise is gone: a Scalar return is no longer passed by '
                                                   'value, so there is nothing to promote')

    rendering = render_sdfg(sdfg, validate=False)
    assert isinstance(rendering.sdfg.arrays['__return'], dace.data.Array), 'the return scalar was not promoted'
    assert 'double * __restrict__ __return' in rendering.code, ('the promoted return is still not a pointer '
                                                                'parameter, so the result cannot leave the callee')
    assert_standalone(rendering.code, 'mpr_ret_promoted')

    a = np.random.rand(20)
    result = np.zeros(1)
    call_standalone(build_standalone(rendering.code, 'mpr_ret_promoted'), rendering.sdfg, {'a': a, '__return': result})
    assert np.allclose(result[0], a[0] * 3.0)


def test_promotion_leaves_no_reference_parameter():
    """A written scalar connector on a nested SDFG must not bind as ``T &``.

    ``cpp.emit_memlet_reference`` spells a written scalar connector as a C++ reference. That is
    valid C++ and invalid C, so the promotion has to remove it rather than the C dialect papering
    over it later.
    """
    sdfg = dace.SDFG('mpr_ret_nested_scalar')
    sdfg.add_array('a', [20], dace.float64)
    sdfg.add_scalar('__return', dace.float64, transient=False)
    state = sdfg.add_state()

    nested = written_scalar_return_sdfg('inner')
    node = state.add_nested_sdfg(nested, {'a'}, {'__return'})
    state.add_edge(state.add_access('a'), None, node, 'a', dace.Memlet('a[0:20]'))
    state.add_edge(node, '__return', state.add_access('__return'), None, dace.Memlet('__return[0]'))

    rendering = render_sdfg(sdfg, validate=False)
    assert not re.search(r'\w+\s*&\s*\w+\s*[,)]', rendering.code), ('a C++ reference parameter survived the '
                                                                    'promotion, so this cannot render as C')

    a = np.random.rand(20)
    result = np.zeros(1)
    call_standalone(build_standalone(rendering.code, 'mpr_ret_nested_scalar'), rendering.sdfg, {
        'a': a,
        '__return': result
    })
    assert np.allclose(result[0], a[0] * 3.0)


def test_unwritten_scalar_return_is_refused():
    """A ``Scalar`` return nothing writes cannot be promoted, and by value it returns nothing."""
    sdfg = dace.SDFG('mpr_ret_byvalue')
    sdfg.add_array('a', [20], dace.float64)
    sdfg.add_scalar('__return', dace.float64, transient=False)
    assert 'double __return' in sdfg.signature(), ('the premise of this test is gone: a Scalar return is no longer '
                                                   'passed by value, so MPR need not refuse it')

    with pytest.raises(NotImplementedError, match='BY VALUE'):
        render_sdfg(sdfg, validate=False)


def test_nested_return_connector_is_allowed():
    """A nested SDFG's ``__return`` is its out-connector, not an entry parameter -- it must render."""

    @dace.program
    def inner(a: dace.float64[20]):
        return a + 1.0

    @dace.program
    def nests_a_return(a: dace.float64[20]):
        return inner(a) * 2.0

    sdfg = nests_a_return.to_sdfg(simplify=False)
    sdfg.name = 'mpr_ret_nested'
    owners = {owner.name for owner, _ in return_containers(sdfg)}
    assert owners > {sdfg.name}, ('the premise is gone: no nested SDFG declares a return container, so this asserts '
                                  'nothing about nesting')

    rendering = render_sdfg(sdfg, validate=False)
    assert_standalone(rendering.code, 'mpr_ret_nested')

    a = np.random.rand(20)
    result = np.zeros(20)
    call_standalone(build_standalone(rendering.code, 'mpr_ret_nested'), rendering.sdfg, {'a': a, '__return': result})
    assert np.allclose(result, (a + 1.0) * 2.0)


def test_transient_return_inside_a_nested_sdfg_is_refused():
    """A transient return one level down is written into a buffer nothing outside can read."""
    sdfg = dace.SDFG('mpr_ret_nested_transient')
    sdfg.add_array('a', [20], dace.float64)
    state = sdfg.add_state()

    nested = dace.SDFG('inner')
    nested.add_array('a', [20], dace.float64)
    nested.add_array('__return', [20], dace.float64, transient=True)
    inner_state = nested.add_state()
    inner_state.add_nedge(inner_state.add_access('a'), inner_state.add_access('__return'),
                          dace.Memlet('a[0:20] -> [0:20]'))
    node = state.add_nested_sdfg(nested, {'a'}, set())
    state.add_edge(state.add_access('a'), None, node, 'a', dace.Memlet('a[0:20]'))

    owners = dict((name, owner.name) for owner, name in return_containers(sdfg))
    assert owners == {'__return': 'inner'}, f'the nested return container was not found: {owners}'

    with pytest.raises(NotImplementedError, match='nested SDFG'):
        render_sdfg(sdfg, validate=False)


def test_transient_return_is_refused():
    """A transient return container is absent from the signature, so the caller could never read it."""
    sdfg = dace.SDFG('mpr_ret_transient')
    sdfg.add_array('a', [20], dace.float64)
    sdfg.add_array('__return', [20], dace.float64, transient=True)
    assert '__return' not in sdfg.arglist(), 'the premise is gone: a transient return reaches the signature'

    with pytest.raises(NotImplementedError, match='transient'):
        render_sdfg(sdfg, validate=False)


if __name__ == '__main__':
    pytest.main([__file__])
