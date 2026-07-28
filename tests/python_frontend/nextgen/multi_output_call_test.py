# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for calls whose result is SEVERAL containers (``numpy.split``,
``numpy.divmod``, ``numpy.frexp``, ``numpy.linspace(retstep=True)``).

Every lowering mechanism writes one target, so these take their own path: one
:class:`~dace.sdfg.analysis.schedule_tree.treenodes.ReplacementCallNode`
carrying one result container per output, with the assigned name bound to a
static sequence of those containers. Canonicalization has already reduced the
unpacking (``p, q = ...``) to element reads (``p = __unpack0[0]``), which fold
to direct accesses against that sequence.
"""
import numpy as np
import pytest

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _nodes_of_type(tree, node_type):
    return [node for node in tree.preorder_traversal() if isinstance(node, node_type)]


def test_split_emits_one_call_with_two_targets():

    @dace.program
    def prog(a: dace.float64[8]):
        p, q = np.split(a, 2)
        return p + q

    tree = nextgen.parse_program(prog, np.zeros(8))
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    calls = _nodes_of_type(tree, tn.ReplacementCallNode)
    assert len(calls) == 1
    assert calls[0].qualname == 'numpy.split'
    # One result container per output, the first in ``target`` and the rest in
    # ``extra_targets``.
    assert len(calls[0].targets) == 2
    assert calls[0].extra_targets and calls[0].target not in calls[0].extra_targets
    assert all(name in tree.containers for name in calls[0].targets)
    assert all(tuple(tree.containers[name].shape) == (4, ) for name in calls[0].targets)


def test_divmod_is_a_ufunc_call_with_two_targets():

    @dace.program
    def prog(a: dace.float64[8], b: dace.float64[8]):
        d, m = np.divmod(a, b)
        return d + m

    tree = nextgen.parse_program(prog, np.zeros(8), np.ones(8))
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    calls = _nodes_of_type(tree, tn.ReplacementCallNode)
    assert len(calls) == 1
    assert calls[0].ufunc_name == 'divmod' and calls[0].ufunc_method is None
    assert len(calls[0].targets) == 2


@pytest.mark.parametrize('name', ['split', 'hsplit', 'divmod', 'frexp', 'modf'])
def test_multi_output_execution(name):
    """Each form runs and agrees with NumPy."""
    rng = np.random.default_rng(0)
    a = rng.random(8) * 10.0
    b = rng.random(8) * 3.0 + 1.0
    matrix = rng.random((4, 8))

    if name == 'split':

        @dace.program
        def prog(a: dace.float64[8]):
            p, q = np.split(a, 2)
            return p + q

        arguments, expected = dict(a=a), np.split(a, 2)[0] + np.split(a, 2)[1]
    elif name == 'hsplit':

        @dace.program
        def prog(a: dace.float64[4, 8]):
            p, q = np.hsplit(a, 2)
            return p + q

        arguments = dict(a=matrix)
        expected = np.hsplit(matrix, 2)[0] + np.hsplit(matrix, 2)[1]
    elif name == 'divmod':

        @dace.program
        def prog(a: dace.float64[8], b: dace.float64[8]):
            d, m = np.divmod(a, b)
            return d + m

        arguments = dict(a=a, b=b)
        expected = np.divmod(a, b)[0] + np.divmod(a, b)[1]
    elif name == 'frexp':

        @dace.program
        def prog(a: dace.float64[8]):
            mantissa, _ = np.frexp(a)
            return mantissa

        arguments, expected = dict(a=a), np.frexp(a)[0]
    else:

        @dace.program
        def prog(a: dace.float64[8]):
            frac, whole = np.modf(a)
            return frac + whole

        arguments, expected = dict(a=a), np.modf(a)[0] + np.modf(a)[1]

    tree = nextgen.parse_program(prog, *arguments.values())
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    result = tree.as_sdfg().compile()(**arguments)
    assert np.allclose(np.asarray(result).reshape(np.shape(expected)), expected)


def test_unused_second_output():
    """Only one of the results is read; the other is still produced."""

    @dace.program
    def prog(a: dace.float64[8]):
        p, _ = np.split(a, 2)
        return p * 2.0

    a = np.arange(8, dtype=np.float64)
    tree = nextgen.parse_program(prog, a)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    result = tree.as_sdfg().compile()(a=a)
    assert np.allclose(np.asarray(result).ravel(), a[:4] * 2.0)


def test_bare_statement_still_falls_back():
    """A multi-output call with no assignment target discards every result, so
    there is nothing to bind: it stays a callback rather than being lowered to
    dataflow nobody reads."""

    @dace.program
    def prog(a: dace.float64[8]):
        np.split(a, 2)
        return a * 2.0

    tree = nextgen.parse_program(prog, np.zeros(8))
    assert _nodes_of_type(tree, tn.PythonCallbackNode)


if __name__ == '__main__':
    test_split_emits_one_call_with_two_targets()
    test_divmod_is_a_ufunc_call_with_two_targets()
    for _name in ['split', 'hsplit', 'divmod', 'frexp', 'modf']:
        test_multi_output_execution(_name)
    test_unused_second_output()
    test_bare_statement_still_falls_back()
