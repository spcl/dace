# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for callback provenance categories in the next-generation frontend:
every interpreter fallback carries a stable kebab-case ``[category]`` prefix
on its reason, raise-site categories propagate through the fallback plumbing,
and batching preserves the prefixes of every merged statement run.
"""
import re

import dace
from dace.frontend.python import nextgen
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _callbacks(root: tn.ScheduleTreeRoot):
    return [node for node in root.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]


def _categories(node: tn.PythonCallbackNode):
    """All category prefixes of a (possibly merged) callback reason."""
    return re.findall(r'(?:^|; )\[([^\]]+)\]', node.reason)


def dace_inhibitor(f):
    return f


@dace_inhibitor
def uninlinable(x):
    return x * 3


def test_detected_callback_category():
    """A call preprocessing wrapped as a callback is intended interpreter work."""

    @dace.program
    def calls_callback(A: dace.float64[N]):
        y = uninlinable(5)
        A[0] = y

    tree = nextgen.parse_program(calls_callback)
    callbacks = _callbacks(tree)
    assert len(callbacks) == 1
    assert _categories(callbacks[0])[0] == 'detected-callback'


def test_opaque_syntax_category():
    """Statements outside the CPA subset are categorized at canonicalization."""

    @dace.program
    def dict_literal(A: dace.float64[N]):
        d = {'a': 1}
        A[0] = d['a']

    tree = nextgen.parse_program(dict_literal)
    callbacks = _callbacks(tree)
    assert len(callbacks) == 1
    categories = _categories(callbacks[0])
    assert categories[0] == 'opaque-syntax:Assign'
    # The consumer of the opaque dict is a pyobject-propagation fallback,
    # batched into the same callback with its own prefix preserved.
    assert 'pyobject-propagation' in categories


def test_raise_site_category_propagates(monkeypatch):
    """A category set at the raise site survives the fallback plumbing."""
    from dace.frontend.python.nextgen.lowering.mechanisms import elementwise

    def raising(*args, **kwargs):
        raise UnsupportedFeatureError('forced failure', category='test-category')

    monkeypatch.setattr(elementwise, 'emit_computation', raising)

    @dace.program
    def computes(A: dace.float64[N]):
        A[0] = A[1] + 1.0

    tree = nextgen.parse_program(computes)
    callbacks = _callbacks(tree)
    assert len(callbacks) == 1
    assert _categories(callbacks[0]) == ['test-category']
    assert 'forced failure' in callbacks[0].reason


def test_uncategorized_default(monkeypatch):
    """A category-less raise site falls back with the explicit default."""
    from dace.frontend.python.nextgen.lowering.mechanisms import elementwise

    def raising(*args, **kwargs):
        raise UnsupportedFeatureError('forced failure')

    monkeypatch.setattr(elementwise, 'emit_computation', raising)

    @dace.program
    def computes(A: dace.float64[N]):
        A[0] = A[1] + 1.0

    tree = nextgen.parse_program(computes)
    callbacks = _callbacks(tree)
    assert len(callbacks) == 1
    assert _categories(callbacks[0]) == ['uncategorized']


def test_every_callback_reason_is_categorized():
    """No fallback path emits a reason without a leading category prefix."""

    @dace.program
    def mixed(A: dace.float64[N]):
        print(A)
        y = uninlinable(2)
        d = [x for x in range(int(y))]
        A[0] = len(d)

    tree = nextgen.parse_program(mixed)
    callbacks = _callbacks(tree)
    assert callbacks
    for node in callbacks:
        # Every '; '-separated reason segment carries its own prefix
        segments = node.reason.split('; ')
        assert len(_categories(node)) == len(segments)


if __name__ == '__main__':
    import pytest
    test_detected_callback_category()
    test_opaque_syntax_category()
    test_raise_site_category_propagates(pytest.MonkeyPatch())
    test_uncategorized_default(pytest.MonkeyPatch())
    test_every_callback_reason_is_categorized()
