# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the per-program callee parse cache: a nested ``@dace.program``
invoked from multiple call sites with the same specialization (argument
descriptors, constant values, bound method object) preprocesses and
canonicalizes exactly once per top-level parse; different specializations
parse separately; failures cache and re-raise per call site (preserving the
callback fallback); and cached canonical bodies are isolated from
lowering-time mutation via per-site deep copies.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.frontend.python.nextgen.lowering.rules import calls
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


class _ParseCounter:
    """Counts invocations of calls._parse_callee (cache misses)."""

    def __init__(self, monkeypatch):
        self.count = 0
        original = calls._parse_callee

        def counting(*args, **kwargs):
            self.count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(calls, '_parse_callee', counting)


@dace.program
def doubler(x: dace.float64[N]):
    x[:] = x[:] * 2.0


def test_same_specialization_parses_once(monkeypatch):
    counter = _ParseCounter(monkeypatch)

    @dace.program
    def caller(a: dace.float64[N], b: dace.float64[N]):
        doubler(a)
        doubler(b)
        doubler(a)

    tree = nextgen.parse_program(caller)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(_nodes_of_type(tree, tn.FunctionCallScope)) == 3
    assert counter.count == 1  # a and b share descriptors -> one parse

    # Execution stays correct through the shared parse.
    func = tree.as_sdfg().compile()
    a = np.random.rand(6)
    b = np.random.rand(6)
    a_ref, b_ref = a * 4.0, b * 2.0
    func(a=a, b=b, N=6)
    assert np.allclose(a, a_ref)
    assert np.allclose(b, b_ref)


def test_distinct_descriptors_parse_separately(monkeypatch):
    counter = _ParseCounter(monkeypatch)

    @dace.program
    def caller(a: dace.float64[N], b: dace.float32[N]):
        doubler(a)
        doubler(b)

    tree = nextgen.parse_program(caller)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    assert counter.count == 2  # different dtypes -> different specializations


def test_constant_specialization_keys(monkeypatch):
    counter = _ParseCounter(monkeypatch)

    @dace.program
    def scaler(x: dace.float64[N], factor: dace.compiletime):
        x[:] = x[:] * factor

    @dace.program
    def caller(a: dace.float64[N]):
        scaler(a, 2.0)
        scaler(a, 3.0)
        scaler(a, 2.0)

    tree = nextgen.parse_program(caller)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    assert counter.count == 2  # 2.0 and 3.0 specialize separately; 2.0 reuses


def test_methodobj_discrimination(monkeypatch):

    class Adder:

        def __init__(self, value):
            self.value = value

        def __call__(self, x):
            x[:] = x[:] + self.value

    add_one = Adder(1.0)
    add_two = Adder(2.0)

    @dace.program
    def caller(a: dace.float64[N]):
        add_one(a)
        add_two(a)

    counter = _ParseCounter(monkeypatch)
    tree = nextgen.parse_program(caller)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    # Same __call__ source, different bound objects: attribute values enter
    # the parse through preprocessing, so the cache must not share entries.
    assert counter.count == 2
    codes = [t.node.code.as_string for t in _nodes_of_type(tree, tn.TaskletNode)]
    assert any('1.0' in code for code in codes)
    assert any('2.0' in code for code in codes)


def test_failure_caches_per_site_fallback(monkeypatch):
    attempts = {'count': 0}

    def failing_parse(*args, **kwargs):
        attempts['count'] += 1
        raise ValueError('synthetic parse failure')

    monkeypatch.setattr(calls, '_parse_callee', failing_parse)

    @dace.program
    def caller(a: dace.float64[N]):
        doubler(a)
        doubler(a)

    tree = nextgen.parse_program(caller)
    # Both call sites fall back to callbacks; the parse was attempted once
    # and its cached failure re-raised for the second site (adjacent
    # callbacks may batch into one node).
    assert len(_nodes_of_type(tree, tn.PythonCallbackNode)) >= 1
    assert not _nodes_of_type(tree, tn.FunctionCallScope)
    assert attempts['count'] == 1


def test_mutation_isolation():
    # A callee whose early return gets restructured (lowering mutates the
    # canonical body): the second inline must see a pristine copy.
    @dace.program
    def clamped(x: dace.float64[N]):
        if x[0] > 0.5:
            return x[0]
        return x[0] * 2.0

    @dace.program
    def caller(a: dace.float64[N], b: dace.float64[N], out: dace.float64[2]):
        out[0] = clamped(a)
        out[1] = clamped(b)

    tree = nextgen.parse_program(caller)
    scopes = _nodes_of_type(tree, tn.FunctionCallScope)
    assert len(scopes) == 2
    # Both inlines produced the same structure from the shared cached body.
    counts = [sum(1 for _ in scope.preorder_traversal()) for scope in scopes]
    assert counts[0] == counts[1]


if __name__ == '__main__':

    class _Patch:
        """Minimal monkeypatch stand-in for direct execution."""

        def __init__(self):
            self._saved = []

        def setattr(self, target, name, value):
            self._saved.append((target, name, getattr(target, name)))
            setattr(target, name, value)

        def undo(self):
            for target, name, value in reversed(self._saved):
                setattr(target, name, value)
            self._saved.clear()

    for test in (test_same_specialization_parses_once, test_distinct_descriptors_parse_separately,
                 test_constant_specialization_keys, test_methodobj_discrimination,
                 test_failure_caches_per_site_fallback):
        patch = _Patch()
        try:
            test(patch)
        finally:
            patch.undo()
    test_mutation_isolation()
