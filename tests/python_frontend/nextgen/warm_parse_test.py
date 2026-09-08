# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the speculative parallel warm-up of nested ``@dace.program``
parses: statically-resolvable callees pre-parse in a thread pool before
sequential lowering, sharing work with the lowering path through the
per-program parse cache (so nothing ever parses twice), recursing bottom-up
inside workers, and terminating on recursive programs.
"""
import threading

import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.frontend.python.nextgen.lowering.rules import calls
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


class _ParseRecorder:
    """Records calls._parse_callee invocations and their threads."""

    def __init__(self, monkeypatch):
        self.calls = []
        original = calls._parse_callee

        def recording(callee, *args, **kwargs):
            # DaceProgram names are module-qualified; keep the bare suffix
            self.calls.append((callee.name.rsplit('_', 1)[-1], threading.current_thread().name))
            return original(callee, *args, **kwargs)

        monkeypatch.setattr(calls, '_parse_callee', recording)


@dace.program
def scale2(x: dace.float64[N]):
    x[:] = x[:] * 2.0


@dace.program
def add1(x: dace.float64[N]):
    x[:] = x[:] + 1.0


@dace.program
def sub3(x: dace.float64[N]):
    x[:] = x[:] - 3.0


def test_parallel_warmup(monkeypatch):
    recorder = _ParseRecorder(monkeypatch)

    @dace.program
    def caller(a: dace.float64[N]):
        scale2(a)
        add1(a)
        sub3(a)

    tree = nextgen.parse_program(caller)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    # Each unique callee parsed exactly once (warm-up and lowering share the
    # cache), and the warm-up ran the parses on pool worker threads.
    assert sorted(name for name, _ in recorder.calls) == ['add1', 'scale2', 'sub3']
    assert all('ThreadPoolExecutor' in thread for _, thread in recorder.calls)

    func = tree.as_sdfg().compile()
    a = np.random.rand(8)
    expected = (a * 2.0 + 1.0) - 3.0
    func(a=a, N=8)
    assert np.allclose(a, expected)


def test_bottom_up_nested_warmup(monkeypatch):

    @dace.program
    def middle(x: dace.float64[N]):
        scale2(x)
        x[:] = x[:] + 0.5

    @dace.program
    def caller(a: dace.float64[N], b: dace.float64[N]):
        middle(a)
        add1(b)

    recorder = _ParseRecorder(monkeypatch)
    tree = nextgen.parse_program(caller)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    # The worker that parsed 'middle' recursively warmed its callee 'scale2'
    # in the same thread (bottom-up), and lowering re-parsed nothing.
    names = [name for name, _ in recorder.calls]
    assert sorted(names) == ['add1', 'middle', 'scale2']
    threads = dict((name, thread) for name, thread in recorder.calls)
    assert threads['scale2'] == threads['middle']


def test_recursive_program_terminates():

    @dace.program
    def recurse(x: dace.float64[N]):
        recurse(x)

    @dace.program
    def caller(a: dace.float64[N]):
        recurse(a)

    # The warm-up recursion guard stops on the self-call; lowering falls back
    # to a callback for the recursive invocation (established behavior).
    tree = nextgen.parse_program(caller)
    assert len(_nodes_of_type(tree, tn.PythonCallbackNode)) >= 1


def test_non_static_arguments_skip_warmup(monkeypatch):
    recorder = _ParseRecorder(monkeypatch)

    @dace.program
    def caller(a: dace.float64[N]):
        t = a[0:4]
        scale2(t)

    tree = nextgen.parse_program(caller)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    # The argument is a lowering-created view, unknown at warm time: the call
    # is skipped by the warm-up and parses inline exactly once.
    assert [name for name, _ in recorder.calls] == ['scale2']
    assert all('ThreadPoolExecutor' not in thread for _, thread in recorder.calls)


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

    for test in (test_parallel_warmup, test_bottom_up_nested_warmup, test_non_static_arguments_skip_warmup):
        patch = _Patch()
        try:
            test(patch)
        finally:
            patch.undo()
    test_recursive_program_terminates()
