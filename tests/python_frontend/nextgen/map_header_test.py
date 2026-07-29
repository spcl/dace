# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for ``for``-loop headers over DaCe iteration constructs in the nextgen
frontend.

The frontend resolves a header by *evaluating* it (see
:mod:`~dace.frontend.python.iterators`) rather than by matching its
syntax: ``dace.map[...]`` is a real object in the program's closure, and the
operators written on it (``@ ScheduleType``) are its own dunder methods. These
tests pin that generality — every spelling that produces the same generator is
one form to the frontend, and an operator the frontend has never heard of works
as soon as the construct defines it.
"""
import ast

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.frontend.python import iterators, nextgen
from dace.frontend.python.interface import MapGenerator
from dace.frontend.python.nextgen.canonical import cpa
from dace.sdfg.analysis.schedule_tree import treenodes as tn

#: The built-in under another name, to show that ``range`` is resolved.
builtin_range = range


def two_element_range(stop):
    """A program-defined ``range`` that is not the built-in."""
    return [0, 1]


def _map_schedules(root: tn.ScheduleTreeRoot):
    """The (params, schedule) of every map scope in a schedule tree."""
    return [(tuple(node.node.map.params), node.node.map.schedule) for node in root.preorder_traversal()
            if isinstance(node, tn.MapScope)]


def _callbacks(root: tn.ScheduleTreeRoot):
    return [node for node in root.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]


def test_schedule_annotation_reaches_the_map():
    """``dace.map[...] @ ScheduleType`` sets the schedule of the emitted map."""

    @dace.program
    def annotated(a: dace.float64[20, 20]):
        for i in dace.map[0:20]:
            for j in dace.map[0:20] @ dace.ScheduleType.Sequential:
                a[i, j] = 1.0

    root = nextgen.parse_program(annotated, np.zeros((20, 20)))
    assert _map_schedules(root) == [(('i', ), dtypes.ScheduleType.Default), (('j', ), dtypes.ScheduleType.Sequential)]
    assert not _callbacks(root)


def test_schedule_annotation_spellings():
    """Any expression that evaluates to a ScheduleType annotates the map."""
    from dace.dtypes import ScheduleType
    sequential = ScheduleType.Sequential

    @dace.program
    def annotated(a: dace.float64[3, 20]):
        for i in dace.map[0:20] @ ScheduleType.Sequential:
            a[0, i] = 1.0
        for i in dace.map[0:20] @ dtypes.ScheduleType.Sequential:
            a[1, i] = 1.0
        for i in dace.map[0:20] @ sequential:
            a[2, i] = 1.0

    root = nextgen.parse_program(annotated, np.zeros((3, 20)))
    assert [schedule for _, schedule in _map_schedules(root)] == [dtypes.ScheduleType.Sequential] * 3


def test_annotated_map_with_compound_bounds():
    """A bound needing a temporary is still hoisted when the header has an operator."""

    @dace.program
    def annotated(a: dace.float64[20]):
        for i in dace.map[0:a.shape[0] // 2] @ dace.ScheduleType.Sequential:
            a[i] = 1.0

    root = nextgen.parse_program(annotated, np.zeros(20))
    (params, schedule), = _map_schedules(root)
    assert params == ('i', ) and schedule == dtypes.ScheduleType.Sequential
    assert not _callbacks(root)


def test_annotated_map_executes():
    a = np.random.rand(20, 20)
    y = np.zeros(20)

    @dace.program
    def rowsum(a: dace.float64[20, 20], y: dace.float64[20]):
        for i in dace.map[0:20]:
            for j in dace.map[0:20] @ dace.ScheduleType.Sequential:
                y[i] += a[i, j]

    nextgen.parse_program(rowsum, a, y).as_sdfg().compile()(a=a, y=y)
    assert np.allclose(y, a.sum(axis=1))


def test_unknown_operator_needs_no_frontend_change():
    """
    The frontend knows nothing about ``@`` specifically: any operator the
    iteration construct defines works. ``MapGenerator`` is given a second
    schedule operator here, and the header resolves through it without the
    frontend having been taught the syntax.
    """
    MapGenerator.__or__ = lambda self, schedule: MapGenerator(self.rng, schedule)
    try:

        @dace.program
        def annotated(a: dace.float64[20]):
            for i in dace.map[0:20] | dace.ScheduleType.Sequential:
                a[i] = 1.0

        root = nextgen.parse_program(annotated, np.zeros(20))
    finally:
        del MapGenerator.__or__
    assert _map_schedules(root) == [(('i', ), dtypes.ScheduleType.Sequential)]
    assert not _callbacks(root)


def test_header_evaluation_rejects_non_constructs():
    """Headers that are not iteration constructs resolve to nothing."""
    globals_ = {'dace': dace, 'numpy': np, 'rows': [1, 2, 3]}
    for source in ('range(0, 10, 1)', 'rows', 'numpy.zeros(10)', 'dace.map', 'dace.map[0:10] @ 5',
                   'dace.map[0:10] @ numpy'):
        header = ast.parse(source, mode='eval').body
        assert iterators.iteration_object(header, globals_) is None, source
        assert not cpa.is_dace_map_iterator(header, globals_), source


def test_header_resolution_calls_nothing():
    """
    Resolving a header must not execute the program's functions: a ``for`` over
    a generator function is asked about like any other header.
    """
    calls = []

    def generator():
        calls.append(1)
        return [1, 2]

    header = ast.parse('generator()', mode='eval').body
    assert iterators.iteration_object(header, {'generator': generator}) is None
    assert not calls


def test_header_evaluation_keeps_bounds_symbolic():
    """The generator's range stays an AST — bounds are lowering's business."""
    header = ast.parse('dace.map[0:N, 0:M] @ dace.ScheduleType.Sequential', mode='eval').body
    generator = iterators.iteration_object(header, {'dace': dace})
    assert generator.schedule == dtypes.ScheduleType.Sequential
    assert isinstance(generator.rng, ast.Tuple)
    assert all(isinstance(dimension, ast.Slice) for dimension in generator.rng.elts)


def test_map_spellings_are_one_canonical_form():
    """Aliased imports of ``dace.map`` are recognized like the plain spelling."""
    import dace as dc
    from dace import map as dmap
    for source, globals_ in (('dace.map[0:10]', {
            'dace': dace
    }), ('dc.map[0:10]', {
            'dc': dc
    }), ('dmap[0:10]', {
            'dmap': dmap
    }), ('dmap[0:10] @ dace.ScheduleType.Sequential', {
            'dmap': dmap,
            'dace': dace
    })):
        header = ast.parse(source, mode='eval').body
        assert cpa.is_dace_map_iterator(header, globals_), source


def test_range_is_resolved_not_name_matched():
    """
    ``range`` is recognized as the object it resolves to, like ``dace.map``:
    an aliased built-in is the same loop, and a rebound name is not one.
    """
    header = ast.parse('range(0, 10, 1)', mode='eval').body
    assert cpa.is_range_iterator(header, {})
    assert cpa.is_range_iterator(header, {'range': range})
    assert cpa.is_range_iterator(ast.parse('rng(0, 10, 1)', mode='eval').body, {'rng': range})
    assert not cpa.is_range_iterator(header, {'range': lambda *bounds: [0, 1]})


def test_aliased_range_loop_is_a_loop():
    """A program looping over an aliased built-in gets a loop, not a callback."""

    @dace.program
    def counted(a: dace.float64[20]):
        for i in builtin_range(20):
            a[i] = 1.0

    root = nextgen.parse_program(counted, np.zeros(20))
    assert not _callbacks(root)


def test_shadowed_range_is_not_read_as_the_builtin():
    """
    A program that rebinds ``range`` iterates what it actually bound. The
    frontend has no compile-time reading for that, so the loop degrades to the
    ordinary Python-iterator path instead of silently becoming ``0..stop``.
    """

    @dace.program
    def shadowed(a: dace.float64[20]):
        for i in two_element_range(20):
            a[i] = 1.0

    root = nextgen.parse_program(shadowed, np.zeros(20))
    assert _callbacks(root)


def test_bad_schedule_operand_degrades():
    """
    An operand the construct rejects (``@ 5``) makes the header something the
    frontend cannot evaluate, which is the same situation as an arbitrary
    Python iterator: the loop becomes a callback rather than a map, and the
    construct's own ``TypeError`` is what the interpreter raises there.
    """

    @dace.program
    def annotated(a: dace.float64[20]):
        for i in dace.map[0:20] @ 5:
            a[i] = 1.0

    root = nextgen.parse_program(annotated, np.zeros(20))
    assert not _map_schedules(root)
    assert len(_callbacks(root)) == 1
    with pytest.raises(TypeError):
        dace.map[0:20] @ 5


if __name__ == '__main__':
    test_schedule_annotation_reaches_the_map()
    test_schedule_annotation_spellings()
    test_annotated_map_with_compound_bounds()
    test_annotated_map_executes()
    test_unknown_operator_needs_no_frontend_change()
    test_header_evaluation_rejects_non_constructs()
    test_header_resolution_calls_nothing()
    test_header_evaluation_keeps_bounds_symbolic()
    test_map_spellings_are_one_canonical_form()
    test_range_is_resolved_not_name_matched()
    test_aliased_range_loop_is_a_loop()
    test_shadowed_range_is_not_read_as_the_builtin()
    test_bad_schedule_operand_degrades()
