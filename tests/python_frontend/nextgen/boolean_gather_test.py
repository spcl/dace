# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``B = A[mask]`` for a full-shape boolean array ``mask`` -- the bare
top-level boolean-mask *read*. The result's element count is unknown until
the mask has been read at runtime, so unlike every other advanced-indexing
read this frontend supports, this one mints a fresh SDFG symbol from a
runtime-computed count (see
:func:`~dace.frontend.python.nextgen.lowering.mechanisms.advanced_indexing.emit_boolean_gather`)
instead of allocating a build-time-known shape.

Two configurable strategies (``frontend.boolean_index_strategy``), both
proven independently of the frontend in
``tests/sdfg/deferred_symbol_boolean_filter_test.py``:

- ``'view'`` (default) -- one pass, an upper-bound-sized backing buffer, the
  result exposed as a VIEW over its first ``M`` elements. Since a view is
  always transient, it cannot itself cross a program's external call
  boundary (DaCe's ctypes ABI needs the caller to already have every
  argument's memory before the call, and the count is only known mid-call) --
  every test here that checks *content* correctness for this strategy
  therefore consumes ``B`` inside the SAME program (a slice-write into a
  caller-provided upper-bound array), which is also the realistic use case
  the strategy targets: further computation on the exact- or view-sized
  result without a host round-trip. A ``B`` that stays unused is legitimate
  dead code and is cleanly eliminated by ``simplify()`` -- verified as its
  own case below, since that combination (a WCR accumulator whose only
  consumer is an interstate-edge symbol assignment, feeding a Stream-typed
  conditional push) tripped a real, narrow, now-fixed core bug
  (``dtypes.pointer`` never set ``self.typename``, crashing
  ``DeadDataflowElimination``'s connector-removal hint injection with an
  ``AttributeError`` instead of correctly stringifying the type -- nothing
  to do with whether the dependency was tracked, which ``AccessSets``
  already gets right via ``InterstateEdge.free_symbols``).
- ``'exact'`` -- two passes (a count-only reduction, then a compaction sized
  exactly to the count), never over-allocates. ``B`` is registered
  non-transient here, so it survives regardless of downstream use (matching
  classic's ``filter.py`` sample, whose equivalent output the caller may or
  may not read back) -- correctness tests call the compiled SDFG directly,
  supplying ``B``/``M`` themselves.

Every content-correctness test compares SORTED results: the underlying
compaction runs inside an OpenMP-parallel map, so source order is not
preserved (also matching ``filter.py``, whose own regression test sorts
before comparing).
"""
import numpy as np
import pytest

import dace
from dace.config import Config
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _callbacks(root: tn.ScheduleTreeRoot):
    found = []

    def walk(node):
        for child in getattr(node, 'children', None) or []:
            if isinstance(child, tn.PythonCallbackNode):
                found.append(child)
            walk(child)

    walk(root)
    return found


@pytest.fixture(autouse=True)
def _restore_strategy_config():
    original = Config.get('frontend', 'boolean_index_strategy')
    yield
    Config.set('frontend', 'boolean_index_strategy', value=original)


def _rng_case(seed: int, n: int, fraction: float):
    rng = np.random.default_rng(seed)
    A = rng.random(n)
    if fraction <= 0.0:
        mask = np.zeros(n, dtype=bool)
    elif fraction >= 1.0:
        mask = np.ones(n, dtype=bool)
    else:
        mask = rng.random(n) < fraction
    return A, mask


@pytest.mark.parametrize('strategy', ['view', 'exact'])
def test_boolean_gather_no_callback(strategy):
    """The bare top-level form lowers with zero callbacks, for both strategies."""
    Config.set('frontend', 'boolean_index_strategy', value=strategy)

    @dace.program
    def program(A: dace.float64[N], mask: dace.bool[N]):
        B = A[mask]

    tree = nextgen.parse_program(program)
    assert not _callbacks(tree), [node.reason for node in _callbacks(tree)]


@pytest.mark.parametrize('strategy', ['view', 'exact'])
@pytest.mark.parametrize('seed,n,fraction', [
    (1, 137, 0.4),
    (2, 137, 0.0),
    (3, 137, 1.0),
    (4, 1, 1.0),
    (5, 1, 0.0),
    (6, 500, 0.9),
])
def test_boolean_gather_matches_numpy(strategy, seed, n, fraction):
    """Content correctness across normal and boundary (empty/full/size-1) masks."""
    Config.set('frontend', 'boolean_index_strategy', value=strategy)

    @dace.program
    def program(A: dace.float64[N], mask: dace.bool[N], out: dace.float64[N]):
        B = A[mask]
        out[0:B.shape[0]] = B

    tree = nextgen.parse_program(program)
    assert not _callbacks(tree), [node.reason for node in _callbacks(tree)]
    A, mask = _rng_case(seed, n, fraction)
    m = int(mask.sum())
    expected = A[mask]
    out = np.zeros(n, dtype=np.float64)
    tree.as_sdfg()(A=A, mask=mask, out=out, N=n)
    if m:
        assert np.allclose(np.sort(out[:m]), np.sort(expected))
    else:
        assert m == 0  # nothing meaningful was written; only the count matters


@pytest.mark.parametrize('strategy', ['view', 'exact'])
def test_boolean_gather_unused_result_is_eliminated_cleanly(strategy):
    """
    An assigned-but-never-used ``B`` is legitimate dead code. This must not
    crash simplify() -- see the module docstring's note on the ``dtypes.pointer``
    bugfix this test would have caught.
    """
    Config.set('frontend', 'boolean_index_strategy', value=strategy)

    @dace.program
    def program(A: dace.float64[N], mask: dace.bool[N]):
        B = A[mask]

    tree = nextgen.parse_program(program)
    assert not _callbacks(tree), [node.reason for node in _callbacks(tree)]
    sdfg = tree.as_sdfg()  # must not raise
    compiled = sdfg.compile()
    A, mask = _rng_case(8, 30, 0.5)
    compiled(A=A, mask=mask, N=30)  # must not raise


# --- Cases outside the narrow bare-top-level-assignment scope: unchanged,
# still fall back to a callback exactly as before this feature.


def test_boolean_gather_nested_in_expression_still_falls_back():

    @dace.program
    def program(A: dace.float64[N], mask: dace.bool[N]):
        return A[mask] + 1.0

    tree = nextgen.parse_program(program)
    callbacks = _callbacks(tree)
    assert callbacks
    assert any('advanced-indexing' in node.reason for node in callbacks)


def test_boolean_gather_combined_with_integer_index_still_falls_back():

    @dace.program
    def program(A: dace.float64[N, N], mask: dace.bool[N], ind: dace.int32[N]):
        B = A[mask, ind]

    tree = nextgen.parse_program(program)
    assert _callbacks(tree)


if __name__ == '__main__':
    test_boolean_gather_no_callback('view')
    test_boolean_gather_no_callback('exact')
    for strategy in ('view', 'exact'):
        for case in [(1, 137, 0.4), (2, 137, 0.0), (3, 137, 1.0), (4, 1, 1.0), (5, 1, 0.0), (6, 500, 0.9)]:
            test_boolean_gather_matches_numpy(strategy, *case)
        test_boolean_gather_unused_result_is_eliminated_cleanly(strategy)
    test_boolean_gather_nested_in_expression_still_falls_back()
    test_boolean_gather_combined_with_integer_index_still_falls_back()
