# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the per-stage canonicalize debugging harness
(``dace.transformation.passes.canonicalize.debug``).

Covers the building blocks (random-input construction, output comparison),
the clean-kernel integration path (every stage valid + numerically
correct), and the *detection* path: when a stage corrupts values or
structure, the harness must flag it.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.canonicalize import debug as cdbg
from dace.transformation.passes.canonicalize.debug import (canonicalize_with_stage_checks, first_failing_stage,
                                                           StageCheckResult, _build_random_inputs, _compare)

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def elementwise(a: dace.float64[N, M], b: dace.float64[N, M]):
    for i in dace.map[0:N]:
        for j in dace.map[0:M]:
            b[i, j] = a[i, j] * 2.0 + 1.0


@dace.program
def per_row_reduction(a: dace.float64[N, M], b: dace.float64[N]):
    for i in dace.map[0:N]:
        s = 0.0
        for j in range(M):
            s += a[i, j]
        b[i] = s


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------


def test_build_random_inputs_small_symbols_and_data():
    """Free symbols are set to the small ``symbol_value``; arrays get
    shapes resolved from those symbols and small-magnitude data."""
    sdfg = elementwise.to_sdfg(simplify=True)
    rng = np.random.default_rng(0)
    symbols, arrays = _build_random_inputs(sdfg, symbol_value=3, rng=rng)
    assert symbols == {'N': 3, 'M': 3}
    assert set(arrays.keys()) == {'a', 'b'}
    assert arrays['a'].shape == (3, 3)
    assert arrays['b'].shape == (3, 3)  # b is dace.float64[N, M]
    # Small magnitude (within the harness float range, generously bounded).
    assert np.all(np.abs(arrays['a']) <= 2.0)


def test_compare_identical_and_diff():
    ref = {'x': np.ones((4, ), np.float64)}
    same = {'x': np.ones((4, ), np.float64)}
    diff = {'x': np.array([1.0, 1.0, 1.0, 2.5])}
    ok, md = _compare(ref, same, rtol=1e-9, atol=1e-12)
    assert ok and md == 0.0
    ok2, md2 = _compare(ref, diff, rtol=1e-9, atol=1e-12)
    assert not ok2 and md2 == pytest.approx(1.5)


def test_stage_result_ok_and_str():
    good = StageCheckResult(0, 'fuse', 'P', True, None, True, 0.0, None)
    assert good.ok and 'numeric-ok' in str(good)
    invalid = StageCheckResult(1, 'fuse', 'P', False, 'boom', None, None, None)
    assert not invalid.ok and 'INVALID' in str(invalid)
    mismatch = StageCheckResult(2, 'fuse', 'P', True, None, False, 3.0, None)
    assert not mismatch.ok and 'NUMERIC-MISMATCH' in str(mismatch)


# ----------------------------------------------------------------------
# Clean integration: every stage stays valid + numerically correct
# ----------------------------------------------------------------------


def test_elementwise_all_stages_ok():
    sdfg = elementwise.to_sdfg(simplify=True)
    # Structural fingerprint before the harness runs.
    n_nodes_before = len(list(sdfg.all_nodes_recursive()))
    n_states_before = len(list(sdfg.all_states()))
    results = canonicalize_with_stage_checks(sdfg, symbol_value=3)
    assert results, 'expected at least one stage'
    for r in results:
        assert r.ok, f'stage failed: {r}'
    # The input SDFG must NOT have been mutated by the harness (it
    # canonicalizes an internal copy).
    assert len(list(sdfg.all_nodes_recursive())) == n_nodes_before
    assert len(list(sdfg.all_states())) == n_states_before


def test_per_row_reduction_all_stages_ok():
    sdfg = per_row_reduction.to_sdfg(simplify=True)
    results = canonicalize_with_stage_checks(sdfg, symbol_value=4)
    for r in results:
        assert r.ok, f'stage failed: {r}'


def test_first_failing_stage_none_for_clean_kernel():
    sdfg = elementwise.to_sdfg(simplify=True)
    assert first_failing_stage(sdfg) is None


# ----------------------------------------------------------------------
# Detection path: a corrupting stage must be caught
# ----------------------------------------------------------------------


class _PerturbOutputValues(ppl.Pass):
    """A deliberately value-corrupting pass: appends ``+ 1000`` to the
    first tasklet's output so the result diverges from the reference,
    while keeping the SDFG structurally valid."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified) -> bool:
        return False

    def apply_pass(self, sdfg, _):
        for n, _p in sdfg.all_nodes_recursive():
            if isinstance(n, nodes.Tasklet) and n.out_connectors:
                oc = next(iter(n.out_connectors))
                n.code.as_string = f'{n.code.as_string}\n{oc} = {oc} + 1000.0'
                return 1
        return None


class _InjectInvalidNode(ppl.Pass):
    """Adds an AccessNode referencing non-existent data, making the SDFG
    fail structural validation."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States

    def should_reapply(self, modified) -> bool:
        return False

    def apply_pass(self, sdfg, _):
        st = sdfg.add_state('corrupt_state')
        st.add_node(nodes.AccessNode('this_data_does_not_exist_xyz'))
        return 1


def test_harness_detects_value_corrupting_stage(monkeypatch):
    """Monkeypatch the stage list to a single value-corrupting pass; the
    harness must report a numerical mismatch (and ``first_failing_stage``
    must find it)."""
    monkeypatch.setattr(cdbg, '_build_stages', lambda: [('inject', _PerturbOutputValues())])
    sdfg = elementwise.to_sdfg(simplify=True)
    results = canonicalize_with_stage_checks(sdfg, symbol_value=3)
    assert len(results) == 1
    r = results[0]
    assert r.valid, 'value corruption keeps the SDFG structurally valid'
    assert r.numerically_correct is False, 'harness must catch the value divergence'
    assert r.max_abs_diff is not None and r.max_abs_diff > 1.0


def test_harness_detects_invalid_stage(monkeypatch):
    """Monkeypatch the stage list to a structure-corrupting pass; the
    harness must report the stage as invalid."""
    monkeypatch.setattr(cdbg, '_build_stages', lambda: [('inject', _InjectInvalidNode())])
    sdfg = elementwise.to_sdfg(simplify=True)
    results = canonicalize_with_stage_checks(sdfg, symbol_value=3)
    assert len(results) == 1
    assert results[0].valid is False
    assert not results[0].ok


def test_stop_on_failure_halts_at_first_bad_stage(monkeypatch):
    monkeypatch.setattr(cdbg, '_build_stages', lambda: [('inject', _PerturbOutputValues()),
                                                        ('inject2', _PerturbOutputValues())])
    sdfg = elementwise.to_sdfg(simplify=True)
    results = canonicalize_with_stage_checks(sdfg, symbol_value=3, stop_on_failure=True)
    assert len(results) == 1  # halted after the first failing stage


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
