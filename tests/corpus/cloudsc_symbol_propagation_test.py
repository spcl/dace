# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end guard that ``SymbolPropagation`` does not corrupt the real,
inlined CloudSC kernel.

On simplified CloudSC, symbol propagation is in fact a no-op: its only
propagatable symbols (``kfdia_plus_1_N = kfdia + 1`` etc.) reference the
horizontal-bound scalar ARGUMENTS ``kidia`` / ``kfdia``, which are ``dt.Scalar``
and (correctly) skipped by the scalar filter -- the default
``ScalarToSymbolPromotion`` does not promote argument scalars. This test
therefore checks that applying the pass leaves CloudSC bit-faithful (a
regression guard should symbol propagation ever start touching it). The actual
propagation behaviour after promoting the scalar arguments to symbols is unit-
tested in ``tests/passes/symbol_propagation_test.py``
(``test_cloudsc_kidia_kfdia_promote_then_propagate``).

The comparison runs both SDFGs **sequentially** (``run_and_compare`` forces
sequential schedules): CloudSC's parallel maps reorder floating-point
reductions run-to-run, which alone makes two identical computations differ by
~1e-5 -- noise that would mask a real numerical difference. Uses the inlined
``cloudsc_py`` (no callbacks) and the shared data/compare helpers.
"""
import copy

import pytest

from dace.transformation.passes.symbol_propagation import SymbolPropagation
from tests.corpus.generate_data_for_cloudsc import build_cloudsc_sdfg, run_and_compare


def test_cloudsc_symbol_propagation_is_numerically_faithful():
    """``SymbolPropagation`` leaves a simplified CloudSC SDFG bit-faithful
    versus the same SDFG without it (sequential execution, so the comparison is
    deterministic)."""
    reference = build_cloudsc_sdfg(simplify=True)

    candidate = copy.deepcopy(reference)
    SymbolPropagation().apply_pass(candidate, {})
    candidate.validate()

    assert run_and_compare(reference, candidate, verbose=True), \
        'SymbolPropagation changed CloudSC outputs vs the simplified reference'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
