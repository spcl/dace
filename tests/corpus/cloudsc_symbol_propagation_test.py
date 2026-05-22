# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end numerical check that ``SymbolPropagation`` (run after simplify)
is faithful on the real, inlined CloudSC kernel.

The candidate is exactly the simplified SDFG with ``SymbolPropagation`` applied
on top, compared against the simplified SDFG alone. This isolates symbol
propagation: it rewrites symbolic (integer-exact) quantities only, so it must
not change a single floating-point output. Comparing against the simplified
baseline (rather than the raw, un-simplified SDFG) keeps the comparison clean
of ``simplify``'s own floating-point reassociation. Uses the inlined
``cloudsc_py`` (no callbacks) and the shared data/compare helpers.
"""
import copy

import pytest

from dace.transformation.passes.symbol_propagation import SymbolPropagation
from tests.corpus.generate_data_for_cloudsc import build_cloudsc_sdfg, run_and_compare


@pytest.mark.xfail(strict=True,
                   reason=('SymbolPropagation has a numerical bug: applied to the simplified CloudSC '
                           'SDFG it changes floating-point outputs (8 arrays differ, ~1e-4..1e-6) versus '
                           'the same simplified SDFG without it. It rewrites symbolic (integer-exact) '
                           'quantities only, so it must not change a single float -- this pins the bug as '
                           'a fix target (see CORE_BUGFIXES SymbolPropagation TODO).'))
def test_cloudsc_symbol_propagation_is_numerically_faithful():
    """``SymbolPropagation`` applied to a simplified CloudSC SDFG produces
    bit-faithful outputs versus the same simplified SDFG without it."""
    reference = build_cloudsc_sdfg(simplify=True)

    candidate = copy.deepcopy(reference)
    SymbolPropagation().apply_pass(candidate, {})
    candidate.validate()

    assert run_and_compare(reference, candidate, verbose=True), \
        'SymbolPropagation changed CloudSC outputs vs the simplified reference'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
