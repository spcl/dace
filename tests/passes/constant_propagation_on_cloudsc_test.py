# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ConstantPropagation on the CloudSC corpus kernel.

CloudSC is the scaling case for this pass: thousands of blocks and loop regions nested many levels
deep, which is where the collection schedule dominates the runtime. The last test closes the loop
by compiling and running, so a schedule change that quietly altered a value cannot pass.
"""
import pytest

import dace
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising
from tests.corpus.cloudsc.generate_data_for_cloudsc import build_cloudsc_sdfg, run_and_compare


def promoted_cloudsc() -> dace.SDFG:
    """CloudSC as SimplifyPass hands it to ConstantPropagation.

    Without the promotion no symbol carries a constant and the pass early-exits.
    """
    sdfg = build_cloudsc_sdfg(simplify=False)
    ScalarToSymbolPromotion().apply_pass(sdfg, {})
    ControlFlowRaising().apply_pass(sdfg, {})
    return sdfg


@pytest.mark.long
def test_constant_propagation_cloudsc():
    sdfg = promoted_cloudsc()

    propagated = ConstantPropagation().apply_pass(sdfg, {})
    assert propagated, 'expected constants to propagate in cloudsc'
    sdfg.validate()

    # No constant assignment may survive on an interstate edge for a symbol that was propagated away.
    propagated_symbols = {sym for cfg_id, sym in propagated if cfg_id == sdfg.cfg_id}
    for edge in sdfg.all_interstate_edges():
        assert not (propagated_symbols & edge.data.assignments.keys())

    # An empty set here instead of None would spin SimplifyPass's FixedPointPipeline forever.
    assert ConstantPropagation().apply_pass(sdfg, {}) is None


@pytest.mark.long
def test_constant_propagation_cloudsc_is_numerically_faithful():
    """End to end: promote, propagate, compile, run, compare against the un-propagated kernel."""
    reference = build_cloudsc_sdfg(simplify=False)
    candidate = promoted_cloudsc()
    # Under the 'name' cache config the build folder is just the SDFG name, so equal names collide.
    candidate.name = f'{candidate.name}_propagated'

    assert ConstantPropagation().apply_pass(candidate, {})
    assert run_and_compare(reference, candidate)


@pytest.mark.long
def test_simplified_cloudsc_is_numerically_faithful():
    """The same check through the full ``simplify``, which runs ConstantPropagation in a pipeline."""
    reference = build_cloudsc_sdfg(simplify=False)
    candidate = build_cloudsc_sdfg(simplify=False)
    candidate.name = f'{candidate.name}_simplified'

    candidate.simplify(validate=True)
    assert sum(1 for _ in candidate.all_control_flow_blocks()) < sum(1 for _ in reference.all_control_flow_blocks())

    assert run_and_compare(reference, candidate)


if __name__ == '__main__':
    test_constant_propagation_cloudsc()
    test_constant_propagation_cloudsc_is_numerically_faithful()
    test_simplified_cloudsc_is_numerically_faithful()
