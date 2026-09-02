# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression guards on simplifying the CloudSC corpus kernel: what it costs, and what it answers.

CloudSC is the scaling case: thousands of blocks nested many levels deep, where a pass whose cost
is superlinear in nesting depth shows up as minutes rather than seconds. ConstantPropagation is the
dominant term in ``simplify`` on this input.
"""
import copy
import os
import statistics
import time

import numpy as np
import pytest

import dace
from tests.corpus.cloudsc.generate_data_for_cloudsc import (CLOUDSC_INPUT_RANGES, CLOUDSC_INT_RANGES, IEEE_CPU_ARGS,
                                                            build_cloudsc_sdfg, generate_cloudsc_inputs)

#: Wall-clock budget for one ``simplify()`` of CloudSC.
SIMPLIFY_BUDGET_SECONDS: float = 140.0
SIMPLIFY_REPS: int = 5

#: Per-output tolerance for the simplify-vs-reference comparison, taken against that output's own
#: magnitude. Sized for fp64 reassociation and not for bit equality: simplify may reorder an
#: accumulation, and two orderings of the same fp64 sum legitimately differ in the last ulps.
#: Reordering the kernel's longest sums (``klev`` = 32 levels, ``nclv`` = 5 species) costs
#: ``O(n * eps)`` ~ 1e-14 relative, so 1e-12 leaves two orders of headroom while staying far below
#: a semantic divergence, which moves digits and not ulps. As measured on this input set the two
#: sides are in fact bit-identical on every output -- CloudSC parallelises over the horizontal, so
#: no accumulation crosses a thread -- and the headroom is for a future reordering, not cover for a
#: drift that exists today.
PARALLEL_NUMERIC_TOLERANCE: float = 1e-12

#: The three outputs this input set never reaches: the dwarf's reference temperature range is
#: entirely below freezing, so no liquid precipitation forms and its flux, its enthalpy flux and
#: the rain-freezing fraction stay zero. They are still compared -- simplify must not make them
#: nonzero -- but they cannot stand as proof that the kernel computed anything.
UNWRITTEN_OUTPUTS: frozenset = frozenset({'pfhpsl', 'pfplsl', 'prainfrac_toprfz'})


@pytest.mark.long
def test_build_cloudsc_sdfg_hands_out_private_copies():
    """The corpus memoizes the parse; every caller must still get a copy it can transform."""
    first = build_cloudsc_sdfg(simplify=False)
    second = build_cloudsc_sdfg(simplify=False)
    assert first is not second
    first.add_symbol('canary', dace.int32)
    assert 'canary' not in second.symbols


@pytest.mark.long
def test_simplify_stays_within_its_time_budget():
    """One run per copy: ``simplify`` is idempotent, so a second run on the same SDFG measures
    nothing. The median absorbs a slow rep from a loaded machine without needing a wide margin.
    """
    durations = []
    for _ in range(SIMPLIFY_REPS):
        sdfg = build_cloudsc_sdfg(simplify=False)  # deliberately outside the timer
        start = time.perf_counter()
        sdfg.simplify()
        durations.append(time.perf_counter() - start)

    median = statistics.median(durations)
    reps = ', '.join('%.1f' % d for d in durations)
    assert median < SIMPLIFY_BUDGET_SECONDS, (f'median simplify took {median:.1f}s, budget is '
                                              f'{SIMPLIFY_BUDGET_SECONDS:.0f}s; reps={reps}')


@pytest.mark.long
def test_simplify_preserves_cloudsc_numerics_in_parallel():
    """Simplify claims to preserve values; run CloudSC before and after it and check that it does.

    Deliberately on the shipped multicore schedule. Forcing both sides sequential (or pinning
    ``OMP_NUM_THREADS=1``) would make them agree bit-for-bit for free and would grade a
    configuration nobody runs, hiding any defect that only appears once a map is split across
    threads -- so the comparison is at a reassociation-sized tolerance instead.
    """
    assert os.environ.get('OMP_NUM_THREADS') != '1', 'single-threaded, this comparison proves nothing'

    reference = build_cloudsc_sdfg(simplify=False)
    candidate = build_cloudsc_sdfg(simplify=False)
    # Under the 'name' cache config the build folder is just the SDFG name, so equal names collide.
    candidate.name = f'{candidate.name}_simplified'
    candidate.simplify(validate=True)

    # A value comparison also passes when the pass did nothing, so pin that simplify rewrote the
    # graph: on this kernel it takes the control-flow graph from some 6300 blocks to some 560.
    reference_blocks = sum(1 for _ in reference.all_control_flow_blocks())
    assert sum(1 for _ in candidate.all_control_flow_blocks()) < reference_blocks

    inputs = generate_cloudsc_inputs(reference)
    ref_run, cand_run = copy.deepcopy(inputs), copy.deepcopy(inputs)
    saved_args = dace.Config.get('compiler', 'cpu', 'args')
    try:
        # -O0, no fast-math, no FP contraction: the compiler may not reassociate, so whatever
        # difference remains comes from simplify or from the thread schedule, not from -O3.
        dace.Config.set('compiler', 'cpu', 'args', value=IEEE_CPU_ARGS)
        reference(**ref_run)
        candidate(**cand_run)
    finally:
        dace.Config.set('compiler', 'cpu', 'args', value=saved_args)

    # Outputs are the arrays the generator leaves zeroed because the reference has no input range
    # for them. Two sides that both computed nothing agree perfectly, so grade them first.
    outputs = [
        name for name, value in ref_run.items() if isinstance(value, np.ndarray) and value.size > 1
        and name not in CLOUDSC_INPUT_RANGES and name not in CLOUDSC_INT_RANGES
    ]
    assert outputs, 'no output arrays to compare'
    for name in outputs:
        assert np.all(np.isfinite(ref_run[name])), f'{name} is not finite'
        if name not in UNWRITTEN_OUTPUTS:
            assert np.any(ref_run[name] != 0.0), f'{name} is still at its zero initialisation'

    # Graded against each array's own scale: an absolute floor tied to the magnitude of the values
    # keeps an element that cancelled to near-zero from reading as a huge relative error, while the
    # relative term still holds the large elements to the same tolerance.
    failures = []
    for name in outputs:
        ref_val, cand_val = ref_run[name], cand_run[name]
        scale = float(np.max(np.abs(ref_val)))
        if not np.allclose(ref_val, cand_val, rtol=PARALLEL_NUMERIC_TOLERANCE, atol=PARALLEL_NUMERIC_TOLERANCE * scale):
            worst = float(np.max(np.abs(ref_val - cand_val)))
            failures.append(f'{name}: max_abs={worst:.3e} scale={scale:.3e} rel={worst / scale:.3e}')
    assert not failures, 'simplify changed the answer:\n' + '\n'.join(failures)


if __name__ == '__main__':
    test_build_cloudsc_sdfg_hands_out_private_copies()
    test_simplify_stays_within_its_time_budget()
    test_simplify_preserves_cloudsc_numerics_in_parallel()
