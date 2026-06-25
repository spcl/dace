# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end integration test: a transform pipeline on CloudSC stays numerically
faithful to the un-transformed reference, step by step.

Starting from the simplified CloudSC SDFG, the chain applies
``UniqueLoopIterators`` (with post-value materialization) -> ``SymbolPropagation``
-> ``simplify`` -> ``LoopToMap``, and after **each** step re-runs the candidate on
identical physical inputs and compares every output array to the non-transformed
reference (built with ``simplify=False``).

It runs in two build regimes (see :func:`~tests.corpus.cloudsc.generate_data_for_cloudsc`):

* ``ieee``    -- ``-O0``, no fast-math, no FP contraction, sequential schedules.
  The transforms are value-preserving, so this reproduces the reference
  *bit-for-bit* (tolerance ``1e-15``).
* ``release`` -- the configured ``-O3 -ffast-math`` flags, parallel schedules.
  The transcendental intrinsics are approximated and the flux prefix sums and
  parallel reductions reordered, so agreement is to ~``1e-10``.

This is a slow integration test: it builds the full CloudSC SDFG once
(``simplify=False`` parse is minutes) and compiles it several times.
"""
import contextlib
import copy
import gc
import os

import pytest

import dace
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes import SymbolPropagation
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators
from tests.corpus.cloudsc.generate_data_for_cloudsc import (IEEE_CPU_ARGS, build_cloudsc_sdfg, compare_outputs,
                                                    generate_cloudsc_inputs, make_sequential)

#: (ieee_build, sequential, rtol, atol) per regime.
_REGIMES = {
    'ieee': (True, True, 1e-15, 1e-15),
    'release': (False, False, 1e-10, 1e-10),
}


def _apply_uli(sdfg: dace.SDFG):
    UniqueLoopIterators(assign_loop_iterator_post_value=True).apply_pass(sdfg, {})


def _apply_symbol_propagation(sdfg: dace.SDFG):
    SymbolPropagation().apply_pass(sdfg, {})


def _apply_simplify(sdfg: dace.SDFG):
    sdfg.simplify()


def _apply_loop_to_map(sdfg: dace.SDFG):
    # LoopToMap logs every refused loop; keep the test output readable.
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        sdfg.apply_transformations_repeated(LoopToMap)


#: Ordered chain of (label, in-place transform) applied to the candidate.
_CHAIN = [
    ('uli_postamble', _apply_uli),
    ('symbol_propagation', _apply_symbol_propagation),
    ('simplify', _apply_simplify),
    ('loop_to_map', _apply_loop_to_map),
]


def _run(sdfg: dace.SDFG, inputs, ieee_build: bool, sequential: bool, tag: str):
    """Run ``sdfg`` once on a private copy of ``inputs`` under the given build
    regime, returning the mutated buffers. ``sdfg`` is renamed (fresh build dir)
    and run in place; the prior ``compiler.cpu.args`` is restored afterwards."""
    sdfg.name = f'cloudsc_chain_{tag}'
    if sequential:
        make_sequential(sdfg)
    saved_args = dace.Config.get('compiler', 'cpu', 'args')
    try:
        if ieee_build:
            dace.Config.set('compiler', 'cpu', 'args', value=IEEE_CPU_ARGS)
        args = copy.deepcopy(inputs)
        sdfg(**args)
    finally:
        dace.Config.set('compiler', 'cpu', 'args', value=saved_args)
    return args


@pytest.fixture(scope='module')
def reference_sdfg_file(tmp_path_factory):
    """Build the un-transformed CloudSC SDFG once and persist it; each regime
    reloads it (the build's multi-minute parse is shared across regimes)."""
    ref = build_cloudsc_sdfg(simplify=False)
    path = str(tmp_path_factory.mktemp('cloudsc') / 'cloudsc_nosimplify.sdfgz')
    ref.save(path, compress=True)
    return path


@pytest.mark.integration
@pytest.mark.parametrize('regime', list(_REGIMES))
def test_cloudsc_transform_chain(reference_sdfg_file, regime):
    ieee_build, sequential, rtol, atol = _REGIMES[regime]

    ref = dace.SDFG.from_file(reference_sdfg_file)
    inputs = generate_cloudsc_inputs(ref, seed=0)
    reference_out = _run(ref, inputs, ieee_build, sequential, tag=f'{regime}_ref')
    del ref
    gc.collect()

    candidate = dace.SDFG.from_file(reference_sdfg_file)
    candidate.simplify()

    for label, transform in _CHAIN:
        transform(candidate)
        candidate.validate()
        out = _run(candidate, inputs, ieee_build, sequential, tag=f'{regime}_{label}')
        report = compare_outputs(out, reference_out, rtol=rtol, atol=atol)
        bad = {name: (max_abs, max_rel) for name, (max_abs, max_rel, ok) in report.items() if not ok}
        assert not bad, (f'{regime}/{label}: outputs diverge from the un-transformed reference '
                         f'(rtol={rtol}, atol={atol}): {bad}')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
