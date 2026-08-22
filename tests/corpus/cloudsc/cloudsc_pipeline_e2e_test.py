# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end integration test for the CloudSC optimization pipelines (:mod:`pipelines`).

Drives each of the three variants -- ``parallelize`` / ``canon_cpu`` / ``canon_gpu`` -- on the
un-simplified python-frontend CloudSC SDFG to MAXIMUM parallelism, checking correctness at **every
phase boundary**: after each phase the transformed graph is run on the same physical inputs and every
output array is compared to the un-transformed reference (bit-exact under IEEE on value-preserving
phases, relaxed from the first reassociating phase). A divergence is pinned to the exact phase that
introduced it, and the per-phase ``.sdfgz`` checkpoints left under the dump dir let you reload and
post-mortem any stage.

This extends the numeric coverage the ``cloudsc_parallelize_chain_test`` gives (parallelize only) to
the two canonicalization variants as well.

Slow: the ``simplify=False`` parse is minutes and each phase boundary compiles + runs CloudSC
(``canon_*`` has many phases). Built once, shared across variants; the reference is run once.

Manual run (single-core, IEEE, 8 GB cap -- see README)::

    OMP_NUM_THREADS=1 pytest tests/corpus/cloudsc/cloudsc_pipeline_e2e_test.py -v -s -m integration -n1

    # one variant:
    OMP_NUM_THREADS=1 pytest tests/corpus/cloudsc/cloudsc_pipeline_e2e_test.py -v -s -m integration \\
        -k parallelize
"""
import gc

import pytest

import dace
from tests.corpus.cloudsc.generate_data_for_cloudsc import build_cloudsc_sdfg
from tests.corpus.cloudsc.pipelines import (VARIANTS, build_reference_outputs, numeric_check_from, run_pipeline,
                                            uniquely_named)

#: Species PARAMETER constants baked in as the ``specialize`` phase (config propagation), so the
#: species / LU loops are constant-trip. Matches the parallelize chain test's specialization.
_CONSTANTS = {'nclv': 5}

#: IEEE, single-core, deterministic -- value-preserving phases stay bit-exact to the reference.
_REGIME = 'ieee'


@pytest.fixture(scope='module')
def reference_path(tmp_path_factory):
    """Build the un-transformed CloudSC SDFG once and persist it; every variant reloads a fresh copy
    (the multi-minute ``simplify=False`` parse is shared)."""
    ref = build_cloudsc_sdfg(simplify=False)
    path = str(tmp_path_factory.mktemp('cloudsc') / 'cloudsc_nosimplify.sdfgz')
    ref.save(path, compress=True)
    del ref
    gc.collect()
    return path


@pytest.fixture(scope='module')
def reference_bundle(reference_path):
    """Run the un-transformed reference ONCE (IEEE, sequential); share ``(inputs, reference_out)``
    across every variant -- all three must reproduce this same output."""
    ref = dace.SDFG.from_file(reference_path)
    inputs, reference_out = build_reference_outputs(ref, regime=_REGIME, seed=0)
    del ref
    gc.collect()
    return inputs, reference_out


@pytest.mark.integration
@pytest.mark.parametrize('variant', VARIANTS)
def test_pipeline_numeric_e2e(variant, reference_path, reference_bundle, tmp_path):
    inputs, reference_out = reference_bundle
    # Fresh check per variant (its own sticky strict->relaxed tolerance state).
    check = numeric_check_from(inputs, reference_out, regime=_REGIME)

    sdfg = uniquely_named(dace.SDFG.from_file(reference_path), f'cloudsc_{variant}')
    # resume=False: a test always runs the full pipeline (checkpoints are for the interactive driver,
    # where the plan signature guards against stale reuse; here we want every phase re-verified).
    run_pipeline(sdfg,
                 variant,
                 tmp_path / 'dump',
                 constants=_CONSTANTS,
                 tag=f'{variant}_python',
                 numeric_check=check,
                 resume=False)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'integration'])
