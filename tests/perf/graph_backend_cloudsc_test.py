# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Correctness gate for the graph-backend benchmark (see graph_backend_cloudsc_bench.py):
running simplify() and then specializing kidia/kfdia/nclv ("config-prop") and fully
unrolling the resulting constant-bounded loops must stay numerically faithful to the
un-transformed CloudSC reference, under EITHER graph backend. This is what makes the
benchmark's timings trustworthy -- a backend that produced a wrong SDFG faster would not
be a real win.
"""
import copy
import importlib.util

import pytest

import dace
from tests.corpus.cloudsc.generate_data_for_cloudsc import (IEEE_CPU_ARGS, build_cloudsc_sdfg, compare_outputs,
                                                    generate_cloudsc_inputs, make_sequential)
from tests.perf.cloudsc_backend_pipeline import filtered_inputs, run_pipeline

_BACKENDS = ['networkx'] + (['rustworkx'] if importlib.util.find_spec('rustworkx') is not None else [])


@pytest.fixture(scope='module')
def reference_sdfg_file(tmp_path_factory):
    """Build the un-transformed CloudSC SDFG once; each backend reloads its own private
    copy (the multi-minute frontend parse is backend-independent, see graph.py's
    OrderedDiGraph._nx staying real networkx unconditionally -- so there's nothing to
    gain from re-parsing per backend, only from re-loading a private mutable copy)."""
    ref = build_cloudsc_sdfg(simplify=False)
    path = str(tmp_path_factory.mktemp('cloudsc') / 'cloudsc_nosimplify.sdfgz')
    ref.save(path, compress=True)
    return path


@pytest.mark.integration
@pytest.mark.parametrize('backend', _BACKENDS)
def test_pipeline_matches_reference(reference_sdfg_file, backend):
    reference = dace.SDFG.from_file(reference_sdfg_file)
    candidate = dace.SDFG.from_file(reference_sdfg_file)

    make_sequential(reference)
    make_sequential(candidate)

    run_pipeline(candidate, backend)

    ref_inputs = generate_cloudsc_inputs(reference, seed=0)
    cand_inputs = filtered_inputs(candidate, copy.deepcopy(ref_inputs))

    saved_args = dace.Config.get('compiler', 'cpu', 'args')
    try:
        dace.Config.set('compiler', 'cpu', 'args', value=IEEE_CPU_ARGS)
        reference(**ref_inputs)
        candidate(**cand_inputs)
    finally:
        dace.Config.set('compiler', 'cpu', 'args', value=saved_args)

    report = compare_outputs(ref_inputs, cand_inputs, rtol=1e-15, atol=1e-15)
    bad = {name: (max_abs, max_rel) for name, (max_abs, max_rel, ok) in report.items() if not ok}
    assert not bad, (f'backend={backend!r}: simplify+config-prop+loopunroll output diverges from the '
                     f'un-transformed reference: {bad}')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
