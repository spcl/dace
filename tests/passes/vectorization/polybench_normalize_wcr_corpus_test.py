# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""polybench corpus: :class:`NormalizeWCR` safety -- value-preservation + idempotency.

``NormalizeWCR`` is a SHARED reduction-normalize pass (canonicalize AND the multi-dim vectorizer
both run it), so a change to it must not regress either consumer. This guards it on the polybench
corpus under the two pipelines it must survive, plus idempotency:

* ``simplify``               -- simplify only, then ``NormalizeWCR``.
* ``simplify_l2m_mapfusion`` -- simplify + ``LoopToMap`` + ``MapFusion`` (the vectorizer's base
  pipeline, :func:`corpus_multidim.base_pipeline`), then ``NormalizeWCR``.

polybench ships no numpy oracle, so correctness = value-preservation: the run output after
``NormalizeWCR`` must equal the untransformed baseline SDFG output on identical inputs.
Idempotency = a SECOND ``NormalizeWCR`` application rewrites nothing (returns ``None``) and leaves a
valid SDFG -- a normalized WCR must not re-trigger the pass.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import pytest

from tests.corpus.polybench import polybench
from tests.passes.vectorization.helpers.corpus_multidim import base_pipeline

try:
    from dace.transformation.passes.normalize_wcr import NormalizeWCR
except ImportError:  # pre-rename fallback (same pass, older module name)
    from dace.transformation.passes.normalize_nested_reduction import NormalizeWCR

_KERNELS = [k.name for k in polybench.collect()]
_PIPELINES = ("simplify", "simplify_l2m_mapfusion")

_PREP: dict = {}


def _prep(name):
    """Memoized ``(kernel, call_arrays, psize, baseline_reference)`` for one kernel."""
    if name not in _PREP:
        kernel = polybench.collect(name=name)[0]
        call_arrays, psize = polybench.make_inputs(kernel)
        _PREP[name] = (kernel, call_arrays, psize, polybench.reference(kernel, call_arrays, psize))
    return _PREP[name]


def _pipelined_sdfg(kernel, pipeline):
    """A fresh polybench SDFG through ``pipeline`` (``NormalizeWCR`` not yet applied)."""
    if pipeline == "simplify":
        return polybench.fresh_sdfg(kernel, simplify=True)
    sdfg = polybench.fresh_sdfg(kernel, simplify=False)
    base_pipeline(sdfg)  # simplify + LoopToMap + MapFusion
    return sdfg


@pytest.mark.parametrize("name", _KERNELS)
@pytest.mark.parametrize("pipeline", _PIPELINES)
def test_polybench_normalize_wcr(name, pipeline):
    """``NormalizeWCR`` preserves semantics after ``pipeline`` (value vs untransformed baseline)."""
    kernel, call_arrays, psize, ref = _prep(name)
    sdfg = _pipelined_sdfg(kernel, pipeline)
    NormalizeWCR().apply_pass(sdfg, {})
    sdfg.validate()
    # Per-(kernel, pipeline) name: concurrent xdist builds must not share .dacecache.
    sdfg.name = f"{sdfg.name}_{pipeline}_nwcr"
    got = polybench.run(sdfg, call_arrays, psize)
    assert polybench.outputs_match(ref, got), f"{name}/{pipeline}: NormalizeWCR changed output vs baseline"


@pytest.mark.parametrize("name", _KERNELS)
@pytest.mark.parametrize("pipeline", _PIPELINES)
def test_polybench_normalize_wcr_idempotent(name, pipeline):
    """A second ``NormalizeWCR`` rewrites nothing (returns ``None``) and leaves a valid SDFG."""
    kernel, _call_arrays, _psize, _ref = _prep(name)
    sdfg = _pipelined_sdfg(kernel, pipeline)
    NormalizeWCR().apply_pass(sdfg, {})
    sdfg.validate()
    second = NormalizeWCR().apply_pass(sdfg, {})
    sdfg.validate()
    assert second is None, f"{name}/{pipeline}: NormalizeWCR not idempotent -- second pass rewrote {second}"
