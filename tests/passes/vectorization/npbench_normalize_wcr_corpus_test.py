# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""npbench corpus: :class:`NormalizeWCR` safety -- value-preservation + idempotency.

``NormalizeWCR`` is a SHARED reduction-normalize pass (canonicalize AND the multi-dim vectorizer
both run it), so a change to it must not regress either consumer. This guards it on the npbench
corpus under the two pipelines it must survive, plus idempotency:

* ``simplify``               -- simplify only, then ``NormalizeWCR``.
* ``simplify_l2m_mapfusion`` -- simplify + ``LoopToMap`` + ``MapFusion`` (the vectorizer's base
  pipeline, :func:`corpus_multidim.base_pipeline`), then ``NormalizeWCR``.

Correctness = the run output still matches the untransformed npbench numpy reference (``NormalizeWCR``
preserves semantics). Idempotency = a SECOND ``NormalizeWCR`` application rewrites nothing (returns
``None``) and leaves a valid SDFG -- a normalized WCR must not re-trigger the pass.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import pytest

from tests.corpus.npbench import npbench
from tests.passes.vectorization.helpers.corpus_multidim import base_pipeline

try:
    from dace.transformation.passes.normalize_wcr import NormalizeWCR
except ImportError:  # pre-rename fallback (same pass, older module name)
    from dace.transformation.passes.normalize_nested_reduction import NormalizeWCR

_CORPUS = {c["name"]: c for c in npbench.collect()}
_KERNELS = sorted(_CORPUS)
_PIPELINES = ("simplify", "simplify_l2m_mapfusion")

_PREP: dict = {}


def _prep(name):
    """Memoized ``(arrays, params, reference)`` for one benchmark."""
    if name not in _PREP:
        c = _CORPUS[name]
        arrays, params = npbench.make_inputs(c)
        _PREP[name] = (arrays, params, npbench.reference_outputs(c, arrays, params))
    return _PREP[name]


def _pipelined_sdfg(name, pipeline):
    """A fresh npbench SDFG through ``pipeline`` (``NormalizeWCR`` not yet applied)."""
    c = _CORPUS[name]
    if pipeline == "simplify":
        return npbench.fresh_sdfg(c, simplify=True)
    sdfg = npbench.fresh_sdfg(c, simplify=False)
    base_pipeline(sdfg)  # simplify + LoopToMap + MapFusion
    return sdfg


@pytest.mark.parametrize("name", _KERNELS)
@pytest.mark.parametrize("pipeline", _PIPELINES)
def test_npbench_normalize_wcr(name, pipeline):
    """``NormalizeWCR`` preserves semantics after ``pipeline`` (value vs npbench reference)."""
    arrays, params, ref = _prep(name)
    sdfg = _pipelined_sdfg(name, pipeline)
    NormalizeWCR().apply_pass(sdfg, {})
    sdfg.validate()
    # Per-(kernel, pipeline) name: concurrent xdist builds must not share .dacecache.
    sdfg.name = f"{sdfg.name}_{pipeline}_nwcr"
    got = npbench.run_outputs(_CORPUS[name], sdfg, arrays, params)
    assert npbench.outputs_match(ref, got), f"{name}/{pipeline}: NormalizeWCR changed output vs npbench reference"


@pytest.mark.parametrize("name", _KERNELS)
@pytest.mark.parametrize("pipeline", _PIPELINES)
def test_npbench_normalize_wcr_idempotent(name, pipeline):
    """A second ``NormalizeWCR`` rewrites nothing (returns ``None``) and leaves a valid SDFG."""
    sdfg = _pipelined_sdfg(name, pipeline)
    NormalizeWCR().apply_pass(sdfg, {})
    sdfg.validate()
    second = NormalizeWCR().apply_pass(sdfg, {})
    sdfg.validate()
    assert second is None, f"{name}/{pipeline}: NormalizeWCR not idempotent -- second pass rewrote {second}"
