# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TSVC corpus: base(``simplify`` + ``LoopToMap`` + ``MapFusion``) then multi-dim tile-op vectorize; numerical
correctness vs numpy reference.

TSVC sibling of ``npbench_simplify_multidim_vectorize_corpus_test`` / ``polybench_..._test``: the *light* P1 path
(no canonicalize). Fresh SDFG per kernel through the shared
:func:`tests.passes.vectorization.helpers.corpus_multidim.base_pipeline` (``simplify`` -> ``LoopToMap`` ->
``MapFusion``, value-preserving); each config deep-copies the base and is checked e2e vs numpy
(:data:`tests.corpus.tsvc.tsvc_numpy.REFERENCES`).

Base is the numerically-checkable root: a recurrence that does not parallelize under plain ``LoopToMap`` stays a loop,
the tile vectorizer no-ops, ``base`` == config. Configs only tile the maps that formed -- a strictly *easier* surface
than the canonicalize path (which forms more maps), so this corpus is the floor the multi-dim CPU vectorizer must hold.

Scope: multi-dim CPU tile path only.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import copy

import numpy as np
import pytest

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES
from tests.passes.vectorization.helpers.corpus_multidim import PHASES, base_pipeline, make_pass, select_widths

_KERNELS = [k.name for k in tsvc.collect()]

_BASE: dict = {}


def _base(name):
    """Memoized ``(base_sdfg, arrays, call_kwargs, reference, widths)`` for one kernel.

    Base built once (fresh un-simplified ``to_sdfg`` + ``base_pipeline``); each phase deep-copies it. Reference =
    numpy oracle run in place on a private copy of the inputs.
    """
    if name not in _BASE:
        kernel = tsvc.collect(name=name)[0]
        sdfg = tsvc.to_sdfg(kernel, tag="p1_base", simplify=False)
        base_pipeline(sdfg)
        arrays, ck = tsvc.make_inputs(kernel, seed=1234)
        ref = {n: a.copy() for n, a in arrays.items()}
        REFERENCES[name](**ref, **ck)
        _BASE[name] = (sdfg, arrays, ck, ref, select_widths(sdfg))
    return _BASE[name]


def _assert_matches(name: str, got: dict, ref: dict, phase: str):
    """Every float output array must match the reference (``nan``/``inf`` equal).

    ``np.allclose(equal_nan=True)`` so a recurrence that overflows to ``inf`` compares ``inf == inf`` rather than
    ``nanmax(inf - inf)``. Integer arrays are read-only indices, skipped.
    """
    for n, a in got.items():
        if not (isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.floating) and a.size):
            continue
        assert np.allclose(np.asarray(a), np.asarray(ref[n]), rtol=1e-9, atol=1e-9, equal_nan=True), (
            f"{name}/{n}: {phase} diverges from numpy reference, "
            f"max|diff|={np.nanmax(np.abs(np.asarray(a) - np.asarray(ref[n]))):.3e}")


@pytest.mark.parametrize("name", _KERNELS)
@pytest.mark.parametrize("phase", PHASES)
def test_tsvc_corpus(name, phase):
    """base(simplify+loop2map+mapfusion) [+ multidim vectorize] -> verify vs numpy."""
    base, arrays, ck, ref, widths = _base(name)
    sdfg = copy.deepcopy(base)
    # Per-(kernel, phase) name so two phases building concurrently under -n xdist don't collide on shared .dacecache.
    sdfg.name = f"{sdfg.name}_{phase}"
    if phase != "base":
        make_pass(widths, phase).apply_pass(sdfg, {})
    sdfg.validate()
    work = {n: a.copy() for n, a in arrays.items()}
    sdfg.compile()(**work, **ck)
    _assert_matches(name, work, ref, phase)
