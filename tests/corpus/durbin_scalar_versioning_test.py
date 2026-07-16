# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression: ``PrivatizeScalars`` (``ScalarFission``) must not version the two
sides of a NestedSDFG inout connector into different array versions.

polybench ``durbin`` carries ``beta`` (and ``alpha``) across the outer ``k`` loop
via the in-place recurrence ``beta = (1 - alpha*alpha) * beta``. Late in the
canonicalize pipeline the peeled ``k`` iteration is wrapped in a Map whose
``loop_body`` NestedSDFG reads and writes ``beta`` through ONE inout connector.
The write-shadow analysis, blind to that join, split the read side into ``beta_0``
and the write side into ``beta_1``; ``_rename_memlet_path`` renamed the boundary
memlets, but the connector (which must name the SAME array on both sides) stayed
``beta``, leaving an invalid inout connector

    ValueError: Inout connector beta is connected to different input ({'beta_0'})
    and output ({'beta_1'}) arrays

The failure is deterministic within a process but process-order dependent (it
sometimes wired both sides to the same version). Both tests below run 10x
in-process so a single run reproduces it reliably.
"""
import copy
import os

# Pin a deterministic, single-threaded, no-MPI-init run before DaCe/OpenMP load.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import multiprocessing as mp

import numpy as np
import pytest

from dace.transformation.passes.canonicalize import canonicalize
from tests.corpus.polybench import polybench as PB

#: The CPU canonicalize knobs the numerical gate uses (peeling + anti-dep break
#: are exactly what expose the peeled-``k`` Map + inout-connector shape).
_KNOBS = dict(peel_limit=4, break_anti_dependence=True)


def _durbin_kernel():
    kernels = PB.collect("durbin")
    assert kernels, "polybench durbin kernel not found in the corpus"
    return kernels[0]


def test_durbin_canonicalize_valid_deterministic():
    """canonicalize(durbin) is a VALID SDFG on every run (10x defeats the flake).

    Before the fix this raised the inout-connector ValueError; the split-vs-consistent
    outcome is process-order dependent, so a single 10x in-process loop reproduces it.
    """
    base = PB.fresh_sdfg(_durbin_kernel())
    for _ in range(10):
        # validate=True raises InvalidSDFG* if any stage left the SDFG invalid.
        canonicalize(copy.deepcopy(base), validate=True, validate_all=False, **_KNOBS)


def _compile_and_run(kernel, sdfg, call_arrays, psize, queue):
    """Child-process body: compile + run the candidate, marshal outputs back.

    Repo rule: always fork when running a compiled kernel so a segfault cannot
    take down the pytest process.
    """
    got = PB.run(sdfg, call_arrays, psize)
    queue.put({name: np.asarray(value) for name, value in got.items()})


def test_durbin_canonicalize_bit_exact():
    """canonicalize(durbin) is value-preserving BIT-EXACT vs the untransformed baseline."""
    kernel = _durbin_kernel()
    call_arrays, psize = PB.make_inputs(kernel)
    reference = PB.reference(kernel, call_arrays, psize)

    base = PB.fresh_sdfg(kernel)
    candidate = canonicalize(copy.deepcopy(base), validate=True, **_KNOBS)

    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_compile_and_run, args=(kernel, candidate, call_arrays, psize, queue))
    proc.start()
    got = queue.get(timeout=600)
    proc.join(600)
    assert proc.exitcode == 0, f"candidate kernel run crashed (exit {proc.exitcode})"

    # Bit-exact: same SDFG values as the baseline, zero tolerance (no reassociation
    # is introduced for durbin's carried scalar, so the result is reproduced exactly).
    assert PB.outputs_match(reference, got, rtol=0.0, atol=0.0), "canonicalized durbin is not bit-exact vs reference"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
