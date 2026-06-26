# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for the (numerically verified) canonicalization/frontend fixes
landed for the npbench+polybench corpus.

Only the fixes confirmed value-preserving are pinned here:

* ``_find_new_name`` is connector-aware (sdfg.py) -- fixed symm.
* bare-name dtype casts (``f32(0)`` where ``f32 = dace.float32``) parse
  (newast.py) -- fixed azimint_naive/resnet parsing.

NOTE: three earlier fix attempts (expand_nested_sdfg_inputs WCR handling,
split_tasklets symbol-only-substatement anchoring, isolate_nested_sdfg re-clone
guard) were REVERTED -- they let canonicalization *complete* but produced silent
miscompiles (atax/cholesky/correlation/covariance numerically wrong, correlation
-inf). Loud failure beats a silent wrong answer; those kernels canon-fail again
until a value-preserving fix is found. Do not re-add their tests without a numeric
(canon-output == baseline-output) assertion.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np

import dace


def test_find_new_name_avoids_tasklet_connector():
    """``_find_new_name`` must not return a name already used as a tasklet
    connector (regression: ``add_scalar('tmp', find_new_name=True)`` collided with
    a ``tmp`` connector -> 'Connector already used as a symbol' in symm)."""
    sdfg = dace.SDFG("find_new_name_conn")
    state = sdfg.add_state()
    tasklet = state.add_tasklet("t", set(), {"tmp"}, "tmp = 0.0")
    sdfg.add_scalar("out", dace.float64, transient=True)
    state.add_edge(tasklet, "tmp", state.add_access("out"), None, dace.Memlet("out[0]"))

    new_name = sdfg.add_scalar("tmp", dace.float64, transient=True, find_new_name=True)
    assert new_name != "tmp"
    assert "tmp" not in sdfg.arrays


def test_bare_name_dtype_cast_parses():
    """A dtype typeclass bound to a plain name and used as a cast (``f32(0)``)
    must parse like the attribute form ``dace.float32(0)`` (regression: npbench
    ``dc_float(0)`` -> 'Unexpected call expression type Constant: float')."""
    N = dace.symbol("N")
    f32 = dace.float32

    @dace.program
    def caster(a: dace.float64[N]):
        return a + f32(0)

    sdfg = caster.to_sdfg()  # used to raise TypeError during parsing
    a = np.ones(8, dtype=np.float64)
    assert np.allclose(sdfg(a=a, N=8), a)
