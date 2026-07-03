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


def test_finalize_does_not_persist_reduction_scalar():
    """``finalize_for_target`` must not leave a scalar / length-1 array in persistent
    storage. A size-1 WCR accumulator made persistent lands in the state struct
    (``__state->x``) and breaks the OpenMP ``reduction(op:var)`` clause -- the go_fast
    ``trace`` failure (``reduction(+:__state->__0_trace)`` did not compile). The
    accumulator must stay a non-persistent (register) local."""
    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    Nsym = dace.symbol("N", dtype=dace.int64)

    @dace.program
    def reduce_prog(a: dace.float64[Nsym]):
        s = 0.0
        for i in range(Nsym):
            s += a[i]
        return a + s

    sdfg = reduce_prog.to_sdfg(simplify=False)
    sdfg = canonicalize(sdfg,
                        validate=True,
                        target="cpu",
                        peel_limit=4,
                        break_anti_dependence=True,
                        interchange_carry_with_map=True,
                        scatter_to_guarded_maps=True)
    finalize_for_target(sdfg, "cpu")

    persistent_scalars = [
        f"{sd.label}:{nm}" for sd in sdfg.all_sdfgs_recursive() for nm, desc in sd.arrays.items()
        if desc.transient and desc.total_size == 1 and desc.lifetime == dace.dtypes.AllocationLifetime.Persistent
    ]
    assert not persistent_scalars, f"finalize left size-1 transient(s) persistent: {persistent_scalars}"

    # And it compiles: the reduction accumulator is a local lvalue, not __state->.
    csdfg = sdfg.compile()
    n = 256
    rng = np.random.default_rng(0)
    a = rng.random(n)
    assert np.allclose(csdfg(a=a, N=n), a + a.sum())
