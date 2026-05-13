"""Top-half-CLOUDSC reproducer for the cloudsc_full xfail.

The full-CLOUDSC test (test_cloudsc_full.py) diverges at PCOVPTOT
26/548 cells with a root-cause cascade traced to a 1-ulp drift in
ZQXN[JM=3] @ JK=15 (NCLDTOP).  Probe + per-loopnest tests have
ruled out the LU solver in isolation, the LU+assembly subsystem in
isolation, and the multi-JK precip chain in isolation -- each passes
at strict rtol=atol=1e-15.  So the bug only manifests when those
subsystems are stitched together.

This reproducer is ``cloudsc.F90`` with the body truncated just
before the SEDIMENTATION block (line 4.2, ~line 2620) and ZSOLQA /
ZSOLQB captured per JK into INTENT(OUT) dummies.  Tests roughly the
top half of the JK loop body (~1100 LoC of physics: source/sink
accumulation into ZSOLQA/ZSOLQB across many sub-loops, plus all of
section 3 + section 4.1 of the kernel) without involving the LU
solver, sedimentation, flux/tendency updates.

If the bridge reproduces a divergence in ZSOLQA / ZSOLQB at strict
tolerance, the bug is in the source/sink accumulation cross-talk.
If it doesn't, the divergence comes from the bottom half (lines
2620..3700) or the assembly/solver/clip combination only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import dace
from _util import build_sdfg, f2py_compile, have_flang
from cloudsc_full._registries import (
    get_inputs,
    get_outputs,
)

_HERE = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.fixture
def _strict_fp_cpu_args():
    prev = dace.Config.get('compiler', 'cpu', 'args')
    dace.Config.set(
        'compiler',
        'cpu',
        'args',
        value='-fPIC -Wall -Wextra -O0 -fno-fast-math -ffp-contract=off '
        '-Wno-unused-parameter -Wno-unused-label',
    )
    try:
        yield
    finally:
        dace.Config.set('compiler', 'cpu', 'args', value=prev)


def _sdfg_call_args(sdfg, scalar_values):
    """Same as test_cloudsc_full -- route scalars to Scalar-or-length-1."""
    from dace.data import Scalar
    arglist = sdfg.arglist()
    out = {}
    for k, v in scalar_values.items():
        desc = arglist.get(k)
        if desc is None or isinstance(desc, Scalar):
            out[k] = v
        else:
            decl_dtype = str(desc.dtype) if hasattr(desc, "dtype") else ""
            if "bool" in decl_dtype.lower():
                out[k] = np.array([bool(v)], dtype=np.bool_)
            elif isinstance(v, float):
                out[k] = np.array([v], dtype=np.float64)
            else:
                out[k] = np.array([v], dtype=np.int32)
    return out


def _lower_keys(d):
    return {k.lower(): v for k, v in d.items()}


def _f2py_argnames(fn):
    import re
    doc = fn.__doc__ or ""
    match = re.match(r"\s*\w+\((.*?)\)", doc, re.DOTALL)
    if not match:
        return set()
    arglist = match.group(1)
    optional = set()
    for m in re.finditer(r"\[([^\]]+)\]", arglist):
        optional.update(s.strip() for s in m.group(1).split(","))
    arglist = re.sub(r"\[[^\]]*\]", "", arglist)
    return {s.strip() for s in arglist.split(",") if s.strip()} | optional


@pytest.fixture(scope="module")
def _f2py_top_half(tmp_path_factory):
    src = (_HERE / "cloudsc_top_half.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_top_half_ref")
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_top_half_ref",
        extra_f90flags="-finit-local-zero -ffree-line-length-none",
        only=("cloudscouter", ),
    )


@pytest.mark.xfail(
    strict=False,
    reason="big top-half reproducer for the cloudsc_full xfail.  Exercises "
    "the source/sink ZSOLQA/ZSOLQB accumulation across ~1100 LoC of body "
    "(sections 3 + 4.1 of CLOUDSC) without the LU solver / sedimentation / "
    "flux blocks.  If this passes at 1e-15, the cross-talk bug is in the "
    "bottom half; if it fails, the bug is in the top half.",
)
def test_cloudsc_top_half_zsolqa_zsolqb(tmp_path, _f2py_top_half, _strict_fp_cpu_args):
    """SDFG-vs-f2py equivalence on ZSOLQA / ZSOLQB after the source/sink
    accumulation block (top half of CLOUDSC's JK loop body)."""
    src = (_HERE / "cloudsc_top_half.F90").read_text()

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="cloudsc_top_half", entry="_QPcloudscouter").build()

    rng = np.random.default_rng(42)
    inputs = get_inputs(rng)
    outputs_ref = {k.lower(): v for k, v in get_outputs(rng).items()}
    outputs_sdfg = {k: v.copy(order="F") for k, v in outputs_ref.items()}

    # Two new INTENT(OUT) dummies on CLOUDSCOUTER from cloudsc_top_half.F90.
    # Shape (KLON, KLEV, NCLV, NCLV, NBLOCKS) matches ZSOLQA per JK per block.
    klon = inputs["KLON"]
    klev = inputs["KLEV"]
    nclv = inputs["NCLV"]
    nblocks = inputs["NBLOCKS"]
    shape = (klon, klev, nclv, nclv, nblocks)
    zsolqa_out_ref = np.zeros(shape, dtype=np.float64, order="F")
    zsolqb_out_ref = np.zeros(shape, dtype=np.float64, order="F")
    zsolqa_out_sdfg = np.zeros_like(zsolqa_out_ref, order="F")
    zsolqb_out_sdfg = np.zeros_like(zsolqb_out_ref, order="F")

    accepted = _f2py_argnames(_f2py_top_half.cloudscouter)
    all_kw_ref = {
        **_lower_keys(inputs),
        **_lower_keys(outputs_ref),
        "zsolqa_out": zsolqa_out_ref,
        "zsolqb_out": zsolqb_out_ref,
    }
    _f2py_top_half.cloudscouter(**{k: v for k, v in all_kw_ref.items() if k in accepted})

    _scalar_types = (bool, int, float, np.bool_, np.integer, np.floating)
    scalar_kwargs = {k.lower(): v for k, v in inputs.items() if isinstance(v, _scalar_types)}
    sdfg_kwargs = {k.lower(): v for k, v in inputs.items() if not isinstance(v, _scalar_types)}
    sdfg_kwargs.update(_lower_keys(outputs_sdfg))
    sdfg_kwargs["zsolqa_out"] = zsolqa_out_sdfg
    sdfg_kwargs["zsolqb_out"] = zsolqb_out_sdfg
    sdfg_kwargs.update(_sdfg_call_args(sdfg, scalar_kwargs))
    sdfg(**sdfg_kwargs)

    # Compare ZSOLQA / ZSOLQB across all JK iterations.  Strict tolerance
    # to catch ulp-level drift the cloudsc_full cascade is rooted in.
    np.testing.assert_allclose(
        zsolqa_out_sdfg,
        zsolqa_out_ref,
        rtol=1e-12,
        atol=1e-12,
        err_msg="ZSOLQA diverges between SDFG and f2py top-half references",
    )
    np.testing.assert_allclose(
        zsolqb_out_sdfg,
        zsolqb_out_ref,
        rtol=1e-12,
        atol=1e-12,
        err_msg="ZSOLQB diverges between SDFG and f2py top-half references",
    )
