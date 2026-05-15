"""Bottom-half-CLOUDSC reproducer for the cloudsc_full xfail.

Companion to ``test_cloudsc_top_half.py``.

The top-half reproducer (lines 1766-2617 of CLOUDSC's body, source/
sink accumulation into ZSOLQA/ZSOLQB) passes at rtol=atol=1e-12 -- the
bridge lowers it bit-correctly.  Therefore the cloudsc_full divergence
has to come from the bottom half (sedimentation + LU solver + flux/
tendency updates, lines 2620-3700).

This reproducer is ``cloudsc.F90`` with the source/sink accumulation
block deleted (lines 1879-2617 of the original).  ZSOLQA / ZSOLQB stay
at the zero values they were reset to just above the deletion point;
the LU solver then factors a near-identity matrix.  The
sedimentation/LU/flux chain still runs over all 122 JK iterations and
produces meaningful PCOVPTOT + flux outputs.

The basic init (ZQXFG, ZTP1, ZRHO, ZA, saturation values) is preserved
above line 1879, so the bottom half has all the local state it needs.

Compares SDFG vs f2py on the same source with identical seeded inputs.
If they disagree at strict tolerance, the bug is in the bottom-half
lowering itself.  If they agree, the cloudsc_full bug only manifests
when source/sink and sedimentation/LU/flux are stitched together --
i.e. it's a cross-talk-only bug.
"""

from pathlib import Path

import numpy as np
import pytest

import dace
from _util import build_sdfg, f2py_compile, have_flang
from cloudsc_full._registries import (
    get_inputs,
    get_outputs,
    program_outputs,
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
        value='-fPIC -Wall -Wextra -O0 -fno-fast-math -ffp-contract=off -frounding-math '
        '-Wno-unused-parameter -Wno-unused-label',
    )
    try:
        yield
    finally:
        dace.Config.set('compiler', 'cpu', 'args', value=prev)


def _sdfg_call_args(sdfg, scalar_values):
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
def _f2py_bottom_half(tmp_path_factory):
    src = (_HERE / "cloudsc_bottom_half.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_bottom_half_ref")
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_bottom_half_ref",
        extra_f90flags="-O0 -fno-fast-math -ffp-contract=off -frounding-math -finit-local-zero -ffree-line-length-none",
        only=("cloudscouter", ),
    )


@pytest.mark.xfail(
    strict=False,
    reason="bottom-half reproducer for the cloudsc_full xfail.  Drops "
    "the source/sink accumulation block (lines 1879-2617 of cloudsc.F90), "
    "so the LU solver factors a near-identity matrix.  The remaining "
    "sedimentation + LU + flux/tendency chain runs over all 122 JK "
    "iterations.  If this passes at 1e-12, the cross-talk between top "
    "and bottom is what triggers the bug.  If it fails, the bug is in "
    "the bottom-half lowering itself.",
)
def test_cloudsc_bottom_half_numerical(tmp_path, _f2py_bottom_half, _strict_fp_cpu_args):
    src = (_HERE / "cloudsc_bottom_half.F90").read_text()

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="cloudsc_bottom_half", entry="_QPcloudscouter").build()

    rng = np.random.default_rng(42)
    inputs = get_inputs(rng)
    outputs_ref = {k.lower(): v for k, v in get_outputs(rng).items()}
    outputs_sdfg = {k: v.copy(order="F") for k, v in outputs_ref.items()}

    accepted = _f2py_argnames(_f2py_bottom_half.cloudscouter)
    all_kw_ref = {**_lower_keys(inputs), **_lower_keys(outputs_ref)}
    _f2py_bottom_half.cloudscouter(**{k: v for k, v in all_kw_ref.items() if k in accepted})

    _scalar_types = (bool, int, float, np.bool_, np.integer, np.floating)
    scalar_kwargs = {k.lower(): v for k, v in inputs.items() if isinstance(v, _scalar_types)}
    sdfg_kwargs = {k.lower(): v for k, v in inputs.items() if not isinstance(v, _scalar_types)}
    sdfg_kwargs.update(_lower_keys(outputs_sdfg))
    sdfg_kwargs.update(_sdfg_call_args(sdfg, scalar_kwargs))
    sdfg(**sdfg_kwargs)

    for name in program_outputs:
        np.testing.assert_allclose(
            outputs_sdfg[name.lower()],
            outputs_ref[name.lower()],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"mismatch on output {name}",
        )
