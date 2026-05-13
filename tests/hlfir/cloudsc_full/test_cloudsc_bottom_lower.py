"""Bottom-LOWER reproducer (companion to bottom-half / bottom-upper).

Drops the bottom-upper physics (sedimentation + autoconversion + melting
+ freezing + evaporation, lines 2620-3355 of cloudsc.F90) and keeps the
solvers + flux + tendency block (lines 3356-3710).

ZSOLQA / ZSOLQB / ZCOVPTOT remain at zero (from their reset above the
deleted region), and the solvers + flux/tendency chain runs on those
zeros.

Expectation: this PASSES at rtol=atol=1e-12.  The bisection chain says:
  * test_cloudsc_top_half          PASS  -> bug in bottom half
  * test_cloudsc_bottom_half       FAIL  -> bug confirmed in bottom half
  * test_cloudsc_bottom_upper      FAIL  -> bug in bottom-upper physics
  * test_cloudsc_bottom_upper_a    PASS  -> bug specifically in 4.5 EVAP
  * test_cloudsc_bottom_lower      PASS  -> bug NOT in solvers/flux/tendency

So this loopnest's role is to confirm the negative: the solvers + flux +
tendency portion of cloudsc lowers bit-correctly under the bridge.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
import dace
from _util import build_sdfg, f2py_compile, have_flang
from cloudsc_full._registries import get_inputs, get_outputs, program_outputs

_HERE = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.fixture
def _strict_fp_cpu_args():
    prev = dace.Config.get('compiler', 'cpu', 'args')
    dace.Config.set('compiler',
                    'cpu',
                    'args',
                    value='-fPIC -Wall -Wextra -O0 -fno-fast-math -ffp-contract=off -frounding-math '
                    '-Wno-unused-parameter -Wno-unused-label')
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
            decl = str(desc.dtype) if hasattr(desc, "dtype") else ""
            if "bool" in decl.lower():
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
    m = re.match(r"\s*\w+\((.*?)\)", doc, re.DOTALL)
    if not m:
        return set()
    al = m.group(1)
    opt = set()
    for mm in re.finditer(r"\[([^\]]+)\]", al):
        opt.update(s.strip() for s in mm.group(1).split(","))
    al = re.sub(r"\[[^\]]*\]", "", al)
    return {s.strip() for s in al.split(",") if s.strip()} | opt


@pytest.fixture(scope="module")
def _f2py_lo(tmp_path_factory):
    src = (_HERE / "cloudsc_bottom_lower.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_bottom_lower_ref")
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_bottom_lower_ref",
        extra_f90flags="-O0 -fno-fast-math -ffp-contract=off -frounding-math -finit-local-zero -ffree-line-length-none",
        only=("cloudscouter", ))


def test_cloudsc_bottom_lower_numerical(tmp_path, _f2py_lo, _strict_fp_cpu_args):
    src = (_HERE / "cloudsc_bottom_lower.F90").read_text()
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="cloudsc_bottom_lower", entry="_QPcloudscouter").build()

    rng = np.random.default_rng(42)
    inputs = get_inputs(rng)
    outputs_ref = {k.lower(): v for k, v in get_outputs(rng).items()}
    outputs_sdfg = {k: v.copy(order="F") for k, v in outputs_ref.items()}

    accepted = _f2py_argnames(_f2py_lo.cloudscouter)
    all_kw_ref = {**_lower_keys(inputs), **_lower_keys(outputs_ref)}
    _f2py_lo.cloudscouter(**{k: v for k, v in all_kw_ref.items() if k in accepted})

    _scalar_types = (bool, int, float, np.bool_, np.integer, np.floating)
    sk = {k.lower(): v for k, v in inputs.items() if isinstance(v, _scalar_types)}
    sdfg_kwargs = {k.lower(): v for k, v in inputs.items() if not isinstance(v, _scalar_types)}
    sdfg_kwargs.update(_lower_keys(outputs_sdfg))
    sdfg_kwargs.update(_sdfg_call_args(sdfg, sk))
    sdfg(**sdfg_kwargs)

    # Expectation: PASS at strict tolerance.  Compare every program_output
    # (the solvers + flux + tendency chain writes them all).
    for name in program_outputs:
        np.testing.assert_allclose(outputs_sdfg[name.lower()],
                                   outputs_ref[name.lower()],
                                   rtol=1e-12,
                                   atol=1e-12,
                                   err_msg=f"PCOVPTOT mismatch in bottom-lower (solvers/flux/tendency only): {name}")
