"""Bisection step: bottom-UPPER reproducer.

After test_cloudsc_bottom_half.py FAILS with the same 26/548 PCOVPTOT
divergence as test_cloudsc_full, we split the bottom half at the
boundary between physics (sections 4.2-4.5) and solvers/flux/tendency
(sections 5.x, 6.x).

This reproducer keeps the bottom-half physics (sedimentation +
autoconversion + melting + freezing + evaporation, lines 2620-3355
of original cloudsc.F90) and drops the solvers/flux/tendency
(lines 3356-3710), replacing them with a direct
``PCOVPTOT(JL,JK) = ZCOVPTOT(JL)`` writeback.

If this PASSES, the cloudsc bug is in the bottom-lower (solvers +
flux + tendency, lines 3356-3710).  If it FAILS with the same
26/548 PCOVPTOT mismatch, the bug is in the bottom-upper physics.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang
from cloudsc_full._registries import (
    CLOUDSC_F90FLAGS,
    get_inputs_physical,
    get_outputs,
)

_HERE = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


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
def _f2py_bottom_upper(tmp_path_factory):
    src = (_HERE / "cloudsc_bottom_upper.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_bottom_upper_ref")
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_bottom_upper_ref",
        # ``-ffree-line-length-none`` is the sole intentionally
        # gfortran-only flag: it is a non-semantic parser necessity for
        # the long-line cloudsc source; LLVM-flang has no line limit and
        # needs no equivalent.  The FP set is the flang-portable core.
        extra_f90flags=CLOUDSC_F90FLAGS,
        only=("cloudscouter", ),
    )


# Physical (NaN-free) inputs: the bridge is bit-identical to gfortran here.
def test_cloudsc_bottom_upper_numerical(tmp_path, _f2py_bottom_upper, _strict_fp_cpu_args):
    src = (_HERE / "cloudsc_bottom_upper.F90").read_text()

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="cloudsc_bottom_upper", entry="_QPcloudscouter").build()

    rng = np.random.default_rng(42)
    inputs = get_inputs_physical(rng)
    outputs_ref = {k.lower(): v for k, v in get_outputs(rng).items()}
    outputs_sdfg = {k: v.copy(order="F") for k, v in outputs_ref.items()}

    accepted = _f2py_argnames(_f2py_bottom_upper.cloudscouter)
    all_kw_ref = {**_lower_keys(inputs), **_lower_keys(outputs_ref)}
    _f2py_bottom_upper.cloudscouter(**{k: v for k, v in all_kw_ref.items() if k in accepted})

    _scalar_types = (bool, int, float, np.bool_, np.integer, np.floating)
    scalar_kwargs = {k.lower(): v for k, v in inputs.items() if isinstance(v, _scalar_types)}
    sdfg_kwargs = {k.lower(): v for k, v in inputs.items() if not isinstance(v, _scalar_types)}
    sdfg_kwargs.update(_lower_keys(outputs_sdfg))
    sdfg_kwargs.update(_sdfg_call_args(sdfg, scalar_kwargs))
    sdfg(**sdfg_kwargs)

    # Only PCOVPTOT is meaningful here (other outputs zero -- their writes
    # got dropped along with the solver/flux section).
    np.testing.assert_allclose(
        outputs_sdfg["pcovptot"],
        outputs_ref["pcovptot"],
        rtol=1e-15,
        atol=1e-15,
        err_msg="PCOVPTOT mismatch in bottom-upper (sedimentation/physics) reproducer",
    )
