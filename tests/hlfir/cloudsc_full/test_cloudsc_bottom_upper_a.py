"""Sub-bisection: bottom-UPPER-A (drops 4.5 EVAPORATION section).

After test_cloudsc_bottom_upper.py FAILS with 51/548 PCOVPTOT mismatch,
this drops the 4.5 EVAPORATION OF RAIN/SNOW section (~330 lines of
the bottom-upper) and keeps only the Sedimentation (4.2) +
Autoconversion (4.3a/b) + Melting (4.4a) + Freezing (4.4b/c) sections.

If this passes, the bug is in 4.5 EVAPORATION.
If it fails, the bug is in 4.2-4.4 (Sedimentation/Autoconv/Melt/Freeze).
"""
from pathlib import Path
import numpy as np
import pytest
from _util import build_sdfg, f2py_compile, have_flang
from cloudsc_full._registries import CLOUDSC_F90FLAGS, get_inputs_physical, get_outputs

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
def _f2py_a(tmp_path_factory):
    src = (_HERE / "cloudsc_bottom_upper_a.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_bottom_upper_a_ref")
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_bottom_upper_a_ref",
        # ``-ffree-line-length-none`` is the sole intentionally
        # gfortran-only flag: it is a non-semantic parser necessity for
        # the long-line cloudsc source; LLVM-flang has no line limit and
        # needs no equivalent.  The FP set is the flang-portable core.
        extra_f90flags=CLOUDSC_F90FLAGS,
        only=("cloudscouter", ))


# Physical (NaN-free) inputs: the bridge is bit-identical to gfortran here.
def test_cloudsc_bottom_upper_a_numerical(tmp_path, _f2py_a, _strict_fp_cpu_args):
    src = (_HERE / "cloudsc_bottom_upper_a.F90").read_text()
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="cloudsc_bottom_upper_a", entry="_QPcloudscouter").build()

    rng = np.random.default_rng(42)
    inputs = get_inputs_physical(rng)
    outputs_ref = {k.lower(): v for k, v in get_outputs(rng).items()}
    outputs_sdfg = {k: v.copy(order="F") for k, v in outputs_ref.items()}

    accepted = _f2py_argnames(_f2py_a.cloudscouter)
    all_kw_ref = {**_lower_keys(inputs), **_lower_keys(outputs_ref)}
    _f2py_a.cloudscouter(**{k: v for k, v in all_kw_ref.items() if k in accepted})

    _scalar_types = (bool, int, float, np.bool_, np.integer, np.floating)
    sk = {k.lower(): v for k, v in inputs.items() if isinstance(v, _scalar_types)}
    sdfg_kwargs = {k.lower(): v for k, v in inputs.items() if not isinstance(v, _scalar_types)}
    sdfg_kwargs.update(_lower_keys(outputs_sdfg))
    sdfg_kwargs.update(_sdfg_call_args(sdfg, sk))
    sdfg(**sdfg_kwargs)

    np.testing.assert_allclose(outputs_sdfg["pcovptot"],
                               outputs_ref["pcovptot"],
                               rtol=1e-15,
                               atol=1e-15,
                               err_msg="PCOVPTOT mismatch in bottom-upper-A (Sed/Autoconv/Melt/Freeze)")
