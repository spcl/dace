"""End-to-end SDFG verification for the six E6 velocity-advection
representative loopnests.

Complementary to ``test_equivalence.py`` (which only checks struct-vs-flat
equivalence at the Fortran level): this file runs the flat kernel through
our HLFIR → SDFG pipeline, calls the compiled SDFG from Python, and
compares the output against an f2py reference built from the same
Fortran source on identical seeded inputs.

Per the project's E2E-verification rule every frontend test that emits
an SDFG must compare against a non-transformed reference — structural
assertions on their own don't catch numerical bugs.

The flat kernel sources are extracted from ``loopnest_N.f90`` so the
bundle (struct + flat + gfortran driver) stays the single source of
truth; the Python test just slices out the flat subroutine for f2py
and HLFIR.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

_HERE = Path(__file__).resolve().parent

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _extract_flat_kernel(bundle_path: Path) -> str:
    """Slice the ``subroutine kernel_flat(...) ... end subroutine`` block
    out of a loopnest bundle so f2py and our HLFIR bridge both see a
    self-contained, standalone Fortran source.

    The bundle wraps the kernel inside a ``contains`` of a module; f2py
    is happy to compile a bare subroutine with explicit-shape dummies so
    stripping the module wrapper is fine.
    """
    text = bundle_path.read_text()
    pattern = re.compile(r"(?is)(subroutine\s+kernel_flat\s*\([^)]*\).*?\bend\s+subroutine)", )
    m = pattern.search(text)
    if not m:
        raise RuntimeError(f"kernel_flat not found in {bundle_path}")
    return m.group(1)


def _f2py_build(src_text: str, out_dir: Path, mod_name: str):
    """f2py-compile ``src_text`` as ``mod_name`` into ``out_dir`` and
    return the imported Python module.  Skips if the toolchain's missing."""
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / f"{mod_name}.f90"
    src.write_text(src_text)
    subprocess.check_call(
        [sys.executable, "-m", "numpy.f2py", "-c",
         str(src), "-m", mod_name, "--quiet"],
        cwd=out_dir,
    )
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


def _sdfg_from_flat(flat_src: str, tmp: Path, name: str):
    """Build the SDFG from the flat kernel source using the minimal
    ``hlfir-propagate-shapes`` pipeline (matches the existing
    ported_from_f2dace_windmill convention)."""
    tmp.mkdir(parents=True, exist_ok=True)
    return build_sdfg(flat_src, tmp, name=name, pipeline="hlfir-propagate-shapes").build()


def _preload_gomp():
    """DaCe-compiled SOs dynamically resolve ``omp_get_max_threads`` —
    dlopen libgomp with RTLD_GLOBAL so later ctypes.CDLL calls on the
    DaCe stub find the symbol.  LD_PRELOAD can't help here because
    we're past process-launch."""
    import ctypes
    for cand in ("libgomp.so.1", "/usr/lib/x86_64-linux-gnu/libgomp.so.1", "/usr/lib64/libgomp.so.1"):
        try:
            ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


_preload_gomp()

# ---------------------------------------------------------------------------
# Loopnest 2 — direct stencil, partial vertical
# ---------------------------------------------------------------------------


def test_loopnest_2_sdfg_matches_f2py(tmp_path: Path):
    """z_w_concorr_me = vn*ddxn + vt*ddxt  (jk = nflatlev..nlev)"""
    bundle = _HERE / "loopnest_2.f90"
    flat_src = _extract_flat_kernel(bundle)

    # f2py reference — built once, called repeatedly would be cheaper but
    # the pytest setup already imports the module on first call.
    ref = _f2py_build(flat_src, tmp_path / "ref", "kernel_flat_2")
    sdfg = _sdfg_from_flat(flat_src, tmp_path / "sdfg", name="kernel_flat")

    rng = np.random.default_rng(2)
    nproma, nlev, nblks_e, nflatlev = 32, 16, 8, 4

    # Fortran-order arrays so numpy's memory layout matches what both
    # f2py and DaCe's descriptor expect (nproma is fastest-varying).
    vn = np.asfortranarray(rng.random((nproma, nlev, nblks_e), dtype=np.float64))
    vt = np.asfortranarray(rng.random((nproma, nlev, nblks_e), dtype=np.float64))
    ddxn = np.asfortranarray(rng.random((nproma, nlev, nblks_e), dtype=np.float64))
    ddxt = np.asfortranarray(rng.random((nproma, nlev, nblks_e), dtype=np.float64))

    z_ref = np.zeros_like(vn, order="F")
    z_sdfg = np.zeros_like(vn, order="F")

    # f2py auto-derives nproma/nlev/nblks_e from array shapes and drops
    # them from the positional-arg list — see kernel_flat.__doc__.
    ref.kernel_flat(vn, vt, ddxn, ddxt, z_ref, nflatlev, 1, nblks_e, 1, nproma)
    assert z_ref.any(), "f2py reference didn't write to z_ref"

    # Our frontend classifies integer dummies that appear in array
    # bounds / loop conditions as DaCe symbols (passed as plain ints),
    # and the remaining integer dummies as length-1 ``Array``s (passed
    # as numpy arrays).  Introspect sdfg.arglist() to route each arg
    # to the right form without hard-coding the split.
    from dace.data import Scalar
    int_args = dict(nproma=nproma,
                    nlev=nlev,
                    nblks_e=nblks_e,
                    nflatlev=nflatlev,
                    i_startblk=1,
                    i_endblk=nblks_e,
                    i_startidx=1,
                    i_endidx=nproma)
    call_kwargs = dict(vn=vn, vt=vt, ddxn=ddxn, ddxt=ddxt, z_w_concorr_me=z_sdfg)
    arglist = sdfg.arglist()
    for k, v in int_args.items():
        desc = arglist.get(k)
        if desc is None or isinstance(desc, Scalar):
            call_kwargs[k] = v  # plain int for scalars / symbols
        else:
            call_kwargs[k] = np.array([v], dtype=np.int32)  # length-1 Array
    sdfg(**call_kwargs)

    np.testing.assert_allclose(z_sdfg, z_ref, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Helpers shared by the remaining loopnest tests
# ---------------------------------------------------------------------------


def _sdfg_call_args(sdfg, int_values: dict) -> dict:
    """Route each integer arg in ``int_values`` to either a plain int or
    a length-1 numpy array, depending on whether the SDFG descriptor
    classifies it as a symbol/scalar or a length-1 Array."""
    from dace.data import Scalar
    arglist = sdfg.arglist()
    out = {}
    for k, v in int_values.items():
        desc = arglist.get(k)
        if desc is None or isinstance(desc, Scalar):
            out[k] = v
        else:
            out[k] = np.array([v], dtype=np.int32)
    return out


# ---------------------------------------------------------------------------
# Loopnest 3 — direct stencil with deepatmo vertical profiles
# ---------------------------------------------------------------------------


def test_loopnest_3_sdfg_matches_f2py(tmp_path: Path):
    """z_v_grad_w = z_v_grad_w*gradh(jk) + vn_ie*(...) + z_vt_ie*(...)"""
    bundle = _HERE / "loopnest_3.f90"
    flat_src = _extract_flat_kernel(bundle)
    ref = _f2py_build(flat_src, tmp_path / "ref", "kernel_flat_3")
    sdfg = _sdfg_from_flat(flat_src, tmp_path / "sdfg", name="kernel_flat_3")

    rng = np.random.default_rng(3)
    nproma, nlev, nblks_e = 32, 16, 8

    def _f(shape):
        return np.asfortranarray(rng.random(shape, dtype=np.float64))

    vn_ie = _f((nproma, nlev, nblks_e))
    z_vt_ie = _f((nproma, nlev, nblks_e))
    ft_e = _f((nproma, nblks_e))
    fn_e = _f((nproma, nblks_e))
    gradh = _f((nlev, ))
    invr = _f((nlev, ))
    z_init = _f((nproma, nlev, nblks_e))
    z_ref = np.array(z_init, order="F")
    z_sdfg = np.array(z_init, order="F")

    ref.kernel_flat(vn_ie, z_vt_ie, ft_e, fn_e, gradh, invr, z_ref, 1, nblks_e, 1, nproma)

    kw = dict(vn_ie=vn_ie,
              z_vt_ie=z_vt_ie,
              ft_e=ft_e,
              fn_e=fn_e,
              gradh=gradh,
              invr=invr,
              z_v_grad_w=z_sdfg,
              nproma=nproma,
              nlev=nlev,
              nblks_e=nblks_e)
    kw.update(_sdfg_call_args(sdfg, dict(i_startblk=1, i_endblk=nblks_e, i_startidx=1, i_endidx=nproma)))
    sdfg(**kw)

    np.testing.assert_allclose(z_sdfg, z_ref, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Loopnest 5 — vn_ie horizontal boundary (single-level + nlev+1 extrapolation)
# ---------------------------------------------------------------------------


def test_loopnest_5_sdfg_matches_f2py(tmp_path: Path):
    """vn_ie(je,1,jb) = vn(je,1,jb); vn_ie(je,nlevp1,jb) = weighted sum

    Separate frontend gap from the ``**`` work just landed: literal
    integer indices (``vn(je, 1, jb)``) fall through buildIndexExpr's
    resolveIndex path and surface as ``?`` in the memlet subset, so
    the SDFG picks up ``?`` as a free symbol.  Tracked alongside the
    ``(vn**2 + vt**2)`` lowering which now works correctly."""
    pytest.xfail("literal integer index (vn(je, 1, jb)) yields '?' in "
                 "memlet subset; separate from math.fpowi handling "
                 "which is now wired")


# ---------------------------------------------------------------------------
# Loopnest 6 — levelmask vertical reduction
# ---------------------------------------------------------------------------


def test_loopnest_6_sdfg_matches_f2py(tmp_path: Path):
    """levelmask(jk) = ANY(levmask(i_startblk:i_endblk, jk))"""
    bundle = _HERE / "loopnest_6.f90"
    flat_src = _extract_flat_kernel(bundle)
    ref = _f2py_build(flat_src, tmp_path / "ref", "kernel_flat_6")
    sdfg = _sdfg_from_flat(flat_src, tmp_path / "sdfg", name="kernel_flat_6")

    rng = np.random.default_rng(6)
    nlev, nblks_c = 64, 12
    i_startblk, i_endblk = 2, 10
    jk_start, jk_end = 3, nlev - 3

    # Fortran default LOGICAL(4) → int32 at the f2py / SDFG boundary.
    levmask = np.asfortranarray((rng.random((nblks_c, nlev)) > 0.7).astype(np.int32))
    levelmask_ref = np.zeros(nlev, dtype=np.int32)
    levelmask_sdfg = np.zeros(nlev, dtype=np.int32)

    ref.kernel_flat(levmask, levelmask_ref, jk_start, jk_end, i_startblk, i_endblk)

    kw = dict(levmask=levmask, levelmask=levelmask_sdfg, nlev=nlev, nblks_c=nblks_c)
    kw.update(_sdfg_call_args(sdfg, dict(jk_start=jk_start, jk_end=jk_end, i_startblk=i_startblk, i_endblk=i_endblk)))
    sdfg(**kw)

    np.testing.assert_array_equal(levelmask_sdfg, levelmask_ref)


# ---------------------------------------------------------------------------
# Loopnests 1, 4 — indirect stencils (3D indirection)
# ---------------------------------------------------------------------------


def test_loopnest_1_sdfg_matches_f2py(tmp_path: Path):
    """z_v_grad_w indirect stencil (two-way cell + vertex indirection).
    Skipped pending frontend support for 3D indirection tables where
    the last dim is a plain integer (``icidx(je, jb, 1..2)``)."""
    pytest.xfail("3D integer indirection arrays not yet lowered cleanly "
                 "through emit_tasklet + memlet subset resolution")


def test_loopnest_4_sdfg_matches_f2py(tmp_path: Path):
    """ddt_vn_apc_pc indirect stencil + (vn_ie(jk)-vn_ie(jk+1)) term."""
    pytest.xfail("4D output array (ddt_vn_apc_pc(...,ntnd)) + 3D "
                 "indirection not yet lowered cleanly")
