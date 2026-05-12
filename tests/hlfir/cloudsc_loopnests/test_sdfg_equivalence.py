"""End-to-end SDFG verification for the five CLOUDSC representative
loopnests (microphysics kernels extracted from ECMWF's cloud-scheme
benchmark).

For each kernel: build the SDFG via the HLFIR bridge AND an f2py
reference from the same Fortran source on identical seeded inputs,
then assert numerical equivalence.

Unlike the icon_loopnests bundle (which carries both struct-typed and
flat versions for cross-checking at the gfortran level), the cloudsc
loopnests are bare flat subroutines — the SDFG-vs-f2py comparison is
the only meaningful correctness check.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

_HERE = Path(__file__).resolve().parent

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _f2py_build(src_text: str, out_dir: Path, mod_name: str):
    """f2py-compile ``src_text`` as ``mod_name`` into ``out_dir`` and
    return the imported Python module.  Skips if the toolchain is missing."""
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


def _sdfg_from_src(src: str, tmp: Path, name: str):
    """Build an SDFG from raw Fortran source via the HLFIR bridge using
    the minimal ``hlfir-propagate-shapes`` pipeline (matches icon's
    convention)."""
    tmp.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, tmp, name=name, pipeline="hlfir-propagate-shapes").build()


def _sdfg_call_args(sdfg, int_values: dict) -> dict:
    """Route each integer arg in ``int_values`` to either a plain int
    (if the SDFG classified it as a symbol / Scalar) or a length-1
    numpy array (if classified as a length-1 Array).  Mirrors the
    icon_loopnests helper."""
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


def _preload_gomp():
    """DaCe-compiled SOs dynamically resolve ``omp_get_max_threads``;
    dlopen libgomp with RTLD_GLOBAL so later ctypes.CDLL loads find
    the symbol.  Past process-launch LD_PRELOAD can't help."""
    import ctypes
    for cand in ("libgomp.so.1", "/usr/lib/x86_64-linux-gnu/libgomp.so.1", "/usr/lib64/libgomp.so.1"):
        try:
            ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


_preload_gomp()


def test_cloudsc_lu_solver_sdfg_matches_f2py(tmp_path: Path):
    """LU forward-substitute + back-substitute for the microphysics
    species block.  Pure linear-algebra inner triple-nested loop with
    nclv-bounded iteration ranges."""
    src = (_HERE / "cloudsc_lu_solver.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "lu_solver_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="lu_solver_microphysics")

    rng = np.random.default_rng(101)
    klon, nclv = 8, 5
    kidia, kfdia = 1, klon

    # Diagonal-dominant random matrix so the LU back-substitute is stable.
    zqlhs = rng.standard_normal((klon, nclv, nclv)).astype(np.float64)
    for d in range(nclv):
        zqlhs[:, d, d] += 10.0  # diagonal dominance
    zqlhs = np.asfortranarray(zqlhs)
    zqxn = np.asfortranarray(rng.standard_normal((klon, nclv)).astype(np.float64))

    zqlhs_ref = np.array(zqlhs, order="F")
    zqxn_ref = np.array(zqxn, order="F")
    zqlhs_sdfg = np.array(zqlhs, order="F")
    zqxn_sdfg = np.array(zqxn, order="F")

    # f2py drops klon / nclv (auto-derived from array shapes).
    ref.lu_solver_microphysics(kidia, kfdia, zqlhs_ref, zqxn_ref)

    kw = dict(zqlhs=zqlhs_sdfg, zqxn=zqxn_sdfg)
    kw.update(_sdfg_call_args(sdfg, dict(kidia=kidia, kfdia=kfdia, klon=klon, nclv=nclv)))
    sdfg(**kw)

    np.testing.assert_allclose(zqlhs_sdfg, zqlhs_ref, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(zqxn_sdfg, zqxn_ref, atol=1e-10, rtol=1e-10)


def test_cloudsc_saturation_sdfg_matches_f2py(tmp_path: Path):
    """Saturation-pressure computation across (klon, klev).  Reads
    temperature + pressure, writes seven saturation-related outputs.
    Pure elementwise math (no loop-carried dependence)."""
    src = (_HERE / "cloudsc_saturation_calculation.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "saturation_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="compute_saturation_values")

    rng = np.random.default_rng(102)
    klon, klev = 16, 8
    kidia, kfdia = 1, klon

    # Realistic atmospheric temperature [200, 300] K + pressure [1e3, 1e5] Pa.
    ztp1 = np.asfortranarray(200.0 + 100.0 * rng.random((klon, klev), dtype=np.float64))
    pap = np.asfortranarray(1e3 + 1e5 * rng.random((klon, klev), dtype=np.float64))

    # Physical constants from CLOUDSC defaults (won't change the
    # equivalence — both backends see the same numbers).
    consts = dict(rtt=273.16,
                  retv=0.608,
                  r2es=611.21,
                  r3les=17.502,
                  r3ies=22.587,
                  r4les=32.19,
                  r4ies=-0.7,
                  rtice=250.0,
                  rtwat=273.16,
                  rtwat_rtice_r=1.0 / (273.16 - 250.0))

    outs_sdfg = {
        k: np.zeros((klon, klev), order="F")
        for k in ("zfoealfa", "zfoeewmt", "zqsmix", "zfoeew", "zqsice", "zfoeeliqt", "zqsliq")
    }

    # f2py converts INTENT(OUT) arrays to a return tuple; only INTENT(IN)
    # / scalars are positional.  klon / klev are auto-derived.
    out_tuple = ref.compute_saturation_values(kidia, kfdia, ztp1, pap, **consts)
    outs_ref = dict(zip(("zfoealfa", "zfoeewmt", "zqsmix", "zfoeew", "zqsice", "zfoeeliqt", "zqsliq"), out_tuple))

    kw = dict(ztp1=ztp1, pap=pap, **outs_sdfg, **consts)
    kw.update(_sdfg_call_args(sdfg, dict(kidia=kidia, kfdia=kfdia, klon=klon, klev=klev)))
    sdfg(**kw)

    for k in outs_ref:
        np.testing.assert_allclose(outs_sdfg[k], outs_ref[k], atol=1e-10, rtol=1e-10, err_msg=f"output {k} differs")


def test_cloudsc_autoconversion_snow_sdfg_matches_f2py(tmp_path: Path):
    """Snow autoconversion: writes ``zsnowaut(jl)`` and an INOUT
    contribution to ``zsolqb(jl, ncldqs, ncldqi)``.  ``laericeauto``
    is a 0/1 flag controlling whether the aerosol-coupled formulation
    runs."""
    src = (_HERE / "cloudsc_autoconversion_snow.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "autoconv_snow_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="autoconversion_snow")

    rng = np.random.default_rng(103)
    klon, ncldqs, ncldqi = 8, 5, 4
    kidia, kfdia = 1, klon

    ztp1 = np.asfortranarray(250.0 + 30.0 * rng.random((klon, ), dtype=np.float64))
    zicecld = np.asfortranarray(1e-5 + 1e-4 * rng.random((klon, ), dtype=np.float64))
    pnice = np.asfortranarray(1e3 + 1e4 * rng.random((klon, ), dtype=np.float64))
    consts = dict(rtt=273.16, rlcritsnow=0.5e-4, rsnowlin1=1e-3, rsnowlin2=0.018, rnice=1.0, ptsphy=600.0, zepsec=1e-14)
    laericeauto = 1

    zsolqb_ref = np.zeros((klon, ncldqs, ncldqi), order="F")
    zsolqb_sdfg = np.zeros_like(zsolqb_ref, order="F")

    # f2py: zsnowaut (OUT) → return; zsolqb (INOUT) → positional.
    zsnowaut_ref = ref.autoconversion_snow(kidia,
                                           kfdia,
                                           ztp1,
                                           zicecld,
                                           pnice,
                                           zsolqb_ref,
                                           rtt=consts["rtt"],
                                           rlcritsnow=consts["rlcritsnow"],
                                           rsnowlin1=consts["rsnowlin1"],
                                           rsnowlin2=consts["rsnowlin2"],
                                           rnice=consts["rnice"],
                                           ptsphy=consts["ptsphy"],
                                           zepsec=consts["zepsec"],
                                           laericeauto=laericeauto)
    zsnowaut_sdfg = np.zeros((klon, ), order="F")

    kw = dict(ztp1=ztp1,
              zicecld=zicecld,
              pnice=pnice,
              zsolqb=zsolqb_sdfg,
              zsnowaut=zsnowaut_sdfg,
              laericeauto=laericeauto,
              **consts)
    kw.update(_sdfg_call_args(sdfg, dict(kidia=kidia, kfdia=kfdia, klon=klon, ncldqs=ncldqs, ncldqi=ncldqi)))
    sdfg(**kw)

    np.testing.assert_allclose(zsnowaut_sdfg, zsnowaut_ref, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(zsolqb_sdfg, zsolqb_ref, atol=1e-12, rtol=1e-10)


def test_cloudsc_ice_supersat_sdfg_matches_f2py(tmp_path: Path):
    """Ice supersaturation adjustment: branchy elementwise body with
    in/out updates to ``zsolqa``, ``zsolac``, ``zqxfg``."""
    src = (_HERE / "cloudsc_ice_supersaturation_adjustment.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "ice_supersat_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="ice_supersaturation_adjustment")

    rng = np.random.default_rng(104)
    klon, nclv = 8, 5
    kidia, kfdia = 1, klon
    ncldql, ncldqi, ncldqv = 1, 2, 3
    nssopt = 1

    ztp1 = np.asfortranarray(220.0 + 30.0 * rng.random((klon, ), dtype=np.float64))
    za = np.asfortranarray(rng.random((klon, ), dtype=np.float64))
    zqx_ncldqv = np.asfortranarray(1e-5 + 1e-3 * rng.random((klon, ), dtype=np.float64))
    zqsice = np.asfortranarray(1e-5 + 1e-3 * rng.random((klon, ), dtype=np.float64))
    zcorqsice = np.asfortranarray(1e-5 + 1e-3 * rng.random((klon, ), dtype=np.float64))
    zfokoop = np.asfortranarray(rng.random((klon, ), dtype=np.float64))
    consts = dict(rtt=273.16, ramin=1e-12, rthomo=235.0, rkooptau=1e-4, ptsphy=600.0, zepsec=1e-14)

    zsolqa_ref = np.asfortranarray(rng.random((klon, nclv, nclv), dtype=np.float64) * 1e-3)
    zsolac_ref = np.asfortranarray(rng.random((klon, ), dtype=np.float64))
    zqxfg_ref = np.asfortranarray(rng.random((klon, nclv), dtype=np.float64) * 1e-3)
    zsolqa_sdfg = np.array(zsolqa_ref, order="F")
    zsolac_sdfg = np.array(zsolac_ref, order="F")
    zqxfg_sdfg = np.array(zqxfg_ref, order="F")

    # f2py positional: kidia, kfdia, ztp1, za, zqx_ncldqv, zqsice, zcorqsice, zfokoop,
    #                  zsolqa, zsolac, zqxfg, rtt, ramin, rthomo, nssopt, rkooptau, ptsphy, zepsec,
    #                  ncldql, ncldqi, ncldqv   ([klon, nclv] auto-derived)
    ref.ice_supersaturation_adjustment(kidia, kfdia, ztp1, za, zqx_ncldqv, zqsice, zcorqsice, zfokoop, zsolqa_ref,
                                       zsolac_ref, zqxfg_ref, consts["rtt"], consts["ramin"], consts["rthomo"], nssopt,
                                       consts["rkooptau"], consts["ptsphy"], consts["zepsec"], ncldql, ncldqi, ncldqv)

    kw = dict(ztp1=ztp1,
              za=za,
              zqx_ncldqv=zqx_ncldqv,
              zqsice=zqsice,
              zcorqsice=zcorqsice,
              zfokoop=zfokoop,
              zsolqa=zsolqa_sdfg,
              zsolac=zsolac_sdfg,
              zqxfg=zqxfg_sdfg,
              **consts)
    kw.update(
        _sdfg_call_args(
            sdfg,
            dict(kidia=kidia,
                 kfdia=kfdia,
                 klon=klon,
                 nclv=nclv,
                 ncldql=ncldql,
                 ncldqi=ncldqi,
                 ncldqv=ncldqv,
                 nssopt=nssopt)))
    sdfg(**kw)

    np.testing.assert_allclose(zsolqa_sdfg, zsolqa_ref, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(zsolac_sdfg, zsolac_ref, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(zqxfg_sdfg, zqxfg_ref, atol=1e-12, rtol=1e-10)


def test_cloudsc_rain_evap_sdfg_matches_f2py(tmp_path: Path):
    """Rain-evaporation (Abel-Boutle 2012 scheme): branchy elementwise
    update to ``zsolqa(jl, ncldqv, ncldqr)`` and several scalars.
    LOGICAL local + many physical constants."""
    src = (_HERE / "cloudsc_rain_evaporation_abel_boutle.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "rain_evap_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="rain_evaporation_abel_boutle")

    rng = np.random.default_rng(105)
    klon, nclv = 8, 5
    kidia, kfdia = 1, klon
    ncldqv, ncldqr = 3, 4

    ztp1 = np.asfortranarray(250.0 + 40.0 * rng.random((klon, ), dtype=np.float64))
    zqx_ncldqv = np.asfortranarray(1e-5 + 1e-3 * rng.random((klon, ), dtype=np.float64))
    za = np.asfortranarray(rng.random((klon, ), dtype=np.float64))
    zqsliq = np.asfortranarray(1e-5 + 1e-3 * rng.random((klon, ), dtype=np.float64))
    zqxfg_ncldqr = np.asfortranarray(1e-6 + 1e-4 * rng.random((klon, ), dtype=np.float64))
    zcovptot = np.asfortranarray(rng.random((klon, ), dtype=np.float64))
    zcovpclr = np.asfortranarray(rng.random((klon, ), dtype=np.float64))
    zcovpmax = np.asfortranarray(rng.random((klon, ), dtype=np.float64))
    zrho = np.asfortranarray(1.0 + 0.3 * rng.random((klon, ), dtype=np.float64))
    pap = np.asfortranarray(1e4 + 1e5 * rng.random((klon, ), dtype=np.float64))
    consts = dict(rtt=273.16,
                  rv=461.5,
                  rd=287.04,
                  rprecrhmax=1.0,
                  rcovpmin=1e-3,
                  rdensref=1.0,
                  ptsphy=600.0,
                  zepsec=1e-14,
                  rcl_fac1=1.0,
                  rcl_fac2=1.0,
                  rcl_cdenom1=1.0,
                  rcl_cdenom2=1.0,
                  rcl_cdenom3=1.0,
                  rcl_ka273=2.4e-2,
                  rcl_const1r=1.0,
                  rcl_const2r=1.0,
                  rcl_const3r=1.0,
                  rcl_const4r=1.0)

    zsolqa_ref = np.asfortranarray(rng.random((klon, nclv, nclv), dtype=np.float64) * 1e-4)
    zevap_ref = np.zeros((klon, ), order="F")
    zsolqa_sdfg = np.array(zsolqa_ref, order="F")
    zevap_sdfg = np.zeros_like(zevap_ref, order="F")
    zcovptot_ref = np.array(zcovptot, order="F")
    zcovpclr_ref = np.array(zcovpclr, order="F")
    zqxfg_ref = np.array(zqxfg_ncldqr, order="F")
    zcovptot_sdfg = np.array(zcovptot, order="F")
    zcovpclr_sdfg = np.array(zcovpclr, order="F")
    zqxfg_sdfg = np.array(zqxfg_ncldqr, order="F")

    # f2py positional (zevap_out is OUT, returned):
    # kidia, kfdia, ztp1, zqx_ncldqv, za, zqsliq, zqxfg_ncldqr, zcovptot, zcovpclr, zcovpmax,
    # zrho, pap, zsolqa, rtt, rv, rd, rprecrhmax, rcovpmin, rdensref, ptsphy, zepsec,
    # rcl_fac1, rcl_fac2, rcl_cdenom1, rcl_cdenom2, rcl_cdenom3, rcl_ka273,
    # rcl_const1r, rcl_const2r, rcl_const3r, rcl_const4r, ncldqv, ncldqr  ([klon, nclv] auto)
    zevap_ref = ref.rain_evaporation_abel_boutle(
        kidia, kfdia, ztp1, zqx_ncldqv, za, zqsliq, zqxfg_ref, zcovptot_ref, zcovpclr_ref, zcovpmax, zrho, pap,
        zsolqa_ref, consts["rtt"], consts["rv"], consts["rd"], consts["rprecrhmax"], consts["rcovpmin"],
        consts["rdensref"], consts["ptsphy"], consts["zepsec"], consts["rcl_fac1"], consts["rcl_fac2"],
        consts["rcl_cdenom1"], consts["rcl_cdenom2"], consts["rcl_cdenom3"], consts["rcl_ka273"], consts["rcl_const1r"],
        consts["rcl_const2r"], consts["rcl_const3r"], consts["rcl_const4r"], ncldqv, ncldqr)
    zevap_sdfg = np.zeros((klon, ), order="F")

    kw = dict(ztp1=ztp1,
              zqx_ncldqv=zqx_ncldqv,
              za=za,
              zqsliq=zqsliq,
              zqxfg_ncldqr=zqxfg_sdfg,
              zcovptot=zcovptot_sdfg,
              zcovpclr=zcovpclr_sdfg,
              zcovpmax=zcovpmax,
              zrho=zrho,
              pap=pap,
              zsolqa=zsolqa_sdfg,
              zevap_out=zevap_sdfg,
              **consts)
    kw.update(_sdfg_call_args(sdfg, dict(kidia=kidia, kfdia=kfdia, klon=klon, nclv=nclv, ncldqv=ncldqv, ncldqr=ncldqr)))
    sdfg(**kw)

    # Rain-evap uses ``rho**0.78`` (Abel-Boutle 2012 exponent) which
    # amplifies rounding differences between gfortran's libm and DaCe's
    # C++ codegen.  ~1e-7 relative drift is expected; tighter tolerance
    # would surface false positives on benign ulp-level differences.
    rt, at = 1e-6, 1e-8
    np.testing.assert_allclose(zsolqa_sdfg, zsolqa_ref, atol=at, rtol=rt)
    np.testing.assert_allclose(zevap_sdfg, zevap_ref, atol=at, rtol=rt)
    np.testing.assert_allclose(zcovptot_sdfg, zcovptot_ref, atol=at, rtol=rt)
    np.testing.assert_allclose(zcovpclr_sdfg, zcovpclr_ref, atol=at, rtol=rt)
    np.testing.assert_allclose(zqxfg_sdfg, zqxfg_ref, atol=at, rtol=rt)
