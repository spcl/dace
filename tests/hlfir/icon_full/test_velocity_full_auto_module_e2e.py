"""Regression e2e: bridge-auto-detected Fortran module-global
provenance replaces the hand-authored ``module_symbol_sources``.

``velocity_full_bindings_e2e_test`` proves the full ICON
``velocity_tendencies`` binding works *with* a hand-authored
``OriginalInterface.module_symbol_sources`` mapping the nine module
globals (``nproma`` / ``nrdmax`` / ``i_am_accel_node`` / timer
handles / ...) to their owning Fortran modules.

This test proves the same binding works with that map **empty**: the
bridge now decodes each value's ``_QM<module>E<entity>``-mangled
``fir.global`` provenance (``extract_vars.cpp`` ->
``FrozenSignature.module_symbol_origins``) and the emitter
``USE``-imports + assigns every module global from that alone.  The
generated wrapper is gfortran-compiled, linked against the compiled
SDFG ``.so`` and executed; outputs are compared against the gfortran
reference of the un-transformed ``velocity_tendencies`` (per
``feedback_e2e_numerical`` / ``feedback_e2e_valid_fortran``).

Reusing ``velocity_full_bindings_e2e_test``'s helpers keeps one
source of truth for the (large) struct construction + driver shim;
the only delta is the empty override map.
"""

import ctypes
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

from _util import build_sdfg

from dace.frontend.hlfir.bindings import FlattenPlan, emit_bindings
from dace.frontend.hlfir.bindings.block_builders import effective_module_sources

# The sibling e2e test owns the (large) struct construction + driver
# shim + allocation helpers; reuse them verbatim so there is one
# source of truth.  The bindings test directory is not on sys.path
# (conftest only adds ``tests/hlfir``), so load it by file path.
_VF_PATH = Path(__file__).resolve().parent / "test_velocity_full_bindings_e2e.py"
_spec = importlib.util.spec_from_file_location("test_velocity_full_bindings_e2e", _VF_PATH)
vf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vf)

pytestmark = vf.pytestmark

# Same caller flat-array order + the nine module globals the
# hand-authored map used to supply.  We assert the bridge recovers
# exactly these nine, then run the full e2e with the override empty.
_EXPECTED_AUTO_ORIGINS = {
    "nproma": ("mo_parallel_config", "nproma"),
    "timers_level": ("mo_run_config", "timers_level"),
    "nrdmax": ("mo_vertical_grid", "nrdmax"),
    "nflatlev": ("mo_init_vgrid", "nflatlev"),
    "i_am_accel_node": ("mo_mpi", "i_am_accel_node"),
    "lextra_diffu": ("mo_nonhydrostatic_config", "lextra_diffu"),
    "lvert_nest": ("mo_run_config", "lvert_nest"),
    "timer_intp": ("mo_timer", "timer_intp"),
    "timer_solve_nh_veltend": ("mo_timer", "timer_solve_nh_veltend"),
}


def test_velocity_full_auto_module_provenance_e2e(tmp_path: Path):
    """With ``module_symbol_sources={}`` the bridge-auto-detected
    provenance alone must yield a ``velocity_tendencies_dace`` binding
    that reproduces the gfortran reference to 1e-10 on all outputs."""
    # Unique SDFG name: the sibling e2e test also builds a
    # ``velocity_tendencies`` SDFG, and DaCe's build cache /
    # ``libdacestub_<name>.so`` is keyed by SDFG name -- two tests
    # sharing the name collide (OSError on the mapped ``.so``) under
    # xdist.  A distinct name isolates this test's build artifacts.
    _NAME = "velocity_tendencies_auto"
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(vf._DRIVER_PATH.read_text(), sdfg_dir, name="velocity_tendencies", entry=vf._ENTRY)
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.validate()
    sdfg.name = _NAME

    frozen = sdfg._frozen_signature

    # 1. The bridge auto-detected every module global, with the same
    #    (module, entity) the hand-authored map used to specify.
    for sym, origin in _EXPECTED_AUTO_ORIGINS.items():
        assert frozen.module_symbol_origins.get(sym) == origin, (
            f"{sym!r}: auto-detected "
            f"{frozen.module_symbol_origins.get(sym)!r}, expected {origin!r}")

    # 2. Empty the explicit override -> emitter must rely purely on
    #    the bridge's auto-detection.  ``iface.entry`` must equal the
    #    SDFG name (both feed the C-bind ``__program_<entry>`` symbol).
    iface_auto = replace(vf._IFACE, entry=_NAME, module_symbol_sources={})
    merged = effective_module_sources(frozen, iface_auto)
    for sym, origin in _EXPECTED_AUTO_ORIGINS.items():
        assert merged.get(sym) == origin

    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)

    bindings_path = tmp_path / "velocity_tendencies_bindings.f90"
    emit_bindings(frozen, iface_auto, plan, str(bindings_path))

    # The generated wrapper must carry the use-imports + assignments
    # for every module global with NO hand-authored residue.
    text = bindings_path.read_text()
    for sym, (mod, member) in _EXPECTED_AUTO_ORIGINS.items():
        assert f"use {mod}, only:" in text, f"missing use of {mod}"
        assert f"{sym}__mod => {member}" in text, f"missing import of {sym}"

    caller_src = vf._CALLER_PATH.read_text()
    sdfg_shim = vf._make_sdfg_driver(caller_src)
    # ``_make_sdfg_driver`` hardcodes the sibling's
    # ``velocity_tendencies_dace`` symbol names; retarget them to this
    # test's unique-entry binding (``<_NAME>_dace`` /
    # ``<_NAME>_bindings``) so the renamed SDFG resolves.
    sdfg_shim = sdfg_shim.replace("velocity_tendencies_dace_bindings", f"{_NAME}_dace_bindings")
    sdfg_shim = sdfg_shim.replace("velocity_tendencies_dace", f"{_NAME}_dace")

    sdfg_build = tmp_path / "sdfg_build"
    sdfg_build.mkdir(parents=True, exist_ok=True)
    shim_path = sdfg_build / "velocity_sdfg_shim.f90"
    shim_path.write_text(sdfg_shim)
    sdfg_so = sdfg_build / "libvelocity_sdfg.so"
    vf._gfortran(sdfg_so,
                 vf._DRIVER_PATH,
                 vf._CALLER_PATH,
                 bindings_path,
                 shim_path,
                 mod_dir=sdfg_build,
                 link_so=so_path)
    sdfg_lib = ctypes.CDLL(str(sdfg_so))

    ref_build = tmp_path / "ref_build"
    ref_build.mkdir(parents=True, exist_ok=True)
    ref_so = ref_build / "libvelocity_ref.so"
    vf._gfortran(ref_so, vf._DRIVER_PATH, vf._CALLER_PATH, mod_dir=ref_build)
    ref_lib = ctypes.CDLL(str(ref_so))

    nproma, nlev, nblks_c, nblks_e, nblks_v = 8, 6, 4, 4, 4
    nlevp1 = nlev + 1
    dims = (nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v)

    bufs_ref = vf._allocate(*dims)
    init = ref_lib.init_inputs_random_c
    init.restype = None
    init.argtypes = [ctypes.c_int] * 7 + [ctypes.c_void_p] * len(vf._INIT_ARRAY_ORDER)
    init(42, nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v, *[bufs_ref[k].ctypes.data for k in vf._INIT_ARRAY_ORDER])
    bufs_sdfg = {k: v.copy(order='F') for k, v in bufs_ref.items()}

    zshape = ((nproma, nlev, nblks_e), (nproma, nlev, nblks_e), (nproma, nlevp1, nblks_e))
    z_ref = [np.zeros(s, dtype=np.float64, order='F') for s in zshape]
    z_sdfg = [np.zeros(s, dtype=np.float64, order='F') for s in zshape]

    pre = {nm: bufs_ref[nm].copy() for nm in vf._OUTPUT_NAMES if nm in bufs_ref}

    vf._run(ref_lib, "run_velocity_flat_c", dims, bufs_ref, z_ref)
    vf._run(sdfg_lib, "run_velocity_flat_sdfg", dims, bufs_sdfg, z_sdfg)

    extras = dict(zip(('z_w_concorr_me', 'z_kin_hor_e', 'z_vt_ie'), zip(z_sdfg, z_ref)))
    mismatches = []
    mutated = False
    for nm in vf._OUTPUT_NAMES:
        sd, rf = extras[nm] if nm in extras else (bufs_sdfg[nm], bufs_ref[nm])
        if nm in pre and not np.array_equal(rf, pre[nm], equal_nan=True):
            mutated = True
        if not np.allclose(sd, rf, rtol=1e-10, atol=1e-10, equal_nan=True):
            d = np.abs(sd - rf)
            mismatches.append(f"{nm}: max_abs_diff={d.max():.3e} "
                              f"(n_diff={np.count_nonzero(d > 1e-10)})")
    assert mutated, "reference left every output untouched -- kernel did not run"
    assert not mismatches, "\n".join(mismatches)
