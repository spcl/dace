"""f90-binding e2e for the full ICON ``velocity_tendencies`` kernel.

This is the derived-type counterpart of ``icon_loopnest_bindings_e2e_test``
/ ``cloudsc_flux_bindings_e2e_test``: instead of a flat explicit-shape
kernel it drives the real ``velocity_tendencies`` whose interface is
five ICON derived types (``t_nh_prog`` / ``t_patch`` / ``t_int_state``
/ ``t_nh_metrics`` / ``t_nh_diag``) plus three naked rank-3 arrays and
scalars.  The ``OriginalInterface`` is hand-authored (the bridge has
no ``get_fortran_interface``); the bindings emitter consumes it plus
the bridge-produced ``FlattenPlan`` to generate a struct-typed
``velocity_tendencies_dace`` wrapper that ``c_f_pointer``-aliases
every flattened struct member to a flat SDFG companion.

Reference: the proven flat caller ``run_velocity_flat_c`` in
``velocity_full_caller.f90`` (calls the un-transformed
``velocity_tendencies`` directly).  SDFG side: a sibling shim
generated from that same caller -- identical struct construction, but
the final call targets the generated ``velocity_tendencies_dace``
binding.  Both are driven from one flat ``ctypes`` buffer set so the
numerical comparison is apples-to-apples.

Per ``feedback_e2e_numerical`` / ``feedback_e2e_valid_fortran``: the
binding is gfortran-compiled, linked against the compiled SDFG ``.so``
and executed; outputs are compared against the gfortran reference.
"""

import ctypes
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

from dace.frontend.hlfir.bindings import (
    FlattenPlan,
    OriginalArg,
    OriginalInterface,
    emit_bindings,
)

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

_HERE = Path(__file__).resolve().parent
_DRIVER_PATH = _HERE / "velocity_full.f90"
_CALLER_PATH = _HERE / "velocity_full_caller.f90"
_ENTRY = "_QMmo_velocity_advectionPvelocity_tendencies"

# Caller subroutine flat-array dummy order -- matches velocity_full_caller.f90
# (identical list to velocity_full_test.py::_INIT_ARRAY_ORDER).
_INIT_ARRAY_ORDER = (
    'p_patch_cells_area',
    'p_patch_cells_neighbor_idx',
    'p_patch_cells_neighbor_blk',
    'p_patch_cells_edge_idx',
    'p_patch_cells_edge_blk',
    'p_patch_cells_start_index',
    'p_patch_cells_end_index',
    'p_patch_cells_start_block',
    'p_patch_cells_end_block',
    'p_patch_cells_decomp_info_owner_mask',
    'p_patch_edges_cell_idx',
    'p_patch_edges_cell_blk',
    'p_patch_edges_vertex_idx',
    'p_patch_edges_vertex_blk',
    'p_patch_edges_quad_idx',
    'p_patch_edges_quad_blk',
    'p_patch_edges_tangent_orientation',
    'p_patch_edges_inv_primal_edge_length',
    'p_patch_edges_inv_dual_edge_length',
    'p_patch_edges_area_edge',
    'p_patch_edges_f_e',
    'p_patch_edges_fn_e',
    'p_patch_edges_ft_e',
    'p_patch_edges_start_index',
    'p_patch_edges_end_index',
    'p_patch_edges_start_block',
    'p_patch_edges_end_block',
    'p_patch_verts_cell_idx',
    'p_patch_verts_cell_blk',
    'p_patch_verts_edge_idx',
    'p_patch_verts_edge_blk',
    'p_patch_verts_start_index',
    'p_patch_verts_end_index',
    'p_patch_verts_start_block',
    'p_patch_verts_end_block',
    'p_int_c_lin_e',
    'p_int_e_bln_c_s',
    'p_int_cells_aw_verts',
    'p_int_rbf_vec_coeff_e',
    'p_int_geofac_grdiv',
    'p_int_geofac_rot',
    'p_int_geofac_n2s',
    'p_prog_w',
    'p_prog_vn',
    'p_diag_vn_ie_ubc',
    'p_diag_vt',
    'p_diag_vn_ie',
    'p_diag_w_concorr_c',
    'p_diag_ddt_vn_apc_pc',
    'p_diag_ddt_vn_cor_pc',
    'p_diag_ddt_w_adv_pc',
    'p_metrics_ddxn_z_full',
    'p_metrics_ddxt_z_full',
    'p_metrics_ddqz_z_full_e',
    'p_metrics_ddqz_z_half',
    'p_metrics_wgtfac_c',
    'p_metrics_wgtfac_e',
    'p_metrics_wgtfacq_e',
    'p_metrics_coeff_gradekin',
    'p_metrics_coeff1_dwdz',
    'p_metrics_coeff2_dwdz',
    'p_metrics_deepatmo_gradh_mc',
    'p_metrics_deepatmo_invr_mc',
    'p_metrics_deepatmo_gradh_ifc',
    'p_metrics_deepatmo_invr_ifc',
)

_OUTPUT_NAMES = (
    'p_prog_w',
    'p_prog_vn',
    'p_diag_vn_ie_ubc',
    'p_diag_vt',
    'p_diag_vn_ie',
    'p_diag_w_concorr_c',
    'p_diag_ddt_vn_apc_pc',
    'p_diag_ddt_vn_cor_pc',
    'p_diag_ddt_w_adv_pc',
    'z_w_concorr_me',
    'z_kin_hor_e',
    'z_vt_ie',
)


def _scalar(name, ftype, intent, stype=None):
    return OriginalArg(name=name, fortran_type=ftype, rank=0, shape=(), intent=intent, struct_type=stype)


def _arr3(name, intent):
    return OriginalArg(name=name,
                       fortran_type="real(8)",
                       rank=3,
                       shape=(":", ":", ":"),
                       intent=intent,
                       struct_type=None)


_IFACE = OriginalInterface(
    entry="velocity_tendencies",
    args=(
        _scalar("p_prog", "type(t_nh_prog)", "inout", "t_nh_prog"),
        _scalar("p_patch", "type(t_patch)", "in", "t_patch"),
        _scalar("p_int", "type(t_int_state)", "in", "t_int_state"),
        _scalar("p_metrics", "type(t_nh_metrics)", "inout", "t_nh_metrics"),
        _scalar("p_diag", "type(t_nh_diag)", "inout", "t_nh_diag"),
        _arr3("z_w_concorr_me", "inout"),
        _arr3("z_kin_hor_e", "inout"),
        _arr3("z_vt_ie", "inout"),
        _scalar("ntnd", "integer", "in"),
        _scalar("istep", "integer", "in"),
        _scalar("lvn_only", "logical", "in"),
        _scalar("dtime", "real(8)", "in"),
        _scalar("dt_linintp_ubc", "real(8)", "in"),
        _scalar("ldeepatmo", "logical", "in"),
    ),
    struct_types={},
    used_modules={
        "mo_model_domain": ("t_patch", ),
        "mo_intp_data_strc": ("t_int_state", ),
        "mo_nonhydro_types": ("t_nh_prog", "t_nh_metrics", "t_nh_diag"),
    },
    # SDFG free symbols / lifted args the kernel reads from Fortran
    # module data (no dummy to query); the emitter ``use``-imports
    # each under a ``__mod`` alias and assigns / copies it.
    module_symbol_sources={
        "nproma": ("mo_parallel_config", "nproma"),
        "timers_level": ("mo_run_config", "timers_level"),
        "nrdmax": ("mo_vertical_grid", "nrdmax"),
        "nflatlev": ("mo_init_vgrid", "nflatlev"),
        "i_am_accel_node": ("mo_mpi", "i_am_accel_node"),
        "lextra_diffu": ("mo_nonhydrostatic_config", "lextra_diffu"),
        "lvert_nest": ("mo_run_config", "lvert_nest"),
        "timer_intp": ("mo_timer", "timer_intp"),
        "timer_solve_nh_veltend": ("mo_timer", "timer_solve_nh_veltend"),
    },
)


def _make_sdfg_driver(caller_src: str) -> str:
    """Derive the SDFG-side shim from the proven flat caller: rename
    ``run_velocity_flat_c`` -> ``run_velocity_flat_sdfg``, pull in the
    generated binding module, and retarget the final call from
    ``velocity_tendencies`` to ``velocity_tendencies_dace`` (plus a
    finalize).  One source of truth for struct construction."""
    m = re.search(r"(?is)(SUBROUTINE\s+run_velocity_flat_c\b.*?END\s+SUBROUTINE\s+run_velocity_flat_c)", caller_src)
    if not m:
        raise RuntimeError("run_velocity_flat_c not found in caller source")
    shim = m.group(1)
    shim = shim.replace("run_velocity_flat_c", "run_velocity_flat_sdfg")
    shim = shim.replace(
        "USE mo_velocity_advection,  ONLY: velocity_tendencies",
        "USE velocity_tendencies_dace_bindings, ONLY: velocity_tendencies_dace, "
        "velocity_tendencies_dace_finalize",
    )
    # Retarget the kernel call.  The reference call spans several
    # continuation lines; swap just the callee name.
    shim = shim.replace("CALL velocity_tendencies(p_prog, p_patch", "CALL velocity_tendencies_dace(p_prog, p_patch")
    # Finalize the ref-counted SDFG handle before returning.
    shim = re.sub(r"(?i)\bEND\s+SUBROUTINE\s+run_velocity_flat_sdfg", "  CALL velocity_tendencies_dace_finalize()\n"
                  "END SUBROUTINE run_velocity_flat_sdfg", shim)
    return shim


def _gfortran(out_so: Path, *sources, mod_dir: Path, link_so: Path | None = None):
    cmd = [
        "gfortran", "-shared", "-fPIC", "-O0", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none",
        f"-J{mod_dir}"
    ]
    cmd += [str(s) for s in sources]
    cmd += ["-o", str(out_so)]
    if link_so is not None:
        cmd += [f"-L{link_so.parent}", f"-Wl,-rpath,{link_so.parent}", f"-l:{link_so.name}"]
    subprocess.check_call(cmd, cwd=mod_dir)


def _allocate(nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v):
    F = lambda *s: np.zeros(s, dtype=np.float64, order='F')
    I = lambda *s: np.zeros(s, dtype=np.int32, order='F')
    B = lambda *s: np.zeros(s, dtype=np.int8, order='F')
    return dict(
        p_patch_cells_area=F(nproma, nblks_c),
        p_patch_cells_neighbor_idx=I(nproma, nblks_c, 3),
        p_patch_cells_neighbor_blk=I(nproma, nblks_c, 3),
        p_patch_cells_edge_idx=I(nproma, nblks_c, 3),
        p_patch_cells_edge_blk=I(nproma, nblks_c, 3),
        p_patch_cells_start_index=I(33),
        p_patch_cells_end_index=I(33),
        p_patch_cells_start_block=I(33),
        p_patch_cells_end_block=I(33),
        p_patch_cells_decomp_info_owner_mask=B(nproma, nblks_c),
        p_patch_edges_cell_idx=I(nproma, nblks_e, 2),
        p_patch_edges_cell_blk=I(nproma, nblks_e, 2),
        p_patch_edges_vertex_idx=I(nproma, nblks_e, 4),
        p_patch_edges_vertex_blk=I(nproma, nblks_e, 4),
        p_patch_edges_quad_idx=I(nproma, nblks_e, 4),
        p_patch_edges_quad_blk=I(nproma, nblks_e, 4),
        p_patch_edges_tangent_orientation=F(nproma, nblks_e),
        p_patch_edges_inv_primal_edge_length=F(nproma, nblks_e),
        p_patch_edges_inv_dual_edge_length=F(nproma, nblks_e),
        p_patch_edges_area_edge=F(nproma, nblks_e),
        p_patch_edges_f_e=F(nproma, nblks_e),
        p_patch_edges_fn_e=F(nproma, nblks_e),
        p_patch_edges_ft_e=F(nproma, nblks_e),
        p_patch_edges_start_index=I(33),
        p_patch_edges_end_index=I(33),
        p_patch_edges_start_block=I(33),
        p_patch_edges_end_block=I(33),
        p_patch_verts_cell_idx=I(nproma, nblks_v, 6),
        p_patch_verts_cell_blk=I(nproma, nblks_v, 6),
        p_patch_verts_edge_idx=I(nproma, nblks_v, 6),
        p_patch_verts_edge_blk=I(nproma, nblks_v, 6),
        p_patch_verts_start_index=I(33),
        p_patch_verts_end_index=I(33),
        p_patch_verts_start_block=I(33),
        p_patch_verts_end_block=I(33),
        p_int_c_lin_e=F(nproma, 2, nblks_e),
        p_int_e_bln_c_s=F(nproma, 3, nblks_c),
        p_int_cells_aw_verts=F(nproma, 6, nblks_v),
        p_int_rbf_vec_coeff_e=F(4, nproma, nblks_e),
        p_int_geofac_grdiv=F(nproma, 5, nblks_e),
        p_int_geofac_rot=F(nproma, 6, nblks_v),
        p_int_geofac_n2s=F(nproma, 4, nblks_c),
        p_prog_w=F(nproma, nlevp1, nblks_c),
        p_prog_vn=F(nproma, nlev, nblks_e),
        p_diag_vn_ie_ubc=F(nproma, 2, nblks_e),
        p_diag_vt=F(nproma, nlev, nblks_e),
        p_diag_vn_ie=F(nproma, nlevp1, nblks_e),
        p_diag_w_concorr_c=F(nproma, nlev, nblks_c),
        p_diag_ddt_vn_apc_pc=F(nproma, nlev, nblks_e, 3),
        p_diag_ddt_vn_cor_pc=F(nproma, nlev, nblks_e, 3),
        p_diag_ddt_w_adv_pc=F(nproma, nlevp1, nblks_c, 3),
        p_metrics_ddxn_z_full=F(nproma, nlev, nblks_e),
        p_metrics_ddxt_z_full=F(nproma, nlev, nblks_e),
        p_metrics_ddqz_z_full_e=F(nproma, nlev, nblks_e),
        p_metrics_ddqz_z_half=F(nproma, nlevp1, nblks_c),
        p_metrics_wgtfac_c=F(nproma, nlevp1, nblks_c),
        p_metrics_wgtfac_e=F(nproma, nlevp1, nblks_e),
        p_metrics_wgtfacq_e=F(nproma, 3, nblks_e),
        p_metrics_coeff_gradekin=F(nproma, 2, nblks_e),
        p_metrics_coeff1_dwdz=F(nproma, nlev, nblks_c),
        p_metrics_coeff2_dwdz=F(nproma, nlev, nblks_c),
        p_metrics_deepatmo_gradh_mc=F(nlev),
        p_metrics_deepatmo_invr_mc=F(nlev),
        p_metrics_deepatmo_gradh_ifc=F(nlevp1),
        p_metrics_deepatmo_invr_ifc=F(nlevp1),
    )


def _run(lib, fn, dims, bufs, z_arrays):
    f = getattr(lib, fn)
    f.restype = None
    f.argtypes = ([ctypes.c_int] * 6  # dims
                  + [ctypes.c_int, ctypes.c_int]  # ntnd, istep
                  + [ctypes.c_int8, ctypes.c_int8]  # lvn_only, ldeepatmo
                  + [ctypes.c_double, ctypes.c_double]  # dtime, dt_linintp_ubc
                  + [ctypes.c_void_p, ctypes.c_void_p]  # nrdmax_in, nflatlev_in
                  + [ctypes.c_int8, ctypes.c_int8, ctypes.c_int]  # lvert_nest, lextra_diffu, timers_level
                  + [ctypes.c_void_p] * (len(_INIT_ARRAY_ORDER) + 3))
    nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v = dims
    nrdmax_in = np.full(10, nlev, dtype=np.int32, order='F')
    nflatlev_in = np.ones(10, dtype=np.int32, order='F')
    f(nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v, 1, 1, 0, 0, 60.0, 0.0, nrdmax_in.ctypes.data,
      nflatlev_in.ctypes.data, 0, 0, 0, *[bufs[k].ctypes.data for k in _INIT_ARRAY_ORDER],
      *[z.ctypes.data for z in z_arrays])


def test_velocity_full_f90_bindings_e2e(tmp_path: Path):
    """The hand-authored derived-type ``OriginalInterface`` +
    bridge ``FlattenPlan`` must yield a ``velocity_tendencies_dace``
    binding that, linked against the compiled SDFG, reproduces the
    gfortran reference of the un-transformed ``velocity_tendencies``."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(_DRIVER_PATH.read_text(), sdfg_dir, name="velocity_tendencies", entry=_ENTRY)
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.validate()
    sdfg.name = "velocity_tendencies"
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)

    bindings_path = tmp_path / "velocity_tendencies_bindings.f90"
    emit_bindings(sdfg._frozen_signature, _IFACE, plan, str(bindings_path))

    caller_src = _CALLER_PATH.read_text()
    sdfg_shim = _make_sdfg_driver(caller_src)

    # SDFG .so: driver modules + caller (init only) + generated binding
    # + the SDFG shim, linked against the compiled SDFG library.
    sdfg_build = tmp_path / "sdfg_build"
    sdfg_build.mkdir(parents=True, exist_ok=True)
    shim_path = sdfg_build / "velocity_sdfg_shim.f90"
    shim_path.write_text(sdfg_shim)
    sdfg_so = sdfg_build / "libvelocity_sdfg.so"
    _gfortran(sdfg_so, _DRIVER_PATH, _CALLER_PATH, bindings_path, shim_path, mod_dir=sdfg_build, link_so=so_path)
    sdfg_lib = ctypes.CDLL(str(sdfg_so))

    # Reference .so: driver + the proven flat caller (un-transformed).
    ref_build = tmp_path / "ref_build"
    ref_build.mkdir(parents=True, exist_ok=True)
    ref_so = ref_build / "libvelocity_ref.so"
    _gfortran(ref_so, _DRIVER_PATH, _CALLER_PATH, mod_dir=ref_build)
    ref_lib = ctypes.CDLL(str(ref_so))

    nproma, nlev, nblks_c, nblks_e, nblks_v = 8, 6, 4, 4, 4
    nlevp1 = nlev + 1
    dims = (nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v)

    bufs_ref = _allocate(*dims)
    init = ref_lib.init_inputs_random_c
    init.restype = None
    init.argtypes = [ctypes.c_int] * 7 + [ctypes.c_void_p] * len(_INIT_ARRAY_ORDER)
    init(42, nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v, *[bufs_ref[k].ctypes.data for k in _INIT_ARRAY_ORDER])
    bufs_sdfg = {k: v.copy(order='F') for k, v in bufs_ref.items()}

    zshape = ((nproma, nlev, nblks_e), (nproma, nlev, nblks_e), (nproma, nlevp1, nblks_e))
    z_ref = [np.zeros(s, dtype=np.float64, order='F') for s in zshape]
    z_sdfg = [np.zeros(s, dtype=np.float64, order='F') for s in zshape]

    # Pristine snapshots so we can prove the kernel mutated outputs
    # (guards against a vacuous "both untouched -> equal" pass).
    pre = {nm: bufs_ref[nm].copy() for nm in _OUTPUT_NAMES if nm in bufs_ref}

    _run(ref_lib, "run_velocity_flat_c", dims, bufs_ref, z_ref)
    _run(sdfg_lib, "run_velocity_flat_sdfg", dims, bufs_sdfg, z_sdfg)

    extras = dict(zip(('z_w_concorr_me', 'z_kin_hor_e', 'z_vt_ie'), zip(z_sdfg, z_ref)))
    mismatches = []
    mutated = False
    for nm in _OUTPUT_NAMES:
        sd, rf = extras[nm] if nm in extras else (bufs_sdfg[nm], bufs_ref[nm])
        if nm in pre and not np.array_equal(rf, pre[nm], equal_nan=True):
            mutated = True
        if not np.allclose(sd, rf, rtol=1e-10, atol=1e-10, equal_nan=True):
            d = np.abs(sd - rf)
            mismatches.append(f"{nm}: max_abs_diff={d.max():.3e} "
                              f"(n_diff={np.count_nonzero(d > 1e-10)})")
    # The reference must have done real work, and the SDFG side must
    # agree with it to 1e-10 on all 12 outputs.
    assert mutated, "reference left every output untouched -- kernel did not run"
    assert not mismatches, "\n".join(mismatches)
