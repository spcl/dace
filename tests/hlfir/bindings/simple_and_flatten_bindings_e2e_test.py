"""f90-binding e2e for (a) the simplest possible flat SDFG and
(b) a struct-flatten-requiring SDFG, with an explicit assertion that
``hlfir-flatten-structs`` recorded a non-empty ``FlattenPlan`` so the
binding actually reconstructs the derived type (not a silent no-op).

Complements ``cloudsc_flux_bindings_e2e_test`` (CLOUDSC-shaped flat)
and ``struct_bindings_e2e_test`` (the original three struct shapes):
the simple case pins the minimal binding surface; the flatten case
pins that a derived-type dummy round-trips correctly through the
generated ``c_f_pointer`` alias path AND that the recipe was emitted.
"""

import ctypes
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


def _compile_so(out_so: Path, *sources: Path, mod_dir: Path, link_so: Path | None = None):
    cmd = ["gfortran", "-shared", "-fPIC", "-O0", "-fno-fast-math", "-ffp-contract=off", f"-J{mod_dir}"]
    cmd.extend(str(s) for s in sources)
    cmd.extend(["-o", str(out_so)])
    if link_so is not None:
        cmd.extend([f"-L{link_so.parent}", f"-Wl,-rpath,{link_so.parent}", f"-l:{link_so.name}"])
    subprocess.check_call(cmd, cwd=mod_dir)


def _build_sdfg_binding_lib(tmp_path, *, kernel_src, entry, sdfg_name, iface, sdfg_driver_src, drv_name, types_src=""):
    """Build the SDFG, emit its binding, link types+binding+driver
    against the SDFG ``.so``.  Returns ``(ctypes lib, FlattenPlan)``.

    ``types_src`` (the derived-type module) is compiled first so its
    ``.mod`` is on the include path for the binding + driver.
    """
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(kernel_src, sdfg_dir, name=sdfg_name, entry=entry)
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = iface.entry  # match __dace_{init,exit}_<entry> symbols
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)

    bindings_path = tmp_path / f"{iface.entry}_bindings.f90"
    emit_bindings(sdfg._frozen_signature, iface, plan, str(bindings_path))
    drv_path = tmp_path / f"{drv_name}.f90"
    drv_path.write_text(sdfg_driver_src)

    build_dir = tmp_path / "sdfg_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    srcs = []
    if types_src:
        tp = tmp_path / f"{iface.entry}_types.f90"
        tp.write_text(types_src)
        srcs.append(tp)
    srcs += [bindings_path, drv_path]
    drv_so = build_dir / f"{drv_name}.so"
    _compile_so(drv_so, *srcs, mod_dir=build_dir, link_so=so_path)
    return ctypes.CDLL(str(drv_so)), plan


def _build_ref_lib(tmp_path, *, kernel_src, ref_driver_src, name, types_src=""):
    ref_dir = tmp_path / "ref_build"
    ref_dir.mkdir(parents=True, exist_ok=True)
    srcs = []
    if types_src:
        tp = ref_dir / f"{name}_types.f90"
        tp.write_text(types_src)
        srcs.append(tp)
    k = ref_dir / f"{name}_k.f90"
    k.write_text(kernel_src)
    d = ref_dir / f"{name}_d.f90"
    d.write_text(ref_driver_src)
    srcs += [k, d]
    so = ref_dir / f"{name}_ref.so"
    _compile_so(so, *srcs, mod_dir=ref_dir)
    return ctypes.CDLL(str(so))


# ---------------------------------------------------------------------------
# (a) Simplest flat SDFG: scalar n + two rank-1 arrays.
# ---------------------------------------------------------------------------

_SIMPLE_SRC = """
subroutine saxpy2(n, x, y)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in)    :: x(n)
  real(8), intent(inout) :: y(n)
  integer :: i
  do i = 1, n
    y(i) = 2.0d0 * x(i) + y(i)
  end do
end subroutine saxpy2
"""

_SIMPLE_SDFG_DRIVER = """
subroutine run_saxpy2(n, x, y) bind(c, name='run_saxpy2')
  use iso_c_binding
  use saxpy2_dace_bindings
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(in)    :: x(n)
  real(c_double), intent(inout) :: y(n)
  call saxpy2_dace(n, x, y)
  call saxpy2_dace_finalize()
end subroutine run_saxpy2
"""

_SIMPLE_REF_DRIVER = """
subroutine run_saxpy2_ref(n, x, y) bind(c, name='run_saxpy2_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(in)    :: x(n)
  real(c_double), intent(inout) :: y(n)
  external :: saxpy2
  call saxpy2(n, x, y)
end subroutine run_saxpy2_ref
"""

_SIMPLE_IFACE = OriginalInterface(
    entry="saxpy2",
    args=(
        OriginalArg(name="n", fortran_type="integer", rank=0, shape=(), intent="in", struct_type=None),
        OriginalArg(name="x", fortran_type="real(8)", rank=1, shape=("n", ), intent="in", struct_type=None),
        OriginalArg(name="y", fortran_type="real(8)", rank=1, shape=("n", ), intent="inout", struct_type=None),
    ),
    struct_types={},
    used_modules={},
)


def test_simple_flat_f90_binding(tmp_path: Path):
    """Minimal binding surface: one scalar + two rank-1 arrays, no
    flattening (plan must be empty)."""
    lib, plan = _build_sdfg_binding_lib(tmp_path,
                                        kernel_src=_SIMPLE_SRC,
                                        entry="_QPsaxpy2",
                                        sdfg_name="saxpy2",
                                        iface=_SIMPLE_IFACE,
                                        sdfg_driver_src=_SIMPLE_SDFG_DRIVER,
                                        drv_name="saxpy2_drv")
    assert not plan.entries, "flat kernel must not produce flatten entries"
    ref = _build_ref_lib(tmp_path, kernel_src=_SIMPLE_SRC, ref_driver_src=_SIMPLE_REF_DRIVER, name="saxpy2")

    n = 17
    rng = np.random.default_rng(3)
    x = np.asfortranarray(rng.standard_normal(n))

    def _call(l, fn):
        f = getattr(l, fn)
        f.restype = None
        f.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        y = np.asfortranarray(np.ones(n))
        f(n, x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return y

    np.testing.assert_allclose(_call(lib, "run_saxpy2"), _call(ref, "run_saxpy2_ref"), rtol=1e-13, atol=1e-13)


# ---------------------------------------------------------------------------
# (b) Flatten-requiring SDFG: a derived-type dummy with two 2-D real
#     array members + a scalar member -> hlfir-flatten-structs must
#     unpack it and the binding must reconstruct it via c_f_pointer.
# ---------------------------------------------------------------------------

_FLAT_TYPES = """
module mo_state
  use iso_c_binding
  implicit none
  integer, parameter :: NX = 6, NY = 4
  type :: t_state
     real(c_double) :: u(NX, NY)
     real(c_double) :: v(NX, NY)
     real(c_double) :: scal
  end type t_state
end module mo_state
"""

_FLAT_KERNEL = """
subroutine update_state(s)
  use mo_state
  implicit none
  type(t_state), intent(inout) :: s
  integer :: i, j
  do j = 1, NY
     do i = 1, NX
        s%u(i, j) = s%u(i, j) + s%scal * s%v(i, j)
     end do
  end do
end subroutine update_state
"""

_FLAT_SRC = _FLAT_TYPES + _FLAT_KERNEL

_FLAT_SDFG_DRIVER = """
subroutine run_state(u, v, scal) bind(c, name='run_state')
  use iso_c_binding
  use mo_state, only: t_state, NX, NY
  use update_state_dace_bindings
  implicit none
  real(c_double), intent(inout) :: u(NX, NY), v(NX, NY)
  real(c_double), value :: scal
  type(t_state), target :: s
  s%u = u
  s%v = v
  s%scal = scal
  call update_state_dace(s)
  call update_state_dace_finalize()
  u = s%u
  v = s%v
end subroutine run_state
"""

_FLAT_REF_DRIVER = """
subroutine run_state_ref(u, v, scal) bind(c, name='run_state_ref')
  use iso_c_binding
  use mo_state, only: t_state, NX, NY
  implicit none
  real(c_double), intent(inout) :: u(NX, NY), v(NX, NY)
  real(c_double), value :: scal
  type(t_state) :: s
  external :: update_state
  s%u = u
  s%v = v
  s%scal = scal
  call update_state(s)
  u = s%u
  v = s%v
end subroutine run_state_ref
"""

_FLAT_IFACE = OriginalInterface(
    entry="update_state",
    args=(OriginalArg(name="s", fortran_type="type(t_state)", rank=0, shape=(), intent="inout",
                      struct_type="t_state"), ),
    struct_types={},
    used_modules={"mo_state": ("t_state", )},
)


def test_struct_flatten_f90_binding(tmp_path: Path):
    """A ``type(t_state)`` dummy with two 2-D members + a scalar must
    be flattened by the bridge (non-empty FlattenPlan) and round-trip
    correctly through the generated binding's struct reconstruction."""
    lib, plan = _build_sdfg_binding_lib(tmp_path,
                                        kernel_src=_FLAT_SRC,
                                        entry="_QPupdate_state",
                                        sdfg_name="update_state",
                                        iface=_FLAT_IFACE,
                                        sdfg_driver_src=_FLAT_SDFG_DRIVER,
                                        drv_name="state_drv",
                                        types_src=_FLAT_TYPES)

    # The flatten recipe must actually have been generated: one entry
    # per flattened struct member reaching the SDFG surface.
    flat_targets = {fn for e in plan.entries for fn in e.recipe.flat_names}
    assert plan.entries, "struct dummy must produce a non-empty FlattenPlan"
    assert any("u" in t for t in flat_targets), flat_targets
    assert any("v" in t for t in flat_targets), flat_targets

    ref = _build_ref_lib(tmp_path,
                         kernel_src=_FLAT_KERNEL,
                         ref_driver_src=_FLAT_REF_DRIVER,
                         name="state",
                         types_src=_FLAT_TYPES)

    nx, ny = 6, 4
    rng = np.random.default_rng(11)
    u0 = np.asfortranarray(rng.standard_normal((nx, ny)))
    v0 = np.asfortranarray(rng.standard_normal((nx, ny)))
    scal = 0.75

    def _call(l, fn):
        f = getattr(l, fn)
        f.restype = None
        f.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_double]
        u = u0.copy(order="F")
        v = v0.copy(order="F")
        f(u.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
          ctypes.c_double(scal))
        return u, v

    u_s, v_s = _call(lib, "run_state")
    u_r, v_r = _call(ref, "run_state_ref")
    np.testing.assert_allclose(u_s, u_r, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(v_s, v_r, rtol=1e-13, atol=1e-13)
