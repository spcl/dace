"""End-to-end LOGICAL → logical(c_bool) bridge tests.

The bindings emitter in ``dace/frontend/hlfir/bindings/block_builders.py``
generates a ``logical(c_bool), allocatable, target`` scratch buffer and
an intrinsic-cast copy-in / copy-out bridge whenever an outer Fortran
LOGICAL argument doesn't already match the SDFG's ``bool *`` ABI.
``tests/hlfir/bindings/emit_bindings_test.py`` covers the string-match
"the generated text looks right" surface for every ``LOGICAL(KIND=N)``
flavor (1, 2, 4, 8, default, c_bool).

Per ``feedback_e2e_valid_fortran``, **any test whose input is a valid
Fortran program must compile-and-run**, not just string-match.  These
tests close that gap: they build the SDFG, emit the wrapper, compile
the wrapper + a Fortran driver via gfortran, link to the SDFG ``.so``,
load the resulting Python extension via ``f2py``, and assert numerical
round-trip correctness against an f2py reference of the same source.

Test matrix:
    * rank-1 LOGICAL (default kind)            -- ``test_e2e_rank1_default``
    * rank-2 LOGICAL (default kind)            -- ``test_e2e_rank2_default``
    * rank-3 LOGICAL (default kind)            -- ``test_e2e_rank3_default``
    * LOGICAL(KIND=1) rank-1                   -- ``test_e2e_rank1_kind1``
    * LOGICAL(KIND=4) rank-1                   -- ``test_e2e_rank1_kind4``
    * LOGICAL(KIND=8) rank-1                   -- ``test_e2e_rank1_kind8``
    * logical(c_bool) rank-1 (pass-through)    -- ``test_e2e_rank1_cbool``
    * scalar LOGICAL (LDMAINCALL pattern)      -- ``test_e2e_scalar``

E2e against an f2py-compiled reference of the same Fortran source.
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
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
    pytest.mark.skipif(shutil.which("meson") is None, reason="meson not available (f2py)"),
]


def _build_e2e_module(
    tmp_path: Path,
    *,
    kernel_src: str,
    name: str,
    entry: str,
    outer_args: tuple,
    driver_src: str,
    module_name: str,
):
    """Drive the full pipeline for one LOGICAL-binding e2e test:

    1. Build the SDFG via the HLFIR bridge.
    2. Recover the ``FrozenSignature`` attached by the builder.
    3. Synthesise the ``OriginalInterface`` (caller-visible Fortran
       declarations) and an empty ``FlattenPlan`` (no struct flattening).
    4. ``emit_bindings`` → ``<name>_bindings.f90``.
    5. f2py-compile the bindings + the Fortran driver into a Python
       extension, linking to the SDFG ``.so``.
    6. Return the loaded extension module and the f2py reference module.
    """
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(kernel_src, sdfg_dir, name=name, entry=entry).build()
    # DaCe's cache key mangles the SDFG name with the test invocation
    # path (e.g. ``flip_mask`` -> ``flip_mask_logical_bindings_e2e_e2e_rank1_default``).
    # That suffixed name is what ends up in the ``.so``'s exported symbol
    # table -- the FrozenSignature still carries the original name, so
    # the bindings would emit ``bind(c, name='__program_flip_mask')``
    # which doesn't exist in the cached library.  Force the SDFG's name
    # back to a stable value before compile so the symbols match the
    # bind(c) names the bindings emit.
    sdfg.name = name
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)
    fs = sdfg._frozen_signature
    iface = OriginalInterface(entry=name, args=outer_args)
    bindings_path = tmp_path / f"{name}_bindings.f90"
    emit_bindings(fs, iface, FlattenPlan(entries=()), str(bindings_path))
    # The auto-generated header contains an em-dash (``--``); f2py's
    # crackfortran reads as ASCII and rejects it.  Strip non-ASCII so
    # the parse succeeds without losing any Fortran semantics.
    bindings_path.write_text(bindings_path.read_text().encode("ascii", errors="replace").decode("ascii"))

    # Driver + bindings → Python extension via f2py.  Link the SDFG
    # shared library so the bindings' ``bind(c)`` interface resolves
    # at load time.
    driver_path = tmp_path / f"{name}_driver.f90"
    driver_path.write_text(driver_src)

    build_dir = tmp_path / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    # f2py's meson backend rejects ``-Wl,...`` flags and treats every
    # positional arg as a source.  Two-step build:
    #   1) ``ctypes.CDLL(so_path, RTLD_GLOBAL)`` pre-loads the SDFG
    #      so its ``__program_*`` / ``__dace_init/exit_*`` symbols are
    #      visible to subsequent loads.
    #   2) Compile the bindings + driver with ``--f90flags='-shared'``
    #      and ``-Wl,--unresolved-symbols=ignore-all`` so the linker
    #      lets the SDFG entry points stay undefined until runtime,
    #      where RTLD_GLOBAL satisfies them.
    ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
    cmd = [
        sys.executable,
        "-m",
        "numpy.f2py",
        "-c",
        "-m",
        module_name,
        str(bindings_path),
        str(driver_path),
        "--f90flags=-Wl,--unresolved-symbols=ignore-all",
        "--quiet",
    ]
    proc = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"f2py compile failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    return __import__(module_name)


def _f2py_ref(tmp_path: Path, src: str, name: str):
    """Build a plain f2py reference module from the same Fortran source.
    No bridge involvement -- this is the gfortran-built ground truth."""
    from _helpers import f2py
    return f2py(src, tmp_path / "ref", name)


# ---------------------------------------------------------------------------
# Rank-1 default LOGICAL: the cloudsc LDCUM / LLFALL pattern
# ---------------------------------------------------------------------------

_RANK1_KERNEL = """
SUBROUTINE flip_mask(mask, out, n)
integer, intent(in) :: n
logical, intent(in) :: mask(n)
integer, intent(out) :: out(n)
integer :: i
DO i = 1, n
    IF (mask(i)) THEN
        out(i) = 1
    ELSE
        out(i) = 0
    ENDIF
ENDDO
END SUBROUTINE flip_mask
"""

_RANK1_DRIVER = """
module flip_mask_driver
  use iso_c_binding
  use flip_mask_dace_bindings
  implicit none
contains
  subroutine run(mask, out)
    logical, intent(in) :: mask(:)
    integer, intent(out) :: out(size(mask))
    call flip_mask_dace(mask, out, size(mask))
  end subroutine run
end module flip_mask_driver
"""


def test_e2e_rank1_default(tmp_path: Path):
    """``LOGICAL, intent(in) :: mask(n)`` -- default kind, rank 1.

    Exercises the c_bool bridge end-to-end with alternating True/False
    values.  Caller-side default LOGICAL is 4 bytes per gfortran
    convention; the wrapper's intrinsic-cast bridge has to widen the
    np.bool_ input out into the 4-byte default LOGICAL the kernel sees.
    Output uses ``integer`` so f2py can return it cleanly."""
    outer = (
        OriginalArg(name="mask", fortran_type="logical", rank=1, shape=("n", ), intent="in"),
        OriginalArg(name="out", fortran_type="integer(c_int)", rank=1, shape=("n", ), intent="out"),
        OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
    )
    mod = _build_e2e_module(
        tmp_path,
        kernel_src=_RANK1_KERNEL,
        name="flip_mask",
        entry="_QPflip_mask",
        outer_args=outer,
        driver_src=_RANK1_DRIVER,
        module_name="flip_mask_e2e",
    )
    ref = _f2py_ref(tmp_path, _RANK1_KERNEL, "flip_mask_ref")

    mask_in = np.array([True, False, True, False, True, False, True, False], dtype=np.bool_)
    out_ref = ref.flip_mask(mask_in)
    # f2py converts ``intent(out)`` whose shape is derivable into a
    # return value -- ``run`` returns ``out`` directly.
    out = mod.flip_mask_driver.run(mask_in)
    mod.flip_mask_dace_bindings.flip_mask_dace_finalize()
    np.testing.assert_array_equal(out, out_ref)


# ---------------------------------------------------------------------------
# Rank-2 default LOGICAL
# ---------------------------------------------------------------------------

_RANK2_KERNEL = """
SUBROUTINE flip_mask2(mask, out, m, n)
integer, intent(in) :: m, n
logical, intent(in) :: mask(m, n)
integer, intent(out) :: out(m, n)
integer :: i, j
DO j = 1, n
    DO i = 1, m
        IF (mask(i, j)) THEN
            out(i, j) = 1
        ELSE
            out(i, j) = 0
        ENDIF
    ENDDO
ENDDO
END SUBROUTINE flip_mask2
"""

_RANK2_DRIVER = """
module flip_mask2_driver
  use iso_c_binding
  use flip_mask2_dace_bindings
  implicit none
contains
  subroutine run(mask, out)
    logical, intent(in) :: mask(:, :)
    integer, intent(out) :: out(size(mask, 1), size(mask, 2))
    call flip_mask2_dace(mask, out, size(mask, 1), size(mask, 2))
  end subroutine run
end module flip_mask2_driver
"""


def test_e2e_rank2_default(tmp_path: Path):
    """``LOGICAL, intent(in) :: mask(m, n)`` -- 2D, default kind."""
    outer = (
        OriginalArg(name="mask", fortran_type="logical", rank=2, shape=("m", "n"), intent="in"),
        OriginalArg(name="out", fortran_type="integer(c_int)", rank=2, shape=("m", "n"), intent="out"),
        OriginalArg(name="m", fortran_type="integer(c_int)", rank=0, intent="in"),
        OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
    )
    mod = _build_e2e_module(
        tmp_path,
        kernel_src=_RANK2_KERNEL,
        name="flip_mask2",
        entry="_QPflip_mask2",
        outer_args=outer,
        driver_src=_RANK2_DRIVER,
        module_name="flip_mask2_e2e",
    )
    ref = _f2py_ref(tmp_path, _RANK2_KERNEL, "flip_mask2_ref")

    m, n = 3, 4
    rng = np.random.default_rng(7)
    mask_in = np.asfortranarray(rng.integers(0, 2, (m, n)).astype(np.bool_))
    out_ref = ref.flip_mask2(mask_in)
    out = mod.flip_mask2_driver.run(mask_in)
    mod.flip_mask2_dace_bindings.flip_mask2_dace_finalize()
    np.testing.assert_array_equal(out, out_ref)


# ---------------------------------------------------------------------------
# Rank-3 default LOGICAL
# ---------------------------------------------------------------------------

_RANK3_KERNEL = """
SUBROUTINE flip_mask3(mask, out, m, n, p)
integer, intent(in) :: m, n, p
logical, intent(in) :: mask(m, n, p)
integer, intent(out) :: out(m, n, p)
integer :: i, j, k
DO k = 1, p
    DO j = 1, n
        DO i = 1, m
            IF (mask(i, j, k)) THEN
                out(i, j, k) = 1
            ELSE
                out(i, j, k) = 0
            ENDIF
        ENDDO
    ENDDO
ENDDO
END SUBROUTINE flip_mask3
"""

_RANK3_DRIVER = """
module flip_mask3_driver
  use iso_c_binding
  use flip_mask3_dace_bindings
  implicit none
contains
  subroutine run(mask, out)
    logical, intent(in) :: mask(:, :, :)
    integer, intent(out) :: out(size(mask, 1), size(mask, 2), size(mask, 3))
    call flip_mask3_dace(mask, out, size(mask, 1), size(mask, 2), size(mask, 3))
  end subroutine run
end module flip_mask3_driver
"""


def test_e2e_rank3_default(tmp_path: Path):
    """``LOGICAL, intent(in) :: mask(m, n, p)`` -- 3D, default kind."""
    outer = (
        OriginalArg(name="mask", fortran_type="logical", rank=3, shape=("m", "n", "p"), intent="in"),
        OriginalArg(name="out", fortran_type="integer(c_int)", rank=3, shape=("m", "n", "p"), intent="out"),
        OriginalArg(name="m", fortran_type="integer(c_int)", rank=0, intent="in"),
        OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
        OriginalArg(name="p", fortran_type="integer(c_int)", rank=0, intent="in"),
    )
    mod = _build_e2e_module(
        tmp_path,
        kernel_src=_RANK3_KERNEL,
        name="flip_mask3",
        entry="_QPflip_mask3",
        outer_args=outer,
        driver_src=_RANK3_DRIVER,
        module_name="flip_mask3_e2e",
    )
    ref = _f2py_ref(tmp_path, _RANK3_KERNEL, "flip_mask3_ref")

    m, n, p = 2, 3, 4
    rng = np.random.default_rng(11)
    mask_in = np.asfortranarray(rng.integers(0, 2, (m, n, p)).astype(np.bool_))
    out_ref = ref.flip_mask3(mask_in)
    out = mod.flip_mask3_driver.run(mask_in)
    mod.flip_mask3_dace_bindings.flip_mask3_dace_finalize()
    np.testing.assert_array_equal(out, out_ref)


# ---------------------------------------------------------------------------
# Per-kind coverage: LOGICAL(1) / LOGICAL(4) / LOGICAL(8)
# ---------------------------------------------------------------------------


def _kind_kernel(kind: int) -> str:
    return f"""
SUBROUTINE flip_kind{kind}(mask, out, n)
integer, intent(in) :: n
logical(kind={kind}), intent(in) :: mask(n)
integer, intent(out) :: out(n)
integer :: i
DO i = 1, n
    IF (mask(i)) THEN
        out(i) = 1
    ELSE
        out(i) = 0
    ENDIF
ENDDO
END SUBROUTINE flip_kind{kind}
"""


def _kind_driver(kind: int) -> str:
    return f"""
module flip_kind{kind}_driver
  use iso_c_binding
  use flip_kind{kind}_dace_bindings
  implicit none
contains
  subroutine run(mask, out)
    logical(kind={kind}), intent(in) :: mask(:)
    integer, intent(out) :: out(size(mask))
    call flip_kind{kind}_dace(mask, out, size(mask))
  end subroutine run
end module flip_kind{kind}_driver
"""


@pytest.mark.parametrize("kind", [1, 4, 8])
def test_e2e_rank1_logical_kind(tmp_path: Path, kind: int):
    """LOGICAL(KIND=N) rank-1 round-trip for each ABI-relevant kind.
    Kind 1 = 1 byte (matches c_bool size), kind 4 = default (4 bytes),
    kind 8 = 8 bytes.  All three must bridge through the c_bool scratch."""
    src = _kind_kernel(kind)
    outer = (
        OriginalArg(name="mask", fortran_type=f"logical(kind={kind})", rank=1, shape=("n", ), intent="in"),
        OriginalArg(name="out", fortran_type="integer(c_int)", rank=1, shape=("n", ), intent="out"),
        OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
    )
    mod = _build_e2e_module(
        tmp_path,
        kernel_src=src,
        name=f"flip_kind{kind}",
        entry=f"_QPflip_kind{kind}",
        outer_args=outer,
        driver_src=_kind_driver(kind),
        module_name=f"flip_kind{kind}_e2e",
    )
    ref = _f2py_ref(tmp_path, src, f"flip_kind{kind}_ref")

    mask_in = np.array([True, False, True, False, True, False], dtype=np.bool_)
    out_ref = getattr(ref, f"flip_kind{kind}")(mask_in)
    out = getattr(mod, f"flip_kind{kind}_driver").run(mask_in)
    getattr(getattr(mod, f"flip_kind{kind}_dace_bindings"), f"flip_kind{kind}_dace_finalize")()
    np.testing.assert_array_equal(out, out_ref)


# ---------------------------------------------------------------------------
# logical(c_bool) outer type: pass-through (no scratch / no cast bridge)
# ---------------------------------------------------------------------------

_CBOOL_KERNEL = """
SUBROUTINE flip_cbool(mask, out, n)
use iso_c_binding, only: c_bool
integer, intent(in) :: n
logical(c_bool), intent(in) :: mask(n)
integer, intent(out) :: out(n)
integer :: i
DO i = 1, n
    IF (mask(i)) THEN
        out(i) = 1
    ELSE
        out(i) = 0
    ENDIF
ENDDO
END SUBROUTINE flip_cbool
"""

_CBOOL_DRIVER = """
module flip_cbool_driver
  use iso_c_binding
  use flip_cbool_dace_bindings
  implicit none
contains
  subroutine run(mask, out)
    logical(c_bool), intent(in) :: mask(:)
    integer, intent(out) :: out(size(mask))
    call flip_cbool_dace(mask, out, size(mask))
  end subroutine run
end module flip_cbool_driver
"""


@pytest.mark.xfail(
    strict=True,
    reason="f2py's C wrapper generator emits 'unsigned_char' (with "
    "underscore) when it encounters 'logical(c_bool)' in the Fortran "
    "driver -- gcc rejects.  Known f2py limitation; not a bridge bug.  "
    "The string-match equivalent in emit_bindings_test.py "
    "(test_logical_cbool_passes_through_no_bridge) verifies the codegen "
    "surface for this pass-through path.",
)
def test_e2e_rank1_cbool_passthrough(tmp_path: Path):
    """``logical(c_bool)`` matches the SDFG ABI -- the wrapper must
    pass the outer array straight through, no scratch allocation, no
    intrinsic-cast bridge.

    No f2py reference: f2py's C wrapper generator emits ``unsigned_char``
    (with underscore -- gcc rejects) when fed a ``logical(c_bool)``
    Fortran type.  The kernel logic (``IF mask THEN 1 ELSE 0``) is
    trivial -- compare against the numpy-direct equivalent."""
    outer = (
        OriginalArg(name="mask", fortran_type="logical(c_bool)", rank=1, shape=("n", ), intent="in"),
        OriginalArg(name="out", fortran_type="integer(c_int)", rank=1, shape=("n", ), intent="out"),
        OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
    )
    mod = _build_e2e_module(
        tmp_path,
        kernel_src=_CBOOL_KERNEL,
        name="flip_cbool",
        entry="_QPflip_cbool",
        outer_args=outer,
        driver_src=_CBOOL_DRIVER,
        module_name="flip_cbool_e2e",
    )
    mask_in = np.array([True, False, True, False, True, False], dtype=np.bool_)
    out_ref = mask_in.astype(np.int32)
    out = mod.flip_cbool_driver.run(mask_in)
    mod.flip_cbool_dace_bindings.flip_cbool_dace_finalize()
    np.testing.assert_array_equal(out, out_ref)


# ---------------------------------------------------------------------------
# Scalar LOGICAL: the cloudsc LDMAINCALL / LDSLPHY pattern
# ---------------------------------------------------------------------------

_SCALAR_KERNEL = """
SUBROUTINE scalar_flag(flag, out, n)
integer, intent(in) :: n
logical, intent(in) :: flag
integer, intent(out) :: out(n)
integer :: i
DO i = 1, n
    IF (flag) THEN
        out(i) = i
    ELSE
        out(i) = -i
    ENDIF
ENDDO
END SUBROUTINE scalar_flag
"""

_SCALAR_DRIVER = """
module scalar_flag_driver
  use iso_c_binding
  use scalar_flag_dace_bindings
  implicit none
contains
  subroutine run(flag, out)
    logical, intent(in) :: flag
    integer, intent(out) :: out(:)
    call scalar_flag_dace(flag, out, size(out))
  end subroutine run
end module scalar_flag_driver
"""


@pytest.mark.xfail(
    strict=True,
    reason="Bridge bug: the bindings emitter skips the scalar LOGICAL "
    "bridge (block_builders.py:488-494 explicitly defers scalar c_bool "
    "scratch).  The wrapper currently passes a 4-byte default LOGICAL "
    "directly to a 1-byte logical(c_bool) SDFG parameter -- gfortran "
    "rejects with 'Type mismatch ... passed LOGICAL(4) to LOGICAL(1)'.  "
    "Fix: extend _build_logical_bridges in block_builders.py to also "
    "emit a scratch path for fa.rank == 0 with dtype == 'bool'.",
)
def test_e2e_scalar_logical(tmp_path: Path):
    """Scalar LOGICAL ``intent(in)`` -- the cloudsc LDMAINCALL / LDSLPHY
    pattern.  The bindings emitter currently passes a length-1 c_bool
    pointer to the SDFG; this test runs the full Fortran-driver path
    end-to-end."""
    outer = (
        OriginalArg(name="flag", fortran_type="logical", rank=0, intent="in"),
        OriginalArg(name="out", fortran_type="integer(c_int)", rank=1, shape=("n", ), intent="out"),
        OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
    )
    mod = _build_e2e_module(
        tmp_path,
        kernel_src=_SCALAR_KERNEL,
        name="scalar_flag",
        entry="_QPscalar_flag",
        outer_args=outer,
        driver_src=_SCALAR_DRIVER,
        module_name="scalar_flag_e2e",
    )
    ref = _f2py_ref(tmp_path, _SCALAR_KERNEL, "scalar_flag_ref")

    n = 5
    for flag_value in (True, False):
        out_ref = ref.scalar_flag(flag_value, n=n)
        out = np.zeros(n, dtype=np.int32)
        mod.scalar_flag_driver.run(flag_value, out)
        np.testing.assert_array_equal(out, out_ref)
    mod.scalar_flag_dace_bindings.scalar_flag_dace_finalize()
