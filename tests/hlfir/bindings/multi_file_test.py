"""Multi-HLFIR driver: parse several files, merge, drop dead siblings,
raise on unresolved calls.  Tests use pre-compiled HLFIR files written
to tmp_path so the flang toolchain is exercised exactly once per test.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _util import have_flang  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "dace" / "frontend" / "hlfir"))
from hlfir_to_sdfg import SDFGBuilder  # noqa: E402

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_FLANG = "flang-new-21"


def _hlfir(src: str, out: Path) -> Path:
    f90 = out.with_suffix(".f90")
    f90.write_text(src)
    subprocess.check_call([_FLANG, "-fc1", "-emit-hlfir", str(f90), "-o", str(out)])
    return out


def test_two_files_merge_drops_dead_helper(tmp_path: Path):
    """Entry in file B, unused helper in file A.  After the multi-file
    pipeline only the entry should remain — symbol-dce drops the
    helper since nothing references it."""
    _hlfir(
        """
subroutine unused_helper(x, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  integer :: i
  do i = 1, n
    x(i) = x(i) + 1.0d0
  end do
end subroutine
""", tmp_path / "a.hlfir")
    _hlfir(
        """
subroutine kernel(x, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  integer :: i
  do i = 1, n
    x(i) = dble(i)
  end do
end subroutine
""", tmp_path / "b.hlfir")

    b = SDFGBuilder.from_files([str(tmp_path / "a.hlfir"), str(tmp_path / "b.hlfir")], entry="_QPkernel")
    remaining = b.module.list_functions()
    assert remaining == ["_QPkernel"], \
        f"symbol-dce should have dropped unused_helper; got {remaining}"
    sdfg = b.build()
    assert sdfg.name == "kernel"
    assert "x" in sdfg.arrays
    assert "n" in sdfg.free_symbols


def test_unresolved_call_raises(tmp_path: Path):
    """An external call with no matching definition in the loaded set
    must surface as an error before SDFG construction."""
    _hlfir(
        """
subroutine caller(x, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  interface
    subroutine external_helper(x, n)
      integer, intent(in) :: n
      real(8), intent(inout) :: x(n)
    end subroutine
  end interface
  call external_helper(x, n)
end subroutine
""", tmp_path / "caller.hlfir")
    with pytest.raises(RuntimeError, match="pipeline failed"):
        SDFGBuilder.from_files([str(tmp_path / "caller.hlfir")], entry="_QPcaller")


def test_entry_missing_raises(tmp_path: Path):
    """Passing an entry name that isn't in any loaded file must
    raise — the driver relies on set_entry_symbol to find it."""
    _hlfir(
        """
subroutine foo(x, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  x(1) = 0.0d0
end subroutine
""", tmp_path / "foo.hlfir")
    with pytest.raises(RuntimeError, match="not found|dropped"):
        SDFGBuilder.from_files([str(tmp_path / "foo.hlfir")], entry="_QPnonexistent")


def test_cross_file_callee_gets_inlined(tmp_path: Path):
    """Entry in one file calls a helper defined in another file.  The
    multi-file pipeline must inline the helper into the entry so the
    SDFG sees the combined behaviour (the current _emit dispatch has no
    "call" handler; without inlining the helper would be silently
    dropped).  After pipeline the helper should be gone from the
    module and its assignment should appear inside the entry.
    """
    _hlfir(
        """
subroutine helper(x, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  integer :: i
  do i = 1, n
    x(i) = x(i) * 2.0d0
  end do
end subroutine
""", tmp_path / "helper.hlfir")
    _hlfir(
        """
subroutine kernel(x, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  interface
    subroutine helper(x, n)
      integer, intent(in) :: n
      real(8), intent(inout) :: x(n)
    end subroutine
  end interface
  integer :: i
  do i = 1, n
    x(i) = dble(i)
  end do
  call helper(x, n)
end subroutine
""", tmp_path / "kernel.hlfir")

    b = SDFGBuilder.from_files([str(tmp_path / "kernel.hlfir"), str(tmp_path / "helper.hlfir")], entry="_QPkernel")
    remaining = b.module.list_functions()
    assert remaining == ["_QPkernel"], f"helper should have inlined + dce'd; got {remaining}"
    # The helper's ``* 2.0`` multiply should live in the entry's body.
    dump = b.module.dump()
    assert "2.000000e+00" in dump, \
        "helper's x(i) * 2.0d0 should appear inlined inside kernel"


def test_parse_files_declaration_loses_to_definition(tmp_path: Path):
    """When two files both expose a symbol, the real definition must
    win over an external declaration so the full body (x(1)=99) ends
    up inlined into the entry — not a verify-no-unresolved-calls error."""
    _hlfir(
        """
subroutine kernel(x, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  interface
    subroutine shared(x, n)
      integer, intent(in) :: n
      real(8), intent(inout) :: x(n)
    end subroutine
  end interface
  call shared(x, n)
end subroutine
""", tmp_path / "caller.hlfir")
    _hlfir(
        """
subroutine shared(x, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  x(1) = 99.0d0
end subroutine
""", tmp_path / "shared.hlfir")
    b = SDFGBuilder.from_files([str(tmp_path / "caller.hlfir"), str(tmp_path / "shared.hlfir")], entry="_QPkernel")
    # Once the definition wins, inline-all folds shared into kernel and
    # symbol-dce drops the helper.  The 99.0 constant must land inside
    # kernel's body as proof the real definition was used — if the
    # declaration had won, verify-no-unresolved-calls would have errored.
    assert b.module.list_functions() == ["_QPkernel"]
    assert "9.900000e+01" in b.module.dump()
