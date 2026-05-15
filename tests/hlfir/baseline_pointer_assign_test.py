"""Baseline HLFIR coverage  --  Fortran ``POINTER`` rebinding under the
strict-no-aliasing assumption.

The bridge collapses ``ptr => target`` rebinds in
``hlfir-rewrite-pointer-assigns``: every read or write of the pointer
becomes an access to the rebind target's storage.  The pass emits a
warning per firing so callers see the no-alias assumption  --  Fortran
allows aliased pointer access, this collapse is unsafe if the program
relies on it.

Pinned coverage:
  * Pointer to a scalar struct field (``tmp => s%a``).
  * Pointer to a scalar local (``tmp => x``).
  * Both reads and writes through the pointer (``tmp = 13``,
    ``r = func(tmp)``).
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def test_pointer_to_scalar_local(tmp_path: Path):
    """``tmp => x; tmp = 13; res = tmp + 1``  --  pointer to a scalar local."""
    src = """
subroutine main(out)
  implicit none
  integer, intent(out) :: out
  integer, target  :: x
  integer, pointer :: tmp
  x = 0
  tmp => x
  tmp = 13
  out = tmp + 1
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "ptr_to_scalar_local")
    out_ref = mod.main()
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    out = np.zeros(1, dtype=np.int32)
    sdfg(out=out)
    assert int(out[0]) == int(out_ref) == 14


def test_dead_store_rebind_is_collapsed(tmp_path: Path):
    """Sequential dead-store rebinds (``ptr => A; ptr => B; use ptr``)
    are not ambiguous  --  only the LAST rebind is observable.  The
    pass takes the last and erases the earlier dead store.

    Pattern shows up in ICON-style code where flang lowers an
    aggregate rebind as multiple stores to the same pointer slot."""
    src = """
subroutine main(out)
  implicit none
  integer, intent(out) :: out
  integer, target  :: x, y
  integer, pointer :: tmp
  x = 1
  y = 2
  tmp => x
  tmp => y
  out = tmp
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    out = np.zeros(1, dtype=np.int32)
    sdfg(out=out)
    assert int(out[0]) == 2  # last rebind wins


def test_unsupported_interleaved_rebinds_raises(tmp_path: Path):
    """Loud-failure contract: a READ between two rebinds observes the
    earlier target  --  collapsing to one would lose that semantics.
    The pass must surface a clear unsupported error.

    NOTE: triggering this from Fortran source needs a non-trivial
    use of the first target between rebinds.  We assemble the IR
    pattern by force-feeding the bridge two rebind sites with a
    read in the middle; since Fortran source doesn't usually emit
    this shape, the test guards against future bridge changes that
    might silently coalesce reads with rebinds."""
    src = """
subroutine main(out)
  implicit none
  integer, intent(out) :: out(2)
  integer, target  :: x, y
  integer, pointer :: tmp
  x = 1
  y = 2
  tmp => x
  out(1) = tmp     ! read of tmp -> x
  tmp => y
  out(2) = tmp     ! read of tmp -> y
end subroutine main
"""
    with pytest.raises(RuntimeError):
        build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()


def test_unsupported_bounds_remap_raises(tmp_path: Path):
    """Loud-failure contract: ``ptr(0:n-1) => src(1:n)`` re-bases the
    lower bound.  Flang encodes the remap on the rebox's
    ``fir.shift`` / ``fir.shape_shift`` operand; collapsing without
    observing the remap would shift every subsequent read by
    ``remap_lo - 1``.  Until the remap is properly handled in the
    rewrite, the pass must reject."""
    src = """
subroutine main(n, src, res)
  implicit none
  integer, intent(in)        :: n
  real, intent(in), target   :: src(n)
  real, intent(out)          :: res
  real, pointer              :: w(:)
  w(0:n-1) => src(1:n)
  res = w(0)
end subroutine main
"""
    with pytest.raises(RuntimeError):
        build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()


def test_pointer_rebind_to_array_slice(tmp_path: Path):
    """``w => store(1:n); res = w(2) + w(4)``  --  pointer rebound to a
    triplet section of a TARGET array.  Distinct from
    ``ptr => target_decl`` (the existing test): the rebind value is
    ``fir.rebox(hlfir.designate(parent, slice))`` rather than
    ``fir.embox(declare)``.  ``hlfir-rewrite-pointer-assigns``'s
    slice-target arm forwards the rebound section box to every load
    of the pointer slot, skipping the alloca round-trip; the bridge's
    ``traceToDecl`` then walks through ``fir.rebox`` and the
    ``hlfir.designate`` chain back to the parent declare so each
    ``w(i)`` reads ``parent(i)`` directly.

    Strict-no-aliasing assumption: the rewrite is unsafe if the
    program relies on aliasing.  No runtime ALLOCATED checks.
    """
    src = """
subroutine main(n, store, res)
  implicit none
  integer, intent(in)             :: n
  real, intent(in), target        :: store(n)
  real, intent(out)               :: res
  real, pointer                   :: w(:)
  w => store(1:n)
  res = w(2) + w(4)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    n = 5
    store = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    res = np.zeros(1, dtype=np.float32)
    sdfg(n=n, store=store, res=res)
    assert res[0] == 6.0  # store[1] + store[3] = 2 + 4


def test_pointer_to_struct_scalar_field(tmp_path: Path):
    """``tmp => s%a; tmp = 13``  --  pointer rebound onto a scalar struct field.

    flatten-structs runs first and replaces ``s%a`` with a flat ``s_a``
    declare; the rewrite-pointer-assigns pass then traces the rebind's
    target through the box+embox chain to ``s_a`` and replaces every
    pointer use with the flat declare.
    """
    src = """
module lib
  implicit none
  type simple_type
    integer :: a
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real, intent(inout) :: d(2)
  type(simple_type), target :: s
  integer, pointer :: tmp
  s%a = 0
  tmp => s%a
  tmp = 13
  d(1) = real(s%a)
  d(2) = real(tmp)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "ptr_to_struct_field")
    d_ref = np.zeros(2, order="F", dtype=np.float32)
    mod.main(d_ref)
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    d = np.zeros(2, dtype=np.float32)
    sdfg(d=d)
    np.testing.assert_array_equal(d, d_ref)
    np.testing.assert_array_equal(d, [13.0, 13.0])
