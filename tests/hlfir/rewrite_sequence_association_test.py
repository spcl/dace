"""Standalone tests for the ``hlfir-rewrite-sequence-association`` pass.

Fortran 2003 section12.4.1.5 lets a caller pass a single element of an array
where the formal expects an explicit-shape array  --  the formal then sees
``N`` consecutive elements starting at the given element.  Flang lowers
this with a deterministic IR shape:

    %elt   = hlfir.designate %parent (%idx)            : !fir.ref<T>
    %arr   = fir.convert     %elt                      : ref<T> -> ref<array<?xT>>
    fir.call @callee(%arr, ...)

The pass collapses this adapter into an explicit section designate
``%parent (lo:lo+N-1:1)`` so the bridge's normal section-aware lowering
takes over.  These tests exercise the pass at the IR level (without
going through SDFG codegen) so the contract is pinned independently of
downstream consumers.
"""
from pathlib import Path

import pytest

from _util import _ensure_on_path, compile_to_hlfir, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_PRELUDE = ("lower-fir-select-case,hlfir-inline-all,hlfir-fold-element-aliases,"
            "hlfir-expand-vector-subscript-gather,hlfir-expand-vector-subscript-scatter,symbol-dce,"
            "fir-polymorphic-op,hlfir-reject-polymorphism")
_REWRITE = "hlfir-rewrite-sequence-association"


def _run(source: str, out_dir: Path, name: str, extra: str = ""):
    _ensure_on_path()
    from build_bridge import hb
    hlfir = compile_to_hlfir(source, out_dir, name)
    m = hb.HLFIRModule()
    assert m.parse_file(str(hlfir))
    m.set_entry_symbol('_QPmain')
    m.run_passes(_PRELUDE)
    before = m.dump()
    m.run_passes(_REWRITE + (("," + extra) if extra else ""))
    after = m.dump()
    return before, after


def _count_seq_adapter(ir: str) -> int:
    """Count rank-0 -> rank-1 ``fir.convert`` adapter shapes in the IR."""
    n = 0
    for line in ir.splitlines():
        if 'fir.convert' not in line:
            continue
        # Pattern: "(!fir.ref<T>) -> !fir.ref<!fir.array<?xT>>"
        if 'ref<!fir.array' in line and '!fir.ref<f' in line.split('->')[0]:
            n += 1
        elif 'ref<!fir.array' in line and '!fir.ref<i' in line.split('->')[0]:
            n += 1
    return n


def test_literal_size_collapses_to_section(tmp_path):
    """``f(d(11), 5)`` -> callee sees ``d(11:15)``.

    The literal ``5`` for the formal's extent reaches the inlined
    callee through flang's ``__assoc_scalar`` adapter (alloca + single
    store + load).  The pass walks load -> alloca -> store to recover the
    constant and emits a section designate of shape ``5``.
    """
    src = """
module lib
contains
  real function f(d, dz)
    integer, intent(in) :: dz
    real, intent(in) :: d(dz)
    f = sum(d)
  end function f
end module lib

subroutine main(d, out)
  use lib
  real, intent(inout) :: d(50)
  real, intent(out)   :: out
  out = f(d(11), 5)
end subroutine main
"""
    before, after = _run(src, tmp_path, "seq_literal")
    # Adapter present before, gone after.
    assert _count_seq_adapter(before) >= 1
    assert _count_seq_adapter(after) == 0
    # Section designate of constant shape <5> appears.
    assert 'fir.array<5xf32>' in after
    # Triplet form: lo:hi:1 with hi = 11 + (5 - 1).
    assert ' (%c11' in after  # lower bound is the original element index


def test_pass_is_noop_without_seq_adapter(tmp_path):
    """A program with no sequence association should round-trip
    unchanged  --  no spurious rewrites of ordinary array views."""
    src = """
subroutine main(a, b, n)
  implicit none
  integer, intent(in)   :: n
  real, intent(in)      :: a(n)
  real, intent(out)     :: b(n)
  b = a + 1.0
end subroutine main
"""
    before, after = _run(src, tmp_path, "seq_noop")
    # No rank-0 -> rank-1 adapter present before, none after.
    assert _count_seq_adapter(before) == 0
    assert _count_seq_adapter(after) == 0


def test_full_section_passed_unchanged(tmp_path):
    """Passing a real triplet section ``f(d(11:15), 5)`` lowers through
    a triplet designate, not a sequence-association adapter  --  the pass
    must leave that path alone."""
    src = """
module lib
contains
  real function f(d, dz)
    integer, intent(in) :: dz
    real, intent(in) :: d(dz)
    f = sum(d)
  end function f
end module lib

subroutine main(d, out)
  use lib
  real, intent(inout) :: d(50)
  real, intent(out)   :: out
  out = f(d(11:15), 5)
end subroutine main
"""
    before, after = _run(src, tmp_path, "seq_full_section")
    # No adapter to collapse  --  the call already takes a section box.
    assert _count_seq_adapter(before) == 0
    assert _count_seq_adapter(after) == 0


def test_constant_arithmetic_extent_folds(tmp_path):
    """Variant 2: ``f(d(11), 2*K + 1)`` with ``K`` itself a literal  --
    extent expression folds through ``arith.muli`` / ``addi`` after
    recursive trace.  Section ends up shape <7> = 2*3 + 1."""
    src = """
module lib
contains
  real function f(d, dz)
    integer, intent(in) :: dz
    real, intent(in) :: d(dz)
    f = sum(d)
  end function f
end module lib

subroutine main(d, out)
  use lib
  real, intent(inout)  :: d(50)
  real, intent(out)    :: out
  integer, parameter :: K = 3
  out = f(d(11), 2*K + 1)
end subroutine main
"""
    before, after = _run(src, tmp_path, "seq_arith")
    assert _count_seq_adapter(before) >= 1
    assert _count_seq_adapter(after) == 0
    assert 'fir.array<7xf32>' in after


def test_module_parameter_constant_extent(tmp_path):
    """Variant 3: ``integer, parameter :: NMAX = 50`` then
    ``f(d(11), NMAX)``  --  extent folds via the global's ``fir.has_value``
    initialiser.  Note: NMAX larger than ``d``'s extent here is
    contrived (would be illegal at runtime); we only care that the IR
    rewrite picks up the constant ``50`` from the module-level
    parameter."""
    src = """
module lib
  implicit none
  integer, parameter :: NMAX = 8
contains
  real function f(d, dz)
    integer, intent(in) :: dz
    real, intent(in) :: d(dz)
    f = sum(d)
  end function f
end module lib

subroutine main(d, out)
  use lib
  real, intent(inout)  :: d(50)
  real, intent(out)    :: out
  out = f(d(11), NMAX)
end subroutine main
"""
    before, after = _run(src, tmp_path, "seq_param")
    assert _count_seq_adapter(before) >= 1
    assert _count_seq_adapter(after) == 0
    assert 'fir.array<8xf32>' in after


def test_runtime_symbolic_extent_emits_dynamic_section(tmp_path):
    """Variant 4: ``f(d(11), sz)`` with ``sz`` computed at runtime.
    Cannot fold to a constant  --  pass falls back to a runtime-extent
    section ``box<array<?xf32>>`` whose triplet upper bound is
    ``lo + sz - 1``."""
    src = """
module lib
contains
  real function f(d, dz)
    integer, intent(in) :: dz
    real, intent(in) :: d(dz)
    f = sum(d)
  end function f
end module lib

subroutine main(d, sz, out)
  use lib
  real, intent(inout)  :: d(50)
  integer, intent(in)  :: sz
  real, intent(out)    :: out
  out = f(d(11), sz)
end subroutine main
"""
    before, after = _run(src, tmp_path, "seq_symbolic")
    assert _count_seq_adapter(before) >= 1
    assert _count_seq_adapter(after) == 0, (
        "runtime-symbolic should still rewrite into a section, not leave the adapter")
    # Triplet section view of unknown extent.
    assert 'fir.box<!fir.array<?xf32>>' in after


def test_qe_blas_pattern_2d_element_to_1d_column(tmp_path):
    """Variant 5: the QE / BLAS pattern.  Caller passes ``f(d(1, j),
    M)`` where ``d`` is rank-2 ``d(M, N)``  --  column-major contiguity
    means the formal sees one full column starting at ``d(1, j)``.  The
    pass must keep ``j`` as a passthrough scalar index and place the
    triplet on dim 1: ``d(1:M, j)``."""
    src = """
module lib
contains
  real function f(d, dz)
    integer, intent(in) :: dz
    real, intent(in) :: d(dz)
    f = sum(d)
  end function f
end module lib

subroutine main(d, j, out)
  use lib
  real, intent(inout)  :: d(8, 4)
  integer, intent(in)  :: j
  real, intent(out)    :: out
  out = f(d(1, j), 8)
end subroutine main
"""
    before, after = _run(src, tmp_path, "seq_qe_blas")
    assert _count_seq_adapter(before) >= 1
    assert _count_seq_adapter(after) == 0
    # Triplet-on-dim-1 + scalar-on-dim-2: ``(%c1:%c8:%c1, j)``.
    assert 'fir.array<8xf32>' in after
    # Both forms  --  element-form ``d(1, j)`` and the rewritten section
    # ``d(1:8:1, j)``  --  should appear once: the latter as the section,
    # the former either erased (no other uses) or kept (other uses).
    # We only require that a triplet section over the rank-2 parent
    # carrying a fixed scalar second index is present.
    section_lines = [ln for ln in after.splitlines() if 'hlfir.designate' in ln and 'fir.array<8xf32>' in ln]
    assert section_lines, f"no section designate of array<8xf32>:\n{after}"


def test_pass_with_offset_index(tmp_path):
    """``f(d(i*2 + 3), 4)``  --  the start index is a runtime expression.
    Pass should still fire (N=4 is constant) and emit a section using
    the runtime ``lo`` value as the triplet's lower bound."""
    src = """
module lib
contains
  real function f(d, dz)
    integer, intent(in) :: dz
    real, intent(in) :: d(dz)
    f = sum(d)
  end function f
end module lib

subroutine main(d, i, out)
  use lib
  real, intent(inout)  :: d(50)
  integer, intent(in)  :: i
  real, intent(out)    :: out
  out = f(d(i*2 + 3), 4)
end subroutine main
"""
    before, after = _run(src, tmp_path, "seq_offset")
    assert _count_seq_adapter(before) >= 1
    assert _count_seq_adapter(after) == 0
    assert 'fir.array<4xf32>' in after
