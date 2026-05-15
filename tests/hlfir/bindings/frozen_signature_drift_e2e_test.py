"""Frozen-signature drift, exercised through the *real* generation path.

``frozen_signature_test.py`` checks ``FrozenSignature.verify_against``
against hand-built synthetic SDFGs.  This module closes the gap the
contract actually guards: a signature snapshotted by
``SDFGBuilder.build()`` (``sdfg._frozen_signature``), then drifted by a
post-build SDFG mutation, must be rejected at codegen -- not silently
emit a C header that disagrees with the already-generated Fortran
binding.

The positive control compiles the untouched SDFG (snapshot matches
live arglist -> no raise); the negative cases mutate the live SDFG
*after* the snapshot was frozen and assert ``generate_code`` raises
``SignatureDriftError``.
"""

from pathlib import Path

import dace
import pytest

from _util import build_sdfg, have_flang
from dace.codegen.codegen import generate_code
from dace.frontend.hlfir.bindings import SignatureDriftError

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
subroutine axpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: a
  real(8), intent(in) :: x(n)
  real(8), intent(inout) :: y(n)
  integer :: i
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine axpy
"""


def _build(tmp_path: Path):
    """Build the ``axpy`` SDFG through the bridge.

    :param tmp_path: pytest scratch dir.
    :returns: built SDFG with ``_frozen_signature`` auto-attached.
    """
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, sdfg_dir, name="axpy", entry="_QPaxpy").build()
    sdfg.validate()
    assert getattr(sdfg, "_frozen_signature",
                   None) is not None, ("SDFGBuilder.build() must auto-attach a frozen signature")
    return sdfg


def test_untouched_sdfg_codegens(tmp_path: Path):
    """Positive control: a build that nobody mutated still matches its
    own snapshot, so codegen does not raise."""
    sdfg = _build(tmp_path)
    generate_code(sdfg)  # no SignatureDriftError


def test_added_arg_after_freeze_raises(tmp_path: Path):
    """A transformation that adds a non-transient array after the
    signature was frozen drifts ``sdfg.arglist()``; codegen must
    reject it instead of emitting a header the binding can't call."""
    sdfg = _build(tmp_path)
    # New caller-visible array -> appears in arglist(), absent from snap.
    sdfg.add_array("z_drift", shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)
    with pytest.raises(SignatureDriftError, match="signature drift"):
        generate_code(sdfg)


def test_dtype_change_after_freeze_raises(tmp_path: Path):
    """Silently retyping an existing arg (float64 -> float32) is the
    most dangerous drift -- order/count look identical.  The per-arg
    dtype guard in ``verify_against`` must catch it."""
    sdfg = _build(tmp_path)
    sdfg.arrays["y"].dtype = dace.float32
    with pytest.raises(SignatureDriftError, match="dtype"):
        generate_code(sdfg)


def test_extra_free_symbol_after_freeze_raises(tmp_path: Path):
    """A pass that introduces a new *used* free symbol changes the
    SDFG's callable surface; the free-symbol set guard must fire."""
    sdfg = _build(tmp_path)
    sdfg.add_symbol("drift_sym", dace.int64)
    # Append a state onto the existing sink so the graph stays
    # connected (codegen validates before the drift check); the new
    # edge's condition makes ``drift_sym`` a *used* free symbol.
    sink = sdfg.sink_nodes()[0]
    tail = sdfg.add_state("drift_tail")
    sdfg.add_edge(sink, tail, dace.InterstateEdge(condition="drift_sym > 0"))
    with pytest.raises(SignatureDriftError):
        generate_code(sdfg)
