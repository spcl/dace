"""Iterator value after a Fortran DO loop must match gfortran's convention.

After a counted `DO i = lo, hi[, step]` loop completes normally (no EXIT),
gfortran leaves ``i`` at the value that failed the bound check —
**one stride past the last attained value**:

* forward  ``DO i = 1, N``    →  exit value ``N + 1``
* reverse  ``DO i = N, 1, -1`` →  exit value ``0``
* strided  ``DO i = 1, N, 2`` →  exit value ``(N | N+1) + 2`` depending
  on parity, again last attained + step

The Fortran 2018+ standard says the iterator is technically undefined
after the loop, but every mainstream compiler (gfortran, ifort, flang)
leaves it at ``last + step``.  The bridge's SSA loop-iterator
reconstruction pass therefore has to match this convention — any kernel
that reads the iterator past the loop end would otherwise diverge from
its f2py reference (no current cloudsc loop nest does this, but it is a
real class of subtle off-by-one bugs and worth pinning).

E2e against an f2py-compiled reference of the same Fortran source.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang
from _helpers import f2py

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _run(tmp_path, name: str, src: str, *, expected: int, n: int):
    """Build SDFG, build f2py reference, call both, assert they agree on
    the captured exit-value and on the int returned by ``get_iter_after``."""
    ref = f2py(src, tmp_path / 'ref', f'{name}_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name=name, entry='_QMkernel_modPdriver').build()

    # f2py: ``out_iter`` is intent(out) integer scalar -- returned as Python int.
    iter_ref = ref.kernel_mod.driver(n=n)

    iter_sd = np.zeros(1, dtype=np.int32)
    sdfg(out_iter=iter_sd, n=n)
    assert int(iter_ref) == expected, f"f2py disagrees with expected {expected}: got {iter_ref}"
    assert int(iter_sd[0]) == expected, f"SDFG disagrees with expected {expected}: got {iter_sd[0]}"


def test_fortran_frontend_loop_iterator_exit_forward(tmp_path):
    """Forward loop ``DO i = 1, N``: post-loop ``i = N + 1``."""
    src = """
MODULE kernel_mod
CONTAINS
SUBROUTINE driver(out_iter, n)
integer, intent(in) :: n
integer, intent(out) :: out_iter
integer :: i, sink
sink = 0
DO i = 1, n
  sink = sink + i
ENDDO
out_iter = i
END SUBROUTINE driver
END MODULE kernel_mod
"""
    _run(tmp_path, name='loop_exit_fwd', src=src, expected=11, n=10)


def test_fortran_frontend_loop_iterator_exit_reverse(tmp_path):
    """Reverse loop ``DO i = N, 1, -1``: post-loop ``i = 0``."""
    src = """
MODULE kernel_mod
CONTAINS
SUBROUTINE driver(out_iter, n)
integer, intent(in) :: n
integer, intent(out) :: out_iter
integer :: i, sink
sink = 0
DO i = n, 1, -1
  sink = sink + i
ENDDO
out_iter = i
END SUBROUTINE driver
END MODULE kernel_mod
"""
    _run(tmp_path, name='loop_exit_rev', src=src, expected=0, n=10)


def test_fortran_frontend_loop_iterator_exit_strided(tmp_path):
    """Strided forward loop ``DO i = 1, N, 2``: post-loop ``i = last + 2``.

    For N=10, iterations cover i = 1, 3, 5, 7, 9; post-loop ``i = 11``.
    """
    src = """
MODULE kernel_mod
CONTAINS
SUBROUTINE driver(out_iter, n)
integer, intent(in) :: n
integer, intent(out) :: out_iter
integer :: i, sink
sink = 0
DO i = 1, n, 2
  sink = sink + i
ENDDO
out_iter = i
END SUBROUTINE driver
END MODULE kernel_mod
"""
    _run(tmp_path, name='loop_exit_strided', src=src, expected=11, n=10)
