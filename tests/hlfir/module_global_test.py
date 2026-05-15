"""Module-level (host-associated) global array passed through to a
kernel via ``USE mo_mod, ONLY: gtable``.

The Fortran source declares ``INTEGER :: gtable(8)`` at module scope
with no initialiser.  A subroutine in a separate module imports
``gtable`` via USE and reads from it inside a loop.  This is the
same shape as ICON's ``mo_vertical_grid::nrdmax``, ``mo_init_vgrid::
nflatlev``, etc.: module-scope arrays the caller is expected to fill
once at init time and the kernel reads.

Bridge contract: an uninitialised module-scope global is an
**external input** to the kernel.  ``extract_vars`` traces the
declare's memref back through ``fir.address_of`` to the
``fir.global`` it references; if the global carries no
``fir.has_value`` body, the variable lands in the SDFG signature
as a non-transient array (``intent='inout'``) the caller must
supply.  Parameter-attributed globals and initialised module data
(``fir.global`` with a dense init) stay transient and get baked
into the constant pool.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
MODULE mo_module_data
  IMPLICIT NONE
  INTEGER :: gtable(8)
END MODULE mo_module_data

MODULE mo_kernel
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE read_table(out, idx, n)
    USE mo_module_data, ONLY: gtable
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n
    INTEGER, INTENT(IN) :: idx(n)
    INTEGER, INTENT(OUT) :: out(n)
    INTEGER :: i
    DO i = 1, n
      out(i) = gtable(idx(i))
    END DO
  END SUBROUTINE read_table
END MODULE mo_kernel
"""


def test_module_global_array_passthrough(tmp_path: Path):
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, sdfg_dir, name="read_table", entry="_QMmo_kernelPread_table").build()
    sdfg.validate()

    # Without the fix, ``gtable`` is a transient -- arglist won't carry it.
    # With the fix, it lands as a non-transient input kwarg.
    assert 'gtable' in sdfg.arglist(), (f"expected `gtable` in SDFG arglist (module-level global must surface "
                                        f"as a non-transient kwarg); arglist keys: {sorted(sdfg.arglist().keys())}")

    ref = f2py_compile(_SRC, tmp_path / "ref", "module_global_ref")

    rng = np.random.default_rng(0)
    n = 16
    gtable_vals = np.asfortranarray(rng.integers(0, 1000, size=8, dtype=np.int32))
    idx = np.asfortranarray(rng.integers(1, 9, size=n, dtype=np.int32))

    out_sdfg = np.zeros(n, dtype=np.int32, order='F')

    # f2py exposes module data on the module object; assign in place.
    # ``out`` is INTENT(OUT) so f2py treats it as a return value.
    ref.mo_module_data.gtable[:] = gtable_vals
    out_ref = ref.mo_kernel.read_table(idx)

    sdfg(gtable=gtable_vals, idx=idx, out=out_sdfg, n=n)

    np.testing.assert_array_equal(out_sdfg, out_ref)
