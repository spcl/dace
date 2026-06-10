# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""2D vectorization on Fortran-layout arrays.

The vectorizer always strides the *innermost* map dimension — and that
dimension does NOT have to be the array's last subscript. What matters
is that the innermost map indexes the array's unit-stride dimension:

- C layout      → unit stride is the LAST subscript  → innermost map
  indexes ``A[i, j]``'s ``j``.
- Fortran layout → unit stride is the FIRST subscript → innermost map
  indexes ``A[i, j]``'s ``i``.

This pins the Fortran case: arrays carry ``strides=(1, M)`` and the 2D
map nests ``j`` (outer) over ``i`` (innermost), so the vectorized dim
``i`` lands on the unit-stride leftmost subscript. The 1D ``VectorizeCPU``
path handles this directly (a contiguous, non-last-dim vectorization),
validated here against the unvectorized reference across the
branch/remainder/emission matrix.

The K-dim tile-op path (``vectorize_config="tile_nodes"``) is NOT
exercised here yet: its ``pure`` expansion lays the register tile out
C-row-major and maps tile dims to map params positionally, so a
Fortran-strided array is read at the wrong offsets. Carrying per-dim
strides into the tile load/store expansion is a tracked follow-up; this
test is its future regression target.

TODO: add a Fortran-packed indirection (gather/scatter) test built on the
indirect-access subgraph pattern in ``indirect/test_strided_gather_scatter``
(a bare ``Memlet("B[idx[i], j]")`` string is not valid — the gather index
must flow through its own connector / dynamic subset).
"""

import pytest
import numpy as np

import dace

from tests.passes.vectorization.helpers.harness import run_vectorization_test

# Fortran-layout (non-last-dim contiguous) — also exercise the tile-op config.
pytestmark = pytest.mark.tile_nodes


def _build_fortran_2d_axpy() -> dace.SDFG:
    """``C[i, j] = A[i, j] + B[i, j]`` on Fortran-layout (M, N) arrays.

    Strides ``(1, M)`` make the leftmost subscript ``i`` unit-stride; the
    2D map nests ``j`` outer / ``i`` innermost so the vectorized dim is
    contiguous.
    """
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg = dace.SDFG("fortran_2d_axpy")
    for nm in ("A", "B", "C"):
        sdfg.add_array(nm, (M, N), dace.float64, strides=(1, M))
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "axpy",
        {
            "j": "0:N",
            "i": "0:M"
        },
        {
            "_a": dace.Memlet("A[i, j]"),
            "_b": dace.Memlet("B[i, j]")
        },
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i, j]")},
        external_edges=True,
    )
    return sdfg


def test_fortran_2d_axpy(branch_mode, remainder_strategy, emission_style):
    """Contiguous 2D store on Fortran-layout arrays (innermost map indexes
    the unit-stride leftmost subscript) vectorizes against the
    unvectorized reference across the branch/remainder/emission matrix."""
    M_val, N_val = 16, 9  # M divisible by W=8; N (outer) carries a remainder
    rng = np.random.default_rng(seed=7)
    A = rng.random((M_val, N_val)).copy(order="F")
    B = rng.random((M_val, N_val)).copy(order="F")
    C = np.zeros((M_val, N_val), order="F")
    run_vectorization_test(
        dace_func=_build_fortran_2d_axpy(),
        arrays={
            "A": A,
            "B": B,
            "C": C
        },
        params={
            "M": M_val,
            "N": N_val
        },
        sdfg_name="fortran_2d_axpy",
        from_sdfg=True,
        branch_mode=branch_mode,
        remainder_strategy=remainder_strategy,
        emission_style=emission_style,
    )
