"""
DaCe indirect stencil matching ICON's z_ekinh interpolation pattern:

  B[jb,jk,jc] = e_bln_c_s[jb,0,jc] * A[ieblk[0,jb,jc], jk, ieidx[0,jb,jc]]
              + e_bln_c_s[jb,1,jc] * A[ieblk[1,jb,jc], jk, ieidx[1,jb,jc]]
              + e_bln_c_s[jb,2,jc] * A[ieblk[2,jb,jc], jk, ieidx[2,jb,jc]]

Array shapes:
  A, B:       dace.float64[nblks_c, nlev, nproma]
  e_bln_c_s:  dace.float64[nblks_c, 3, nproma]
  ieidx:      dace.int32[3, nblks_c, nproma]    (cell index of neighbor)
  ieblk:      dace.int32[3, nblks_c, nproma]    (block index of neighbor)
"""

import dace
import numpy as np
import pytest

# Symbols
nblks_c = dace.symbol("nblks_c")
nlev = dace.symbol("nlev")
nproma = dace.symbol("nproma")


@dace.program
def indirect_stencil(
    A: dace.float64[nblks_c, nlev, nproma],
    B: dace.float64[nblks_c, nlev, nproma],
    e_bln_c_s: dace.float64[nblks_c, 3, nproma],
    ieidx: dace.int32[3, nblks_c, nproma],
    ieblk: dace.int32[3, nblks_c, nproma],
):
    for jb, jk, jc in dace.map[0:nblks_c, 0:nlev, 1:nproma - 1]:
        B[jb, jk, jc] = (
            e_bln_c_s[jb, 0, jc] * A[ieblk[0, jb, jc], jk, ieidx[0, jb, jc]]
            + e_bln_c_s[jb, 1, jc] * A[ieblk[1, jb, jc], jk, ieidx[1, jb, jc]]
            + e_bln_c_s[jb, 2, jc] * A[ieblk[2, jb, jc], jk, ieidx[2, jb, jc]]
        )

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_input(N_blks: int, N_lev: int, N_proma: int, rng=None):
    """Generate A with random values."""
    if rng is None:
        rng = np.random.default_rng(42)
    A = rng.random((N_blks, N_lev, N_proma))
    B = np.zeros_like(A)
    return A, B


def generate_permutation_indices(N_blks: int, N_lev: int, N_proma: int, rng=None):
    """
    Variant 1 — Permutation:
    For each (jb, jc), the 3 neighbors point to randomly permuted block and cell
    indices. This creates scattered, non-local access patterns.

    ieidx[n, jb, jc] ∈ [0, N_proma)   (cell index of neighbor)
    ieblk[n, jb, jc] ∈ [0, N_blks)    (block index of neighbor)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    ieidx = np.empty((3, N_blks, N_proma), dtype=np.int32)
    ieblk = np.empty((3, N_blks, N_proma), dtype=np.int32)

    for jb in range(N_blks):
        for jc in range(N_proma):
            ieblk[:, jb, jc] = rng.choice(N_blks, size=3, replace=True)
            ieidx[:, jb, jc] = rng.choice(np.arange(1, N_proma - 1), size=3, replace=True)

    return ieidx, ieblk


def generate_linear_indices(N_blks: int, N_lev: int, N_proma: int):
    """
    Variant 2 — Linear consecutive:
    Neighbors map to consecutive cells in row-major (block, cell) order.

      flat_base = jb * N_proma + jc
      neighbor n → flat_base + (n + 1)
      ieblk = flat // N_proma   (block)
      ieidx = flat %  N_proma   (cell)
    """
    total_cells = N_blks * N_proma

    ieidx = np.empty((3, N_blks, N_proma), dtype=np.int32)
    ieblk = np.empty((3, N_blks, N_proma), dtype=np.int32)

    for jb in range(N_blks):
        for jc in range(N_proma):
            flat_base = jb * N_proma + jc
            for n in range(3):
                flat_nb = (flat_base + n + 1) % total_cells
                ieblk[n, jb, jc] = flat_nb // N_proma
                ieidx[n, jb, jc] = flat_nb % N_proma

    return ieidx, ieblk


# ---------------------------------------------------------------------------
# Reference & validation
# ---------------------------------------------------------------------------

def generate_weights(N_blks: int, N_lev: int, N_proma: int, rng=None):
    """Generate e_bln_c_s interpolation weights."""
    if rng is None:
        rng = np.random.default_rng(123)
    return rng.random((N_blks, 3, N_proma))


def reference(A, e_bln_c_s, ieidx, ieblk):
    """NumPy reference implementation."""
    N_blks, N_lev, N_proma = A.shape
    B = np.zeros_like(A)
    for jb in range(N_blks):
        for jk in range(N_lev):
            for jc in range(1, N_proma - 1):
                B[jb, jk, jc] = (
                    e_bln_c_s[jb, 0, jc] * A[ieblk[0, jb, jc], jk, ieidx[0, jb, jc]]
                    + e_bln_c_s[jb, 1, jc] * A[ieblk[1, jb, jc], jk, ieidx[1, jb, jc]]
                    + e_bln_c_s[jb, 2, jc] * A[ieblk[2, jb, jc], jk, ieidx[2, jb, jc]]
                )
    return B



N_BLKS, N_LEV, N_PROMA = 2, 4, 16


@pytest.fixture
def input_data():
    A, B = generate_input(N_BLKS, N_LEV, N_PROMA)
    e_bln_c_s = generate_weights(N_BLKS, N_LEV, N_PROMA)
    return A, B, e_bln_c_s


def test_sanity_permutation_indices(input_data):
    A, B, e_bln_c_s = input_data
    ieidx, ieblk = generate_permutation_indices(N_BLKS, N_LEV, N_PROMA)
    B_ref = reference(A, e_bln_c_s, ieidx, ieblk)

    B[:] = 0.0
    indirect_stencil(A, B, e_bln_c_s, ieidx, ieblk,
                     nblks_c=N_BLKS, nlev=N_LEV, nproma=N_PROMA)

    np.testing.assert_allclose(B, B_ref, atol=1e-14)


def test_sanity_linear_indices(input_data):
    A, B, e_bln_c_s = input_data
    ieidx, ieblk = generate_linear_indices(N_BLKS, N_LEV, N_PROMA)
    B_ref = reference(A, e_bln_c_s, ieidx, ieblk)

    B[:] = 0.0
    indirect_stencil(A, B, e_bln_c_s, ieidx, ieblk,
                     nblks_c=N_BLKS, nlev=N_LEV, nproma=N_PROMA)

    np.testing.assert_allclose(B, B_ref, atol=1e-14)

def test_permute_on_linear_indices(input_data):
    A, B, e_bln_c_s = input_data
    ieidx, ieblk = generate_linear_indices(N_BLKS, N_LEV, N_PROMA)
    B_ref = reference(A, e_bln_c_s, ieidx, ieblk)

    B[:] = 0.0
    sdfg = indirect_stencil.to_sdfg()
    sdfg.save("before_permute.sdfgz", compress=True)
