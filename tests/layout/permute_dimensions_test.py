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
import copy
from dace.transformation.layout.permute_dimensions import PermuteDimensions

# Symbols
nblks_c = dace.symbol("nblks_c")
nlev = dace.symbol("nlev")
nproma = dace.symbol("nproma")
nblk = 2

N_BLKS, N_LEV, N_PROMA = 2, 4, 16


@dace.program
def indirect_stencil(
    A: dace.float64[nblks_c, nlev, nproma],
    B: dace.float64[nblks_c, nlev, nproma],
    e_bln_c_s: dace.float64[nblks_c, 3, nproma],
    ieidx: dace.int32[3, nblks_c, nproma],
    ieblk: dace.int32[3, nblks_c, nproma],
):
    for jb, jk, jc in dace.map[0:nblks_c, 0:nlev, 1:nproma - 1]:
        B[jb, jk, jc] = (e_bln_c_s[jb, 0, jc] * A[ieblk[0, jb, jc], jk, ieidx[0, jb, jc]] +
                         e_bln_c_s[jb, 1, jc] * A[ieblk[1, jb, jc], jk, ieidx[1, jb, jc]] +
                         e_bln_c_s[jb, 2, jc] * A[ieblk[2, jb, jc], jk, ieidx[2, jb, jc]])


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


def generate_safe_indices(N_BLKS, N_LEV, N_PROMA):
    rng = np.random.default_rng(42)

    # ieblk must be in [0, N_BLKS)
    ieblk = rng.integers(0, N_BLKS, size=(3, N_BLKS, N_PROMA), dtype=np.int32)

    # ieidx must be in [0, N_PROMA)
    # If your stencil is (1 : N_PROMA-1), the neighbors can technically
    # be anywhere in the range [0, N_PROMA).
    ieidx = rng.integers(0, N_PROMA, size=(3, N_BLKS, N_PROMA), dtype=np.int32)

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
                B[jb, jk, jc] = (e_bln_c_s[jb, 0, jc] * A[ieblk[0, jb, jc], jk, ieidx[0, jb, jc]] +
                                 e_bln_c_s[jb, 1, jc] * A[ieblk[1, jb, jc], jk, ieidx[1, jb, jc]] +
                                 e_bln_c_s[jb, 2, jc] * A[ieblk[2, jb, jc], jk, ieidx[2, jb, jc]])
    return B


@pytest.fixture
def input_data():
    A, B = generate_input(N_BLKS, N_LEV, N_PROMA)
    e_bln_c_s = generate_weights(N_BLKS, N_LEV, N_PROMA)
    return A, B, e_bln_c_s


def test_sanity_permutation_indices(input_data):
    A, B, e_bln_c_s = input_data
    ieidx, ieblk = generate_safe_indices(N_BLKS, N_LEV, N_PROMA)
    B_ref = reference(A, e_bln_c_s, ieidx, ieblk)

    B[:] = 0.0
    indirect_stencil(A, B, e_bln_c_s, ieidx, ieblk, nblks_c=N_BLKS, nlev=N_LEV, nproma=N_PROMA)

    np.testing.assert_allclose(B, B_ref, atol=1e-14)


def test_permute_on_safe_indices(input_data):
    A, B, e_bln_c_s = input_data
    ieidx, ieblk = generate_safe_indices(N_BLKS, N_LEV, N_PROMA)
    B_ref = reference(A, e_bln_c_s, ieidx, ieblk)

    B[:] = 0.0
    sdfg = indirect_stencil.to_sdfg()
    sdfg.validate()

    sdfg.save("before_permute.sdfgz", compress=True)

    sdfg2 = copy.deepcopy(sdfg)

    PermuteDimensions(permute_map={
        "A": [0, 2, 1],
        "B": [0, 2, 1],
        "e_bln_c_s": [0, 2, 1],
        "ieidx": [1, 2, 0],
        "ieblk": [1, 2, 0]
    },
                      add_permute_maps=True).apply_pass(sdfg2, {})

    sdfg2.save("after_permute.sdfgz", compress=True)
    sdfg2.validate()

    sdfg2(A, B, e_bln_c_s, ieidx, ieblk, nblks_c=N_BLKS, nlev=N_LEV, nproma=N_PROMA)

    np.testing.assert_allclose(B, B_ref, atol=1e-14)


@dace.program
def interp_ekinh(
    z_kin_hor_e: dace.float64[nblk, nlev, nproma],
    e_bln_c_s: dace.float64[nblk, 3, nproma],
    ieidx: dace.int32[3, nblk, nproma],
    ieblk: dace.int32[3, nblk, nproma],
    z_ekinh: dace.float64[nblk, nlev, nproma],
):
    for jb in range(nblk):
        for jk, jc in dace.map[0:nlev, 0:nproma]:
            z_ekinh[jb, jk, jc] = (e_bln_c_s[jb, 0, jc] * z_kin_hor_e[ieblk[0, jb, jc], jk, ieidx[0, jb, jc]] +
                                   e_bln_c_s[jb, 1, jc] * z_kin_hor_e[ieblk[1, jb, jc], jk, ieidx[1, jb, jc]] +
                                   e_bln_c_s[jb, 2, jc] * z_kin_hor_e[ieblk[2, jb, jc], jk, ieidx[2, jb, jc]])


@pytest.mark.parametrize("seed", [42, 123, 7])
def test_interp_ekinh(seed):
    NPROMA = 16
    NLEV = 10
    rng = np.random.default_rng(seed)
    z_kin_hor_e = rng.random((nblk, NLEV, NPROMA))
    e_bln_c_s = rng.random((nblk, 3, NPROMA))
    ieidx = rng.integers(0, NPROMA, size=(3, nblk, NPROMA), dtype=np.int32)
    ieblk = rng.integers(0, nblk, size=(3, nblk, NPROMA), dtype=np.int32)
    z_ekinh_dace = np.zeros((nblk, NLEV, NPROMA))
    z_ekinh_ref = np.zeros_like(z_ekinh_dace)

    interp_ekinh(
        z_kin_hor_e=z_kin_hor_e,
        e_bln_c_s=e_bln_c_s,
        ieidx=ieidx,
        ieblk=ieblk,
        z_ekinh=z_ekinh_ref,
        nlev=NLEV,
        nproma=NPROMA,
    )

    sdfg = copy.deepcopy(interp_ekinh.to_sdfg())
    sdfg.save("before_permute_interp_ekinh.sdfgz", compress=True)
    PermuteDimensions(permute_map={
        "z_kin_hor_e": [0, 2, 1],
        "e_bln_c_s": [0, 2, 1],
        "ieidx": [1, 2, 0],
        "ieblk": [1, 2, 0],
        "z_ekinh": [0, 2, 1],
    },
                      add_permute_maps=True).apply_pass(sdfg, {})
    sdfg.save("after_permute_interp_ekinh.sdfgz", compress=True)

    sdfg(
        z_kin_hor_e=z_kin_hor_e,
        e_bln_c_s=e_bln_c_s,
        ieidx=ieidx,
        ieblk=ieblk,
        z_ekinh=z_ekinh_dace,
        nlev=NLEV,
        nproma=NPROMA,
    )

    diff = np.max(np.abs(z_ekinh_dace - z_ekinh_ref))
    print(f"[seed={seed}] Max error: {diff}")
    print(f"DaCe:  {z_ekinh_dace.flat[:10]}")
    print(f"Ref:   {z_ekinh_ref.flat[:10]}")
    assert np.allclose(z_ekinh_dace, z_ekinh_ref), f"FAILED (seed={seed})"


if __name__ == "__main__":
    for seed in [42, 123, 7]:
        test_interp_ekinh(seed)

    A, B = generate_input(N_BLKS, N_LEV, N_PROMA)
    e_bln_c_s = generate_weights(N_BLKS, N_LEV, N_PROMA)
    test_permute_on_safe_indices((A, B, e_bln_c_s))
