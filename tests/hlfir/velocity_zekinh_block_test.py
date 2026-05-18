"""End-to-end test for the ``z_ekinh`` reconstruction block.

Bisecting ``velocity_full`` localised a runtime SDFG segfault to lines
510-530 of ``velocity_full.f90`` -- the DO jb / DO jk / DO jc nest
that builds ``z_ekinh`` via the indirect ``p_patch%cells%edge_idx``
/ ``edge_blk`` tables.  This test runs the same loop nest in
isolation; if it passes here, the original segfault is interaction-
based (state shared with earlier blocks), not in this block itself.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC_PATH = Path(__file__).resolve().parent / "velocity_zekinh_block.f90"
_ENTRY = "_QMmo_velocity_zekinhPzekinh_block"


def _numpy_reference(edge_idx, edge_blk, e_bln_c_s, z_kin_hor_e, z_ekinh, nproma, nlev, nblks_c, i_startblk, i_endblk):
    """Pure-numpy z_ekinh reconstruction matching the Fortran kernel.

    :param edge_idx: shape ``(nproma, nblks_c, 3)``.
    :param edge_blk: same shape; 1-based block index per neighbour.
    :param e_bln_c_s: shape ``(nproma, 3, nblks_c)``; coefficient weights.
    :param z_kin_hor_e: edge-side gather source, shape ``(nproma, nlev, nblks_e)``.
    :param z_ekinh: cell-side output, shape ``(nproma, nlev, nblks_c)``.
    :param nproma: column count.
    :param nlev: level count.
    :param nblks_c: cell-block count (unused but kept for symmetry).
    :param i_startblk: 1-based start block from ``start_block(4)``.
    :param i_endblk: 1-based end block from ``end_block(-5)``.
    """
    for jb in range(i_startblk - 1, i_endblk):
        for jk in range(nlev):
            for jc in range(nproma):
                acc = 0.0
                for k in range(3):
                    ie = edge_idx[jc, jb, k] - 1
                    ib = edge_blk[jc, jb, k] - 1
                    acc += e_bln_c_s[jc, k, jb] * z_kin_hor_e[ie, jk, ib]
                z_ekinh[jc, jk, jb] = acc


def test_velocity_zekinh_block_builds_and_calls(tmp_path: Path):
    """Build + e2e call.  Verifies the isolated block doesn't segfault
    and matches the numpy reference to ``rtol=1e-13``.
    """
    src = _SRC_PATH.read_text()
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="zekinh_block", entry=_ENTRY).build()
    sdfg.validate()

    nproma, nlev, nblks_c, nblks_e = 8, 6, 4, 4

    rng = np.random.default_rng(0)

    def rr(*shape):
        """Random float64 buffer in Fortran order."""
        return np.asfortranarray(rng.standard_normal(shape))

    def ii(hi, *shape):
        """Random in-bounds integer table ``[1, hi]`` in Fortran order."""
        return np.asfortranarray(rng.integers(1, hi + 1, size=shape, dtype=np.int32))

    edge_idx = ii(nproma, nproma, nblks_c, 3)
    edge_blk = ii(nblks_e, nproma, nblks_c, 3)
    # 33-element bounds buffer to cover ICON's [-16, 16] refined-cell-tag range.
    # start_block(4) -> first valid block; end_block(-5) -> last cell block.
    start_block = np.zeros(33, dtype=np.int32, order='F')
    end_block = np.zeros(33, dtype=np.int32, order='F')
    # offset_d0 specialises to 1 for start_block (only positive literal seen)
    # and -5 for end_block (negative-literal inference fires).  So
    # buf[3]=start_block(4) and buf[0]=end_block(-5).
    start_block[3] = 1
    end_block[0] = nblks_c
    e_bln_c_s = rr(nproma, 3, nblks_c)
    z_kin_hor_e = rr(nproma, nlev, nblks_e)
    z_ekinh = np.zeros((nproma, nlev, nblks_c), dtype=np.float64, order='F')
    z_ekinh_ref = z_ekinh.copy(order='F')

    sdfg(
        p_patch_cells_edge_idx=edge_idx,
        p_patch_cells_edge_blk=edge_blk,
        p_patch_cells_start_block=start_block,
        p_patch_cells_end_block=end_block,
        p_int_e_bln_c_s=e_bln_c_s,
        z_kin_hor_e=z_kin_hor_e,
        z_ekinh=z_ekinh,
        nproma=np.int32(nproma),
        p_patch_nblks_c=np.asfortranarray([nblks_c], dtype=np.int32),
        p_patch_nlev=np.int32(nlev),
        # Extent + offset symbols.
        p_int_e_bln_c_s_d0=np.int64(nproma),
        p_int_e_bln_c_s_d1=np.int64(3),
        p_patch_cells_edge_idx_d0=np.int64(nproma),
        p_patch_cells_edge_idx_d1=np.int64(nblks_c),
        p_patch_cells_edge_blk_d0=np.int64(nproma),
        p_patch_cells_edge_blk_d1=np.int64(nblks_c),
        z_kin_hor_e_d0=np.int64(nproma),
        z_kin_hor_e_d1=np.int64(nlev),
        z_ekinh_d0=np.int64(nproma),
        z_ekinh_d1=np.int64(nlev),
        offset_p_int_e_bln_c_s_d0=np.int64(1),
        offset_p_int_e_bln_c_s_d2=np.int64(1),
        offset_p_patch_cells_edge_idx_d0=np.int64(1),
        offset_p_patch_cells_edge_idx_d1=np.int64(1),
        offset_p_patch_cells_edge_blk_d0=np.int64(1),
        offset_p_patch_cells_edge_blk_d1=np.int64(1),
    )

    _numpy_reference(edge_idx,
                     edge_blk,
                     e_bln_c_s,
                     z_kin_hor_e,
                     z_ekinh_ref,
                     nproma,
                     nlev,
                     nblks_c,
                     i_startblk=1,
                     i_endblk=nblks_c)
    # 3-term FMA-reorderable sum -- bridge and numpy may pick different
    # accumulation orders.  Hold a tight rel/abs but not bit-exact.
    np.testing.assert_allclose(z_ekinh, z_ekinh_ref, rtol=1e-13, atol=1e-13)
