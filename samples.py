import numpy
import pytest
import dace
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

KLEV = dace.symbol("KLEV")
KLON = dace.symbol("KLON")
NCLV = dace.symbol("NCLV")
NB = dace.symbol("NB")
NLEV = dace.symbol("NLEV")
NPROMA = dace.symbol("NPROMA")

_PTSPHY = 50.0
_RLMIN = 1.0e-8
_RAMIN = 1.0e-8
_RALVDCP = 2.5008e6 / 1004.7
_RALSDCP = 2.8345e6 / 1004.7
_ZQTMST = 1.0 / _PTSPHY

@dace.program
def cloudsc_tidy_branch(zqx_l: dace.float64[KLEV, KLON], zqx_i: dace.float64[KLEV, KLON],
                        zqx_v: dace.float64[KLEV, KLON], za: dace.float64[KLEV, KLON],
                        ptend_q: dace.float64[KLEV, KLON], ptend_t: dace.float64[KLEV, KLON]):
    # cloudsc_bottom_lower.F90: "Tidy up very small cloud cover or total
    # cloud water" — guarded read-modify-write over several arrays (the
    # CLOUDSC-characteristic conditional accumulation pattern).
    for jk in range(KLEV):
        for jl in range(KLON):
            if zqx_l[jk, jl] + zqx_i[jk, jl] < _RLMIN or za[jk, jl] < _RAMIN:
                zqadj_l = zqx_l[jk, jl] * _ZQTMST
                ptend_q[jk, jl] = ptend_q[jk, jl] + zqadj_l
                ptend_t[jk, jl] = ptend_t[jk, jl] - _RALVDCP * zqadj_l
                zqx_v[jk, jl] = zqx_v[jk, jl] + zqx_l[jk, jl]
                zqx_l[jk, jl] = 0.0
                zqadj_i = zqx_i[jk, jl] * _ZQTMST
                ptend_q[jk, jl] = ptend_q[jk, jl] + zqadj_i
                ptend_t[jk, jl] = ptend_t[jk, jl] - _RALSDCP * zqadj_i
                zqx_v[jk, jl] = zqx_v[jk, jl] + zqx_i[jk, jl]
                zqx_i[jk, jl] = 0.0
                za[jk, jl] = 0.0


@dace.program
def icon_zekinh_gather(e_bln: dace.float64[NB, 3, NPROMA],
                       edge_idx: dace.int32[NB, NPROMA, 3],
                       edge_blk: dace.int32[NB, NPROMA, 3],
                       z_kin_hor_e: dace.float64[NB, NLEV, NPROMA],
                       z_ekinh: dace.float64[NB, NLEV, NPROMA]):
    # ICON velocity_zekinh_block.f90: bilinear cell-from-edges interpolation
    # — 3-edge data-dependent gather (z_kin_hor_e[edge_blk[..], jk, edge_idx[..]])
    # weighted by per-cell coefficients e_bln. Canonical loop-nest exercising
    # the K-dim tile descent's structured gather path.
    for jb in range(NB):
        for jk in range(NLEV):
            for jc in range(NPROMA):
                z_ekinh[jb, jk, jc] = (
                    e_bln[jb, 0, jc] * z_kin_hor_e[edge_blk[jb, jc, 0], jk, edge_idx[jb, jc, 0]] +
                    e_bln[jb, 1, jc] * z_kin_hor_e[edge_blk[jb, jc, 1], jk, edge_idx[jb, jc, 1]] +
                    e_bln[jb, 2, jc] * z_kin_hor_e[edge_blk[jb, jc, 2], jk, edge_idx[jb, jc, 2]])


if __name__ == "__main__":
    sdfg = cloudsc_tidy_branch.to_sdfg()
    sdfg.save("scalar_cloudsc_tidy_branch.sdfg")
    VectorizeCPUMultiDim(widths=(8, 8),
                        target_isa="SCALAR",
                        remainder_strategy="scalar_postamble",
                        branch_mode="merge",
                        loop_to_map_permissive=False,
                        nest_map_bodies=True,
                        insert_copies=True,
                        fuse_overlapping_loads=False,
                        scalar_remainder_emit="tile_k1",
                        expand_tile_nodes=False).apply_pass(sdfg, {})
    sdfg.save("vectorized_cloudsc_tidy_branch.sdfg")

    sdfg = icon_zekinh_gather.to_sdfg()
    sdfg.save("scalar_icon_zekinh_gather.sdfg")
    VectorizeCPUMultiDim(widths=(8, 8),
                        target_isa="SCALAR",
                        remainder_strategy="scalar_postamble",
                        branch_mode="merge",
                        loop_to_map_permissive=False,
                        nest_map_bodies=True,
                        insert_copies=True,
                        fuse_overlapping_loads=False,
                        scalar_remainder_emit="tile_k1",
                        expand_tile_nodes=False).apply_pass(sdfg, {})
    sdfg.save("vectorized_icon_zekinh_gather.sdfg")