# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileStore`` broadcast source kinds.

The base ``TileStore`` reads a tile-shaped transient (``src_kind="Tile"``,
default) and streams it to the destination. Two new source kinds let the
same lib node express constant / scalar broadcasts without a CPP fill
tasklet:

- ``src_kind="Symbol"`` — broadcast a symbolic expression / numeric
  literal (``src_expr``) to every lane. The ``_src`` connector is
  omitted.
- ``src_kind="Scalar"`` — broadcast a length-1 array value read via
  ``_src``.

The descent's ``PromoteNSDFGBodyToTiles._promote_const_stores`` uses
the Symbol kind to lower a body-level ``zqx_l[jk, jl] = 0.0`` constant
store to a single tile op (no nested-loop CPP fill).

The 2-D ``cloudsc_tidy_branch`` kernel exercises this end-to-end: the
guarded body sets six arrays to ``0.0`` inside the ``if`` arm. After
``VectorizeCPUMultiDim`` runs (``expand_tile_nodes=False`` so the lib
nodes survive), the SDFG must hold zero ``Tasklet`` nodes — every
constant store is now a ``TileStore`` lib node.
"""
import dace
import pytest

from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (
    VectorizeCPUMultiDim, )

KLEV = dace.symbol("KLEV")
KLON = dace.symbol("KLON")

_PTSPHY = 50.0
_RLMIN = 1.0e-8
_RAMIN = 1.0e-8
_RALVDCP = 2.5008e6 / 1004.7
_RALSDCP = 2.8345e6 / 1004.7
_ZQTMST = 1.0 / _PTSPHY


@dace.program
def _tidy_branch(
    zqx_l: dace.float64[KLEV, KLON],
    zqx_i: dace.float64[KLEV, KLON],
    zqx_v: dace.float64[KLEV, KLON],
    za: dace.float64[KLEV, KLON],
    ptend_q: dace.float64[KLEV, KLON],
    ptend_t: dace.float64[KLEV, KLON],
):
    # cloudsc_bottom_lower.F90 "Tidy up very small cloud cover or total
    # cloud water" — six guarded zero-memsets inside the same ``if``
    # arm, which is the canonical TileStore broadcast pattern.
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


def _tile_lib_node_count(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if "tileops" in type(n).__module__)


def _tasklet_count(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))


def test_tilestore_symbol_broadcast_minimal():
    """Constructing a ``TileStore(src_kind='Symbol')`` declares no
    ``_src`` connector and embeds the literal in ``src_expr``."""
    from dace.libraries.tileops import TileStore
    node = TileStore("ts_sym", widths=(8, 8), src_kind="Symbol", src_expr="0.0")
    assert "_src" not in node.in_connectors, "Symbol-source TileStore must not declare ``_src``"
    assert "_dst" in node.out_connectors
    assert node.src_expr == "0.0"


def test_tilestore_symbol_requires_expr():
    """``src_kind='Symbol'`` without a ``src_expr`` raises at
    construction (loud failure)."""
    from dace.libraries.tileops import TileStore
    with pytest.raises(ValueError, match="src_expr"):
        TileStore("ts_sym_bad", widths=(8, ), src_kind="Symbol")


def test_tidy_branch_emits_zero_cpp_tasklets():
    """End-to-end: after ``VectorizeCPUMultiDim`` runs on the
    ``cloudsc_tidy_branch`` kernel (with ``expand_tile_nodes=False`` so
    the lib nodes survive), the SDFG must hold zero ``Tasklet`` nodes —
    every constant store is now a ``TileStore`` lib node, no CPP fills."""
    sdfg = _tidy_branch.to_sdfg()
    sdfg.name = "tidy_branch_tile_only"
    sdfg.validate()

    VectorizeCPUMultiDim(
        widths=(8, 8),
        target_isa="SCALAR",
        remainder_strategy="scalar_postamble",
        branch_mode="merge",
        loop_to_map_permissive=False,
        nest_map_bodies=True,
        insert_copies=True,
        fuse_overlapping_loads=False,
        scalar_remainder_emit="tile_k1",
        expand_tile_nodes=False,
    ).apply_pass(sdfg, {})
    sdfg.validate()

    n_tasklets = _tasklet_count(sdfg)
    n_tile = _tile_lib_node_count(sdfg)
    assert n_tasklets == 0, (f"tidy_branch must emit zero CPP tasklets after the descent — every store "
                             f"should be a tile lib node; got {n_tasklets} Tasklet nodes (and {n_tile} "
                             f"tile lib nodes).")
    assert n_tile > 0, f"tile lib nodes must be present pre-expansion; got {n_tile}"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
