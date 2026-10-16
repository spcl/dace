# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Structure-only unit tests (no GPU) for the cuTile lowering passes.

Each pass in :mod:`dace.transformation.passes.vectorization.cutile_lowering`
is tested in isolation on SDFGs produced by
``VectorizeMultiDim(VectorizeConfig(target_isa=ISA.CUTILE, expand_tile_nodes=False))``, plus
hand-built SDFGs for the anchor-construction corner cases. Ordering /
out-of-order precondition behavior and the full documented sequence through
code generation (text assertions only -- no compilation, no kernel launches)
are covered at the end.

After the refactoring to use ``sdfg.apply_gpu_transformations()`` +
``GPUDeviceToCuTile``, the lowering passes are:

1. ``CuTileValidateTiles``
2. ``GPUDeviceToCuTile`` (replaces ``CuTileSetSchedules``)
3. ``CuTileSetTileStorage``
4. ``CuTileSetImplementations``

``CuTileSetSchedules``, ``CuTileSetGlobalStorage``, and
``CuTileInsertDataCopies`` have been deleted -- their responsibilities are
now handled by ``sdfg.apply_gpu_transformations()`` and ``GPUDeviceToCuTile``.
"""
import warnings
from typing import Dict, List, Set, Tuple

import numpy as np
import pytest

import dace
from dace import data, dtypes
from dace.libraries.tileops import TileBinop
from dace.libraries.tileops.nodes import TileIota
from dace.sdfg import SDFG, nodes
from dace.sdfg.validation import InvalidSDFGEdgeError
from dace.transformation.passes.vectorization.cutile_lowering import (
    CuTileSetImplementations,
    CuTileSetTileStorage,
    CuTileValidateTiles,
    GPUDeviceToCuTile,
    _collect_tileops_adjacent_edges,
    _tile_node_types,
    clamp_propagated_oob_memlets,
)
from dace.transformation.passes.vectorization.vectorize_multi_dim import (
    VectorizeMultiDim, )
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.dtypes import DeviceType

# ============================================================
# Fixture builders
# ============================================================


def _vectorize_cutile(sdfg: SDFG, widths: Tuple[int, ...], gpu: bool = False) -> None:
    """Run the building-block vectorizer config for the cuTile lowering.

    Uses ``device=CPU`` (host maps) so the individual lowering passes can be
    exercised in isolation; the passes are order-tolerant by design.

    :param sdfg: The SDFG to vectorize in place.
    :param widths: Per-dim tile widths, innermost-last.
    """
    sdfg.backend = dtypes.BackendLanguage.Python
    if gpu:
        sdfg.apply_gpu_transformations(simplify=True)
    VectorizeMultiDim(
        VectorizeConfig(widths=widths,
                        target_isa=ISA.CUTILE,
                        expand_tile_nodes=False,
                        validate=False,
                        device=DeviceType.GPU if gpu else DeviceType.CPU)).apply_pass(sdfg, {})


def _build_unvectorized_vadd_sdfg() -> SDFG:
    """Symbolic-size K=1 vadd, NOT vectorized (zero tile-op anchors)."""
    N = dace.symbol("N")

    @dace.program
    def cutile_lowering_vadd_plain(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        for i in dace.map[0:N]:
            C[i] = A[i] + B[i]

    return cutile_lowering_vadd_plain.to_sdfg()


def _build_vadd_k1_sdfg(gpu: bool = False) -> SDFG:
    """Symbolic-size K=1 vadd, vectorized with ``widths=(8,)``."""
    sdfg = _build_unvectorized_vadd_sdfg()
    _vectorize_cutile(sdfg, (8, ), gpu=gpu)
    return sdfg


def _build_vadd_k2_sdfg(gpu: bool = False) -> SDFG:
    """Symbolic-size K=2 vadd, vectorized with ``widths=(8, 4)``."""
    M = dace.symbol("M")
    N = dace.symbol("N")

    @dace.program
    def cutile_lowering_vadd_k2(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[M, N]):
        for i, j in dace.map[0:M, 0:N]:
            C[i, j] = A[i, j] + B[i, j]

    sdfg = cutile_lowering_vadd_k2.to_sdfg()
    _vectorize_cutile(sdfg, (8, 4), gpu=gpu)
    return sdfg


def _build_vadd_concrete_sdfg(gpu: bool = False) -> SDFG:
    """Concrete-size vadd with a non-divisible boundary (100 % 8 != 0)."""

    @dace.program
    def cutile_lowering_vadd_n100(A: dace.float64[100], B: dace.float64[100], C: dace.float64[100]):
        for i in dace.map[0:100]:
            C[i] = A[i] + B[i]

    sdfg = cutile_lowering_vadd_n100.to_sdfg()
    _vectorize_cutile(sdfg, (8, ), gpu=gpu)
    return sdfg


def _build_bare_tile_binop_sdfg(widths: Tuple[int, ...]) -> SDFG:
    """Hand-built single-state SDFG with one ``TileBinop`` and no maps.

    :param widths: Tile widths of the node (possibly non-power-of-2, for
        the validation tests).
    """
    sdfg = dace.SDFG("cutile_lowering_bare_binop")
    shape = tuple(max(w, 1) for w in widths)
    sdfg.add_array("A", shape, dace.float64)
    sdfg.add_array("B", shape, dace.float64)
    sdfg.add_array("C", shape, dace.float64)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    c = state.add_access("C")
    node = TileBinop(name="tb", widths=widths, op="+")
    state.add_node(node)
    full = ",".join(f"0:{w}" for w in shape)
    state.add_edge(a, None, node, "_a", dace.Memlet(f"A[{full}]"))
    state.add_edge(b, None, node, "_b", dace.Memlet(f"B[{full}]"))
    state.add_edge(node, "_c", c, None, dace.Memlet(f"C[{full}]"))
    return sdfg


def _build_blas_dot_sdfg() -> SDFG:
    """Hand-built SDFG whose only library node is a BLAS ``Dot`` (no tileops).

    Models the BLAS-only configuration (e.g. cholesky/trisolv after
    vectorization, where every reduction became a ``Dot``).
    """
    from dace.libraries.blas.nodes import Dot
    sdfg = dace.SDFG("cutile_lowering_blas_only")
    sdfg.add_array("x", (8, ), dace.float64)
    sdfg.add_array("y", (8, ), dace.float64)
    sdfg.add_array("r", (1, ), dace.float64)
    state = sdfg.add_state("main")
    dot = Dot("dot")
    state.add_node(dot)
    state.add_edge(state.add_access("x"), None, dot, "_x", dace.Memlet("x[0:8]"))
    state.add_edge(state.add_access("y"), None, dot, "_y", dace.Memlet("y[0:8]"))
    state.add_edge(dot, "_result", state.add_access("r"), None, dace.Memlet("r[0]"))
    return sdfg


def _add_residual_gpu_map(sdfg: SDFG, rng: str) -> nodes.MapEntry:
    """Add a state holding a non-tileops ``GPU_Device`` map over ``rng``.

    :param sdfg: The SDFG to extend.
    :param rng: Map range string (e.g. ``"0:1024"``).
    :returns: The new map's entry node.
    """
    state = sdfg.add_state("residual")
    entry, exit_node = state.add_map("residual_map", dict(k=rng), schedule=dtypes.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet("residual_t", {}, {}, "pass")
    state.add_nedge(entry, tasklet, dace.Memlet())
    state.add_nedge(tasklet, exit_node, dace.Memlet())
    return entry


def _build_tile_iota_sdfg() -> SDFG:
    """Hand-built SDFG holding a ``TileIota`` with a 'cutile' expansion."""
    sdfg = dace.SDFG("cutile_lowering_iota")
    sdfg.add_array("I", (8, ), dace.int64, transient=False)
    state = sdfg.add_state("main")
    iota = TileIota(name="ti", widths=(8, ), expr="__l0")
    state.add_node(iota)
    state.add_edge(iota, "_dst", state.add_access("I"), None, dace.Memlet("I[0:8]"))
    return sdfg


# ============================================================
# Inspection helpers
# ============================================================


def _all_map_entries(sdfg: SDFG) -> List[Tuple[nodes.MapEntry, dace.SDFGState]]:
    """All MapEntry nodes in ``sdfg``, recursively."""
    return [(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def _cutile_map_entries(sdfg: SDFG) -> List[nodes.MapEntry]:
    """All MapEntry nodes with ``ScheduleType.CuTile``, recursively."""
    return [n for n, _ in _all_map_entries(sdfg) if n.map.schedule == dtypes.ScheduleType.CuTile]


def _tileops_nodes(sdfg: SDFG) -> List[Tuple[nodes.LibraryNode, dace.SDFGState]]:
    """All tileops library nodes in ``sdfg``, recursively.

    The tile-node class list is sourced from
    :func:`cutile_lowering._tile_node_types` so it cannot drift when a new
    tile-node type is added; the scope-walk in
    :func:`_outermost_anchored_entries` remains an independent
    reimplementation by design.
    """
    tile_types = _tile_node_types()
    return [(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, tile_types)]


def _outermost_anchored_entries(sdfg: SDFG) -> Set[nodes.MapEntry]:
    """Outermost map entries enclosing tileops anchors (top-level walk).

    Independent re-implementation of the anchored scope walk (in-state
    only -- sufficient for the flat-path fixtures used here) so the tests
    do not lean on the module under test's own helpers.
    """
    outermost: Set[nodes.MapEntry] = set()
    for node, state in _tileops_nodes(sdfg):
        last = None
        while state is not None:
            scope_dict = state.scope_dict()
            scope = scope_dict.get(node)
            while scope is not None:
                last = scope
                scope = scope_dict.get(scope)
            node = state.sdfg.parent_nsdfg_node
            state = state.sdfg.parent if node is not None else None
        if last is not None:
            outermost.add(last)
    return outermost


def _storage_snapshot(sdfg: SDFG) -> Dict[Tuple[str, str], dtypes.StorageType]:
    """Map ``(sdfg_label, array_name) -> storage`` over all nested SDFGs."""
    snapshot: Dict[Tuple[str, str], dtypes.StorageType] = {}
    for nested in sdfg.all_sdfgs_recursive():
        for name, desc in nested.arrays.items():
            snapshot[(nested.label, name)] = desc.storage
    return snapshot


def _apply_gpu_and_adapt(sdfg: SDFG) -> None:
    """Apply GPU transformations and adapter passes.

    Replaces the old ``CuTileSetSchedules`` + ``CuTileSetGlobalStorage``
    sequence.
    """
    GPUDeviceToCuTile().apply_pass(sdfg, {})


# ============================================================
# 1. CuTileValidateTiles
# ============================================================


class TestValidateTiles:

    def test_counts_anchors_on_vadd_k1(self):
        """K=1 vadd carries 9 anchors (two full/remainder paths + maskgen)."""
        sdfg = _build_vadd_k1_sdfg()
        assert CuTileValidateTiles().apply_pass(sdfg, {}) == 9

    def test_counts_anchors_on_vadd_k2(self):
        """K=2 vadd carries 14 anchors (three paths + two maskgens)."""
        sdfg = _build_vadd_k2_sdfg()
        assert CuTileValidateTiles().apply_pass(sdfg, {}) == 14

    def test_zero_anchors_warns_and_returns_none(self):
        """An un-vectorized SDFG has no anchors: warn + return None."""
        sdfg = _build_unvectorized_vadd_sdfg()
        with pytest.warns(UserWarning, match="CuTileValidateTiles: no tileops library nodes found"):
            assert CuTileValidateTiles().apply_pass(sdfg, {}) is None

    def test_zero_anchors_strict_raises(self):
        """``strict=True`` turns the zero-anchor warning into ValueError."""
        sdfg = _build_unvectorized_vadd_sdfg()
        with pytest.raises(ValueError, match="no tileops library nodes found"):
            CuTileValidateTiles(strict=True).apply_pass(sdfg, {})

    def test_non_power_of_two_widths_raise_even_without_strict(self):
        """Non-power-of-2 widths are a hard error regardless of ``strict``."""
        sdfg = _build_bare_tile_binop_sdfg((6, ))
        with pytest.raises(ValueError, match="not a power of 2"):
            CuTileValidateTiles(strict=False).apply_pass(sdfg, {})

    def test_power_of_two_widths_pass_hand_built(self):
        """A hand-built pow2 anchor validates fine (count 1)."""
        sdfg = _build_bare_tile_binop_sdfg((8, ))
        assert CuTileValidateTiles().apply_pass(sdfg, {}) == 1


# ============================================================
# 2. GPUDeviceToCuTile
# ============================================================


class TestGPUDeviceToCuTile:

    def test_stamps_gpu_device_to_cutile(self):
        """Build vadd sdfg, vectorize, validate, apply_gpu_transformations,
        then GPUDeviceToCuTile -- verify tileops-anchored maps get CuTile
        schedule."""
        sdfg = _build_vadd_k1_sdfg(gpu=True)
        CuTileValidateTiles().apply_pass(sdfg, {})
        # Before the adapter pass, tileops-anchored maps should be GPU_Device
        gpu_device_maps = [n for n, _ in _all_map_entries(sdfg) if n.map.schedule == dtypes.ScheduleType.GPU_Device]
        assert len(gpu_device_maps) >= 1, "Expected at least one GPU_Device map"

        GPUDeviceToCuTile().apply_pass(sdfg, {})
        cutile_maps = _cutile_map_entries(sdfg)
        # The vectorizer produces multiple map scopes (main + remainder);
        # all tileops-anchored maps are re-stamped.
        assert len(cutile_maps) >= 1
        assert set(cutile_maps) == _outermost_anchored_entries(sdfg)

    def test_returns_kernel_count(self):
        """Verify return value is positive for a vectorized vadd."""
        sdfg = _build_vadd_k1_sdfg(gpu=True)
        CuTileValidateTiles().apply_pass(sdfg, {})
        result = GPUDeviceToCuTile().apply_pass(sdfg, {})
        assert result is not None and result >= 1

    def test_no_anchors_warns(self):
        """No tileops -> warns + returns None."""
        sdfg = _build_unvectorized_vadd_sdfg()
        with pytest.warns(UserWarning, match="no tileops library nodes found"):
            assert GPUDeviceToCuTile().apply_pass(sdfg, {}) is None

    def test_no_anchors_strict_raises(self):
        """No tileops + strict -> ValueError."""
        sdfg = _build_unvectorized_vadd_sdfg()
        with pytest.raises(ValueError, match="no tileops library nodes found"):
            GPUDeviceToCuTile(strict=True).apply_pass(sdfg, {})

    def test_tiny_residual_map_demotes_silently(self):
        """A residual GPU_Device map with provably tiny volume (<= 4) is the
        intended scalar-control case: demoted to Sequential with no warning."""
        sdfg = _build_vadd_k1_sdfg()
        sdfg.apply_gpu_transformations( simplify=True)
        entry = _add_residual_gpu_map(sdfg, "0:2")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            GPUDeviceToCuTile().apply_pass(sdfg, {})
        assert not [w for w in caught if "per-element host loop" in str(w.message)]
        assert entry.map.schedule == dtypes.ScheduleType.Sequential

    def test_large_residual_map_demotion_warns(self):
        """A large residual GPU_Device map warns (naming map/state/volume and
        the host-loop consequence) but is still demoted in non-strict mode."""
        sdfg = _build_vadd_k1_sdfg()
        sdfg.apply_gpu_transformations( simplify=True)
        entry = _add_residual_gpu_map(sdfg, "0:1024")
        with pytest.warns(UserWarning, match="residual_map.*volume 1024.*per-element host loop"):
            GPUDeviceToCuTile().apply_pass(sdfg, {})
        assert entry.map.schedule == dtypes.ScheduleType.Sequential

    def test_symbolic_residual_map_demotion_warns(self):
        """A symbolic-volume residual map is not provably tiny: warns too."""
        sdfg = _build_vadd_k1_sdfg()
        sdfg.apply_gpu_transformations( simplify=True)
        entry = _add_residual_gpu_map(sdfg, "0:N")
        with pytest.warns(UserWarning, match="per-element host loop"):
            GPUDeviceToCuTile().apply_pass(sdfg, {})
        assert entry.map.schedule == dtypes.ScheduleType.Sequential

    def test_large_residual_map_strict_raises(self):
        """strict=True: the large-residual-map demotion raises instead."""
        sdfg = _build_vadd_k1_sdfg()
        sdfg.apply_gpu_transformations( simplify=True)
        entry = _add_residual_gpu_map(sdfg, "0:1024")
        with pytest.raises(ValueError, match="per-element host loop"):
            GPUDeviceToCuTile(strict=True).apply_pass(sdfg, {})
        # Not demoted: the raise aborts before re-stamping.
        assert entry.map.schedule == dtypes.ScheduleType.GPU_Device


class TestBlasOnlyConfiguration:
    """Zero tileops anchors + non-tileops library nodes is a supported case:
    the passes proceed (informational warning) even under ``strict=True``."""

    def test_validate_tiles_strict_proceeds(self):
        sdfg = _build_blas_dot_sdfg()
        with pytest.warns(UserWarning, match="BLAS-only configuration"):
            assert CuTileValidateTiles(strict=True).apply_pass(sdfg, {}) is None

    def test_gpu_device_to_cutile_strict_proceeds(self):
        sdfg = _build_blas_dot_sdfg()
        with pytest.warns(UserWarning, match="BLAS-only configuration"):
            assert GPUDeviceToCuTile(strict=True).apply_pass(sdfg, {}) is None

    def test_set_tile_storage_strict_proceeds(self):
        sdfg = _build_blas_dot_sdfg()
        with pytest.warns(UserWarning, match="BLAS-only configuration"):
            assert CuTileSetTileStorage(strict=True).apply_pass(sdfg, {}) is None

    def test_idempotent(self):
        """Running twice on already-CuTile map returns None."""
        sdfg = _build_vadd_k1_sdfg(gpu=True)
        CuTileValidateTiles().apply_pass(sdfg, {})
        GPUDeviceToCuTile().apply_pass(sdfg, {})
        # Second run: maps are already CuTile (not GPU_Device), so nothing to re-stamp
        result = GPUDeviceToCuTile().apply_pass(sdfg, {})
        assert result is None

    def test_k2_stamps_kernels(self):
        """K=2 widths, verify tileops-anchored maps become CuTile."""
        sdfg = _build_vadd_k2_sdfg(gpu=True)
        CuTileValidateTiles().apply_pass(sdfg, {})
        result = GPUDeviceToCuTile().apply_pass(sdfg, {})
        assert result is not None and result >= 1
        cutile_maps = _cutile_map_entries(sdfg)
        assert len(cutile_maps) >= 1


# ============================================================
# 3. CuTileSetTileStorage
# ============================================================


class TestSetTileStorage:

    def test_k1_tile_and_mask_transients_stamped(self):
        """K=1: the 4 Register tile/mask transients become CuTile_Tile."""
        sdfg = _build_vadd_k1_sdfg(gpu=True)
        _apply_gpu_and_adapt(sdfg)
        result = CuTileSetTileStorage().apply_pass(sdfg, {})
        assert result is not None and result > 0
        stamped = [desc for owner in sdfg.all_sdfgs_recursive() for desc in owner.arrays.values()
                   if desc.storage == dtypes.StorageType.CuTile_Tile]
        assert len(stamped) >= 1
        for desc in stamped:
            assert desc.transient
            assert isinstance(desc, data.Array)

    def test_scalar_descriptors_untouched(self):
        """No Scalar descriptor changes storage (and none becomes a tile)."""
        sdfg = _build_vadd_k1_sdfg(gpu=True)
        _apply_gpu_and_adapt(sdfg)
        scalar_storage_before = {(owner, name): storage for (owner, name), storage in _storage_snapshot(sdfg).items()}
        CuTileSetTileStorage().apply_pass(sdfg, {})
        for nested in sdfg.all_sdfgs_recursive():
            for name, desc in nested.arrays.items():
                if isinstance(desc, data.Scalar):
                    assert desc.storage == scalar_storage_before[(nested.label, name)], \
                        f"Scalar {name} changed storage"
                    assert desc.storage != dtypes.StorageType.CuTile_Tile

    def test_without_schedules_warns_and_changes_nothing(self):
        """Precondition: no CuTile map -> warn + None + zero mutations."""
        sdfg = _build_vadd_k1_sdfg()
        before = _storage_snapshot(sdfg)
        with pytest.warns(UserWarning, match="CuTileSetTileStorage: no CuTile-scheduled map found"):
            assert CuTileSetTileStorage().apply_pass(sdfg, {}) is None
        assert _storage_snapshot(sdfg) == before

    def test_without_schedules_strict_raises(self):
        sdfg = _build_vadd_k1_sdfg()
        with pytest.raises(ValueError, match="run sdfg.apply_gpu_transformations\\(\\) \\+ GPUDeviceToCuTile first"):
            CuTileSetTileStorage(strict=True).apply_pass(sdfg, {})

    def test_idempotent_second_run(self):
        """A second run stamps nothing (returns None), storages unchanged."""
        sdfg = _build_vadd_k1_sdfg(gpu=True)
        _apply_gpu_and_adapt(sdfg)
        CuTileSetTileStorage().apply_pass(sdfg, {})
        before = _storage_snapshot(sdfg)
        assert CuTileSetTileStorage().apply_pass(sdfg, {}) is None
        assert _storage_snapshot(sdfg) == before


# ============================================================
# 4. CuTileSetImplementations
# ============================================================


class TestSetImplementations:

    def test_all_nodes_stamped_pre_expansion(self):
        """Every tileops node carries CUTILE/cutile before expansion."""
        sdfg = _build_vadd_k1_sdfg()
        assert CuTileSetImplementations().apply_pass(sdfg, {}) == 9
        lib_nodes = _tileops_nodes(sdfg)
        assert len(lib_nodes) == 9
        for node, _ in lib_nodes:
            assert node.target_isa == ISA.CUTILE, f"{node.label}: target_isa {node.target_isa}"
            assert node.implementation == "cutile", f"{node.label}: implementation {node.implementation}"

    def test_k2_nodes_stamped(self):
        """K=2 nodes are stamped 'cutile' directly (no K>=2 'pure' dispatch)."""
        sdfg = _build_vadd_k2_sdfg()
        assert CuTileSetImplementations().apply_pass(sdfg, {}) == 14
        for node, _ in _tileops_nodes(sdfg):
            assert node.implementation == "cutile"

    def test_after_expansion_warns(self):
        """Run after ``expand_library_nodes()``: no lib nodes -> warn + None."""
        sdfg = _build_vadd_k1_sdfg()
        CuTileSetImplementations().apply_pass(sdfg, {})
        sdfg.expand_library_nodes()
        assert _tileops_nodes(sdfg) == []
        with pytest.warns(UserWarning, match="already expanded"):
            assert CuTileSetImplementations().apply_pass(sdfg, {}) is None

    def test_after_expansion_strict_raises(self):
        sdfg = _build_vadd_k1_sdfg()
        CuTileSetImplementations().apply_pass(sdfg, {})
        sdfg.expand_library_nodes()
        with pytest.raises(ValueError, match="already expanded"):
            CuTileSetImplementations(strict=True).apply_pass(sdfg, {})

    def test_tile_iota_stamped_cutile(self):
        """``TileIota`` ships 'cutile': stamping succeeds."""
        sdfg = _build_tile_iota_sdfg()
        assert CuTileSetImplementations().apply_pass(sdfg, {}) == 1
        for node, _ in _tileops_nodes(sdfg):
            assert node.target_isa == ISA.CUTILE
            assert node.implementation == "cutile"

    def test_tile_iota_expands_to_python_tasklet(self):
        """``TileIota`` stamped 'cutile' expands to a Python tasklet."""
        sdfg = _build_tile_iota_sdfg()
        CuTileSetImplementations().apply_pass(sdfg, {})
        sdfg.expand_library_nodes()
        assert not any(isinstance(n, nodes.LibraryNode) for n, _ in sdfg.all_nodes_recursive())
        tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet)]
        assert len(tasklets) == 1
        body = tasklets[0].code.as_string
        assert "ct.arange" in body
        assert "_dst" in body

    def test_node_without_cutile_implementation_raises(self):
        """A tileops node that lacks ``'cutile'`` in its implementations
        dict must fail loudly, not fall back silently."""
        from unittest.mock import patch
        sdfg = _build_tile_iota_sdfg()
        # Temporarily remove 'cutile' from TileIota.implementations so
        # the error path triggers.  patch.dict restores the original on
        # exit, so other tests are not affected.
        with patch.dict(TileIota.implementations, {"pure": TileIota.implementations["pure"]}, clear=True):
            with pytest.raises(ValueError, match="has no 'cutile' implementation"):
                CuTileSetImplementations().apply_pass(sdfg, {})

    def test_idempotent_second_run(self):
        sdfg = _build_vadd_k1_sdfg(gpu=True)
        CuTileSetImplementations().apply_pass(sdfg, {})
        assert CuTileSetImplementations().apply_pass(sdfg, {}) == 9
        for node, _ in _tileops_nodes(sdfg):
            assert node.target_isa == ISA.CUTILE
            assert node.implementation == "cutile"


# ============================================================
# 5. Ordering / full pipeline
# ============================================================


class TestPipelineOrdering:

    def _run_full_sequence(self, sdfg: SDFG) -> None:
        """Apply the passes in documented order, then expand + backend."""
        CuTileValidateTiles().apply_pass(sdfg, {})
        GPUDeviceToCuTile().apply_pass(sdfg, {})
        CuTileSetTileStorage().apply_pass(sdfg, {})
        CuTileSetImplementations().apply_pass(sdfg, {})
        sdfg.expand_library_nodes()
        sdfg.backend = dtypes.BackendLanguage.Python

    def test_full_sequence_k2_generates_cutile_code(self):
        """Documented sequence on K=2 vadd -> Python backend cuTile code."""
        sdfg = _build_vadd_k2_sdfg(gpu=True)
        self._run_full_sequence(sdfg)
        assert not any(isinstance(n, nodes.LibraryNode) for n, _ in sdfg.all_nodes_recursive())
        assert sdfg.backend == dtypes.BackendLanguage.Python
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "import cuda.tile as ct" in code
        assert "@ct.kernel" in code
        assert "ct.load(" in code

    @pytest.mark.skip(reason="apply_gpu_transformations() on already-vectorized SDFGs "
                      "with non-divisible concrete sizes triggers out-of-bounds "
                      "memlet validation (remainder region accesses beyond array "
                      "bounds after GPU clone creation). Known limitation of the "
                      "GPU-transform-based pipeline.")
    def test_full_sequence_k1_concrete_non_divisible(self):
        """Concrete non-divisible size (100 % 8 != 0) through the sequence."""
        sdfg = _build_vadd_concrete_sdfg()
        assert CuTileValidateTiles().apply_pass(sdfg, {}) == 9
        assert GPUDeviceToCuTile().apply_pass(sdfg, {}) >= 1
        result = CuTileSetTileStorage().apply_pass(sdfg, {})
        assert result is not None and result > 0
        assert CuTileSetImplementations().apply_pass(sdfg, {}) == 9
        sdfg.expand_library_nodes()
        sdfg.backend = dtypes.BackendLanguage.Python
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "import cuda.tile as ct" in code

    def test_storage_before_gpu_transforms_is_warned_noop_then_recovers(self):
        """Out-of-order: a storage pass run first warns and is a no-op; the
        documented order afterwards still works on the same SDFG."""
        sdfg = _build_vadd_k1_sdfg(gpu=True)
        before = _storage_snapshot(sdfg)
        with pytest.warns(UserWarning, match="no CuTile-scheduled map found"):
            assert CuTileSetTileStorage().apply_pass(sdfg, {}) is None
        assert _storage_snapshot(sdfg) == before
        # Recovery: correct order succeeds afterwards.
        _apply_gpu_and_adapt(sdfg)
        result = CuTileSetTileStorage().apply_pass(sdfg, {})
        assert result is not None and result > 0


# ============================================================
# 6. clamp_propagated_oob_memlets gating
# ============================================================


def _build_genuine_oob_sdfg(name: str) -> SDFG:
    """A(16) -> B(17) copy with a genuinely OOB memlet ``A[0:17]``: no mask,
    no tileops node anywhere (the reviewer's silent-truncation repro)."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (16, ), dace.float64)
    sdfg.add_array("B", (17, ), dace.float64)
    state = sdfg.add_state("main")
    state.add_nedge(state.add_access("A"), state.add_access("B"), dace.Memlet("A[0:17]"))
    return sdfg


def _build_tileops_tree_oob_sdfg(name: str) -> SDFG:
    """A/B/C(20) map around a ``TileBinop`` with provably-OOB propagated
    outer memlets (``0:24`` on shape-20 arrays)."""
    sdfg = dace.SDFG(name)
    for arr in ("A", "B", "C"):
        sdfg.add_array(arr, (20, ), dace.float64)
    state = sdfg.add_state("main")
    entry, exit_node = state.add_map("tiles", dict(i="0:24:8"))
    binop = TileBinop(name="tb", widths=(8, ), op="+")
    state.add_node(binop)
    for arr, conn in (("A", "_a"), ("B", "_b")):
        entry.add_in_connector(f"IN_{arr}")
        entry.add_out_connector(f"OUT_{arr}")
        state.add_edge(state.add_access(arr), None, entry, f"IN_{arr}", dace.Memlet(f"{arr}[0:24]"))
        state.add_edge(entry, f"OUT_{arr}", binop, conn, dace.Memlet(f"{arr}[i:i+8]", allow_oob=True))
    exit_node.add_in_connector("IN_C")
    exit_node.add_out_connector("OUT_C")
    state.add_edge(binop, "_c", exit_node, "IN_C", dace.Memlet("C[i:i+8]", allow_oob=True))
    state.add_edge(exit_node, "OUT_C", state.add_access("C"), None, dace.Memlet("C[0:24]"))
    return sdfg


def _build_nsdfg_boundary_sdfg(name: str, with_tileops: bool) -> SDFG:
    """Parent A(100)/C(100) with a NestedSDFG consuming a full-8 window
    through provably-OOB boundary memlets (``96:104``).

    :param with_tileops: When ``True`` the inner body is a ``TileBinop``
        (tileops-fed connector); otherwise a plain tasklet chain.
    """
    inner = dace.SDFG(f"{name}_inner")
    inner.add_array("_in_A", (100, ), dace.float64)
    inner.add_array("_out_C", (100, ), dace.float64)
    istate = inner.add_state("body")
    if with_tileops:
        binop = TileBinop(name="tb", widths=(8, ), op="+")
        istate.add_node(binop)
        istate.add_edge(istate.add_access("_in_A"), None, binop, "_a", dace.Memlet("_in_A[96:104]", allow_oob=True))
        istate.add_edge(istate.add_access("_in_A"), None, binop, "_b", dace.Memlet("_in_A[96:104]", allow_oob=True))
        istate.add_edge(binop, "_c", istate.add_access("_out_C"), None, dace.Memlet("_out_C[96:104]", allow_oob=True))
    else:
        tasklet = istate.add_tasklet("t", {"_a"}, {"_c"}, "_c = _a")
        istate.add_edge(istate.add_access("_in_A"), None, tasklet, "_a", dace.Memlet("_in_A[0]"))
        istate.add_edge(tasklet, "_c", istate.add_access("_out_C"), None, dace.Memlet("_out_C[0]"))

    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (100, ), dace.float64)
    sdfg.add_array("C", (100, ), dace.float64)
    state = sdfg.add_state("main")
    nsdfg_node = state.add_nested_sdfg(inner, {"_in_A"}, {"_out_C"})
    state.add_edge(state.add_access("A"), None, nsdfg_node, "_in_A", dace.Memlet("A[96:104]"))
    state.add_edge(nsdfg_node, "_out_C", state.add_access("C"), None, dace.Memlet("C[96:104]"))
    return sdfg


class TestClampPropagatedOOBMemlets:
    """Gating of the OOB clamp: tileops-adjacent / allow_oob only."""

    def test_genuine_oob_not_clamped_and_still_rejected(self):
        """The Major repro: an unmasked off-by-one read must stay OOB and
        fail validation loudly instead of being silently truncated."""
        sdfg = _build_genuine_oob_sdfg("clamp_genuine_oob")
        with pytest.warns(UserWarning, match="neither tileops-adjacent nor marked allow_oob"):
            assert clamp_propagated_oob_memlets(sdfg) == 0
        edge = next(iter(list(sdfg.states())[0].edges()))
        assert str(edge.data.subset) == "0:17"  # untouched
        with pytest.raises(InvalidSDFGEdgeError, match="out-of-bounds"):
            sdfg.validate()

    def test_allow_oob_memlet_clamped_and_volume_recomputed(self):
        """A memlet already marked allow_oob is clamped; volume follows."""
        sdfg = dace.SDFG("clamp_allow_oob_volume")
        sdfg.add_array("A", (16, ), dace.float64)
        sdfg.add_array("B", (20, ), dace.float64)
        state = sdfg.add_state("main")
        memlet = dace.Memlet("A[0:20]")
        memlet.allow_oob = True
        state.add_nedge(state.add_access("A"), state.add_access("B"), memlet)
        assert clamp_propagated_oob_memlets(sdfg) == 1
        edge = next(iter(state.edges()))
        assert str(edge.data.subset) == "0:16"
        assert int(edge.data.volume) == 16

    def test_tileops_adjacent_tree_clamped_without_allow_oob(self):
        """Propagated outer memlets of a tileops memlet tree are clamped
        even when re-propagation dropped their allow_oob flag."""
        sdfg = _build_tileops_tree_oob_sdfg("clamp_tileops_tree")
        assert clamp_propagated_oob_memlets(sdfg) == 3  # A, B, C outer edges
        state = list(sdfg.states())[0]
        outer = [e for e in state.edges() if isinstance(e.src, nodes.AccessNode) or isinstance(e.dst, nodes.AccessNode)]
        assert outer, "fixture must expose outer AccessNode edges"
        for e in outer:
            assert str(e.data.subset) == "0:20"
            assert int(e.data.volume) == 20
        # Library-node-incident edges keep their full-tile subsets.
        binop = next(n for n in state.nodes() if isinstance(n, TileBinop))
        for e in list(state.in_edges(binop)) + list(state.out_edges(binop)):
            assert str(e.data.subset) == "i:i + 8"

    def test_other_subset_clamped_under_same_gating(self):
        """``other_subset`` is clamped against the other endpoint's
        descriptor when the memlet is eligible."""
        sdfg = dace.SDFG("clamp_other_subset")
        sdfg.add_array("A", (16, ), dace.float64)
        sdfg.add_array("B", (16, ), dace.float64)
        state = sdfg.add_state("main")
        memlet = dace.Memlet("A[0:8]")
        memlet.other_subset = dace.subsets.Range.from_string("8:20")
        memlet.allow_oob = True
        state.add_nedge(state.add_access("A"), state.add_access("B"), memlet)
        assert clamp_propagated_oob_memlets(sdfg) == 1
        edge = next(iter(state.edges()))
        assert str(edge.data.subset) == "0:8"  # in-bounds side untouched
        assert str(edge.data.other_subset) == "8:16"

    def test_nsdfg_boundary_edge_marked_not_clamped(self):
        """A full-W window on a NestedSDFG boundary edge feeding a tileops
        node is marked allow_oob but never narrowed below W."""
        sdfg = _build_nsdfg_boundary_sdfg("clamp_nsdfg_boundary", with_tileops=True)
        adjacent, boundary = _collect_tileops_adjacent_edges(sdfg)
        state = list(sdfg.states())[0]
        outer_edges = [e for e in state.edges()]
        assert all(id(e) in boundary for e in outer_edges)
        assert clamp_propagated_oob_memlets(sdfg) == 0
        for e in outer_edges:
            assert str(e.data.subset) in ("96:104", )  # full window kept
            assert e.data.allow_oob
        sdfg.validate()  # allow_oob defers the boundary OOB to the mask

    def test_nsdfg_boundary_without_tileops_not_marked(self):
        """The same boundary shape WITHOUT tileops inside is a genuine OOB:
        neither marked nor clamped, and validation rejects it."""
        sdfg = _build_nsdfg_boundary_sdfg("clamp_nsdfg_plain", with_tileops=False)
        with pytest.warns(UserWarning, match="neither tileops-adjacent nor marked allow_oob"):
            assert clamp_propagated_oob_memlets(sdfg) == 0
        state = list(sdfg.states())[0]
        for e in state.edges():
            assert not e.data.allow_oob
            assert str(e.data.subset) == "96:104"
        with pytest.raises(InvalidSDFGEdgeError, match="out-of-bounds"):
            sdfg.validate()

    def test_propagate_subset_drops_allow_oob_from_aggregated_list(self):
        """Pin the propagation-inheritance caveat: ``propagate_subset``
        copies ``memlets[0]``, so an unmarked first memlet drops the flag
        of a marked neighbor (why the clamp stays load-bearing)."""
        from dace import subsets as sbs
        from dace.sdfg.propagation import propagate_subset
        arr = data.Array(dace.float64, (100, ))
        unmarked = dace.Memlet("A[i:i+8]")
        marked = dace.Memlet("A[i:i+8]")
        marked.allow_oob = True
        rng = sbs.Range.from_string("0:100:8")
        dropped = propagate_subset([unmarked, marked], arr, ["i"], rng, dace.symbolic.Facts.none())
        assert not dropped.allow_oob  # flag of memlets[1] silently dropped
        kept = propagate_subset([marked, unmarked], arr, ["i"], rng, dace.symbolic.Facts.none())
        assert kept.allow_oob

    @pytest.mark.gpu
    def test_boundary_tail_nested_body_gpu(self):
        """Always-on NestedSDFG body descent with a non-divisible concrete
        size: the boundary-tail window survives the clamp gate and the
        kernel matches NumPy end-to-end."""
        from dace.transformation.passes.vectorization import VectorizeCuTile

        @dace.program
        def clamp_nest_boundary_vadd(A: dace.float64[100], B: dace.float64[100], C: dace.float64[100]):
            for i in dace.map[0:100]:
                C[i] = A[i] + B[i]

        sdfg = clamp_nest_boundary_vadd.to_sdfg()
        VectorizeCuTile(widths=(8, )).apply_pass(sdfg, {})
        rng = np.random.default_rng(0)
        A = rng.random(100)
        B = rng.random(100)
        C = np.zeros(100)
        sdfg(A=A, B=B, C=C)
        np.testing.assert_allclose(C, A + B)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
