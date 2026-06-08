# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Inside-body staging pass (design section 3 + section 3.3).

For each tile-tagged Map's body NSDFG, walks every non-transient
AccessNode and stages each per-tile-subset access through a fresh
transient. The shape of the staged transient is picked from the per-
dim lattice (section 4):

* CONSTANT on every tile dim
    -> ``Scalar`` (or length-1 Array) transient. Direct AN -> AN scalar
       copy (no tasklet); the lib node downstream consumes it via the
       ``Scalar`` operand kind with hardware splat.

* At least one dim in {LINEAR, AFFINE, REPLICATE, MODULAR}
    -> ``(K_0, ..., K_{K-1})`` Array transient. AN -> TileLoad ->
       transient (read) / transient -> TileStore -> AN (write).

* Any dim is GATHER
    -> same tile-shape transient; the TileLoad / TileStore carries
       ``gather_dims=...`` + variable-shape ``_idx_<d>`` connectors
       fed by PreparePerLaneIndices.

This is the inside-body counterpart of the staging design
(STAGE_GLOBAL_THROUGH_SCALARS_SPEC.md): the outer global accesses now
flow through staged transients first, and the lib nodes are emitted at
the boundary inside the body. No non-transient AccessNode survives in
the middle of the body's dataflow.

Status (incremental landing):

* Step 1 (this commit): module-level helper :func:`stage_constant_access`
  that handles the CONSTANT-only case (direct AN -> AN scalar copy).
  Pass scaffold with a stub ``apply_pass`` that documents the contract.
  Unit tests for the helper.
* Step 2 (follow-up): :func:`stage_tile_access` for the
  LINEAR / AFFINE / REPLICATE / MODULAR case with TileLoad emission.
* Step 3 (follow-up): GATHER case + integration with
  :class:`PreparePerLaneIndices`.
* Step 4 (follow-up): Pass walker drives the helpers across every
  body NSDFG.
"""
from typing import Any, Dict, Optional, Tuple

from dace import dtypes
from dace.libraries.tileops import TileLoad
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg.nodes import AccessNode
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation


def stage_constant_access(state: SDFGState, an: AccessNode, name_hint: str = "constant_bridge") -> str:
    """Stage a CONSTANT-only access on ``an`` through a fresh ``Scalar`` transient.

    Mints a transient ``dace.data.Scalar`` of ``an``'s element dtype, adds it
    to ``state.sdfg.arrays``, and emits a direct AccessNode -> AccessNode edge
    from ``an`` to a new AccessNode wrapping the scalar -- **no copy tasklet
    between** per design section 3.6.

    Per design section 3.1: when every tile dim's per-dim kind is CONSTANT,
    the source value is loop-invariant across every tile lane. The lib node
    consuming the staged transient sees a ``Scalar`` operand kind and emits
    a hardware splat (AVX-512 ``_mm512_set1_pd``, SVE ``svdup_f64``, etc.).

    :param state: State holding ``an``.
    :param an: Non-transient AccessNode that's being staged.
    :param name_hint: Hint for the bridge transient name; uniquified via
        ``find_new_name=True``.
    :returns: Name of the staged transient (an AccessNode for it is added
        to ``state`` with the AN -> AN edge already wired).
    """
    sdfg = state.sdfg
    desc = sdfg.arrays[an.data]
    dtype = desc.dtype
    bridge_name, _ = sdfg.add_scalar(name_hint,
                                     dtype,
                                     transient=True,
                                     storage=dtypes.StorageType.Register,
                                     find_new_name=True)
    bridge_an = state.add_access(bridge_name)
    # Direct AN -> AN scalar copy per section 3.6: the memlet names the
    # source array on its data side; other_subset is omitted (full-shape
    # implicit), which the staged consumer's an_side_subset helper handles.
    # The 1-element subset on `an` side is naturally the loop-invariant
    # access (the classifier upstream guarantees this is CONSTANT-only).
    state.add_edge(an, None, bridge_an, None, Memlet(data=an.data, subset=f"0:1" if len(desc.shape) == 1 else None))
    return bridge_name


def stage_tile_access(state: SDFGState,
                      an: AccessNode,
                      widths: Tuple[int, ...],
                      src_subset: Memlet,
                      name_hint: str = "tile_bridge",
                      dim_strides: Optional[Tuple[int, ...]] = None,
                      replicate_factor_per_dim: Optional[Tuple[int, ...]] = None,
                      src_dims: Optional[Tuple[int, ...]] = None) -> Tuple[str, "TileLoad"]:
    """Stage a tile-shaped access on ``an`` through a fresh `(widths,)` Array transient.

    Mints a transient ``Array(shape=widths, ...)`` of ``an``'s element
    dtype, adds an :class:`TileLoad` lib node between ``an`` and the new
    transient, and wires the source / destination memlets. Used by the
    LINEAR / AFFINE / REPLICATE / MODULAR case (the access varies per
    tile lane but the index is affine / structured).

    Per design section 3.1: the consuming lib node reads the staged tile
    transient as a ``Tile`` operand at register granularity.

    :param state: State holding ``an``.
    :param an: Non-transient AccessNode being staged.
    :param widths: Tile widths ``(W_0, ..., W_{K-1})``.
    :param src_subset: The memlet that will be attached to the
        ``an -> TileLoad._src`` edge (carries the per-tile region).
    :param name_hint: Hint for the transient name; uniquified.
    :param dim_strides: Per-dim stride coefficients forwarded to
        :class:`TileLoad`. Defaults to all 1s (LINEAR).
    :param replicate_factor_per_dim: Per-dim REPLICATE factors;
        defaults to all 1s (no replication).
    :param src_dims: Source-array dim permutation. Defaults to the
        ``TileLoad`` default (innermost ``K`` dims in order).
    :returns: ``(bridge_name, load_node)`` -- the staged transient's
        name and the inserted :class:`TileLoad` instance.
    """
    sdfg = state.sdfg
    desc = sdfg.arrays[an.data]
    dtype = desc.dtype
    bridge_name, _ = sdfg.add_array(name_hint,
                                    shape=widths,
                                    dtype=dtype,
                                    transient=True,
                                    storage=dtypes.StorageType.Register,
                                    find_new_name=True)
    bridge_an = state.add_access(bridge_name)
    load = TileLoad(name=f"load_{bridge_name}",
                    widths=widths,
                    dim_strides=dim_strides,
                    replicate_factor_per_dim=replicate_factor_per_dim,
                    src_dims=src_dims)
    state.add_node(load)
    state.add_edge(an, None, load, "_src", src_subset)
    dst_subset_str = ", ".join(f"0:{w}" for w in widths)
    state.add_edge(load, "_dst", bridge_an, None, Memlet(f"{bridge_name}[{dst_subset_str}]"))
    return bridge_name, load


def stage_gather_access(state: SDFGState,
                        an: AccessNode,
                        widths: Tuple[int, ...],
                        src_subset: Memlet,
                        gather_dims: Tuple[int, ...],
                        idx_sources: Dict[int, AccessNode],
                        name_hint: str = "gather_bridge",
                        dim_strides: Optional[Tuple[int, ...]] = None,
                        replicate_factor_per_dim: Optional[Tuple[int, ...]] = None,
                        src_dims: Optional[Tuple[int, ...]] = None) -> Tuple[str, "TileLoad"]:
    """Stage a gather-shaped access on ``an`` through a fresh tile transient + ``TileLoad`` with ``gather_dims``.

    Mirrors :func:`stage_tile_access` but wires the per-dim
    ``_idx_<d>`` connectors using AccessNodes already in scope
    (typically materialised by :func:`materialise_per_lane_index_tile`
    from :mod:`prepare_per_lane_indices`).

    Per design section 3.1: gather staging produces the same tile
    transient shape as the structured case; only the lib node's
    ``gather_dims`` + the wired ``_idx_<d>`` connectors differ.

    :param state: State holding ``an``.
    :param an: Non-transient AccessNode being staged.
    :param widths: Tile widths.
    :param src_subset: The memlet on ``an -> TileLoad._src``.
    :param gather_dims: Sorted tuple of tile dims that gather (subset
        of ``range(K)``).
    :param idx_sources: ``{d: AccessNode}`` for each ``d in
        gather_dims``. Each AN's descriptor shape must be a Cartesian
        product of widths for some sorted subset of tile dims (design
        section 9.2 lane-dependency rule); :class:`TileLoad.validate`
        re-checks this.
    :param name_hint: Hint for the bridge transient name.
    :param dim_strides: Forwarded to :class:`TileLoad`.
    :param replicate_factor_per_dim: Forwarded to :class:`TileLoad`.
    :param src_dims: Forwarded to :class:`TileLoad`.
    :returns: ``(bridge_name, load_node)``.
    :raises ValueError: When ``set(gather_dims) != set(idx_sources)``.
    """
    if set(gather_dims) != set(idx_sources):
        raise ValueError(f"stage_gather_access: gather_dims {gather_dims!r} must match the keys of "
                         f"idx_sources {sorted(idx_sources)}")
    sdfg = state.sdfg
    desc = sdfg.arrays[an.data]
    dtype = desc.dtype
    bridge_name, _ = sdfg.add_array(name_hint,
                                    shape=widths,
                                    dtype=dtype,
                                    transient=True,
                                    storage=dtypes.StorageType.Register,
                                    find_new_name=True)
    bridge_an = state.add_access(bridge_name)
    load = TileLoad(name=f"load_{bridge_name}",
                    widths=widths,
                    dim_strides=dim_strides,
                    replicate_factor_per_dim=replicate_factor_per_dim,
                    src_dims=src_dims,
                    gather_dims=gather_dims)
    state.add_node(load)
    state.add_edge(an, None, load, "_src", src_subset)
    # Wire each _idx_<d> connector from the supplied AccessNode.
    for d in gather_dims:
        idx_an = idx_sources[d]
        idx_desc = sdfg.arrays[idx_an.data]
        idx_subset = ", ".join(f"0:{s}" for s in idx_desc.shape)
        state.add_edge(idx_an, None, load, f"_idx_{d}", Memlet(f"{idx_an.data}[{idx_subset}]"))
    dst_subset_str = ", ".join(f"0:{w}" for w in widths)
    state.add_edge(load, "_dst", bridge_an, None, Memlet(f"{bridge_name}[{dst_subset_str}]"))
    return bridge_name, load


@transformation.explicit_cf_compatible
class StageInsideBody(ppl.Pass):
    """Inside-body staging pass scaffold (design section 3.3).

    Walks every tile-tagged Map's body NSDFG and stages each non-transient
    access through a fresh transient sized per the classifier output. Step 1
    landed the CONSTANT helper; subsequent steps wire the Tile + Gather
    helpers + the walker that drives them across the body.
    """

    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors | ppl.Modifies.Tasklets
                | ppl.Modifies.Nodes)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Stub -- the walker lands in step 4.

        :returns: ``None`` (no rewrites performed by the stub).
        """
        return None
