# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Explicit SIMD vectorization pass: tiles innermost maps to the vector width,
rewrites tasklets/memlets to vector form, and vectorizes nested SDFG bodies."""
import dace
import copy
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from dace import SDFG, Memlet, SDFGState, properties, transformation
from dace import typeclass
import dace.sdfg.construction_utils as cutil
from dace.properties import CodeBlock
from dace.sdfg.graph import Edge
from dace.sdfg.nodes import CodeNode
from dace.sdfg.sdfg import InterstateEdge
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.clean_access_node_to_scalar_slice_to_tasklet_pattern import CleanAccessNodeToScalarSliceToTaskletPattern
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import RemoveFPTypeCasts, RemoveIntTypeCasts, PowerOperatorExpansion
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.passes.vectorization.utils import *
from dace.transformation.passes.vectorization.utils.nsdfg_reshape import (
    _setup_strided_inside_nsdfg,
    _setup_multi_element_strided_inside_nsdfg,
    emit_staging_copy,
)


# NOTE: post_descent_invariants was deleted with the legacy descent (PromoteNSDFGBodyToTiles).
# The walker-primary path (StageInsideBody + ClearPerLaneIndexSymbols audit) replaces both.
# Local stubs below preserve the surface for any 1D-legacy ``vectorize.py`` callers that still
# reach for the descent invariants helper -- they become no-ops since the descent is gone.
def assert_post_descent_invariants(*args, **kwargs):
    """No-op stub -- legacy descent was deleted in the walker-primary migration."""
    return None


def cleanup_an_to_an_edges(*args, **kwargs):
    """No-op stub -- legacy descent was deleted in the walker-primary migration."""
    return None


from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
import dace.sdfg.tasklet_utils as tutil
from dace.transformation.passes.vectorization.utils.symbolic_polymorphism import free_symbol_names, free_symbols


@properties.make_properties
@transformation.explicit_cf_compatible
class Vectorize(ppl.Pass):
    """Vectorizes innermost maps to a fixed vector width, including nested SDFG bodies."""

    templates = properties.DictProperty(
        key_type=str,
        value_type=str,
    )
    vector_width = properties.SymbolicProperty(default=8)
    vector_input_storage = properties.Property(dtype=dace.dtypes.StorageType, default=dace.dtypes.StorageType.Register)
    vector_output_storage = properties.Property(dtype=dace.dtypes.StorageType, default=dace.dtypes.StorageType.Register)
    global_code = properties.Property(dtype=str, default="")
    global_code_location = properties.Property(dtype=str, default="")
    vector_op_numeric_type = properties.Property(dtype=typeclass, default=dace.float64)
    try_to_demote_symbols_in_nsdfgs = properties.Property(dtype=bool, default=False)
    fuse_overlapping_loads = properties.Property(dtype=bool, default=False)
    insert_copies = properties.Property(dtype=bool, default=True, allow_none=False)
    fail_on_unvectorizable = properties.Property(dtype=bool, default=False, allow_none=False)
    eliminate_trivial_vector_map = properties.Property(dtype=bool, default=True, allow_none=False)
    user_skip_nsdfg_arrays = properties.SetProperty(element_type=str, default=set())

    def __init__(self,
                 templates: Dict[str, str],
                 vector_width: str,
                 vector_input_storage: dace.dtypes.StorageType,
                 vector_output_storage: dace.dtypes.StorageType,
                 vector_op_numeric_type: typeclass,
                 global_code: str,
                 global_code_location: str,
                 try_to_demote_symbols_in_nsdfgs: bool,
                 apply_on_maps: Optional[List[str]],
                 insert_copies: bool,
                 fail_on_unvectorizable: bool,
                 eliminate_trivial_vector_map: bool,
                 user_skip_nsdfg_arrays: Optional[Set[str]] = None,
                 fuse_overlapping_loads: bool = False):
        """Configure the vectorizer.

        :param templates: op-name to C++ vector-template string mapping.
        :param vector_width: number of lanes per vector.
        :param vector_input_storage: storage type for vector inputs.
        :param vector_output_storage: storage type for vector outputs.
        :param vector_op_numeric_type: numeric type used for vector ops.
        :param global_code: C++ global code to append to the SDFG.
        :param global_code_location: location key for the global code.
        :param try_to_demote_symbols_in_nsdfgs: attempt to demote NSDFG symbols.
        :param apply_on_maps: optional whitelist of map entries to vectorize.
        :param insert_copies: insert copy-in/copy-out around vectorized maps.
        :param fail_on_unvectorizable: raise instead of skipping unvectorizable maps.
        :param eliminate_trivial_vector_map: remove trivial single-lane vector maps.
        :param user_skip_nsdfg_arrays: array names to skip during NSDFG copying.
        :param fuse_overlapping_loads: when True, the NSDFG-boundary copy
            classifier fuses multiple overlapping read subsets of the same
            array into a single shared union-window staging buffer (the
            consumers become offset views into it) instead of emitting one
            independent copy per subset. Baked into the copy-emission
            classifier so it composes with the staging path, replacing the
            former standalone post-vectorizer ``FuseOverlappingLoads`` pass.
        """
        super().__init__()

        self.templates = templates
        self.vector_width = str(vector_width)
        self.vector_input_storage = vector_input_storage
        self.vector_output_storage = vector_output_storage
        self.global_code = global_code
        self.global_code_location = global_code_location
        self.vector_op_numeric_type = vector_op_numeric_type
        self.try_to_demote_symbols_in_nsdfgs = try_to_demote_symbols_in_nsdfgs
        self.insert_copies = insert_copies
        self.fuse_overlapping_loads = fuse_overlapping_loads
        self.fail_on_unvectorizable = fail_on_unvectorizable
        self._used_names = set()
        self._apply_on_maps = apply_on_maps
        self.eliminate_trivial_vector_map = eliminate_trivial_vector_map
        self.user_skip_nsdfg_arrays = set(user_skip_nsdfg_arrays) if user_skip_nsdfg_arrays else set()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies):
        return False

    def depends_on(self):
        return {
            PowerOperatorExpansion, SplitTasklets, RemoveFPTypeCasts, RemoveIntTypeCasts,
            CleanAccessNodeToScalarSliceToTaskletPattern
        }

    def _vectorize_map(self, state: SDFGState, inner_map_entry: dace.nodes.MapEntry, vectorization_number: int):
        """Tile, rewrite, and vectorize a single innermost map (and its NSDFG body if any).

        :param state: the state containing the map.
        :param inner_map_entry: the innermost map entry to vectorize.
        :param vectorization_number: unique index used to name per-map vector arrays.
        """
        # Get the innermost maps
        assert isinstance(inner_map_entry, dace.nodes.MapEntry)
        assert inner_map_entry in state.nodes()

        # Before anything try to clean other subset going from map entry
        try_clean_other_subset_going_out_from_map_entry(state, inner_map_entry)

        # Normalize every body NSDFG's AN -> AN edges to the canonical
        # ``AN -> [_out=_in] -> AN`` form BEFORE any widening. The post-
        # descent audit refuses surviving AN -> AN edges; running cleanup
        # here keeps the legacy path in lock-step with the K-dim descent.
        for n in state.scope_subgraph(inner_map_entry).nodes():
            if isinstance(n, dace.nodes.NestedSDFG):
                cleanup_an_to_an_edges(n.sdfg)

        tile_sizes = [1 for _ in inner_map_entry.map.range]
        tile_sizes[-1] = self.vector_width
        assert tile_sizes != []
        assert tile_sizes != [1 for _ in inner_map_entry.map.range]

        MapTiling.apply_to(
            sdfg=state.sdfg,
            map_entry=inner_map_entry,
            options={
                "tile_sizes": tile_sizes,
                "skew": False,
                "divides_evenly": True,
            },
        )

        new_inner_map = inner_map_entry
        new_inner_map.schedule = dace.dtypes.ScheduleType.Sequential
        new_inner_map.map.label = "vectorloop_" + new_inner_map.map.label

        # If it has any branching out then move the branching one level up.
        if map_has_branching_memlets(state, new_inner_map):
            cutil.duplicate_memlets_sharing_single_in_connector(state, new_inner_map, True)
            state.validate()

        (b, e, s) = new_inner_map.map.range[0]
        assert len(new_inner_map.map.range) == 1
        vector_map_param = new_inner_map.map.params[0]
        try:
            int_size = int(e + 1 - b)
            int_vwidth = int(self.vector_width)
        except:
            int_size = None
            int_vwidth = None
        assert (int_size is not None and int_size == int_vwidth) or (
            e - b + 1
        ).approx == self.vector_width, f"MapTiling should have created a map with range of size {self.vector_width}, found {(e - b + 1)}"
        assert s == 1, f"MapTiling should have created a map with stride 1, found {s}"

        # Copy over the subsets of the map above by just replacing the memlet subsets
        # Use the map parameter of the previous map, makes implementation of future parts easier

        # We do this only if we have no nestedSDFGs
        nodes = state.all_nodes_between(new_inner_map, state.exit_node(new_inner_map))
        has_single_nested_sdfg = len(nodes) == 1 and isinstance(next(iter(nodes)), dace.nodes.NestedSDFG)

        if not has_single_nested_sdfg:
            vectorizable_arrays = collect_non_unit_stride_accesses_in_map(state.sdfg, state, new_inner_map)
            use_previous_subsets(state, inner_map_entry, self.vector_width, vectorizable_arrays)
        else:
            use_previous_subsets(state, inner_map_entry, self.vector_width, set())

        state.sdfg.validate()
        new_inner_map.map.range = dace.subsets.Range([(b, e, dace.symbolic.SymExpr(self.vector_width))])

        nodes = state.all_nodes_between(new_inner_map, state.exit_node(new_inner_map))

        # Updates memlets from [k, i] to [k, i:i+4]
        if not has_single_nested_sdfg:
            array_accesses_to_be_packed = {k for k, v in vectorizable_arrays.items() if v is False}
            # Squeeeze the memlets that have more volume than the vector with (strided access) that need to be packed
            # e.g. stride-2 access will be something like [2*i:2*i+15] (vector length 8) before packing we need to make it back to [2*i:2*i].
            squeeze_memlets_of_packed_arrays(state, new_inner_map, array_accesses_to_be_packed)
            # This function also fixes strided loads
            modified_nodes, modified_edges = self._generate_strided_loads_to_packed_storage(
                state.sdfg, state, {vector_map_param}, new_inner_map, array_accesses_to_be_packed)
            modified_nodes2, modified_edges2 = self._generate_strided_stores_from_packed_storage(
                state.sdfg, state, {vector_map_param}, new_inner_map, array_accesses_to_be_packed)
            modified_nodes = modified_nodes.union(modified_nodes2)
            modified_edges = modified_edges.union(modified_edges2)
        else:
            modified_nodes = set()
            modified_edges = set()

        self._extend_memlets(state, new_inner_map, modified_edges)
        # Special case, // 2 (or integer dividing the thing nicely) access -> need to multiplex elements to be able to vectorize
        new_modified_nodes, new_modified_edges = detect_halve_index(state,
                                                                    new_inner_map,
                                                                    vector_length=self.vector_width)
        modified_edges = modified_edges.union(new_modified_edges)
        modified_nodes = modified_nodes.union(new_modified_nodes)
        self._extend_temporary_scalars(state, new_inner_map, modified_nodes, modified_edges)
        state.sdfg.validate()

        if not has_single_nested_sdfg:
            if self.insert_copies:
                self._copy_in_and_copy_out(state, new_inner_map, vectorization_number, vectorizable_arrays)

        # Replactes tasklets of form A op B to vectorized_op(A, B)
        self._replace_tasklets(state, new_inner_map, vector_map_param, modified_nodes)
        # Copies in data to the storage needed by the vector unit
        # Copies out data from the storage needed by the vector unit
        # Copy-in out needs to skip scalars and arrays that pass complete dimensions (over approximation must be due to a reason)

        # If tasklet -> sclar -> tasklet, now we have,
        # vector_tasklet -> scalar -> vector_tasklet
        # makes the scalar into vector
        # If the inner node is a NestedSDFG we need to vectorize that too
        state.sdfg.validate()

        if has_single_nested_sdfg:
            nsdfg_node = next(iter(nodes))
            # Strided NSDFG edges (bbox in the stride-1 dim wider than
            # ``vector_width`` *because* the begin scales with the map
            # param at coefficient > 1) get the strided-load /
            # strided-store rewrite inside the NSDFG body via
            # ``_setup_strided_inside_nsdfg``.  Two routes wire it in:
            # ``insert_copies=True`` runs ``add_copies_before_and_after_nsdfg``
            # later in this method which calls into ``_process_edges``;
            # ``insert_copies=False`` instead routes here so kernels
            # like TSVC s127 / s1111 with stride-2 writes inside a
            # P1-wrapped body still vectorize correctly under scalar /
            # masked remainder.  A contiguous full-slice ``A[0:32]``
            # (begin doesn't reference the map param) is NOT strided
            # and is left for the standard reshape path to handle.
            if not self.insert_copies:
                self._setup_strided_nsdfg_edges_inline(state, nsdfg_node, vector_map_param)
            fix_nsdfg_connector_array_shapes_mismatch(state, nsdfg_node, vector_width=int(self.vector_width))
            check_nsdfg_connector_array_shapes_match(state, nsdfg_node)
            unstructured_data = self._vectorize_nested_sdfg(state, nsdfg_node, vector_map_param)

            if self.insert_copies:
                # Combine the pass's per-call "unstructured" arrays (computed during nested-SDFG
                # vectorization) with any user-supplied opt-in skip list, then forward as the
                # `skip` parameter — replaces the previously-hardcoded cloudsc array names.
                copy_skip = unstructured_data | self.user_skip_nsdfg_arrays
                inserted_array_names = add_copies_before_and_after_nsdfg(
                    state,
                    nsdfg_node,
                    self.vector_width,
                    self.vector_input_storage,
                    copy_skip,
                    fuse_overlapping_loads=self.fuse_overlapping_loads)

        # Post-vectorization invariants shared with the K-dim descent: no
        # residual ``other_subset`` (codegen would emit a wrong-stride
        # ``CopyND``) and at most one WCR per map terminating at the map
        # exit (multiple chains race, mid-chain WCR breaks accumulator
        # atomicity). See
        # :mod:`dace.transformation.passes.vectorization.utils.post_descent_invariants`.
        body_sdfgs = [
            n.sdfg for n in state.scope_subgraph(new_inner_map).nodes() if isinstance(n, dace.nodes.NestedSDFG)
        ]
        assert_post_descent_invariants(state, new_inner_map, body_sdfgs)

    def parent_connection_is_scalar(self, state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG,
                                    scalar_name: str) -> bool:
        """Return whether an NSDFG connector is backed by a Scalar in the parent.

        :param state: the state containing the nested SDFG.
        :param nsdfg: the nested SDFG node.
        :param scalar_name: the connector name to inspect.
        :returns: True if the parent-side data is a Scalar.
        :raises Exception: if the connector is not found on the NSDFG.
        """
        for ie in state.in_edges(nsdfg):
            if ie.dst_conn == scalar_name:
                dataname = ie.data.data
                if isinstance(state.sdfg.arrays[dataname], dace.data.Scalar):
                    return True
        for oe in state.out_edges(nsdfg):
            if oe.src_conn == scalar_name:
                dataname = oe.data.data
                if isinstance(state.sdfg.arrays[dataname], dace.data.Scalar):
                    return True
        raise Exception("Parent connect is scalar called but the scalar is not found in the connectors"
                        )  # Shouldn't happen but whatever

    def _setup_strided_nsdfg_edges_inline(self, state: dace.SDFGState, nsdfg_node: dace.nodes.NestedSDFG,
                                          map_param: str):
        """Emit a strided load/store inside the NSDFG body for non-contiguous boundary edges.

        Handles both single-dim strided accesses (one stride-1 dim with a
        wider-than-W bounding box and the map param in its begin) and
        multi-dim strided accesses (e.g. a diagonal ``A[i, i]``), linearising
        the inter-lane stride through array strides. Contiguous full-slice
        edges are left for the standard reshape path.

        :param state: the state containing the nested SDFG.
        :param nsdfg_node: the nested SDFG node.
        :param map_param: the vectorized map parameter name.
        :raises NotImplementedError: if a strided edge does not match a
            supported stride/element pattern.
        """
        inner_sdfg = nsdfg_node.sdfg
        for direction in ("in", "out"):
            edges = (list(state.in_edges(nsdfg_node)) if direction == "in" else list(state.out_edges(nsdfg_node)))
            for e in edges:
                if e.data.data is None or e.data.subset is None:
                    continue
                arr = state.sdfg.arrays[e.data.data]

                # Classify each dim:
                # - param_dims_wide: param appears in begin, bbox > 1 (the lanes scatter here)
                # - any_param_dims: param appears in begin (regardless of bbox)
                param_dims_wide = []
                any_param_dims = []
                map_sym = dace.symbolic.symbol(map_param)
                for d, (b, ee, s) in enumerate(e.data.subset):
                    if not free_symbols(b):
                        continue
                    if map_param not in {str(sym) for sym in b.free_symbols}:
                        continue
                    any_param_dims.append(d)
                    try:
                        bbox_vol_d = int(ee - b + 1)
                    except (TypeError, ValueError):
                        bbox_vol_d = None
                    if s == 1 and bbox_vol_d is not None and bbox_vol_d > 1:
                        param_dims_wide.append((d, bbox_vol_d))

                if not param_dims_wide:
                    continue

                # 1D strided path (existing): exactly one stride-1 dim is
                # the param-bearing wide dim, bbox > W. Stride is
                # ``(bbox - 1) / (W - 1)``.
                #
                # Strided vs contiguous is decided by the *inter-lane
                # stride* — the memory displacement between what lane
                # ``l`` and lane ``l+1`` access on the wide dim, i.e.
                # ``begin(map_param + 1) - begin(map_param)``.  This is
                # robust to stencil halo width and arbitrary affine
                # offsets:
                #
                # - ``A[2*i]``           → 2*(j+1) - 2*j           = 2  (strided)
                # - ``A[i+1, j+1]``      → (j+1+1) - (j+1)         = 1  (contiguous,
                #   5-point jacobi: bbox = W + 2 > W but lanes are 1 apart)
                # - 9-box / wide stencil → still 1  (bbox = W + 2*halo,
                #   inter-lane stride unchanged — correctly contiguous)
                # - ``A[i + LEN//2]``    → 1  (contiguous, offset only)
                #
                # Only an inter-lane stride > 1 is a genuine strided
                # access for the ``(W-1)*S+K`` handler; stride == 1
                # (any halo width) must fall through to the normal
                # contiguous reshape path.
                stride_one_dims = [d for d, st in enumerate(arr.strides) if st == 1]
                inter_lane_stride = None
                if len(param_dims_wide) == 1:
                    wb = e.data.subset[param_dims_wide[0][0]][0]
                    # This is the same inter-lane step that
                    # ``lane_access.classify_lane_access`` exposes as
                    # ``LaneAccess.inter_lane_stride``; it is recomputed
                    # locally (not delegated) because the strided-handler
                    # selects the wide dim by per-dim bounding-box volume,
                    # which the lane-access classifier intentionally does
                    # not model — delegating would change which dim is
                    # picked for multi-param edges.
                    # Inter-lane stride = begin(map_param + 1) - begin(map_param),
                    # computed on the dace symbolic begin expression itself.
                    # ``map_sym`` is the dace symbol named ``map_param`` (not a
                    # raw ``sympy.Symbol``): a raw sympy symbol does not unify
                    # with the dace symbol carried in ``wb``, so ``.subs``
                    # would no-op and the ``int(...)`` would raise "Cannot
                    # convert symbols to int", silently disabling every
                    # genuinely strided edge (the K-elements-per-iter bug).
                    try:
                        inter_lane_stride = int(dace.symbolic.simplify(wb.subs(map_sym, map_sym + 1) - wb))
                    except (TypeError, ValueError):
                        inter_lane_stride = None
                is_single_dim_strided = (len(param_dims_wide) == 1 and len(stride_one_dims) == 1
                                         and param_dims_wide[0][0] == stride_one_dims[0]
                                         and param_dims_wide[0][1] > self.vector_width and inter_lane_stride is not None
                                         and inter_lane_stride > 1)

                if is_single_dim_strided and self.vector_width > 1:
                    bbox_vol = param_dims_wide[0][1]
                    W = self.vector_width
                    inner_conn = e.dst_conn if direction == "in" else e.src_conn

                    # Generalised K-elements-per-iter strided: each lane
                    # accesses K consecutive elements per iter, consecutive
                    # lanes are ``S`` (inter-lane stride) apart. Total
                    # bbox = (W-1)*S + K.
                    #
                    # - K=1, any S: single-element-per-iter at stride S
                    #   (TSVC s1111-shape outer scatter / s293 strided).
                    # - K>1, S=K: K-element contiguous bbox (TSVC s127).
                    # - K>1, S>K: K-element scatter/gather with gaps.
                    #
                    # ``K`` comes from the inner connector's stride-1 dim
                    # size — this is the per-iter element count carried
                    # through the NSDFG boundary by ``prepare_vectorized_array``
                    # / ``fix_nsdfg_connector_array_shapes_mismatch``.
                    inner_arr = inner_sdfg.arrays.get(inner_conn)
                    inner_dim0 = None
                    if inner_arr is not None and len(inner_arr.shape) >= 1:
                        try:
                            inner_dim0 = int(inner_arr.shape[-1])
                        except Exception:
                            inner_dim0 = None
                    K_candidate = inner_dim0 if (inner_dim0 is not None and inner_dim0 >= 1) else 1
                    handled = False
                    if (bbox_vol - K_candidate) % (W - 1) == 0:
                        S_value = int(dace.symbolic.int_floor(bbox_vol - K_candidate,
                                                              W - 1))  # concrete pass-time int (never C++ '/')
                        if S_value >= K_candidate:
                            if K_candidate == 1:
                                _setup_strided_inside_nsdfg(state,
                                                            nsdfg_node,
                                                            inner_sdfg,
                                                            e,
                                                            inner_conn,
                                                            e.data.data,
                                                            arr,
                                                            W,
                                                            S_value,
                                                            direction=direction)
                            else:
                                _setup_multi_element_strided_inside_nsdfg(state,
                                                                          nsdfg_node,
                                                                          inner_sdfg,
                                                                          e,
                                                                          inner_conn,
                                                                          e.data.data,
                                                                          arr,
                                                                          W,
                                                                          elements_per_iter=K_candidate,
                                                                          stride=S_value,
                                                                          direction=direction)
                            handled = True
                    if not handled:
                        # Fall back to the older K=1 single-element check
                        # (some pre-Slice-1 callers don't populate
                        # ``inner_arr`` with the K value).
                        if (bbox_vol - 1) % (W - 1) == 0:
                            stride_value = int(dace.symbolic.int_floor(bbox_vol - 1,
                                                                       W - 1))  # concrete pass-time int (never C++ '/')
                            _setup_strided_inside_nsdfg(state,
                                                        nsdfg_node,
                                                        inner_sdfg,
                                                        e,
                                                        inner_conn,
                                                        e.data.data,
                                                        arr,
                                                        W,
                                                        stride_value,
                                                        direction=direction)
                            handled = True
                    if not handled:
                        raise NotImplementedError(
                            f"Vectorize: strided NSDFG edge on {e.data.data} has bbox "
                            f"volume {bbox_vol}; doesn't match (W-1)*S + K for any K in "
                            f"[1, {inner_arr.shape if inner_arr else None}] and vector_width={W}.")
                    continue

                # Single NON-contiguous wide param dim: the vectorized
                # innermost dim indexes a non-unit-stride array dim (e.g.
                # C-layout ``bb[i, j]`` with ``i`` innermost,
                # ``strides=(N, 1)``). Each lane is one element; consecutive
                # lanes are ``coeff * arr.strides[d]`` apart -> strided
                # gather. (The contiguous single-dim case is handled above;
                # this is its non-contiguous sibling, judged by the
                # descriptor stride, not dim position.)
                if (len(param_dims_wide) == 1 and self.vector_width > 1 and arr.strides[param_dims_wide[0][0]] != 1
                        and param_dims_wide[0][1] == self.vector_width):
                    d0 = param_dims_wide[0][0]
                    # Guard: a strided access widens ONLY the lane dim; every
                    # other dim must be a single element (unit). A wide
                    # non-lane dim (e.g. an inner sequential loop's full
                    # range, ``bb[j:j+8, 0:N]``) is the mixed strided+gather
                    # edge case -- refuse (-> clean skip) rather than emit a
                    # 2D access the downstream strided-store can't express.
                    for _d, (_b, _ee, _s) in enumerate(e.data.subset):
                        if _d == d0:
                            continue
                        try:
                            _ln = int(_ee - _b + 1)
                        except (TypeError, ValueError):
                            _ln = None
                        if _ln is None or _ln > 1:
                            raise NotImplementedError(
                                f"Vectorize: non-contiguous strided edge on {e.data.data} has a wide "
                                f"non-lane dim {_d} (mixed strided+gather; lane dim {d0}); not supported.")
                    b0, _ee0, _s0 = e.data.subset[d0]
                    try:
                        coeff0 = b0.coeff(map_sym)
                    except Exception:
                        coeff0 = None
                    if coeff0 is None:
                        raise NotImplementedError(
                            f"Vectorize: non-contiguous strided NSDFG edge on {e.data.data} dim {d0} "
                            f"has non-linear param dependence ({b0}); only linear begins supported.")
                    inner_conn = e.dst_conn if direction == "in" else e.src_conn
                    _setup_strided_inside_nsdfg(state,
                                                nsdfg_node,
                                                inner_sdfg,
                                                e,
                                                inner_conn,
                                                e.data.data,
                                                arr,
                                                self.vector_width,
                                                coeff0 * arr.strides[d0],
                                                direction=direction,
                                                multi_dim_param_dims=(d0, ))
                    continue

                # Multi-dim strided path: param-bearing dims expand to bbox W
                # each, and the linearised inter-lane stride is the sum of
                # ``arr.strides[d] * coeff(d)`` where ``coeff(d)`` is the
                # coefficient of ``map_param`` in dim d's begin expression.
                # For ``A[i, i]``: dim 0 has coeff 1 + stride N; dim 1 has
                # coeff 1 + stride 1 → linear stride = N + 1.
                if len(param_dims_wide) >= 2 and self.vector_width > 1:
                    # Every wide param-dim's bbox must equal W (sanity).
                    if not all(bw == self.vector_width for _, bw in param_dims_wide):
                        raise NotImplementedError(f"Vectorize: multi-dim strided NSDFG edge on {e.data.data} has "
                                                  f"non-W bbox per param-dim: {param_dims_wide}; only W per dim is "
                                                  f"supported today.")
                    linear_stride = 0
                    for d, _bw in param_dims_wide:
                        b, _ee, _s = e.data.subset[d]
                        # Coefficient of map_param in b. ``b.coeff(sym)`` returns
                        # the integer/symbolic coefficient or 0 if absent.
                        try:
                            coeff = b.coeff(map_sym)
                        except Exception:
                            coeff = None
                        if coeff is None:
                            raise NotImplementedError(f"Vectorize: multi-dim strided NSDFG edge on {e.data.data} dim "
                                                      f"{d} has non-linear param dependence ({b}); only linear "
                                                      f"begins supported.")
                        linear_stride = linear_stride + coeff * arr.strides[d]
                    inner_conn = e.dst_conn if direction == "in" else e.src_conn
                    _setup_strided_inside_nsdfg(state,
                                                nsdfg_node,
                                                inner_sdfg,
                                                e,
                                                inner_conn,
                                                e.data.data,
                                                arr,
                                                self.vector_width,
                                                linear_stride,
                                                direction=direction,
                                                multi_dim_param_dims=tuple(d for d, _ in param_dims_wide))

    def _boundary_lane_dims(self, state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG, vector_map_param: str) -> dict:
        """Map each NSDFG connector to its lane dim in inner-array coordinates.

        The connector name equals the inner array name. For each boundary
        edge, the lane dim is the subset dim whose ``begin`` references
        ``vector_map_param`` (e.g. ``bb[i, j]`` -> dim 0 for innermost
        ``i``). Used by ``expand_memlet_expression`` when the inner body
        access has lost the param to a length-1 connector view.

        ``prepare_vectorized_array`` collapses every length-1 boundary
        subset dim, so the inner connector array can have fewer dims than
        the boundary subset (e.g. a ``B[i+1, j+1:j+9]`` write whose inner
        descriptor is the 1-D ``(W,)`` vector). The returned dim is
        therefore translated into the *collapsed inner-array* coordinates:
        the count of surviving (non-length-1) boundary dims preceding the
        param dim. Without this, a stale boundary index (1 for ``B`` above)
        is out of range for the 1-D inner subset and the lane fallback in
        ``expand_memlet_expression`` silently never widens the write.

        :param state: State holding the NSDFG node.
        :param nsdfg: The nested SDFG node.
        :param vector_map_param: The vectorized map parameter name.
        :returns: ``{connector_name: lane_dim}`` (inner-array coordinates)
            for connectors whose boundary edge carries the param in exactly
            one dim.
        """
        lane_dims: dict = {}
        for edge, conn in ([(e, e.dst_conn) for e in state.in_edges(nsdfg)] + [(e, e.src_conn)
                                                                               for e in state.out_edges(nsdfg)]):
            if conn is None or edge.data is None or edge.data.subset is None:
                continue
            boundary_subset = edge.data.subset
            param_dims = []
            for d, (b, _e, _s) in enumerate(boundary_subset):
                if vector_map_param in free_symbol_names(b):
                    param_dims.append(d)
            if len(param_dims) != 1:
                continue
            param_dim = param_dims[0]
            inner = nsdfg.sdfg.arrays.get(conn)
            inner_ndim = len(inner.shape) if inner is not None else len(boundary_subset)
            if inner_ndim < len(boundary_subset):
                kept_before = 0
                for d, (b, e, s) in enumerate(boundary_subset):
                    if d == param_dim:
                        break
                    length = e - b + 1
                    try:
                        if dace.symbolic.simplify(length) != 1:
                            kept_before += 1
                    except (TypeError, ValueError, AttributeError):
                        kept_before += 1
                lane_dims[conn] = kept_before
            else:
                lane_dims[conn] = param_dim
        return lane_dims

    def _vectorize_nested_sdfg(self, state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG, vector_map_param: str):
        """Vectorize the body of a nested SDFG sitting inside a vectorized map.

        Reconciles connector shapes, reshapes transients to the vector width,
        packs indirect accesses, duplicates interstate symbols per lane, and
        rewrites memlets and tasklets to vector form.

        :param state: the state containing the nested SDFG.
        :param nsdfg: the nested SDFG node.
        :param vector_map_param: the vectorized map parameter name.
        :returns: the set of array names handled as unstructured (already packed).
        :raises Exception: if a scalar source and sink both remain after reduction lift.
        """
        inner_sdfg: dace.SDFG = nsdfg.sdfg
        # Imagine the case where
        # On vectorization of interstate edges (Step 3):
        # We do A[idx[i]]
        # DaCe generates this as | State 1 | -(sym = idx[i])-> | State 2 code = A[sym] ... |
        # This means when we vectorize access to 8 elements then we need to access:
        # idx[i:i+8]
        # And access A accordingly: A[idx[i], idx[i+1], ..., idx[i+8]]
        # Since we can't improve this we need to load the individual sym1, sym2, ..., sym8 individually

        # First we track which inputs have the shape of the vector unit
        # We can copy all of them in the state e.g. idx[i:i+8]
        # Scalars need to be skipped (input subset is still a scalar)
        # Views that have different dimensions have indirect accesses
        # We need to load them individually, for that we need to add "A_packed"
        # and anytime we detect a load from A we need load the 8 elements accordingly

        # Any time we find a load from A[sym] we need to expand load to sym1, sym2, sym3, sym4 (need to find the sym assignment before)
        # and extend that state

        # Step 1. Analyze
        # 0. Try to demote some symbols into scalars
        # 1.1. Detect input and output shapes
        # 1.1.1 Make all non-transient data within the nested SDFG match the connector shapes.
        # 1.1.2 All transient arrays should match the vector unit shape
        # 1.2 Detect sink and source scalars (non-transient scalar access nodes without out_edges or without in_edges)
        # 1.3 Scalar sources are not supported (because duplicating input scalar data results in a different program), raise Error
        # 1.3.1 Unless has flops in that case we can move it, if it involved in the last floating point operation
        # 1.4 Scalar sinks are supported as they can be de-duplicated when writing, track them
        # 1.5 Detect indirect accesses that need a packed intermediate storage
        # 1.6 Each access parameter needs to becomes is own array make its own array

        # After replacing all arrays to match, vectorize:
        # Step 2. Duplicate all interstate symbols to respect lane-ids
        # 2.1 Generate packed loads
        # Step 3. For all scalar sink nodes de-duplicate writes
        # Step 4. Replace all memlet subsets and names
        # Step 5. Replace all tasklets to use vectorized types
        # Step 6. Collect all data used, make sure their type matches the vector op type

        # 0
        # Collect all assigned symbols, check what we can demote
        if self.try_to_demote_symbols_in_nsdfgs:
            demoted_scalars = try_demoting_vectorizable_symbols(inner_sdfg)

        # 1.1.1
        fix_nsdfg_connector_array_shapes_mismatch(state, nsdfg, vector_width=int(self.vector_width))
        ConvertLengthOneArraysToScalars(recursive=True, transient_only=True).apply_pass(inner_sdfg, {})

        # 1.1.2
        transient_arrays = {arr_name for arr_name, arr in inner_sdfg.arrays.items() if arr.transient}
        # ``_iter_mask`` arrays attached by ``GenerateIterationMask`` (P3) are
        # already correctly typed (``bool[W]``) and must NOT be reshaped or
        # have their dtype rewritten to ``vector_op_numeric_type`` (which
        # would convert ``bool`` to ``float64`` and break the masked emitter
        # contract). Exclude them from the W-wide-transient set below.
        transient_arrays = {n for n in transient_arrays if n != "_iter_mask" and not n.startswith("_iter_mask_")}
        vector_width_transient_arrays = {
            arr_name
            for arr_name in transient_arrays if inner_sdfg.arrays[arr_name].shape == (self.vector_width, )
        }
        if self.try_to_demote_symbols_in_nsdfgs:
            for k in demoted_scalars:
                assert k in transient_arrays
        if self.try_to_demote_symbols_in_nsdfgs:
            replace_arrays_with_new_shape(inner_sdfg, demoted_scalars, (self.vector_width, ), None)

        # If a scalar is only used on an interstate edge we have a pattern such as:
        # scalar1 = i + scalar0
        # A[scalar1]
        # This vectorizable as A[scalar1:scalar1+8] (vector width = 8)
        # This would be not vectorizable if it would be:
        # scalar1 = array1[i]
        # Then would need to duplicate the accesses as we can't
        # guarantee that the resulting access will be contigous

        # Get scalars
        scalars = {arr_name for arr_name, arr in inner_sdfg.sdfg.arrays.items() if isinstance(arr, dace.data.Scalar)}

        scalars_only_used_on_interstate_edges = {
            scalar
            for scalar in scalars if not cutil.array_is_used_in_sdfg_states(inner_sdfg, set(), scalar, True)
        }
        # If a scalar is transient + invariant or passed from parent also as a scalar and not as an array
        # Then we can use the same variable across all lanes
        invariant_scalars = {
            scalar
            for scalar in scalars_only_used_on_interstate_edges
            if inner_sdfg.arrays[scalar].transient or self.parent_connection_is_scalar(state, nsdfg, scalar)
        }

        # Do not make scalars used only interstate edges into vector-width arrays
        non_vector_width_transient_arrays = transient_arrays - (vector_width_transient_arrays.union(invariant_scalars))
        replace_arrays_with_new_shape(inner_sdfg, vector_width_transient_arrays, (self.vector_width, ),
                                      self.vector_op_numeric_type)
        replace_arrays_with_new_shape(inner_sdfg, non_vector_width_transient_arrays, (self.vector_width, ), None)

        inner_sdfg.reset_cfg_list()

        vector_width_arrays = {
            arr_name
            for arr_name, arr in inner_sdfg.arrays.items()
            if isinstance(arr, dace.data.Array) and arr.shape == (self.vector_width, )
        }
        scalars = {
            arr_name
            for arr_name, arr in inner_sdfg.arrays.items()
            if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array) and arr.shape == (1, ))
        }
        state.sdfg.validate()

        # 1.2
        scalar_source_nodes: List[Tuple[dace.SDFGState, dace.nodes.AccessNode]] = get_scalar_source_nodes(
            inner_sdfg, True, invariant_scalars)
        array_source_nodes: List[Tuple[dace.SDFGState,
                                       dace.nodes.AccessNode]] = get_array_source_nodes(inner_sdfg, True)
        scalar_sink_nodes: List[Tuple[dace.SDFGState,
                                      dace.nodes.AccessNode]] = get_scalar_sink_nodes(inner_sdfg, True, set())
        array_sink_nodes: List[Tuple[dace.SDFGState, dace.nodes.AccessNode]] = get_array_sink_nodes(inner_sdfg, True)

        # 1.3 and 1.3.1
        unstructured_data = set()
        if len(scalar_source_nodes) == 1 and len(scalar_sink_nodes) == 1:
            has_reduction, _, _ = move_out_reduction(scalar_source_nodes, state, nsdfg, inner_sdfg, self.vector_width)
            # Reduction src and dst are not unstructured, no need to add them
            # Moving out reduction changes the source and sink nodes
            if has_reduction:
                scalar_source_nodes: List[Tuple[dace.SDFGState,
                                                dace.nodes.AccessNode]] = get_scalar_source_nodes(inner_sdfg, True)
                scalar_sink_nodes: List[Tuple[dace.SDFGState,
                                              dace.nodes.AccessNode]] = get_scalar_sink_nodes(inner_sdfg, True, set())
                array_source_nodes: List[Tuple[dace.SDFGState,
                                               dace.nodes.AccessNode]] = get_array_source_nodes(inner_sdfg, True)
                array_sink_nodes: List[Tuple[dace.SDFGState,
                                             dace.nodes.AccessNode]] = get_array_sink_nodes(inner_sdfg, True)

        # No scalar sink nodes should be left at this point (should be either vectors or gone)
        if len(scalar_source_nodes) > 0 and len(scalar_sink_nodes) > 0:
            raise Exception(
                f"Pass tried to lift a reduction within the nested SDFG to enable auto-vectorization but failed. remainign sink nodes: {scalar_sink_nodes}, remaining scalar source nodes: {scalar_source_nodes}"
            )

        # 1.5
        # Generate subset to packed array name map
        # This analysis needs to be more detailed
        # Consider x = A[0, 0, _for_it_52]
        # This can be vectorized but the input shape will not be the (1,) or (vector_width,)
        # use the utility function that returns the accesses that are vectorizable:
        # vectorizable access means that all subets to an array depends purely on constants or loop parameters
        # And the loop parameter is not involved in a multiplication, otherwise wee need to pack it:
        # Consider loop (i=0; i<N; i++) and accessing an array [i*2] this means we have stride-2 access and this also
        # needs to be packed
        vectorizable_arrays_dict = collect_vectorizable_arrays(inner_sdfg, nsdfg, state, invariant_scalars,
                                                               int(self.vector_width))
        non_vectorizable_arrays = {k for k, v in vectorizable_arrays_dict.items() if v is False} - invariant_scalars

        non_vectorizable_array_descs = [(arr_name, inner_sdfg.arrays[arr_name]) for arr_name in non_vectorizable_arrays]
        non_vectorizable_array_infos = {(PackedNameScheme.make(arr_name), (self.vector_width, ),
                                         self.vector_input_storage, arr.dtype)
                                        for arr_name, arr in non_vectorizable_array_descs}
        if self.try_to_demote_symbols_in_nsdfgs:
            for k in demoted_scalars:
                assert k in vectorizable_arrays_dict

        add_transient_arrays_from_list(inner_sdfg, non_vectorizable_array_infos)

        modified_nodes: Set[dace.nodes.Node] = set()
        modified_edges: Set[Edge[Memlet]] = set()

        # 2 and 2.1
        new_mn, new_me, packed_data = self._generate_loads_to_packed_storage(inner_sdfg, non_vectorizable_arrays,
                                                                             vector_width_arrays, vector_map_param)
        modified_nodes = modified_nodes.union(new_mn)
        modified_edges = modified_edges.union(new_me)

        # Bookkeep unstructured data to avoid copying twice
        unstructured_data = unstructured_data.union(packed_data)

        # 3
        check_writes_to_scalar_sinks_happen_through_assign_tasklets(inner_sdfg, scalar_sink_nodes)
        new_mn, new_me = self._duplicate_unstructured_writes(inner_sdfg, non_vectorizable_arrays, vector_map_param)
        modified_nodes = modified_nodes.union(new_mn)
        modified_edges = modified_edges.union(new_me)

        # 4
        for inner_state in inner_sdfg.all_states():
            # Skip the data data that are still scalar and source nodes
            # Do not replace invariant scalars either
            scalar_source_data = {n.data for _, n in scalar_source_nodes}.union(invariant_scalars)
            array_data = {n.data for _, n in array_source_nodes}.union({n.data for _, n in array_sink_nodes})
            # Scalar source nodes can't be replaced.
            # These are non-transient scalars of the SDFG as we do not know how to expand them.
            # If it was a scalar view of an array that could be expanded, that should have been done before
            # by the previous steps
            # If it is an array it will be done in 4.1
            edges_to_replace = {
                edge
                for edge in inner_state.edges() if edge not in modified_edges and edge.data is not None
                and edge.data.data not in scalar_source_data and edge.data.data not in array_data
            }
            old_subset = dace.subsets.Range([(0, 0, 1)])
            new_subset = dace.subsets.Range([(dace.symbolic.SymExpr(0), dace.symbolic.SymExpr(self.vector_width) - 1,
                                              dace.symbolic.SymExpr(1))])
            replace_memlet_expression(inner_state, edges_to_replace, old_subset, new_subset, True, modified_edges,
                                      self.vector_op_numeric_type)

        # 4.1 Do it for arrays
        # Expand memlets accessing arrays, consider nested SDFG receives A[0:2, 0:2, 0:X] as a view
        # and within the nested SDFG we access A[0, 0, for_it_0] (map parameter), then this is vectorizable
        # and should be expanded to A[0:1, 0:1, for_it_0:for_it_0+8] (exclusive range).
        # This should be performed only for arrays.
        #
        # When the body access consumed the map param into a length-1
        # connector view (``bb[0, 0]`` for an outer ``bb[i, j]`` with ``i``
        # innermost), the inner subset no longer names the lane param, so
        # ``expand_memlet_expression`` can't tell which dim is the lane dim
        # and would fall back to the storage-stride-1 dim — wrong for a
        # non-contiguous vectorised access. Recover the lane dim per
        # connector from the NSDFG boundary edge (the dim whose ``begin``
        # carries ``vector_map_param``).
        connector_lane_dim = self._boundary_lane_dims(state, nsdfg, vector_map_param)
        for inner_state in inner_sdfg.all_states():
            array_data = {n.data for _, n in array_source_nodes}.union({n.data for _, n in array_sink_nodes})
            readwrite_data = set()
            for n in inner_state.nodes():
                # Step 4.1 expands ARRAY memlets to a W-wide view ("This
                # should be performed only for arrays"). A non-transient
                # Scalar / length-1 source (e.g. the loop-invariant kernel
                # arg ``c`` in ``a[i,j] > c``, which the ITE-CFG arm
                # clone leaves with an in+out edge) is not an array view:
                # widening its 1-element memlet to ``[0:W]`` reads past it
                # (OOB) and it cannot be reshaped (parent-fed connector).
                # It must stay ``[0]`` and be broadcast. Mirror step 4's
                # scalar exclusion.
                if not isinstance(n, dace.nodes.AccessNode):
                    continue
                desc = inner_state.sdfg.arrays[n.data]
                if (inner_state.in_degree(n) > 0 and inner_state.out_degree(n) > 0 and desc.transient is False
                        and isinstance(desc, dace.data.Array) and str(desc.total_size) != "1"):
                    readwrite_data.add(n.data)
            for rw in readwrite_data:
                array_data.add(rw)
            edges_to_replace = {
                edge
                for edge in inner_state.edges()
                if edge not in modified_edges and edge.data is not None and edge.data.data in array_data
            }
            expand_memlet_expression(inner_state, edges_to_replace, modified_edges, self.vector_width, vector_map_param)

        # Extend interstate edges for all symbols used in tasklets / or interstate edges that access vectorized data
        # There two types of doing this, assume the map parameters are (i, j) and we vectorize over j with vector simd length > 2
        # The map parameter used in the loop is expanded via: _sym1 = j -> _sym1_0 = j, _sym1_1 = j + 1, ...
        # The others are duplicated (for convenience) _sym1 = i -> _sym1_0 = i, _sym1_1 = i, ...
        # `sym = 0`
        # Would become
        # `sym_laneid_0 = 0, sym=sym_laneid_0, sym_laneid_1 = 0, sym_laneid_2 = 0, ....`
        # Assume:
        # `sym = A[_for_it] + 1`
        # Would become:
        # `sym_laneid_0 = A[_for_it + 0] + 1`, `sym = sym_laneid_0`, `sym_laneid_1 = A[_for_it + 1] + 1`, ...
        expand_interstate_assignments_to_lanes(inner_sdfg, nsdfg, state, self.vector_width, invariant_scalars,
                                               vector_map_param)

        # 5
        for inner_state in inner_sdfg.all_states():
            nodes = {n for n in inner_state.nodes() if n not in modified_nodes}
            self._replace_tasklets_from_node_list(inner_state, nodes, vector_map_param)
            modified_nodes = modified_nodes.union(nodes)

        # Add missing symbols
        # There might be missing expanded loop symbols, they are of form `loop_var{id}` where `{id}` is an integer
        # Construct back the loop variable and add assignments for them
        resolve_missing_laneid_symbols(inner_sdfg, nsdfg, state, vector_map_param)

        # Fix in connectors, if scalar it might be already set to float64, but not it should be None
        # If the connector type is not void (typeclass(None)) then it means it was prob. set to its scalar type before,
        # reset it
        reset_connectors(inner_sdfg, nsdfg)

        return unstructured_data

    def _duplicate_unstructured_writes(self, inner_sdfg: dace.SDFG, non_vectorizable_arrays: Set[str],
                                       vector_map_param: str):
        """Fan out writes to non-vectorizable array sinks into one access per lane.

        :param inner_sdfg: the nested SDFG being vectorized.
        :param non_vectorizable_arrays: array names that must be packed/duplicated.
        :param vector_map_param: the vectorized map parameter name.
        :returns: the modified nodes and modified edges as a pair of sets.
        :raises Exception: if a write to a non-transient scalar sink remains.
        """
        modified_edges = set()
        modified_nodes = set()
        for state in inner_sdfg.all_states():
            for node in state.nodes():
                if state.out_degree(node) == 0:
                    arr = state.sdfg.arrays[node.data]
                    if (arr.transient is False and
                        (isinstance(arr, dace.data.Scalar) or isinstance(arr, dace.data.Array) and arr.shape == (1, ))):
                        # If it is a reduction tasklet + number of edges matching vector unit it is ok
                        srcs = {ie.src for ie in state.in_edges(node)}
                        if not (len(srcs) == 1 and state.in_degree(next(iter(srcs))) == self.vector_width
                                and isinstance(next(iter(srcs)), dace.nodes.Tasklet)):
                            raise Exception(
                                "At this point of the pass, no write to non-transient scalar sinks should remain")
                    if arr.transient is False and (isinstance(arr, dace.data.Array) and
                                                   (arr.shape != (1, ) and arr.shape != (self.vector_width, ))
                                                   and node.data in non_vectorizable_arrays):
                        touched_nodes, touched_edges = duplicate_access(state, node, self.vector_width,
                                                                        vector_map_param)
                        modified_edges = modified_edges.union(touched_edges)
                        modified_nodes = modified_nodes.union(touched_nodes)
        return modified_nodes, modified_edges

    def _generate_loads_to_packed_storage(
            self, sdfg: dace.SDFG, array_accessed_to_be_packed: Set[str], candidate_arrays: Set[str],
            vector_map_param: str) -> Tuple[Set[dace.nodes.Node], Set[Edge[Memlet]], Set[str]]:
        """Build per-lane gather loads into packed transient storage for indirect arrays.

        :param sdfg: the nested SDFG being vectorized.
        :param array_accessed_to_be_packed: array names needing packed storage.
        :param candidate_arrays: arrays that may be promoted to vector width.
        :param vector_map_param: the vectorized map parameter name.
        :returns: modified nodes, modified edges, and the packed data names.
        """
        modified_nodes: Set[dace.nodes.Node] = set()
        modified_edges: Set[Edge[Memlet]] = set()
        expanded_symbols = set()
        modified_data = set()

        # First expand intersate assignments
        for state in sdfg.all_states():
            for edge in state.edges():
                if edge.data is not None and edge.data.data in array_accessed_to_be_packed:
                    free_symbols = edge.data.free_symbols
                    # Look for the assignments in the interstate edges and expand them
                    non_expanded_free_symbols = free_symbols - expanded_symbols
                    expanded_symbols = expanded_symbols.union(free_symbols)
                    self._expand_interstate_assignments(sdfg, non_expanded_free_symbols, candidate_arrays)

        # Build packed gather buffers. An array may be gathered MULTIPLE
        # times in one state with DISTINCT index expressions (e.g. the
        # ICON cell-from-edges interpolation reads
        # ``z_kin_hor_e[edge_blk_m, jk, edge_idx_m]`` for m = 0,1,2).
        # Group the consuming edges by their gather subset: identical
        # gathers share ONE packed buffer + one set of per-lane loads
        # (dedup by subset key -> no codegen bloat); each DISTINCT gather
        # gets its own packed buffer and EVERY consuming edge of it is
        # rewritten. The previous code renamed the one source AccessNode
        # in place and rewrote a single edge, so a second distinct
        # gather of the same array was left with a stale ``<arr>`` memlet
        # on the renamed source -> ``InvalidSDFGEdgeError: Memlet data
        # does not match source``.
        #
        # Two naming invariants the masked-remainder collapse relies on
        # (``utils.lane_fanout``): (1) per-lane tasklet label must match
        # ``^assign_<digits>$`` (``_ASSIGN_LABEL_RE``); (2) the packed
        # buffer name must END with ``_packed``
        # (``PackedNameScheme.is_packed`` == ``endswith``). The n>0
        # buffer therefore uses ``PackedNameScheme.make(f"{arr}_{n}")``
        # -> ``<arr>_<n>_packed`` (still ends ``_packed``), NOT
        # ``<arr>_packed_<n>``. ``repl_subset_to_use_laneid_offset``
        # already lane-expands ALL gathered index components in the
        # subset, so a multi-component (two index-table) gather is
        # handled for free.
        for state in sdfg.all_states():
            groups = {}
            orig_src_nodes = {}
            for edge in list(state.edges()):
                if edge.data is None or edge.data.data not in array_accessed_to_be_packed:
                    continue
                # Skip edges sourced from non-AccessNodes (e.g. the CPP
                # ``_iter_mask_fill`` tasklet emitted by P3).
                if not isinstance(edge.src, dace.nodes.AccessNode):
                    continue
                if not (state.in_degree(edge.src) == 0 and edge.src.data == edge.data.data):
                    continue
                groups.setdefault((edge.data.data, str(edge.data.subset)), []).append(edge)
                orig_src_nodes.setdefault(edge.data.data, edge.src)

            per_array_idx = {}
            for (data_name, _subset_key), edges in groups.items():
                n = per_array_idx.get(data_name, 0)
                per_array_idx[data_name] = n + 1
                # n == 0 reuses the pre-created ``<arr>_packed``; extra
                # distinct gathers get ``<arr>_<n>_packed`` (ends
                # ``_packed`` so PackedNameScheme.is_packed holds).
                packed_name = (PackedNameScheme.make(data_name)
                               if n == 0 else PackedNameScheme.make(f"{data_name}_{n}"))
                if packed_name not in sdfg.arrays:
                    base = sdfg.arrays[data_name]
                    sdfg.add_array(packed_name, (self.vector_width, ),
                                   base.dtype,
                                   storage=self.vector_input_storage,
                                   transient=True,
                                   find_new_name=False)
                modified_data.add(data_name)
                modified_data.add(packed_name)

                packed_src = state.add_access(packed_name)
                non_packed_access = state.add_access(data_name)
                non_packed_access.setzero = True
                modified_nodes.add(packed_src)
                modified_nodes.add(non_packed_access)

                # A -[j]-> B becomes
                # A -[j_laneid_0..j_laneid_{W-1}]-> A_packed -[0:W]-> B.
                # Tasklet label MUST stay ``assign_{i}`` (lane-fanout
                # collapse matcher); duplicate labels across distinct
                # packed fans are fine -- the matcher scopes to one
                # packed node's neighbour tasklets.
                rep_subset = edges[0].data.subset
                for i in range(self.vector_width):
                    new_subset = repl_subset_to_use_laneid_offset(sdfg=state.sdfg,
                                                                  subset=copy.deepcopy(rep_subset),
                                                                  symbol_offset=str(i),
                                                                  vector_map_param=vector_map_param)
                    at = state.add_tasklet(name=f"assign_{i}", inputs={"_in"}, outputs={"_out"}, code="_out = _in")
                    at.add_in_connector("_in")
                    at.add_out_connector("_out")
                    e1 = state.add_edge(non_packed_access, None, at, "_in",
                                        dace.memlet.Memlet(data=data_name, subset=new_subset))
                    e2 = state.add_edge(at, "_out", packed_src, None, dace.memlet.Memlet(f"{packed_name}[{i}]"))
                    modified_nodes.add(at)
                    if isinstance(e1, dace.nodes.Node) or isinstance(e2, dace.nodes.Node):
                        raise RuntimeError(f"state.add_edge returned a Node for {packed_name}; "
                                           "SDFG API contract broken")
                    modified_edges.add(e1)
                    modified_edges.add(e2)

                # Rewrite EVERY consuming edge of this distinct gather to
                # read the packed buffer (not just the first one).
                for e in edges:
                    dst, dst_conn = e.dst, e.dst_conn
                    state.remove_edge(e)
                    ne = state.add_edge(packed_src, None, dst, dst_conn,
                                        dace.memlet.Memlet(expr=f"{packed_name}[0:{self.vector_width}]"))
                    if isinstance(ne, dace.nodes.Node):
                        raise RuntimeError(f"state.add_edge returned a Node for {packed_name} consumer; "
                                           "SDFG API contract broken")
                    modified_edges.add(ne)

            # The original gather source nodes lost all their consumers
            # (every out-edge was rewritten to a packed buffer); drop the
            # now-isolated nodes.
            for src in set(orig_src_nodes.values()):
                if src in state.nodes() and state.degree(src) == 0:
                    state.remove_node(src)
        return modified_nodes, modified_edges, modified_data

    def _generate_strided_loads_to_packed_storage(
            self, sdfg: dace.SDFG, state: dace.SDFGState, symbols_to_offset: Set[str], map_entry: dace.nodes.MapEntry,
            array_accessed_to_be_packed: Set[str]) -> Tuple[set[dace.nodes.Node], Set[Edge[Memlet]]]:
        """Replace strided array loads inside a map with per-lane gathers into packed storage.

        :param sdfg: the SDFG containing the map.
        :param state: the state containing the map.
        :param symbols_to_offset: symbols to offset per lane (the map params).
        :param map_entry: the vectorized map entry.
        :param array_accessed_to_be_packed: array names needing packed storage.
        :returns: the modified nodes and modified edges as a pair of sets.
        """

        all_nodes = state.all_nodes_between(map_entry, state.exit_node(map_entry))

        modified_nodes = set()
        modified_edges = set()

        array_loads_to_be_packed = set()

        for edge in state.all_edges(*all_nodes):
            if edge.data.data in array_accessed_to_be_packed and (isinstance(edge.dst, dace.nodes.AccessNode)
                                                                  or isinstance(edge.src, dace.nodes.MapEntry)):
                array_loads_to_be_packed.add(edge.data.data)

        # Then do the other stuff
        for edge in state.all_edges(*all_nodes):
            if edge.data is not None and edge.data.data in array_loads_to_be_packed:
                # Create a packed copy
                # If it is a source node (no in edges, copy in), otherwise replace with the packed data
                src = edge.src

                # Distinguish between the first access and consequent accesses
                if edge.data.data in array_loads_to_be_packed:
                    if src == map_entry:
                        new_data_name = f"{edge.data.data}_packed"
                        old_data_name = f"{edge.data.data}"

                        #non_packed_access = state.add_access(old_data_name)
                        packed_access = state.add_access(new_data_name)
                        packed_access.setzero = True
                        modified_nodes.add(packed_access)

                        if new_data_name not in sdfg.arrays:
                            sdfg.add_array(name=new_data_name,
                                           shape=(self.vector_width, ),
                                           dtype=self.vector_op_numeric_type,
                                           storage=dace.dtypes.StorageType.Register,
                                           transient=True)

                        # Replace all symbols used e.g. i,j with `i`, `j_laneid_0`
                        # A -[j]> B becomes now
                        # A -[j_laneid_0,j_laneid_1,...,j_laneid_7]-> A_packed -[0:8]-> B
                        for i in range(self.vector_width):
                            new_subset = repl_subset_to_use_with_int_offset(sdfg=state.sdfg,
                                                                            symbols_to_offset=symbols_to_offset,
                                                                            subset=copy.deepcopy(edge.data.subset),
                                                                            int_offset=i)
                            at = state.add_tasklet(
                                name=f"assign_{i}",
                                inputs={
                                    "_in",
                                },
                                outputs={
                                    "_out",
                                },
                                code="_out = _in",
                            )
                            at.add_in_connector("_in")
                            at.add_out_connector("_out")

                            e1 = state.add_edge(edge.src, edge.src_conn, at, "_in",
                                                dace.memlet.Memlet(
                                                    data=old_data_name,
                                                    subset=new_subset,
                                                ))
                            e2 = state.add_edge(at, "_out", packed_access, None,
                                                dace.memlet.Memlet(f"{new_data_name}[{i}]"))

                            modified_edges.add(e1)
                            modified_edges.add(e2)
                            modified_nodes.add(at)

                        e3 = state.add_edge(packed_access, None, edge.dst, edge.dst_conn,
                                            dace.memlet.Memlet(f"{new_data_name}[0:{self.vector_width}]"))
                        modified_edges.add(e3)

                        state.remove_edge(edge)
                        modified_edges.add(edge)
                    else:
                        new_data_name = f"{edge.data.data}_packed"
                        old_data_name = f"{edge.data.data}"

                        edge.data = dace.memlet.Memlet(data=new_data_name, subset=copy.deepcopy(edge.data.subset))

                        if isinstance(edge.src, dace.nodes.AccessNode) and edge.src.data == old_data_name:
                            edge.src.data = new_data_name
                        if isinstance(edge.dst, dace.nodes.AccessNode) and edge.dst.data == old_data_name:
                            edge.dst.data = new_data_name

        return modified_nodes, modified_edges

    def _generate_strided_stores_from_packed_storage(
            self, sdfg: dace.SDFG, state: dace.SDFGState, symbols_to_offset: Set[str], map_entry: dace.nodes.MapEntry,
            array_accessed_to_be_packed: Set[str]) -> Tuple[set[dace.nodes.Node], Set[Edge[Memlet]]]:
        """Replace strided array stores inside a map with per-lane scatters from packed storage.

        :param sdfg: the SDFG containing the map.
        :param state: the state containing the map.
        :param symbols_to_offset: symbols to offset per lane (the map params).
        :param map_entry: the vectorized map entry.
        :param array_accessed_to_be_packed: array names needing packed storage.
        :returns: the modified nodes and modified edges as a pair of sets.
        """

        all_nodes = state.all_nodes_between(map_entry, state.exit_node(map_entry))

        modified_nodes = set()
        modified_edges = set()

        array_stores_to_be_packed = set()

        for edge in state.all_edges(*all_nodes):
            if edge.data.data in array_accessed_to_be_packed and isinstance(edge.dst, dace.nodes.MapExit):
                array_stores_to_be_packed.add(edge.data.data)

        map_exit = state.exit_node(map_entry)

        # Then do the other stuff
        for edge in state.all_edges(*all_nodes):
            if edge.data is not None and edge.data.data in array_stores_to_be_packed:
                # Create a packed copy
                # If it is a source node (no in edges, copy in), otherwise replace with the packed data
                dst = edge.dst

                # Distinguish between the first access and consequent accesses
                if edge.data.data in array_stores_to_be_packed:
                    if dst == map_exit:
                        new_data_name = f"{edge.data.data}_packed"
                        old_data_name = f"{edge.data.data}"

                        #non_packed_access = state.add_access(old_data_name)
                        packed_access = state.add_access(new_data_name)
                        packed_access.setzero = True
                        modified_nodes.add(packed_access)

                        if new_data_name not in sdfg.arrays:
                            sdfg.add_array(name=new_data_name,
                                           shape=(self.vector_width, ),
                                           dtype=self.vector_op_numeric_type,
                                           storage=dace.dtypes.StorageType.Register,
                                           transient=True)

                        # Replace all symbols used e.g. i,j with `i`, `j_laneid_0`
                        # A -[j]> B becomes now
                        # A -[j_laneid_0,j_laneid_1,...,j_laneid_7]-> A_packed -[0:8]-> B
                        for i in range(self.vector_width):
                            new_subset = repl_subset_to_use_with_int_offset(sdfg=state.sdfg,
                                                                            symbols_to_offset=symbols_to_offset,
                                                                            subset=copy.deepcopy(edge.data.subset),
                                                                            int_offset=i)
                            at = state.add_tasklet(
                                name=f"assign_{i}",
                                inputs={
                                    "_in",
                                },
                                outputs={
                                    "_out",
                                },
                                code="_out = _in",
                            )
                            at.add_in_connector("_in")
                            at.add_out_connector("_out")

                            e1 = state.add_edge(packed_access, None, at, "_in",
                                                dace.memlet.Memlet(f"{new_data_name}[{i}]"))

                            e2 = state.add_edge(at, "_out", edge.dst, edge.dst_conn,
                                                dace.memlet.Memlet(
                                                    data=old_data_name,
                                                    subset=new_subset,
                                                ))
                            modified_edges.add(e1)
                            modified_edges.add(e2)
                            modified_nodes.add(at)

                        e3 = state.add_edge(edge.src, edge.src_conn, packed_access, None,
                                            dace.memlet.Memlet(f"{new_data_name}[0:{self.vector_width}]"))
                        modified_edges.add(e3)

                        state.remove_edge(edge)
                        modified_edges.add(edge)
                    else:
                        new_data_name = f"{edge.data.data}_packed"
                        old_data_name = f"{edge.data.data}"

                        edge.data = dace.memlet.Memlet(data=new_data_name, subset=copy.deepcopy(edge.data.subset))

                        if isinstance(edge.src, dace.nodes.AccessNode) and edge.src.data == old_data_name:
                            edge.src.data = new_data_name
                        if isinstance(edge.dst, dace.nodes.AccessNode) and edge.dst.data == old_data_name:
                            edge.dst.data = new_data_name

        return modified_nodes, modified_edges

    def _expand_interstate_assignment(self, sdfg: dace.SDFG, edge: Edge[InterstateEdge], syms: Set[str],
                                      candidate_arrays: Set[str]):
        """Fan out interstate symbol assignments on one edge into one per lane.

        :param sdfg: the SDFG being vectorized.
        :param edge: the interstate edge whose assignments are expanded.
        :param syms: the symbol names to fan out.
        :param candidate_arrays: arrays referenced in assignments to index per lane.
        :returns: the set of duplicated symbol names.
        """
        duplicated_symbols = set()
        syms_to_rm = set()
        new_assignments = dict()
        # Lets say we have
        # k = idx
        # then we need to do:
        # k0 = idx[0]
        # k1 = idx[1]
        # ...
        # k7 = idx[7]
        # Also need to consider the case where multiple symbols are involved
        # k0 = idx[0] + idy[0]
        for k, v in edge.data.assignments.items():
            if k in syms:

                for i in range(0, self.vector_width):
                    # Get all scalar accesses from v and replace with the array equivalent
                    # if we have j = k1 + k2
                    # we need to have j0 = k1[0] + k2[0], j1 = k1[1] + k2[1], ...
                    vcopy = copy.deepcopy(v)
                    nv = vcopy
                    for ca in candidate_arrays:
                        assert ca in sdfg.arrays
                        ca_data = sdfg.arrays[ca]
                        if isinstance(ca_data, dace.data.Scalar) or (isinstance(ca_data, dace.data.Array)
                                                                     and ca_data.shape == (1, )):
                            ca_scl = ca_data
                            assert ca_scl.transient
                            sdfg.remove_data(ca, validate=False)
                            sdfg.add_array(
                                name=ca,
                                shape=(self.vector_width, ),
                                dtype=ca_scl.dtype,
                                storage=ca_scl.storage,
                                location=ca_scl.location,
                                alignment=parse_int_or_default(self.vector_width, 8) * ca_scl.dtype.bytes,
                                transient=True,
                                lifetime=ca_scl.lifetime,
                                find_new_name=False,
                            )
                        nv = tutil.token_replace_dict(nv, {ca: f"{ca}[{i}]"})
                    new_assignments[LaneIdScheme.make_dim(k, 0, i)] = nv
                    if i == 0:
                        new_assignments[k] = nv
                duplicated_symbols.add(k)
                syms_to_rm.add(k)
            else:
                new_assignments[k] = v
        edge.data.assignments = new_assignments
        return duplicated_symbols

    def _expand_interstate_assignments(self, sdfg: dace.SDFG, syms: Set[str], candidate_arrays: Set[str]):
        """Fan out interstate symbol assignments per lane across all interstate edges.

        :param sdfg: the SDFG being vectorized.
        :param syms: the symbol names to fan out.
        :param candidate_arrays: arrays referenced in assignments to index per lane.
        :returns: the set of duplicated symbol names.
        """
        duplicated_symbols = set()
        for edge in sdfg.all_interstate_edges():
            duplicated_symbols = duplicated_symbols.union(
                self._expand_interstate_assignment(sdfg, edge, syms, candidate_arrays))
        return duplicated_symbols

    def _extend_temporary_scalars(self,
                                  state: SDFGState,
                                  map_entry: dace.nodes.MapEntry,
                                  modified_nodes: Set[dace.nodes.Node] = set(),
                                  modified_edges: Set[Edge[Memlet]] = set()):
        """Promote scalar/length-1 transients inside a vectorized map to vector-width arrays.

        :param state: the state containing the map.
        :param map_entry: the vectorized map entry.
        :param modified_nodes: nodes already handled, skipped here.
        :param modified_edges: edges already handled, skipped here.
        """
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        edges_to_rm = set()
        edges_to_add = set()
        nodes_to_rm = set()
        for node in nodes:
            if isinstance(node, dace.nodes.AccessNode):
                if node in modified_nodes:
                    continue
                desc = state.parent_graph.sdfg.arrays[node.data]
                if (isinstance(desc, dace.data.Scalar) or (isinstance(desc, dace.data.Array) and desc.shape == (1, ))):
                    if f"{node.data}_vec" not in state.parent_graph.sdfg.arrays:
                        state.sdfg.add_array(
                            name=f"{node.data}_vec",
                            shape=(self.vector_width, ),
                            dtype=self.vector_op_numeric_type,
                            storage=self.vector_input_storage,
                            transient=True,
                            alignment=parse_int_or_default(self.vector_width, 8) * desc.dtype.bytes,
                            find_new_name=False,
                        )

                new_an = state.add_access(f"{node.data}_vec")
                new_an.setzero = True

                for ie in state.in_edges(node):
                    new_edge_tuple = (ie.src, ie.src_conn, new_an, None, copy.deepcopy(ie.data))
                    edges_to_rm.add(ie)
                    edges_to_add.add(new_edge_tuple)
                for oe in state.out_edges(node):
                    new_edge_tuple = (new_an, None, oe.dst, oe.dst_conn, copy.deepcopy(oe.data))
                    edges_to_rm.add(oe)
                    edges_to_add.add(new_edge_tuple)
                nodes_to_rm.add(node)

        rmed_data_names = set()
        for edge in edges_to_rm:
            state.remove_edge(edge)
        for node in nodes_to_rm:
            state.remove_node(node)
            rmed_data_names.add(node.data)
        for edge_tuple in edges_to_add:
            state.add_edge(*edge_tuple)

        # Refresh nodes
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        for edge in state.all_edges(*nodes):
            if edge in modified_edges:
                continue
            if edge.data is not None and edge.data.data in rmed_data_names:
                new_memlet = dace.memlet.Memlet(
                    data=f"{edge.data.data}_vec",
                    subset=dace.subsets.Range([(dace.symbolic.SymExpr(0), dace.symbolic.SymExpr(self.vector_width) - 1,
                                                dace.symbolic.SymExpr(1))]),
                )
                state.remove_edge(edge)
                state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, new_memlet)

    def _extend_memlets(self,
                        state: SDFGState,
                        map_entry: dace.nodes.MapEntry,
                        modified_edges: Set[Edge[Memlet]] = set()):
        """Extend memlets inside a vectorized map to vector-width slices.

        :param state: the state containing the map.
        :param map_entry: the vectorized map entry.
        :param modified_edges: edges already handled, skipped here.
        """
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        assert not any({
            isinstance(node, dace.nodes.MapEntry)
            for node in nodes
        }), f"No map entry nodes are allowed within the vectorized map entry - this case is not supported yet"
        edges = set(state.all_edges(*nodes)) - modified_edges
        self._extend_memlets_from_node_and_edge_list(state, edges)

    def _extend_memlets_from_node_and_edge_list(self, state: SDFGState, edges: Iterable[Edge[Memlet]]):
        """Extend each edge's param-bearing stride-1 dim to a vector-width slice.

        :param state: the state containing the edges.
        :param edges: the edges whose memlets are rewritten.
        :raises NotImplementedError: if the map param spans multiple dims
            non-pointwise or the param-bearing stride-1 dim has a non-1 step.
        """
        # For each memlet, find the dimension whose range expression contains the inner map param
        # and (if it is stride-1 in the array) extend it to a vector-width slice.
        #
        # Behavior matches the original F/C-driven extension on the common case (param in the
        # contiguous dim) and is a no-op when the param sits in a non-stride-1 dim — those
        # gather-shaped accesses are handled by the strided-load / multiplex path elsewhere; it
        # is *not* this helper's job to refuse them.
        #
        # We keep `raise NotImplementedError` only for cases that are genuinely impossible to
        # interpret: param appearing in two dims at once, or the param-bearing stride-1 dim
        # having a non-1 step.
        for edge in edges:
            memlet: dace.memlet.Memlet = edge.data
            map_entry: dace.nodes.MapEntry = state.entry_node(
                edge.src) if not isinstance(edge.src, dace.nodes.MapEntry) else state.entry_node(edge.dst)
            used_param = map_entry.map.params[-1]
            param_sym = dace.symbolic.symbol(used_param)

            if memlet.subset is None:
                new_memlet = dace.memlet.Memlet(None)
            else:
                arr_strides = state.sdfg.arrays[memlet.data].strides if memlet.data is not None else None
                new_range_list = [(b, e, s) for (b, e, s) in memlet.subset]

                param_dims = []
                for d, (b, e, _) in enumerate(new_range_list):
                    free_syms = free_symbol_names(b) | free_symbol_names(e)
                    if used_param in free_syms:
                        param_dims.append(d)

                if len(param_dims) > 1:
                    # Linear-combination access (e.g. A[i,i], A[2*i,i], A[i,2*i]).
                    # Bare-tasklet path: ``_generate_strided_loads_to_packed_storage``
                    # emits per-lane fan-out, the post-emit ``DetectMultiDimStrided{Load,Store}``
                    # then collapses to a single ``strided_load`` intrinsic.  In that path
                    # the subset stays scalar — the per-lane tasklets carry the
                    # individual ``i`` values via the cloned inner map.
                    #
                    # NSDFG-wrapped path (P1+P2 scalar/masked): the bare-tasklet
                    # fan-out is skipped because ``has_single_nested_sdfg=True``;
                    # ``_setup_strided_nsdfg_edges_inline`` instead emits a
                    # multi-dim ``strided_load`` *inside* the NSDFG. For that to
                    # work the outer memlet must be the **W-lane bbox** so the
                    # connector array gets reshaped to the full window the
                    # diagonal walks.  Expand each param-dim's end via
                    # ``i -> i + (W - 1)`` (same formula as the 1D branch).
                    for d in param_dims:
                        b, e, _ = new_range_list[d]
                        if b != e:
                            raise NotImplementedError(
                                f"Vectorize: multi-dim param {used_param} on a non-point subset "
                                f"(dim {d}: {b}..{e}) of memlet {memlet} on edge {edge} "
                                f"(state {state.label}); only point accesses A[c*i+d, ...] supported")
                    # When the edge crosses an NSDFG boundary (multi-dim
                    # strided rewrite will fire inside the NSDFG), expand
                    # each param-bearing dim to the W-lane bbox.  For the
                    # bare-tasklet path (no NSDFG below the map) leave
                    # subsets scalar so per-lane fan-out keeps working.
                    edge_crosses_nsdfg = (isinstance(edge.dst, dace.nodes.NestedSDFG)
                                          or isinstance(edge.src, dace.nodes.NestedSDFG))
                    if edge_crosses_nsdfg:
                        new_range_list_expanded = list(new_range_list)
                        for d in param_dims:
                            lb, le, ls = new_range_list_expanded[d]
                            if isinstance(le, int):
                                le = dace.symbolic.SymExpr(str(le))
                            new_le = le.subs(param_sym,
                                             dace.symbolic.SymExpr(f"({self.vector_width} - 1) + {used_param}"))
                            new_range_list_expanded[d] = (lb, new_le, ls)
                        new_memlet = dace.memlet.Memlet(data=memlet.data,
                                                        subset=dace.subsets.Range(new_range_list_expanded))
                    else:
                        new_memlet = dace.memlet.Memlet(data=memlet.data, subset=dace.subsets.Range(new_range_list))
                    self._assert_no_other_subset(memlet, edge, state)
                    state.remove_edge(edge)
                    state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, new_memlet)
                    continue

                # Extend only when the param is in a stride-1 dim. If the param sits in a
                # non-stride-1 dim, leave the memlet alone — the gather/strided-load path will
                # handle it (mirrors the original code, which would no-op-substitute in such
                # cases by extending the contiguous dim that did not contain the param).
                non_contig_lane_crosses_nsdfg = (len(param_dims) == 1 and arr_strides is not None
                                                 and arr_strides[param_dims[0]] != 1
                                                 and (isinstance(edge.dst, dace.nodes.NestedSDFG)
                                                      or isinstance(edge.src, dace.nodes.NestedSDFG)))
                if len(param_dims) == 1 and arr_strides is not None and arr_strides[param_dims[0]] == 1:
                    d = param_dims[0]
                    lb, le, ls = new_range_list[d]
                    if isinstance(le, int):
                        le = dace.symbolic.SymExpr(str(le))
                    if ls != 1:
                        raise NotImplementedError(
                            f"Vectorize: param-bearing dim {d} has non-1 step {ls} on memlet {memlet} "
                            f"(edge {edge}, state {state.label})")

                    # Extend the upper bound forward by ``W - 1`` lanes
                    # via ``i -> i + (W - 1)``. Works uniformly for both
                    # point (``lb == le``) accesses like ``a[2*i]`` and
                    # ranged (``lb != le``) accesses like ``a[2*i:2*i+1]``
                    # — in either case the bbox spans the W consecutive
                    # ``i``-iterations the vectorizer is about to compute.
                    new_le = le.subs(param_sym, dace.symbolic.SymExpr(f"({self.vector_width} - 1) + {used_param}"))
                    new_range_list[d] = (lb, new_le, ls)
                elif non_contig_lane_crosses_nsdfg:
                    # The vectorized param sits in a NON-contiguous dim and
                    # the edge crosses an NSDFG boundary (e.g. C-layout
                    # ``bb[i, j]`` with ``i`` innermost, ``strides=(N, 1)``).
                    # Widen that dim to the W-lane bbox (``i -> i + (W-1)``)
                    # so the NSDFG receives ``bb[i:i+W, j]``; the strided-load
                    # path inside the NSDFG then gathers the W lanes at the
                    # array's per-dim stride. Without this the boundary edge
                    # stays a single element and the connector cannot hold
                    # the lane window.
                    d = param_dims[0]
                    lb, le, ls = new_range_list[d]
                    if isinstance(le, int):
                        le = dace.symbolic.SymExpr(str(le))
                    if ls != 1:
                        raise NotImplementedError(
                            f"Vectorize: non-contiguous param dim {d} has non-1 step {ls} on memlet {memlet} "
                            f"(edge {edge}, state {state.label})")
                    new_le = le.subs(param_sym, dace.symbolic.SymExpr(f"({self.vector_width} - 1) + {used_param}"))
                    new_range_list[d] = (lb, new_le, ls)

                new_memlet = dace.memlet.Memlet(
                    data=memlet.data,
                    subset=dace.subsets.Range(new_range_list),
                )

            self._assert_no_other_subset(memlet, edge, state)
            state.remove_edge(edge)
            state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, new_memlet)

    def _replace_tasklets(self, state: SDFGState, map_entry: dace.nodes.MapEntry, vector_map_param: str,
                          modified_nodes: Set[dace.nodes.AccessNode]):
        """Replace scalar tasklets inside a vectorized map with vector-template tasklets.

        :param state: the state containing the map.
        :param map_entry: the vectorized map entry.
        :param vector_map_param: the vectorized map parameter name.
        :param modified_nodes: nodes already handled, skipped here.
        """
        nodes = set(state.all_nodes_between(map_entry, state.exit_node(map_entry))) - modified_nodes
        assert not any({
            isinstance(node, dace.nodes.MapEntry)
            for node in nodes
        }), f"No map entry nodes are allowed within the vectorized map entry - this case is not supported yet"
        self._replace_tasklets_from_node_list(state, nodes, vector_map_param)

    def _replace_tasklets_from_node_list(self, state: SDFGState, nodes: Iterable[dace.nodes.Node],
                                         vector_map_param: str):
        """Rewrite each Python tasklet in the list to its vector-template form.

        Invariant scalar-only tasklets are left as-is; when an iteration-mask
        transient is in scope, a ``_mask`` connector is wired so masked
        template variants can be selected.

        :param state: the state containing the tasklets.
        :param nodes: the candidate nodes to rewrite.
        :param vector_map_param: the vectorized map parameter name.
        """
        # C.2-b: when the inner SDFG has a P3-generated ``_iter_mask: bool[W]``
        # transient in scope, every vectorized tasklet that writes to an array
        # picks up a ``_mask`` input connector wired from that array. The
        # emitter then routes to the ``_masked`` template variant (RMW on the
        # destination so inactive lanes stay untouched). For tasklets whose
        # op has no ``_masked`` template entry, ``_template_key`` falls back
        # to the unsuffixed variant and the dangling ``_mask`` connector is a
        # tolerated dead read.
        iter_mask_name = None
        for name in state.sdfg.arrays:
            if name == "_iter_mask" or name.startswith("_iter_mask_"):
                iter_mask_name = name
                break

        for node in nodes:
            if isinstance(node, dace.nodes.Tasklet):
                # Skip non-Python tasklets: already-emitted CPP intrinsics
                # (vector_<op> bodies, _iter_mask_fill, gather_load /
                # scatter_store intrinsic emissions) are in their final form;
                # classify_tasklet asserts Language.Python and would fail.
                if node.code.language != dace.dtypes.Language.Python:
                    continue
                tasklet_info = tutil.classify_tasklet(state, node)
                ttype: tutil.TaskletType = tasklet_info.get("type")
                # If we still have scalar-scalar or scalar-symbol or symbol-symbol op
                # that is not writing to an array (all length 1 arrays have been made into scalasr before)
                # it means they are invariant and we should keep them as they are
                if ttype in {
                        tutil.TaskletType.SCALAR_SCALAR,
                        tutil.TaskletType.SCALAR_SYMBOL,
                        tutil.TaskletType.SCALAR_SYMBOL,
                        tutil.TaskletType.SYMBOL_SYMBOL,
                        tutil.TaskletType.SCALAR_SCALAR_ASSIGNMENT,
                        tutil.TaskletType.UNARY_SCALAR,
                        tutil.TaskletType.UNARY_SYMBOL,
                        tutil.TaskletType.SCALAR_SYMBOL_ASSIGNMENT,
                }:
                    # If output is not an array
                    oe = state.out_edges(node).pop()
                    if not isinstance(state.sdfg.arrays[oe.data.data], dace.data.Array):
                        continue

                mask_connector_arg = None
                if iter_mask_name is not None and "_mask" not in node.in_connectors:
                    # Declare the connector as a bool pointer so DaCe codegen
                    # emits ``const bool* _mask`` rather than inferring
                    # ``double*`` from the tasklet's other in-connectors.
                    mask_ptr_type = dace.dtypes.pointer(dace.bool_)
                    node.add_in_connector("_mask", dtype=mask_ptr_type, force=True)
                    mask_an = state.add_access(iter_mask_name)
                    state.add_edge(mask_an, None, node, "_mask",
                                   dace.memlet.Memlet(f"{iter_mask_name}[0:{self.vector_width}]"))
                    mask_connector_arg = "_mask"
                elif iter_mask_name is not None:
                    mask_connector_arg = "_mask"

                instantiate_tasklet_from_info(state,
                                              node,
                                              tasklet_info,
                                              self.vector_width,
                                              self.templates,
                                              vector_map_param,
                                              self.vector_op_numeric_type,
                                              mask_connector=mask_connector_arg)

    def _offset_all_memlets(self, state: SDFGState, map_entry: dace.nodes.MapEntry, dataname: str, new_dataname: str):
        """Rewrite all memlets on ``dataname`` inside the map to the vector copy.

        :param state: the state containing the map.
        :param map_entry: the vectorized map entry.
        :param dataname: the original array name.
        :param new_dataname: the vector-copy array name.
        """
        nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        assert not any({
            isinstance(node, dace.nodes.MapEntry)
            for node in nodes
        }), f"No map entry nodes are allowed within the vectorized map entry - this case is not supported yet"
        edges = state.all_edges(*nodes)
        self._offset_memlets_from_edge_list(state, edges, dataname, new_dataname)

    def _offset_memlets_from_edge_list(self, state: SDFGState, edges: Iterable[Edge[Memlet]], dataname: str,
                                       new_dataname: str):
        """Rewrite matching memlets in the edge list to the vector copy.

        :param state: the state containing the edges.
        :param edges: the edges to inspect.
        :param dataname: the original array name to match.
        :param new_dataname: the vector-copy array name.
        """
        for edge in edges:
            if edge.data.data is None or edge.data.data != dataname:
                continue
            memlet: dace.memlet.Memlet = edge.data

            assert memlet.other_subset is None, f"Other subset not supported in vectorization yet, found {memlet.other_subset} that is None for {memlet} (edge: {edge}) (state: {state})"

            state.remove_edge(edge)
            state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, self._vector_memlet(new_dataname))

    def _iterate_on_path_from_map_entry_to_exit(self, state: SDFGState, map_exit: dace.nodes.MapExit,
                                                data_in_edge: Edge[Memlet], dataname: str, new_dataname: str):
        """Forward-rewrite memlets from a map-entry in-edge toward the map exit.

        :param state: the state containing the dataflow.
        :param map_exit: the map exit that bounds the traversal.
        :param data_in_edge: the starting in-edge into the map.
        :param dataname: the original array name to match.
        :param new_dataname: the vector-copy array name.
        """
        # BFS along the dataflow rewriting matching memlets to the new (vector) data name.
        # The traversal pops one edge at a time, replaces its memlet, then enqueues the out-edges
        # of the new edge's dst — so the next iteration sees the up-to-date graph. Mutation and
        # iteration are decoupled per-edge.
        edges_to_check = {data_in_edge}
        sink_node = None
        while edges_to_check:
            edge = edges_to_check.pop()
            sink_node = edge.dst
            if edge.dst == map_exit:
                continue

            if edge.data.data is None or (edge.data.data != dataname and edge.data.data != new_dataname):
                continue

            memlet: dace.memlet.Memlet = edge.data
            self._assert_no_other_subset(memlet, edge, state)

            state.remove_edge(edge)
            edge = state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, self._vector_memlet(new_dataname))

            if isinstance(edge.dst, dace.nodes.MapEntry):
                new_edges = {
                    oe
                    for oe in state.out_edges_by_connector(edge.dst, edge.dst_conn.replace("IN_", "OUT_"))
                    if not isinstance(oe.dst, (dace.nodes.AccessNode, dace.nodes.NestedSDFG))
                }
            else:
                new_edges = {
                    oe
                    for oe in state.out_edges(edge.dst)
                    if not isinstance(oe.dst, (dace.nodes.AccessNode, dace.nodes.NestedSDFG))
                }
            edges_to_check = edges_to_check.union(new_edges)

        # the sink node is a code node and out data has the same array then we have a problem (not that we can fix but it needs to be preprocessed)
        if isinstance(sink_node, (CodeNode, dace.nodes.Tasklet)):
            for ie in state.out_edges(sink_node):
                if ie.data.data is not None and ie.data.data == dataname:
                    warnings.warn(
                        f"Vectorize: data {dataname} still used after sink node; read-write detection assumes "
                        f"the first occurrence flows out (state {state.label})")

    def _iterate_on_path_from_map_exit_to_entry(self, state: SDFGState, map_entry: dace.nodes.MapEntry,
                                                data_out_edge: Edge[Memlet], dataname: str, new_dataname: str):
        """Backward-rewrite memlets from a map-exit out-edge toward the map entry.

        :param state: the state containing the dataflow.
        :param map_entry: the map entry that bounds the traversal.
        :param data_out_edge: the starting out-edge from the map exit.
        :param dataname: the original array name to match.
        :param new_dataname: the vector-copy array name.
        """
        # Walk backward from a map_exit out-edge through the producing dataflow, rewriting matching
        # memlets to the new (vector) data name. The traversal stops as soon as the producing
        # path forks (multiple in-edges into a node) — this is intentional: the rewrite then leaves
        # the un-traversed branches alone, which matches the read-write detection convention used
        # in `_iterate_on_path_from_map_entry_to_exit` (warn afterwards if a downstream user might
        # still see the old name).
        edges_to_check = state.memlet_path(data_out_edge)

        while edges_to_check:
            source_node = None
            for edge in edges_to_check:
                if edge.src == map_entry:
                    edges_to_check = None
                    break

                if edge.data.data is None or edge.data.data != dataname:
                    continue

                memlet: dace.memlet.Memlet = edge.data
                self._assert_no_other_subset(memlet, edge, state)

                state.remove_edge(edge)
                state.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, self._vector_memlet(new_dataname))

            if edges_to_check is not None:
                source_node = edges_to_check[0].src
                new_in_edges = state.in_edges(source_node)
                if len(new_in_edges) == 1:
                    new_in_edge = new_in_edges[0]
                    edges_to_check = state.memlet_path(new_in_edge)
                else:
                    edges_to_check = None

            # the sink node is a code node and out data has the same array then we have a problem (not that we can fix but it needs to be preprocessed)
            if isinstance(source_node, (CodeNode, dace.nodes.Tasklet)):
                for ie in state.in_edges(source_node):
                    if ie.data.data is not None and ie.data.data == dataname:
                        warnings.warn(
                            f"Vectorize: data {dataname} still used after source node; read-write detection assumes "
                            f"the first occurrence flows out (state {state.label})")

    def _offset_memlets_on_path(self, state: SDFGState, map_entry: dace.nodes.MapEntry, dataname: str,
                                new_dataname: str):
        """Rewrite the in- and out-paths of a vectorized map to the vector copy.

        :param state: the state containing the map.
        :param map_entry: the vectorized map entry.
        :param dataname: the original array name.
        :param new_dataname: the vector-copy array name.
        """
        # Get memlet paths
        # And while memlet we have not encountered the map exit continue
        # if we find data name then we will replace
        # Precondition: memlet-path and no tree (previous passes to explicit vectorization should have fixed that)
        # one in edge to the map entry with the vector data
        map_exit = state.exit_node(map_entry)

        # Get all in edges (need to do the same for the map exit later)
        # Go from map_entry -> map_exit for input data
        # map_exit -> map_entry for output data
        in_edges = state.in_edges(map_entry)
        # Filter by the data
        data_in_edges = {e for e in in_edges if e.data.data == new_dataname}
        #assert len(data_in_edges) <= 1, f"{data_in_edges} length is not <= 1, but {len(data_in_edges)}"
        for data_in_edge in data_in_edges:
            self._iterate_on_path_from_map_entry_to_exit(state, map_exit, data_in_edge, dataname, new_dataname)

        # Now for map exit
        out_edges = state.out_edges(map_exit)
        # Filter by the data
        data_out_edges = {e for e in out_edges if e.data.data == new_dataname}
        assert len(data_out_edges) <= 1
        if len(data_out_edges) == 1:
            data_out_edge: Edge[Memlet] = next(iter(data_out_edges))
            # IMPORTANT!
            # Get memlet paths until we reach map exit
            # This will create a problem if the input flows into the exit because we can't distinguish,
            # We will assume the first occurence flow to the exit
            self._iterate_on_path_from_map_exit_to_entry(state, map_entry, data_out_edge, dataname, new_dataname)

        assert len(data_in_edges) == 1 or len(
            data_out_edges
        ) == 1, f"{dataname} -> {new_dataname} no data in our out edges found | {in_edges}, {out_edges}"

    def _find_new_name(self, candidate: str):
        """Return a unique array name derived from ``candidate``.

        :param candidate: the desired base name.
        :returns: ``candidate`` or a suffixed variant not yet used.
        """
        candidate2 = candidate
        i = 0
        while candidate2 in self._used_names:
            candidate2 = candidate + f"_{i}"
            i += 1
        self._used_names.add(candidate2)
        return candidate2

    def _vector_memlet(self, new_dataname: str) -> dace.memlet.Memlet:
        return dace.memlet.Memlet(
            data=new_dataname,
            subset=dace.subsets.Range([(dace.symbolic.SymExpr(0), dace.symbolic.SymExpr(self.vector_width) - 1,
                                        dace.symbolic.SymExpr(1))]),
        )

    @staticmethod
    def _assert_no_other_subset(memlet: dace.memlet.Memlet, edge: Edge[Memlet], state: SDFGState) -> None:
        """Raise if ``memlet.other_subset`` is set, which vectorization does not support.

        :param memlet: the memlet to check.
        :param edge: the edge carrying the memlet (for the error message).
        :param state: the state containing the edge (for the error message).
        :raises NotImplementedError: if ``other_subset`` is not None.
        """
        if memlet.other_subset is not None:
            raise NotImplementedError(
                f"Vectorize: other_subset not supported, found {memlet.other_subset} on memlet {memlet} "
                f"(edge {edge}, state {state.label})")

    @staticmethod
    def _map_is_masked_remainder(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
        """Whether ``map_entry`` is a masked vector remainder.

        ``GenerateIterationMask`` (P3) attaches an ``_iter_mask: bool[W]``
        transient only to the body NSDFG of the masked remainder; the main
        map and the scalar remainder never carry it. Detect it by scanning
        the map scope for a NestedSDFG whose inner SDFG declares one.

        :param state: State containing the map.
        :param map_entry: The vectorized map entry.
        :returns: True iff this map is the masked remainder.
        """
        for n in state.scope_subgraph(map_entry).nodes():
            if isinstance(n, dace.nodes.NestedSDFG):
                for nm in n.sdfg.arrays:
                    if nm == "_iter_mask" or nm.startswith("_iter_mask_"):
                        return True
        return False

    def _copy_in_and_copy_out(self, state: SDFGState, map_entry: dace.nodes.MapEntry, vectorization_number: int,
                              vectorizable_arrays: Dict[str, bool]):
        """Insert vector-staging copies before and after a vectorized map.

        :param state: the state containing the map.
        :param map_entry: the vectorized map entry.
        :param vectorization_number: unique index used to name staging arrays.
        :param vectorizable_arrays: per-array vectorizability flags.
        :raises RuntimeError: if a map-exit out-edge has no data.
        """
        map_exit = state.exit_node(map_entry)
        masked = self._map_is_masked_remainder(state, map_entry)
        data_and_offsets = list()
        in_datas = set()
        for ie in state.in_edges(map_entry):
            # If input storage is not registers need to copy in
            if ie.data.data is None:
                continue

            ie_arr = state.sdfg.arrays[ie.data.data]
            if isinstance(ie_arr, dace.data.Scalar):
                continue

            if ie.data.data in vectorizable_arrays and vectorizable_arrays[ie.data.data] is False:
                continue

            array = state.parent_graph.sdfg.arrays[ie.data.data]
            if array.storage != self.vector_input_storage:
                # Sanity check: the strided-pattern case used to hit this path with
                # ``ie.data.subset`` volume > vector_width and overflow the W-wide
                # destination via a contiguous CopyND. With the NSDFG-body strided
                # handling (``_setup_strided_inside_nsdfg``) and the bare-body
                # lane-fanout path (``_generate_strided_loads_to_packed_storage``)
                # both producing the right shapes, ``src`` no longer reaches
                # ``_copy_in_and_copy_out`` with a wider-than-W subset.
                try:
                    src_volume_int = int(ie.data.subset.num_elements())
                except (TypeError, ValueError):
                    src_volume_int = None
                assert src_volume_int is None or src_volume_int <= self.vector_width, (
                    f"_copy_in_and_copy_out: source memlet on {ie.data.data} has "
                    f"subset {ie.data.subset} (volume {src_volume_int}) > vector_width "
                    f"({self.vector_width}); strided handling should have intercepted earlier.")
                # Add new array, if not there
                arr_name_to_use = self._find_new_name(f"{VecNameScheme.make_k(ie.data.data)}{vectorization_number}")
                if arr_name_to_use not in state.parent_graph.sdfg.arrays:
                    state.parent_graph.sdfg.add_array(name=arr_name_to_use,
                                                      shape=(self.vector_width, ),
                                                      dtype=array.dtype,
                                                      storage=self.vector_input_storage,
                                                      transient=True,
                                                      allow_conflicts=False,
                                                      alignment=parse_int_or_default(self.vector_width, 8) *
                                                      array.dtype.bytes,
                                                      find_new_name=False,
                                                      may_alias=False)
                in_datas.add(arr_name_to_use)
                an = state.add_access(arr_name_to_use)
                an.setzero = True
                src, src_conn, dst, dst_conn, data = ie
                state.remove_edge(ie)
                if masked:
                    emit_staging_copy(state,
                                      src,
                                      src_conn,
                                      an,
                                      None,
                                      data,
                                      arr_name_to_use,
                                      int(self.vector_width),
                                      "in",
                                      gate_extent=True)
                else:
                    state.add_edge(src, src_conn, an, None, copy.deepcopy(data))
                state.add_edge(an, None, map_entry, ie.dst_conn,
                               dace.memlet.Memlet(f"{arr_name_to_use}[0:{self.vector_width}]"))

                memlet: dace.memlet.Memlet = ie.data
                dataname: str = memlet.data
                offsets = [b for (b, e, s) in memlet.subset]
                data_and_offsets.append((dataname, arr_name_to_use, offsets))

        out_datas = set()
        for oe in state.out_edges(map_exit):
            if oe.data.data is None:
                raise RuntimeError(f"Map exit out-edge has no data in {state.label}; pre-flatten before vectorization "
                                   f"(map_entry={map_entry})")
            oe_arr = state.sdfg.arrays[oe.data.data]
            assert isinstance(oe_arr, dace.data.Array)

            if oe.data.data in vectorizable_arrays and vectorizable_arrays[oe.data.data] is False:
                continue

            array = state.parent_graph.sdfg.arrays[oe.data.data]
            if array.storage != self.vector_output_storage:
                # Sanity check (symmetric to the in-edge side): same condition,
                # same reason for being unreachable.
                try:
                    out_volume_int = int(oe.data.subset.num_elements())
                except (TypeError, ValueError):
                    out_volume_int = None
                assert out_volume_int is None or out_volume_int <= self.vector_width, (
                    f"_copy_in_and_copy_out: output memlet on {oe.data.data} has "
                    f"subset {oe.data.subset} (volume {out_volume_int}) > vector_width "
                    f"({self.vector_width}); strided handling should have intercepted earlier.")
                # If the name exists in the inputs, reuse the name. No
                # ``_find_new_name`` here (unlike the in-edge side): the
                # out-edge buffer must keep the *same* name as the
                # matching in-edge buffer so an inout connector's two
                # directions agree (the ``VecNameScheme`` contract).
                arr_name_to_use = f"{VecNameScheme.make_k(oe.data.data)}{vectorization_number}"

                if arr_name_to_use not in state.parent_graph.sdfg.arrays:
                    state.parent_graph.sdfg.add_array(name=arr_name_to_use,
                                                      shape=(self.vector_width, ),
                                                      dtype=array.dtype,
                                                      storage=self.vector_input_storage,
                                                      transient=True,
                                                      allow_conflicts=False,
                                                      alignment=parse_int_or_default(self.vector_width, 8) *
                                                      array.dtype.bytes,
                                                      find_new_name=False,
                                                      may_alias=False)
                out_datas.add(arr_name_to_use)
                an = state.add_access(arr_name_to_use)
                an.setzero = True
                src, src_conn, dst, dst_conn, data = oe
                state.remove_edge(oe)
                state.add_edge(map_exit, src_conn, an, None,
                               dace.memlet.Memlet(f"{arr_name_to_use}[0:{self.vector_width}]"))
                if masked:
                    emit_staging_copy(state,
                                      an,
                                      None,
                                      dst,
                                      dst_conn,
                                      data,
                                      arr_name_to_use,
                                      int(self.vector_width),
                                      "out",
                                      gate_extent=True)
                else:
                    state.add_edge(an, None, dst, dst_conn, copy.deepcopy(data))

                memlet: dace.memlet.Memlet = oe.data
                dataname: str = memlet.data
                offsets = [b for (b, e, s) in memlet.subset]
                data_and_offsets.append((dataname, arr_name_to_use, offsets))

        for dataname, new_dataname, offsets in data_and_offsets:
            self._offset_memlets_on_path(state, map_entry, dataname, new_dataname)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """Vectorize every eligible innermost map in ``sdfg``.

        :param sdfg: the SDFG vectorized in place.
        :param pipeline_results: unused pipeline results.
        :returns: the set of map entries that were vectorized.
        :raises NotImplementedError: if input/output storage differ or an
            NSDFG without a parent map scope is found.
        """
        # Sanity check: every array must have a consistent unit-stride dim; raises on mixed layouts.
        # The result is no longer cached because callers now resolve the contiguous dim per-array
        # (via desc.strides.index(1)) rather than from a global F/C classification.
        assert_strides_are_packed_C_or_packed_Fortran(sdfg)

        sdfg.validate()

        if self.vector_input_storage != self.vector_output_storage:
            raise NotImplementedError("Different input and output storage types not implemented yet")

        # 1. Broadcast used scalars to vectorized type
        # 1.1 E.g. if we do 2.0 * A[i:i+4] then we need to have [2.0, 2.0, 2.0, 2.0] * A[i:i+4]

        # 2. Vectorize Maps and Tasklets
        # 2.1 Map needs to be tiled using the vector unit length
        # 2.1.1 Assumption - the inner dimension of the map should be generating unit-stride accesses
        # 2.1.2 If not, we need to re-order the maps so that the innermost map is the one with unit-stride accesses
        # A op B -> needs to be replaced with  vectorized_op(A, B)
        # 2.2 All memlets need to be updated to reflect the vectorized access
        # 2.3 All tasklets ened to be replaced with the vectorized code (using templates)

        # 3. Insert data transfers
        # for (o = 0; o < 4; o ++){
        #     A[i + o] ...;
        # }
        # Needs to become:
        # vecA[0:4] = A[i:i+4]; if source of A is not input location of the vector unit
        # Same for the output
        # This needs to be done before the vectorized map from source(A) ->  source(vecA)
        # And after the map for destination(vecA) -> destination(A)

        # 4. Recursively done to nested SDFGs
        # If the inner body - if the nestedSDFG is within a map then the inner SDFG
        # needs to be vectorized as well

        # For all maps:
        # Map1 [ Map2 [ Map3 [Body]]]
        # Can vectorize the innermost map only
        # If NestedSDFG we need to know if we have a parent map

        # Need to vectorize innermost maps always, if a map has a map inside, vectorize that
        map_entries = list()
        for node, graph in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry):
                map_entry: dace.nodes.MapEntry = node
                map_entries.append((map_entry, graph))

        # We allow a map to have no nested SDFGs within its body, or only one node which is a nestedSDFG
        # If there is only a nested SDFG inside we need vectorize that.
        num_vectorized = 0
        vectorized_maps = set()
        sdfgs_to_vectorize = set()

        applied = 0
        applied_set = set()
        for i, (map_entry, state) in enumerate(map_entries):
            if is_innermost_map(state, map_entry):
                num_vectorized += 1
                all_nodes_between = state.all_nodes_between(map_entry, state.exit_node(map_entry))

                if self._apply_on_maps is not None and map_entry not in self._apply_on_maps:
                    continue

                # If no unit stride dimension continue
                strides = [s for (b, e, s) in map_entry.map.range]
                if not any({s == 1 for s in strides}):
                    warnings.warn(f"Vectorize: skipping {map_entry} ({state.label}) - no unit-stride dimension")
                    continue

                if map_entry.map.label.startswith("vectorloop_"):
                    continue

                # Skip Sequential-scheduled maps: P2's scalar postamble runs
                # at step 1 and is explicitly marked Sequential — tiling it
                # to step W would emit a step-W loop over the trailing R<W
                # elements and overrun the kernel bounds.
                if map_entry.map.schedule == dace.dtypes.ScheduleType.Sequential:
                    continue

                # If map has a nested SDFG - and that has more nested SDFGs we cant vectorize it
                if has_nsdfg_depth_more_than_one(state, map_entry):
                    warnings.warn(
                        f"Vectorize: skipping {map_entry} ({state.label}) - multiple levels of nested SDFGs inside")
                    continue

                if not last_dim_of_map_is_contiguous_accesses(state, map_entry):
                    warnings.warn(
                        f"Vectorize: {map_entry} ({state.label}) last dimension does not fall on contiguous accesses; "
                        f"indirect accesses might not always be packed")

                if not map_consists_of_single_nsdfg_or_no_nsdfg(state, map_entry):
                    if self.fail_on_unvectorizable:
                        raise Exception(f"Map contains more than 1 NSDFG within both, or both {map_entry}, {state}")
                    else:
                        continue

                if map_param_appears_in_multiple_dimensions(state, map_entry):
                    if not map_param_dim_usage_is_linear_combo(state, map_entry):
                        if self.fail_on_unvectorizable:
                            raise NotImplementedError(
                                f"Vectorize: {map_entry} ({state.label}) - map param accesses multiple "
                                f"dimensions non-linearly; gather/scatter lowering for this shape is not "
                                f"yet implemented")
                        warnings.warn(f"Vectorize: skipping {map_entry} ({state.label}) - map param "
                                      f"accesses multiple dimensions non-linearly")
                        continue
                    # Linear-combination uses (A[i,i], A[2*i,i], A[i,2*i], ...) flow through
                    # the per-lane fan-out path; the strided-load / strided-store detector
                    # linearises the multi-dim subsets through array strides at post-emit.

                opt_nsdfg = get_single_nsdfg_inside_map(state, map_entry)
                if opt_nsdfg is not None:
                    nsdfg: dace.nodes.NestedSDFG = opt_nsdfg
                    if not has_only_states(nsdfg.sdfg):
                        # If it has a conditional block with only break inside we can still do it
                        if has_only_states_or_single_block_with_break_only(nsdfg.sdfg):
                            pass
                        else:
                            warnings.warn(f"Vectorize: skipping {map_entry} ({state.label}) - nested SDFG contains "
                                          f"non-state nodes other than break-only conditionals")
                            continue

                if not no_other_subset(state, map_entry):
                    if self.fail_on_unvectorizable:
                        raise Exception(f"Other subset is not supported, it appears in {map_entry}, {state}")
                    else:
                        continue

                if not no_wcr(state, map_entry):
                    if self.fail_on_unvectorizable:
                        raise Exception(f"WCR is not supported, it appears in {map_entry}, {state}")
                    else:
                        continue

                state.sdfg.validate()
                self._vectorize_map(state, map_entry, vectorization_number=i)
                applied += 1
                applied_set.add(map_entry)
                vectorized_maps.add(map_entry)
                if len(all_nodes_between) == 1 and isinstance(next(iter(all_nodes_between)), dace.nodes.NestedSDFG):
                    sdfgs_to_vectorize.add((next(iter(all_nodes_between)), state))

        # Assume we have :
        # --- Map Entry ---
        # -----------------
        #     NestedSDFG
        # tasklet without map
        # --- Map Exit  ---
        #
        # If the inside has no maps
        # We have problems. If no maps,
        # We need to add a parent map
        for node, state in sdfgs_to_vectorize:
            parent_scope_is_none = state.scope_dict()[node] is None
            if parent_scope_is_none:
                raise NotImplementedError(
                    "NestedSDFGs without parent map scopes are not supported, they must have been inlined if the pipeline has been called."
                    "If pipeline has been called verify why InlineSDFG failed, otherwise call InlineSDFG")

        current_global_code = sdfg.global_code[self.global_code_location]
        if isinstance(current_global_code, CodeBlock):
            current_global_code = current_global_code.as_string
        if self.global_code not in current_global_code:
            sdfg.append_global_code(cpp_code=self.global_code, location=self.global_code_location)
        # Set zero for all transients
        # Vectorization requires all transient to be 0 to not accidentally read trash data
        # All access nodes of the same array need to be setzero=True so the first node that triggers allocation
        # determines if we set zero or not
        for n, g in sdfg.all_nodes_recursive():
            if isinstance(n, dace.nodes.AccessNode):
                arr = g.sdfg.arrays[n.data]
                if arr.transient:
                    n.setzero = True

        return applied_set
