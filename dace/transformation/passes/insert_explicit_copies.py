# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass replacing implicit copy patterns (path between two access nodes without
and intermediate tasklet) with explicit ``CopyLibraryNode`` instances.
"""
import copy as _copy
from typing import Any, Dict, Iterable, Optional

import dace
from dace import dtypes, nodes, properties
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.sdfg import is_devicelevel_gpu
from dace.transformation import pass_pipeline as ppl, transformation
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode


def _derive_matching_dst_subset(src_subset, dst_desc, src_desc):
    """Pick a destination subset for a memlet that omits ``other_subset``/``dst_subset``.

    Convention used by implicit copy edges: if the destination array's shape
    matches the source subset's volume shape (either directly, or after
    squeezing singleton dimensions), the implicit destination range is the
    full destination (offset 0 on every dimension). Otherwise we fall back
    to ``src_subset`` for backward compatibility — that path is only
    correct when the two arrays share the same shape.
    """
    from dace import subsets as _subsets

    src_size = list(src_subset.size())
    dst_shape = list(dst_desc.shape)

    # DaCe symbols are interned by name but `sympy.simplify` can leave
    # `N - N` un-simplified when the two `N` instances belong to different
    # symbol objects. Compare via string repr, which yields a canonical form
    # for symbols sharing a name.
    def _eq(a, b):
        if a is b or a == b:
            return True
        try:
            return str(a) == str(b)
        except Exception:
            return False

    def _shapes_match(a, b):
        return len(a) == len(b) and all(_eq(s, d) for s, d in zip(a, b))

    # Direct match: ranks and per-dim sizes line up exactly.
    if _shapes_match(src_size, dst_shape):
        return _subsets.Range.from_array(dst_desc)

    # Rank-reducing case: drop singleton dims from the source subset and
    # try again. Pattern: ``A[i, j, 0:N]`` -> 1-D destination of size ``N``
    # where the singleton index dims are implicit on the source side only.
    src_size_squeezed = [s for s in src_size if not _eq(s, 1)]
    if _shapes_match(src_size_squeezed, dst_shape):
        return _subsets.Range.from_array(dst_desc)

    # Rank match (after squeezing) but per-dim sizes differ symbolically.
    # This catches cases where the subset's stepped range produces a size
    # symbolically different from the destination's declared shape (e.g.
    # ``1:N-1:2`` -> ``ceiling(N/2) - 1`` vs ``floor(N/2) - 1``) even
    # though they agree at runtime for valid ``N``. Trust the user's
    # intent that the volumes line up and pick the destination's full
    # natural range.
    if len(src_size_squeezed) == len(dst_shape):
        return _subsets.Range.from_array(dst_desc)

    # Consecutive-rank reshape: a contiguous run of source dims may collapse
    # into a single destination dim (or vice versa, splitting a source dim
    # into a run of destination dims). Walk both shapes left-to-right with
    # two pointers, accumulating the running product on whichever side has
    # the smaller current value, and advance both when products meet.
    # Example: ``[8, 12, 5, 3] -> [96, 5, 3]`` collapses dims 0-1; ``[80, 12]
    # -> [8, 10, 12]`` splits dim 0. Squeeze 1s on both sides first so a
    # ``[8, 1, 12]`` source matches a ``[8, 12]`` destination through this
    # path even though squeezing the source alone would have caught it
    # earlier.
    dst_shape_squeezed = [s for s in dst_shape if not _eq(s, 1)]
    if _is_consecutive_reshape(src_size_squeezed, dst_shape_squeezed):
        return _subsets.Range.from_array(dst_desc)

    return src_subset


def _is_consecutive_reshape(src_size, dst_shape):
    """True iff ``dst_shape`` is reachable from ``src_size`` by collapsing
    contiguous runs of dimensions (or splitting one dim into a contiguous
    run). Both inputs are expected to be 1-squeezed."""
    i = j = 0
    src_acc = 1
    dst_acc = 1
    while i < len(src_size) and j < len(dst_shape):
        if src_acc == dst_acc:
            src_acc *= src_size[i]
            dst_acc *= dst_shape[j]
            i += 1
            j += 1
        elif _expr_lt(src_acc, dst_acc):
            src_acc *= src_size[i]
            i += 1
        else:
            dst_acc *= dst_shape[j]
            j += 1
    while i < len(src_size):
        src_acc *= src_size[i]
        i += 1
    while j < len(dst_shape):
        dst_acc *= dst_shape[j]
        j += 1
    try:
        return bool((src_acc - dst_acc) == 0)
    except Exception:
        return str(src_acc) == str(dst_acc)


def _expr_lt(a, b):
    """``a < b`` for sympy/numeric mixes, falling back to ``False`` on
    indeterminate symbolic comparisons (which sends both pointers forward
    in lockstep — safe under the equal-product guard)."""
    try:
        return bool((a - b) < 0)
    except Exception:
        return False


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertExplicitCopies(ppl.Pass):
    """Replaces implicit copy patterns with ``CopyLibraryNode`` instances.

    Detected patterns:
    - ``AccessNode -> AccessNode`` (direct copy edge)
    - ``AccessNode -> MapEntries -> AccessNode`` (stage-in)
    - ``AccessNode -> MapExits -> AccessNode`` (stage-out)
    """

    src_locations = properties.SetProperty(
        element_type=dtypes.StorageType,
        default=set(),
        desc="Only lift copies whose source storage is in this set. "
        "Empty set means any source storage is accepted.",
    )
    dst_locations = properties.SetProperty(
        element_type=dtypes.StorageType,
        default=set(),
        desc="Only lift copies whose destination storage is in this set. "
        "Empty set means any destination storage is accepted.",
    )
    skip_inside_device_scope = properties.Property(
        dtype=bool,
        default=False,
        desc="When True, copies whose endpoints sit inside a GPU device-level scope are left "
        "alone (cudaMemcpyAsync cannot be issued from device code, and the codegen handles "
        "intra-kernel copies through other paths).",
    )

    def __init__(self,
                 src_locations: Optional[Iterable[dtypes.StorageType]] = None,
                 dst_locations: Optional[Iterable[dtypes.StorageType]] = None,
                 skip_inside_device_scope: bool = False):
        super().__init__()
        self.src_locations = set(src_locations) if src_locations else set()
        self.dst_locations = set(dst_locations) if dst_locations else set()
        self.skip_inside_device_scope = skip_inside_device_scope

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _storage_allowed(self, src_storage: dtypes.StorageType, dst_storage: dtypes.StorageType) -> bool:
        """Return True when the (src, dst) storage pair passes the configured filter."""
        if self.src_locations and src_storage not in self.src_locations:
            return False
        if self.dst_locations and dst_storage not in self.dst_locations:
            return False
        return True

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Return the number of copy nodes inserted, or ``None`` if nothing changed."""
        count = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                # Pre-pass: rewrite AN<->View edges as `AN -> AN_inter -> View`
                # (or `View -> AN_inter -> AN`) so the View aliases a fresh
                # transient and the real data movement becomes a copy on the
                # AN side that the next step lifts normally.
                self._rewrite_view_edges(nsdfg, state)
                count += self._replace_direct_copies(nsdfg, state)
                count += self._replace_map_staging_copies(nsdfg, state)
        return count if count > 0 else None

    _VIEW_BUF_PREFIX = "_view_buf_"

    def _rewrite_view_edges(self, sdfg: SDFG, state: SDFGState) -> None:
        """Insert a packed transient between a direct AN<->View edge so the
        data-movement side becomes an AN<->AN edge that ``_replace_direct_copies``
        can lift to a ``CopyLibraryNode``. Idempotent across repeated calls
        and a no-op for non-packed views (which the codegen handles as
        pointer-offset aliases)."""
        from dace import data as dt
        from dace import subsets as _subsets

        for edge in list(state.edges()):
            if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
                continue
            if edge.data.is_empty():
                continue
            src_desc = sdfg.arrays[edge.src.data]
            dst_desc = sdfg.arrays[edge.dst.data]
            src_is_view = isinstance(src_desc, dt.View)
            dst_is_view = isinstance(dst_desc, dt.View)
            if src_is_view == dst_is_view:
                # Both Views or neither — nothing to do here. (Both-View
                # is an unusual nested-view pattern we leave alone.)
                continue

            non_view_node = edge.src if dst_is_view else edge.dst
            if non_view_node.data.startswith(self._VIEW_BUF_PREFIX):
                continue

            view_node = edge.dst if dst_is_view else edge.src
            view_desc = dst_desc if dst_is_view else src_desc

            if not view_desc.is_packed_c_strides():
                continue

            if dst_is_view:
                if not self._storage_allowed(src_desc.storage, view_desc.storage):
                    continue
            else:
                if not self._storage_allowed(view_desc.storage, dst_desc.storage):
                    continue

            inter_name, inter_desc = sdfg.add_array(
                f"{self._VIEW_BUF_PREFIX}{view_node.data}",
                shape=view_desc.shape,
                dtype=view_desc.dtype,
                storage=view_desc.storage,
                transient=True,
                find_new_name=True,
            )
            inter_node = state.add_access(inter_name)

            # The new alias-side memlet (intermediate <-> View) carries
            # the intermediate's full range; the original memlet stays on
            # the data-movement side (AN_orig <-> intermediate).
            inter_full_subset = _subsets.Range.from_array(inter_desc)
            alias_memlet = Memlet(data=inter_name, subset=_copy.deepcopy(inter_full_subset))
            alias_memlet.dynamic = edge.data.dynamic

            state.remove_edge(edge)
            if dst_is_view:
                # AN_orig -> View  =>  AN_orig -> AN_inter -> View
                state.add_edge(edge.src, edge.src_conn, inter_node, None, edge.data)
                state.add_edge(inter_node, None, view_node, edge.dst_conn, alias_memlet)
            else:
                # View -> AN_orig  =>  View -> AN_inter -> AN_orig
                state.add_edge(view_node, edge.src_conn, inter_node, None, alias_memlet)
                state.add_edge(inter_node, None, edge.dst, edge.dst_conn, edge.data)

    def _replace_direct_copies(self, sdfg: SDFG, state: SDFGState) -> int:
        """Replace direct ``AccessNode -> AccessNode`` edges with ``CopyLibraryNode`` instances."""
        edges = list(state.edges())
        count = 0
        for edge in edges:
            if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
                continue

            src_node: nodes.AccessNode = edge.src
            dst_node: nodes.AccessNode = edge.dst
            memlet: Memlet = edge.data

            if memlet.is_empty():
                continue

            # WCR edges aren't copies — they're reductions. Lifting them
            # into a ``CopyLibraryNode`` would lose the conflict-resolution
            # semantics (write-without-merge). They're left in place so a
            # later pass can lower them to a proper reduction node.
            if memlet.wcr is not None:
                continue

            src_desc = sdfg.arrays[src_node.data]
            dst_desc = sdfg.arrays[dst_node.data]

            # Views alias their underlying array; an Array<->View edge is an
            # aliasing reference, not a copy. Lifting it into a CopyLibraryNode
            # would (a) emit a memcpy between two pointers into the same buffer
            # and (b) break `sdutil.get_view_edge`, which requires the View's
            # neighbor on at least one side to be an AccessNode — it walks the
            # adjacent edge to the underlying buffer. Same convention the
            # legacy CUDA codegen already follows in `_compute_cudastreams`
            # ("Skip views").
            if isinstance(src_desc, dace.data.View) or isinstance(dst_desc, dace.data.View):
                continue

            if not self._storage_allowed(src_desc.storage, dst_desc.storage):
                continue

            in_device = (is_devicelevel_gpu(sdfg, state, src_node) or is_devicelevel_gpu(sdfg, state, dst_node))
            if in_device and self.skip_inside_device_scope:
                continue

            src_name = src_node.data
            dst_name = dst_node.data

            # `Memlet` carries `data` (which array `subset` refers to) plus an
            # optional `other_subset`. For self-copies (src_name == dst_name)
            # `memlet.data` matches both endpoints; the DaCe convention there
            # is that `subset` is the destination range, so check dst first.
            if memlet.data == dst_name:
                dst_subset = memlet.subset
                src_subset = memlet.other_subset
            elif memlet.data == src_name:
                src_subset = memlet.subset
                dst_subset = memlet.other_subset
            else:
                src_subset = memlet.subset
                dst_subset = memlet.other_subset

            # Fill in either side that wasn't carried by the memlet, deriving
            # a matching range on the absent side from the array shape when
            # the volumes line up (common for implicit copies between
            # different-shaped but same-volume arrays).
            if src_subset is None:
                src_subset = _derive_matching_dst_subset(dst_subset, src_desc, dst_desc)
            if dst_subset is None:
                dst_subset = _derive_matching_dst_subset(src_subset, dst_desc, src_desc)

            in_memlet = Memlet(data=src_name, subset=_copy.deepcopy(src_subset))
            in_memlet.dynamic = memlet.dynamic
            out_memlet = Memlet(data=dst_name, subset=_copy.deepcopy(dst_subset))
            out_memlet.dynamic = memlet.dynamic

            label = f"copy_{src_name}_to_{dst_name}"
            libnode = CopyLibraryNode(name=label)

            state.remove_edge(edge)
            state.add_node(libnode)
            state.add_edge(src_node, None, libnode, "_cpy_in", in_memlet)
            state.add_edge(libnode, "_cpy_out", dst_node, None, out_memlet)
            count += 1

        return count

    def _replace_map_staging_copies(self, sdfg: SDFG, state: SDFGState) -> int:
        """Replace map-boundary staging paths with ``CopyLibraryNode`` instances.

        Both staging directions are handled at any nesting depth:
          * ``AccessNode -> MapEntry (-> MapEntry)* -> AccessNode`` (stage-in)
          * ``AccessNode -> (MapExit ->)* MapExit -> AccessNode`` (stage-out)

        The outer AccessNode is resolved via ``state.memlet_path(edge)`` so
        intermediate map scopes between the two access nodes are transparent.
        """
        edges_to_process = []

        for e in state.edges():
            if e.data.is_empty():
                continue
            # WCR edges are reductions, not copies — leave for the
            # reduction-lowering pass.
            if e.data.wcr is not None:
                continue
            if isinstance(e.src, nodes.MapEntry) and isinstance(e.dst, nodes.AccessNode):
                direction = 'in'
            elif isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.MapExit):
                direction = 'out'
            else:
                continue

            mpath = state.memlet_path(e)
            # stage-in: innermost edge is (innermost MapEntry -> AccessNode); source of the path is the outer AccessNode.
            # stage-out: innermost edge is (AccessNode -> innermost MapExit); sink of the path is the outer AccessNode.
            outer_an = mpath[0].src if direction == 'in' else mpath[-1].dst
            if not isinstance(outer_an, nodes.AccessNode):
                continue

            # Skip lifts where either endpoint is a View — Views are
            # logical aliases, not real data movement. Forcing a copy
            # would write through the alias and break the View's read
            # semantics. The codegen handles View access as pointer
            # arithmetic; leave it alone here.
            local_an = e.dst if direction == 'in' else e.src
            from dace import data as dt
            if isinstance(sdfg.arrays[local_an.data], dt.View):
                continue
            if isinstance(sdfg.arrays[outer_an.data], dt.View):
                continue

            edges_to_process.append((direction, e, outer_an))

        count = 0
        for direction, edge, outer_an in edges_to_process:
            if edge not in state.edges():
                continue

            outer_desc = sdfg.arrays[outer_an.data]
            if direction == 'in':
                scope_node, local_an = edge.src, edge.dst
                local_desc = sdfg.arrays[local_an.data]
                src_storage, dst_storage = outer_desc.storage, local_desc.storage
            else:
                local_an, scope_node = edge.src, edge.dst
                local_desc = sdfg.arrays[local_an.data]
                src_storage, dst_storage = local_desc.storage, outer_desc.storage

            if not self._storage_allowed(src_storage, dst_storage):
                continue

            in_device = (is_devicelevel_gpu(sdfg, state, local_an) or is_devicelevel_gpu(sdfg, state, scope_node))
            if in_device and self.skip_inside_device_scope:
                continue

            outer_memlet = edge.data
            local_memlet = Memlet(data=local_an.data, subset=dace.subsets.Range.from_array(local_desc))
            outer_copy = Memlet(data=outer_memlet.data, subset=_copy.deepcopy(outer_memlet.subset))
            name = (f"copy_{outer_an.data}_to_{local_an.data}"
                    if direction == 'in' else f"copy_{local_an.data}_to_{outer_an.data}")
            libnode = CopyLibraryNode(name=name)

            state.remove_edge(edge)
            state.add_node(libnode)
            if direction == 'in':
                state.add_edge(scope_node, edge.src_conn, libnode, "_cpy_in", outer_copy)
                state.add_edge(libnode, "_cpy_out", local_an, None, local_memlet)
            else:
                state.add_edge(local_an, None, libnode, "_cpy_in", local_memlet)
                state.add_edge(libnode, "_cpy_out", scope_node, edge.dst_conn, outer_copy)

            count += 1

        return count
