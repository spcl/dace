# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
SplitArray Pass
===============
Splits arrays along dimensions whose extent is a known compile-time symbol.

Example: ``iphase[0:nclv]`` with ``nclv=3`` and a dimension name list `di, dq, dq` becomes ``iphase_di, iphase_dq, iphase_ds``.

When the index is data-dependent (e.g. ``iphase[idx[i]]``), the pass generates
a ConditionalBlock with one branch per possible value.

Prerequisites: Loops/maps over the split dimension must be unrollable.
Follow-up: Runnign simplify might be good after this pass
"""

import copy
import itertools
import sympy as sp
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import dace
from dace import SDFG, properties
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.dataflow.map_unroll import MapUnroll
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.passes.analysis import loop_analysis


def reverse_bfs_assignments(cfg: ControlFlowRegion, start_node) -> Dict[str, str]:
    """Walk backward from ``start_node`` collecting interstate-edge assignments.

    Returns the first (closest) assignment found for each symbol key.
    """
    result = {}
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)

    while queue:
        node = queue.popleft()
        for edge in cfg.in_edges(node):
            for key, val in edge.data.assignments.items():
                # Keep only the closest assignment (first encountered in BFS)
                if key not in result:
                    result[key] = val
            if edge.src not in visited:
                visited.add(edge.src)
                queue.append(edge.src)

    return result


def resolve_aliases(
    data_dependent_dims: List[Optional[dace.symbolic.SymExpr]],
    iedge_assignments: Dict[str, str],
) -> List[Optional[dace.symbolic.SymExpr]]:
    """Deduplicate aliased symbols in data-dependent index expressions.

    DaCe can generate multiple symbols mapping to the same value
    (e.g. ``imelt_index_0 = imelt[0], imelt_index_1 = imelt[0]``).
    This picks the first key per value as canonical and substitutes the rest.
    """
    val_to_canonical = {}
    alias_map = {}
    for k, v in iedge_assignments.items():
        v_str = str(v)
        if v_str not in val_to_canonical:
            val_to_canonical[v_str] = k
        else:
            alias_map[k] = val_to_canonical[v_str]

    resolved = []
    for expr in data_dependent_dims:
        if expr is None:
            resolved.append(None)
            continue
        for old, new in alias_map.items():
            expr = expr.subs(sp.Symbol(old), sp.Symbol(new))
        resolved.append(expr)
    return resolved


def copy_state_contents(old_state: SDFGState, new_state: SDFGState) -> Dict[dace.nodes.Node, dace.nodes.Node]:
    """Deep-copy all nodes and edges from ``old_state`` into ``new_state``.

    Returns a mapping from original nodes to their copies.
    """
    node_map = {}

    for n in old_state.nodes():
        c_n = copy.deepcopy(n)
        node_map[n] = c_n
        new_state.add_node(c_n)

    for e in old_state.edges():
        new_state.add_edge(
            node_map[e.src],
            e.src_conn,
            node_map[e.dst],
            e.dst_conn,
            copy.deepcopy(e.data),
        )

    return node_map


@properties.make_properties
@transformation.explicit_cf_compatible
class SplitArray(ppl.Pass):
    """Splits arrays along dimensions whose extent matches a known symbol.

    Args:
        symbol_map: Maps symbol names to their integer extents (e.g. ``{"nclv": 5}``).
        name_map: Maps symbol names to per-index name suffixes
                  (e.g. ``{"nclv": ["rain", "ice", "snow", "graupel", "hail"]}``).
    """

    CATEGORY: str = "Layout"

    # Class-level counter for unique ConditionalBlock labels
    c = 0

    def __init__(self, symbol_map: Dict[str, int], name_map: Dict[str, List[str]]):
        super().__init__()
        self._symbol_map = symbol_map
        self._name_map = name_map
        # Snapshot of array shapes before symbol replacement (symbols still symbolic)
        self._array_dim_map: Dict[str, tuple] = {}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    # ------------------------------------------------------------------ #
    #  Phase 0: Unroll loops/maps over split dimensions
    # ------------------------------------------------------------------ #
    def _unroll_loops_that_depend_only_on_split_dimensions(self, sdfg: dace.SDFG):
        """Unroll maps and loops whose iteration range matches a split-dimension extent.
        After ``replace_dict``, extents that depended only on split symbols become
        concrete integers. We unroll one at a time and rescan, because each
        unroll mutates the graph and invalidates node references.
        """
        assert self._array_dim_map, "Expected _array_dim_map to be populated before unrolling loops/maps"
        sdfg.replace_dict(self._symbol_map)
        potential_ranges = set(self._symbol_map.values())

        while True:
            target = None
            target_extent = None

            for n, g in sdfg.all_nodes_recursive():
                if isinstance(n, dace.nodes.MapEntry):
                    has_split_dim = False
                    for _, r in zip(n.map.params, n.map.range):
                        extent = ((r[1] + 1) - r[0]) // r[2]
                        if extent.free_symbols:
                            continue
                        try:
                            val = int(extent)
                        except (TypeError, ValueError):
                            continue
                        if val in potential_ranges:
                            has_split_dim = True
                            target_extent = val
                            break
                    if has_split_dim:
                        target = ("map", n, g)
                        break

                elif isinstance(n, LoopRegion):
                    beg = loop_analysis.get_init_assignment(n)
                    end = loop_analysis.get_loop_end(n)
                    step = loop_analysis.get_loop_stride(n)
                    if beg is None or end is None or step is None:
                        continue
                    extent = ((end + 1) - beg) // step
                    if extent.free_symbols:
                        continue
                    try:
                        val = int(extent)
                    except (TypeError, ValueError):
                        continue
                    if val in potential_ranges:
                        target = ("loop", n, g)
                        target_extent = val
                        break

            if target is None:
                break

            kind, n, g = target
            if kind == "map":
                MapUnroll().apply_to(sdfg=g.sdfg, options={}, map_entry=n)
            else:
                LoopUnroll().apply_to(sdfg=n.sdfg, options={"inline_iterations": True}, loop=n)

    # ------------------------------------------------------------------ #
    #  Phase 1: Identify which arrays/dimensions to split
    # ------------------------------------------------------------------ #

    def _collect_arrays_to_split(self, sdfg: dace.SDFG) -> Dict[str, List[Optional[str]]]:
        """Build a map from array name to per-dimension split config.

        Each entry is ``None`` (keep dimension) or a symbol name (split along it).
        Arrays with no split dimensions are omitted.
        """
        if not self._array_dim_map:
            for arr, desc in sdfg.arrays.items():
                self._array_dim_map[arr] = copy.deepcopy(desc.shape)

        split_map = {}
        for arrname in sdfg.arrays:
            # If not in present (added due to loop/map unroll, we cant split)
            if arrname in self._array_dim_map:
                desc_shape = self._array_dim_map[arrname]
                split_list = [None] * len(desc_shape)
                for i, d_expr in enumerate(desc_shape):
                    if str(d_expr) in self._symbol_map:
                        split_list[i] = str(d_expr)
                if not all(s is None for s in split_list):
                    split_map[arrname] = split_list
        return split_map

    # ------------------------------------------------------------------ #
    #  Phase 2: Create new split data descriptors
    # ------------------------------------------------------------------ #

    def _split_data_descriptors(self, sdfg: dace.SDFG, split_map: Dict[str, List[Optional[str]]]):
        """Create one new array descriptor per Cartesian-product combination of split indices.

        For ``arr[a, kept, b]`` with ``a`` split into 4 and ``b`` into 3,
        this creates 12 new arrays of shape ``[kept]``.
        """
        new_descs = {}

        for arr, desc in sdfg.arrays.items():
            if arr not in split_map:
                continue
            assert isinstance(desc, dace.data.Array)

            split_config = split_map[arr]

            assert desc.is_packed_c_strides() or desc.is_packed_fortran_strides()

            # Non-split dimensions form the shape of each new array
            filtered_shape = [dim for dim, split in zip(desc.shape, split_config) if split is None]

            # Recompute packed strides for the reduced shape
            if desc.is_packed_c_strides():
                filtered_strides = [1] * len(filtered_shape)
                for i in range(len(filtered_shape) - 2, -1, -1):
                    filtered_strides[i] = filtered_strides[i + 1] * filtered_shape[i + 1]
            else:
                filtered_strides = [1] * len(filtered_shape)
                for i in range(1, len(filtered_shape)):
                    filtered_strides[i] = filtered_strides[i - 1] * filtered_shape[i - 1]

            # Map positional index in split dims to their symbol name
            split_dims = [(dname, dim) for dname, dim in zip(split_config, desc.shape) if dname is not None]
            array_name_map = {}
            idx = 0
            for d in split_config:
                if d is not None:
                    array_name_map[idx] = d
                    idx += 1

            # Generate all combinations even if not all are accessed,
            # because the full array may be passed to nested SDFGs
            ranges = [range(extent) for _, extent in split_dims]
            for indices in itertools.product(*ranges):
                suffix = "_".join(str(self._name_map[array_name_map[i]][int(idx)]) for i, idx in enumerate(indices))
                new_descs[f"{arr}_{suffix}"] = dace.data.Array(
                    shape=filtered_shape,
                    strides=filtered_strides,
                    dtype=desc.dtype,
                    storage=desc.storage,
                    transient=desc.transient,
                    lifetime=desc.lifetime,
                )

        for name, desc in new_descs.items():
            sdfg.add_datadesc(name, desc)

    # ------------------------------------------------------------------ #
    #  Memlet rewriting helpers
    # ------------------------------------------------------------------ #

    def _get_corresponding_array(
        self,
        edge: MultiConnectorEdge[dace.Memlet],
        split_map: Dict[str, List[Optional[str]]],
        access_mapping: Optional[Dict[str, int]] = None,
    ) -> Tuple[str, dace.subsets.Range]:
        """Compute the new array name and subset for a memlet after splitting.

        For compile-time-known indices, the index is folded into the array name.
        For data-dependent indices, ``access_mapping`` must provide the concrete
        value (used inside ConditionalBlock branches).
        """
        if edge.data.data not in split_map:
            return edge.data.data, edge.data.subset

        dim_filter = split_map[edge.data.data]
        new_name_expr = []
        new_subset_expr = []

        for i, (splitd, access_expr) in enumerate(zip(dim_filter, edge.data.subset)):
            b, e, s = access_expr
            if splitd is not None:
                # Split dimensions must be single-element accesses
                assert ((e + 1) - b) // s == 1
                try:
                    access_offset = int(b)
                    new_name_expr.append(self._name_map[splitd][access_offset])
                except Exception as e:
                    if access_mapping is None:
                        raise Exception(f"Expression {b} is not an integer, can't resolve array"
                                        f" for {edge.data.data}[{edge.data.subset}],"
                                        f" {edge.src} -> {edge.dst}: {e}")
                    if str(b) not in access_mapping:
                        raise Exception(f"(Internal) access_mapping {access_mapping} missing key {b}")
                    new_name_expr.append(self._name_map[splitd][access_mapping[str(b)]])
            else:
                new_subset_expr.append((b, e, s))

        new_name = f"{edge.data.data}_{'_'.join(map(str, new_name_expr))}"
        return new_name, dace.subsets.Range(new_subset_expr)

    def _get_data_dependent_dims(
        self,
        state: SDFGState,
        edge: MultiConnectorEdge[dace.Memlet],
        split_map: Dict[str, List[Optional[str]]],
    ) -> Tuple[List[Optional[dace.symbolic.SymExpr]], List[Optional[dace.symbolic.SymExpr]]]:
        """Identify split-dimension indices that cannot be resolved at compile time.

        Returns two lists (one alias-resolved, one raw) with ``None`` for
        non-split or compile-time-known dims, and the symbolic expression otherwise.
        """
        if edge.data.data not in split_map:
            return edge.data.data, edge.data.subset

        dim_filter = split_map[edge.data.data]
        data_dependent_dims = []

        for i, (splitd, access_expr) in enumerate(zip(dim_filter, edge.data.subset)):
            b, e, s = access_expr
            if splitd is not None:
                assert ((e + 1) - b) // s == 1
                try:
                    int(b)  # Compile-time resolvable — not data-dependent
                except Exception:
                    data_dependent_dims.append(b)
            else:
                data_dependent_dims.append(None)

        # Resolve symbol aliases introduced by DaCe's interstate assignments
        iedge_assignments = reverse_bfs_assignments(state.parent_graph, state)
        all_exprs = copy.deepcopy(data_dependent_dims)
        data_dependent_dims = resolve_aliases(data_dependent_dims, iedge_assignments)

        return data_dependent_dims, all_exprs

    def _has_non_integer_access(
        self,
        sdfg: dace.SDFG,
        state: SDFGState,
        split_map: Dict[str, List[Optional[str]]],
    ) -> Tuple[Optional[str], Optional[Set[str]]]:
        """Check if ``state`` has any data-dependent accesses on split dimensions.

        Returns ``(canonical_symbol, all_alias_expressions)`` or ``(None, None)``.
        """
        all_data_dependent_dims: Set[str] = set()
        all_access_exprs: Set[str] = set()

        for edge in state.edges():
            if edge.data.data is not None and edge.data.data in split_map:
                data_dependent_dims, access_exprs = self._get_data_dependent_dims(state, edge, split_map)
                for d in data_dependent_dims:
                    if d is not None:
                        all_data_dependent_dims.add(str(d))
                        all_access_exprs.update(str(e) for e in access_exprs)

        if all_data_dependent_dims:
            # Multiple distinct data-dependent symbols would require nested branching
            if len(all_data_dependent_dims) > 1:
                sdfg.save("failing.sdfgz", compress=True)
            assert len(all_data_dependent_dims) == 1, (
                f"Multiple data-dependent dims not supported: {all_data_dependent_dims} in {state}")
            return all_data_dependent_dims.pop(), all_access_exprs
        return None, None

    def _get_non_int_access_dims(
        self,
        state: SDFGState,
        split_map: Dict[str, List[Optional[str]]],
        exprs: Set[str],
        canonical_access: str,
    ) -> Set[str]:
        """Find which split-dimension symbols are accessed data-dependently in ``state``."""
        dims: Set[str] = set()

        for edge in state.edges():
            if edge.data.data not in split_map:
                continue
            dim_filter = split_map[edge.data.data]

            for i, (splitd, access_expr) in enumerate(zip(dim_filter, edge.data.subset)):
                b, e, s = access_expr
                if splitd is not None:
                    assert ((e + 1) - b) // s == 1
                    try:
                        int(b)
                    except Exception:
                        if not any(str(expr) in str(b) for expr in exprs):
                            raise Exception(f"Expected expression {exprs} in access {b}")
                        if splitd != "None":
                            dims.add(splitd)
        return dims

    # ------------------------------------------------------------------ #
    #  Phase 3: Rewrite memlets and generate branches for dynamic accesses
    # ------------------------------------------------------------------ #

    def _replace_memlets(self, sdfg: dace.SDFG, split_map: Dict[str, List[Optional[str]]]):
        """Rewrite memlets to point to the new split arrays.

        For states with data-dependent accesses, a ConditionalBlock is inserted
        with one branch per possible index value, each containing a copy of the
        original state with the appropriate concrete array substituted.
        """
        for state in sdfg.all_states():
            canonical_access, all_access_exprs = self._has_non_integer_access(sdfg, state, split_map)

            if canonical_access is not None:
                dims = self._get_non_int_access_dims(state, split_map, all_access_exprs, canonical_access)
                assert len(dims) == 1
                dim = dims.pop()
                extent = self._symbol_map[dim]

                # Replace the state with a ConditionalBlock that branches on the
                # data-dependent index. Each branch gets a full copy of the state
                # with the split-dimension access resolved to a concrete array.
                cb = ConditionalBlock(
                    label=f"extent_check_{SplitArray.c}",
                    sdfg=sdfg,
                    parent=state.parent_graph,
                )
                g = state.parent_graph

                # Splice cb into the CFG in place of the original state
                g.add_node(cb, g.start_block == state)
                for ie in g.in_edges(state):
                    g.add_edge(ie.src, cb, dace.InterstateEdge())
                for oe in g.out_edges(state):
                    g.add_edge(cb, oe.dst, dace.InterstateEdge())
                g.remove_node(state)

                for i in range(extent):
                    cfg = ControlFlowRegion(
                        label=f"extent_check_{SplitArray.c}_branch_{i}",
                        sdfg=cb.sdfg,
                        parent=cb,
                    )
                    ns = cfg.add_state(f"extent_check_{SplitArray.c}_body_{i}", True)
                    copy_state_contents(state, ns)
                    cb.add_branch(
                        condition=properties.CodeBlock(f"{canonical_access} == {i}"),
                        branch=cfg,
                    )
                    # Map all alias expressions to the same concrete index
                    access_mapping = {k: i for k in all_access_exprs}
                    for edge in ns.edges():
                        mapped_data, new_subset = self._get_corresponding_array(edge, split_map, access_mapping)
                        if mapped_data != edge.data.data:
                            edge.data = dace.memlet.Memlet(data=mapped_data, subset=new_subset)

                SplitArray.c += 1
            else:
                # Static case: every split-dimension index is a compile-time constant
                for edge in state.edges():
                    mapped_data, new_subset = self._get_corresponding_array(edge, split_map)
                    if mapped_data != edge.data.data:
                        edge.data = dace.memlet.Memlet(data=mapped_data, subset=new_subset)

    # ------------------------------------------------------------------ #
    #  Phase 3b: Rewrite interstate-edge symbolic expressions
    # ------------------------------------------------------------------ #

    def _replace_iedges(self, sdfg: dace.SDFG, split_map: Dict[str, List[Optional[str]]]):
        """Rewrite array-as-function calls in interstate-edge assignments.

        DaCe represents array accesses in interstate edges as SymPy function
        applications (e.g. ``zsolqa(4, 2, j)``). This rewrites them to reference
        the split array (e.g. ``zsolqa_rain_snow(j)``).
        """

        def _rewrite_expr(expr):
            """Recursively rewrite SymPy expr, folding split-dim args into the name."""
            if not expr.args:
                return expr

            new_args = tuple(_rewrite_expr(a) for a in expr.args)

            if not expr.is_Function:
                return expr.func(*new_args)

            fname = type(expr).__name__
            if fname not in split_map:
                # Preserve argument order with non-commutative function
                return sp.Function(fname, commutative=False)(*new_args)

            dim_filter = split_map[fname]
            assert len(dim_filter) == len(new_args), (f"{fname}: split_config has {len(dim_filter)} dims "
                                                      f"but call has {len(new_args)} args: {expr}")

            name_parts = []
            kept_args = []
            for splitd, arg in zip(dim_filter, new_args):
                if splitd is not None:
                    name_parts.append(str(self._name_map[splitd][int(arg)]))
                else:
                    kept_args.append(arg)

            new_name = f"{fname}_{'_'.join(name_parts)}"

            if kept_args:
                return sp.Function(new_name, commutative=False)(*kept_args)
            else:
                # All dims split — collapses to a scalar symbol
                return sp.Symbol(new_name)

        for iedge in sdfg.all_interstate_edges():
            changed = False
            new_assignments = {}
            for k, v in iedge.data.assignments.items():
                v_sym = dace.symbolic.pystr_to_symbolic(v)
                v_new = _rewrite_expr(v_sym)
                new_assignments[k] = dace.symbolic.symstr(v_new, arrayexprs=frozenset(sdfg.arrays.keys()))
                if v_new is not v_sym:
                    changed = True
            if changed:
                iedge.data.assignments = new_assignments

    def _replace_access_nodes(self, sdfg: dace.SDFG, split_map: Dict[str, List[Optional[str]]]):
        """Point access nodes to the new split arrays.
        If an access node has different input and output array names (read from
        one split variant, write to another), the node is duplicated so each
        variant gets its own access node.
        """
        for state in sdfg.all_states():
            for dnode in list(state.data_nodes()):
                if dnode.data not in split_map:
                    continue

                in_edges = state.in_edges(dnode)
                out_edges = state.out_edges(dnode)
                in_names = {ie.data.data for ie in in_edges if ie.data.data is not None}
                out_names = {oe.data.data for oe in out_edges if oe.data.data is not None}
                unique_names = in_names | out_names

                # All edges agree on one name (or no edges at all) -> just rename
                if len(unique_names) <= 1:
                    if unique_names:
                        dnode.data = next(iter(unique_names))
                    continue

                # Multiple in-names combined with out-names is unsupported
                assert len(in_names) <= 1 or len(out_names) == 0, (f"Multiple in-names with out-names unsupported: "
                                                                   f"in={in_names}, out={out_names} in {state}")

                if len(out_names) > 0:
                    # <=1 in-name, multiple out-names -> split out-edges
                    if in_names:
                        dnode.data = next(iter(in_names))
                    for oe in list(out_edges):
                        dup = state.add_access(oe.data.data)
                        state.remove_edge(oe)
                        state.add_edge(dup, None, oe.dst, oe.dst_conn, copy.deepcopy(oe.data))
                        if in_names:
                            state.add_edge(dnode, None, dup, None, dace.memlet.Memlet())
                    if not in_names:
                        state.remove_node(dnode)
                else:
                    # Multiple in-names, no out-names -> split in-edges
                    for ie in list(in_edges):
                        dup = state.add_access(ie.data.data)
                        state.remove_edge(ie)
                        state.add_edge(ie.src, ie.src_conn, dup, None, copy.deepcopy(ie.data))
                    state.remove_node(dnode)

    def _remove_split_arrays(self, sdfg: dace.SDFG, split_map: Dict[str, List[Optional[str]]]):
        """Remove original (pre-split) array descriptors from the SDFG."""
        for arr in split_map:
            sdfg.remove_data(arr)

    def _pass_to_nsdfgs(self, sdfg: dace.SDFG, split_map: Dict[str, List[Optional[str]]]):
        """Propagate split arrays into nested SDFGs. (Not yet implemented.)"""
        for state in sdfg.all_states():
            for n in state.nodes():
                if isinstance(n, dace.nodes.NestedSDFG):
                    connectors = set(n.in_connectors) | set(n.out_connectors)
                    if any(name in split_map for name in connectors):
                        raise Exception(f"TODO: Split arrays passed to nested SDFGs not supported yet."
                                        f" Found in {n} of {state}. Split arrays: {list(split_map.keys())}")

    # ------------------------------------------------------------------ #
    #  Entry point
    # ------------------------------------------------------------------ #

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """Run the SplitArray pass.

        Pipeline:
          1. Unroll loops/maps over split dimensions (requires symbol replacement).
          2. Create new per-index array descriptors.
          3. Rewrite memlets, access nodes, and interstate edges.
          4. Propagate to nested SDFGs (TODO).
          5. Remove original arrays.
        """
        # Collect split config before symbol replacement (needs symbolic shapes)
        split_map = self._collect_arrays_to_split(sdfg)

        # Phase 0: Replace symbols with constants and unroll
        self._unroll_loops_that_depend_only_on_split_dimensions(sdfg)
        sdfg.validate()

        # Phase 2: Create split array descriptors
        self._split_data_descriptors(sdfg, split_map)

        # Phase 3: Rewrite all references to split arrays
        self._replace_memlets(sdfg, split_map)
        self._replace_access_nodes(sdfg, split_map)
        self._replace_iedges(sdfg, split_map)

        # Phase 4: Nested SDFG propagation (currently raises not-implemented error)
        self._pass_to_nsdfgs(sdfg, split_map)
        sdfg.save("split_array_intermediate.sdfgz", compress=True)
        # Phase 5: Cleanup
        self._remove_split_arrays(sdfg, split_map)
