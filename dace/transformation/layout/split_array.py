# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Splits arrays along dimensions whose extent is a known compile-time symbol."""

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
    """Walk backward from ``start_node``, keeping the closest assignment per symbol key."""
    result = {}
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)

    while queue:
        node = queue.popleft()
        for edge in cfg.in_edges(node):
            for key, val in edge.data.assignments.items():
                # first (closest) assignment wins
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
    """Deduplicate symbols in data-dependent index expressions that alias the same value."""
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
    """Deep-copy all nodes/edges from ``old_state`` into ``new_state``; returns old->new node map."""
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

    :param symbol_map: Symbol name -> integer extent (e.g. ``{"nclv": 5}``).
    :param name_map: Symbol name -> per-index name suffixes (e.g. ``{"nclv": ["rain", "ice", ...]}``).
    """

    CATEGORY: str = "Layout"

    # unique ConditionalBlock label counter
    c = 0

    def __init__(self, symbol_map: Dict[str, int], name_map: Dict[str, List[str]]):
        super().__init__()
        self._symbol_map = symbol_map
        self._name_map = name_map
        # shapes before symbol replacement (still symbolic)
        self._array_dim_map: Dict[str, tuple] = {}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    # ------------------------------------------------------------------ #
    #  Phase 0: Unroll loops/maps over split dimensions
    # ------------------------------------------------------------------ #
    def _unroll_loops_that_depend_only_on_split_dimensions(self, sdfg: dace.SDFG) -> int:
        """Unroll maps/loops whose range matches a split extent, one at a time (unrolling invalidates node refs).

        :param sdfg: The SDFG to specialize and unroll in place.
        :returns: Number of maps/loops unrolled.
        """
        assert self._array_dim_map, "Expected _array_dim_map to be populated before unrolling loops/maps"
        sdfg.replace_dict(self._symbol_map)
        potential_ranges = set(self._symbol_map.values())
        unrolled = 0

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
                    extent = dace.symbolic.int_floor((end + 1) - beg, step)  # int_floor, never `//`
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
            unrolled += 1

        return unrolled

    # ------------------------------------------------------------------ #
    #  Phase 1: Identify which arrays/dimensions to split
    # ------------------------------------------------------------------ #

    def _collect_arrays_to_split(self, sdfg: dace.SDFG) -> Dict[str, List[Optional[str]]]:
        """Map each array to a per-dim split config: None (keep) or symbol name (split); omits unsplit arrays."""
        if not self._array_dim_map:
            for arr, desc in sdfg.arrays.items():
                self._array_dim_map[arr] = copy.deepcopy(desc.shape)

        split_map = {}
        for arrname in sdfg.arrays:
            # skip arrays added post-snapshot (unroll) -- can't split those
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
        """Create one new array descriptor per Cartesian-product combination of split indices."""
        new_descs = {}

        for arr, desc in sdfg.arrays.items():
            if arr not in split_map:
                continue
            assert isinstance(desc, dace.data.Array)

            split_config = split_map[arr]

            assert desc.is_packed_c_strides() or desc.is_packed_fortran_strides()

            # shape of the new array = non-split dims
            filtered_shape = [dim for dim, split in zip(desc.shape, split_config) if split is None]

            # fully-split element: length-1 array (not scalar) so it stays writable, at index 0
            if not filtered_shape:
                filtered_shape = [1]

            # recompute packed strides for the reduced shape
            if desc.is_packed_c_strides():
                filtered_strides = [1] * len(filtered_shape)
                for i in range(len(filtered_shape) - 2, -1, -1):
                    filtered_strides[i] = filtered_strides[i + 1] * filtered_shape[i + 1]
            else:
                filtered_strides = [1] * len(filtered_shape)
                for i in range(1, len(filtered_shape)):
                    filtered_strides[i] = filtered_strides[i - 1] * filtered_shape[i - 1]

            # positional index -> split symbol name
            split_dims = [(dname, dim) for dname, dim in zip(split_config, desc.shape) if dname is not None]
            array_name_map = {}
            idx = 0
            for d in split_config:
                if d is not None:
                    array_name_map[idx] = d
                    idx += 1

            # generate all combinations; unaccessed ones may still be passed to nested SDFGs
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
        """Compute the new split-array name and subset for a memlet; access_mapping resolves data-dependent indices."""
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
        """Identify split-dim indices unresolved at compile time; returns (alias-resolved, raw) lists, None for the rest."""
        if edge.data.data not in split_map:
            return edge.data.data, edge.data.subset

        dim_filter = split_map[edge.data.data]
        data_dependent_dims = []

        for i, (splitd, access_expr) in enumerate(zip(dim_filter, edge.data.subset)):
            b, e, s = access_expr
            if splitd is not None:
                assert ((e + 1) - b) // s == 1
                try:
                    int(b)  # compile-time resolvable, not data-dependent
                except Exception:
                    data_dependent_dims.append(b)
            else:
                data_dependent_dims.append(None)

        # resolve DaCe-introduced symbol aliases
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
        """Check state for data-dependent accesses on split dims; returns (canonical_symbol, alias_exprs) or (None, None)."""
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
        """Rewrite memlets to the new split arrays; data-dependent accesses get a ConditionalBlock per index value."""
        for state in sdfg.all_states():
            canonical_access, all_access_exprs = self._has_non_integer_access(sdfg, state, split_map)

            if canonical_access is not None:
                dims = self._get_non_int_access_dims(state, split_map, all_access_exprs, canonical_access)
                assert len(dims) == 1
                dim = dims.pop()
                extent = self._symbol_map[dim]

                # replace state with a ConditionalBlock; each branch is a full state copy with a concrete array
                cb = ConditionalBlock(
                    label=f"extent_check_{SplitArray.c}",
                    sdfg=sdfg,
                    parent=state.parent_graph,
                )
                g = state.parent_graph

                # splice cb in place of state; interstate edges preserved verbatim (in-edge defines the branch symbol)
                g.add_node(cb, g.start_block == state)
                for ie in g.in_edges(state):
                    g.add_edge(ie.src, cb, copy.deepcopy(ie.data))
                for oe in g.out_edges(state):
                    g.add_edge(cb, oe.dst, copy.deepcopy(oe.data))
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
                    # alias expressions -> same concrete index
                    access_mapping = {k: i for k in all_access_exprs}
                    for edge in ns.edges():
                        mapped_data, new_subset = self._get_corresponding_array(edge, split_map, access_mapping)
                        if mapped_data != edge.data.data:
                            # keep wcr/other_subset/dynamic (dropping wcr turns a reduction into an overwrite)
                            edge.data = dace.memlet.Memlet(data=mapped_data,
                                                           subset=new_subset,
                                                           other_subset=copy.deepcopy(edge.data.other_subset),
                                                           wcr=edge.data.wcr,
                                                           wcr_nonatomic=edge.data.wcr_nonatomic,
                                                           dynamic=edge.data.dynamic)

                SplitArray.c += 1
            else:
                # static case: every split-dim index is a compile-time constant
                for edge in state.edges():
                    mapped_data, new_subset = self._get_corresponding_array(edge, split_map)
                    if mapped_data != edge.data.data:
                        # keep wcr/other_subset/dynamic -- dropping any breaks reductions, copies, or dynamic edges
                        edge.data = dace.memlet.Memlet(data=mapped_data,
                                                       subset=new_subset,
                                                       other_subset=copy.deepcopy(edge.data.other_subset),
                                                       wcr=edge.data.wcr,
                                                       wcr_nonatomic=edge.data.wcr_nonatomic,
                                                       dynamic=edge.data.dynamic)

    # ------------------------------------------------------------------ #
    #  Phase 3b: Rewrite interstate-edge symbolic expressions
    # ------------------------------------------------------------------ #

    def _replace_iedges(self, sdfg: dace.SDFG, split_map: Dict[str, List[Optional[str]]]):
        """Rewrite interstate-edge array accesses to the split name (handles both AppliedUndef and Subscript SymPy shapes)."""

        def _rewrite_expr(expr):
            """Recursively rewrite SymPy expr, folding split-dim indices into the name."""
            if not expr.args:
                return expr

            new_args = tuple(_rewrite_expr(a) for a in expr.args)

            if not expr.is_Function:
                return expr.func(*new_args)

            # normalize both shapes to (array name, indices)
            fname = type(expr).__name__
            if fname == 'Subscript':
                base, index_args = new_args[0], new_args[1:]
                fname = str(base)
            else:
                index_args = new_args

            if fname not in split_map:
                # expr.func (not sp.Function(fname)) preserves the class so the name doesn't leak into free_symbols
                return expr.func(*new_args)

            dim_filter = split_map[fname]
            assert len(dim_filter) == len(index_args), (f"{fname}: split_config has {len(dim_filter)} dims "
                                                        f"but access has {len(index_args)} indices: {expr}")

            name_parts = []
            kept_args = []
            for splitd, arg in zip(dim_filter, index_args):
                if splitd is not None:
                    name_parts.append(str(self._name_map[splitd][int(arg)]))
                else:
                    kept_args.append(arg)

            new_name = f"{fname}_{'_'.join(name_parts)}"

            if kept_args:
                return sp.Function(new_name, commutative=False)(*kept_args)
            else:
                # all dims split: length-1 array (see _split_data_descriptors); index 0 dereferences it
                return sp.Function(new_name, commutative=False)(sp.Integer(0))

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
        """Point access nodes to the new split arrays; duplicates a node when in/out names differ."""
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
        """Remove original pre-split array descriptors."""
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
        """Split every array whose shape carries a mapped symbol into one array per index.

        :param sdfg: The SDFG to transform in place.
        :param _: Pipeline results (unused).
        :returns: Number of changes made (arrays split + symbols specialized + maps/loops unrolled), or ``None``
                  if the SDFG uses none of the mapped symbols.
        """
        # needs symbolic shapes, before symbol replacement
        split_map = self._collect_arrays_to_split(sdfg)
        # Phase 0 specializes ``_symbol_map`` into the SDFG, so it modifies the graph even when no
        # array ends up being split. Count the symbols it will substitute before they disappear.
        specialized = sum(1 for sym in self._symbol_map if sym in sdfg.symbols)

        # Phase 0
        unrolled = self._unroll_loops_that_depend_only_on_split_dimensions(sdfg)
        sdfg.validate()

        # Phase 2
        self._split_data_descriptors(sdfg, split_map)

        # Phase 3
        self._replace_memlets(sdfg, split_map)
        self._replace_access_nodes(sdfg, split_map)
        self._replace_iedges(sdfg, split_map)

        # Phase 4 (not yet implemented, raises)
        self._pass_to_nsdfgs(sdfg, split_map)
        # Phase 5
        self._remove_split_arrays(sdfg, split_map)

        return (len(split_map) + specialized + unrolled) or None
