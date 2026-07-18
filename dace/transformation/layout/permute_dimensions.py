import dace
import warnings

from typing import Dict, List, Any, Tuple
from dace.transformation import pass_pipeline as ppl
from dace.sdfg import nodes as nd
from dataclasses import dataclass


def _is_full_extent(memlet, arr) -> bool:
    """True iff ``memlet`` covers ``arr``'s full extent."""
    if memlet is None or memlet.subset is None:
        return False
    full = 1
    for s in arr.shape:
        full = full * s
    try:
        return memlet.subset.num_elements() == full
    except Exception:
        return False


def _nested_inner_permutation(sdfg, node, outer_name: str, inner_name: str, permute_map: Dict[str,
                                                                                              List[int]]) -> List[int]:
    """Inner permutation for outer_name in node's nested SDFG; None if full/1-D rank match, else raises (partial-slice rank is unreachable after prepare_for_layout)."""
    if outer_name not in permute_map:
        return None
    if inner_name is None or inner_name not in node.sdfg.arrays:
        return None
    outer_rank = len(sdfg.arrays[outer_name].shape)
    inner_rank = len(node.sdfg.arrays[inner_name].shape)
    if inner_rank == outer_rank:
        return permute_map[outer_name]
    if inner_rank == 1:
        return None
    raise NotImplementedError(
        f"PermuteDimensions: nested SDFG {node.label!r} receives {outer_name!r} (rank {outer_rank}) as "
        f"{inner_name!r} at rank {inner_rank} -- a partial slice. The induced permutation on the surviving "
        f"axes is not computed, and skipping it would leave the nested body reading the old layout. Run "
        f"prepare_for_layout first (ExpandNestedSDFGInputs widens nested inputs to the full array).")


def _is_memcpy_tasklet_between(state, src_an, dst_an) -> bool:
    """True iff a single memcpy_ tasklet with full-extent memlets connects src_an to dst_an."""
    for oe in state.out_edges(src_an):
        if not isinstance(oe.dst, nd.Tasklet):
            continue
        if not oe.dst.label.startswith('memcpy_'):
            continue
        for tasklet_oe in state.out_edges(oe.dst):
            if tasklet_oe.dst is dst_an:
                src_arr = state.parent.arrays.get(src_an.data)
                dst_arr = state.parent.arrays.get(dst_an.data)
                if (src_arr and dst_arr and _is_full_extent(oe.data, src_arr)
                        and _is_full_extent(tasklet_oe.data, dst_arr)):
                    return True
    return False


def _has_final_copy_in(state, t_name: str) -> bool:
    """True iff state drains t_name to a non-transient via a full-extent copy (AN->AN or AN->memcpy_tasklet->AN)."""
    sdfg = state.parent
    t_arr = sdfg.arrays.get(t_name)
    if t_arr is None or not t_arr.transient:
        return False
    for an in state.data_nodes():
        if an.data != t_name:
            continue
        for oe in state.out_edges(an):
            if isinstance(oe.dst, nd.AccessNode):
                dst_arr = sdfg.arrays.get(oe.dst.data)
                if dst_arr and not dst_arr.transient and _is_full_extent(oe.data, t_arr):
                    return True
            elif isinstance(oe.dst, nd.Tasklet) and oe.dst.label.startswith('memcpy_'):
                for tout in state.out_edges(oe.dst):
                    if isinstance(tout.dst, nd.AccessNode):
                        dst_arr = sdfg.arrays.get(tout.dst.data)
                        if (dst_arr and not dst_arr.transient and _is_memcpy_tasklet_between(state, an, tout.dst)):
                            return True
    return False


def _find_final_copy_state(sdfg, t_name: str):
    for state in sdfg.all_states():
        if _has_final_copy_in(state, t_name):
            return state
    return None


def _warn_unhandled_full_extent_ops(sdfg, t_name: str, init_state, final_state) -> None:
    """Warns on any top-level full-extent write/read of t_name that isn't the init copy, final copy, or body Map."""
    arr = sdfg.arrays[t_name]
    for state in sdfg.all_states():
        if state is init_state or state is final_state:
            continue
        sdict = state.scope_dict()
        for an in state.data_nodes():
            if an.data != t_name or sdict[an] is not None:
                continue
            for ie in state.in_edges(an):
                if isinstance(ie.src, nd.MapExit):
                    continue  # body kernel write -- handled by rename loop
                if _is_full_extent(ie.data, arr):
                    warnings.warn(
                        f"PermuteDimensions: full-extent write to transient "
                        f"'{t_name}' in state '{state.label}' is neither an "
                        f"init copy nor a body Map writer; the permutation "
                        f"may produce wrong output. Source node type: "
                        f"{type(ie.src).__name__}.",
                        stacklevel=2,
                    )
            for oe in state.out_edges(an):
                if isinstance(oe.dst, nd.MapEntry):
                    continue
                if _is_full_extent(oe.data, arr):
                    warnings.warn(
                        f"PermuteDimensions: full-extent read of transient "
                        f"'{t_name}' in state '{state.label}' is neither a "
                        f"final copy nor a body Map reader; the permutation "
                        f"may produce wrong output. Sink node type: "
                        f"{type(oe.dst).__name__}.",
                        stacklevel=2,
                    )


def _is_zero_init_tasklet(t: 'nd.Tasklet') -> bool:
    if not isinstance(t, nd.Tasklet):
        return False
    if len(t.in_connectors) != 0 or len(t.out_connectors) != 1:
        return False
    out_conn = next(iter(t.out_connectors))
    code = t.code.as_string.strip().rstrip(';').strip()
    if t.language == dace.Language.Python:
        return code in (f"{out_conn} = 0", f"{out_conn} = 0.0")
    return code in (f"{out_conn} = 0", f"{out_conn} = 0.0")


def _find_full_extent_writer(sdfg: dace.SDFG, name: str) -> Tuple[dace.SDFGState, 'nd.Node']:
    """Locates the unique (state, producer) that initializes name; raises ValueError if not exactly one."""
    candidates: List[Tuple[dace.SDFGState, 'nd.Node']] = []
    desc = sdfg.arrays[name]
    full_volume = 1
    for s in desc.shape:
        full_volume = full_volume * s
    for state in sdfg.all_states():
        for an in state.data_nodes():
            if an.data != name:
                continue
            in_edges = state.in_edges(an)
            if not in_edges:
                continue
            covered = False
            producer = None
            for ie in in_edges:
                m = ie.data
                if m is None or m.data is None:
                    continue
                try:
                    if m.subset is not None and m.subset.num_elements() == full_volume:
                        covered = True
                        producer = ie.src
                        break
                except Exception:
                    pass
            if covered:
                candidates.append((state, producer))
    if len(candidates) != 1:
        raise ValueError(f"Cannot permute transient '{name}': expected exactly one full-extent writer state, "
                         f"found {len(candidates)}.")
    return candidates[0]


def _is_zero_initialized(sdfg: dace.SDFG, name: str) -> bool:
    """True iff name's initialization writer is a MapEntry with a single full-extent zero-write Tasklet."""
    try:
        state, producer = _find_full_extent_writer(sdfg, name)
    except ValueError:
        return False
    if isinstance(producer, nd.MapExit):
        scope = state.scope_subgraph(state.entry_node(producer))
        tasklets = [n for n in scope.nodes() if isinstance(n, nd.Tasklet)]
        return len(tasklets) == 1 and _is_zero_init_tasklet(tasklets[0])
    if isinstance(producer, nd.Tasklet):
        return _is_zero_init_tasklet(producer)
    return False


@dataclass
class PermuteDimensions(ppl.Pass):

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.States | ppl.Modifies.AccessNodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors
                | ppl.Modifies.NestedSDFGs | ppl.Modifies.Memlets)

    def __init__(self,
                 permute_map: Dict[str, List[int]],
                 add_permute_maps: bool,
                 use_permute_libnodes: bool = False,
                 column_major: bool = False):
        self._permute_map = permute_map
        self._use_permute_libnodes = use_permute_libnodes
        self._add_permute_maps = add_permute_maps
        self._column_major = column_major

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # permutation applied once; never re-apply
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        # precondition (prepare_for_layout): no views except at library nodes, no implicit AN->AN copies;
        # WCR edges permuted like any other memlet, wcr preserved
        self._permute_index(sdfg, sdfg, self._permute_map, self._add_permute_maps)
        return 0

    def _add_permute_map(self, sdfg: dace.SDFG, state: dace.SDFGState, old_shape: List[int], new_shape: List[int],
                         permute_indices: List[int], old_name: str, new_name: str):
        """Copies old_name to new_name via transpose; no implementation chosen here (picked later by select_layout_lowering)."""
        if self._use_permute_libnodes:
            from dace.libraries.linalg import TensorTranspose

            old_access = state.add_access(old_name)
            new_access = state.add_access(new_name)

            assert len(old_shape) == len(new_shape), \
                f"Old shape {old_shape} and new shape {new_shape} must have the same length"

            tnode = TensorTranspose(f"permute_{old_name}_to_{new_name}", axes=permute_indices)
            state.add_node(tnode)

            state.add_edge(old_access, None, tnode, "_inp_tensor",
                           dace.Memlet.from_array(old_name, sdfg.arrays[old_name]))
            state.add_edge(tnode, "_out_tensor", new_access, None,
                           dace.Memlet.from_array(new_name, sdfg.arrays[new_name]))
        else:
            # Map iterates over the OLD shape
            map_params = [f"__i{d}" for d in range(len(old_shape))]
            map_ranges = {p: f"0:{s}" for p, s in zip(list(reversed(map_params)), list(reversed(old_shape)))}

            # read indices: natural order
            read_indices = ", ".join(map_params)

            # write indices: permuted order
            write_indices = ", ".join(map_params[permute_indices[d]] for d in range(len(permute_indices)))

            state.add_mapped_tasklet(
                name=f"permute_{old_name}_to_{new_name}",
                map_ranges=map_ranges,
                inputs={"__inp": dace.Memlet.simple(old_name, read_indices)},
                code="__out = __inp",
                outputs={"__out": dace.Memlet.simple(new_name, write_indices)},
                external_edges=True,
            )

    def _inverse_permute_indices(self, permute_indices: List[int]) -> List[int]:
        return inverse_permutation(permute_indices)

    def _note_copy_side(self, sides: Dict, edge, permute_indices: List[int]) -> None:
        note_copy_side(sides, edge, permute_indices)

    def _retranspose_copies(self, state: dace.SDFGState, sides: Dict) -> None:
        retranspose_copies(state, sides, context="PermuteDimensions")

    def _permute_index(self, root: dace.SDFG, sdfg: dace.SDFG, permute_map: Dict[str, List[int]],
                       add_permute_maps: bool):
        # top-level SDFG: may add transpose states/maps; nested: just replace array shapes
        name_map = dict()
        permute_states_to_skip = set()

        for arr_name, arr in list(sdfg.arrays.items()):
            if arr_name in permute_map:
                permute_indices = permute_map[arr_name]

                arr_shape = arr.shape

                permuted_shape = []
                assert len(permute_indices) == len(
                    arr_shape
                ), f"Permute indices {permute_indices} and array shape {arr_shape} must have the same length {arr_name}"
                for i in permute_indices:
                    permuted_shape.append(arr_shape[i])

                # permuted array is packed, contiguous
                strides = None
                if self._column_major:
                    strides = [1]
                    for i in range(len(permuted_shape) - 1):
                        strides.append(strides[-1] * permuted_shape[i])

                permuted_arr = dace.data.Array(
                    dtype=arr.dtype,
                    shape=permuted_shape,
                    strides=strides,
                    transient=True if (add_permute_maps and root == sdfg) else arr.transient,
                    allow_conflicts=arr.allow_conflicts,
                    storage=arr.storage,
                    alignment=arr.alignment,
                    lifetime=arr.lifetime,
                )

                # outer name gets a permuted_ prefix; nested SDFG keeps identity mapping
                if add_permute_maps and root == sdfg:
                    sdfg.add_datadesc(name="permuted_" + arr_name, datadesc=permuted_arr, find_new_name=False)
                else:
                    sdfg.remove_data(name=arr_name, validate=False)
                    sdfg.add_datadesc(name=arr_name,
                                      datadesc=permuted_arr)  # memlets permuted before validation

                name_map[arr_name] = "permuted_" + arr_name if (add_permute_maps and root == sdfg) else arr_name

        if root == sdfg:
            if add_permute_maps:
                # non-transient inputs: permute_in/out wrap states; transients: handled per-array below
                input_name_map = {n: m for n, m in name_map.items() if not sdfg.arrays[n].transient}
                transient_name_map = {n: m for n, m in name_map.items() if sdfg.arrays[n].transient}

                if input_name_map:
                    permute_state = sdfg.add_state_before(sdfg.start_state, "permute_in")
                    permute_states_to_skip.add(permute_state)
                    final_block = [v for v in sdfg.nodes() if sdfg.out_degree(v) == 0][0]
                    permute_out_state = sdfg.add_state_after(final_block, "permute_out")
                    permute_states_to_skip.add(permute_out_state)

                    # forward: original -> permuted
                    for old_name, new_name in input_name_map.items():
                        old_shape = sdfg.arrays[old_name].shape
                        new_shape = sdfg.arrays[new_name].shape
                        permute_indices = permute_map[old_name]

                        self._add_permute_map(sdfg=sdfg,
                                              state=permute_state,
                                              old_shape=old_shape,
                                              new_shape=new_shape,
                                              permute_indices=permute_indices,
                                              old_name=old_name,
                                              new_name=new_name)

                    # inverse: permuted -> original
                    for old_name, new_name in input_name_map.items():
                        old_shape = sdfg.arrays[old_name].shape
                        new_shape = sdfg.arrays[new_name].shape
                        # map is old->new; invert for the return trip
                        inverse_permute_indices = self._inverse_permute_indices(permute_map[old_name])

                        self._add_permute_map(sdfg=sdfg,
                                              state=permute_out_state,
                                              old_shape=new_shape,
                                              new_shape=old_shape,
                                              permute_indices=inverse_permute_indices,
                                              old_name=new_name,
                                              new_name=old_name)

                # per transient T: zero-init needs no transpose; else transpose after init (+ inverse
                # before drain if any); unique full-extent writer required, else error
                for old_name, new_name in transient_name_map.items():
                    if _is_zero_initialized(sdfg, old_name):
                        permute_states_to_skip.add(None)  # no-op marker
                        continue

                    # init state keeps old layout, excluded from rename below; permute_after_<T> does the transpose
                    init_state, _ = _find_full_extent_writer(sdfg, old_name)
                    final_state = _find_final_copy_state(sdfg, old_name)
                    _warn_unhandled_full_extent_ops(sdfg, old_name, init_state, final_state)
                    permute_states_to_skip.add(init_state)

                    old_shape = sdfg.arrays[old_name].shape
                    new_shape = sdfg.arrays[new_name].shape
                    permute_indices = permute_map[old_name]

                    after = sdfg.add_state_after(init_state, f"permute_after_{old_name}")
                    permute_states_to_skip.add(after)
                    self._add_permute_map(sdfg=sdfg,
                                          state=after,
                                          old_shape=old_shape,
                                          new_shape=new_shape,
                                          permute_indices=permute_indices,
                                          old_name=old_name,
                                          new_name=new_name)

                    if final_state is not None:
                        # mirror of init: drain keeps old layout, fed by an inverse transpose
                        permute_states_to_skip.add(final_state)
                        before = sdfg.add_state_before(final_state, f"permute_before_{old_name}")
                        permute_states_to_skip.add(before)
                        inverse = self._inverse_permute_indices(permute_indices)
                        self._add_permute_map(sdfg=sdfg,
                                              state=before,
                                              old_shape=new_shape,
                                              new_shape=old_shape,
                                              permute_indices=inverse,
                                              old_name=new_name,
                                              new_name=old_name)

        # shapes/maps added above; memlets not yet permuted. recurse into nested SDFGs first
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    new_permute_map = dict()
                    # nested SDFG names permuted only when the full array is passed, not a scalar slice
                    # TODO: views, do they need any changes?
                    for ie in state.in_edges(node):
                        inner = _nested_inner_permutation(sdfg, node, ie.data.data, ie.dst_conn, permute_map)
                        if inner is not None:
                            new_permute_map[ie.dst_conn] = inner
                    for oe in state.out_edges(node):
                        inner = _nested_inner_permutation(sdfg, node, oe.data.data, oe.src_conn, permute_map)
                        if inner is not None:
                            new_permute_map[oe.src_conn] = inner

                    self._permute_index(root=root, sdfg=node.sdfg, permute_map=new_permute_map, add_permute_maps=False)

        for state in sdfg.all_states():
            if sdfg == root and (state in permute_states_to_skip):
                continue
            permuted_copy_sides = rewrite_state_for_permute(state, name_map, permute_map, self._note_copy_side)
            self._retranspose_copies(state, permuted_copy_sides)

        for edge in sdfg.all_interstate_edges():
            new_assignments = dict()
            for k, v in edge.data.assignments.items():
                if any(name in v or name in k for name in permute_map.keys()):
                    new_v = _parse_interstate_edge(v, permute_map, sdfg)
                    new_assignments[k] = new_v
                else:
                    new_assignments[k] = v
            edge.data.assignments = new_assignments


def inverse_permutation(permute_indices: List[int]) -> List[int]:
    """The inverse axis permutation: ``inverse[permute_indices[i]] = i``."""
    inverse_map = {p: i for i, p in enumerate(permute_indices)}
    return [inverse_map[i] for i in sorted(inverse_map)]


def note_copy_side(sides: Dict, edge, permute_indices: List[int]) -> None:
    """Marks edge (a CopyLibraryNode operand) as relaid out; may turn an elementwise copy transposing."""
    from dace.libraries.standard.nodes.copy_node import CopyLibraryNode

    if isinstance(edge.dst, CopyLibraryNode):
        sides.setdefault(edge.dst, {})['in'] = list(permute_indices)
    if isinstance(edge.src, CopyLibraryNode):
        sides.setdefault(edge.src, {})['out'] = list(permute_indices)


def spanned_dims(memlet) -> int:
    """How many dimensions the memlet actually SPANS (extent > 1). Zero means a unit element."""
    if memlet is None or not isinstance(memlet.subset, dace.subsets.Range):
        return 0
    return sum(1 for begin, end, _ in memlet.subset.ranges if dace.symbolic.simplify(end - begin) != 0)


def covers_full_array(memlet, desc) -> bool:
    """True iff the memlet spans the WHOLE array (every dimension, full extent, unit step)."""
    if memlet is None or not isinstance(memlet.subset, dace.subsets.Range):
        return False
    ranges = memlet.subset.ranges
    if len(ranges) != len(desc.shape):
        return False
    for (begin, end, step), size in zip(ranges, desc.shape):
        if dace.symbolic.simplify(begin) != 0:
            return False
        if dace.symbolic.simplify(end - (size - 1)) != 0:
            return False
        if dace.symbolic.simplify(step - 1) != 0:
            return False
    return True


def retranspose_copies(state: dace.SDFGState, sides: Dict, context: str = "PermuteDimensions") -> None:
    """Replaces transposing copies with TensorTranspose; must run now, the permutation isn't recoverable later (axes = P^-1 if input relaid, P if output relaid)."""
    from dace.libraries.linalg import TensorTranspose
    from dace.libraries.standard.nodes.copy_node import CopyLibraryNode

    for copy_node, permuted in sides.items():
        if 'in' in permuted and 'out' in permuted:
            if permuted['in'] == permuted['out']:
                continue  # both operands moved the same way -- still elementwise
            raise NotImplementedError(f"{context}: the two operands of copy '{copy_node.label}' were permuted "
                                      f"differently ({permuted['in']} vs {permuted['out']}); composing the two "
                                      f"permutations into one transpose is not supported.")
        axes = (inverse_permutation(permuted['in']) if 'in' in permuted else list(permuted['out']))
        in_edge = next((e for e in state.in_edges(copy_node) if e.dst_conn == CopyLibraryNode.INPUT_CONNECTOR_NAME),
                       None)
        out_edge = next((e for e in state.out_edges(copy_node) if e.src_conn == CopyLibraryNode.OUTPUT_CONNECTOR_NAME),
                        None)
        if in_edge is None or out_edge is None:
            continue

        # only unit-element or full-array copies are valid here; any other subset spans dims a
        # whole-array TensorTranspose cannot express (it reads array descriptors, not memlet subsets)
        if max(spanned_dims(in_edge.data), spanned_dims(out_edge.data)) == 0:
            continue  # unit element

        in_desc = state.sdfg.arrays[in_edge.data.data]
        out_desc = state.sdfg.arrays[out_edge.data.data]
        if not (covers_full_array(in_edge.data, in_desc) and covers_full_array(out_edge.data, out_desc)):
            raise NotImplementedError(
                f"{context}: copy '{copy_node.label}' was made transposing by the layout "
                f"change, but it copies a SUB-REGION ({in_edge.data.subset} -> "
                f"{out_edge.data.subset}). Only a full-array copy (which the permutation IS) or a "
                f"unit-element copy (which maps trivially) can be relaid out; any other subset "
                f"needs the permutation induced on the dimensions it spans, which a whole-array "
                f"TensorTranspose cannot express.")

        # implementation left unset; lowering chosen later
        transpose = TensorTranspose(f"{copy_node.label}_transpose", axes=axes)
        state.add_node(transpose)
        state.add_edge(in_edge.src, in_edge.src_conn, transpose, "_inp_tensor", dace.Memlet.from_memlet(in_edge.data))
        state.add_edge(transpose, "_out_tensor", out_edge.dst, out_edge.dst_conn,
                       dace.Memlet.from_memlet(out_edge.data))
        state.remove_edge(in_edge)
        state.remove_edge(out_edge)
        state.remove_node(copy_node)


def rewrite_state_for_permute(state: dace.SDFGState,
                              name_map: Dict[str, str],
                              permute_map: Dict[str, List[int]],
                              note_copy_side=None) -> Dict:
    """Renames access nodes/connectors per name_map and permutes memlet subsets (new_subset[i] = old_subset[perm[i]]); shared rewrite core of PermuteDimensions and apply_assignment."""
    sides: Dict = {}
    for node in state.nodes():
        if isinstance(node, dace.nodes.AccessNode) and node.data in name_map:
            node.data = name_map[node.data]

    for edge in state.edges():
        if edge.data is not None and edge.data.data is not None and edge.data.data in name_map:
            # rename map connectors for the permuted array; nested-SDFG connectors untouched
            if edge.dst_conn == "IN_" + edge.data.data:
                edge.dst_conn = "IN_" + name_map[edge.data.data]
                edge.dst.remove_in_connector("IN_" + edge.data.data)
                edge.dst.add_in_connector("IN_" + name_map[edge.data.data])
            if edge.src_conn == "OUT_" + edge.data.data:
                edge.src_conn = "OUT_" + name_map[edge.data.data]
                edge.src.remove_out_connector("OUT_" + edge.data.data)
                edge.src.add_out_connector("OUT_" + name_map[edge.data.data])

            old_name = edge.data.data
            edge.data.data = name_map[old_name]

            new_subset = []
            permute_indices = permute_map[old_name]
            for i in range(len(permute_indices)):
                new_subset.append(edge.data.subset[permute_indices[i]])
            edge.data.subset = dace.subsets.Range(new_subset)

            if note_copy_side is not None:
                note_copy_side(sides, edge, permute_indices)
    return sides


def permute_args(expr, permute_map: dict[str, list[int]]):
    """Recursively permutes call args in a SymPy expr; permute_map[func][new_pos] = old_pos."""
    if not expr.args:
        return expr
    args = tuple(permute_args(a, permute_map) for a in expr.args)
    name = str(expr.func)
    if name in permute_map:
        perm = permute_map[name]
        args = tuple(args[perm[i]] for i in range(len(args)))
    if args == expr.args:
        return expr
    return expr.func(*args)


def _parse_interstate_edge(edge_data: str, permute_map: dict[str, list[int]], sdfg: dace.SDFG = None):
    symbolic_expr: dace.symbolic.SymExpr = dace.symbolic.pystr_to_symbolic(edge_data)
    permuted_symbolic_expr: dace.symbolic.SymExpr = permute_args(symbolic_expr, permute_map)
    permuted_str_expr: str = dace.symbolic.symstr(sym=permuted_symbolic_expr, arrayexprs=frozenset(sdfg.arrays.keys()))
    return permuted_str_expr
