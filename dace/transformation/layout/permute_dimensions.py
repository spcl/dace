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


def _is_memcpy_tasklet_between(state, src_an, dst_an) -> bool:
    """True iff there's a single tasklet whose name starts with
    ``memcpy_`` between ``src_an`` and ``dst_an``, with full-extent
    memlets on both edges."""
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
    """``state`` drains ``t_name`` to a non-transient via a full-extent
    copy (implicit AN->AN, or AN->memcpy_tasklet->AN)."""
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
    """Emit a loud warning for any top-level full-extent write/read of
    ``t_name`` that ISN'T the init copy, the final copy, or a body
    Map writer/reader (which the rename loop handles via subscript
    rewrite)."""
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
    """Locate the unique state + producer node that initializes ``name``.

    Returns (state, producer). Raises ``ValueError`` if zero or more than
    one full-extent writer state qualifies.
    """
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
    """True iff the initialization writer for ``name`` is a map-zero pattern.

    The pattern is a MapEntry whose body has a single zero-write Tasklet
    feeding the AccessNode for ``name`` over its full extent.
    """
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
        # This is an analysis pass, so it does not modify anything
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
        # Once permutaiton is done, no re-application is needed, ever
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        # Precondition (established by prepare_for_layout): no views except at library nodes,
        # no implicit AN->AN copies (lifted to CopyLibraryNodes). WCR reduction edges are
        # supported -- their subset is permuted like any other memlet, the wcr is preserved.
        self._permute_index(sdfg, sdfg, self._permute_map, self._add_permute_maps)
        return 0

    def _add_permute_map(self, sdfg: dace.SDFG, state: dace.SDFGState, old_shape: List[int], new_shape: List[int],
                         permute_indices: List[int], old_name: str, new_name: str):
        """
        Adds a transpose that copies data from the original layout to the permuted layout
        using the TensorTranspose library node.

        For GPU arrays, the cuTENSOR implementation is used (cutensorPermute).
        For CPU arrays, the pure (map-based) implementation is used.
        """
        if self._use_permute_libnodes:
            from dace.libraries.linalg import TensorTranspose

            old_access = state.add_access(old_name)
            new_access = state.add_access(new_name)

            assert len(old_shape) == len(new_shape), \
                f"Old shape {old_shape} and new shape {new_shape} must have the same length"

            impl = "pure"

            tnode = TensorTranspose(f"permute_{old_name}_to_{new_name}", axes=permute_indices)
            tnode.implementation = impl
            state.add_node(tnode)

            state.add_edge(old_access, None, tnode, "_inp_tensor",
                           dace.Memlet.from_array(old_name, sdfg.arrays[old_name]))
            state.add_edge(tnode, "_out_tensor", new_access, None,
                           dace.Memlet.from_array(new_name, sdfg.arrays[new_name]))
        else:
            # Map iterates over the OLD shape
            map_params = [f"__i{d}" for d in range(len(old_shape))]
            map_ranges = {p: f"0:{s}" for p, s in zip(list(reversed(map_params)), list(reversed(old_shape)))}

            # Read indices: i0, i1, i2, ...
            read_indices = ", ".join(map_params)

            # Write indices: permuted, e.g. [1,0,2] → __i1, __i0, __i2
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
        # implicit([0, 1, 2, 3]) -> [0, 3, 1, 2]
        # 1. get as a dictionary {0:0, 1:3, 2:1, 3:2}
        # 2. invert keys and values to get {0:0, 3:1, 1:2, 2:3}
        # 3. sort by keys to get {0:0, 1:3, 2:1, 3:2} -> [0, 3, 1, 2]
        # 1: Create mapping dictionary
        perm_map = {i: p for i, p in enumerate(permute_indices)}
        # 2: Invert the dictionary
        inverse_map = {v: k for k, v in perm_map.items()}
        # 3: Sort by key and extract values
        inverse_perm = [inverse_map[i] for i in sorted(inverse_map)]
        return inverse_perm

    def _permute_index(self, root: dace.SDFG, sdfg: dace.SDFG, permute_map: Dict[str, List[int]],
                       add_permute_maps: bool):
        # If top-level SDFG, namely the root is equal to the sdfg, we might need to add a transpose state and maps to
        # permute the arrays, otherwise we just replace the arrays with the permuted shape
        name_map = dict()
        permute_states_to_skip = set()

        for arr_name, arr in list(sdfg.arrays.items()):
            if arr_name in permute_map:
                permute_indices = permute_map[arr_name]

                arr_shape = arr.shape

                # Generate new shape
                permuted_shape = []
                assert len(permute_indices) == len(
                    arr_shape
                ), f"Permute indices {permute_indices} and array shape {arr_shape} must have the same length {arr_name}"
                for i in permute_indices:
                    permuted_shape.append(arr_shape[i])

                # Permuted array is packed (contiguous one-dimensional memory)
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

                # Change the in connector name
                # If before it was A -> (A)(NestedSDFG)
                # it will be per_A -> (A)(NestedSDFG)
                # If before it was A -> (nA)(NestedSDFG)
                # it will be per_A -> (nA)(NestedSDFG)
                # The nested SDFG needs to have identity as the name map
                if add_permute_maps and root == sdfg:
                    sdfg.add_datadesc(name="permuted_" + arr_name, datadesc=permuted_arr, find_new_name=False)
                else:
                    sdfg.remove_data(name=arr_name, validate=False)
                    sdfg.add_datadesc(name=arr_name,
                                      datadesc=permuted_arr)  # Need to transpose memlets before validation

                name_map[arr_name] = "permuted_" + arr_name if (add_permute_maps and root == sdfg) else arr_name

        if root == sdfg:
            if add_permute_maps:
                # Split into input (non-transient) and transient sub-maps.
                # Inputs go through the wrap-around permute_in/permute_out
                # states. Transients are handled per-array: zero-initialized
                # transients need no copy; non-zero-initialized transients
                # get an in-place permute right after their initialization
                # (assumed to cover the full extent — see
                # _find_full_extent_writer).
                input_name_map = {n: m for n, m in name_map.items() if not sdfg.arrays[n].transient}
                transient_name_map = {n: m for n, m in name_map.items() if sdfg.arrays[n].transient}

                if input_name_map:
                    permute_state = sdfg.add_state_before(sdfg.start_state, "permute_in")
                    permute_states_to_skip.add(permute_state)
                    final_block = [v for v in sdfg.nodes() if sdfg.out_degree(v) == 0][0]
                    permute_out_state = sdfg.add_state_after(final_block, "permute_out")
                    permute_states_to_skip.add(permute_out_state)

                    # Add maps to permute the input arrays to their permuted shape
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

                    # Add maps to permute the arrays back to their original shape
                    for old_name, new_name in input_name_map.items():
                        old_shape = sdfg.arrays[old_name].shape
                        new_shape = sdfg.arrays[new_name].shape
                        # Permute map is of form map[old] = new, we need to invert it
                        inverse_permute_indices = self._inverse_permute_indices(permute_map[old_name])

                        self._add_permute_map(sdfg=sdfg,
                                              state=permute_out_state,
                                              old_shape=new_shape,
                                              new_shape=old_shape,
                                              permute_indices=inverse_permute_indices,
                                              old_name=new_name,
                                              new_name=old_name)

                # Per-transient handling. For each permuted transient T:
                #
                #   * Zero-initialized T needs no transpose: the init Map is
                #     renamed and re-zeros over the permuted domain (handled by
                #     the ``_is_zero_initialized`` skip below).
                #   * Otherwise T has a unique full-extent writer (init Map or
                #     copy). That writer keeps the original layout, and a
                #     ``permute_after_<T>`` state transposes T -> permuted_T so
                #     the body reads the permuted layout.
                #   * A T that is drained to a non-transient via a full-extent
                #     copy gets the mirror ``permute_before_<T>`` (inverse
                #     transpose) just before that drain.
                #
                # A transient with no unique full-extent writer is a hard error
                # (``_find_full_extent_writer`` raises). Any other top-level
                # full-extent write/read of T outside a body Map triggers a loud
                # warning.
                for old_name, new_name in transient_name_map.items():
                    if _is_zero_initialized(sdfg, old_name):
                        permute_states_to_skip.add(None)  # no-op marker
                        continue

                    # The initializer (Map or full-extent copy) keeps writing the
                    # original layout, so its state is excluded from the rename
                    # below; the real transpose into the permuted layout happens
                    # in the inserted ``permute_after_<T>`` state. Raises when no
                    # unique full-extent writer exists.
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
                        # Mirror of the init: the drain copy keeps reading the
                        # original layout, fed by an inverse transpose inserted
                        # just before it.
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

        # The transformation has added the permuted shapes and maps to permute them if the user requested it.
        # The transformation has yet permuted the memlets as we want to access the previous defined arrays
        # The arrays passed to the NestedSDFG nodes need to be permuted as well, recursively go deeper
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    new_permute_map = dict()
                    # Change the in connector name
                    # If before it was A -> (A)(NestedSDFG)
                    # it will be per_A -> (A)(NestedSDFG)
                    # If before it was A -> (nA)(NestedSDFG)
                    # it will be per_A -> (nA)(NestedSDFG)

                    # The nested SDFG needs to have identity as the name map
                    # Update the names for the nested SDFG
                    # But only if the full array is passed, for example A[i] (array) -> tmp_X (scalar) does not require replacement

                    # TODO: views, do they need any changes?
                    # Open issue, what to do when we get all subsets, vs. only some of the subsets
                    # If we permute [a, b, c] -> [b, a, c], if we get subset [c] nothing changes,
                    # but if we get subset [a, b] we need to change it to [b, a]
                    # For now: the length of shape should be either 1 (no change needed), or the full length of the array (full change needed), otherwise we raise an exception as we do not know how to permute it
                    for ie in state.in_edges(node):
                        src_name = ie.data.data
                        dst_name = ie.dst_conn
                        dst_dimensionality = len(node.sdfg.arrays[dst_name].shape)
                        src_dimensionality = len(sdfg.arrays[src_name].shape)
                        if src_name in permute_map and src_dimensionality == dst_dimensionality:
                            new_permute_map[dst_name] = permute_map[src_name]
                    for oe in state.out_edges(node):
                        dst_name = oe.src_conn
                        src_name = oe.data.data
                        dst_dimensionality = len(node.sdfg.arrays[dst_name].shape)
                        src_dimensionality = len(sdfg.arrays[src_name].shape)
                        if src_name in permute_map and src_dimensionality == dst_dimensionality:
                            new_permute_map[dst_name] = permute_map[src_name]

                    self._permute_index(root=root, sdfg=node.sdfg, permute_map=new_permute_map, add_permute_maps=False)

        for state in sdfg.all_states():
            if sdfg == root and (state in permute_states_to_skip):
                continue
            for node in state.nodes():
                # Replace array access with the new Name (can be identity if we have not added permute maps)
                if isinstance(node, dace.nodes.AccessNode):
                    if node.data in name_map:
                        node.data = name_map[node.data]

            # Go through all memlets
            for edge in state.edges():
                if edge.data is not None and edge.data.data is not None and edge.data.data in name_map:
                    # Replace map connectors to reference to correct permuted array (e.g. IN_A -> IN_per_A)
                    # Do not change nested SDFG connectors
                    if edge.dst_conn == "IN_" + edge.data.data:
                        edge.dst_conn = "IN_" + name_map[edge.data.data]
                        edge.dst.remove_in_connector("IN_" + edge.data.data)
                        edge.dst.add_in_connector("IN_" + name_map[edge.data.data])
                    if edge.src_conn == "OUT_" + edge.data.data:
                        edge.src_conn = "OUT_" + name_map[edge.data.data]
                        edge.src.remove_out_connector("OUT_" + edge.data.data)
                        edge.src.add_out_connector("OUT_" + name_map[edge.data.data])

                    # Change data of the memlet
                    old_name = edge.data.data
                    edge.data.data = name_map[old_name]

                    # Permute the memlet subset
                    new_subset = []
                    permute_indices = permute_map[old_name]
                    for i in range(len(permute_indices)):
                        new_subset.append(edge.data.subset[permute_indices[i]])
                    edge.data.subset = dace.subsets.Range(new_subset)

        # Go through all interstate edges
        for edge in sdfg.all_interstate_edges():
            new_assignments = dict()
            for k, v in edge.data.assignments.items():
                # Replace array names if present, according to the permute conditions
                if any(name in v or name in k for name in permute_map.keys()):
                    # Time to replace
                    new_v = _parse_interstate_edge(v, permute_map, sdfg)
                    new_assignments[k] = new_v
                else:
                    new_assignments[k] = v
            edge.data.assignments = new_assignments


def permute_args(expr, permute_map: dict[str, list[int]]):
    """Recursively permute function call arguments in a SymPy/DaCe expression.

    permute_map: {func_name: perm} where perm[new_pos] = old_pos.
    Returns the original expression object if nothing changed.
    """
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
