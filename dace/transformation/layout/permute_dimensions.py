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
    """The permutation to apply inside ``node``'s nested SDFG for an edge carrying ``outer_name``.

    Returns ``None`` when the nested body needs no rewrite, otherwise the permutation for
    ``inner_name``. Three cases, keyed on how much of the outer array the nested SDFG receives:

    * FULL RANK -- it sees every dimension, so the outer permutation applies verbatim. This is what
      ``prepare_for_layout`` leaves behind: ``ExpandNestedSDFGInputs`` widens a narrowed in/out
      subset to the full outer array and mirrors the outer shape onto the inner descriptor.
    * ONE DIMENSION -- a scalar or a single surviving axis (``A[i, j, 0:N] -> nA[N]``). A
      permutation reorders axes relative to one another, so with one axis there is nothing to
      reorder and no rewrite is needed.
    * ANYTHING BETWEEN -- a partial slice (``A[0:a, j, 0:c] -> nA[a, c]``). Which axes survive the
      squeeze decides the induced inner permutation, and we do not compute it. Refuse loudly: the
      alternative is to leave the body reading the OLD layout while the outer array is relaid, and
      when the surviving extents happen to be equal (``A[N, N, N]``) nothing downstream would catch
      the mismatch -- a silent transpose. Unreachable after ``prepare_for_layout``; this is the
      invariant check that keeps it that way.
    """
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

        The node is inserted WITHOUT an implementation: the transform does not choose a library
        lowering. The device-appropriate expansion (``pure`` map on CPU, ``cuTENSOR`` on GPU) is
        selected later, at compile time, by ``select_layout_lowering`` / the node default.
        """
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
        return inverse_permutation(permute_indices)

    def _note_copy_side(self, sides: Dict, edge, permute_indices: List[int]) -> None:
        note_copy_side(sides, edge, permute_indices)

    def _retranspose_copies(self, state: dace.SDFGState, sides: Dict) -> None:
        retranspose_copies(state, sides, context="PermuteDimensions")

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


def inverse_permutation(permute_indices: List[int]) -> List[int]:
    """The inverse axis permutation: ``inverse[permute_indices[i]] = i``."""
    inverse_map = {p: i for i, p in enumerate(permute_indices)}
    return [inverse_map[i] for i in sorted(inverse_map)]


def note_copy_side(sides: Dict, edge, permute_indices: List[int]) -> None:
    """Record that this edge -- which we just relaid out -- is an operand of a ``CopyLibraryNode``.

    A copy is elementwise: it moves logical index ``(i, j)`` to ``(i, j)``. Relaying out ONE of
    its operands makes it transposing, and a plain copy cannot express that. Shared bookkeeping of
    ``PermuteDimensions`` (in-place mode) and ``apply_assignment`` (trajectory mode)."""
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
    """Replace every copy the layout change made transposing with a ``TensorTranspose`` carrying the
    permutation.

    This MUST happen in the pass that applied the permutation, because the permutation is not
    recoverable afterwards: for a square array declared with one symbol (``A[N, N]``) the two
    operands of the copy end up with identical shapes, identical strides and identical subsets, so
    nothing downstream can tell an elementwise copy from a transposing one. Left as a plain copy it
    silently produces ``C = A^T`` instead of ``C = A``.

    Axes: with ``out = transpose(in, axes)`` and ``out_sizes[k] == in_sizes[axes[k]]``, relaying
    out the INPUT by ``P`` gives ``in_sizes[m] = original[P[m]]``, so ``axes = P^-1``; relaying
    out the OUTPUT by ``P`` gives ``axes = P``.
    """
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

        # Exactly two copy shapes are valid under a layout change:
        #
        #   * a UNIT-ELEMENT copy (every dimension degenerate) -- one element maps through the
        #     whole algebra trivially, whatever the permutation does to the strides around it, so
        #     it needs no transpose;
        #   * a FULL-array copy -- the array IS the region, so the permutation of the array is
        #     exactly the transpose of the copy.
        #
        # Every other SUBSET copy is invalid: it needs the permutation induced on the dimensions
        # it happens to span, and a whole-array TensorTranspose cannot express that (its validate
        # reads the array DESCRIPTORS, not the memlet subsets, so it would move data the copy
        # never touched). Copying half of one dimension into another half is not a permutation of
        # the array at all.
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

        # No implementation is set: choosing the library lowering is not a transform's job.
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
    """Rewrite ONE state for a dimension permutation: rename access nodes per ``name_map`` (old ->
    new array name; identity in in-place mode, a segment clone in trajectory mode), permute every
    matching memlet subset (``new_subset[i] = old_subset[perm[i]]``), and rename the scope
    ``IN_``/``OUT_`` connectors to match.

    This is the shared per-state rewrite core of ``PermuteDimensions`` (whole-SDFG, all states) and
    ``apply_assignment`` (task A5: a layout TRAJECTORY rewrites only the states of one segment).
    ``permute_map`` is keyed by the OLD name. ``note_copy_side(sides, edge, perm)`` is
    ``PermuteDimensions``' copy-retranspose bookkeeping hook; the noted sides are returned."""
    sides: Dict = {}
    for node in state.nodes():
        if isinstance(node, dace.nodes.AccessNode) and node.data in name_map:
            node.data = name_map[node.data]

    for edge in state.edges():
        if edge.data is not None and edge.data.data is not None and edge.data.data in name_map:
            # Replace map connectors to reference the permuted array (e.g. IN_A -> IN_per_A);
            # nested-SDFG connectors are left alone.
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
