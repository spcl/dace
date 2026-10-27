# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Global memlet-path utilities shared by the memlet access policy passes: finding leaf memlets, following a leaf
memlet's path through nested SDFGs (:class:`GlobalPath`), and the loop / lane symbols that enclose it."""

from collections.abc import Iterator
from dataclasses import dataclass

import sympy as sp

from dace import SDFG, SDFGState, dtypes, symbolic
from dace import data as dt
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import LoopRegion
from dace.subsets import Range
from dace.transformation import gpu_helpers


# ---------------------------------------------------------------------------------------------------------------
# Symbolic helpers
# ---------------------------------------------------------------------------------------------------------------
def symbol_named(expr: sp.Basic, name: str) -> sp.Symbol | None:
    """The free symbol of ``expr`` called ``name``, or ``None``."""
    for s in expr.free_symbols:
        if str(s) == name:
            return s
    return None


def symbol_names(expr: sp.Basic) -> set[str]:
    """Names of the free symbols of ``expr``."""
    return {str(s) for s in expr.free_symbols}


# ---------------------------------------------------------------------------------------------------------------
# Containers and offsets
# ---------------------------------------------------------------------------------------------------------------
def base_offset(desc: dt.Data, memlet: Memlet) -> sp.Basic:
    """The physical element offset of the first element of ``memlet`` in ``desc`` (descriptor offset included;
    what ``cpp_offset_expr`` emits). The flat reference is set to the array's physical element 0."""
    subset = memlet.subset.offset_new(desc.offset, False)
    return symbolic.pystr_to_symbolic(subset.at([0] * len(desc.strides), desc.strides))


def flat_length(desc: dt.Data, subset: Range) -> symbolic.SymbolicType | None:
    """Number of elements of ``subset`` if it is a contiguous run in memory (at most one dimension of size > 1,
    with unit step and unit stride), otherwise ``None``. Callers only pass :class:`~dace.subsets.Range` subsets
    (memlets with other subsets get no loop cursor)."""
    sizes = subset.size()
    wide = [d for d, s in enumerate(sizes) if sp.sympify(s) != 1]
    if not wide:
        return sp.Integer(1)
    if len(wide) > 1:
        return None
    d = wide[0]
    if sp.sympify(subset.ranges[d][2]) != 1 or sp.sympify(desc.strides[d]) != 1:
        return None
    return sp.sympify(sizes[d])


def flat_addressable_array(desc: dt.Data | None) -> bool:
    """Arrays whose base address a flat reference can hold: plain arrays (no views, references, scalars, streams)."""
    return isinstance(desc, dt.Array) and not isinstance(desc, dt.View)


# ---------------------------------------------------------------------------------------------------------------
# Leaf memlets
# ---------------------------------------------------------------------------------------------------------------
def leaf_edges(state: SDFGState) -> Iterator[MultiConnectorEdge[Memlet]]:
    """Edges whose memlet produces an address in generated code: the innermost edge of each memlet path, i.e. the
    edge whose endpoint away from the memlet's data container is not a scope (map entry/exit) node. This covers
    connector bindings of tasklets and library nodes, access nodes inside scopes fed through a map entry, and
    access-node-to-access-node copies. View-defining edges (``views`` connector) and reference ``set`` edges are not
    leaves, and neither are nested SDFG bindings: a nested SDFG receives the whole container (an equivalent
    descriptor, no offset), so the path continues inside it, to its own leaf memlets (see :class:`GlobalPath`)."""
    for e in state.edges():
        if e.data.is_empty() or e.data.data is None:
            continue
        if e.dst_conn in ("set", "views") or e.src_conn == "views":
            continue
        if isinstance(e.src, nodes.NestedSDFG) or isinstance(e.dst, nodes.NestedSDFG):
            continue
        src_scope = isinstance(e.src, (nodes.EntryNode, nodes.ExitNode))
        dst_scope = isinstance(e.dst, (nodes.EntryNode, nodes.ExitNode))
        if src_scope and dst_scope:
            continue  # between two scope nodes: never innermost
        if not src_scope and not dst_scope:
            yield e  # direct binding or copy
            continue
        # One scope endpoint: innermost iff the other endpoint is not the data's own access node (the path root).
        other = e.dst if src_scope else e.src
        if not (isinstance(other, nodes.AccessNode) and other.data == e.data.data):
            yield e


def scope_node(edge: MultiConnectorEdge[Memlet]) -> nodes.Node:
    """A node whose scope is the scope the leaf memlet is accessed in (used to find lane symbols)."""
    return edge.dst if not isinstance(edge.dst, (nodes.EntryNode, nodes.ExitNode)) else edge.src


def path_root(state: SDFGState, edge: MultiConnectorEdge[Memlet]) -> tuple[nodes.AccessNode | None, bool]:
    """The access node of the memlet's data at the root of the edge's memlet path, and whether the memlet reads
    it. ``(None, False)`` if the data is read and written by the same path or the root is not its access node."""
    path = state.memlet_path(edge)
    data = edge.data.data
    first, last = path[0].src, path[-1].dst
    is_read = isinstance(first, nodes.AccessNode) and first.data == data
    is_write = isinstance(last, nodes.AccessNode) and last.data == data
    if is_read == is_write:
        return None, False
    return (first, True) if is_read else (last, False)


# ---------------------------------------------------------------------------------------------------------------
# Enclosing loops and scopes
# ---------------------------------------------------------------------------------------------------------------
def enclosing_loops(block) -> list[LoopRegion]:
    """Loop regions strictly enclosing a state or control flow region within its SDFG, innermost first."""
    result = []
    region = block.parent_graph
    while region is not None and not isinstance(region, SDFG):
        if isinstance(region, LoopRegion):
            result.append(region)
        region = region.parent_graph
    return result


def outer_loop_variables(loop: LoopRegion) -> set[str]:
    """Induction variables of the loops enclosing ``loop`` (the variables an inner cursor can be chained on)."""
    return {outer.loop_variable for outer in enclosing_loops(loop) if outer.loop_variable}


def inner_symbols(loop: LoopRegion) -> set[str]:
    """Symbols (re)defined inside the loop body: parameters of maps in the body, variables of nested loops,
    and symbols assigned on interstate edges of the body. Terms of a memlet offset that depend on them vary
    *within* an iteration and therefore belong to the per-access immediate, never to the cursor."""
    result: set[str] = set()
    for region in loop.all_control_flow_regions(recursive=False):
        if isinstance(region, LoopRegion) and region is not loop and region.loop_variable:
            result.add(region.loop_variable)
        for edge in region.edges():
            result.update(edge.data.assignments.keys())
    for state in loop.states():
        for node in state.nodes():
            if isinstance(node, nodes.EntryNode):
                result.update(node.map.params if hasattr(node, "map") else [])
    return result


def lane_symbols(
    state: SDFGState,
    node: nodes.Node,
    kernel_dims: dict[nodes.MapEntry, gpu_helpers.KernelDimensions | None] | None = None,
) -> set[str]:
    """Symbols of ``state``'s SDFG, in scope at ``node``, whose value differs between the threads of one GPU thread
    block (the *lanes*), following the thread indices code generation assigns to map parameters (see
    :func:`~dace.transformation.gpu_helpers.kernel_dimensions`):

    * parameters of ``GPU_ThreadBlock``, ``GPU_ThreadBlock_Dynamic`` and ``GPU_Warp`` maps;
    * parameters of a (non-persistent) kernel map without ``GPU_ThreadBlock`` maps (or with dynamic ones), in grid
      dimensions with more than one thread per block, where code generation adds ``threadIdx`` to the block index;
    * parameters of ``GPU_Device`` maps nested in a kernel, in grid dimensions with more than one thread per block
      (by default their dimensions have one thread per block, so their parameters vary per block only);
    * symbols of a nested SDFG in a kernel that its symbol mapping binds to lanes of the enclosing SDFG.

    Other maps in a kernel (e.g., sequential ones) run within each thread and contribute no lanes. The
    classification only steers which address terms a loop cursor carries; it never affects correctness.

    :param kernel_dims: A cache of kernel dimensions per kernel map entry, shared across calls.
    """
    return _lanes_and_kernel(state, node, {} if kernel_dims is None else kernel_dims)[0]


def _lanes_and_kernel(
    state: SDFGState,
    node: nodes.Node,
    kernel_dims: dict[nodes.MapEntry, gpu_helpers.KernelDimensions | None],
) -> tuple[set[str], tuple[nodes.MapEntry, gpu_helpers.KernelDimensions | None] | None]:
    """The lanes at ``node`` (see :func:`lane_symbols`) and the enclosing kernel map with its dimensions
    (``None`` dimensions if code generation could not derive them), or ``None`` outside of GPU kernels."""
    lanes: set[str] = set()
    kernel = None
    parent_node, parent_state = state.sdfg.parent_nsdfg_node, state.sdfg.parent
    if parent_node is not None and parent_state is not None:
        outer_lanes, kernel = _lanes_and_kernel(parent_state, parent_node, kernel_dims)
        for name, value in parent_node.symbol_mapping.items():
            value = value if isinstance(value, sp.Basic) else symbolic.pystr_to_symbolic(value)
            if symbol_names(value) & outer_lanes:
                lanes.add(str(name))

    sdict = state.scope_dict()
    scopes = []
    cur = sdict.get(node)
    while cur is not None:
        scopes.append(cur)
        cur = sdict.get(cur)
    for entry in reversed(scopes):  # Outermost first
        schedule = entry.map.schedule
        if kernel is None:
            if schedule in dtypes.GPU_KERNEL_SCHEDULES:
                if entry not in kernel_dims:
                    try:
                        kernel_dims[entry] = gpu_helpers.kernel_dimensions(entry, state.scope_subgraph(entry))
                    except NotImplementedError:
                        kernel_dims[entry] = None  # Code generation rejects the kernel too
                kernel = (entry, kernel_dims[entry])
                if schedule != dtypes.ScheduleType.GPU_Persistent:  # A persistent kernel defines no indices
                    lanes |= _thread_varying_params(entry.map.params, kernel[1], 0, by_thread_block_maps=True)
            continue  # Outside of GPU kernels there are no lanes
        if schedule in (
            dtypes.ScheduleType.GPU_ThreadBlock,
            dtypes.ScheduleType.GPU_ThreadBlock_Dynamic,
            dtypes.ScheduleType.GPU_Warp,
        ):
            lanes.update(entry.map.params)
        elif schedule == dtypes.ScheduleType.GPU_Device:
            kernel_entry, dims = kernel
            if kernel_entry.map.schedule == dtypes.ScheduleType.GPU_Persistent:
                # A grid-stride loop over the persistent grid's leading dimensions
                lanes |= _thread_varying_params(entry.map.params, dims, 0, by_thread_block_maps=True)
            else:
                # Extra grid dimensions after the kernel map's own
                lanes |= _thread_varying_params(
                    entry.map.params, dims, len(kernel_entry.map.params), by_thread_block_maps=False
                )
    return lanes, kernel


def _thread_varying_params(
    params: list[str], dims: gpu_helpers.KernelDimensions | None, first_dim: int, by_thread_block_maps: bool
) -> set[str]:
    """The parameters of a device-level map whose index code generation offsets by ``threadIdx``: the parameter
    in grid dimension ``d`` (the last parameter is in ``first_dim``, beyond the third dimension they share it) if
    the kernel's block has more than one thread in ``d``.

    :param by_thread_block_maps: Whether ``GPU_ThreadBlock`` maps take over the threads (then only a kernel with
                                 dynamic thread-block maps offsets the index), as for the kernel map itself.
    """
    if dims is None:
        return set(params)  # Unknown dimensions: conservatively all
    if by_thread_block_maps and dims.has_tbmap and not dims.has_dtbmap:
        return set()
    result = set()
    for i, param in enumerate(reversed(params)):
        d = min(first_dim + i, 2) if by_thread_block_maps else first_dim + i
        if d < 3 and (dims.block_size[d] != 1) is not False:
            result.add(param)
    return result


# ---------------------------------------------------------------------------------------------------------------
# Global memlet paths
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class PathLevel:
    """One SDFG along a :class:`GlobalPath`."""

    sdfg: SDFG
    state: SDFGState  #: The state of this SDFG holding the path.
    node: nodes.Node  #: The accessing node at the leaf level, the nested SDFG node of the level below otherwise.
    array: str  #: The name of the path's container in this SDFG.


class GlobalPath:
    """The memlet path of a leaf memlet through the chain of nested SDFGs that pass its container on whole.

    Level 0 is the leaf memlet's SDFG; level ``k + 1`` is the SDFG holding the nested SDFG node of level ``k``,
    for as long as the container is a connector whose descriptor is equivalent to the outer one (the nested SDFG
    contract). An address in the leaf SDFG's symbols is then the same physical element offset in the outer
    container once its symbols are restated through the nested SDFG nodes' symbol mappings, so loops of every
    level enclose the leaf memlet and can carry its cursor.

    Restating an expression outward (:meth:`to_level`) replaces each symbol by its symbol mapping entry; a symbol
    defined inside a nested SDFG (a map parameter, a loop variable, an interstate assignment) has no outer meaning
    and becomes an *atom*, a placeholder that varies within any loop enclosing that SDFG. Restating inward
    (:meth:`to_leaf`) passes every symbol of an outer level down to the leaf SDFG, adding symbol mapping entries
    where the nested SDFGs do not already receive it.
    """

    def __init__(
        self,
        state: SDFGState,
        edge: MultiConnectorEdge[Memlet],
        is_read: bool,
        kernel_dims: dict[nodes.MapEntry, gpu_helpers.KernelDimensions | None] | None = None,
    ):
        """
        :param kernel_dims: A cache of GPU kernel dimensions (see :func:`lane_symbols`), shared across paths.
        """
        self.is_read = is_read
        self._kernel_dims = {} if kernel_dims is None else kernel_dims
        self.levels: list[PathLevel] = [PathLevel(state.sdfg, state, scope_node(edge), edge.data.data)]
        self._mappings: list[dict[str, sp.Basic] | None] = [None]  #: Per level, its node's symbol mapping.
        self._atoms: dict[str, tuple[int, sp.Basic]] = {}  #: Atom name -> (level, symbol it stands for).
        sdfg, array = state.sdfg, edge.data.data
        while True:
            node, outer_state = sdfg.parent_nsdfg_node, sdfg.parent
            desc = sdfg.arrays[array]
            if node is None or outer_state is None or desc.transient:
                break
            bindings = (
                outer_state.in_edges_by_connector(node, array)
                if is_read
                else outer_state.out_edges_by_connector(node, array)
            )
            binding = next(iter(bindings), None)
            if binding is None or binding.data.data is None:
                break
            outer_desc = outer_state.sdfg.arrays.get(binding.data.data)
            if not flat_addressable_array(outer_desc):
                break
            if not desc.is_equivalent(outer_desc, symbol_mapping=node.symbol_mapping):
                break
            sdfg, array = outer_state.sdfg, binding.data.data
            self.levels.append(PathLevel(sdfg, outer_state, node, array))
            self._mappings.append(
                {
                    str(k): (v if isinstance(v, sp.Basic) else symbolic.pystr_to_symbolic(v))
                    for k, v in node.symbol_mapping.items()
                }
            )

    @property
    def leaf(self) -> PathLevel:
        return self.levels[0]

    @property
    def atom_names(self) -> set[str]:
        """Names of the atoms created so far (symbols defined inside a nested SDFG of the path)."""
        return set(self._atoms)

    def loops(self) -> list[tuple[int, LoopRegion]]:
        """``(level, loop)`` of every loop enclosing the leaf memlet along the path, innermost first."""
        return [(k, loop) for k, level in enumerate(self.levels) for loop in enclosing_loops(level.state)]

    def find_loop(self, label: str, variable: str) -> tuple[int, LoopRegion] | None:
        """The innermost enclosing loop with the given label and induction variable, or ``None``."""
        for k, loop in self.loops():
            if loop.label == label and loop.loop_variable == variable:
                return k, loop
        return None

    def to_level(self, expr, level: int, source: int = 0) -> sp.Basic:
        """Restate an expression in the symbols of level ``source`` (default: the leaf SDFG) in the symbols of
        ``level`` and atoms."""
        expr = sp.sympify(expr)
        for k in range(source, level):
            mapping = self._mappings[k + 1]
            repl = {}
            for s in expr.free_symbols:
                if s.name in self._atoms:
                    continue
                if s.name in mapping:
                    repl[s] = mapping[s.name]
                else:
                    repl[s] = self._atom(k, s)
            expr = expr.xreplace(repl)
        return expr

    def lanes(self, level: int) -> set[str]:
        """Thread/lane symbols of the scopes between ``level`` and the leaf, in the symbols of ``level``."""
        result: set[str] = set()
        for k in range(level + 1):
            lv = self.levels[k]
            for name in lane_symbols(lv.state, lv.node, self._kernel_dims):
                result |= symbol_names(self.to_level(symbolic.pystr_to_symbolic(name), level, source=k))
        return result

    def to_leaf(self, expr, level: int) -> sp.Basic:
        """Restate an expression in the symbols of ``level`` (and atoms) in the leaf SDFG's symbols. Symbols that
        a nested SDFG on the way does not receive yet are added to it and to its node's symbol mapping."""
        expr = self._restore_atoms(sp.sympify(expr), level)
        for k in range(level, 0, -1):
            node: nodes.NestedSDFG = self.levels[k].node
            repl = {}
            for s in expr.free_symbols:
                if s.name in self._atoms:
                    continue
                repl[s] = self._pass_down(node, s, k)
            expr = self._restore_atoms(expr.xreplace(repl), k - 1)
        return expr

    def _pass_down(self, node: nodes.NestedSDFG, sym: sp.Basic, level: int) -> sp.Basic:
        """The symbol of ``node``'s SDFG that receives the level-``level`` symbol ``sym``, created if needed."""
        inner = node.sdfg
        # The node's live mapping: another path may already have passed the symbol down.
        for key, value in node.symbol_mapping.items():
            value = value if isinstance(value, sp.Basic) else symbolic.pystr_to_symbolic(value)
            if isinstance(value, sp.Symbol) and value.name == sym.name and str(key) in inner.symbols:
                return symbolic.symbol(str(key), inner.symbols[str(key)])
        dtype = self.levels[level].sdfg.symbols.get(sym.name, getattr(sym, "dtype", symbolic.DEFAULT_SYMBOL_TYPE))
        base = sym.name if sym.name.startswith("__dace_") else f"__dace_ap_{sym.name}"
        name, n = base, 1
        while name in inner.symbols or name in inner.arrays or name in node.symbol_mapping:
            name = f"{base}_{n}"
            n += 1
        inner.add_symbol(name, dtype)
        node.symbol_mapping[name] = sym
        self._mappings[level][name] = sym
        return symbolic.symbol(name, dtype)

    def _atom(self, level: int, sym: sp.Basic) -> sp.Basic:
        name = f"__dace_atom{level}_{sym.name}"
        self._atoms.setdefault(name, (level, sym))
        return symbolic.symbol(name)

    def _restore_atoms(self, expr: sp.Basic, level: int) -> sp.Basic:
        repl = {s: self._atoms[s.name][1] for s in expr.free_symbols if self._atoms.get(s.name, (None,))[0] == level}
        return expr.xreplace(repl) if repl else expr
