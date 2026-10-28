# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that moves kernel-local arrays GPU registers cannot hold out of GPU kernels into device-global memory."""

import ast
import copy
import itertools
import logging
import numbers
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import sympy
from ordered_set import OrderedSet

from dace import SDFG, SDFGState, dtypes, properties, subsets, symbolic, utils
from dace import data as dt
from dace.memlet import Memlet
from dace.sdfg import is_devicelevel_gpu, nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.replace import replace_properties_dict
from dace.sdfg.state import LoopRegion
from dace.transformation import helpers, transformation
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.length_one_array_scalar_conversion import rewrite_code_slots

logger = logging.getLogger(__name__)

#: Map schedules that launch a kernel; each of their iterations gets a slice of a moved array.
KERNEL_SCHEDULES = (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_Persistent)
# Not ``dtypes.GPU_SCHEDULES``: that also holds dynamic thread-block schedules, which are not lifted.
GPU_HIERARCHY_SCHEDULES = (*KERNEL_SCHEDULES, dtypes.ScheduleType.GPU_ThreadBlock)
# Storage a kernel declares as a plain per-thread local.
THREAD_LOCAL_STORAGE = (dtypes.StorageType.Register, dtypes.StorageType.Default)

Scope = tuple[nodes.MapEntry, SDFGState]
Dim = tuple[symbolic.SymbolicType, symbolic.SymbolicType, symbolic.SymbolicType]
Prefix = list[Dim]


class LiftedLayout(NamedTuple):
    shape: list[symbolic.SymbolicType]
    strides: list[symbolic.SymbolicType]
    total_size: symbolic.SymbolicType
    offset: list[int]


def tile_extent(max_elem: symbolic.SymbolicType, min_elem: symbolic.SymbolicType) -> symbolic.SymbolicType:
    """Per-iteration extent of an inner-map range; a ``Min``-bounded tile yields its static width."""
    if isinstance(max_elem, sympy.Min):
        for arg in max_elem.args:
            diff = symbolic.simplify(arg - min_elem)
            if diff.is_Integer and diff >= 0:
                return diff + 1
    return max_elem + 1 - min_elem


def constant_size(desc: dt.Data) -> int | None:
    """The element count of ``desc``, or None if it is symbolic."""
    total = utils.prod(desc.shape)
    return int(total) if isinstance(total, numbers.Integral) else None


def fits_registers(desc: dt.Data, register_limit: int) -> bool:
    """A constant size of at most ``register_limit`` elements; persistent and external arrays outlive a thread."""
    if desc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
        return False
    size = constant_size(desc)
    return size is not None and size <= register_limit


def is_variable_length(desc: dt.Data) -> bool:
    """A per-thread local whose size is symbolic: C++ has no variable-length arrays, so no kernel can declare it."""
    return desc.storage in THREAD_LOCAL_STORAGE and constant_size(desc) is None


def is_move_candidate(desc: dt.Data, register_limit: int, only_variable_length: bool = False) -> bool:
    """A transient array a kernel cannot declare as is: ``GPU_Global``, or a per-thread local of symbolic size or
    larger than ``register_limit``; with ``only_variable_length`` just the symbolic-size per-thread local."""
    # A view or reference names other memory, set by its connector, rather than owning any
    if not isinstance(desc, dt.Array) or isinstance(desc, (dt.View, dt.Reference)) or not desc.transient:
        return False
    if only_variable_length:
        return is_variable_length(desc)
    if desc.storage is dtypes.StorageType.GPU_Global:
        return True
    return desc.storage in THREAD_LOCAL_STORAGE and not fits_registers(desc, register_limit)


def enclosing_maps(state: SDFGState, node: nodes.Node) -> list[Scope]:
    """Map scopes enclosing ``node``, innermost first, continuing through enclosing nested SDFGs."""
    scopes: list[Scope] = []
    parent = helpers.get_parent_map(state, node)
    while parent is not None:
        scopes.append(parent)
        parent = helpers.get_parent_map(parent[1], parent[0])
    return scopes


def gpu_levels(state: SDFGState, node: nodes.Node) -> list[Scope] | None:
    """GPU hierarchy maps enclosing ``node`` up to its innermost kernel, outermost first; ``None`` outside kernels."""
    levels: list[Scope] = []
    for scope in enclosing_maps(state, node):
        if scope[0].map.schedule in GPU_HIERARCHY_SCHEDULES:
            levels.append(scope)
        if scope[0].map.schedule in KERNEL_SCHEDULES:
            return levels[::-1]
    return None


def lift_prefix(levels: list[Scope], state: SDFGState, src: nodes.Node) -> Prefix:
    """Leading subset an edge leaving ``src`` gains: its own iteration inside a level, the whole level outside it."""
    prefix: Prefix = []
    for entry, entry_state in levels:
        # The exit node closes the scope, so an edge leaving it runs outside.
        closes = entry_state is state and src is state.exit_node(entry)
        inside = not closes and helpers.contained_in(state, src, entry)
        for param, (start, end, step), origin in zip(
            entry.map.params, entry.map.range, entry.map.range.min_element(), strict=True
        ):
            if inside:
                index = symbolic.symbol(param) - origin
                prefix.append((index, index, sympy.S.One))
            else:
                prefix.append((start - origin, end - origin, step))
    return prefix


def single_point(prefix: Prefix, label: str, name: str) -> list[str]:
    """The one slice index ``prefix`` names.

    :raises NotImplementedError: ``prefix`` spans several slices, so ``label`` has no single index into ``name``.
    """
    if any(begin != end for begin, end, _ in prefix):
        raise NotImplementedError(f"{label} reads '{name}' outside the kernel levels it is lifted by")
    return [symbolic.symstr(begin) for begin, _, _ in prefix]


def assigns_symbol(sdfg: SDFG, name: str) -> bool:
    """Whether ``sdfg`` gives ``name`` its own value rather than reading one from its caller."""
    if any(isinstance(cfr, LoopRegion) and cfr.loop_variable == name for cfr in sdfg.all_control_flow_regions()):
        return True
    return any(name in edge.data.assignments for edge in sdfg.all_interstate_edges())


class SubscriptPrefixer(ast.NodeTransformer):
    """Prepend fixed leading index expressions to every subscript of one array name."""

    __slots__ = ("array_name", "changed", "prefix")

    def __init__(self, array_name: str, prefix: list[str]):
        self.array_name = array_name
        self.prefix = [ast.parse(expr, mode="eval").body for expr in prefix]
        self.changed = False

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == self.array_name:
            existing = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            node.slice = ast.Tuple(elts=copy.deepcopy(self.prefix) + existing, ctx=ast.Load())
            self.changed = True
        return node


def prepend_subscript_indices(code: str, array_name: str, prefix: list[str]) -> str:
    """``code`` with ``prefix`` prepended to each ``array_name`` subscript; code that is not Python stays."""
    if array_name not in code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    prefixer = SubscriptPrefixer(array_name, prefix)
    tree = prefixer.visit(tree)
    # Unparsing reformats, so untouched code is returned as it came.
    return ast.unparse(ast.fix_missing_locations(tree)) if prefixer.changed else code


def prefix_tasklet(tasklet: nodes.Tasklet, name: str, point: list[str]) -> None:
    """Prepend ``point`` to every ``name`` subscript in ``tasklet``'s body.

    :raises NotImplementedError: A non-Python body subscripts ``name``; only Python bodies can be rewritten.
    """
    code = tasklet.code.as_string
    if tasklet.language is not dtypes.Language.Python:
        if re.search(rf"\b{re.escape(name)}\s*\[", code):
            raise NotImplementedError(f"Tasklet {tasklet.label} subscripts '{name}' in {tasklet.language.name}")
        return
    tasklet.code = properties.CodeBlock(prepend_subscript_indices(code, name, point), tasklet.language)


def prefix_memlet(edge: MultiConnectorEdge, name: str, prefix: Prefix) -> None:
    memlet = edge.data
    if memlet.data == name:
        memlet.subset = subsets.Range(prefix + memlet.subset.ndrange())
    elif isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == name and memlet.dst_subset is not None:
        memlet.dst_subset = subsets.Range(prefix + memlet.dst_subset.ndrange())
    elif isinstance(edge.src, nodes.AccessNode) and edge.src.data == name and memlet.src_subset is not None:
        memlet.src_subset = subsets.Range(prefix + memlet.src_subset.ndrange())


def prefix_control_flow(sdfg: SDFG, name: str, point: list[str] | None) -> None:
    """Prepend ``point`` to the subscripts of ``name`` that interstate edges, loops and branches read.

    :raises NotImplementedError: ``point`` is ``None`` (the array varies per thread) and control flow reads it.
    """

    def rewrite(code: str) -> str:
        new_code = prepend_subscript_indices(code, name, point or ["0"])
        if point is None and new_code != code:
            raise NotImplementedError(f"Control flow of {sdfg.name} reads '{name}', which varies per GPU thread")
        return new_code

    rewrite_code_slots(sdfg, rewrite)


def sdfg_chain(inner: SDFG, outer: SDFG) -> list[SDFG]:
    """``inner`` and its ancestors up to and including ``outer``, innermost first."""
    chain = [inner]
    while chain[-1] is not outer:
        chain.append(chain[-1].parent_sdfg)
    return chain


def size_known_at_launch(
    desc: dt.Array, hierarchy: list[SDFG], kernel: nodes.MapEntry, kernel_state: SDFGState
) -> bool:
    """``desc``'s shape, restated through the symbol mapping of every nest up to the kernel's SDFG, names only symbols
    defined where the kernel launches: a size the kernel computes (a CSR row length) cannot size an allocation made
    before it."""
    shape = [symbolic.pystr_to_symbolic(extent) for extent in desc.shape]
    for sdfg in hierarchy[:-1]:
        mapping = sdfg.parent_nsdfg_node.symbol_mapping
        names = {str(sym) for extent in shape for sym in extent.free_symbols}
        if any(name not in mapping or assigns_symbol(sdfg, name) for name in names):
            return False
        replacements = {symbolic.symbol(name): symbolic.pystr_to_symbolic(mapping[name]) for name in names}
        shape = [extent.subs(replacements) for extent in shape]
    defined = kernel_state.symbols_defined_at(kernel)
    return all(
        str(sym) in defined and str(sym) not in kernel.map.params for extent in shape for sym in extent.free_symbols
    )


def binding_conflict(hierarchy: list[SDFG], needed: list[str]) -> tuple[SDFG, str] | None:
    """A nest between the kernel and the owner that means something else by one of ``needed``, if any."""
    for sdfg in reversed(hierarchy[:-1]):
        nsdfg_node = sdfg.parent_nsdfg_node
        defined = sdfg.parent.symbols_defined_at(nsdfg_node)
        for name in needed:
            bound = nsdfg_node.symbol_mapping.get(name)
            if name in defined and (assigns_symbol(sdfg, name) if bound is None else str(bound) != name):
                return sdfg, name
    return None


def bind_symbols(hierarchy: list[SDFG], needed: list[str]) -> None:
    """Bind each of ``needed`` by name into every nest from the kernel's SDFG down to the owner."""
    for sdfg in reversed(hierarchy[:-1]):
        nsdfg_node = sdfg.parent_nsdfg_node
        defined = sdfg.parent.symbols_defined_at(nsdfg_node)
        # A name missing here is defined further down, by a map inside this SDFG.
        for name in (n for n in needed if n in defined):
            if name not in sdfg.symbols:
                sdfg.add_symbol(name, defined[name])
            nsdfg_node.symbol_mapping[name] = name


def lifted_layout(desc: dt.Array, levels: list[nodes.MapEntry], coalesced: bool = True) -> LiftedLayout:
    """``desc`` with one leading dimension per GPU map parameter.

    ``gpu_A[64]`` under ``map[0:128, 0:32]`` becomes ``[128, 32, 64]`` indexed ``gpu_A[x, y, :]``. Coalesced, the map
    parameters vary fastest (strides ``[32, 1, 4096]``), so neighboring threads touch neighboring elements; contiguous,
    they vary slowest (``[2048, 64, 1]``), so the slice of one iteration is consecutive. Either way the own axes keep
    their C or Fortran order.

    :param levels: GPU map entries between the array and the kernel exit, innermost first.
    :param coalesced: Map parameters vary fastest instead of slowest.
    :raises NotImplementedError: The array is neither packed-C nor packed-Fortran.
    """
    if desc.is_packed_c_strides():
        inner_order = list(reversed(range(len(desc.shape))))
    elif desc.is_packed_fortran_strides():
        inner_order = list(range(len(desc.shape)))
    else:
        raise NotImplementedError(f"Cannot lift {desc}: only packed C or Fortran strides are supported.")

    extended_size: list[symbolic.SymbolicType] = []
    offset = list(desc.offset)
    for level in levels:
        rng = level.map.range
        extended_size = [
            tile_extent(mx, mn) for mx, mn in zip(rng.max_element(), rng.min_element(), strict=True)
        ] + extended_size
        offset = [0] * len(level.map.params) + offset

    prepended = len(extended_size)
    # ``strides_from_layout`` takes dimensions innermost-first
    own, mapped = [d + prepended for d in inner_order], list(reversed(range(prepended)))
    layout = mapped + own if coalesced else own + mapped
    lifted = desc.clone()
    lifted.set_shape(extended_size + list(desc.shape))
    strides, total_size = lifted.strides_from_layout(*layout)
    return LiftedLayout(list(lifted.shape), list(strides), total_size, offset)


def slice_levels(name: str, accesses: list[tuple[nodes.AccessNode, SDFGState]]) -> list[Scope]:
    """GPU levels that get a dimension: the deepest access's, which every other access's must prefix."""
    chains = [gpu_levels(state, node) or [] for node, state in accesses]
    deepest = max(chains, key=len)
    for chain in chains:
        if [entry for entry, _ in chain] != [entry for entry, _ in deepest[: len(chain)]]:
            raise NotImplementedError(f"Cannot lift '{name}': it is accessed under sibling GPU maps")
    return deepest


def nearest_access_node(access_nodes: list[nodes.AccessNode], node: nodes.Node, state: SDFGState) -> nodes.Node:
    """Closest of ``access_nodes`` to ``node`` in ``state``, by undirected graph distance."""
    visited = OrderedSet([node])
    queue = deque([node])
    while queue:
        current = queue.popleft()
        if current in access_nodes:
            return current
        for neighbor in state.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    raise RuntimeError(f"No access node is connected to {node}")


@dataclass(slots=True)
class LiftPlan:
    """What lifting one transient rewrites, computed before anything changes."""

    name: str
    owner: SDFG
    desc: dt.Array
    levels: list[Scope]
    accesses: list[tuple[nodes.AccessNode, SDFGState]]
    prefixes: list[tuple[MultiConnectorEdge, Prefix]]
    layout: LiftedLayout
    hierarchy: list[SDFG]
    needed: list[str]


def plan_lift(
    name: str, owner: SDFG, kernel: nodes.MapEntry, kernel_state: SDFGState, coalesced: bool
) -> LiftPlan | None:
    """Everything the lift of ``owner``'s ``name`` rewrites, or ``None`` if its size is only known inside the kernel
    or a nest would misread its index."""
    hierarchy = sdfg_chain(owner, kernel_state.sdfg)
    if not size_known_at_launch(owner.arrays[name], hierarchy, kernel, kernel_state):
        logger.debug("Not lifting '%s': the kernel computes its size", name)
        return None
    accesses = [(node, state) for state in owner.states() for node in state.data_nodes() if node.data == name]
    levels = slice_levels(name, accesses)
    prefixes = [
        (edge, lift_prefix(levels, state, edge.src))
        for state in owner.states()
        for edge in state.edges()
        if edge.data.data == name
        or any(isinstance(node, nodes.AccessNode) and node.data == name for node in (edge.src, edge.dst))
    ]
    desc = owner.arrays[name]
    layout = lifted_layout(desc, [entry for entry, _ in reversed(levels)], coalesced)
    bounds = [bound for _, prefix in prefixes for dim in prefix for bound in dim]
    bounds += layout.shape[: len(layout.shape) - len(desc.shape)]
    needed = sorted({str(sym) for bound in bounds for sym in symbolic.pystr_to_symbolic(bound).free_symbols})
    conflict = binding_conflict(hierarchy, needed)
    if conflict is not None:
        logger.debug("Not lifting '%s': %s gives '%s' its own meaning", name, conflict[0].name, conflict[1])
        return None
    return LiftPlan(name, owner, desc, levels, accesses, prefixes, layout, hierarchy, needed)


def reshape_descendants(
    sdfg: SDFG, name: str, desc: dt.Array, prefix_at: Callable[[SDFGState, nodes.Node], Prefix]
) -> None:
    """Give every nest below ``sdfg`` that ``name`` reaches the lifted descriptor and index its accesses by the slice
    the nest sees: a nested SDFG's data has the shape of the data connected to it (No-View nested SDFGs)."""
    for state in sdfg.states():
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            conns = {e.dst_conn for e in state.in_edges(node) if e.data.data == name} | {
                e.src_conn for e in state.out_edges(node) if e.data.data == name
            }
            if not conns:
                continue
            prefix = prefix_at(state, node)
            point = single_point(prefix, f"Nest {node.label}", name)
            inner = node.sdfg
            defined = state.symbols_defined_at(node)
            needed = {
                str(sym)
                for expr in (*desc.shape, *(begin for begin, _, _ in prefix))
                for sym in symbolic.pystr_to_symbolic(expr).free_symbols
            }
            for sym in sorted(needed):
                if sym not in inner.symbols:
                    inner.add_symbol(sym, defined[sym])
                node.symbol_mapping[sym] = sym
            for conn in sorted(conns):
                inner.arrays[conn] = copy.deepcopy(desc)
                inner.arrays[conn].transient = False
                for inner_state in inner.states():
                    for edge in inner_state.edges():
                        prefix_memlet(edge, conn, prefix)
                    for tasklet in (n for n in inner_state.nodes() if isinstance(n, nodes.Tasklet)):
                        prefix_tasklet(tasklet, conn, point)
                prefix_control_flow(inner, conn, point)
                reshape_descendants(inner, conn, inner.arrays[conn], lambda *_, fixed=prefix: fixed)


def carry_out_of_kernel(plan: LiftPlan, name: str, kernel: nodes.MapEntry, state: SDFGState) -> None:
    """Order an access outside the kernel after the access nearest its exit, so the array is allocated outside.

    The route carries no data: through a scope exit a memlet from ``name`` to ``name`` is a copy, which every thread
    would run onto the first slice.
    """
    exit_node = state.exit_node(kernel)
    source = nearest_access_node([node for node, _ in plan.accesses], exit_node, state)
    entries = [entry for entry, _ in enclosing_maps(state, source)]
    exits = [state.exit_node(entry) for entry in entries[: entries.index(kernel) + 1]]
    route = [source, *exits, state.add_access(name)]
    for src, dst in itertools.pairwise(route):
        state.add_nedge(src, dst, Memlet())


def lift_through_nested_sdfgs(name: str, kernel: nodes.MapEntry, hierarchy: list[SDFG]) -> None:
    """Declare the array at every level from the owner up to the kernel's SDFG and connect it outward."""
    for inner, outer in itertools.pairwise(hierarchy):
        nsdfg_node = inner.parent_nsdfg_node
        state = inner.parent
        new_desc = copy.deepcopy(inner.arrays[name])
        symbolic.safe_replace(
            nsdfg_node.symbol_mapping, lambda repl, desc=new_desc: replace_properties_dict(desc, repl)
        )
        outer.add_datadesc(name, new_desc)

        exits = []
        for entry, entry_state in enclosing_maps(state, nsdfg_node):
            if entry_state is not state:
                break
            exits.append(state.exit_node(entry))
            if entry is kernel:
                break
        nsdfg_node.add_out_connector(name)
        state.add_memlet_path(
            nsdfg_node, *exits, state.add_access(name), src_conn=name, memlet=Memlet.from_array(name, new_desc)
        )
    # The outermost descriptor is allocated by codegen instead of expected as a kernel input.
    hierarchy[-1].arrays[name].transient = True


def move_array(plan: LiftPlan, kernel: nodes.MapEntry, kernel_state: SDFGState) -> None:
    """Give the planned transient one slice per kernel iteration and allocate it outside the kernel."""
    name = plan.name
    taken = OrderedSet(n for sdfg in plan.hierarchy[1:] for n in (*sdfg.arrays, *sdfg.symbols))
    if name in taken:
        name = utils.find_new_name(name, taken | OrderedSet(plan.owner.arrays) | OrderedSet(plan.owner.symbols))
        plan.owner.replace(plan.name, name)
    for edge, prefix in plan.prefixes:
        prefix_memlet(edge, name, prefix)
    # A tasklet with inlined connectors names the array in its body and gains the same leading index.
    for state in plan.owner.states():
        for node in state.nodes():
            if isinstance(node, nodes.Tasklet) and name in node.code.as_string:
                prefix_tasklet(node, name, single_point(lift_prefix(plan.levels, state, node), node.label, name))
    plan.desc.set_shape(
        new_shape=plan.layout.shape,
        strides=plan.layout.strides,
        total_size=plan.layout.total_size,
        offset=plan.layout.offset,
    )
    # A nest below the owner indexes by the kernel parameters, so the owner must already be bound to them.
    bind_symbols(plan.hierarchy, plan.needed)
    reshape_descendants(plan.owner, name, plan.desc, lambda state, node: lift_prefix(plan.levels, state, node))

    if len(plan.hierarchy) == 1:
        carry_out_of_kernel(plan, name, kernel, kernel_state)
        return
    # Control flow runs outside the owner's own maps, so only levels above the owner give it an index.
    point = None
    if all(state.sdfg is not plan.owner for _, state in plan.levels):
        node, state = plan.accesses[0]
        point = [symbolic.symstr(begin) for begin, _, _ in lift_prefix(plan.levels, state, node)]
    prefix_control_flow(plan.owner, name, point)
    plan.desc.transient = False
    lift_through_nested_sdfgs(name, kernel, plan.hierarchy)


def move_candidates(
    sdfg: SDFG, register_limit: int, only_variable_length: bool = False
) -> list[tuple[str, dt.Array, SDFG, nodes.MapEntry, SDFGState]]:
    """Transients that :func:`is_move_candidate`, accessed only inside one kernel map.

    :raises NotImplementedError: A candidate is accessed by several kernels.
    """
    users: dict[tuple[SDFG, str], list[Scope | None]] = {}
    for owner in sdfg.all_sdfgs_recursive():
        for state in owner.states():
            for node in state.data_nodes():
                if is_move_candidate(owner.arrays[node.data], register_limit, only_variable_length):
                    kernel = next(
                        (s for s in enclosing_maps(state, node) if s[0].map.schedule in KERNEL_SCHEDULES),
                        None,
                    )
                    found = users.setdefault((owner, node.data), [])
                    if kernel not in found:
                        found.append(kernel)
    result = []
    for (owner, name), found in users.items():
        kernels = [k for k in found if k is not None]
        if len(kernels) < len(found):
            continue
        if len(kernels) > 1:
            raise NotImplementedError(f"Transient '{name}' is shared by the kernels {[k for k, _ in kernels]}")
        kernel, kernel_state = kernels[0]
        result.append((name, owner.arrays[name], owner, kernel, kernel_state))
    return result


def check_no_global_copy_in_kernel(sdfg: SDFG) -> None:
    """:raises ValueError: A copy into or out of a transient ``GPU_Global`` array survives inside a kernel, where
    nothing can allocate it."""
    offenders: list[str] = []
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for edge in state.edges():
                if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
                    continue
                if edge.data.is_empty() or edge.data.wcr is not None:
                    continue
                descs = (nsdfg.arrays[edge.src.data], nsdfg.arrays[edge.dst.data])
                # An edge into or out of a view aliases its data; it copies nothing.
                if any(isinstance(desc, dt.View) for desc in descs):
                    continue
                if (
                    all(desc.storage is dtypes.StorageType.GPU_Global for desc in descs)
                    and any(desc.transient for desc in descs)
                    and (is_devicelevel_gpu(nsdfg, state, edge.src) or is_devicelevel_gpu(nsdfg, state, edge.dst))
                ):
                    offenders.append(f"  - {edge.src.data} -> {edge.dst.data} in state '{state.label}' ({nsdfg.name})")
    if offenders:
        raise ValueError(
            "Transient GPU_Global arrays cannot live inside a kernel scope. Offenders:\n" + "\n".join(offenders)
        )


@properties.make_properties
@transformation.explicit_cf_compatible
class MoveArrayOutOfKernel(ppl.Pass):
    """Move transient arrays a GPU kernel cannot hold in registers out of ``GPU_Device`` maps (kernels).

    A per-thread (``Register`` or ``Default``) array of constant size up to ``register_limit`` elements stays where it
    is, and a ``GPU_Global`` one that small is demoted to ``Register``. Every other kernel-local array (symbolic size,
    larger than the limit, or ``GPU_Global`` and accumulated into) becomes a ``GPU_Global`` array outside the kernel
    with one disjoint slice per kernel iteration.
    """

    register_limit = properties.Property(
        dtype=int,
        default=16,
        desc="Largest element count of a constant-size kernel-local array kept in per-thread registers; larger and "
        "symbolically-sized ones are moved to GPU_Global memory outside the kernel.",
    )

    only_variable_length = properties.Property(
        dtype=bool,
        default=False,
        desc="Move only per-thread (Register or Default) arrays of symbolic size, which no kernel can declare; "
        "code generation runs the pass this way.",
    )

    layout = properties.Property(
        dtype=str,
        default="coalesced",
        choices=["coalesced", "contiguous"],
        desc="Order of a moved array: ``coalesced`` puts the kernel iterations fastest, so neighboring GPU threads "
        "access neighboring elements; ``contiguous`` keeps the slice of one iteration consecutive.",
    )

    def __init__(
        self, register_limit: int | None = None, only_variable_length: bool | None = None, layout: str | None = None
    ) -> None:
        super().__init__()
        if layout is not None:
            self.layout = layout
        if register_limit is not None:
            self.register_limit = register_limit
        if only_variable_length is not None:
            self.only_variable_length = only_variable_length

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, object]) -> int | None:
        """Demote or move every kernel-local array registers cannot hold.

        :returns: Number of arrays demoted or moved, or ``None`` if there were none.
        :raises NotImplementedError: An array cannot be given one disjoint slice per kernel iteration.
        :raises ValueError: A transient ``GPU_Global`` copy is left inside a kernel.
        """
        # A register cannot accumulate across threads, so a WCR target stays in memory.
        accumulated = OrderedSet(
            e.data.data
            for nsdfg in sdfg.all_sdfgs_recursive()
            for state in nsdfg.states()
            for e in state.edges()
            if e.data.wcr is not None
        )
        handled = 0
        for name, desc, owner, kernel, kernel_state in move_candidates(
            sdfg, self.register_limit, self.only_variable_length
        ):
            if name not in accumulated and fits_registers(desc, self.register_limit):
                desc.storage = dtypes.StorageType.Register
                handled += 1
                continue
            plan = plan_lift(name, owner, kernel, kernel_state, self.layout == "coalesced")
            if plan is None:
                continue
            desc.storage = dtypes.StorageType.GPU_Global
            move_array(plan, kernel, kernel_state)
            handled += 1
        if not self.only_variable_length:
            check_no_global_copy_in_kernel(sdfg)
        return handled or None
