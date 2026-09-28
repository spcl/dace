# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that hoists kernel-local transients out of GPU kernels into device-global allocations."""
import ast
import copy
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import sympy

from dace import SDFG, SDFGState, data as dt, dtypes, properties, subsets, symbolic, utils
from dace.memlet import Memlet
from dace.sdfg import is_devicelevel_gpu, nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.replace import replace_properties_dict
from dace.sdfg.state import LoopRegion
from dace.transformation import helpers, pass_pipeline as ppl, transformation
from dace.transformation.passes.length_one_array_scalar_conversion import rewrite_code_slots
from ordered_set import OrderedSet

# Deliberately NOT ``dtypes.GPU_SCHEDULES``: that also includes dynamic/persistent thread-block
# schedules this pass does not lift.
logger = logging.getLogger(__name__)

GPU_HIERARCHY_SCHEDULES = (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_ThreadBlock)

#: A map scope together with the state holding it.
Scope = Tuple[nodes.MapEntry, SDFGState]


def tile_extent(max_elem, min_elem):
    """Per-iteration extent of an inner-map range."""
    if isinstance(max_elem, sympy.Min):
        for arg in max_elem.args:
            diff = symbolic.simplify(arg - min_elem)
            if diff.is_Integer and diff >= 0:
                return diff + 1
    return max_elem + 1 - min_elem


def is_register_demotable(desc: dt.Data, max_elements: int) -> bool:
    """True if ``desc`` has a literal shape of at most ``max_elements`` elements, so it fits in registers."""
    if any(symbolic.issymbolic(dim) for dim in desc.shape):
        return False
    try:
        total = int(utils.prod(desc.shape))
    except (TypeError, ValueError):
        return False  # e.g. sympy.oo: not symbolic, but not a finite integer either
    return 0 < total <= max_elements


def has_wcr_incoming(sdfg: SDFG, data_name: str) -> bool:
    """True if any memlet accumulates into ``data_name``, which a per-thread register would break."""
    return any(e.data.wcr is not None and e.data.data == data_name for nsdfg in sdfg.all_sdfgs_recursive()
               for state in nsdfg.states() for e in state.edges())


def enclosing_maps(state: SDFGState, node: nodes.Node) -> List[Scope]:
    """Map scopes enclosing ``node``, innermost first, continuing through enclosing nested SDFGs."""
    scopes: List[Scope] = []
    parent = helpers.get_parent_map(state, node)
    while parent is not None:
        scopes.append(parent)
        parent = helpers.get_parent_map(parent[1], parent[0])
    return scopes


def gpu_levels(state: SDFGState, node: nodes.Node) -> Optional[List[Scope]]:
    """GPU hierarchy maps enclosing ``node`` up to its innermost kernel, outermost first; ``None`` outside kernels."""
    levels: List[Scope] = []
    for entry, entry_state in enclosing_maps(state, node):
        if entry.map.schedule in GPU_HIERARCHY_SCHEDULES:
            levels.append((entry, entry_state))
        if entry.map.schedule == dtypes.ScheduleType.GPU_Device:
            return levels[::-1]
    return None


def encloses(scope: Scope, state: SDFGState, src: nodes.Node) -> bool:
    """Whether an edge leaving ``src`` in ``state`` runs inside ``scope``."""
    entry, entry_state = scope
    if entry_state is not state:
        # A scope in another state of the same SDFG is a sibling; one in an ancestor SDFG encloses it.
        return entry_state.sdfg is not state.sdfg
    if src is entry:
        return True
    if src is state.exit_node(entry):
        return False
    parent = state.entry_node(src)
    while parent is not None and parent is not entry:
        parent = state.entry_node(parent)
    return parent is entry


def lift_prefix(levels: List[Scope], state: SDFGState, src: nodes.Node) -> List[Tuple]:
    """Leading subset an edge leaving ``src`` gains: its own iteration inside a level, the whole level outside it."""
    prefix = []
    for scope in levels:
        entry = scope[0]
        inside = encloses(scope, state, src)
        for param, (start, end, step), origin in zip(entry.map.params, entry.map.range, entry.map.range.min_element()):
            if inside:
                index = symbolic.symbol(param) - origin
                prefix.append((index, index, 1))
            else:
                prefix.append((start - origin, end - origin, step))
    return prefix


def assigns_symbol(sdfg: SDFG, name: str) -> bool:
    """Whether ``sdfg`` gives ``name`` its own value rather than reading one from its caller."""
    if any(isinstance(cfr, LoopRegion) and cfr.loop_variable == name for cfr in sdfg.all_control_flow_regions()):
        return True
    return any(name in edge.data.assignments for edge in sdfg.all_interstate_edges())


class SubscriptPrefixer(ast.NodeTransformer):
    """Prepend fixed leading index expressions to every subscript of one array name."""

    __slots__ = ('array_name', 'prefix')

    def __init__(self, array_name: str, prefix: List[str]):
        self.array_name = array_name
        self.prefix = [ast.parse(expr, mode='eval').body for expr in prefix]

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == self.array_name:
            existing = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            node.slice = ast.Tuple(elts=copy.deepcopy(self.prefix) + existing, ctx=ast.Load())
        return node


def prepend_subscript_indices(code: str, array_name: str, prefix: List[str]) -> str:
    """``code`` with ``prefix`` prepended to each ``array_name`` subscript; code that is not Python stays."""
    if array_name not in code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    return ast.unparse(ast.fix_missing_locations(SubscriptPrefixer(array_name, prefix).visit(tree)))


def free_symbol_names(exprs) -> OrderedSet:
    return OrderedSet(str(sym) for expr in exprs for sym in symbolic.pystr_to_symbolic(expr).free_symbols)


def sdfg_chain(inner: SDFG, outer: SDFG) -> List[SDFG]:
    """``inner`` and its ancestors up to and including ``outer``, innermost first."""
    chain = [inner]
    while chain[-1] is not outer:
        chain.append(chain[-1].parent_sdfg)
    return chain


def binding_conflict(hierarchy: List[SDFG], needed: OrderedSet) -> Optional[Tuple[SDFG, str]]:
    """A nest between the kernel and the owner that means something else by one of ``needed``, if any."""
    for sdfg in reversed(hierarchy[:-1]):
        nsdfg_node = sdfg.parent_nsdfg_node
        defined = sdfg.parent.symbols_defined_at(nsdfg_node)
        for name in needed:
            bound = nsdfg_node.symbol_mapping.get(name)
            if name in defined and ((bound is not None and str(bound) != name) or
                                    (bound is None and assigns_symbol(sdfg, name))):
                return sdfg, name
    return None


def bind_symbols(hierarchy: List[SDFG], needed: OrderedSet) -> None:
    """Bind each of ``needed`` by name into every nest from the kernel's SDFG down to the owner."""
    for sdfg in reversed(hierarchy[:-1]):
        nsdfg_node = sdfg.parent_nsdfg_node
        defined = sdfg.parent.symbols_defined_at(nsdfg_node)
        # A name missing here is defined further down, by a map inside this SDFG.
        for name in (n for n in needed if n in defined):
            if name not in sdfg.symbols:
                sdfg.add_symbol(name, defined[name])
            nsdfg_node.symbol_mapping[name] = name


@dataclass(slots=True)
class LiftPlan:
    """What lifting one transient rewrites, computed before anything changes."""
    name: str
    owner: SDFG
    desc: dt.Array
    levels: List[Scope]
    accesses: List[Tuple[nodes.AccessNode, SDFGState]]
    prefixes: List[Tuple[MultiConnectorEdge, List[Tuple]]]
    shape_info: Tuple
    hierarchy: List[SDFG]
    needed: OrderedSet


@properties.make_properties
@transformation.explicit_cf_compatible
class MoveArrayOutOfKernel(ppl.Pass):
    """Lift transient ``GPU_Global`` arrays out of ``GPU_Device`` maps (kernels).

    Each array is replicated per map iteration into a disjoint outer array
    (correct per-iteration semantics instead of a single racing array). GPUs
    have no per-thread ``GPU_Device`` memory, so this is backward-compat only
    and discouraged.
    """

    register_demotion_max_elements = properties.Property(
        dtype=int,
        default=64,
        desc="Max ``prod(shape)`` for a literal-shape kernel-internal transient to be demoted "
        "from GPU_Global to per-thread Register storage. Larger transients are hoisted.",
    )

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Demote or hoist every transient ``GPU_Global`` array defined inside a kernel.

        :returns: Number of arrays handled, or ``None`` if there were none.
        :raises NotImplementedError: An array cannot be given one disjoint slice per kernel iteration.
        """
        handled = 0
        for name, desc, owner, kernel, kernel_state in self.kernel_internal_gpu_global_transients(sdfg):
            if is_register_demotable(desc, self.register_demotion_max_elements) and not has_wcr_incoming(sdfg, name):
                desc.storage = dtypes.StorageType.Register
                handled += 1
                continue
            plan = self.plan_lift(name, owner, kernel_state)
            if plan is None:
                continue
            warnings.warn(f"Transient array '{name}' with storage type GPU_Global detected inside kernel "
                          f"{kernel}. GPU_Global memory cannot be allocated within GPU kernels; the array "
                          f"will be lifted outside the kernel as a non-transient GPU_Global array.")
            self.move_array(plan, kernel, kernel_state)
            handled += 1
        self.fail_on_in_kernel_global_global(sdfg)
        return handled or None

    @staticmethod
    def kernel_internal_gpu_global_transients(
            sdfg: SDFG) -> List[Tuple[str, dt.Array, SDFG, nodes.MapEntry, SDFGState]]:
        """Transient ``GPU_Global`` arrays accessed only inside one ``GPU_Device`` map."""
        kernels: Dict[Tuple[SDFG, str], OrderedSet] = {}
        for owner in sdfg.all_sdfgs_recursive():
            for state in owner.states():
                for node in state.data_nodes():
                    desc = owner.arrays[node.data]
                    if (isinstance(desc, dt.Array) and desc.transient
                            and desc.storage is dtypes.StorageType.GPU_Global):
                        for entry, entry_state in enclosing_maps(state, node):
                            if entry.map.schedule == dtypes.ScheduleType.GPU_Device:
                                kernels.setdefault((owner, node.data), OrderedSet()).add((entry, entry_state))
                                break
                        else:
                            kernels.setdefault((owner, node.data), OrderedSet()).add(None)
        result = []
        for (owner, name), users in kernels.items():
            if None in users:
                continue
            if len(users) > 1:
                raise NotImplementedError(f"Transient '{name}' is shared by the kernels {[k for k, _ in users]}")
            kernel, kernel_state = users[0]
            result.append((name, owner.arrays[name], owner, kernel, kernel_state))
        return result

    @staticmethod
    def fail_on_in_kernel_global_global(sdfg: SDFG) -> None:
        """Raise if a transient ``GPU_Global`` copy survives inside a kernel, where nothing can allocate it."""
        offenders: List[str] = []
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for edge in state.edges():
                    if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
                        continue
                    if edge.data.is_empty() or edge.data.wcr is not None:
                        continue
                    src_desc, dst_desc = nsdfg.arrays[edge.src.data], nsdfg.arrays[edge.dst.data]
                    if not (src_desc.storage is dtypes.StorageType.GPU_Global
                            and dst_desc.storage is dtypes.StorageType.GPU_Global):
                        continue
                    if not (src_desc.transient or dst_desc.transient):
                        continue
                    if not (is_devicelevel_gpu(nsdfg, state, edge.src) or is_devicelevel_gpu(nsdfg, state, edge.dst)):
                        continue
                    offenders.append(f"  - {edge.src.data} -> {edge.dst.data} in state "
                                     f"'{state.label}' (SDFG '{nsdfg.name}')")
        if offenders:
            raise ValueError("Transient GPU_Global arrays cannot live inside a kernel scope. Offenders:\n" +
                             "\n".join(offenders))

    def plan_lift(self, name: str, owner: SDFG, kernel_state: SDFGState) -> Optional['LiftPlan']:
        """Everything the lift of ``owner``'s ``name`` rewrites, or ``None`` if a nest would misread its index."""
        accesses = [(node, state) for state in owner.all_states() for node in state.data_nodes() if node.data == name]
        levels = self.slice_levels(name, accesses)
        prefixes = [(edge, lift_prefix(levels, state, edge.src)) for state in owner.all_states()
                    for edge in state.edges() if self.touches(edge, name)]
        desc = owner.arrays[name]
        shape_info = self.get_new_shape_info(desc, [entry for entry, _ in reversed(levels)])
        needed = free_symbol_names(bound for _, prefix in prefixes for rng in prefix for bound in rng)
        needed |= free_symbol_names(shape_info[0][:len(shape_info[0]) - len(desc.shape)])
        hierarchy = sdfg_chain(owner, kernel_state.sdfg)
        conflict = binding_conflict(hierarchy, needed)
        if conflict is not None:
            logger.debug("Not lifting '%s': %s gives '%s' its own meaning", name, conflict[0].name, conflict[1])
            return None
        return LiftPlan(name, owner, desc, levels, accesses, prefixes, shape_info, hierarchy, needed)

    def move_array(self, plan: 'LiftPlan', kernel: nodes.MapEntry, kernel_state: SDFGState) -> None:
        """Give the planned transient one slice per kernel iteration and allocate it outside the kernel."""
        name = self.free_name(plan.name, plan.hierarchy)
        for edge, prefix in plan.prefixes:
            self.prefix_memlet(edge, name, prefix)
        new_shape, new_strides, new_total_size, new_offsets = plan.shape_info
        plan.desc.set_shape(new_shape=new_shape, strides=new_strides, total_size=new_total_size, offset=new_offsets)

        if len(plan.hierarchy) == 1:
            self.carry_out_of_kernel(name, plan.desc, plan.levels, plan.accesses, kernel, kernel_state)
            return
        # Control flow runs outside the owner's own maps, so only levels above the owner give it an index.
        point = None
        if all(state.sdfg is not plan.owner for _, state in plan.levels):
            node, state = plan.accesses[0]
            point = [symbolic.symstr(begin) for begin, _, _ in lift_prefix(plan.levels, state, node)]
        self.prefix_control_flow(plan.owner, name, point)
        bind_symbols(plan.hierarchy, plan.needed)
        plan.desc.transient = False
        self.lift_array_through_nested_sdfgs(name, kernel, plan.hierarchy)

    @staticmethod
    def free_name(name: str, hierarchy: List[SDFG]) -> str:
        """``name``, or a fresh one renamed into the owner if an enclosing SDFG already uses it."""
        taken = OrderedSet(n for sdfg in hierarchy[1:] for n in (*sdfg.arrays, *sdfg.symbols))
        if name not in taken:
            return name
        owner = hierarchy[0]
        new_name = utils.find_new_name(name, taken | OrderedSet(owner.arrays) | OrderedSet(owner.symbols))
        owner.replace(name, new_name)
        return new_name

    @staticmethod
    def touches(edge, name: str) -> bool:
        return edge.data.data == name or any(
            isinstance(node, nodes.AccessNode) and node.data == name for node in (edge.src, edge.dst))

    @staticmethod
    def slice_levels(name: str, accesses: List[Tuple[nodes.AccessNode, SDFGState]]) -> List[Scope]:
        """GPU levels that get a dimension: the deepest access's, which every other access's must prefix."""
        chains = [gpu_levels(state, node) for node, state in accesses]
        deepest = max(chains, key=len)
        for chain in chains:
            if [entry for entry, _ in chain] != [entry for entry, _ in deepest[:len(chain)]]:
                raise NotImplementedError(f"Cannot lift '{name}': it is accessed under sibling GPU maps")
        return deepest

    @staticmethod
    def prefix_memlet(edge, name: str, prefix: List[Tuple]) -> None:
        memlet = edge.data
        if memlet.data == name:
            memlet.subset = subsets.Range(prefix + memlet.subset.ndrange())
        elif isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == name and memlet.dst_subset is not None:
            memlet.dst_subset = subsets.Range(prefix + memlet.dst_subset.ndrange())
        elif isinstance(edge.src, nodes.AccessNode) and edge.src.data == name and memlet.src_subset is not None:
            memlet.src_subset = subsets.Range(prefix + memlet.src_subset.ndrange())

    @staticmethod
    def prefix_control_flow(sdfg: SDFG, name: str, point: Optional[List[str]]) -> None:
        """Prepend ``point`` to the subscripts of ``name`` that interstate edges, loops and branches read."""

        def rewrite(code: str) -> str:
            new_code = prepend_subscript_indices(code, name, point or ['0'])
            if point is None and new_code != code:
                raise NotImplementedError(f"Control flow of {sdfg.name} reads '{name}', which varies per GPU thread")
            return new_code

        rewrite_code_slots(sdfg, rewrite)

    def carry_out_of_kernel(self, name: str, desc: dt.Array, levels: List[Scope], accesses: List[Tuple[nodes.AccessNode,
                                                                                                       SDFGState]],
                            kernel: nodes.MapEntry, state: SDFGState) -> None:
        """Route the array from its access nearest the kernel exit out through every map exit, one slice per edge."""
        exit_node = state.exit_node(kernel)
        source = self.get_nearest_access_node([node for node, _ in accesses], exit_node, state)
        entries = [entry for entry, _ in enclosing_maps(state, source)]
        exits = [state.exit_node(entry) for entry in entries[:entries.index(kernel) + 1]]
        whole = subsets.Range.from_array(desc).ndrange()
        for src, dst in zip([source] + exits[:-1], exits):
            prefix = lift_prefix(levels, state, src)
            conn = f'IN_{name}'
            dst.add_in_connector(conn)
            dst.add_out_connector(f'OUT_{name}')
            src_conn = None if src is source else f'OUT_{name}'
            state.add_edge(src, src_conn, dst, conn,
                           Memlet(data=name, subset=subsets.Range(prefix + whole[len(prefix):])))
        state.add_edge(exit_node, f'OUT_{name}', state.add_access(name), None, Memlet.from_array(name, desc))

    def lift_array_through_nested_sdfgs(self, name: str, kernel: nodes.MapEntry, hierarchy: List[SDFG]) -> None:
        """Declare the array at every level from the owner up to the kernel's SDFG and connect it outward."""
        for inner, outer in zip(hierarchy, hierarchy[1:]):
            nsdfg_node = inner.parent_nsdfg_node
            state = inner.parent
            new_desc = copy.deepcopy(inner.arrays[name])
            symbolic.safe_replace(nsdfg_node.symbol_mapping, lambda repl: replace_properties_dict(new_desc, repl))
            outer.add_datadesc(name, new_desc)

            exits = []
            for entry, entry_state in enclosing_maps(state, nsdfg_node):
                if entry_state is not state:
                    break
                exits.append(state.exit_node(entry))
                if entry is kernel:
                    break
            nsdfg_node.add_out_connector(name)
            state.add_memlet_path(nsdfg_node,
                                  *exits,
                                  state.add_access(name),
                                  src_conn=name,
                                  memlet=Memlet.from_array(name, new_desc))
        # Transient at the outermost SDFG, so codegen allocates it instead of expecting a kernel input.
        new_desc.transient = True

    def get_new_shape_info(self, array_desc: dt.Array, map_exit_chain: List[nodes.MapEntry]):
        """New shape, strides, total size and offsets for a transient array lifted out of a kernel.

        Each GPU map prepends dimensions for per-thread disjoint slices, e.g. ``gpu_A`` of shape
        ``[64]`` under ``map[0:128, 0:32]`` becomes ``[128, 32, 64]`` (indexed ``gpu_A[x, y, :]``).
        The prepended dimensions are made the slowest-varying ones while the original dimensions
        keep their own layout, so a packed-Fortran array stays packed-Fortran on its own axes.

        :param map_exit_chain: MapEntry nodes between array and kernel exit, innermost first.
        :returns: ``(new_shape, new_strides, new_total_size, new_offsets)``.
        :raises NotImplementedError: The array is neither packed-C nor packed-Fortran.
        """
        if array_desc.is_packed_c_strides():
            inner_order = list(reversed(range(len(array_desc.shape))))
        elif array_desc.is_packed_fortran_strides():
            inner_order = list(range(len(array_desc.shape)))
        else:
            raise NotImplementedError(f'Cannot lift {array_desc}: only packed C or Fortran strides are supported.')

        extended_size = []
        new_offsets = list(array_desc.offset)
        for next_map in map_exit_chain:
            if next_map.map.schedule not in GPU_HIERARCHY_SCHEDULES:
                continue
            extended_size = [
                tile_extent(mx, mn)
                for mx, mn in zip(next_map.map.range.max_element(), next_map.map.range.min_element())
            ] + extended_size
            new_offsets = [0 for _ in next_map.map.params] + new_offsets

        prepended = len(extended_size)
        # ``strides_from_layout`` takes the dimensions innermost-first: the original axes in their
        # own order, then the prepended ones outermost-last so they end up slowest-varying.
        layout = [d + prepended for d in inner_order] + list(reversed(range(prepended)))

        lifted = array_desc.clone()
        lifted.set_shape(extended_size + list(array_desc.shape))
        new_strides, new_total_size = lifted.strides_from_layout(*layout)
        return list(lifted.shape), list(new_strides), new_total_size, new_offsets

    @staticmethod
    def get_nearest_access_node(access_nodes: List[nodes.AccessNode], node: nodes.Node,
                                state: SDFGState) -> nodes.AccessNode:
        """Closest of ``access_nodes`` to ``node`` in ``state``, by undirected graph distance.

        :raises RuntimeError: No candidate is connected to ``node``.
        """
        visited = OrderedSet([node])
        queue = [node]
        while queue:
            current = queue.pop(0)
            if current in access_nodes:
                return current
            for neighbor in state.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        raise RuntimeError(f"No access node found connected to the given node {node}. ")
