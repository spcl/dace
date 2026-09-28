# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that hoists kernel-local transients out of GPU kernels into device-global allocations."""
import ast
import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple

import sympy

from dace import SDFG, SDFGState, data as dt, dtypes, properties, subsets, symbolic, utils
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import is_devicelevel_gpu, nodes
from dace.sdfg.replace import replace_properties_dict
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
from ordered_set import OrderedSet

# Deliberately NOT ``dtypes.GPU_SCHEDULES``: that also includes dynamic/persistent thread-block
# schedules this pass does not lift.
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
    while True:
        entry = state.entry_node(node)
        while entry is not None:
            scopes.append((entry, state))
            entry = state.entry_node(entry)
        if state.sdfg.parent_nsdfg_node is None:
            return scopes
        node, state = state.sdfg.parent_nsdfg_node, state.sdfg.parent


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

    def __init__(self, array_name: str, prefix: List[str]):
        self.array_name = array_name
        self.prefix = [ast.parse(expr, mode='eval').body for expr in prefix]
        self.changed = False

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == self.array_name:
            existing = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            node.slice = ast.Tuple(elts=copy.deepcopy(self.prefix) + existing, ctx=ast.Load())
            self.changed = True
        return node


def prepend_subscript_indices(code: str, array_name: str, prefix: List[str]) -> Optional[str]:
    """``code`` with ``prefix`` prepended to each ``array_name`` subscript, or ``None`` if nothing changed."""
    if array_name not in code:
        return None
    prefixer = SubscriptPrefixer(array_name, prefix)
    tree = prefixer.visit(ast.parse(code))
    return ast.unparse(ast.fix_missing_locations(tree)) if prefixer.changed else None


def free_symbol_names(exprs) -> OrderedSet:
    return OrderedSet(str(sym) for expr in exprs for sym in symbolic.pystr_to_symbolic(expr).free_symbols)


def sdfg_chain(inner: SDFG, outer: SDFG) -> List[SDFG]:
    """``inner`` and its ancestors up to and including ``outer``, innermost first."""
    chain = [inner]
    while chain[-1] is not outer:
        chain.append(chain[-1].parent_sdfg)
    return chain


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
            else:
                warnings.warn(f"Transient array '{name}' with storage type GPU_Global detected inside kernel "
                              f"{kernel}. GPU_Global memory cannot be allocated within GPU kernels; the array "
                              f"will be lifted outside the kernel as a non-transient GPU_Global array.")
                self.move_array(name, owner, kernel, kernel_state)
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

    def move_array(self, name: str, owner: SDFG, kernel: nodes.MapEntry, kernel_state: SDFGState) -> None:
        """Give ``owner``'s transient ``name`` one slice per kernel iteration and allocate it outside the kernel."""
        accesses = [(node, state) for state in owner.all_states() for node in state.data_nodes() if node.data == name]
        levels = self.slice_levels(name, accesses)
        desc = owner.arrays[name]
        new_shape, new_strides, new_total_size, new_offsets = self.get_new_shape_info(
            desc, [entry for entry, _ in reversed(levels)])

        hierarchy = sdfg_chain(owner, kernel_state.sdfg)
        name = self.free_name(name, hierarchy)
        needed = self.prefix_accesses(owner, name, levels)
        needed |= free_symbol_names(new_shape[:len(new_shape) - len(desc.shape)])
        desc.set_shape(new_shape=new_shape, strides=new_strides, total_size=new_total_size, offset=new_offsets)

        if len(hierarchy) == 1:
            self.carry_out_of_kernel(name, desc, levels, accesses, kernel, kernel_state)
            return
        # Control flow runs outside the owner's own maps, so only levels above the owner give it an index.
        point = None
        if all(state.sdfg is not owner for _, state in levels):
            point = [symbolic.symstr(begin) for begin, _, _ in lift_prefix(levels, accesses[0][1], accesses[0][0])]
        self.prefix_control_flow(owner, name, point)
        self.bind_symbols(hierarchy, needed)
        desc.transient = False
        self.lift_array_through_nested_sdfgs(name, kernel, hierarchy)

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

    def prefix_accesses(self, owner: SDFG, name: str, levels: List[Scope]) -> OrderedSet:
        """Prefix every memlet of ``name`` in ``owner`` with its slice; returns the symbols the prefixes name."""
        needed = OrderedSet()
        for state in owner.all_states():
            for edge in state.edges():
                if self.touches(edge, name):
                    prefix = lift_prefix(levels, state, edge.src)
                    needed |= free_symbol_names(bound for rng in prefix for bound in rng)
                    self.prefix_memlet(edge, name, prefix)
        return needed

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

        def rewrite(code: str) -> Optional[str]:
            new_code = prepend_subscript_indices(code, name, point or ['0'])
            if new_code is not None and point is None:
                raise NotImplementedError(f"Control flow of {sdfg.name} reads '{name}', which varies per GPU thread")
            return new_code

        def block(code: Optional[CodeBlock]) -> Optional[CodeBlock]:
            if code is None or code.language is not dtypes.Language.Python:
                return code
            new_code = rewrite(code.as_string)
            return code if new_code is None else CodeBlock(new_code, dtypes.Language.Python)

        for cfg in sdfg.all_control_flow_regions():
            for edge in cfg.edges():
                for var, value in edge.data.assignments.items():
                    edge.data.assignments[var] = rewrite(str(value)) or value
                edge.data.condition = block(edge.data.condition)
            if isinstance(cfg, LoopRegion):
                cfg.init_statement = block(cfg.init_statement)
                cfg.loop_condition = block(cfg.loop_condition)
                cfg.update_statement = block(cfg.update_statement)
            elif isinstance(cfg, ConditionalBlock):
                cfg.branches[:] = [(block(cond), branch) for cond, branch in cfg.branches]

    @staticmethod
    def bind_symbols(hierarchy: List[SDFG], needed: OrderedSet) -> None:
        """Bind each of ``needed`` by name into every nested SDFG from the kernel's down to the owner."""
        for sdfg in reversed(hierarchy[:-1]):
            nsdfg_node = sdfg.parent_nsdfg_node
            defined = sdfg.parent.symbols_defined_at(nsdfg_node)
            for name in needed:
                if name not in defined:
                    continue  # Defined further down, by a map inside this SDFG.
                bound = nsdfg_node.symbol_mapping.get(name)
                if (bound is not None and str(bound) != name) or (bound is None and assigns_symbol(sdfg, name)):
                    raise NotImplementedError(f"Cannot index the lifted array by '{name}' inside {sdfg.name}, "
                                              "which gives the name its own meaning")
                if name not in sdfg.symbols:
                    sdfg.add_symbol(name, defined[name])
                nsdfg_node.symbol_mapping[name] = name

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
