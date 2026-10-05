# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that hoists kernel-local transients out of GPU kernels into device-global allocations."""
import ast
import copy
import itertools
import logging
import numbers
import re
import warnings
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass

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

logger = logging.getLogger(__name__)

# Not ``dtypes.GPU_SCHEDULES``: that also holds dynamic and persistent thread-block schedules, which are not lifted.
GPU_HIERARCHY_SCHEDULES = (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_ThreadBlock)

Scope = tuple[nodes.MapEntry, SDFGState]
Dim = tuple[symbolic.SymbolicType, symbolic.SymbolicType, symbolic.SymbolicType]
Prefix = list[Dim]


def tile_extent(max_elem: symbolic.SymbolicType, min_elem: symbolic.SymbolicType) -> symbolic.SymbolicType:
    """Per-iteration extent of an inner-map range; a ``Min``-bounded tile yields its static width."""
    if isinstance(max_elem, sympy.Min):
        for arg in max_elem.args:
            diff = symbolic.simplify(arg - min_elem)
            if diff.is_Integer and diff >= 0:
                return diff + 1
    return max_elem + 1 - min_elem


def is_register_demotable(desc: dt.Data, max_elements: int) -> bool:
    """A literal shape of at most ``max_elements``; persistent and external arrays outlive a thread's registers."""
    if desc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
        return False
    total = utils.prod(desc.shape)
    return isinstance(total, numbers.Integral) and 0 < total <= max_elements


# Storage emitted as a plain local declaration in device code; shared memory has its own sizing rules.
DEVICE_LOCAL_STORAGE = (dtypes.StorageType.Register, dtypes.StorageType.Default)


def needs_global_memory(desc: dt.Data) -> bool:
    """A kernel-internal transient that can only live in device-global memory: ``GPU_Global``, or device-local
    with a symbolic extent, which would be a variable-length array that nvcc rejects."""
    if not isinstance(desc, dt.Array) or isinstance(desc, dt.View) or not desc.transient:
        return False
    if desc.storage is dtypes.StorageType.GPU_Global:
        return True
    return desc.storage in DEVICE_LOCAL_STORAGE and any(symbolic.issymbolic(dim) for dim in desc.shape)


def gpu_levels(state: SDFGState, node: nodes.Node) -> list[Scope] | None:
    """GPU hierarchy maps enclosing ``node`` up to its innermost kernel, outermost first; ``None`` outside kernels."""
    levels: list[Scope] = []
    for scope in helpers.get_parent_maps(state, node):
        if scope[0].map.schedule in GPU_HIERARCHY_SCHEDULES:
            levels.append(scope)
        if scope[0].map.schedule == dtypes.ScheduleType.GPU_Device:
            return levels[::-1]
    return None


def lift_prefix(levels: list[Scope], state: SDFGState, src: nodes.Node) -> Prefix:
    """Leading subset an edge leaving ``src`` gains: its own iteration inside a level, the whole level outside it."""
    prefix: Prefix = []
    for entry, entry_state in levels:
        # The exit node closes the scope, so an edge leaving it runs outside.
        closes = entry_state is state and src is state.exit_node(entry)
        inside = not closes and helpers.contained_in(state, src, entry)
        for param, (start, end, step), origin in zip(entry.map.params,
                                                     entry.map.range,
                                                     entry.map.range.min_element(),
                                                     strict=True):
            if inside:
                index = symbolic.symbol(param) - origin
                prefix.append((index, index, sympy.S.One))
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

    __slots__ = ('array_name', 'changed', 'prefix')

    def __init__(self, array_name: str, prefix: list[str]):
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


def sdfg_chain(inner: SDFG, outer: SDFG) -> list[SDFG]:
    """``inner`` and its ancestors up to and including ``outer``, innermost first."""
    chain = [inner]
    while chain[-1] is not outer:
        chain.append(chain[-1].parent_sdfg)
    return chain


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


@dataclass(slots=True)
class LiftPlan:
    """What lifting one transient rewrites, computed before anything changes."""
    name: str
    owner: SDFG
    desc: dt.Array
    levels: list[Scope]
    accesses: list[tuple[nodes.AccessNode, SDFGState]]
    prefixes: list[tuple[MultiConnectorEdge, Prefix]]
    shape_info: tuple[list[symbolic.SymbolicType], list[symbolic.SymbolicType], symbolic.SymbolicType, list[int]]
    hierarchy: list[SDFG]
    needed: list[str]


@properties.make_properties
@transformation.explicit_cf_compatible
class MoveArrayOutOfKernel(ppl.Pass):
    """Lift transient ``GPU_Global`` arrays out of ``GPU_Device`` maps (kernels).

    Each array is replicated per map iteration into a disjoint outer array. GPUs have no per-thread
    ``GPU_Device`` memory, so this exists for backward compatibility and is discouraged.
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

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, object]) -> int | None:
        """Demote or hoist every transient ``GPU_Global`` array defined inside a kernel.

        :returns: Number of arrays handled, or ``None`` if there were none.
        :raises NotImplementedError: An array cannot be given one disjoint slice per kernel iteration.
        """
        # A register cannot accumulate across threads, so a WCR target stays in memory.
        accumulated = OrderedSet(e.data.data for nsdfg in sdfg.all_sdfgs_recursive() for state in nsdfg.states()
                                 for e in state.edges() if e.data.wcr is not None)
        handled = 0
        for name, desc, owner, kernel, kernel_state in self.kernel_internal_gpu_global_transients(sdfg):
            if name not in accumulated and is_register_demotable(desc, self.register_demotion_max_elements):
                desc.storage = dtypes.StorageType.Register
                handled += 1
                continue
            plan = self.plan_lift(name, owner, kernel_state)
            if plan is None:
                continue
            reason = ('with storage type GPU_Global'
                      if desc.storage is dtypes.StorageType.GPU_Global else f'of symbolic shape {list(desc.shape)}')
            warnings.warn(
                f"Transient array '{name}' {reason} detected inside kernel {kernel}. Neither GPU_Global "
                f"memory nor a variable-length local array can be allocated within a GPU kernel; the "
                f"array will be lifted outside the kernel as a non-transient GPU_Global array.",
                stacklevel=2)
            desc.storage = dtypes.StorageType.GPU_Global
            self.move_array(plan, kernel, kernel_state)
            handled += 1
        self.fail_on_in_kernel_global_global(sdfg)
        return handled or None

    @staticmethod
    def kernel_internal_gpu_global_transients(
            sdfg: SDFG) -> list[tuple[str, dt.Array, SDFG, nodes.MapEntry, SDFGState]]:
        """Transients that :func:`needs_global_memory`, accessed only inside one ``GPU_Device`` map."""
        users: dict[tuple[SDFG, str], OrderedSet[Scope | None]] = {}
        for owner in sdfg.all_sdfgs_recursive():
            for state in owner.states():
                for node in state.data_nodes():
                    if needs_global_memory(owner.arrays[node.data]):
                        kernel = next((scope for scope in helpers.get_parent_maps(state, node)
                                       if scope[0].map.schedule == dtypes.ScheduleType.GPU_Device), None)
                        users.setdefault((owner, node.data), OrderedSet()).add(kernel)
        result = []
        for (owner, name), kernels in users.items():
            if None in kernels:
                continue
            if len(kernels) > 1:
                raise NotImplementedError(f"Transient '{name}' is shared by the kernels {[k for k, _ in kernels]}")
            kernel, kernel_state = kernels[0]
            result.append((name, owner.arrays[name], owner, kernel, kernel_state))
        return result

    @staticmethod
    def fail_on_in_kernel_global_global(sdfg: SDFG) -> None:
        """Raise if a transient ``GPU_Global`` copy survives inside a kernel, where nothing can allocate it."""
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
                    if (all(desc.storage is dtypes.StorageType.GPU_Global for desc in descs)
                            and any(desc.transient for desc in descs) and
                        (is_devicelevel_gpu(nsdfg, state, edge.src) or is_devicelevel_gpu(nsdfg, state, edge.dst))):
                        offenders.append(f"  - {edge.src.data} -> {edge.dst.data} in state "
                                         f"'{state.label}' (SDFG '{nsdfg.name}')")
        if offenders:
            raise ValueError("Transient GPU_Global arrays cannot live inside a kernel scope. Offenders:\n" +
                             "\n".join(offenders))

    def plan_lift(self, name: str, owner: SDFG, kernel_state: SDFGState) -> LiftPlan | None:
        """Everything the lift of ``owner``'s ``name`` rewrites, or ``None`` if a nest would misread its index."""
        accesses = [(node, state) for state in owner.all_states() for node in state.data_nodes() if node.data == name]
        levels = self.slice_levels(name, accesses)
        prefixes = [(edge, lift_prefix(levels, state, edge.src)) for state in owner.all_states()
                    for edge in state.edges() if edge.data.data == name or any(
                        isinstance(node, nodes.AccessNode) and node.data == name for node in (edge.src, edge.dst))]
        desc = owner.arrays[name]
        shape_info = self.get_new_shape_info(desc, [entry for entry, _ in reversed(levels)])
        bounds = [bound for _, prefix in prefixes for dim in prefix for bound in dim]
        bounds += shape_info[0][:len(shape_info[0]) - len(desc.shape)]
        needed = sorted({str(sym) for bound in bounds for sym in symbolic.pystr_to_symbolic(bound).free_symbols})
        hierarchy = sdfg_chain(owner, kernel_state.sdfg)
        conflict = binding_conflict(hierarchy, needed)
        if conflict is not None:
            logger.debug("Not lifting '%s': %s gives '%s' its own meaning", name, conflict[0].name, conflict[1])
            return None
        return LiftPlan(name, owner, desc, levels, accesses, prefixes, shape_info, hierarchy, needed)

    def move_array(self, plan: LiftPlan, kernel: nodes.MapEntry, kernel_state: SDFGState) -> None:
        """Give the planned transient one slice per kernel iteration and allocate it outside the kernel."""
        name = plan.name
        taken = OrderedSet(n for sdfg in plan.hierarchy[1:] for n in (*sdfg.arrays, *sdfg.symbols))
        if name in taken:
            name = utils.find_new_name(name, taken | OrderedSet(plan.owner.arrays) | OrderedSet(plan.owner.symbols))
            plan.owner.replace(plan.name, name)
        for edge, prefix in plan.prefixes:
            self.prefix_memlet(edge, name, prefix)
        # A tasklet with inlined connectors names the array in its body and gains the same leading index.
        for state in plan.owner.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.Tasklet) and name in node.code.as_string:
                    self.prefix_tasklet(node, state, name, plan.levels)
        new_shape, new_strides, new_total_size, new_offsets = plan.shape_info
        plan.desc.set_shape(new_shape=new_shape, strides=new_strides, total_size=new_total_size, offset=new_offsets)
        self.reshape_descendants(plan.owner, name, plan.desc, lambda state, node: lift_prefix(plan.levels, state, node))

        if len(plan.hierarchy) == 1:
            self.carry_out_of_kernel(plan, name, kernel, kernel_state)
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

    def reshape_descendants(self, sdfg: SDFG, name: str, desc: dt.Array, prefix_at: Callable[[SDFGState, nodes.Node],
                                                                                             Prefix]) -> None:
        """Give every nest below ``sdfg`` that ``name`` reaches the lifted descriptor and index its accesses by the
        slice the nest sees: a nested SDFG's data has the shape of the data connected to it (No-View nested SDFGs).

        :raises NotImplementedError: A nest sees more than one slice, so its accesses have no single index.
        """
        for state in sdfg.all_states():
            for node in state.nodes():
                if not isinstance(node, nodes.NestedSDFG):
                    continue
                conns = ({e.dst_conn
                          for e in state.in_edges(node) if e.data.data == name}
                         | {e.src_conn
                            for e in state.out_edges(node) if e.data.data == name})
                if not conns:
                    continue
                prefix = prefix_at(state, node)
                if any(begin != end for begin, end, _ in prefix):
                    raise NotImplementedError(
                        f"Nest {node.label} reads '{name}' outside the kernel levels it is lifted by")
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
                point = [symbolic.symstr(begin) for begin, _, _ in prefix]
                for conn in sorted(conns):
                    inner.arrays[conn] = copy.deepcopy(desc)
                    inner.arrays[conn].transient = False
                    for inner_state in inner.all_states():
                        for edge in inner_state.edges():
                            self.prefix_memlet(edge, conn, prefix)
                        for tasklet in (n for n in inner_state.nodes() if isinstance(n, nodes.Tasklet)):
                            code = tasklet.code.as_string
                            if tasklet.language is not dtypes.Language.Python:
                                if re.search(rf'\b{re.escape(conn)}\s*\[', code):
                                    raise NotImplementedError(
                                        f"Tasklet {tasklet.label} subscripts '{conn}' in {tasklet.language.name}")
                                continue
                            tasklet.code = properties.CodeBlock(prepend_subscript_indices(code, conn, point),
                                                                tasklet.language)
                    self.prefix_control_flow(inner, conn, point)
                    self.reshape_descendants(inner, conn, inner.arrays[conn], lambda _state, _node: prefix)

    @staticmethod
    def slice_levels(name: str, accesses: list[tuple[nodes.AccessNode, SDFGState]]) -> list[Scope]:
        """GPU levels that get a dimension: the deepest access's, which every other access's must prefix."""
        chains = [gpu_levels(state, node) for node, state in accesses]
        deepest = max(chains, key=len)
        for chain in chains:
            if [entry for entry, _ in chain] != [entry for entry, _ in deepest[:len(chain)]]:
                raise NotImplementedError(f"Cannot lift '{name}': it is accessed under sibling GPU maps")
        return deepest

    @staticmethod
    def prefix_memlet(edge: MultiConnectorEdge, name: str, prefix: Prefix) -> None:
        memlet = edge.data
        if memlet.data == name:
            memlet.subset = subsets.Range(prefix + memlet.subset.ndrange())
        elif isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == name and memlet.dst_subset is not None:
            memlet.dst_subset = subsets.Range(prefix + memlet.dst_subset.ndrange())
        elif isinstance(edge.src, nodes.AccessNode) and edge.src.data == name and memlet.src_subset is not None:
            memlet.src_subset = subsets.Range(prefix + memlet.src_subset.ndrange())

    @staticmethod
    def prefix_tasklet(tasklet: nodes.Tasklet, state: SDFGState, name: str, levels: list[Scope]) -> None:
        """Prepend this iteration's slice index to every ``name`` subscript in ``tasklet``'s body.

        :raises NotImplementedError: The tasklet runs outside a level (no single slice), or its C++ body
                                     subscripts ``name``, which only a Python body can be rewritten for.
        """
        prefix = lift_prefix(levels, state, tasklet)
        if any(dim[0] != dim[1] for dim in prefix):
            raise NotImplementedError(
                f"Tasklet {tasklet.label} reads '{name}' outside the kernel levels it is lifted by")
        code = tasklet.code.as_string
        if tasklet.language is not dtypes.Language.Python:
            if re.search(rf'\b{re.escape(name)}\s*\[', code):
                raise NotImplementedError(f"Tasklet {tasklet.label} subscripts '{name}' in {tasklet.language.name}")
            return
        point = [symbolic.symstr(dim[0]) for dim in prefix]
        tasklet.code = properties.CodeBlock(prepend_subscript_indices(code, name, point), tasklet.language)

    @staticmethod
    def prefix_control_flow(sdfg: SDFG, name: str, point: list[str] | None) -> None:
        """Prepend ``point`` to the subscripts of ``name`` that interstate edges, loops and branches read."""

        def rewrite(code: str) -> str:
            new_code = prepend_subscript_indices(code, name, point or ['0'])
            if point is None and new_code != code:
                raise NotImplementedError(f"Control flow of {sdfg.name} reads '{name}', which varies per GPU thread")
            return new_code

        rewrite_code_slots(sdfg, rewrite)

    def carry_out_of_kernel(self, plan: LiftPlan, name: str, kernel: nodes.MapEntry, state: SDFGState) -> None:
        """Route the array from its access nearest the kernel exit out through every map exit, one slice per edge."""
        desc = plan.desc
        exit_node = state.exit_node(kernel)
        source = self.get_nearest_access_node([node for node, _ in plan.accesses], exit_node, state)
        entries = [entry for entry, _ in helpers.get_parent_maps(state, source)]
        exits = [state.exit_node(entry) for entry in entries[:entries.index(kernel) + 1]]
        whole = subsets.Range.from_array(desc).ndrange()
        for src, dst in zip([source, *exits[:-1]], exits, strict=True):
            prefix = lift_prefix(plan.levels, state, src)
            dst.add_in_connector(f'IN_{name}')
            dst.add_out_connector(f'OUT_{name}')
            state.add_edge(src, None if src is source else f'OUT_{name}', dst, f'IN_{name}',
                           Memlet(data=name, subset=subsets.Range(prefix + whole[len(prefix):])))
        state.add_edge(exit_node, f'OUT_{name}', state.add_access(name), None, Memlet.from_array(name, desc))

    def lift_array_through_nested_sdfgs(self, name: str, kernel: nodes.MapEntry, hierarchy: list[SDFG]) -> None:
        """Declare the array at every level from the owner up to the kernel's SDFG and connect it outward."""
        for inner, outer in itertools.pairwise(hierarchy):
            nsdfg_node = inner.parent_nsdfg_node
            state = inner.parent
            new_desc = copy.deepcopy(inner.arrays[name])
            symbolic.safe_replace(nsdfg_node.symbol_mapping,
                                  lambda repl, desc=new_desc: replace_properties_dict(desc, repl))
            outer.add_datadesc(name, new_desc)

            exits = []
            for entry, entry_state in helpers.get_parent_maps(state, nsdfg_node):
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
        # The outermost descriptor is allocated by codegen instead of expected as a kernel input.
        hierarchy[-1].arrays[name].transient = True

    def get_new_shape_info(
        self, array_desc: dt.Array, levels: list[nodes.MapEntry]
    ) -> tuple[list[symbolic.SymbolicType], list[symbolic.SymbolicType], symbolic.SymbolicType, list[int]]:
        """New shape, strides, total size and offsets of ``array_desc`` with one dimension per GPU map parameter.

        The prepended dimensions are the slowest-varying, so ``gpu_A[64]`` under ``map[0:128, 0:32]``
        becomes ``[128, 32, 64]`` indexed ``gpu_A[x, y, :]`` and the own axes keep their C or Fortran layout.

        :param levels: GPU map entries between the array and the kernel exit, innermost first.
        :raises NotImplementedError: The array is neither packed-C nor packed-Fortran.
        """
        if array_desc.is_packed_c_strides():
            inner_order = list(reversed(range(len(array_desc.shape))))
        elif array_desc.is_packed_fortran_strides():
            inner_order = list(range(len(array_desc.shape)))
        else:
            raise NotImplementedError(f'Cannot lift {array_desc}: only packed C or Fortran strides are supported.')

        extended_size: list[symbolic.SymbolicType] = []
        new_offsets = list(array_desc.offset)
        for level in levels:
            extended_size = [
                tile_extent(mx, mn)
                for mx, mn in zip(level.map.range.max_element(), level.map.range.min_element(), strict=True)
            ] + extended_size
            new_offsets = [0 for _ in level.map.params] + new_offsets

        prepended = len(extended_size)
        # ``strides_from_layout`` takes dimensions innermost-first: the own axes, then the prepended ones.
        layout = [d + prepended for d in inner_order] + list(reversed(range(prepended)))
        lifted = array_desc.clone()
        lifted.set_shape(extended_size + list(array_desc.shape))
        new_strides, new_total_size = lifted.strides_from_layout(*layout)
        return list(lifted.shape), list(new_strides), new_total_size, new_offsets

    @staticmethod
    def get_nearest_access_node(access_nodes: list[nodes.AccessNode], node: nodes.Node,
                                state: SDFGState) -> nodes.AccessNode:
        """Closest of ``access_nodes`` to ``node`` in ``state``, by undirected graph distance.

        :raises RuntimeError: No candidate is connected to ``node``.
        """
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
        raise RuntimeError(f"No access node found connected to the given node {node}. ")
