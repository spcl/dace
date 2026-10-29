# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that hoists kernel-local transients out of GPU kernels into device-global allocations."""

import ast
import copy
import itertools
import logging
import numbers
import re
import warnings
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sympy

from dace import SDFG, SDFGState, dtypes, properties, subsets, symbolic, utils
from dace import data as dt
from dace.memlet import Memlet
from dace.optionals import required
from dace.ordered import OrderedSet
from dace.sdfg import is_devicelevel_gpu, nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.replace import replace_properties_dict
from dace.sdfg.state import LoopRegion
from dace.transformation import helpers, transformation
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.length_one_array_scalar_conversion import rewrite_code_slots

logger = logging.getLogger(__name__)

# Not ``dtypes.GPU_SCHEDULES``: that also holds dynamic and persistent thread-block schedules, which are not lifted.
GPU_HIERARCHY_SCHEDULES = (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_ThreadBlock)

Scope = tuple[nodes.MapEntry, SDFGState]
Dim = tuple[symbolic.SymbolicType, symbolic.SymbolicType, symbolic.SymbolicType]
Prefix = list[Dim]


def bound_over_levels(extent: symbolic.SymbolicType, levels: list[nodes.MapEntry]) -> symbolic.SymbolicType:
    """``extent``'s maximum over the iterations of ``levels`` (innermost first): the lift allocates before they run.

    :raises NotImplementedError: ``extent`` is not linear in each parameter, so no corner bounds it.
    """
    for level in levels:
        names = {str(s) for s in symbolic.symlist(extent)}
        bounds = {
            param: (low, high)
            for param, low, high in zip(
                level.map.params, level.map.range.min_element(), level.map.range.max_element(), strict=True
            )
            if param in names
        }
        if not bounds:
            continue
        if any(symbolic.affine_coefficients(extent, param) is None for param in bounds):
            raise NotImplementedError(f"Cannot bound the extent {extent} over the kernel parameters {list(bounds)}.")
        corners = [
            extent.subs({symbolic.pystr_to_symbolic(param): value for param, value in zip(bounds, corner)})
            for corner in itertools.product(*bounds.values())
        ]
        # The range and the descriptor name one symbol through different instances.
        extent = symbolic.simplify(sympy.Max(*symbolic.equalize_symbols_across(*corners)))
    return extent


def is_register_demotable(desc: dt.Data, max_elements: int) -> bool:
    """A literal shape of at most ``max_elements``; persistent and external arrays outlive a thread's registers."""
    if desc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
        return False
    total = utils.prod(desc.shape)
    return isinstance(total, numbers.Integral) and 0 < total <= max_elements


# Storage emitted as a plain local declaration in device code; shared memory has its own sizing rules.
DEVICE_LOCAL_STORAGE = (dtypes.StorageType.Register, dtypes.StorageType.Default)


def needs_global_memory(desc: dt.Data, max_register_elements: int) -> bool:
    """A kernel-internal transient that belongs in device-global memory: ``GPU_Global``, or device-local with a
    symbolic extent (a variable-length array nvcc rejects) or more than ``max_register_elements`` elements (more
    per-thread registers than an intermediate should take; it spills to local memory anyway)."""
    # A view or a reference owns no memory: it points into what it binds
    if not isinstance(desc, dt.Array) or isinstance(desc, (dt.View, dt.Reference)) or not desc.transient:
        return False
    if desc.storage is dtypes.StorageType.GPU_Global:
        return True
    if desc.storage not in DEVICE_LOCAL_STORAGE:
        return False
    return any(symbolic.issymbolic(dim) for dim in desc.shape) or utils.prod(desc.shape) > max_register_elements


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
        for param, (start, end, step), origin in zip(
            entry.map.params, entry.map.range, entry.map.range.min_element(), strict=True
        ):
            if inside:
                index = slot(symbolic.symbol(param) - origin, step)
                prefix.append((index, index, sympy.S.One))
            elif step == 1:
                prefix.append((start - origin, end - origin, step))
            else:
                prefix.append((slot(start - origin, step), slot(end - origin, step), sympy.S.One))
    return prefix


def path_ends(state: SDFGState, edge: MultiConnectorEdge) -> tuple[nodes.Node, nodes.Node]:
    """The first source and last destination of ``edge``'s memlet path."""
    path = state.memlet_path(edge)
    return path[0].src, path[-1].dst


def slot(offset: symbolic.SymbolicType, step: symbolic.SymbolicType) -> symbolic.SymbolicType:
    """The slot of the iteration ``offset`` past a level's first one: one slot per iteration, also for a strided level
    (a tile grid), whose slots would otherwise leave ``step - 1`` unused between two."""
    return offset if step == 1 else symbolic.int_floor(offset, step)


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


@dataclass(slots=True)
class ExtentBound:
    """A host symbol bounding an extent symbol assigned inside the kernel: ``max(0, container[index])`` over the
    kernel's iterations, ``container`` and ``index`` in the kernel SDFG's names. ``index`` reads the containers in
    ``reads`` (``row_bin[rows[b]]``) through their connectors, :func:`read_connector`."""

    symbol: str
    bound: str
    container: str
    index: list[str]
    dtype: dtypes.typeclass
    on_device: bool
    reads: list[str]
    folded: bool = False


def read_connector(container: str) -> str:
    return f"__read_{container}"


def kernel_assigned_read(hierarchy: list[SDFG], name: str) -> tuple[SDFG, str, str] | None:
    """The one element read ``name`` is assigned inside the kernel, as ``(sdfg, container, index)``; ``None`` if a
    nest assigns it anything else, or several values."""
    found: set[tuple[int, str]] = set()
    for depth, sdfg in enumerate(hierarchy[:-1]):
        if any(isinstance(cfr, LoopRegion) and cfr.loop_variable == name for cfr in sdfg.all_control_flow_regions()):
            return None
        found |= {
            (depth, edge.data.assignments[name])
            for edge in sdfg.all_interstate_edges()
            if name in edge.data.assignments
        }
    if len(found) != 1:
        return None
    depth, value = next(iter(found))
    expr = ast.parse(value.strip(), mode="eval").body
    if not isinstance(expr, ast.Subscript) or not isinstance(expr.value, ast.Name):
        return None
    return hierarchy[depth], expr.value.id, ast.unparse(expr.slice)


def read_in_kernel_sdfg(
    hierarchy: list[SDFG], sdfg: SDFG, container: str, index: str
) -> tuple[str, list[symbolic.SymbolicType]] | None:
    """``container[index]`` read in ``sdfg`` (a nest of ``hierarchy``) in the names of ``hierarchy[-1]``: through
    every nest's input binding and symbol mapping; ``None`` when a nest is handed something other than an array
    window of the outer one."""
    idx = [symbolic.pystr_to_symbolic(part) for part in index.split(",")] if index else []
    level = hierarchy.index(sdfg)
    for inner in hierarchy[level:-1]:
        node, state = inner.parent_nsdfg_node, inner.parent
        bound = [e for e in state.in_edges(node) if e.dst_conn == container and not e.data.is_empty()]
        if len(bound) != 1 or not isinstance(bound[0].data.subset, subsets.Range):
            return None
        window = bound[0].data.subset
        outer_desc = state.sdfg.arrays.get(bound[0].data.data)
        if outer_desc is None or isinstance(outer_desc, dt.View) or len(window) != len(idx):
            return None
        mapping = {
            symbolic.pystr_to_symbolic(k): symbolic.pystr_to_symbolic(str(v)) for k, v in node.symbol_mapping.items()
        }
        idx = [begin + symbolic.pystr_to_symbolic(i).subs(mapping) for (begin, _, _), i in zip(window, idx)]
        container = bound[0].data.data
    return container, idx


def inline_kernel_reads(
    hierarchy: list[SDFG], index: list[symbolic.SymbolicType], seen: tuple[str, ...] = ()
) -> tuple[list[str], OrderedSet[str]] | None:
    """``index`` as text in the names of ``hierarchy[-1]``, each symbol a nest assigns replaced by the one element
    read it is assigned (:func:`kernel_assigned_read`) through :func:`read_connector`, and the containers read;
    ``None`` if a nest assigns one anything else."""
    reads = OrderedSet()
    replacements = {}
    names = sorted({str(s) for i in index for s in symbolic.pystr_to_symbolic(i).free_symbols})
    for name in names:
        if not any(assigns_symbol(sdfg, name) for sdfg in hierarchy[:-1]):
            continue
        read = kernel_assigned_read(hierarchy, name) if name not in seen else None
        located = read and read_in_kernel_sdfg(hierarchy, *read)
        inner = located and inline_kernel_reads(hierarchy, located[1], (*seen, name))
        if not inner:
            return None
        reads.add(located[0])
        reads.update(inner[1])
        replacements[name] = f"{read_connector(located[0])}[{', '.join(inner[0])}]"
    text = [symbolic.symstr(i) for i in index]
    for name, value in replacements.items():
        text = [re.sub(rf"\b{re.escape(name)}\b", value, part) for part in text]
    return text, reads


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
    prefixes: list[tuple[MultiConnectorEdge, Prefix, SDFGState]]
    shape_info: tuple[list[symbolic.SymbolicType], list[symbolic.SymbolicType], symbolic.SymbolicType, list[int]]
    hierarchy: list[SDFG]
    needed: list[str]
    bounds: list[ExtentBound]


@properties.make_properties
@transformation.explicit_cf_compatible
class MoveArrayOutOfKernel(ppl.Pass):
    """Lift kernel-internal transients out of ``GPU_Device`` maps (kernels): ``GPU_Global`` ones, and per-thread
    register arrays of symbolic shape or more than ``register_demotion_max_elements`` elements.

    Each array is replicated per map iteration into a disjoint outer array. A small literal-shape ``GPU_Global``
    one is demoted to registers instead.
    """

    register_demotion_max_elements = properties.Property(
        dtype=int,
        default=16,
        desc="Max ``prod(shape)`` of a kernel-internal transient kept in per-thread Register storage: a "
        "literal-shape GPU_Global one at most this size is demoted to it, and a device-local one larger "
        "than this (or of symbolic shape) is hoisted out of the kernel.",
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
        accumulated = OrderedSet(
            e.data.data
            for nsdfg in sdfg.all_sdfgs_recursive()
            for state in nsdfg.states()
            for e in state.edges()
            if e.data.wcr is not None
        )
        handled = 0
        from dace.codegen.python.dialects.cuda_tile_cpp import is_tile_kernel

        for name, desc, owner, kernel, kernel_state in self.kernel_internal_gpu_global_transients(
            sdfg, self.register_demotion_max_elements
        ):
            if is_tile_kernel(kernel_state, kernel):
                continue  # a CUDA Tile C++ kernel holds its tiles as values
            if name not in accumulated and is_register_demotable(desc, self.register_demotion_max_elements):
                desc.storage = dtypes.StorageType.Register
                handled += 1
                continue
            plan = self.plan_lift(name, owner, kernel, kernel_state)
            if plan is None:
                continue
            reason = (
                "with storage type GPU_Global"
                if desc.storage is dtypes.StorageType.GPU_Global
                else f"of shape {list(desc.shape)} (symbolic, or over {self.register_demotion_max_elements} elements)"
            )
            warnings.warn(
                f"Transient array '{name}' {reason} detected inside kernel {kernel}. Neither GPU_Global "
                f"memory nor a variable-length or oversized local array belongs within a GPU kernel; the "
                f"array will be lifted outside the kernel as a non-transient GPU_Global array.",
                stacklevel=2,
            )
            desc.storage = dtypes.StorageType.GPU_Global
            self.move_array(plan, kernel, kernel_state)
            handled += 1
        self.fail_on_in_kernel_global_global(sdfg)
        return handled or None

    @staticmethod
    def kernel_internal_gpu_global_transients(
        sdfg: SDFG, max_register_elements: int
    ) -> list[tuple[str, dt.Array, SDFG, nodes.MapEntry, SDFGState]]:
        """Transients that :func:`needs_global_memory`, accessed only inside one ``GPU_Device`` map."""
        users: dict[tuple[SDFG, str], OrderedSet[Scope | None]] = {}
        for owner in sdfg.all_sdfgs_recursive():
            for state in owner.states():
                for node in state.data_nodes():
                    if needs_global_memory(owner.arrays[node.data], max_register_elements):
                        kernel = next(
                            (
                                scope
                                for scope in helpers.get_parent_maps(state, node)
                                if scope[0].map.schedule == dtypes.ScheduleType.GPU_Device
                            ),
                            None,
                        )
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
                    # An edge into or out of a view, or setting a reference, aliases its data; it copies nothing.
                    if any(isinstance(desc, (dt.View, dt.Reference)) for desc in descs):
                        continue
                    if (
                        all(desc.storage is dtypes.StorageType.GPU_Global for desc in descs)
                        and any(desc.transient for desc in descs)
                        and (is_devicelevel_gpu(nsdfg, state, edge.src) or is_devicelevel_gpu(nsdfg, state, edge.dst))
                    ):
                        offenders.append(
                            f"  - {edge.src.data} -> {edge.dst.data} in state '{state.label}' (SDFG '{nsdfg.name}')"
                        )
        if offenders:
            raise ValueError(
                "Transient GPU_Global arrays cannot live inside a kernel scope. Offenders:\n" + "\n".join(offenders)
            )

    def plan_lift(self, name: str, owner: SDFG, kernel: nodes.MapEntry, kernel_state: SDFGState) -> LiftPlan | None:
        """Everything the lift of ``owner``'s ``name`` rewrites, or ``None`` if a nest would misread its index."""
        accesses = [(node, state) for state in owner.states() for node in state.data_nodes() if node.data == name]
        levels = self.slice_levels(name, accesses)
        prefixes = [
            (edge, lift_prefix(levels, state, edge.src), state)
            for state in owner.states()
            for edge in state.edges()
            if edge.data.data == name
            or any(isinstance(node, nodes.AccessNode) and node.data == name for node in path_ends(state, edge))
        ]
        # An access that neither enters nor leaves a scope names one slice; above a slicing level it would span all
        for edge, prefix, state in prefixes:
            crosses = (
                edge.data.is_empty() or isinstance(edge.dst, nodes.EntryNode) or isinstance(edge.src, nodes.ExitNode)
            )
            if not crosses and any(dim[0] != dim[1] for dim in prefix):
                raise NotImplementedError(
                    f"Memlet {edge.data} in {state.label} reads '{name}' outside the kernel levels it is lifted by"
                )
        desc = owner.arrays[name]
        hierarchy = sdfg_chain(owner, kernel_state.sdfg)
        extent_bounds = self.extent_bounds(desc, hierarchy, kernel, kernel_state)
        if extent_bounds is None:
            logger.debug("Not lifting '%s': its extent varies per GPU thread with no bound outside the kernel", name)
            return None
        sized = desc.clone()
        sized.set_shape(
            [
                symbolic.pystr_to_symbolic(symbolic.symstr(d)).subs(
                    {symbolic.symbol(b.symbol): symbolic.symbol(b.bound) for b in extent_bounds}
                )
                for d in desc.shape
            ]
        )
        shape_info = self.get_new_shape_info(sized, [entry for entry, _ in reversed(levels)])
        bounds = [bound for _, prefix, _ in prefixes for dim in prefix for bound in dim]
        bounds += shape_info[0][: len(shape_info[0]) - len(desc.shape)]
        needed = sorted({str(sym) for bound in bounds for sym in symbolic.pystr_to_symbolic(bound).free_symbols})
        needed += [b.bound for b in extent_bounds]
        conflict = binding_conflict(hierarchy, needed)
        if conflict is not None:
            logger.debug("Not lifting '%s': %s gives '%s' its own meaning", name, conflict[0].name, conflict[1])
            return None
        return LiftPlan(name, owner, desc, levels, accesses, prefixes, shape_info, hierarchy, needed, extent_bounds)

    @staticmethod
    def extent_bounds(
        desc: dt.Array, hierarchy: list[SDFG], kernel: nodes.MapEntry, kernel_state: SDFGState
    ) -> list[ExtentBound] | None:
        """A host bound for every shape symbol of ``desc`` the kernel assigns, or ``None`` if one has none: each must
        be assigned one integer element read of data the kernel's SDFG holds in device memory or in memory the host
        code before the kernel reads (``count = counts[b]``)."""
        from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES, host_storages

        kernel_sdfg = kernel_state.sdfg
        result = []
        taken = OrderedSet((*kernel_sdfg.arrays, *kernel_sdfg.symbols))
        for sym in sorted({str(s) for d in desc.shape for s in symbolic.pystr_to_symbolic(d).free_symbols}):
            if not any(assigns_symbol(sdfg, sym) for sdfg in hierarchy[:-1]):
                continue
            read = kernel_assigned_read(hierarchy, sym)
            located = read and read_in_kernel_sdfg(hierarchy, *read)
            if not located:
                return None
            container, index = located
            inlined = inline_kernel_reads(hierarchy, index)
            if inlined is None:
                return None
            index, reads = inlined
            ctype = kernel_sdfg.arrays[container].dtype
            on_device = kernel_sdfg.arrays[container].storage in GPU_RESIDENT_STORAGES
            readable = GPU_RESIDENT_STORAGES if on_device else host_storages(kernel, kernel_state)
            if not np.issubdtype(ctype.type, np.integer) or any(
                kernel_sdfg.arrays[name].storage not in readable for name in (container, *reads)
            ):
                return None
            reads = list(reads)
            bound = f"{sym}_max"
            if bound in kernel_sdfg.symbols and not any(b.bound == bound for b in result):
                # Already folded for an array this kernel lifted before
                result.append(ExtentBound(sym, bound, container, index, ctype, on_device, reads, folded=True))
                continue
            bound = utils.find_new_name(bound, taken)
            taken.add(bound)
            result.append(ExtentBound(sym, bound, container, index, ctype, on_device, reads))
        return result

    def move_array(self, plan: LiftPlan, kernel: nodes.MapEntry, kernel_state: SDFGState) -> None:
        """Give the planned transient one slice per kernel iteration and allocate it outside the kernel."""
        for extent in plan.bounds:
            if not extent.folded:
                self.compute_bound(extent, kernel, kernel_state)
        name = plan.name
        taken = OrderedSet(n for sdfg in plan.hierarchy[1:] for n in (*sdfg.arrays, *sdfg.symbols))
        if name in taken:
            name = utils.find_new_name(name, taken | OrderedSet(plan.owner.arrays) | OrderedSet(plan.owner.symbols))
            plan.owner.replace(plan.name, name)
        for edge, prefix, state in plan.prefixes:
            self.prefix_memlet(edge, name, prefix, state)
        # A tasklet with inlined connectors names the array in its body and gains the same leading index.
        for state in plan.owner.states():
            for node in state.nodes():
                if isinstance(node, nodes.Tasklet) and name in node.code.as_string:
                    self.prefix_tasklet(node, state, name, plan.levels)
        new_shape, new_strides, new_total_size, new_offsets = plan.shape_info
        plan.desc.set_shape(new_shape=new_shape, strides=new_strides, total_size=new_total_size, offset=new_offsets)
        # The kernel's params reach the owner first: a nest below it types them from what the owner defines
        if len(plan.hierarchy) > 1:
            bind_symbols(plan.hierarchy, plan.needed)
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
        plan.desc.transient = False
        self.lift_array_through_nested_sdfgs(name, kernel, plan.hierarchy)

    @staticmethod
    def compute_bound(extent: ExtentBound, kernel: nodes.MapEntry, kernel_state: SDFGState) -> None:
        """Assign ``extent.bound`` before ``kernel_state``: the maximum, and at least 0, of the element read over the
        kernel's iterations, folded where the data lives (a device map for device data)."""
        sdfg = kernel_state.sdfg
        on_device = extent.on_device
        storage = dtypes.StorageType.GPU_Global if on_device else dtypes.StorageType.CPU_Heap
        schedule = dtypes.ScheduleType.GPU_Device if on_device else dtypes.ScheduleType.Sequential
        fold, _ = sdfg.add_array(
            f"{extent.bound}_fold", [1], extent.dtype, storage=storage, transient=True, find_new_name=True
        )
        region = kernel_state.parent_graph
        state = region.add_state_before(
            kernel_state, f"bound_{extent.bound}", is_start_block=region.start_block is kernel_state
        )
        state.add_mapped_tasklet(
            f"seed_{extent.bound}",
            {"__s": "0:1"},
            {},
            "o = 0",
            {"o": Memlet(f"{fold}[0]")},
            schedule=schedule,
            external_edges=True,
        )
        fold_state = region.add_state_after(state, f"fold_{extent.bound}")
        params = dict(zip(kernel.map.params, (f"{b}:{e}+1:{st}" for b, e, st in kernel.map.range), strict=True))
        index = ", ".join(extent.index)
        if extent.reads:
            inputs = {
                read_connector(name): Memlet.from_array(name, sdfg.arrays[name])
                for name in OrderedSet((extent.container, *extent.reads))
            }
            code = f"o = {read_connector(extent.container)}[{index}]"
        else:
            inputs, code = {"v": Memlet(f"{extent.container}[{index}]")}, "o = v"
        _, _, fold_exit = fold_state.add_mapped_tasklet(
            f"fold_{extent.bound}",
            params,
            inputs,
            code,
            {"o": Memlet(f"{fold}[0]", wcr="lambda a, b: max(a, b)")},
            schedule=schedule,
            external_edges=True,
        )
        source = fold
        if on_device:
            host, _ = sdfg.add_array(
                f"{extent.bound}_host",
                [1],
                extent.dtype,
                storage=dtypes.StorageType.CPU_Heap,
                transient=True,
                find_new_name=True,
            )
            written = next(e.dst for e in fold_state.out_edges(fold_exit))
            fold_state.add_nedge(written, fold_state.add_write(host), Memlet(f"{host}[0]"))
            source = host
        if extent.bound not in sdfg.symbols:
            sdfg.add_symbol(extent.bound, extent.dtype)
        for edge in region.in_edges(kernel_state):
            if edge.src is fold_state:
                edge.data.assignments[extent.bound] = f"{source}[0]"

    def reshape_descendants(
        self, sdfg: SDFG, name: str, desc: dt.Array, prefix_at: Callable[[SDFGState, nodes.Node], Prefix]
    ) -> None:
        """Give every nest below ``sdfg`` that ``name`` reaches the lifted descriptor and index its accesses by the
        slice the nest sees: a nested SDFG's data has the shape of the data connected to it (No-View nested SDFGs).

        :raises NotImplementedError: A nest sees more than one slice, so its accesses have no single index.
        """
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
                if any(begin != end for begin, end, _ in prefix):
                    raise NotImplementedError(
                        f"Nest {node.label} reads '{name}' outside the kernel levels it is lifted by"
                    )
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
                    for inner_state in inner.states():
                        for edge in inner_state.edges():
                            self.prefix_memlet(edge, conn, prefix, inner_state)
                        for tasklet in (n for n in inner_state.nodes() if isinstance(n, nodes.Tasklet)):
                            code = tasklet.code.as_string
                            if tasklet.language is not dtypes.Language.Python:
                                if re.search(rf"\b{re.escape(conn)}\s*\[", code):
                                    raise NotImplementedError(
                                        f"Tasklet {tasklet.label} subscripts '{conn}' in {tasklet.language.name}"
                                    )
                                continue
                            tasklet.code = properties.CodeBlock(
                                prepend_subscript_indices(code, conn, point), tasklet.language
                            )
                    self.prefix_control_flow(inner, conn, point)
                    self.reshape_descendants(inner, conn, inner.arrays[conn], lambda _state, _node: prefix)

    @staticmethod
    def slice_levels(name: str, accesses: list[tuple[nodes.AccessNode, SDFGState]]) -> list[Scope]:
        """GPU levels that get a dimension: the deepest access's, which every other access's must prefix."""
        chains = [gpu_levels(state, node) for node, state in accesses]
        deepest = max(chains, key=len)
        for chain in chains:
            if [entry for entry, _ in required(chain)] != [entry for entry, _ in required(deepest[: len(chain)])]:
                raise NotImplementedError(f"Cannot lift '{name}': it is accessed under sibling GPU maps")
        return deepest

    @staticmethod
    def prefix_memlet(edge: MultiConnectorEdge, name: str, prefix: Prefix, state: SDFGState) -> None:
        """Prepend ``prefix`` to the subset of ``edge`` that indexes ``name``: its own, or the side of its memlet
        path that ends in an access node of ``name`` (a copy through a scope names its source only so)."""
        memlet = edge.data
        src, dst = path_ends(state, edge)
        if memlet.data == name:
            memlet.subset = subsets.Range(prefix + memlet.subset.ndrange())
        elif isinstance(dst, nodes.AccessNode) and dst.data == name and memlet.dst_subset is not None:
            memlet.dst_subset = subsets.Range(prefix + memlet.dst_subset.ndrange())
        elif isinstance(src, nodes.AccessNode) and src.data == name and memlet.src_subset is not None:
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
                f"Tasklet {tasklet.label} reads '{name}' outside the kernel levels it is lifted by"
            )
        code = tasklet.code.as_string
        if tasklet.language is not dtypes.Language.Python:
            if re.search(rf"\b{re.escape(name)}\s*\[", code):
                raise NotImplementedError(f"Tasklet {tasklet.label} subscripts '{name}' in {tasklet.language.name}")
            return
        point = [symbolic.symstr(dim[0]) for dim in prefix]
        tasklet.code = properties.CodeBlock(prepend_subscript_indices(code, name, point), tasklet.language)

    @staticmethod
    def prefix_control_flow(sdfg: SDFG, name: str, point: list[str] | None) -> None:
        """Prepend ``point`` to the subscripts of ``name`` that interstate edges, loops and branches read."""

        def rewrite(code: str) -> str:
            new_code = prepend_subscript_indices(code, name, point or ["0"])
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
        exits = [state.exit_node(entry) for entry in entries[: entries.index(kernel) + 1]]
        whole = subsets.Range.from_array(desc).ndrange()
        for src, dst in zip([source, *exits[:-1]], exits, strict=True):
            prefix = lift_prefix(plan.levels, state, src)
            required(dst).add_in_connector(f"IN_{name}")
            required(dst).add_out_connector(f"OUT_{name}")
            state.add_edge(
                src,
                None if src is source else f"OUT_{name}",
                dst,
                f"IN_{name}",
                Memlet(data=name, subset=subsets.Range(prefix + whole[len(prefix) :])),
            )
        state.add_edge(exit_node, f"OUT_{name}", state.add_access(name), None, Memlet.from_array(name, desc))

    def lift_array_through_nested_sdfgs(self, name: str, kernel: nodes.MapEntry, hierarchy: list[SDFG]) -> None:
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
            for entry, entry_state in helpers.get_parent_maps(state, nsdfg_node):
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
            raise NotImplementedError(f"Cannot lift {array_desc}: only packed C or Fortran strides are supported.")

        extended_size: list[symbolic.SymbolicType] = []
        new_offsets = list(array_desc.offset)
        for level in levels:
            extended_size = [
                symbolic.tile_extent(mx, mn) if step == 1 else symbolic.int_ceil(symbolic.tile_extent(mx, mn), step)
                for mx, mn, (begin, end, step) in zip(
                    level.map.range.max_element(), level.map.range.min_element(), level.map.range, strict=True
                )
            ] + extended_size
            new_offsets = [0 for _ in level.map.params] + new_offsets

        prepended = len(extended_size)
        # ``strides_from_layout`` takes dimensions innermost-first: the own axes, then the prepended ones.
        layout = [d + prepended for d in inner_order] + list(reversed(range(prepended)))
        lifted = array_desc.clone()
        lifted.set_shape(extended_size + [bound_over_levels(extent, levels) for extent in array_desc.shape])
        new_strides, new_total_size = lifted.strides_from_layout(*layout)
        return list(lifted.shape), list(new_strides), new_total_size, new_offsets

    @staticmethod
    def get_nearest_access_node(
        access_nodes: list[nodes.AccessNode], node: nodes.Node, state: SDFGState
    ) -> nodes.AccessNode:
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
