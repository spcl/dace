# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Move kernel-local arrays that a GPU kernel cannot allocate out of the kernel.

A GPU kernel can neither allocate device-global memory nor hold a symbolically sized or large array in registers;
the CUDA code generator refuses both ("Dynamic allocation of registers not allowed"). Running this pass before GPU
code generation keeps such SDFGs compilable: each such array becomes one ``GPU_Global`` buffer allocated outside the
kernel, with one disjoint slice per GPU thread, and an array small enough for registers is demoted to them instead.
"""

import ast
import copy
import logging
import numbers
import re
import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import sympy
from ordered_set import OrderedSet

from dace import SDFG, SDFGState, dtypes, properties, subsets, symbolic, utils
from dace import data as dt
from dace.config import Config
from dace.memlet import Memlet
from dace.sdfg import dealias, is_devicelevel_gpu, nodes, propagation
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.replace import replace_properties_dict
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.length_one_array_scalar_conversion import rewrite_code_slots

logger = logging.getLogger(__name__)

# Maps whose iterations each own a slice. Dynamic thread-block maps share threads between iterations.
SLICE_SCHEDULES = (*dtypes.GPU_KERNEL_SCHEDULES, dtypes.ScheduleType.GPU_ThreadBlock)

Scope = tuple[nodes.MapEntry, SDFGState]
Dim = tuple[symbolic.SymbolicType, symbolic.SymbolicType, symbolic.SymbolicType]
Prefix = list[Dim]
ScopeDict = dict[nodes.Node, nodes.EntryNode | None]


class LiftedLayout(NamedTuple):
    shape: list[symbolic.SymbolicType]
    strides: list[symbolic.SymbolicType]
    total_size: symbolic.SymbolicType
    offset: list[int]


class Level(NamedTuple):
    entry: nodes.MapEntry
    params: list[symbolic.symbol]


class Hop(NamedTuple):
    """One nested-SDFG boundary the array is passed out through."""

    node: nodes.NestedSDFG
    state: SDFGState
    levels: list[Level]
    exits: list[nodes.ExitNode]
    desc: dt.Array


def tile_extent(max_elem: symbolic.SymbolicType, min_elem: symbolic.SymbolicType) -> symbolic.SymbolicType:
    """Per-iteration extent of an inner-map range; a ``Min``-bounded tile yields its static width."""
    if isinstance(max_elem, sympy.Min):
        for arg in max_elem.args:
            diff = symbolic.simplify(arg - min_elem, symbolic.Facts.none())
            if diff.is_Integer and diff >= 0:
                return diff + 1
    return max_elem + 1 - min_elem


def fits_registers(desc: dt.Data, register_limit: int) -> bool:
    """A constant size of at most ``register_limit`` elements; persistent and external arrays outlive a thread."""
    if desc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
        return False
    total = utils.prod(desc.shape)
    return isinstance(total, numbers.Integral) and total <= register_limit


def is_move_candidate(desc: dt.Data, register_limit: int) -> bool:
    """A transient array a kernel cannot declare as is: ``GPU_Global``, or a ``Register`` array of symbolic size or
    larger than ``register_limit``."""
    # A view or reference names other memory, set by its connector, rather than owning any
    if not isinstance(desc, dt.Array) or isinstance(desc, (dt.View, dt.Reference)) or not desc.transient:
        return False
    if desc.storage is dtypes.StorageType.GPU_Global:
        return True
    return desc.storage is dtypes.StorageType.Register and not fits_registers(desc, register_limit)


def enclosing_entries(
    sdict: ScopeDict, scope: nodes.EntryNode | None, stop: nodes.EntryNode | None
) -> list[nodes.EntryNode]:
    """``scope`` and the scopes around it in its state, innermost first, ending at ``stop`` if given."""
    entries = []
    while scope is not None:
        entries.append(scope)
        if scope is stop:
            break
        scope = sdict[scope]
    return entries


def slicing(entries: list[nodes.EntryNode]) -> list[nodes.MapEntry]:
    """The maps of ``entries`` that give each iteration a slice, outermost first."""
    return [e for e in reversed(entries) if isinstance(e, nodes.MapEntry) and e.map.schedule in SLICE_SCHEDULES]


def make_levels(sdfg: SDFG, state: SDFGState, entries: list[nodes.MapEntry]) -> list[Level]:
    levels = []
    for entry in entries:
        types = entry.new_symbols(sdfg, state, {})
        levels.append(Level(entry, [symbolic.symbol(p, types[p]) for p in entry.map.params]))
    return levels


def level_prefix(levels: list[Level], chain: list[nodes.MapEntry]) -> Prefix:
    """Leading subset of an access inside ``chain``: its own iteration of a level it is in, the whole level otherwise."""
    prefix: Prefix = []
    for level in levels:
        rng = level.entry.map.range
        inside = level.entry in chain
        for param, (start, end, step), origin in zip(level.params, rng, rng.min_element(), strict=True):
            if inside:
                index = slot(param - origin, step)
                prefix.append((index, index, sympy.S.One))
            elif step == 1:
                prefix.append((start - origin, end - origin, step))
            else:
                prefix.append((slot(start - origin, step), slot(end - origin, step), sympy.S.One))
    return prefix


def exact_form(expr: symbolic.SymbolicType) -> symbolic.SymbolicType:
    """The exact form of a ``SymExpr``. ``int_floor`` is a sympy function and cannot take the pair."""
    return expr.expr if isinstance(expr, symbolic.SymExpr) else expr


def approximate_form(expr: symbolic.SymbolicType) -> symbolic.SymbolicType:
    return expr.approx if isinstance(expr, symbolic.SymExpr) else expr


def slot(offset: symbolic.SymbolicType, step: symbolic.SymbolicType) -> symbolic.SymbolicType:
    """The slot of the iteration ``offset`` past a level's first one: one slot per iteration, also for a strided level
    (a tile grid), whose slots would otherwise leave ``step - 1`` unused between two."""
    # A tiled bound is a ``SymExpr`` (exact end, over-approximation). sympy cannot construct ``int_floor`` from it.
    if isinstance(offset, symbolic.SymExpr) or isinstance(step, symbolic.SymExpr):
        return symbolic.SymExpr(
            slot(exact_form(offset), exact_form(step)), slot(approximate_form(offset), approximate_form(step))
        )
    return offset if step == 1 else symbolic.int_floor(offset, step)


def single_point(prefix: Prefix, label: str, name: str) -> list[str]:
    """The one slice index ``prefix`` names.

    :raises NotImplementedError: ``prefix`` spans several slices, so ``label`` has no single index into ``name``.
    """
    if any(begin != end for begin, end, step in prefix):
        raise NotImplementedError(f"{label} reads '{name}' outside the kernel levels it is lifted by")
    return [symbolic.symstr(begin) for begin, end, step in prefix]


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


def needs_prefix(tasklet: nodes.Tasklet, name: str) -> bool:
    """Whether ``tasklet``'s Python body subscripts the array ``name`` directly.

    :raises NotImplementedError: A non-Python body subscripts ``name``; only Python bodies can be rewritten.
    """
    if name in tasklet.in_connectors or name in tasklet.out_connectors:
        return False
    if not re.search(rf"\b{re.escape(name)}\s*\[", tasklet.code.as_string):
        return False
    if tasklet.language is not dtypes.Language.Python:
        raise NotImplementedError(f"Tasklet {tasklet.label} subscripts '{name}' in {tasklet.language.name}")
    return True


def prefix_memlet(edge: MultiConnectorEdge, name: str, prefix: Prefix) -> None:
    memlet = edge.data
    if memlet.data == name:
        memlet.subset = subsets.Range(prefix + memlet.subset.ndrange())
    elif isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == name and memlet.dst_subset is not None:
        memlet.dst_subset = subsets.Range(prefix + memlet.dst_subset.ndrange())
    elif isinstance(edge.src, nodes.AccessNode) and edge.src.data == name and memlet.src_subset is not None:
        memlet.src_subset = subsets.Range(prefix + memlet.src_subset.ndrange())


def innermost_edges(state: SDFGState, name: str) -> list[MultiConnectorEdge]:
    """Edges of ``name`` that start a memlet path; propagation derives the rest of each path from them."""

    def names(edge: MultiConnectorEdge) -> bool:
        ends = (edge.src, edge.dst)
        return edge.data.data == name or any(isinstance(n, nodes.AccessNode) and n.data == name for n in ends)

    return [
        edge
        for edge in state.edges()
        if not edge.data.is_empty()
        and not isinstance(edge.src, nodes.ExitNode)
        and not isinstance(edge.dst, nodes.EntryNode)
        and names(edge)
    ]


def refuse_control_flow_reads(sdfg: SDFG, name: str) -> None:
    """:raises NotImplementedError: Control flow of ``sdfg`` subscripts ``name``, which varies per GPU thread."""
    pattern = re.compile(rf"\b{re.escape(name)}\s*\[")

    def check(code: str) -> str:
        if pattern.search(code):
            raise NotImplementedError(f"Control flow of {sdfg.name} reads '{name}', which varies per GPU thread")
        return code

    rewrite_code_slots(sdfg, check)


def prepend_levels(desc: dt.Array, levels: list[nodes.MapEntry]) -> tuple[list[symbolic.SymbolicType], list[int]]:
    """Shape and offset of ``desc`` with one leading dimension per parameter of ``levels`` (innermost first)."""
    shape, offset = list(desc.shape), list(desc.offset)
    for level in levels:
        rng = level.map.range
        shape = [
            tile_extent(mx, mn) if step == 1 else symbolic.int_ceil(tile_extent(mx, mn), step)
            for mx, mn, (_, _, step) in zip(rng.max_element(), rng.min_element(), rng, strict=True)
        ] + shape
        offset = [0] * len(level.map.params) + offset
    return shape, offset


def lifted_strides(
    shape: list[symbolic.SymbolicType], own: dt.Array, coalesced: bool
) -> tuple[list[symbolic.SymbolicType], symbolic.SymbolicType]:
    """Packed strides and size of ``shape``, ``own``'s shape with leading map dimensions. Coalesced, the map
    dimensions vary fastest, the last (``threadIdx.x``) innermost; contiguous, slowest, so one iteration's slice is
    consecutive. The own axes keep their C or Fortran order.

    :raises NotImplementedError: ``own`` is neither packed-C nor packed-Fortran.
    """
    if own.is_packed_c_strides():
        own_order = list(reversed(range(len(own.shape))))
    elif own.is_packed_fortran_strides():
        own_order = list(range(len(own.shape)))
    else:
        raise NotImplementedError(f"Cannot lift {own}: only packed C or Fortran strides are supported.")
    prepended = len(shape) - len(own.shape)
    # ``strides_from_layout`` takes dimensions innermost-first
    own_dims, mapped = [d + prepended for d in own_order], list(reversed(range(prepended)))
    full = own.clone()
    full.set_shape(shape)
    strides, total_size = full.strides_from_layout(*(mapped + own_dims if coalesced else own_dims + mapped))
    return list(strides), total_size


def lifted_layout(desc: dt.Array, levels: list[nodes.MapEntry], coalesced: bool = False) -> LiftedLayout:
    """``desc`` with one leading dimension per GPU map parameter: ``gpu_A[64]`` under ``map[0:128, 0:32]`` becomes
    ``[128, 32, 64]`` indexed ``gpu_A[x, y, :]``, strides ``[32, 1, 4096]`` coalesced or ``[2048, 64, 1]`` contiguous.

    :param levels: GPU map entries between the array and the kernel exit, innermost first.
    """
    shape, offset = prepend_levels(desc, levels)
    strides, total_size = lifted_strides(shape, desc, coalesced)
    return LiftedLayout(shape, strides, total_size, offset)


def apply_layout(desc: dt.Array, layout: LiftedLayout) -> None:
    desc.set_shape(layout.shape, strides=layout.strides, total_size=layout.total_size, offset=layout.offset)


def deepest_chain(name: str, chains: list[list[nodes.MapEntry]]) -> int:
    """Index of the deepest of ``chains``, which every other chain must prefix."""
    deepest = max(range(len(chains)), key=lambda k: len(chains[k]))
    for chain in chains:
        if chain != chains[deepest][: len(chain)]:
            raise NotImplementedError(f"Cannot lift '{name}': it is accessed under sibling GPU maps")
    return deepest


@dataclass(slots=True)
class LiftPlan:
    """What lifting one transient rewrites, computed before anything changes."""

    name: str
    owner: SDFG
    desc: dt.Array
    kernel: Scope
    levels: list[Level]
    layout: LiftedLayout
    edits: list[tuple[SDFGState, MultiConnectorEdge, Prefix, nodes.EntryNode | None]]
    tasklets: list[tuple[nodes.Tasklet, list[str]]]
    hops: list[Hop]
    bounds: list["ExtentBound"] = field(default_factory=list)


def plan_owner_edits(plan: LiftPlan, sdicts: dict[SDFGState, ScopeDict], stop: nodes.MapEntry | None) -> None:
    """Index every innermost access of the owner by its slice. A nest below takes a one-slice memlet, which
    ``dealias.convert_legacy_nested_sdfgs`` then widens to the whole lifted array."""
    name = plan.name
    for state, sdict in sdicts.items():
        for edge in innermost_edges(state, name):
            scope = edge.src if isinstance(edge.src, nodes.EntryNode) else sdict[edge.src]
            prefix = level_prefix(plan.levels, slicing(enclosing_entries(sdict, scope, stop)))
            for nest in (n for n in (edge.src, edge.dst) if isinstance(n, nodes.NestedSDFG)):
                single_point(prefix, f"Nest {nest.label}", name)
            # An access that neither enters nor leaves a scope names one slice; above a slicing level it would span all
            if not (
                edge.data.is_empty() or isinstance(edge.dst, nodes.EntryNode) or isinstance(edge.src, nodes.ExitNode)
            ):
                single_point(prefix, f"Memlet {edge.data} in {state.label}", name)
            plan.edits.append((state, edge, prefix, scope))
        for node in state.nodes():
            if isinstance(node, nodes.Tasklet) and needs_prefix(node, name):
                chain = slicing(enclosing_entries(sdict, sdict[node], stop))
                plan.tasklets.append((node, single_point(level_prefix(plan.levels, chain), node.label, name)))


def assigns_symbol(sdfg: SDFG, name: str) -> bool:
    """Whether ``sdfg`` gives ``name`` its own value rather than reading one from its caller."""
    if any(isinstance(cfr, LoopRegion) and cfr.loop_variable == name for cfr in sdfg.all_control_flow_regions()):
        return True
    return any(name in edge.data.assignments for edge in sdfg.all_interstate_edges())


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


def sdfg_chain(inner: SDFG, outer: SDFG) -> list[SDFG]:
    """``inner`` and its ancestors up to and including ``outer``, innermost first."""
    chain = [inner]
    while chain[-1] is not outer:
        chain.append(chain[-1].parent_sdfg)
    return chain


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
    for inner in hierarchy[hierarchy.index(sdfg) : -1]:
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
        idx = [begin + symbolic.pystr_to_symbolic(i).subs(mapping) for (begin, end, step), i in zip(window, idx)]
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


def extent_bounds(desc: dt.Array, hierarchy: list[SDFG], kernel: Scope) -> list[ExtentBound] | None:
    """A host bound for every shape symbol of ``desc`` the kernel assigns, or ``None`` if one has none: each must be
    assigned one integer element read of data the kernel's SDFG holds in device memory or in memory the host code
    before the kernel reads (``count = counts[b]``)."""
    from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES, host_storages

    entry, state = kernel
    kernel_sdfg = state.sdfg
    result = []
    taken = OrderedSet((*kernel_sdfg.arrays, *kernel_sdfg.symbols))
    for sym in sorted({str(s) for d in desc.shape for s in symbolic.pystr_to_symbolic(d).free_symbols}):
        if not any(assigns_symbol(sdfg, sym) for sdfg in hierarchy[:-1]):
            continue
        read = kernel_assigned_read(hierarchy, sym)
        located = read and read_in_kernel_sdfg(hierarchy, *read)
        inlined = located and inline_kernel_reads(hierarchy, located[1])
        if not inlined:
            return None
        container = located[0]
        index, reads = inlined
        ctype = kernel_sdfg.arrays[container].dtype
        on_device = kernel_sdfg.arrays[container].storage in GPU_RESIDENT_STORAGES
        readable = GPU_RESIDENT_STORAGES if on_device else host_storages(entry, state)
        if not np.issubdtype(ctype.type, np.integer) or any(
            kernel_sdfg.arrays[name].storage not in readable for name in (container, *reads)
        ):
            return None
        bound = f"{sym}_max"
        if bound in kernel_sdfg.symbols and not any(b.bound == bound for b in result):
            # Already folded for an array this kernel lifted before
            result.append(ExtentBound(sym, bound, container, index, ctype, on_device, list(reads), folded=True))
            continue
        bound = utils.find_new_name(bound, taken)
        taken.add(bound)
        result.append(ExtentBound(sym, bound, container, index, ctype, on_device, list(reads)))
    return result


def compute_bound(extent: ExtentBound, kernel: Scope) -> None:
    """Assign ``extent.bound`` before the kernel's state: the maximum, and at least 0, of the element read over the
    kernel's iterations, folded where the data lives (a device map for device data)."""
    entry, kernel_state = kernel
    sdfg = kernel_state.sdfg
    storage = dtypes.StorageType.GPU_Global if extent.on_device else dtypes.StorageType.CPU_Heap
    schedule = dtypes.ScheduleType.GPU_Device if extent.on_device else dtypes.ScheduleType.Sequential
    fold, desc = sdfg.add_array(
        f"{extent.bound}_fold", [1], extent.dtype, storage=storage, transient=True, find_new_name=True
    )
    region = kernel_state.parent_graph
    seed_state = region.add_state_before(
        kernel_state, f"bound_{extent.bound}", is_start_block=region.start_block is kernel_state
    )
    seed_tasklet, seed_entry, seed_exit = seed_state.add_mapped_tasklet(
        f"seed_{extent.bound}",
        {"__s": "0:1"},
        {},
        "o = 0",
        {"o": Memlet(f"{fold}[0]")},
        schedule=schedule,
        external_edges=True,
    )
    fold_state = region.add_state_after(seed_state, f"fold_{extent.bound}")
    params = dict(zip(entry.map.params, (f"{b}:{e}+1:{st}" for b, e, st in entry.map.range), strict=True))
    index = ", ".join(extent.index)
    if extent.reads:
        inputs = {
            read_connector(name): Memlet.from_array(name, sdfg.arrays[name])
            for name in OrderedSet((extent.container, *extent.reads))
        }
        code = f"o = {read_connector(extent.container)}[{index}]"
    else:
        inputs, code = {"v": Memlet(f"{extent.container}[{index}]")}, "o = v"
    fold_tasklet, fold_entry, fold_exit = fold_state.add_mapped_tasklet(
        f"fold_{extent.bound}",
        params,
        inputs,
        code,
        {"o": Memlet(f"{fold}[0]", wcr="lambda a, b: max(a, b)")},
        schedule=schedule,
        external_edges=True,
    )
    source = fold
    if extent.on_device:
        seed_entry.map.gpu_block_size = [1, 1, 1]
        fold_entry.map.gpu_block_size = [
            int(size) for size in Config.get("compiler", "cuda", "default_block_size").split(",")
        ]
        host, desc = sdfg.add_array(
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


def bind_bounds(plan: "LiftPlan") -> None:
    """Fold every new bound before the launch and pass each bound by name into every nest down to the owner."""
    for extent in plan.bounds:
        if not extent.folded:
            compute_bound(extent, plan.kernel)
    for sdfg in sdfg_chain(plan.owner, plan.kernel[1].sdfg)[:-1]:
        node = sdfg.parent_nsdfg_node
        for extent in plan.bounds:
            if extent.bound not in sdfg.symbols:
                sdfg.add_symbol(extent.bound, extent.dtype)
            node.symbol_mapping[extent.bound] = extent.bound


def size_known_at_launch(desc: dt.Array, kernel: Scope, folded: set[str]) -> bool:
    """Whether ``desc``, written in the symbols of the kernel's SDFG, names only symbols defined before the launch
    or ``folded`` there: an allocation made there cannot use a size the kernel computes."""
    entry, state = kernel
    defined = state.symbols_defined_at(entry)
    names = {str(s) for s in desc.free_symbols} - folded
    return all(n in defined for n in names) and not names & set(entry.map.params)


def plan_hops(plan: LiftPlan, inner: dt.Array, kernel_sdfg: SDFG) -> bool:
    """Plan the descriptor of every SDFG from the owner's parent up to the kernel's, starting from the lifted
    ``inner``; ``False`` if a shape names a symbol a nest computes or the launch cannot see."""
    kernel = plan.kernel[0]
    sdfg = plan.owner
    while sdfg is not kernel_sdfg:
        node, state = sdfg.parent_nsdfg_node, sdfg.parent
        folded = {extent.bound for extent in plan.bounds}
        names = [str(s) for s in inner.free_symbols if str(s) not in folded]
        unknown = [n for n in names if n not in node.symbol_mapping or assigns_symbol(sdfg, n)]
        if unknown:
            logger.debug("Not lifting '%s': %s computes or does not map %s", plan.name, sdfg.name, unknown)
            return False
        outer = copy.deepcopy(inner)
        symbolic.safe_replace(node.symbol_mapping, lambda repl, desc=outer: replace_properties_dict(desc, repl))
        sdict = state.scope_dict()
        entries = enclosing_entries(sdict, sdict[node], kernel if state.sdfg is kernel_sdfg else None)
        levels = make_levels(state.sdfg, state, slicing(entries))
        shape, offset = prepend_levels(outer, [level.entry for level in reversed(levels)])
        outer.set_shape(shape, offset=offset)
        outer.transient = False
        outer.storage = dtypes.StorageType.GPU_Global
        plan.hops.append(Hop(node, state, levels, [state.exit_node(e) for e in entries], outer))
        inner, sdfg = outer, state.sdfg
    if not size_known_at_launch(inner, plan.kernel, {extent.bound for extent in plan.bounds}):
        logger.debug("Not lifting '%s': the kernel computes its size", plan.name)
        return False
    # The outermost descriptor is allocated by codegen instead of expected as a kernel input.
    if plan.hops:
        plan.hops[-1].desc.transient = True
    return True


def assign_strides(plan: LiftPlan, own: dt.Array, coalesced: bool) -> None:
    """Give the outermost lifted array packed strides; every nest's descriptor, a view of its trailing dimensions,
    takes the trailing strides."""
    shape = list(plan.hops[-1].desc.shape) if plan.hops else plan.layout.shape
    strides, total_size = lifted_strides(shape, own, coalesced)
    plan.layout = plan.layout._replace(strides=strides[len(shape) - len(plan.layout.shape) :], total_size=total_size)
    for hop in plan.hops:
        hop.desc.set_shape(hop.desc.shape, strides=strides[len(shape) - len(hop.desc.shape) :], total_size=total_size)


def plan_lift(name: str, owner: SDFG, kernel: Scope, coalesced: bool) -> LiftPlan | None:
    """Everything the lift of ``owner``'s ``name`` rewrites, or ``None`` if its size is only known inside the kernel.

    :raises NotImplementedError: The array has no single slice per kernel iteration.
    """
    kernel_sdfg = kernel[1].sdfg
    stop = kernel[0] if owner is kernel_sdfg else None
    sdicts = {state: state.scope_dict() for state in owner.states()}
    accesses = [(node, state) for state, sdict in sdicts.items() for node in state.data_nodes() if node.data == name]
    chains = [slicing(enclosing_entries(sdicts[state], sdicts[state][node], stop)) for node, state in accesses]
    deepest = deepest_chain(name, chains)
    desc = owner.arrays[name]
    levels = make_levels(owner, accesses[deepest][1], chains[deepest])
    # An extent the kernel reads per thread is sized by its maximum over the kernel, folded before the launch
    bounds = extent_bounds(desc, sdfg_chain(owner, kernel_sdfg), kernel) or []
    sized = copy.deepcopy(desc)
    sized.set_shape(
        [
            symbolic.pystr_to_symbolic(d).subs({symbolic.symbol(b.symbol): symbolic.symbol(b.bound) for b in bounds})
            for d in desc.shape
        ]
    )
    plan = LiftPlan(
        name,
        owner,
        desc,
        kernel,
        levels,
        lifted_layout(sized, [level.entry for level in reversed(levels)]),
        [],
        [],
        [],
        bounds,
    )
    if levels:
        plan_owner_edits(plan, sdicts, stop)
        if stop is None:
            refuse_control_flow_reads(owner, name)
    elif any(isinstance(n, nodes.Tasklet) and needs_prefix(n, name) for state in sdicts for n in state.nodes()):
        # Widening the nest restates memlets and control flow, not tasklet bodies.
        raise NotImplementedError(f"A tasklet of {owner.name} names '{name}' in its body")
    lifted = copy.deepcopy(sized)
    apply_layout(lifted, plan.layout)
    if not plan_hops(plan, lifted, kernel_sdfg):
        return None
    assign_strides(plan, desc, coalesced)
    return plan


def carry_out_of_kernel(plan: LiftPlan) -> None:
    """Place an access of the array after the kernel, so that code generation allocates it outside. The path to it
    moves no data: a data edge between two accesses of one array is a copy into the whole array."""
    kernel, state = plan.kernel
    accesses = [node for node in state.data_nodes() if node.data == plan.name]
    source = next((node for node in accesses if state.out_degree(node) == 0), accesses[-1])
    sdict = state.scope_dict()
    exits = [state.exit_node(entry) for entry in enclosing_entries(sdict, sdict[source], kernel)]
    state.add_memlet_path(source, *exits, state.add_access(plan.name), memlet=Memlet())


def pass_out_through_nests(plan: LiftPlan) -> None:
    """Declare the array in every SDFG up to the kernel's and connect each nest's copy to its parent's slice."""
    inner_name = plan.name
    for hop in plan.hops:
        outer_name = hop.state.sdfg.add_datadesc(plan.name, hop.desc, find_new_name=True)
        hop.node.add_out_connector(inner_name)
        prefix = level_prefix(hop.levels, [level.entry for level in hop.levels])
        whole = subsets.Range.from_array(hop.desc).ndrange()
        hop.state.add_memlet_path(
            hop.node,
            *hop.exits,
            hop.state.add_access(outer_name),
            src_conn=inner_name,
            memlet=Memlet(data=outer_name, subset=subsets.Range(prefix + whole[len(prefix) :])),
        )
        inner_name = outer_name


def restride_views(sdfg: SDFG, name: str, old: list[symbolic.SymbolicType], new: list[symbolic.SymbolicType]) -> None:
    """Views of ``name`` in ``sdfg`` scale their strides as the lift scaled ``old`` to ``new``: one factor for every
    axis, the packed extent of the map dimensions placed inside them (1 when they are outside)."""
    factor = symbolic.simplify(new[-1] / old[-1])
    if factor == 1:
        return
    scaled, pending = {name}, [name]
    while pending:
        viewed = pending.pop()
        for state in sdfg.states():
            for node in state.data_nodes():
                if node.data != viewed:
                    continue
                for edge in state.all_edges(node):
                    other = edge.dst if edge.src is node else edge.src
                    if (
                        isinstance(other, nodes.AccessNode)
                        and other.data not in scaled
                        and isinstance(other.desc(sdfg), dt.View)
                    ):
                        view = other.desc(sdfg)
                        view.set_shape(
                            view.shape,
                            strides=[st * factor for st in view.strides],
                            total_size=view.total_size * factor,
                        )
                        scaled.add(other.data)
                        pending.append(other.data)


def restride_nests(sdfg: SDFG, name: str, old: list[symbolic.SymbolicType], new: list[symbolic.SymbolicType]) -> None:
    """Nests below that describe ``name`` with its own strides ``old`` take ``new``, so their window stays a plain
    slice of the lifted array that ``dealias`` widens; so do the views of it in each of them.

    A nest that already describes the whole container addresses it as the parent does; its edges take the whole of
    it too, so that the window ``dealias`` sees starts at the slice's origin rather than at the accessed elements."""
    restride_views(sdfg, name, old, new)
    outer = sdfg.arrays[name]
    for state in sdfg.states():
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            edges = [(e.dst_conn, e) for e in state.in_edges(node)] + [(e.src_conn, e) for e in state.out_edges(node)]
            for conn in dict.fromkeys(c for c, e in edges if c is not None and e.data.data == name):
                inner = node.sdfg.arrays[conn]
                if list(inner.strides) == old:
                    if list(inner.shape) == list(outer.shape):
                        for c, edge in edges:
                            if c == conn and edge.data.data == name:
                                edge.data.subset = subsets.Range.from_array(outer)
                                edge.data.volume = edge.data.subset.num_elements()
                    inner.set_shape(inner.shape, strides=new, total_size=inner.total_size)
                    restride_nests(node.sdfg, conn, old, new)


def move_array(plan: LiftPlan) -> None:
    """Give the planned transient one slice per kernel iteration and allocate it outside the kernel."""
    bind_bounds(plan)
    own_strides = list(plan.desc.strides)
    restride_nests(plan.owner, plan.name, own_strides, list(plan.layout.strides[-len(own_strides) :]))
    scopes: dict[SDFGState, dict[nodes.EntryNode, None]] = {}
    for state, edge, prefix, scope in plan.edits:
        prefix_memlet(edge, plan.name, prefix)
        if scope is not None:
            scopes.setdefault(state, {})[scope] = None
    for tasklet, point in plan.tasklets:
        code = prepend_subscript_indices(tasklet.code.as_string, plan.name, point)
        tasklet.code = properties.CodeBlock(code, tasklet.language)
    apply_layout(plan.desc, plan.layout)
    for state, entries in scopes.items():
        tree = state.scope_tree()
        propagation.propagate_memlets_scope(plan.owner, state, [tree[entry] for entry in entries])
    if not plan.hops:
        carry_out_of_kernel(plan)
        return
    plan.desc.transient = False
    pass_out_through_nests(plan)


def find_kernel(sdict: ScopeDict, scope: nodes.EntryNode | None, state: SDFGState, outer: Scope | None) -> Scope | None:
    for entry in enclosing_entries(sdict, scope, None):
        if entry.schedule in dtypes.GPU_KERNEL_SCHEDULES:
            return entry, state
    return outer


def collect_users(
    sdfg: SDFG, outer: Scope | None, register_limit: int, users: dict[tuple[SDFG, str], list[Scope | None]]
) -> None:
    """Record the kernel around every access of a move candidate in ``sdfg`` and the SDFGs nested in it."""
    for state in sdfg.states():
        sdict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                collect_users(node.sdfg, find_kernel(sdict, sdict[node], state, outer), register_limit, users)
            elif isinstance(node, nodes.AccessNode) and is_move_candidate(sdfg.arrays[node.data], register_limit):
                kernel = find_kernel(sdict, sdict[node], state, outer)
                found = users.setdefault((sdfg, node.data), [])
                if kernel not in found:
                    found.append(kernel)


def move_candidates(sdfg: SDFG, register_limit: int) -> list[tuple[str, dt.Array, SDFG, list[Scope]]]:
    """Transients that :func:`is_move_candidate`, accessed only inside GPU kernels, with those kernels."""
    users: dict[tuple[SDFG, str], list[Scope | None]] = {}
    collect_users(sdfg, None, register_limit, users)
    return [(name, owner.arrays[name], owner, found) for (owner, name), found in users.items() if None not in found]


def copies_global_transient_in_kernel(sdfg: SDFG, state: SDFGState, edge: MultiConnectorEdge) -> bool:
    if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
        return False
    if edge.data.is_empty() or edge.data.wcr is not None:
        return False
    descs = (sdfg.arrays[edge.src.data], sdfg.arrays[edge.dst.data])
    # An edge into or out of a view, or setting a reference, aliases its data; it copies nothing.
    if any(isinstance(desc, (dt.View, dt.Reference)) for desc in descs):
        return False
    return (
        all(desc.storage is dtypes.StorageType.GPU_Global for desc in descs)
        and any(desc.transient for desc in descs)
        and (is_devicelevel_gpu(sdfg, state, edge.src) or is_devicelevel_gpu(sdfg, state, edge.dst))
    )


def check_no_global_copy_in_kernel(sdfg: SDFG) -> None:
    """:raises ValueError: A copy into or out of a transient ``GPU_Global`` array survives inside a kernel, where
    nothing can allocate it."""
    offenders = [
        f"  - {edge.src.data} -> {edge.dst.data} in state '{state.label}' ({nsdfg.name})"
        for nsdfg in sdfg.all_sdfgs_recursive()
        for state in nsdfg.states()
        for edge in state.edges()
        if copies_global_transient_in_kernel(nsdfg, state, edge)
    ]
    if offenders:
        raise ValueError(
            "Transient GPU_Global arrays cannot live inside a kernel scope. Offenders:\n" + "\n".join(offenders)
        )


@properties.make_properties
@transformation.explicit_cf_compatible
class MoveArrayOutOfKernel(ppl.Pass):
    """Move transient arrays a GPU kernel cannot hold in registers out of the kernel (``GPU_Device`` or
    ``GPU_Persistent`` map).

    A ``Register`` array of constant size up to ``register_limit`` elements stays where it is, and a ``GPU_Global``
    one that small is demoted to ``Register``. Every other kernel-local array (symbolic size, larger than the limit,
    or ``GPU_Global`` and accumulated into) becomes a ``GPU_Global`` array outside the kernel with one disjoint slice
    per kernel and thread-block iteration. Nested SDFGs inside every such level keep their descriptors and receive
    their one slice through the memlet.
    """

    register_limit = properties.Property(
        dtype=int,
        default=16,
        desc="Largest element count of a constant-size kernel-local array kept in per-thread registers; larger and "
        "symbolically-sized ones are moved to GPU_Global memory outside the kernel.",
    )
    layout = properties.Property(
        dtype=str,
        default="coalesced",
        choices=["coalesced", "contiguous"],
        desc="Order of a moved array: ``coalesced`` puts the kernel iterations fastest, so neighboring GPU threads "
        "access neighboring elements; ``contiguous`` keeps the slice of one iteration consecutive.",
    )
    raise_on_unsupported = properties.Property(
        dtype=bool,
        default=True,
        desc="Raise on an array that cannot be given one slice per kernel iteration; otherwise warn and leave it "
        "as it is, as code generation does.",
    )

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def plan_or_refuse(self, name: str, owner: SDFG, kernels: list[Scope]) -> LiftPlan | None:
        # Planning changes nothing, so a refusal leaves the SDFG as it was.
        try:
            if len(kernels) > 1:
                raise NotImplementedError(
                    f"Transient '{name}' is shared by the kernels {[kernel for kernel, scope in kernels]}"
                )
            return plan_lift(name, owner, kernels[0], self.layout == "coalesced")
        except NotImplementedError as ex:
            if self.raise_on_unsupported:
                raise
            warnings.warn(f"Not moving '{name}' out of its GPU kernel: {ex}")
            return None

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, object]) -> int | None:
        """Demote or move every kernel-local array registers cannot hold.

        :returns: Number of arrays demoted or moved, or ``None`` if there were none.
        :raises NotImplementedError: An array cannot be given one disjoint slice per kernel iteration, and
                                     ``raise_on_unsupported`` is set.
        :raises ValueError: A transient ``GPU_Global`` copy is left inside a kernel, and ``raise_on_unsupported`` is
                            set.
        """
        candidates = move_candidates(sdfg, self.register_limit)
        # A register cannot accumulate across threads, so a WCR target stays in memory.
        accumulated = OrderedSet(
            e.data.data
            for nsdfg in (sdfg.all_sdfgs_recursive() if candidates else ())
            for state in nsdfg.states()
            for e in state.edges()
            if e.data.wcr is not None
        )
        handled = 0
        for name, desc, owner, kernels in candidates:
            if name not in accumulated and fits_registers(desc, self.register_limit):
                desc.storage = dtypes.StorageType.Register
                handled += 1
                continue
            plan = self.plan_or_refuse(name, owner, kernels)
            if plan is None:
                continue
            desc.storage = dtypes.StorageType.GPU_Global
            move_array(plan)
            # Nests now take one slice of the lifted array; restate them as the whole array they are connected to.
            dealias.convert_legacy_nested_sdfgs(plan.kernel[1].sdfg)
            handled += 1
        if self.raise_on_unsupported:
            check_no_global_copy_in_kernel(sdfg)
        return handled or None
