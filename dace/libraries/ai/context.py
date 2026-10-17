# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
The context given to the model for a library node (:func:`collect_context`): its connectors and
the containers behind them, the enclosing scopes, what the code may do there (e.g. use ``__state``
or dereference a pointer), and the compiler and hardware.
"""

import copy
import inspect
import platform
from dataclasses import dataclass, field
from typing import Any

import dace.library
from dace import data as dt
from dace import dtypes
from dace.codegen import common, compiler, compiler_family
from dace.codegen.targets.cpp import cpp_offset_expr, sym2cpp
from dace.config import Config
from dace.libraries.ai.nodes import AINode
from dace.sdfg import SDFG, SDFGState, infer_types, nodes
from dace.sdfg.scope import devicelevel_block_size, is_devicelevel_gpu
from dace.sdfg.state import ConditionalBlock, LoopRegion


@dataclass
class ConnectorInfo:
    """One connector of the library node and the container behind it."""

    name: str
    direction: str  #: ``'in'`` or ``'out'``
    ctype: str  #: The C++ type of the connector variable
    is_pointer: bool  #: If False, the connector is a plain scalar value
    element_type: str  #: The C++ type of a single element
    #: The connector type, set on the generated tasklet. Not shown in the prompt.
    conntype: dtypes.typeclass | None = None
    data: str | None = None  #: Name of the data container the memlet refers to
    container_kind: str | None = None  #: ``Array``, ``Scalar``, ``ArrayView``, ...
    dtype: str | None = None
    #: C++ element type of the container, also when the connector type is unknown
    data_ctype: str | None = None
    shape: tuple[str, ...] | None = None
    strides: tuple[str, ...] | None = None
    total_size: str | None = None
    storage: str | None = None
    #: The declared storage, if inference resolved it to another one
    storage_declared: str | None = None
    transient: bool | None = None
    subset: str | None = None
    num_elements: str | None = None
    wcr: str | None = None
    dynamic: bool = False
    #: Whether the generated code may dereference this pointer where it runs (see :class:`Capabilities`)
    dereferenceable: bool = True
    base_offset: str | None = None  #: Offset of the pointer in the container
    view_shape: tuple[str, ...] | None = None  #: Extent of the subset this connector covers
    index_formula: str | None = None  #: How to address one element of the view
    contiguous: bool | None = None  #: Whether the subset is one contiguous block of memory


@dataclass
class NestingFrame:
    """One scope between the library node and the top-level SDFG (see :func:`collect_nesting`)."""

    kind: str  #: ``map``, ``consume``, ``scope``, ``loop``, ``conditional``, ``region``,
    #: ``nested_sdfg`` or ``sdfg``
    label: str
    detail: str = ""  #: The iteration space or condition
    schedule: str | None = None
    unroll: bool | None = None
    block_size: str | None = None
    symbol_mapping: dict[str, str] | None = None


@dataclass
class Capabilities:
    """What the generated code may do where the node is, e.g. in a GPU kernel or on the host."""

    device_level: bool
    gpu_backend: str | None = None
    state_available: bool = True
    current_stream_available: bool = False
    environments_allowed: str = "full"  #: ``full`` or ``device-headers-only``
    #: Per connector, whether the code may dereference the pointer where it runs
    dereferenceable: dict[str, bool] = field(default_factory=dict)


@dataclass
class ResolvedDefaults:
    """
    The schedules and storages DaCe's inference resolves ``Default`` to, computed on a copy of the
    SDFG and keyed by ``guid``, which the copy preserves. See :func:`infer_defaults`.
    """

    #: Node guid -> resolved schedule
    schedules: dict[str, dtypes.ScheduleType] = field(default_factory=dict)
    #: (SDFG guid, container name) -> resolved storage
    storage: dict[tuple[str, str], dtypes.StorageType] = field(default_factory=dict)
    #: Node guid -> the node and its state in the resolved copy.
    nodes: dict[str, tuple[nodes.Node, SDFGState]] = field(default_factory=dict)


@dataclass
class TargetInfo:
    """The compiler and hardware the generated code will be built for."""

    platform: str
    machine: str
    cpu_model: str | None = None
    cpu_features: str | None = None
    host_compiler: str | None = None
    compiler_family: str | None = None
    host_flags: str | None = None
    build_type: str | None = None
    cpp_standard: str | None = None
    gpu_backend: str | None = None
    gpu_flags: str | None = None
    gpu_architectures: str | None = None
    gpu_names: str | None = None


@dataclass
class EnvironmentInfo:
    """A registered DaCe environment that the generated code may use."""

    path: str  #: The full class path
    headers: list[str]  #: The headers it includes in the file of the generated code
    state_fields: list[str]  #: The fields it adds to the state struct


@dataclass
class ExpansionContext:
    """The context given to the model for one library node."""

    node_type: str
    node_name: str
    description: str = ""
    class_docstring: str = ""
    node_properties: dict[str, str] = field(default_factory=dict)
    connectors: list[ConnectorInfo] = field(default_factory=list)
    nesting: list[NestingFrame] = field(default_factory=list)
    capabilities: Capabilities | None = None
    symbols: dict[str, str] = field(default_factory=dict)
    target: TargetInfo | None = None
    available_environments: list[EnvironmentInfo] = field(default_factory=list)
    #: Registered environments that define no installation check.
    unchecked_environments: list[EnvironmentInfo] = field(default_factory=list)


def _scope_frame(
    entry: nodes.EntryNode, state: SDFGState, sdfg: SDFG, defaults: ResolvedDefaults | None = None
) -> NestingFrame:
    """
    Builds a nesting frame for a dataflow scope (map, consume or general scope).

    :param entry: The scope entry node.
    :param state: The state the entry node belongs to.
    :param sdfg: The SDFG the state belongs to.
    :param defaults: Resolved schedules and storage, from :func:`infer_defaults`.
    :return: The corresponding nesting frame.
    """
    if isinstance(entry, nodes.MapEntry):
        params = ", ".join(entry.map.params)
        detail = f"{params} = {entry.map.range}"
        block_size = None
        if entry.map.schedule in dtypes.GPU_SCHEDULES:
            try:
                bs = devicelevel_block_size(sdfg, state, entry)
                block_size = ", ".join(str(b) for b in bs) if bs else None
            except Exception:
                block_size = None
        return NestingFrame(
            kind="map",
            label=entry.map.label,
            detail=detail,
            schedule=_resolved_schedule(entry, defaults),
            unroll=entry.map.unroll,
            block_size=block_size,
        )
    if isinstance(entry, nodes.ConsumeEntry):
        return NestingFrame(
            kind="consume",
            label=entry.consume.label,
            detail=f"{entry.consume.pe_index} < {entry.consume.num_pes}",
            schedule=_resolved_schedule(entry, defaults),
        )
    return NestingFrame(kind="scope", label=str(entry))


def _region_frame(region: Any) -> NestingFrame:
    """
    Builds a nesting frame for a control flow region.

    :param region: The control flow region (loop, conditional, or a plain named region).
    :return: The corresponding nesting frame.
    """
    if isinstance(region, LoopRegion):
        init = region.init_statement.as_string if region.init_statement is not None else ""
        cond = region.loop_condition.as_string if region.loop_condition is not None else ""
        update = region.update_statement.as_string if region.update_statement is not None else ""
        var = region.loop_variable or "?"
        return NestingFrame(
            kind="loop",
            label=region.label,
            detail=f"for {var}: init [{init}]; while [{cond}]; update [{update}]",
            schedule="sequential",
            unroll=region.unroll,
        )
    if isinstance(region, ConditionalBlock):
        conds = [c.as_string if c is not None else "else" for c, _ in region.branches]
        return NestingFrame(kind="conditional", label=region.label, detail=" | ".join(conds))
    return NestingFrame(kind="region", label=region.label)


def collect_nesting(
    node: nodes.Node, state: SDFGState, sdfg: SDFG, defaults: ResolvedDefaults | None = None
) -> list[NestingFrame]:
    """
    Lists the scopes enclosing a node: dataflow scopes, control flow regions and nested SDFGs.

    :param node: The node to start from.
    :param state: The state containing ``node``.
    :param sdfg: The SDFG containing ``state``.
    :return: The enclosing scopes, ordered innermost first. The last frame is always the top-level
             SDFG.
    """
    frames: list[NestingFrame] = []
    cur_node: nodes.Node = node
    cur_state: SDFGState = state
    cur_sdfg: SDFG = sdfg

    while True:
        # 1. Dataflow scopes within the current state
        sdict = cur_state.scope_dict()
        entry = sdict.get(cur_node)
        while entry is not None:
            frames.append(_scope_frame(entry, cur_state, cur_sdfg, defaults))
            entry = sdict.get(entry)

        # 2. Control flow regions between the state and its SDFG
        region = cur_state.parent_graph
        while region is not None and not isinstance(region, SDFG):
            frames.append(_region_frame(region))
            region = region.parent_graph

        # 3. Nested SDFG boundary, or the top level
        nsdfg_node = cur_sdfg.parent_nsdfg_node
        if nsdfg_node is None:
            symbols = ", ".join(f"{k}: {v}" for k, v in sorted(cur_sdfg.symbols.items()))
            frames.append(
                NestingFrame(kind="sdfg", label=cur_sdfg.name, detail=f"symbols: {symbols}" if symbols else "")
            )
            break
        mapping = {k: str(v) for k, v in sorted(nsdfg_node.symbol_mapping.items())}
        frames.append(
            NestingFrame(
                kind="nested_sdfg",
                label=nsdfg_node.label,
                detail=f'inner SDFG "{cur_sdfg.name}"',
                symbol_mapping=mapping,
            )
        )
        cur_node = nsdfg_node
        cur_state = cur_sdfg.parent
        cur_sdfg = cur_sdfg.parent_sdfg

    return frames


def _describe_view(info: ConnectorInfo, desc: Any, memlet: Any) -> None:
    """
    Records where a pointer connector points (the first element of the subset) and how to index it.

    :param info: The connector being described, updated in place.
    :param desc: The data descriptor behind the connector.
    :param memlet: The memlet attached to the connector.
    """
    if not info.is_pointer or memlet.subset is None or not isinstance(desc, dt.Data):
        return
    try:
        info.base_offset = cpp_offset_expr(desc, memlet.subset)
        sizes = list(memlet.subset.size())
        info.view_shape = tuple(str(s) for s in sizes)

        strides = memlet.subset.absolute_strides(desc.strides)
        terms = [f"i{dim}" if str(stride) == "1" else f"i{dim}*{sym2cpp(stride)}" for dim, stride in enumerate(strides)]
        ranges = ", ".join(f"i{dim} in 0:{sym2cpp(size)}" for dim, size in enumerate(sizes))
        info.index_formula = f"{info.name}[{' + '.join(terms)}]  with {ranges}"
        info.contiguous = bool(memlet.subset.is_contiguous_subset(desc))
    except Exception:  # The subset and strides are reported either way
        pass


def _infer_conntype(
    node: nodes.LibraryNode,
    name: str,
    direction: str,
    state: SDFGState,
    sdfg: SDFG,
    edge: Any,
    defaults: ResolvedDefaults | None = None,
) -> dtypes.typeclass | None:
    """
    Determines a connector type with :mod:`dace.sdfg.infer_types`, as code generation would for the
    library node, on the copy with resolved defaults if there is one.

    :param node: The library node being expanded.
    :param name: The connector name.
    :param direction: ``'in'`` or ``'out'``.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :param edge: The edge attached to this connector, or ``None``.
    :return: The inferred type, or ``None`` if it cannot be determined.
    """
    declared = (node.in_connectors if direction == "in" else node.out_connectors).get(name)
    if declared is not None and declared.type is not None:
        return declared
    if edge is None or edge.data.data is None:
        return None

    # Inference reads storage, so it runs where the defaults are resolved
    if defaults is not None and node.guid in defaults.nodes:
        node, state = defaults.nodes[node.guid]
        sdfg = state.sdfg
    if direction == "in":
        return infer_types.infer_in_connector_type(sdfg, state, node, name)
    return infer_types.infer_out_connector_type(sdfg, state, node, name)


def _connector_info(
    name: str,
    conntype: dtypes.typeclass | None,
    direction: str,
    state: SDFGState,
    sdfg: SDFG,
    edge: Any,
    defaults: ResolvedDefaults | None = None,
) -> ConnectorInfo:
    """
    Describes a single connector of the library node.

    :param name: The connector name, which is also the C++ variable name in the generated code.
    :param conntype: The inferred connector type, if any.
    :param direction: ``'in'`` or ``'out'``.
    :param state: The state containing the library node.
    :param sdfg: The SDFG containing the state.
    :param edge: The edge attached to this connector, or ``None``.
    :return: The collected connector information.
    """
    usable = conntype is not None and conntype.type is not None
    is_pointer = isinstance(conntype, dtypes.pointer)
    element_type = conntype.base_type.ctype if is_pointer else (conntype.ctype if usable else "auto")
    info = ConnectorInfo(
        name=name,
        direction=direction,
        ctype=conntype.ctype if usable else "auto",
        is_pointer=is_pointer,
        element_type=element_type,
        conntype=conntype if usable else None,
    )
    if edge is None:
        return info

    memlet = edge.data
    info.subset = str(memlet.subset) if memlet.subset is not None else None
    try:
        info.num_elements = str(memlet.subset.num_elements()) if memlet.subset is not None else None
    except Exception:
        info.num_elements = None
    info.wcr = str(memlet.wcr) if memlet.wcr is not None else None
    info.dynamic = bool(memlet.dynamic)

    # The memlet's own container, even if it is a view: codegen offsets the connector relative to it
    if memlet.data is None or memlet.data not in sdfg.arrays:
        return info
    data_name, desc = memlet.data, sdfg.arrays[memlet.data]
    info.data = data_name
    info.container_kind = type(desc).__name__
    info.dtype = str(desc.dtype)
    info.data_ctype = desc.dtype.ctype
    resolved_storage = (defaults.storage if defaults else {}).get((sdfg.guid, data_name), desc.storage)
    info.storage = resolved_storage.name
    info.storage_declared = desc.storage.name if resolved_storage != desc.storage else None
    info.transient = desc.transient
    if isinstance(desc, dt.Data):
        info.shape = tuple(str(s) for s in desc.shape)
    if isinstance(desc, (dt.Array, dt.View)):
        info.strides = tuple(str(s) for s in desc.strides)
        info.total_size = str(desc.total_size)
    _describe_view(info, desc, memlet)
    return info


def collect_connectors(
    node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG, defaults: ResolvedDefaults | None = None
) -> list[ConnectorInfo]:
    """
    Describes every connector of a library node and the data behind it.

    :param node: The library node being expanded.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :return: One :class:`ConnectorInfo` per connector, inputs first.
    """
    in_edges = {e.dst_conn: e for e in state.in_edges(node) if e.dst_conn is not None}
    out_edges = {e.src_conn: e for e in state.out_edges(node) if e.src_conn is not None}

    result = []
    for direction, connectors, edges in (("in", node.in_connectors, in_edges), ("out", node.out_connectors, out_edges)):
        # Sorted, so that the prompt is the same in every run
        for name in sorted(connectors):
            edge = edges.get(name)
            ctype = _infer_conntype(node, name, direction, state, sdfg, edge, defaults)
            result.append(_connector_info(name, ctype, direction, state, sdfg, edge, defaults))
    return result


def collect_capabilities(
    node: nodes.LibraryNode,
    state: SDFGState,
    sdfg: SDFG,
    connectors: list[ConnectorInfo],
) -> Capabilities:
    """
    Determines what the generated code is allowed to do at this point in the SDFG.

    :param node: The library node being expanded.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :param connectors: The already-collected connector information.
    :return: The capability block for this expansion.
    """
    device_level = is_devicelevel_gpu(sdfg, state, node)
    storages = {c.name: dtypes.StorageType[c.storage] for c in connectors if c.storage is not None}

    execution_space = dtypes.ScheduleType.GPU_Device if device_level else dtypes.ScheduleType.CPU_Multicore
    for conn in connectors:
        storage = storages.get(conn.name)
        conn.dereferenceable = storage is None or bool(dtypes.can_access(execution_space, storage))

    # A GPU stream is in scope for host code that operates on memory the device can reach
    touches_gpu_memory = any(
        storage is not dtypes.StorageType.Register and dtypes.can_access(dtypes.ScheduleType.GPU_Device, storage)
        for storage in storages.values()
    )

    try:
        backend = common.get_gpu_backend()
    except RuntimeError:
        backend = None

    return Capabilities(
        device_level=device_level,
        gpu_backend=backend,
        state_available=not device_level,
        current_stream_available=not device_level and touches_gpu_memory,
        environments_allowed="device-headers-only" if device_level else "full",
        dereferenceable={c.name: c.dereferenceable for c in connectors},
    )


def collect_target_info(device_level: bool) -> TargetInfo:
    """
    Describes the compiler and hardware the generated code will be built for.

    :param device_level: True if the tasklet will be compiled as GPU device code.
    :return: The target description.
    """
    cpu_model, cpu_flags = compiler.host_cpu()
    host_compiler = compiler_family.host_compiler()
    try:
        family = compiler_family.detect(host_compiler)
    except Exception:
        family = None
    try:
        host_flags = compiler_family.cpu_args()
    except Exception:
        host_flags = Config.get("compiler", "cpu", "args")

    info = TargetInfo(
        platform=platform.system(),
        machine=platform.machine(),
        cpu_model=cpu_model,
        cpu_features=" ".join(cpu_flags) or None,
        host_compiler=host_compiler,
        compiler_family=family,
        host_flags=host_flags,
        build_type=Config.get("compiler", "build_type"),
        cpp_standard=Config.get("compiler", "cpp_standard"),
    )

    try:
        backend = common.get_gpu_backend()
    except RuntimeError:
        backend = None
    # What nvcc or hipcc will target: the configured architecture, else the local devices
    configured = Config.get("compiler", "cuda", "cuda_arch" if backend == "cuda" else "hip_arch") if backend else ""
    gpus = common.local_gpus() if backend else ()
    if configured:
        prefix = "sm_" if backend == "cuda" else ""
        architectures = ", ".join(prefix + a.strip() for a in configured.split(",") if a.strip())
    else:
        architectures = ", ".join(sorted({gpu.architecture for gpu in gpus if gpu.architecture}))
    if architectures or device_level:
        info.gpu_backend = backend
        info.gpu_architectures = architectures or None
        info.gpu_names = ", ".join(sorted({gpu.name for gpu in gpus if gpu.name})) or None
        if backend == "hip":
            info.gpu_flags = Config.get("compiler", "cuda", "hip_args")
        elif backend == "cuda":
            info.gpu_flags = Config.get("compiler", "cuda", "args")
    return info


def has_unresolved_defaults(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> bool:
    """
    Checks whether a schedule or storage relevant to a node is unresolved, which is the case when
    it is expanded before compilation.

    :param node: The library node being expanded.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :return: True if :func:`infer_defaults` is needed.
    """
    for edge in state.all_edges(node):
        name = edge.data.data
        if name is None or name not in sdfg.arrays:
            continue
        desc = sdfg.arrays[name]
        if desc.storage is dtypes.StorageType.Default:
            return True
        # Inference replaces the storage of a nested SDFG's non-transient with that of the outer container
        if not desc.transient and sdfg.parent is not None:
            if infer_types.get_storage_from_parent(name, sdfg) != desc.storage:
                return True

    if node.schedule is dtypes.ScheduleType.Default:
        return True
    cur_node: nodes.Node = node
    cur_state: SDFGState = state
    cur_sdfg: SDFG = sdfg
    while cur_state is not None:
        sdict = cur_state.scope_dict()
        entry = sdict.get(cur_node)
        while entry is not None:
            if entry.schedule is dtypes.ScheduleType.Default:
                return True
            entry = sdict.get(entry)
        cur_node = cur_sdfg.parent_nsdfg_node
        cur_state = cur_sdfg.parent
        cur_sdfg = cur_sdfg.parent_sdfg
    return False


def infer_defaults(sdfg: SDFG) -> ResolvedDefaults:
    """
    Resolves the ``Default`` schedules and storages with
    :func:`dace.sdfg.infer_types.set_default_schedule_and_storage_types`, on a copy of the root SDFG.

    :param sdfg: Any SDFG in the tree.
    :return: The resolved values, or none if inference fails (e.g. on an ambiguous scope).
    """
    try:
        copied = copy.deepcopy(sdfg.root_sdfg)
        infer_types.set_default_schedule_and_storage_types(copied, [None])
    except Exception as e:
        if Config.get_bool("debugprint"):
            print(
                f'[ai] Could not infer defaults for "{sdfg.name}" ({e}); reporting schedules and storage as declared.'
            )
        return ResolvedDefaults()

    resolved = ResolvedDefaults()
    for nested in copied.all_sdfgs_recursive():
        for name, desc in nested.arrays.items():
            resolved.storage[(nested.guid, name)] = desc.storage
    for node, parent in copied.all_nodes_recursive():
        if isinstance(parent, SDFGState):
            resolved.nodes[node.guid] = (node, parent)
            if isinstance(node, (nodes.MapEntry, nodes.ConsumeEntry, nodes.LibraryNode)):
                resolved.schedules[node.guid] = node.schedule
    return resolved


def _annotate_resolved(declared: Any, inferred: Any) -> str:
    """
    Renders a value, noting when inference resolved it.

    :param declared: What the SDFG says.
    :param inferred: What inference decided, or ``None``.
    :return: The name to report.
    """
    if inferred is None or inferred == declared:
        return declared.name
    return f"{inferred.name} (resolved from {declared.name})"


def _resolved_schedule(
    node: nodes.MapEntry | nodes.ConsumeEntry | nodes.LibraryNode, defaults: ResolvedDefaults | None
) -> str:
    """
    Renders one node's schedule, noting when it was resolved from ``Default``.

    :param node: The scope entry or library node.
    :param defaults: The result of :func:`infer_defaults`, if available.
    :return: The schedule to report.
    """
    declared = node.schedule
    return _annotate_resolved(declared, (defaults.schedules if defaults else {}).get(node.guid))


def collect_available_environments(device_level: bool) -> tuple[list[EnvironmentInfo], list[EnvironmentInfo]]:
    """
    Lists the registered DaCe library environments.

    :param device_level: Whether the code is GPU device code, which is compiled in the GPU code file
                         and gets that file's headers.
    :return: The environments whose ``is_installed`` check passes, and those that define no check,
             each sorted by class path.
    """
    backend = "cuda" if device_level else "frame"
    installed, unchecked = [], []
    for path, env in sorted(dace.library._DACE_REGISTERED_ENVIRONMENTS.items()):
        check = getattr(env, "is_installed", None)  # Optional in the environment protocol
        if check is not None:
            try:
                if not check():
                    continue
            except Exception:  # A failing check means not installed
                continue
        # A list of headers is included in the host code file only (see the frame code generator)
        headers = env.headers if isinstance(env.headers, dict) else {"frame": env.headers}
        info = EnvironmentInfo(path, list(headers.get(backend, [])), list(env.state_fields))
        (unchecked if check is None else installed).append(info)
    return installed, unchecked


def collect_class_docstring(node: nodes.LibraryNode) -> str:
    """
    Returns the docstring of a library node's class or of a base class below
    :class:`~dace.sdfg.nodes.LibraryNode`.

    :param node: The library node being expanded.
    :return: The documentation, cleaned of indentation, or an empty string.
    """
    for cls in type(node).__mro__:
        if cls is nodes.LibraryNode:
            break
        doc = cls.__dict__.get("__doc__")
        if doc and doc.strip():
            return inspect.cleandoc(doc)
    return ""


def collect_symbols(node: nodes.LibraryNode, state: SDFGState) -> dict[str, str]:
    """
    :param node: The library node being expanded.
    :param state: The state containing the node.
    :return: The symbols in scope at the node (SDFG, interstate, loop and map symbols), with their C++ types.
    """
    return {name: stype.ctype for name, stype in state.symbols_defined_at(node).items()}


def collect_context(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> ExpansionContext:
    """
    Gathers the context of a library node.

    :param node: The library node being expanded.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :return: The complete expansion context.
    """
    # Inference copies the whole SDFG, so it only runs when needed
    defaults = infer_defaults(sdfg) if has_unresolved_defaults(node, state, sdfg) else ResolvedDefaults()
    connectors = collect_connectors(node, state, sdfg, defaults)
    capabilities = collect_capabilities(node, state, sdfg, connectors)
    nesting = collect_nesting(node, state, sdfg, defaults)

    # Properties about the node's place in the graph, which are reported separately
    skipped = (
        "debuginfo",
        "environments",
        "in_connectors",
        "out_connectors",
        "implementation",
        "instrument",
        "location",
        "label",
        "name",
        "description",
        "guid",
        "schedule",
    )
    properties: dict[str, str] = {}
    for prop, value in node.properties():
        if prop.attr_name in skipped or value is None:
            continue
        properties[prop.attr_name] = str(value)

    description = node.description.strip() if isinstance(node, AINode) else ""
    installed, unchecked = collect_available_environments(capabilities.device_level)
    return ExpansionContext(
        node_type=type(node).__name__,
        node_name=node.name,
        description=description,
        # A description replaces the class documentation
        class_docstring="" if description else collect_class_docstring(node),
        node_properties=properties,
        connectors=connectors,
        nesting=nesting,
        capabilities=capabilities,
        symbols=collect_symbols(node, state),
        target=collect_target_info(capabilities.device_level),
        available_environments=installed,
        unchecked_environments=unchecked,
    )
