# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Collection of the context an AI model needs in order to write the body of a tasklet that replaces
a library node.

The central entry point is :func:`collect_context`, which gathers four kinds of information:

* **Connectors** -- the C++ variables the generated code will see, their types, and the data
  descriptors behind them (:class:`ConnectorInfo`).
* **Nesting** -- every map, loop, control flow region and nested SDFG between the library node and
  the top-level SDFG (:func:`collect_nesting`, :class:`NestingFrame`).
* **Capabilities** -- what the generated code is allowed to do in this particular slot, e.g.
  whether ``__state`` exists and whether a pointer may be dereferenced on the host
  (:class:`Capabilities`).
* **Target** -- the compiler, its flags, and the local CPU/GPU architecture
  (:class:`TargetInfo`).
"""

import copy
import inspect
import platform
from dataclasses import dataclass, field
from typing import Any

from dace import data as dt
from dace import dtypes
from dace.config import Config
from dace.sdfg import SDFG, SDFGState, infer_types, nodes
from dace.sdfg.scope import devicelevel_block_size, get_node_schedule, is_devicelevel_gpu


@dataclass
class ConnectorInfo:
    """Everything known about one connector of the library node being expanded."""

    name: str
    direction: str  #: ``'in'`` or ``'out'``
    ctype: str  #: The C++ type of the variable that will be in scope
    is_pointer: bool  #: If False, the connector is a plain scalar value
    element_type: str  #: The C++ type of a single element
    #: The resolved connector type, which the expansion stamps onto the generated tasklet so that
    #: what the model was told stays true through code generation. Not rendered into the prompt.
    conntype: dtypes.typeclass | None = None
    data: str | None = None  #: Name of the data container the memlet refers to
    container_kind: str | None = None  #: ``Array``, ``Scalar``, ``Stream``, ``View``, ...
    dtype: str | None = None
    #: C++ element type of that container. Unlike ``element_type`` this does not depend on the
    #: connector's own type having been inferable, which is what makes it usable as a fallback.
    data_ctype: str | None = None
    shape: tuple[str, ...] | None = None
    strides: tuple[str, ...] | None = None
    total_size: str | None = None
    storage: str | None = None
    #: What the SDFG declared, when inference resolved it to something else (i.e. it said
    #: ``Default``). ``None`` when the two agree.
    storage_declared: str | None = None
    transient: bool | None = None
    subset: str | None = None
    num_elements: str | None = None
    wcr: str | None = None
    dynamic: bool = False
    #: Whether the generated code may dereference this pointer where it runs (set once the
    #: execution space is known; see :class:`Capabilities`).
    dereferenceable: bool = True
    base_offset: str | None = None  #: What the pointer is offset by, relative to the container
    view_shape: tuple[str, ...] | None = None  #: Extent of the subset this connector covers
    index_formula: str | None = None  #: How to address one element of the view
    contiguous: bool | None = None  #: False when the view is strided within a larger container


@dataclass
class NestingFrame:
    """
    One level of nesting between the library node and the top-level SDFG.

    Frames are produced innermost-first by :func:`collect_nesting`.
    """

    kind: str  #: ``map``, ``consume``, ``scope``, ``loop``, ``conditional``, ``region``,
    #: ``nested_sdfg`` or ``sdfg``
    label: str
    detail: str = ""  #: Human-readable description of the iteration space or condition
    schedule: str | None = None
    unroll: bool | None = None
    block_size: str | None = None
    symbol_mapping: dict[str, str] | None = None


@dataclass
class Capabilities:
    """
    What the generated code may and may not do at this particular point in the SDFG.

    The same library node type expands very differently depending on where it sits -- inside a GPU
    kernel, on the host over GPU arrays, or on the host over CPU arrays -- and getting it wrong
    fails either at compile time (``__state`` in device code) or, worse, at run time
    (dereferencing a device pointer on the host). These flags are handed to the model explicitly
    rather than left to be inferred.
    """

    device_level: bool
    effective_schedule: str
    gpu_backend: str | None = None
    state_available: bool = True
    current_stream_available: bool = False
    block_size: str | None = None
    environments_allowed: str = "full"  #: ``full`` or ``device-headers-only``
    #: Per connector, whether the generated code may dereference the pointer *where it runs*. This
    #: is a match between the code's execution space and the data's storage space, not a property
    #: of the storage alone: GPU global memory is unreachable from the host and perfectly
    #: reachable from device code.
    dereferenceable: dict[str, bool] = field(default_factory=dict)


@dataclass
class ResolvedDefaults:
    """
    What DaCe's own inference decides for the ``Default`` schedules and storages of an SDFG.

    Keyed by ``guid``, which a deep copy preserves, so answers computed on a throwaway copy can be
    read back against the caller's graph. See :func:`infer_defaults`.
    """

    #: Node guid -> the schedule that node will actually run under.
    schedules: dict[str, dtypes.ScheduleType] = field(default_factory=dict)
    #: (SDFG guid, container name) -> the storage that container will actually be allocated in.
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
class ExpansionContext:
    """The complete context handed to the model for one library node expansion."""

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
    available_environments: list[str] = field(default_factory=list)


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
    return NestingFrame(kind="scope", label=str(entry), schedule=_resolved_schedule(entry, defaults))


def _region_frame(region: Any) -> NestingFrame:
    """
    Builds a nesting frame for a control flow region.

    :param region: The control flow region (loop, conditional, or a plain named region).
    :return: The corresponding nesting frame.
    """
    from dace.sdfg.state import ConditionalBlock, LoopRegion  # Avoid a cyclic import

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
    Walks outward from a node to the top-level SDFG, recording every enclosing scope.

    The walk crosses all three kinds of boundary in turn: dataflow scopes within a state, control
    flow regions from the state up to its SDFG, and nested SDFG boundaries. It is modeled on
    :func:`dace.sdfg.scope.get_node_schedule`, which performs the same traversal to resolve a
    single schedule.

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
        schedule = getattr(nsdfg_node, "schedule", None)
        frames.append(
            NestingFrame(
                kind="nested_sdfg",
                label=nsdfg_node.label,
                detail=f'inner SDFG "{cur_sdfg.name}"',
                schedule=schedule.name if schedule is not None else None,
                symbol_mapping=mapping,
            )
        )
        cur_node = nsdfg_node
        cur_state = cur_sdfg.parent
        cur_sdfg = cur_sdfg.parent_sdfg

    return frames


def _describe_view(info: ConnectorInfo, desc: Any, memlet: Any) -> None:
    """
    Records where a pointer connector points and how to index it.

    A pointer connector is a *view* into its container, positioned at the first
    element of the memlet subset and strided by the container's strides.
    Getting it wrong produces code that compiles cleanly and computes the wrong
    answer, which the compile check cannot catch.

    :param info: The connector being described, updated in place.
    :param desc: The data descriptor behind the connector.
    :param memlet: The memlet attached to the connector.
    """
    from dace.codegen.targets.cpp import cpp_offset_expr, sym2cpp  # Avoid a cyclic import

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
    except Exception:
        # A subset this helper cannot render must not stop the expansion; the raw subset and
        # strides are reported either way.
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
    Determines the type a connector will have in the generated code.

    Connector types are normally filled in by :func:`dace.sdfg.infer_types.infer_connector_types`,
    which runs after expansion. Since whether a connector is a value or a pointer is exactly what
    the generated code depends on, the same rule is applied here without mutating the SDFG.

    ``infer_types`` refuses to pass GPU global memory to a library node by value, and that guard is
    applied here too -- otherwise a single-element GPU operand would be handed to the generated
    code as a ``float``, which DaCe reads with ``float _a = A[0];`` on the host and which therefore
    faults at run time. The guard is written as ``isinstance(node, nodes.LibraryNode)`` and the
    library node is gone by the time inference runs, so it does not survive expansion on its own:
    :meth:`dace.libraries.ai.expansion.ExpandAI._make_tasklet` stamps the types computed here onto
    the tasklet's connectors, which is what makes this prediction hold rather than merely describe
    an intent. A prediction that did not hold would be worse than none: the model is told the type,
    and the compile check uses it.

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
    # The resolved storage, not the declared one: "Default" says nothing about whether this
    # pointer may be dereferenced where the code runs, and it is what `dereferenceable` is computed
    # from. The declared value is reported alongside so the prompt does not appear to contradict
    # the graph.
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
        # Sorted, because connectors are often declared as a set: an unstable order would make the
        # prompt differ between runs for no reason.
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
    defaults: ResolvedDefaults | None = None,
) -> Capabilities:
    """
    Determines what the generated code is allowed to do at this point in the SDFG.

    :param node: The library node being expanded.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :param connectors: The already-collected connector information.
    :param defaults: Resolved schedules and storage, from :func:`infer_defaults`.
    :return: The capability block for this expansion.
    """
    from dace.codegen import common  # Avoid a cyclic import through the code generator

    device_level = is_devicelevel_gpu(sdfg, state, node)
    storages = {c.name: dtypes.StorageType[c.storage] for c in connectors if c.storage is not None}

    # Reachability is a match between where the code runs and where the data lives. Device code
    # can dereference GPU memory and not host memory; host code, the other way round.
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

    block_size = None
    if device_level:
        try:
            bs = devicelevel_block_size(sdfg, state, node)
            block_size = ", ".join(str(b) for b in bs) if bs else None
        except Exception:
            block_size = None

    # Inference resolves the library node's own schedule too, so prefer its answer over the
    # unresolved one the SDFG carries
    schedule = (defaults.schedules if defaults else {}).get(node.guid) or get_node_schedule(sdfg, state, node)

    return Capabilities(
        device_level=device_level,
        effective_schedule=schedule.name,
        gpu_backend=backend,
        state_available=not device_level,
        current_stream_available=not device_level and touches_gpu_memory,
        block_size=block_size,
        environments_allowed="device-headers-only" if device_level else "full",
        dereferenceable={c.name: c.dereferenceable for c in connectors},
    )


def collect_target_info(device_level: bool) -> TargetInfo:
    """
    Describes the compiler and hardware the generated code will be built for.

    :param device_level: True if the tasklet will be compiled as GPU device code.
    :return: The target description.
    """
    from dace.codegen import common, compiler, compiler_family  # Avoid a cyclic import through the code generator

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
    Checks whether anything the context reports about this slot still says ``Default``.

    The context only depends on the slot itself: the library node's schedule, the schedules of the
    scopes enclosing it, and the storage of the containers behind its connectors. On the usual path
    -- expansion during compilation -- :mod:`dace.codegen.codegen` has already resolved all of
    them, and they can be read off the graph directly. Only a node expanded by hand, before
    compilation, still needs :func:`infer_defaults`.

    :param node: The library node being expanded.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :return: True if a schedule or storage relevant to the node is still ``Default``.
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
    Asks DaCe what every ``Default`` schedule and storage will actually become.

    An SDFG mostly says ``Default`` for both, and reading it back reports that verbatim. Code
    generation does not run on that: :func:`dace.sdfg.infer_types.set_default_schedule_and_storage_types`
    resolves the defaults first. A top-level map becomes ``CPU_Multicore`` -- an OpenMP parallel
    loop -- and a transient with no better answer becomes ``Register``. Both matter to a model
    writing code for the slot:

    * A body told "Default" that in fact runs on every thread is missing the one fact it needs to
      avoid writing a race: a function-local ``static``, a shared scratch buffer, or any
      non-reentrant state would be wrong there.
    * "Default" storage says nothing about whether a pointer may be dereferenced where the code
      runs, which is what separates a working tasklet from one that faults at run time.

    That inference is deliberately *reused* rather than reimplemented here. It is not the table
    lookup it first appears to be: storage types are inferred first and the schedule decision then
    consults them, so a ``Default`` map over GPU arrays becomes ``GPU_Device`` and not the
    ``CPU_Multicore`` a plain ``SCOPEDEFAULT_SCHEDULE`` lookup would give. Reimplementing it is how
    the prompt ends up confidently describing a slot that does not exist -- the same failure mode as
    predicting connector types independently of ``infer_connector_types``.

    Inference mutates, so it runs on a throwaway copy of the whole SDFG: the caller's graph must
    not have its storage and schedule decisions frozen early as a side effect of building a prompt.

    :param sdfg: Any SDFG in the tree; inference always runs from the root, since a nested SDFG's
                 defaults depend on the scopes above it.
    :return: The resolved decisions. Empty if inference could not run, in which case callers fall
             back to what the SDFG says.
    """
    try:
        root = sdfg.root_sdfg
    except Exception:
        root = sdfg

    try:
        copied = copy.deepcopy(root)
        infer_types.set_default_schedule_and_storage_types(copied, [None])
    except Exception as e:
        # Inference raises on an ambiguous scope. Building a prompt must never be the thing that
        # fails an expansion, so fall back to the unresolved schedules.
        if Config.get_bool("debugprint"):
            print(
                f'[ai] Could not infer defaults for "{getattr(sdfg, "name", "?")}" ({e}); reporting schedules '
                "and storage as declared."
            )
        return ResolvedDefaults()

    resolved = ResolvedDefaults()
    for nested in copied.all_sdfgs_recursive():
        for name, desc in nested.arrays.items():
            if isinstance(getattr(desc, "storage", None), dtypes.StorageType):
                resolved.storage[(nested.guid, name)] = desc.storage
    for node, parent in copied.all_nodes_recursive():
        if isinstance(parent, SDFGState):
            resolved.nodes[node.guid] = (node, parent)
            schedule = getattr(node, "schedule", None)
            if isinstance(schedule, dtypes.ScheduleType):
                resolved.schedules[node.guid] = schedule
    return resolved


def _annotate_resolved(declared: Any, inferred: Any) -> str:
    """
    Renders a value, noting when inference changed it.

    Reported as resolved rather than silently rewritten: the SDFG really does say ``Default``, and
    someone comparing the prompt against the graph should not conclude the prompt is lying.

    :param declared: What the SDFG says.
    :param inferred: What inference decided, or ``None``.
    :return: The name to report.
    """
    if inferred is None or inferred == declared:
        return declared.name
    return f"{inferred.name} (resolved from {declared.name})"


def _resolved_schedule(node: nodes.Node, defaults: ResolvedDefaults | None) -> str | None:
    """
    Renders one node's schedule, noting when it was resolved from ``Default``.

    :param node: The scope entry or library node.
    :param defaults: The result of :func:`infer_defaults`, if available.
    :return: The schedule to report, or ``None`` if the node has none.
    """
    declared = getattr(node, "schedule", None)
    if declared is None:
        return None
    return _annotate_resolved(declared, (defaults.schedules if defaults else {}).get(node.guid))


def collect_available_environments() -> list[str]:
    """
    Lists the DaCe library environments that are installed on this machine.

    Environments that do not define an ``is_installed`` check are assumed to be available, since
    that is the convention used throughout ``dace.libraries``.

    :return: Full class paths of usable environments.
    """
    import dace.library  # Avoid a cyclic import

    available = []
    for path, env in dace.library._DACE_REGISTERED_ENVIRONMENTS.items():
        check = getattr(env, "is_installed", None)
        try:
            if check is None or check():
                available.append(path)
        except Exception:
            continue
    return sorted(available)


def collect_class_docstring(node: nodes.LibraryNode) -> str:
    """
    Returns the documentation of a library node's class, if it has any of its own.

    For a library node shipped with DaCe this is usually the closest thing to a specification that
    exists -- ``Gemm``, for instance, documents itself as ``alpha * (A @ B) + beta * C`` -- so it
    is worth far more to the model than the class name alone. The MRO is walked so that a node
    which documents its behavior on a shared base class (as the MPI nodes do) is covered too, but
    it stops at :class:`~dace.sdfg.nodes.LibraryNode`, whose docstring describes the IR rather than
    any particular computation.

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
    Gathers everything the model needs in order to write this tasklet.

    :param node: The library node being expanded.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :return: The complete expansion context.
    """
    # Computed once and shared, since every part of the context that mentions a schedule or a
    # storage type has to agree with it. Inference copies the whole SDFG, so it only runs when the
    # slot itself does not already say what it resolves to.
    defaults = infer_defaults(sdfg) if has_unresolved_defaults(node, state, sdfg) else ResolvedDefaults()
    connectors = collect_connectors(node, state, sdfg, defaults)
    capabilities = collect_capabilities(node, state, sdfg, connectors, defaults)
    nesting = collect_nesting(node, state, sdfg, defaults)

    # Properties the model does not need: they describe the node's place in the graph rather than
    # what it computes, and are reported separately.
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

    description = str(getattr(node, "description", "") or "").strip()
    return ExpansionContext(
        node_type=type(node).__name__,
        node_name=node.name,
        description=description,
        # An explicit description is the specification; the class documentation
        # would only be generic background next to it (and for AINode it merely
        # describes this mechanism).
        class_docstring="" if description else collect_class_docstring(node),
        node_properties=properties,
        connectors=connectors,
        nesting=nesting,
        capabilities=capabilities,
        symbols=collect_symbols(node, state),
        target=collect_target_info(capabilities.device_level),
        available_environments=collect_available_environments(),
    )
