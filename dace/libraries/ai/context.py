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

import inspect
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dace import data as dt
from dace import dtypes
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.scope import devicelevel_block_size, get_node_schedule, is_devicelevel_gpu

#: Storage types that live in GPU memory and are therefore unreachable from host code.
DEVICE_ONLY_STORAGE = (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared)

#: Storage types that live in host memory and are therefore unreachable from device code.
HOST_ONLY_STORAGE = (dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_ThreadLocal)

#: Storage types that live in GPU memory (used to decide whether a GPU stream is in scope).
GPU_STORAGE = (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared, dtypes.StorageType.CPU_Pinned)


@dataclass
class ConnectorInfo:
    """ Everything known about one connector of the library node being expanded. """

    name: str
    direction: str  #: ``'in'`` or ``'out'``
    ctype: str  #: The C++ type of the variable that will be in scope
    is_pointer: bool  #: If False, the connector is a plain scalar value
    element_type: str  #: The C++ type of a single element
    #: The resolved connector type, which the expansion stamps onto the generated tasklet so that
    #: what the model was told stays true through code generation. Not rendered into the prompt.
    conntype: Optional[dtypes.typeclass] = None
    data: Optional[str] = None  #: Name of the data container the memlet refers to
    container_kind: Optional[str] = None  #: ``Array``, ``Scalar``, ``Stream``, ``View``, ...
    dtype: Optional[str] = None
    #: C++ element type of that container. Unlike ``element_type`` this does not depend on the
    #: connector's own type having been inferable, which is what makes it usable as a fallback.
    data_ctype: Optional[str] = None
    shape: Optional[Tuple[str, ...]] = None
    strides: Optional[Tuple[str, ...]] = None
    total_size: Optional[str] = None
    storage: Optional[str] = None
    transient: Optional[bool] = None
    subset: Optional[str] = None
    num_elements: Optional[str] = None
    wcr: Optional[str] = None
    dynamic: bool = False
    #: Whether the generated code may dereference this pointer where it runs (set once the
    #: execution space is known; see :class:`Capabilities`).
    dereferenceable: bool = True
    base_offset: Optional[str] = None  #: What the pointer is offset by, relative to the container
    view_shape: Optional[Tuple[str, ...]] = None  #: Extent of the subset this connector covers
    index_formula: Optional[str] = None  #: How to address one element of the view
    contiguous: Optional[bool] = None  #: False when the view is strided within a larger container


@dataclass
class NestingFrame:
    """
    One level of nesting between the library node and the top-level SDFG.

    Frames are produced innermost-first by :func:`collect_nesting`.
    """

    kind: str  #: ``map``, ``consume``, ``scope``, ``loop``, ``conditional``, ``region``,
    #: ``nested_sdfg`` or ``sdfg``
    label: str
    detail: str = ''  #: Human-readable description of the iteration space or condition
    schedule: Optional[str] = None
    unroll: Optional[bool] = None
    block_size: Optional[str] = None
    symbol_mapping: Optional[Dict[str, str]] = None


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
    gpu_backend: Optional[str] = None
    state_available: bool = True
    current_stream_available: bool = False
    block_size: Optional[str] = None
    environments_allowed: str = 'full'  #: ``full`` or ``device-headers-only``
    #: Per connector, whether the generated code may dereference the pointer *where it runs*. This
    #: is a match between the code's execution space and the data's storage space, not a property
    #: of the storage alone: GPU global memory is unreachable from the host and perfectly
    #: reachable from device code.
    dereferenceable: Dict[str, bool] = field(default_factory=dict)


@dataclass
class TargetInfo:
    """ The compiler and hardware the generated code will be built for. """

    platform: str
    machine: str
    cpu_model: Optional[str] = None
    cpu_features: Optional[str] = None
    host_compiler: Optional[str] = None
    compiler_family: Optional[str] = None
    host_flags: Optional[str] = None
    build_type: Optional[str] = None
    cpp_standard: Optional[str] = None
    gpu_backend: Optional[str] = None
    gpu_flags: Optional[str] = None
    gpu_architectures: Optional[str] = None
    gpu_names: Optional[str] = None


@dataclass
class ExpansionContext:
    """ The complete context handed to the model for one library node expansion. """

    node_type: str
    node_name: str
    description: str = ''
    class_docstring: str = ''
    node_properties: Dict[str, str] = field(default_factory=dict)
    connectors: List[ConnectorInfo] = field(default_factory=list)
    nesting: List[NestingFrame] = field(default_factory=list)
    capabilities: Optional[Capabilities] = None
    symbols: Dict[str, str] = field(default_factory=dict)
    target: Optional[TargetInfo] = None
    available_environments: List[str] = field(default_factory=list)


def _describe_range(rng: Any) -> str:
    """
    Renders a subset or range as a readable string.

    :param rng: The range or subset to render.
    :return: A string description, or ``'?'`` if it cannot be rendered.
    """
    try:
        return str(rng)
    except Exception:
        return '?'


def _scope_frame(entry: nodes.EntryNode, state: SDFGState, sdfg: SDFG) -> NestingFrame:
    """
    Builds a nesting frame for a dataflow scope (map, consume or general scope).

    :param entry: The scope entry node.
    :param state: The state the entry node belongs to.
    :param sdfg: The SDFG the state belongs to.
    :return: The corresponding nesting frame.
    """
    if isinstance(entry, nodes.MapEntry):
        params = ', '.join(entry.map.params)
        detail = f'{params} = {_describe_range(entry.map.range)}'
        block_size = None
        if entry.map.schedule in dtypes.GPU_SCHEDULES:
            try:
                bs = devicelevel_block_size(sdfg, state, entry)
                block_size = ', '.join(str(b) for b in bs) if bs else None
            except Exception:
                block_size = None
        return NestingFrame(kind='map',
                            label=entry.map.label,
                            detail=detail,
                            schedule=entry.map.schedule.name,
                            unroll=entry.map.unroll,
                            block_size=block_size)
    if isinstance(entry, nodes.ConsumeEntry):
        return NestingFrame(kind='consume',
                            label=entry.consume.label,
                            detail=f'{entry.consume.pe_index} < {entry.consume.num_pes}',
                            schedule=entry.consume.schedule.name)
    return NestingFrame(kind='scope', label=str(entry), schedule=getattr(entry, 'schedule', None))


def _region_frame(region: Any) -> NestingFrame:
    """
    Builds a nesting frame for a control flow region.

    :param region: The control flow region (loop, conditional, or a plain named region).
    :return: The corresponding nesting frame.
    """
    from dace.sdfg.state import ConditionalBlock, LoopRegion  # Avoid a cyclic import

    if isinstance(region, LoopRegion):
        init = region.init_statement.as_string if region.init_statement is not None else ''
        cond = region.loop_condition.as_string if region.loop_condition is not None else ''
        update = region.update_statement.as_string if region.update_statement is not None else ''
        var = region.loop_variable or '?'
        return NestingFrame(kind='loop',
                            label=region.label,
                            detail=f'for {var}: init [{init}]; while [{cond}]; update [{update}]',
                            schedule='sequential',
                            unroll=region.unroll)
    if isinstance(region, ConditionalBlock):
        conds = [c.as_string if c is not None else 'else' for c, _ in region.branches]
        return NestingFrame(kind='conditional', label=region.label, detail=' | '.join(conds))
    return NestingFrame(kind='region', label=region.label)


def collect_nesting(node: nodes.Node, state: SDFGState, sdfg: SDFG) -> List[NestingFrame]:
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
    frames: List[NestingFrame] = []
    cur_node: nodes.Node = node
    cur_state: SDFGState = state
    cur_sdfg: SDFG = sdfg

    while True:
        # 1. Dataflow scopes within the current state
        sdict = cur_state.scope_dict()
        entry = sdict.get(cur_node)
        while entry is not None:
            frames.append(_scope_frame(entry, cur_state, cur_sdfg))
            entry = sdict.get(entry)

        # 2. Control flow regions between the state and its SDFG
        region = cur_state.parent_graph
        while region is not None and not isinstance(region, SDFG):
            frames.append(_region_frame(region))
            region = region.parent_graph

        # 3. Nested SDFG boundary, or the top level
        nsdfg_node = cur_sdfg.parent_nsdfg_node
        if nsdfg_node is None:
            symbols = ', '.join(f'{k}: {v}' for k, v in sorted(cur_sdfg.symbols.items()))
            frames.append(
                NestingFrame(kind='sdfg', label=cur_sdfg.name, detail=f'symbols: {symbols}' if symbols else ''))
            break
        mapping = {k: str(v) for k, v in sorted(nsdfg_node.symbol_mapping.items())}
        schedule = getattr(nsdfg_node, 'schedule', None)
        frames.append(
            NestingFrame(kind='nested_sdfg',
                         label=nsdfg_node.label,
                         detail=f'inner SDFG "{cur_sdfg.name}"',
                         schedule=schedule.name if schedule is not None else None,
                         symbol_mapping=mapping))
        cur_node = nsdfg_node
        cur_state = cur_sdfg.parent
        cur_sdfg = cur_sdfg.parent_sdfg

    return frames


def _descriptor_for(state: SDFGState, sdfg: SDFG, edge: Any, incoming: bool) -> Tuple[Optional[str], Optional[Any]]:
    """
    Resolves the data container behind a library node edge.

    Follows the memlet path to the outermost access node rather than trusting ``edge.data.data``,
    which may name an inner or view container when the library node sits inside a map.

    :param state: The state containing the edge.
    :param sdfg: The SDFG containing the state.
    :param edge: The edge to resolve.
    :param incoming: True if the edge is an input to the library node.
    :return: A tuple of (container name, data descriptor), either of which may be ``None``.
    """
    path = state.memlet_path(edge)
    endpoint = path[0].src if incoming else path[-1].dst
    if isinstance(endpoint, nodes.AccessNode) and endpoint.data in sdfg.arrays:
        return endpoint.data, sdfg.arrays[endpoint.data]
    if edge.data.data is not None and edge.data.data in sdfg.arrays:
        return edge.data.data, sdfg.arrays[edge.data.data]
    return None, None


def _describe_view(info: ConnectorInfo, desc: Any, memlet: Any) -> None:
    """
    Records where a pointer connector points and how to index it.

    A pointer connector is a *view* into its container, positioned at the first
    element of the memlet subset and strided by the container's strides.
    Getting it wrong produces code that compiles cleanly and computes the wrong
    answer, which the verification probe cannot catch.

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

        strides = list(desc.strides)
        terms = [f'i{dim}' if str(stride) == '1' else f'i{dim}*{sym2cpp(stride)}' for dim, stride in enumerate(strides)]
        ranges = ', '.join(f'i{dim} in 0:{sym2cpp(size)}' for dim, size in enumerate(sizes))
        info.index_formula = f'{info.name}[{" + ".join(terms)}]  with {ranges}'

        # Contiguous only if the view's own packed strides match the container's
        packed, running = [], 1
        for size in reversed(sizes):
            packed.insert(0, running)
            running *= size
        info.contiguous = all(str(a) == str(b) for a, b in zip(packed, strides))
    except Exception:
        # A subset this helper cannot render must not stop the expansion; the raw subset and
        # strides are reported either way.
        pass


def _infer_conntype(node: nodes.LibraryNode, name: str, direction: str, state: SDFGState, sdfg: SDFG,
                    edge: Any) -> Optional[dtypes.typeclass]:
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
    and the probe compiles against it.

    :param node: The library node being expanded.
    :param name: The connector name.
    :param direction: ``'in'`` or ``'out'``.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :param edge: The edge attached to this connector, or ``None``.
    :return: The inferred type, or ``None`` if it cannot be determined.
    """
    declared = (node.in_connectors if direction == 'in' else node.out_connectors).get(name)
    if declared is not None and declared.type is not None:
        return declared

    if edge is None or edge.data.data is None or edge.data.data not in sdfg.arrays:
        return None
    memlet = edge.data
    desc = sdfg.arrays[memlet.data]

    scalar = bool(memlet.subset) and memlet.subset.num_elements() == 1
    if direction == 'out':
        # A dynamic output without a write-conflict resolution has no single destination to write
        # back to, so it stays a pointer
        scalar &= not memlet.dynamic or memlet.wcr is not None
    scalar |= isinstance(desc, dt.Scalar)
    # Never by value out of GPU global memory: the load or store would run wherever the tasklet
    # does, which for a host tasklet means dereferencing a device pointer
    scalar &= desc.storage is not dtypes.StorageType.GPU_Global
    return desc.dtype if scalar else dtypes.pointer(desc.dtype)


def _connector_info(name: str, conntype: Optional[dtypes.typeclass], direction: str, state: SDFGState, sdfg: SDFG,
                    edge: Any) -> ConnectorInfo:
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
    element_type = conntype.base_type.ctype if is_pointer else (conntype.ctype if usable else 'auto')
    info = ConnectorInfo(name=name,
                         direction=direction,
                         ctype=conntype.ctype if usable else 'auto',
                         is_pointer=is_pointer,
                         element_type=element_type,
                         conntype=conntype if usable else None)
    if edge is None:
        return info

    memlet = edge.data
    info.subset = _describe_range(memlet.subset) if memlet.subset is not None else None
    try:
        info.num_elements = str(memlet.subset.num_elements()) if memlet.subset is not None else None
    except Exception:
        info.num_elements = None
    info.wcr = str(memlet.wcr) if memlet.wcr is not None else None
    info.dynamic = bool(memlet.dynamic)

    data_name, desc = _descriptor_for(state, sdfg, edge, direction == 'in')
    if desc is None:
        return info
    info.data = data_name
    info.container_kind = type(desc).__name__
    info.dtype = str(desc.dtype)
    info.data_ctype = getattr(desc.dtype, 'ctype', None)
    info.storage = desc.storage.name
    info.transient = desc.transient
    if isinstance(desc, dt.Data):
        info.shape = tuple(str(s) for s in desc.shape)
    if isinstance(desc, (dt.Array, dt.View)):
        info.strides = tuple(str(s) for s in desc.strides)
        info.total_size = str(desc.total_size)
    _describe_view(info, desc, memlet)
    return info


def collect_connectors(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> List[ConnectorInfo]:
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
    for direction, connectors, edges in (('in', node.in_connectors, in_edges), ('out', node.out_connectors, out_edges)):
        # Sorted, because connectors are often declared as a set: an unstable order would make the
        # prompt differ between runs for no reason.
        for name in sorted(connectors):
            edge = edges.get(name)
            ctype = _infer_conntype(node, name, direction, state, sdfg, edge)
            result.append(_connector_info(name, ctype, direction, state, sdfg, edge))
    return result


def collect_capabilities(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG,
                         connectors: List[ConnectorInfo]) -> Capabilities:
    """
    Determines what the generated code is allowed to do at this point in the SDFG.

    :param node: The library node being expanded.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :param connectors: The already-collected connector information.
    :return: The capability block for this expansion.
    """
    from dace.codegen import common  # Avoid a cyclic import through the code generator

    device_level = is_devicelevel_gpu(sdfg, state, node)
    touches_gpu_memory = any(c.storage in {s.name for s in GPU_STORAGE} for c in connectors)

    # Reachability is a match between where the code runs and where the data lives. Device code
    # can dereference GPU memory and not host memory; host code, the other way round.
    unreachable = {s.name for s in (HOST_ONLY_STORAGE if device_level else DEVICE_ONLY_STORAGE)}
    for conn in connectors:
        conn.dereferenceable = conn.storage not in unreachable

    try:
        backend = common.get_gpu_backend()
    except RuntimeError:
        backend = None

    block_size = None
    if device_level:
        try:
            bs = devicelevel_block_size(sdfg, state, node)
            block_size = ', '.join(str(b) for b in bs) if bs else None
        except Exception:
            block_size = None

    return Capabilities(device_level=device_level,
                        effective_schedule=get_node_schedule(sdfg, state, node).name,
                        gpu_backend=backend,
                        state_available=not device_level,
                        current_stream_available=not device_level and touches_gpu_memory,
                        block_size=block_size,
                        environments_allowed='device-headers-only' if device_level else 'full',
                        dereferenceable={c.name: c.dereferenceable
                                         for c in connectors})


def collect_target_info(device_level: bool) -> TargetInfo:
    """
    Describes the compiler and hardware the generated code will be built for.

    :param device_level: True if the tasklet will be compiled as GPU device code.
    :return: The target description.
    """
    from dace.codegen import compiler_family  # Avoid a cyclic import through the code generator
    from dace.libraries.ai import sysinfo

    cpu = sysinfo.cpu_description()
    host_compiler = compiler_family.host_compiler()
    try:
        family = compiler_family.detect(host_compiler)
    except Exception:
        family = None
    try:
        host_flags = compiler_family.cpu_args()
    except Exception:
        host_flags = Config.get('compiler', 'cpu', 'args')

    info = TargetInfo(platform=platform.system(),
                      machine=cpu.get('machine', platform.machine()),
                      cpu_model=cpu.get('model'),
                      cpu_features=cpu.get('flags'),
                      host_compiler=host_compiler,
                      compiler_family=family,
                      host_flags=host_flags,
                      build_type=Config.get('compiler', 'build_type'),
                      cpp_standard=Config.get('compiler', 'cpp_standard'))

    gpu_arch = sysinfo.gpu_architectures()
    if gpu_arch is not None or device_level:
        from dace.codegen import common
        try:
            backend = common.get_gpu_backend()
        except RuntimeError:
            backend = None
        info.gpu_backend = backend
        info.gpu_architectures = gpu_arch
        info.gpu_names = sysinfo.gpu_names()
        if backend == 'hip':
            info.gpu_flags = Config.get('compiler', 'cuda', 'hip_args')
        elif backend == 'cuda':
            info.gpu_flags = Config.get('compiler', 'cuda', 'args')
    return info


def collect_available_environments() -> List[str]:
    """
    Lists the DaCe library environments that are installed on this machine.

    Environments that do not define an ``is_installed`` check are assumed to be available, since
    that is the convention used throughout ``dace.libraries``.

    :return: Full class paths of usable environments.
    """
    import dace.library  # Avoid a cyclic import

    available = []
    for path, env in dace.library._DACE_REGISTERED_ENVIRONMENTS.items():
        check = getattr(env, 'is_installed', None)
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
        doc = cls.__dict__.get('__doc__')
        if doc and doc.strip():
            return inspect.cleandoc(doc)
    return ''


def collect_symbols(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> Dict[str, str]:
    """
    Lists the symbols that will be in scope as plain C++ variables inside the tasklet.

    :param node: The library node being expanded.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :return: A mapping from symbol name to its C++ type.
    """
    symbols: Dict[str, str] = {}
    for name, stype in sdfg.symbols.items():
        symbols[name] = stype.ctype if isinstance(stype, dtypes.typeclass) else str(stype)
    sdict = state.scope_dict()
    entry = sdict.get(node)
    while entry is not None:
        if isinstance(entry, nodes.MapEntry):
            for param in entry.map.params:
                symbols.setdefault(param, dtypes.int64.ctype)
        entry = sdict.get(entry)
    return symbols


def collect_context(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> ExpansionContext:
    """
    Gathers everything the model needs in order to write this tasklet.

    :param node: The library node being expanded.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :return: The complete expansion context.
    """
    connectors = collect_connectors(node, state, sdfg)
    capabilities = collect_capabilities(node, state, sdfg, connectors)

    # Properties the model does not need: they describe the node's place in the graph rather than
    # what it computes, and are reported separately.
    skipped = ('debuginfo', 'environments', 'in_connectors', 'out_connectors', 'implementation', 'instrument',
               'location', 'label', 'name', 'description', 'guid', 'schedule')
    properties: Dict[str, str] = {}
    for prop, value in node.properties():
        if prop.attr_name in skipped or value is None:
            continue
        properties[prop.attr_name] = str(value)

    description = str(getattr(node, 'description', '') or '').strip()
    return ExpansionContext(
        node_type=type(node).__name__,
        node_name=node.name,
        description=description,
        # An explicit description is the specification; the class documentation
        # would only be generic background next to it (and for AINode it merely
        # describes this mechanism).
        class_docstring='' if description else collect_class_docstring(node),
        node_properties=properties,
        connectors=connectors,
        nesting=collect_nesting(node, state, sdfg),
        capabilities=capabilities,
        symbols=collect_symbols(node, state, sdfg),
        target=collect_target_info(capabilities.device_level),
        available_environments=collect_available_environments())
