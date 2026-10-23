# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Memory engine (TMA) bulk copy between global and shared memory.
"""
from typing import List, Optional, Set, Tuple

import dace
from dace import data, library, nodes, dtypes, symbolic
from dace.libraries.standard.environments.memory_engine import MemoryEngine
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.standard.nodes.copy.common import (
    _build_memory_engine_barrier_init_code, _build_memory_engine_bulk_copy_code, _build_memory_engine_desc_code,
    mark_memory_engine_shared_static, memory_engine_bulk_reason, memory_engine_geometry, MemoryEngineGeometry,
    CUtensorMap, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME, MEMORY_ENGINE_BARRIER_INPUT_NAME,
    MEMORY_ENGINE_BARRIER_OUTPUT_NAME, MEMORY_ENGINE_DESC_INPUT_NAME, MEMORY_ENGINE_DESC_OUTPUT_NAME,
    MEMORY_ENGINE_GLOBAL_INPUT_NAME)

_KERNEL_SCHEDULES = (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_Persistent)
"""Schedules that start a GPU kernel."""


@library.register_expansion(CopyLibraryNode, "MemoryEngineBulk")
class ExpandMemoryEngineBulk(ExpandTransformation):
    """Memory engine based bulk copy between ``GPU_Global`` and ``GPU_Shared``.

    Currently only the TMA engine on NVIDIA Hopper+ GPUs is supported. The copy becomes a
    block-collective tasklet issuing ``cp.async.bulk.tensor`` from a leader thread. Postprocessing
    adds the support graph:
    - a host tasklet in the closest CPU scope that encodes the ``CUtensorMap``, which reaches the kernel
      as a ``const __grid_constant__`` parameter.
    - for loads, a ``GPU_Shared`` mbarrier and a tasklet initializing it once at the top of the kernel.

    The shared side is pinned to static shared memory (``GPU_Shared(dynamic=False)``), because currently
    only static shared memory is aligned to 128B as the memory engine requires.

    Caller must place this expansion outside any enclosing ``GPU_ThreadBlock`` map. This expansion *is*
    the thread-block-level operation.

    TODO: Add support for dynamic shared memory.
    """
    environments = [MemoryEngine]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        node.validate(parent_sdfg, parent_state, allow_cross_storage=True)
        reason = memory_engine_bulk_reason(node, parent_state)
        if reason is not None:
            raise ValueError(f"Cannot expand {node} with MemoryEngineBulk: {reason}")
        geometry, _ = memory_engine_geometry(node, parent_state)

        inputs = {
            INPUT_CONNECTOR_NAME: dtypes.pointer(geometry.global_desc.dtype),
            # By value through a dynamic memlet: a const reference to the kernel parameter, never a copy
            MEMORY_ENGINE_DESC_INPUT_NAME: CUtensorMap,
        }
        if geometry.is_load:
            inputs[MEMORY_ENGINE_BARRIER_INPUT_NAME] = dtypes.pointer(dtypes.uint64)
        return nodes.Tasklet(node.name,
                             inputs=inputs,
                             outputs={OUTPUT_CONNECTOR_NAME: dtypes.pointer(geometry.global_desc.dtype)},
                             code=_build_memory_engine_bulk_copy_code(node, geometry),
                             language=dace.Language.CPP)

    @staticmethod
    def postprocessing(sdfg: SDFG, state: SDFGState, expansion: nodes.Tasklet):
        geometry, reason = memory_engine_geometry(expansion, state)
        if reason is not None:
            raise ValueError(f"Cannot expand {expansion} with MemoryEngineBulk: {reason}")
        mark_memory_engine_shared_static(expansion, state, geometry)
        _add_tensor_map(sdfg, state, expansion, geometry)
        if geometry.is_load:
            _add_barrier(sdfg, state, expansion)


def _scope_chain(state: SDFGState, node: nodes.Node) -> List[nodes.EntryNode]:
    """Entry nodes enclosing ``node``, outermost first."""
    sdict = state.scope_dict()
    chain = []
    scope = sdict[node]
    while scope is not None:
        chain.insert(0, scope)
        scope = sdict[scope]
    return chain


def _outermost_kernel_entry(state: SDFGState, node: nodes.Node) -> Optional[nodes.EntryNode]:
    """The outermost kernel map entry enclosing ``node`` in ``state``, or ``None``."""
    return next((e for e in _scope_chain(state, node) if e.schedule in _KERNEL_SCHEDULES), None)


def _map_symbols_up(expr: symbolic.SymExpr, nsdfg_node: nodes.NestedSDFG) -> symbolic.SymExpr:
    """Rewrite ``expr`` from the symbols of ``nsdfg_node``'s SDFG to those of its parent."""
    expr = symbolic.pystr_to_symbolic(expr)
    mapping = {symbolic.symbol(k): symbolic.pystr_to_symbolic(v) for k, v in nsdfg_node.symbol_mapping.items()}
    return expr.subs(mapping, simultaneous=True) if mapping else expr


class _Route:
    """Where the producer of a routed input goes, and what it must feed.

    The producer writes the access node ``data`` (created by the caller in ``scope`` of ``state``), which
    is wired through ``entries`` into ``consumer.conn``. ``levels`` are the NestedSDFG nodes crossed on the
    way up (innermost first) and ``inner_params`` the map parameters of every scope that was crossed,
    neither of which the producer can see.
    """

    def __init__(self, sdfg: SDFG, state: SDFGState, scope: Optional[nodes.EntryNode], entries: List[nodes.EntryNode],
                 consumer: nodes.Node, conn: str, levels: List[nodes.NestedSDFG], inner_params: Set[str]):
        self.sdfg = sdfg
        self.state = state
        self.scope = scope
        self.entries = entries
        self.consumer = consumer
        self.conn = conn
        self.levels = levels
        self.inner_params = inner_params

    def connect(self, an: nodes.AccessNode, memlet: Memlet):
        """Wire the producer's access node ``an`` to the consumer."""
        self.state.add_memlet_path(an, *self.entries, self.consumer, dst_conn=self.conn, memlet=memlet)


def _route_input(sdfg: SDFG,
                 state: SDFGState,
                 node: nodes.Node,
                 conn: str,
                 inner_desc: data.Data,
                 name: str,
                 stop: str,
                 dynamic: bool = False) -> _Route:
    """Route a new input ``conn`` of ``node`` outward until the scope where its producer belongs.

    Crossing a NestedSDFG adds a non-transient copy of ``inner_desc`` to the nested SDFG and an input
    connector to the NestedSDFG node. ``inner_desc`` is an array so nested-SDFG code receives a pointer
    and never copies the value (a ``CUtensorMap`` must stay in kernel parameter space).

    :param stop: ``'kernel'`` stops inside the outermost kernel map (device code, run by every thread),
                 ``'host'`` stops in the scope enclosing that map.
    :param dynamic: make the routed memlets dynamic (read by reference instead of pre-read by value).
    """
    levels = []
    inner_params = set()
    while True:
        kernel_entry = _outermost_kernel_entry(state, node)
        if kernel_entry is not None:
            chain = _scope_chain(state, node)
            k = chain.index(kernel_entry)
            if stop == 'kernel':
                scope, entries = kernel_entry, chain[k + 1:]
            else:
                scope, entries = (chain[k - 1] if k > 0 else None), chain[k:]
            inner_params |= {p for e in entries for p in e.map.params}
            return _Route(sdfg, state, scope, entries, node, conn, levels, inner_params)

        nsdfg_node = sdfg.parent_nsdfg_node
        if nsdfg_node is None:
            raise ValueError(f"{node} is not inside a GPU kernel map")
        inner_name = sdfg.add_datadesc(name, inner_desc.clone(), find_new_name=True)
        chain = _scope_chain(state, node)
        inner_params |= {p for e in chain for p in e.map.params}
        memlet = Memlet.from_array(inner_name, sdfg.arrays[inner_name])
        memlet.dynamic = dynamic
        state.add_memlet_path(state.add_access(inner_name), *chain, node, dst_conn=conn, memlet=memlet)
        nsdfg_node.add_in_connector(inner_name)
        levels.append(nsdfg_node)
        node, conn = nsdfg_node, inner_name
        state, sdfg = sdfg.parent, sdfg.parent_sdfg


def _global_pointer(sdfg: SDFG, state: SDFGState, tasklet: nodes.Tasklet, geometry: MemoryEngineGeometry,
                    host_state: SDFGState) -> Tuple[nodes.AccessNode, symbolic.SymExpr]:
    """Follow the global side of the copy outward to ``host_state``.

    :returns: the host-level access node of the global container (only the outermost one; it may sit in
              another scope than the producer), and the element offset of the kernel-visible array from
              its base pointer, in host symbols.
    """
    def is_starting_edge(e):
        if geometry.is_load:
            return e.dst_conn == INPUT_CONNECTOR_NAME
        else:
            return e.src_conn == OUTPUT_CONNECTOR_NAME
    edge = next(e for e in state.all_edges(tasklet) if is_starting_edge(e))
    offset = symbolic.pystr_to_symbolic(0)
    while True:
        path = state.memlet_path(edge)
        an = path[0].src if geometry.is_load else path[-1].dst
        if state is host_state:
            return an, offset
        desc = sdfg.arrays[an.data]
        nsdfg_node = sdfg.parent_nsdfg_node
        if desc.transient or nsdfg_node is None:
            raise NotImplementedError(f"MemoryEngineBulk: cannot follow global container {an.data} out of "
                                      f"{sdfg.name}; it must be passed into the nested SDFG")
        parent_state = sdfg.parent
        if geometry.is_load:
            outer = next(e for e in parent_state.in_edges(nsdfg_node) if e.dst_conn == an.data)
        else:
            outer = next(e for e in parent_state.out_edges(nsdfg_node) if e.src_conn == an.data)
        outer_desc = sdfg.parent_sdfg.arrays[outer.data.data]
        offset = _map_symbols_up(offset, nsdfg_node)
        offset += sum(m * s for m, s in zip(outer.data.subset.min_element(), outer_desc.strides))
        edge, state, sdfg = outer, parent_state, sdfg.parent_sdfg


def _add_tensor_map(sdfg: SDFG, state: SDFGState, tasklet: nodes.Tasklet, geometry: MemoryEngineGeometry):
    """Add the host tasklet encoding the copy's ``CUtensorMap`` and route it to ``tasklet``."""
    inner_desc = data.Array(CUtensorMap, [1], storage=dtypes.StorageType.Register, transient=False)
    route = _route_input(sdfg,
                         state,
                         tasklet,
                         MEMORY_ENGINE_DESC_INPUT_NAME,
                         inner_desc,
                         "tma_desc",
                         stop='host',
                         dynamic=True)
    host_sdfg, host_state = route.sdfg, route.state

    dims, strides = geometry.dims, geometry.strides
    for nsdfg_node in route.levels:
        dims = [_map_symbols_up(d, nsdfg_node) for d in dims]
        strides = [_map_symbols_up(s, nsdfg_node) for s in strides]
    global_an, offset = _global_pointer(sdfg, state, tasklet, geometry, host_state)
    global_desc = host_sdfg.arrays[global_an.data]

    desc_name = host_sdfg.add_scalar("tma_desc",
                                     CUtensorMap,
                                     storage=dtypes.StorageType.Register,
                                     transient=True,
                                     find_new_name=True)[0]
    init = host_state.add_tasklet(f"{tasklet.label}_tma_desc_init",
                                  inputs={MEMORY_ENGINE_GLOBAL_INPUT_NAME: dtypes.pointer(global_desc.dtype)},
                                  outputs={MEMORY_ENGINE_DESC_OUTPUT_NAME: CUtensorMap},
                                  code=_build_memory_engine_desc_code(geometry, dims, strides, offset),
                                  language=dtypes.Language.CPP)
    init.environments = {MemoryEngine.full_class_path()}
    # Place ``init`` in the producer scope through its global pointer input. Only the pointer is read, so
    # a load reuses the access node feeding the kernel when it is in that scope; otherwise a fresh access
    # node is read (a store's sink would close a cycle through the kernel)
    if geometry.is_load and host_state.scope_dict()[global_an] is route.scope:
        src_an, src_chain = global_an, []
    else:
        src_an = host_state.add_access(global_an.data)
        src_chain = (_scope_chain(host_state, route.scope) + [route.scope]) if route.scope is not None else []
    host_state.add_memlet_path(src_an,
                               *src_chain,
                               init,
                               dst_conn=MEMORY_ENGINE_GLOBAL_INPUT_NAME,
                               memlet=Memlet.from_array(global_an.data, global_desc))
    desc_an = host_state.add_access(desc_name)
    host_state.add_edge(init, MEMORY_ENGINE_DESC_OUTPUT_NAME, desc_an, None, Memlet(f"{desc_name}[0]"))
    route.connect(desc_an, Memlet(f"{desc_name}[0]", dynamic=True))

    defined = set(host_state.symbols_defined_at(init).keys()) | set(host_sdfg.arrays.keys())
    used = {str(s) for e in dims + strides + [offset] for s in symbolic.pystr_to_symbolic(e).free_symbols}
    missing = (used - defined) | (used & route.inner_params)
    if missing:
        raise NotImplementedError(f"MemoryEngineBulk: the tensor map of {tasklet} depends on {sorted(missing)}, "
                                  "which are not known on the host where it is encoded")


def _add_barrier(sdfg: SDFG, state: SDFGState, tasklet: nodes.Tasklet):
    """Add a ``GPU_Shared`` mbarrier for ``tasklet``, initialized once at the top of its kernel."""
    inner_desc = data.Array(dtypes.uint64, [1], storage=dtypes.StorageType.GPU_Shared, transient=False)
    route = _route_input(sdfg, state, tasklet, MEMORY_ENGINE_BARRIER_INPUT_NAME, inner_desc, "tma_bar", stop='kernel')
    kernel_sdfg, kernel_state = route.sdfg, route.state

    # Static like the copy's shared side, so the barrier never needs the layout of dynamic shared memory
    bar_name = kernel_sdfg.add_array("tma_bar", [1],
                                     dtypes.uint64,
                                     storage=dtypes.StorageType.GPU_Shared(dynamic=False),
                                     transient=True,
                                     find_new_name=True)[0]
    init = kernel_state.add_tasklet(f"{tasklet.label}_tma_barrier_init",
                                    inputs={},
                                    outputs={MEMORY_ENGINE_BARRIER_OUTPUT_NAME: dtypes.pointer(dtypes.uint64)},
                                    code=_build_memory_engine_barrier_init_code(),
                                    language=dtypes.Language.CPP)
    init.environments = {MemoryEngine.full_class_path()}
    kernel_state.add_nedge(route.scope, init, Memlet())
    bar_an = kernel_state.add_access(bar_name)
    kernel_state.add_edge(init, MEMORY_ENGINE_BARRIER_OUTPUT_NAME, bar_an, None,
                          Memlet.from_array(bar_name, kernel_sdfg.arrays[bar_name]))
    route.connect(bar_an, Memlet.from_array(bar_name, kernel_sdfg.arrays[bar_name]))
