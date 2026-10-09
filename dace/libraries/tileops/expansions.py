# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The kinds of expansion of a tile node.

DaCe ties an expansion class to the one node it expands, so a node declares its own subclass of each backend; the
lowering itself is the node's ``pure_tasklet`` and ``isa_tasklet``. The ``block`` expansion is registered for every
node at once (:mod:`dace.libraries.tileops.nodes`).
"""

import re
from copy import deepcopy as dcpy
from typing import NamedTuple

import dace
from dace import data, dtypes
from dace.libraries.tileops.dispatch import lane_implementation
from dace.libraries.tileops.isa import isa_chunk
from dace.libraries.tileops.lanes import LaneDistribution, distributed_lanes
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

#: The thread index of a ``block`` lowering, the parameter of its thread-block map.
THREAD_INDEX = "__tile_t"
#: The name prefix of the nested SDFG and of the thread-block map of a ``block`` lowering, whose barriers the tile
#: synchronization places.
TILE_THREADS_MAP = "tile_threads_"


class ExpandTilePure(ExpandTransformation):
    """The per-lane C++ loop, which the compiler can still vectorize."""

    environments: list[type] = []

    @classmethod
    def expansion(cls, node: nodes.LibraryNode, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return node.pure_tasklet(parent_state, parent_sdfg)


class ExpandTileIsa(ExpandTransformation):
    """A call into the header of one ISA backend; a subclass names the backend and its environment."""

    backend: str

    @classmethod
    def expansion(cls, node: nodes.LibraryNode, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return node.isa_tasklet(parent_state, parent_sdfg, cls.backend)


class ExpandTileBlock(ExpandTransformation):
    """A ``WARP`` or ``BLOCK`` node in a GPU kernel: a thread-block map over the threads of its group, thread ``t``
    running its share of the output elements (:func:`block_layout`).

    The register tiles the node touches move to shared memory, where every thread of the block sees them. The barriers
    between the nodes come from :class:`~dace.transformation.passes.tile_synchronization.InsertTileSync`.
    """

    environments: list[type] = []

    @classmethod
    def expansion(cls, node: nodes.LibraryNode, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        layout = block_layout(node, parent_state)
        if layout.isa is not None:
            tasklet = chunked_isa_tasklet(node, parent_state, parent_sdfg, node.implementations[layout.isa], layout)
            threads = layout.threads
        else:
            threads, tasklet = distributed_pure_tasklet(node, parent_state, parent_sdfg, layout.threads)

        # An empty memlet only orders the node (a barrier before it), and stays on the expanded node
        edges = [
            edge for edge in (*parent_state.in_edges(node), *parent_state.out_edges(node)) if not edge.data.is_empty()
        ]
        for edge in edges:
            share_tile(parent_sdfg.arrays[edge.data.data])

        nsdfg = dace.SDFG(f"{TILE_THREADS_MAP}{node.label}")
        defined = parent_state.symbols_defined_at(node)
        state = nsdfg.add_state(is_start_block=True)
        entry, exit_node = state.add_map(
            f"{TILE_THREADS_MAP}{node.label}",
            {THREAD_INDEX: f"0:{threads}"},
            schedule=dtypes.ScheduleType.GPU_ThreadBlock,
        )
        state.add_node(tasklet)
        free = set()
        outputs = []
        for edge in edges:
            conn = edge.dst_conn if edge.dst is node else edge.src_conn
            desc = dcpy(parent_sdfg.arrays[edge.data.data])
            desc.transient = False
            nsdfg.add_datadesc(conn, desc)
            # The connector is the outer container, so the memlet keeps the outer subset
            memlet = Memlet(data=conn, subset=dcpy(edge.data.subset))
            free |= {str(s) for s in edge.data.subset.free_symbols} | {str(s) for s in desc.free_symbols}
            if edge.dst is node and lane_connector(conn) in tasklet.in_connectors:
                state.add_memlet_path(
                    state.add_read(conn), entry, tasklet, dst_conn=lane_connector(conn), memlet=memlet
                )
            elif edge.src is node and lane_connector(conn) in tasklet.out_connectors:
                outputs.append((conn, memlet))
        # The tasklet joins the map scope before its outputs leave it
        if state.in_degree(tasklet) == 0:
            state.add_nedge(entry, tasklet, Memlet())
        for conn, memlet in outputs:
            state.add_memlet_path(
                tasklet, exit_node, state.add_write(conn), src_conn=lane_connector(conn), memlet=memlet
            )
        # A descriptor declares the symbols of its shape when it is added
        for name in sorted(free - nsdfg.symbols.keys()):
            nsdfg.add_symbol(name, defined.get(name, dace.int64))
        return nsdfg


def distributed_pure_tasklet(
    node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG, threads: int
) -> tuple[int, nodes.Tasklet]:
    """The threads and the pure tasklet of ``node`` spread one lane at a time over ``threads`` threads; a node that
    computes nothing per lane runs on one thread."""
    if threads > 1:
        distribution = LaneDistribution(THREAD_INDEX, threads)
        with distributed_lanes(distribution):
            tasklet = node.pure_tasklet(state, sdfg)
        if distribution.loops > 0:
            return threads, renamed_connectors(tasklet)
    with distributed_lanes(None):
        return 1, renamed_connectors(node.pure_tasklet(state, sdfg))


class BlockLayout(NamedTuple):
    """How a ``block`` lowering spreads the ``elements`` output elements of a node: thread ``t`` takes ``chunk``
    consecutive elements from ``t * chunk``, then every ``threads * chunk``-th. ``isa`` runs each chunk when set."""

    threads: int
    chunk: int
    elements: int
    isa: str | None


def block_layout(node: nodes.LibraryNode, state: dace.SDFGState) -> BlockLayout:
    """The layout the ``block`` expansion gives ``node``, which the tile synchronization reads as well.

    A node whose lanes depend on each other, or whose output is not a tile of its elements (a lane-invariant output
    combines every lane), runs on one thread. A one-dim node with a target ISA runs it on ``lanes_per_thread``
    consecutive lanes per thread; the rest spread one lane at a time.
    """
    elements = node.output_elements()
    if not node.lanes_independent or any(
        edge.data.subset.num_elements() != elements for edge in state.out_edges(node) if not edge.data.is_empty()
    ):
        return BlockLayout(1, elements, elements, None)
    chunk = node.lanes_per_thread
    if chunk > 1 and len(node.widths) == 1 and elements % chunk == 0:
        isa = lane_implementation(node, state)
        if isa != "pure":
            return BlockLayout(min(elements // chunk, node.group_threads()), chunk, elements, isa)
    return BlockLayout(min(elements, node.group_threads()), 1, elements, None)


def chunked_isa_tasklet(
    node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG, expansion: type[ExpandTileIsa], layout: BlockLayout
) -> nodes.Tasklet:
    """The ISA call of ``node`` on the chunks of its ``layout``. A tile operand is offset to the chunk; a broadcast
    one is read whole."""
    width, chunk, threads = node.widths[0], layout.chunk, layout.threads
    with isa_chunk(chunk):
        call = node.isa_tasklet(state, sdfg, expansion.backend)
    edges = {edge.dst_conn: edge for edge in state.in_edges(node)} | {
        edge.src_conn: edge for edge in state.out_edges(node)
    }
    views = [
        f"auto {conn} = {lane_connector(conn)} + __c;"
        if edges[conn].data.subset.num_elements() == width
        else f"auto {conn} = {lane_connector(conn)};"
        for conn in (*call.in_connectors, *call.out_connectors)
    ]
    body = "\n".join(f"    {line}" for line in (*views, *call.code.as_string.splitlines()))
    code = f"for (std::size_t __c = {THREAD_INDEX} * {chunk}; __c < {width}; __c += {threads * chunk}) {{\n{body}\n}}"
    tasklet = nodes.Tasklet(
        call.label,
        inputs={lane_connector(conn): None for conn in call.in_connectors},
        outputs={lane_connector(conn): None for conn in call.out_connectors},
        code=code,
        language=dtypes.Language.CPP,
    )
    # The ISA header comes with the environment of the backend
    tasklet.environments = {env.full_class_path() for env in expansion.environments}
    return tasklet


def lane_connector(conn: str) -> str:
    """The tasklet connector of the node connector ``conn``, which names an array in the nested SDFG."""
    return f"{conn}_lanes"


def renamed_connectors(tasklet: nodes.Tasklet) -> nodes.Tasklet:
    """``tasklet`` with every connector renamed by :func:`lane_connector`, in its code too."""
    code = tasklet.code.as_string
    for conn in (*tasklet.in_connectors, *tasklet.out_connectors):
        code = re.sub(rf"(?<!\w){re.escape(conn)}(?!\w)", lane_connector(conn), code)
    return nodes.Tasklet(
        tasklet.label,
        inputs={lane_connector(conn): ctype for conn, ctype in tasklet.in_connectors.items()},
        outputs={lane_connector(conn): ctype for conn, ctype in tasklet.out_connectors.items()},
        code=code,
        language=tasklet.language,
    )


def share_tile(desc: data.Data) -> None:
    """Move a register tile to shared memory, where the threads that each compute some of its lanes all see it."""
    if (
        isinstance(desc, data.Array)
        and not isinstance(desc, data.View)
        and desc.transient
        and desc.storage in (dtypes.StorageType.Register, dtypes.StorageType.Default)
    ):
        desc.storage = dtypes.StorageType.GPU_Shared
