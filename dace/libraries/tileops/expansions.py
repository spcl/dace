# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The kinds of expansion of a tile node.

DaCe ties an expansion class to the one node it expands, so a node declares its own subclass of each backend; the
lowering itself is the node's ``pure_tasklet`` and ``isa_tasklet``. The ``block`` expansion is registered for every
node at once (:mod:`dace.libraries.tileops.nodes`).
"""

import re
from copy import deepcopy as dcpy

import dace
from dace import data, dtypes
from dace.libraries.tileops.lanes import LaneDistribution, distributed_lanes
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

#: The thread index of a ``block`` lowering, the parameter of its thread-block map.
THREAD_INDEX = "__tile_t"


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
    running the per-lane body of every ``threads``-th output element from ``t``.

    A node whose lanes depend on each other runs on one thread. The register tiles the node touches move to shared
    memory, where every thread of the block sees them; the barriers between the thread-block maps come from the GPU
    code generator's shared-memory synchronization.
    """

    environments: list[type] = []

    @classmethod
    def expansion(cls, node: nodes.LibraryNode, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        threads = min(node.output_elements(), node.group_threads()) if node.lanes_independent else 1
        tasklet = None
        if threads > 1:
            distribution = LaneDistribution(THREAD_INDEX, threads)
            with distributed_lanes(distribution):
                tasklet = node.pure_tasklet(parent_state, parent_sdfg)
            if distribution.loops == 0:
                threads, tasklet = 1, None
        if tasklet is None:
            with distributed_lanes(None):
                tasklet = node.pure_tasklet(parent_state, parent_sdfg)

        tasklet = renamed_connectors(tasklet)
        edges = [*parent_state.in_edges(node), *parent_state.out_edges(node)]
        for edge in edges:
            share_tile(parent_sdfg.arrays[edge.data.data])

        nsdfg = dace.SDFG(f"{node.label}_{node.group.name.lower()}")
        defined = parent_state.symbols_defined_at(node)
        state = nsdfg.add_state(is_start_block=True)
        entry, exit_node = state.add_map(
            f"{node.label}_threads", {THREAD_INDEX: f"0:{threads}"}, schedule=dtypes.ScheduleType.GPU_ThreadBlock
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
