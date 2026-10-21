# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Schedule-tree pass that overlaps nonblocking communication with independent work.

A nonblocking operation (a ``nonblocking`` collective, ``Isend``, ``Irecv``) starts communicating, and its ``Wait``
completes it. :func:`overlap_communication` moves every start, with the work it depends on, ahead of the other work
before it, and every wait behind the work after it that does not depend on it, so that the work between them runs
while the communication is in flight. It then puts every
communication node in a state of its own, so that code generation keeps this order (communication nodes have side
effects, so state fusion does not merge them with their neighbors). NCCL (and RCCL) nodes stay in the states of
their neighbors: they run on GPU streams, which order them with the work around them on the device, while the end of
a state would make the host wait for them.

The order of the operations that start communication (starts and blocking operations) does not change: collectives
must start in the same order on every process. Waits may move past them.
"""

import dataclasses

import sympy

from dace import data
from dace.config import Config
from dace.libraries.comm.nodes import Irecv, Isend, Wait, Waitall
from dace.libraries.comm.nodes.node import CommNode, NonblockingCommNode
from dace.sdfg.analysis.schedule_tree import treenodes as tn

#: Unknown memory: a container that may alias any other (e.g., a reference)
_ANY = "*"


@dataclasses.dataclass
class _Access:
    """What a schedule tree node accesses, for reordering it with its neighbors."""

    reads: set[str]  #: Containers read (resolved to the containers they view)
    writes: set[str]  #: Containers written
    names: set[str]  #: Container names used (a view is used by name before it is defined)
    symbols: set[str]  #: Free symbols
    defines: set[str]  #: Container and symbol names defined (views, symbol assignments)
    starts: bool  #: Starts communication (a start or a blocking operation), or contains such a node
    barrier: bool  #: Nothing moves past this node (control flow transfer, references)


def _is_comm(node: tn.ScheduleTreeNode) -> bool:
    return isinstance(node, tn.LibraryCall) and isinstance(node.node, CommNode)


def _on_gpu_streams(node: tn.ScheduleTreeNode) -> bool:
    """Whether a communication node runs on GPU streams (NCCL or RCCL), which order it with the work around it on
    the device. A state of its own would make the host wait for it at the end of the state."""
    implementation = node.node.implementation or Config.get("library", "comm", "default_implementation")
    return implementation == "NCCL"


def is_start(node: tn.ScheduleTreeNode) -> bool:
    """Whether ``node`` starts a nonblocking communication operation."""
    if not isinstance(node, tn.LibraryCall):
        return False
    libnode = node.node
    return isinstance(libnode, (Isend, Irecv)) or (isinstance(libnode, NonblockingCommNode) and libnode.nonblocking)


def is_wait(node: tn.ScheduleTreeNode) -> bool:
    """Whether ``node`` completes nonblocking communication operations."""
    return isinstance(node, tn.LibraryCall) and isinstance(node.node, (Wait, Waitall))


def overlap_communication(stree: tn.ScheduleTreeRoot) -> None:
    """
    Moves nonblocking communication starts as early, and their waits as late, as possible within their scopes, and
    puts every communication node in a state of its own. Modifies ``stree`` in place.

    :param stree: The schedule tree.
    """
    aliases = _view_sources(stree)
    _overlap_in(stree, stree, aliases)


def _view_sources(stree: tn.ScheduleTreeRoot) -> dict[str, str]:
    """The container each view of the tree views."""
    sources = {}
    for node in stree.preorder_traversal():
        if isinstance(node, tn.ViewNode):
            sources[node.target] = node.source
    return sources


def _overlap_in(scope: tn.ScheduleTreeScope, root: tn.ScheduleTreeRoot, aliases: dict[str, str]) -> None:
    for child in scope.children:
        if isinstance(child, (tn.ControlFlowScope, tn.ScheduleTreeRoot)):
            _overlap_in(child, root, aliases)
    if not any(_is_comm(child) for child in scope.children):
        return
    children = list(scope.children)
    accesses = {id(child): _access(child, root, aliases) for child in children}
    _wait_for_buffers_in_flight(children, accesses, root, aliases)

    # Each start moves up with the nodes it depends on (its backward slice), ahead of the others; the first start
    # first, so that starts keep their order
    for start in [child for child in children if is_start(child)]:
        position = children.index(start)
        first = _region_start(children, position, accesses)
        before = children[first:position]
        slice_ = _slice(list(reversed(before)), start, accesses, before=True)
        children[first : position + 1] = (
            [n for n in before if id(n) in slice_] + [start] + [n for n in before if id(n) not in slice_]
        )
    # Each wait moves down behind the nodes that do not depend on it, ahead of its forward slice; the last wait first
    for wait in reversed([child for child in children if is_wait(child)]):
        position = children.index(wait)
        last = _region_end(children, position, accesses)
        after = children[position + 1 : last]
        slice_ = _slice(after, wait, accesses, before=False)
        children[position:last] = (
            [n for n in after if id(n) not in slice_] + [wait] + [n for n in after if id(n) in slice_]
        )

    # Every communication node in a state of its own
    pinned: list[tn.ScheduleTreeNode] = []

    def boundary(node: tn.StateBoundaryNode) -> None:
        if not pinned:
            return
        if isinstance(pinned[-1], tn.StateBoundaryNode):
            if node.due_to_control_flow:  # Keeps the boundary that control flow requires
                pinned[-1] = node
            return
        pinned.append(node)

    for child in children:
        if _is_comm(child) and not _on_gpu_streams(child):
            boundary(tn.StateBoundaryNode())
            pinned.append(child)
            boundary(tn.StateBoundaryNode())
        elif isinstance(child, tn.StateBoundaryNode):
            boundary(child)
        else:
            pinned.append(child)
    if pinned and isinstance(pinned[-1], tn.StateBoundaryNode) and not pinned[-1].due_to_control_flow:
        pinned.pop()
    for child in pinned:
        child.parent = scope
    scope.children = pinned


def _region_start(children: list[tn.ScheduleTreeNode], position: int, accesses) -> int:
    """The first position after the last barrier before ``position``."""
    first = position
    while first > 0 and not accesses[id(children[first - 1])].barrier:
        first -= 1
    return first


def _region_end(children: list[tn.ScheduleTreeNode], position: int, accesses) -> int:
    """The position of the first barrier after ``position`` (or the end)."""
    last = position + 1
    while last < len(children) and not accesses[id(children[last])].barrier:
        last += 1
    return last


def _slice(candidates: list[tn.ScheduleTreeNode], node: tn.ScheduleTreeNode, accesses, before: bool) -> set[int]:
    """
    The ids of the ``candidates`` that must stay on their side of ``node``: those that depend on it (or it on them),
    directly or through other such candidates. ``candidates`` are the nodes before ``node`` from the nearest one
    (``before``), or those after it from the nearest one. Communication keeps its order: starts and blocking
    operations stay before a start, and waits after a wait.
    """
    slice_: list[tn.ScheduleTreeNode] = [node]
    for candidate in candidates:
        if before:
            keep = accesses[id(candidate)].starts or any(
                not _can_exchange(candidate, other, accesses) for other in slice_
            )
        else:
            keep = is_wait(candidate) or any(not _can_exchange(other, candidate, accesses) for other in slice_)
        if keep:
            slice_.append(candidate)
    return {id(n) for n in slice_[1:]}


def _wait_for_buffers_in_flight(children: list[tn.ScheduleTreeNode], accesses, root, aliases) -> None:
    """A wait accesses the buffers of the starts whose requests it completes: no node may write their inputs, or
    access their outputs, before the wait."""
    in_flight: dict[str, list[_Access]] = {}
    for child in children:
        if is_start(child):
            for connector, memlet in _items(child.out_memlets):
                if connector == "_request":
                    # Every start into the container: a wait may complete any of its elements
                    in_flight.setdefault(_resolve(memlet.data, root, aliases), []).append(accesses[id(child)])
        elif is_wait(child):
            access = accesses[id(child)]
            for memlet in child.input_memlets(root):
                for start in in_flight.get(_resolve(memlet.data, root, aliases), []):
                    access.reads |= start.reads
                    access.writes |= start.writes


def _items(memlets) -> list[tuple[str | None, object]]:
    return list(memlets.items()) if isinstance(memlets, dict) else [(None, m) for m in memlets]


def _can_exchange(first: tn.ScheduleTreeNode, second: tn.ScheduleTreeNode, accesses) -> bool:
    """Whether ``first`` and ``second`` (in this order) can exchange places: they do not depend on each other."""
    a, b = accesses[id(first)], accesses[id(second)]
    if a.barrier or b.barrier:
        return False
    if a.defines & (b.names | b.symbols) or b.defines & (a.names | a.symbols):
        return False
    return not (_overlap(a.writes, b.reads | b.writes) or _overlap(b.writes, a.reads))


def _overlap(first: set[str], second: set[str]) -> bool:
    if not first or not second:
        return False
    return _ANY in first or _ANY in second or bool(first & second)


def _access(node: tn.ScheduleTreeNode, root: tn.ScheduleTreeRoot, aliases: dict[str, str]) -> _Access:
    barrier = isinstance(
        node, (tn.GotoNode, tn.BreakNode, tn.ContinueNode, tn.RefSetNode, tn.DynScopeCopyNode, tn.StateLabel)
    )
    starts = any(_is_comm(n) and not is_wait(n) for n in node.preorder_traversal())
    if isinstance(node, tn.ViewNode):  # Defines the view; reads and writes go through its name
        return _Access(set(), set(), {node.source}, _symbols(node.memlet.free_symbols), {node.target}, False, False)
    if isinstance(node, tn.AssignNode):  # Defines a symbol, from symbols and containers
        inputs = node.input_memlets(root)
        reads = {_resolve(m.data, root, aliases) for m in inputs if m.data is not None}
        symbols = _symbols(node.edge.free_symbols) - {node.name}
        return _Access(reads, set(), {m.data for m in inputs if m.data is not None}, symbols, {node.name}, False, False)
    try:
        inputs = node.input_memlets(root)
        outputs = node.output_memlets(root)
    except NotImplementedError:
        return _Access(set(), set(), set(), set(), set(), starts, True)
    symbols = set()
    for memlet in list(inputs) + list(outputs):
        symbols |= _symbols(memlet.free_symbols)
    if isinstance(node, tn.LibraryCall):
        symbols |= _symbols(node.node.free_symbols)
    names = {m.data for m in list(inputs) + list(outputs) if m.data is not None}
    return _Access(
        reads={_resolve(m.data, root, aliases) for m in inputs if m.data is not None},
        writes={_resolve(m.data, root, aliases) for m in outputs if m.data is not None},
        names=names,
        symbols=symbols,
        defines=set(),
        starts=starts,
        barrier=barrier,
    )


def _symbols(symbols) -> set[str]:
    return {str(s) for s in symbols if isinstance(s, (str, sympy.Symbol))}


def _resolve(name: str, root: tn.ScheduleTreeRoot, aliases: dict[str, str]) -> str:
    """The container whose memory ``name`` accesses: a view resolves to the container it views."""
    seen = set()
    while name in aliases and name not in seen:
        seen.add(name)
        name = aliases[name]
    desc = root.containers.get(name)
    if isinstance(desc, (data.View, data.Reference)) and name not in aliases:
        return _ANY
    return name
