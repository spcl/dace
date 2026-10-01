# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Write-once bindings for the readable CPU generator: a transient that one assignment tasklet writes, and that is read
only after the write inside its scope, is declared at the write as ``const T x = expr;``.
"""
import ast
from collections.abc import Callable

from dace import data as dt
from dace import dtypes
from dace.sdfg import SDFG, SDFGState, nodes
from dace.transformation.passes.promote_constant_transients import symbolic_reads

NeedsCopy = Callable[[nodes.Tasklet, str], bool]

VALUE_STORAGES = (dtypes.StorageType.Register, dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap)


def is_binding_candidate(desc: dt.Data) -> bool:
    """A scope-lifetime host scalar, or one-element stack array, emitted as a value."""
    if type(desc) not in (dt.Scalar, dt.Array):
        return False
    if not desc.transient or desc.lifetime != dtypes.AllocationLifetime.Scope or desc.allow_conflicts:
        return False
    if isinstance(desc, dt.Scalar):
        return desc.storage in VALUE_STORAGES
    return desc.storage == dtypes.StorageType.Register and all(d == 1 for d in desc.shape)


def is_single_assignment(tasklet: nodes.Tasklet) -> bool:
    """The tasklet is one Python assignment, the only statement a binding can fuse with."""
    if tasklet.language != dtypes.Language.Python:
        return False
    stmts = tasklet.code.code
    return len(stmts) == 1 and isinstance(stmts[0], ast.Assign) and len(stmts[0].targets) == 1


def is_fusable_write(state: SDFGState, node: nodes.AccessNode, needs_copy: NeedsCopy) -> bool:
    """``node`` has one plain write from a tasklet that is emitted as one statement without a scope."""
    in_edges = [e for e in state.in_edges(node) if not e.data.is_empty()]
    if node.setzero or len(in_edges) != 1:
        return False
    edge = in_edges[0]
    tasklet = edge.src
    if edge.data.wcr is not None or edge.data.dynamic or not isinstance(tasklet, nodes.Tasklet):
        return False
    if state.out_degree(tasklet) != 1 or not is_single_assignment(tasklet):
        return False
    # A connector that is not inlined puts the tasklet in its own scope, which would trap the binding
    connectors = [e.dst_conn for e in state.in_edges(tasklet) if not e.data.is_empty()] + [edge.src_conn]
    return not any(needs_copy(tasklet, conn) for conn in connectors)


def reads_follow_write(state: SDFGState, write: nodes.AccessNode, reads: list) -> bool:
    """Every read is in the write's state, after it in dataflow, in a scope the write's scope encloses."""
    scope = state.scope_dict()
    after = set(state.bfs_nodes(write))
    for read_state, read in reads:
        if read_state is not state or read not in after:
            return False
        if any(not isinstance(state.memlet_path(e)[-1].dst, (nodes.Tasklet, nodes.AccessNode))
               for e in state.out_edges(read) if not e.data.is_empty()):
            return False  # a nested SDFG or another consumer may take the value by non-const reference
        enclosing = scope[read]
        while enclosing is not scope[write]:
            if enclosing is None:
                return False
            enclosing = scope[enclosing]
    return True


def find_const_bindings(top_sdfg: SDFG, needs_copy: NeedsCopy) -> dict[int, set[str]]:
    """The data names per ``cfg_id`` (nested SDFGs included) that are bound ``const`` at their single write."""
    result: dict[int, set[str]] = {}
    for sdfg in top_sdfg.all_sdfgs_recursive():
        excluded = symbolic_reads(sdfg) | set(sdfg.constants_prop)
        accesses: dict[str, list] = {}
        for state in sdfg.states():
            for node in state.data_nodes():
                accesses.setdefault(node.data, []).append((state, node))
        for name, sites in accesses.items():
            if name in excluded or not is_binding_candidate(sdfg.arrays[name]):
                continue
            writes = [(s, n) for s, n in sites if any(not e.data.is_empty() for e in s.in_edges(n))]
            if len(writes) != 1 or any(e.dst_conn in ('views', 'set') or e.src_conn == 'views' for s, n in sites
                                       for e in s.all_edges(n)):
                continue
            state, write = writes[0]
            reads = [(s, n) for s, n in sites if any(not e.data.is_empty() for e in s.out_edges(n))]
            if reads and is_fusable_write(state, write, needs_copy) and reads_follow_write(state, write, reads):
                result.setdefault(sdfg.cfg_id, set()).add(name)
    return result
