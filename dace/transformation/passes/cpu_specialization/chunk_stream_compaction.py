# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Chunk a lifted stream compaction on the CPU: count per block, offsets, write per block.

:class:`~dace.transformation.passes.canonicalize.loop_to_stream_compaction.LoopToStreamCompaction` leaves the
device-neutral three-phase form: a map writing ``mask[i]`` (the guard, 0 or 1), an exclusive ``Scan`` of the mask
into ``rank`` and a ``Reduce`` of it into ``total``, and a map scattering every taken iteration to slot
``base + K * rank[i]``. On a GPU that is the right shape. On a CPU the ``int8`` mask and the ``int64`` rank are a
write and two or three reads of memory per element that a chunked compaction never makes: each thread counts the
taken iterations of a contiguous block (the guard alone), one short sequential pass turns the counts into the
blocks' first ranks and the total, and each thread then runs its block in order with a running rank, writing
exactly where the scatter would. Agents wrote this shape by hand at about 1.7x the three-phase form on 72 cores.

This pass rewrites the three-phase form into that shape, reusing the mask map's body and the scatter map's body
as they are: phase 1 is a parallel map over blocks whose body walks the block and sums the mask body's output;
phase 2 is one tasklet over the block counts; phase 3 is a parallel map over blocks whose body walks the block,
runs the mask body to learn whether the iteration is taken, hands the running rank to the scatter body in place of
``rank[i]`` and then advances it. ``mask``, ``rank``, the ``Scan`` and the ``Reduce`` disappear; ``total`` keeps
its one writer. It applies only to the exact three-phase shape the lift emits, one-dimensional and top level.
"""

import copy
import re
from typing import Any, NamedTuple

import dace
from dace import SDFG, SDFGState, data, dtypes, subsets, symbolic
from dace import memlet as mm
from dace.codegen.common import sym2cpp
from dace.libraries.standard.nodes.reduce import Reduce
from dace.libraries.standard.nodes.scan import INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME, Scan, ScanOp
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.cpu_specialization.fuse_scan_chains import expose_symbols

#: Iterations one block covers. A block's guard reads are re-read by phase 3 from cache when the block's inputs fit
#: in L2 at a few arrays of eight bytes.
BLOCK = 16384


class Phase(NamedTuple):
    """A one-dimensional top-level map whose body is a single nested SDFG."""

    entry: nodes.MapEntry
    exit: nodes.MapExit
    body: nodes.NestedSDFG


class Match(NamedTuple):
    mask: Phase
    scatter: Phase
    scan: Scan
    count: Reduce
    mask_node: nodes.AccessNode
    rank_node: nodes.AccessNode
    total_node: nodes.AccessNode
    length: symbolic.SymbolicType


def single_body_map(state: SDFGState, entry: nodes.MapEntry) -> Phase | None:
    """``entry``'s map as a :class:`Phase` when it is one-dimensional, unit-stride from 0, top level and holds
    one nested SDFG and nothing else."""
    if len(entry.map.params) != 1 or state.entry_node(entry) is not None:
        return None
    begin, _, step = entry.map.range[0]
    if begin != 0 or step != 1:
        return None
    inner = [n for n in state.scope_children()[entry] if not isinstance(n, nodes.MapExit)]
    if len(inner) != 1 or not isinstance(inner[0], nodes.NestedSDFG):
        return None
    return Phase(entry, state.exit_node(entry), inner[0])


def inside_map(sdfg: SDFG) -> bool:
    """Whether ``sdfg`` is the body of a map at some level of nesting."""
    while sdfg.parent_nsdfg_node is not None:
        if sdfg.parent.entry_node(sdfg.parent_nsdfg_node) is not None:
            return True
        sdfg = sdfg.parent_sdfg
    return False


def only_access(sdfg: SDFG, node: nodes.AccessNode) -> bool:
    """Whether ``node`` is the one access node of its container anywhere in ``sdfg``."""
    return all(n is node for state in sdfg.states() for n in state.data_nodes() if n.data == node.data)


def match(state: SDFGState, scan: Scan, sdfg: SDFG) -> Match | None:
    """The three-phase compaction around ``scan``, or ``None``."""
    if scan.op is not ScanOp.SUM or not scan.exclusive or scan.chains != 1 or not symbolic.equal_valued(1, scan.stride):
        return None
    inputs = [e for e in state.in_edges(scan) if e.dst_conn == INPUT_CONNECTOR_NAME]
    outputs = [e for e in state.out_edges(scan) if e.src_conn == OUTPUT_CONNECTOR_NAME]
    if len(inputs) != 1 or len(outputs) != 1 or len(state.in_edges(scan)) != 1:
        return None
    mask_node, rank_node = inputs[0].src, outputs[0].dst
    if not isinstance(mask_node, nodes.AccessNode) or not isinstance(rank_node, nodes.AccessNode):
        return None
    for node in (mask_node, rank_node):
        desc = sdfg.arrays[node.data]
        if not desc.transient or len(desc.shape) != 1 or not only_access(sdfg, node):
            return None
    length = sdfg.arrays[mask_node.data].shape[0]
    # The mask: written by one map, read by the scan and one sum Reduce into a scalar.
    writers = state.in_edges(mask_node)
    if len(writers) != 1 or not isinstance(writers[0].src, nodes.MapExit):
        return None
    mask_phase = single_body_map(state, state.entry_node(writers[0].src))
    readers = [e.dst for e in state.out_edges(mask_node)]
    counts = [r for r in readers if isinstance(r, Reduce)]
    if mask_phase is None or len(readers) != 2 or scan not in readers or len(counts) != 1:
        return None
    count = counts[0]
    count_out = state.out_edges(count)
    if len(count_out) != 1 or not isinstance(count_out[0].dst, nodes.AccessNode):
        return None
    total_node = count_out[0].dst
    if sdfg.arrays[total_node.data].total_size != 1 or state.in_degree(total_node) != 1:
        return None
    # The rank: read by one map only.
    rank_readers = state.out_edges(rank_node)
    if len(rank_readers) != 1 or not isinstance(rank_readers[0].dst, nodes.MapEntry):
        return None
    scatter_phase = single_body_map(state, rank_readers[0].dst)
    if scatter_phase is None:
        return None
    for phase in (mask_phase, scatter_phase):
        if not symbolic.equal_valued(phase.entry.map.range[0][1] + 1, length):
            return None
    return Match(mask_phase, scatter_phase, scan, count, mask_node, rank_node, total_node, length)


def body_edges(state: SDFGState, phase: Phase) -> tuple[list, list]:
    """The (connector, memlet) pairs ``phase``'s body reads from and writes to through its map."""
    reads = [(e.dst_conn, e.data) for e in state.in_edges(phase.body)]
    writes = [(e.src_conn, e.data) for e in state.out_edges(phase.body)]
    return reads, writes


def renamed(memlet: mm.Memlet, param: str, loop_var: str) -> mm.Memlet:
    """``memlet`` with the map parameter ``param`` replaced by ``loop_var``."""
    result = copy.deepcopy(memlet)
    replacement = {symbolic.pystr_to_symbolic(param): symbolic.pystr_to_symbolic(loop_var)}
    if result.subset is not None:
        result.subset.replace(replacement)
    if result.other_subset is not None:
        result.other_subset.replace(replacement)
    return result


def clone_body(target: SDFGState, phase: Phase, loop_var: str) -> nodes.NestedSDFG:
    """A copy of ``phase``'s body in ``target`` with the map parameter renamed ``loop_var``."""
    param = phase.entry.map.params[0]
    body = phase.body
    mapping = {
        name: symbolic.pystr_to_symbolic(str(value)).subs(
            {symbolic.pystr_to_symbolic(param): symbolic.pystr_to_symbolic(loop_var)}
        )
        for name, value in body.symbol_mapping.items()
    }
    clone = target.add_nested_sdfg(
        copy.deepcopy(body.sdfg), set(body.in_connectors), set(body.out_connectors), symbol_mapping=mapping
    )
    return clone


def scalarize(clone: nodes.NestedSDFG, connector: str) -> None:
    """Make ``clone``'s container ``connector`` -- one element of a buffer per iteration -- a scalar: its
    descriptor becomes a ``Scalar`` and every memlet on it, at any depth, addresses element 0."""
    inner = clone.sdfg
    old = inner.arrays[connector]
    inner.remove_data(connector, validate=False)
    inner.add_scalar(connector, old.dtype, storage=old.storage, transient=False)
    # An interstate edge reads it by subscript (the scatter's cursor, ``base + K * rank[it]``): the one value now.
    subscript = re.compile(r"\b" + re.escape(connector) + r"\s*\[[^\[\]]*\]")
    for edge in inner.all_interstate_edges():
        for name, value in list(edge.data.assignments.items()):
            edge.data.assignments[name] = subscript.sub(connector, value)
        if subscript.search(edge.data.condition.as_string):
            edge.data.condition = CodeBlock(subscript.sub(connector, edge.data.condition.as_string))
    for sub in inner.all_sdfgs_recursive():
        for state in sub.states():
            for edge in state.edges():
                if edge.data.data == connector and sub is inner:
                    edge.data.subset = subsets.Range([(0, 0, 1)])
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG) and connector in node.in_connectors | node.out_connectors:
                    nested = node.sdfg
                    if connector in nested.arrays and not isinstance(nested.arrays[connector], data.Scalar):
                        scalarize(node, connector)


def in_scope(state: SDFGState, phases: tuple[Phase, ...]) -> dict[str, Any]:
    """Every symbol the phases' bodies can read, with its type: the SDFG's own, those an enclosing loop or interstate
    edge assigns (an outer iterator, a cursor base), minus the phases' own map parameters, which the block loop
    replaces."""
    own = {phase.entry.map.params[0] for phase in phases}
    found: dict[str, Any] = {}
    for phase in phases:
        for name, stype in state.symbols_defined_at(phase.body).items():
            if name not in own and not isinstance(stype, data.Data):
                found.setdefault(name, stype)
    return found


class ChunkStreamCompaction(ppl.Pass):
    """Rewrite each lifted three-phase compaction into the chunked count / offsets / write shape."""

    CATEGORY: str = "CPU Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Nodes)

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        done = 0
        for node, state in list(sdfg.all_nodes_recursive()):
            if isinstance(node, Scan) and isinstance(state, SDFGState) and node in state.nodes():
                if inside_map(state.sdfg):
                    continue  # each iteration of an enclosing parallel map compacts sequentially
                found = match(state, node, state.sdfg)
                if found is not None:
                    self.rewrite(state, state.sdfg, found)
                    done += 1
        return done or None

    def rewrite(self, state: SDFGState, sdfg: SDFG, m: Match) -> None:
        count_dtype = sdfg.arrays[m.total_node.data].dtype
        blocks = symbolic.int_ceil(m.length, BLOCK)
        counts, _ = sdfg.add_array("compaction_counts", [blocks], count_dtype, transient=True, find_new_name=True)
        starts, _ = sdfg.add_array("compaction_starts", [blocks], count_dtype, transient=True, find_new_name=True)
        mask_reads, mask_writes = body_edges(state, m.mask)
        scatter_reads, scatter_writes = body_edges(state, m.scatter)
        mask_out_conn = mask_writes[0][0]
        rank_conn = next(conn for conn, memlet in scatter_reads if memlet.data == m.rank_node.data)

        # Containers each phase reads and writes, by name, with the outer access node that provides them.
        outer_reads = {e.src.data: e.src for e in state.in_edges(m.mask.entry) if isinstance(e.src, nodes.AccessNode)}
        scatter_outer_reads = {
            e.src.data: e.src
            for e in state.in_edges(m.scatter.entry)
            if isinstance(e.src, nodes.AccessNode) and e.src.data != m.rank_node.data
        }
        scatter_outer_writes = {
            e.dst.data: e.dst for e in state.out_edges(m.scatter.exit) if isinstance(e.dst, nodes.AccessNode)
        }

        count_node = self.count_phase(state, sdfg, m, blocks, counts, mask_reads, mask_out_conn, outer_reads)
        starts_node = self.offset_phase(state, sdfg, m, blocks, counts, starts, count_node)
        self.write_phase(
            state,
            sdfg,
            m,
            blocks,
            starts,
            starts_node,
            mask_reads,
            mask_out_conn,
            scatter_reads,
            scatter_writes,
            rank_conn,
            {**outer_reads, **scatter_outer_reads},
            scatter_outer_writes,
        )
        # The three-phase form goes: its maps, the mask and rank buffers, the scan and the reduce.
        for phase in (m.mask, m.scatter):
            state.remove_nodes_from(list(state.scope_subgraph(phase.entry, True, True).nodes()))
        state.remove_nodes_from([m.scan, m.count, m.mask_node, m.rank_node])
        for name in (m.mask_node.data, m.rank_node.data):
            sdfg.remove_data(name, validate=False)
        for node in list(state.data_nodes()):
            if state.degree(node) == 0:
                state.remove_node(node)

    def block_sdfg(self, name: str, sdfg: SDFG, containers: list[str], symbols: dict[str, Any]) -> SDFG:
        """An empty block-body SDFG declaring ``containers`` as non-transient copies of ``sdfg``'s and every outer
        symbol, plus the block index ``block``."""
        inner = SDFG(name)
        for container in containers:
            desc = copy.deepcopy(sdfg.arrays[container])
            desc.transient = False
            inner.add_datadesc(container, desc)
        for symbol, stype in symbols.items():
            inner.add_symbol(symbol, stype)
        inner.add_symbol("block", dace.int64)
        return inner

    def block_loop(self, inner: SDFG, before: dace.SDFGState, length: Any, loop_var: str) -> LoopRegion:
        """The sequential loop over one block's iterations, after ``before``."""
        first = f"block * {BLOCK}"
        last = f"min({sym2cpp(length)}, (block + 1) * {BLOCK})"
        loop = LoopRegion(
            "block_iterations",
            f"{loop_var} < {last}",
            loop_var,
            f"{loop_var} = {first}",
            f"{loop_var} = {loop_var} + 1",
        )
        inner.add_node(loop)
        inner.add_edge(before, loop, dace.InterstateEdge())
        return loop

    def wire_reads(
        self,
        body_state: SDFGState,
        clone: nodes.NestedSDFG,
        reads: list,
        param: str,
        loop_var: str,
        skip: str | None = None,
    ) -> None:
        for conn, memlet in reads:
            if conn == skip:
                continue
            body_state.add_edge(body_state.add_read(memlet.data), None, clone, conn, renamed(memlet, param, loop_var))

    def count_phase(self, state, sdfg, m, blocks, counts, mask_reads, mask_out_conn, outer_reads):
        """Phase 1: per block, the number of taken iterations, into ``counts[block]``."""
        dtype = sdfg.arrays[counts].dtype
        mask_dtype = sdfg.arrays[m.mask_node.data].dtype
        scope = in_scope(state, (m.mask,))
        inner = self.block_sdfg("compaction_count_block", sdfg, sorted(outer_reads) + [counts], scope)
        inner.add_scalar("taken", mask_dtype, transient=True)
        inner.add_scalar("run", dtype, transient=True)
        start = inner.add_state("start", is_start_block=True)
        start.add_edge(
            start.add_tasklet("zero", {}, {"z"}, "z = 0"), "z", start.add_write("run"), None, mm.Memlet("run[0]")
        )
        loop = self.block_loop(inner, start, m.length, "it")
        body = loop.add_state("count", is_start_block=True)
        clone = clone_body(body, m.mask, "it")
        scalarize(clone, mask_out_conn)
        self.wire_reads(body, clone, mask_reads, m.mask.entry.map.params[0], "it")
        taken = body.add_access("taken")
        body.add_edge(clone, mask_out_conn, taken, None, mm.Memlet("taken[0]"))
        add = body.add_tasklet("add", {"r", "t"}, {"o"}, "o = r + t")
        body.add_edge(body.add_read("run"), None, add, "r", mm.Memlet("run[0]"))
        body.add_edge(taken, None, add, "t", mm.Memlet("taken[0]"))
        body.add_edge(add, "o", body.add_write("run"), None, mm.Memlet("run[0]"))
        end = inner.add_state("store")
        inner.add_edge(loop, end, dace.InterstateEdge())
        copy_out = end.add_tasklet("store", {"r"}, {"o"}, "o = r")
        end.add_edge(end.add_read("run"), None, copy_out, "r", mm.Memlet("run[0]"))
        end.add_edge(copy_out, "o", end.add_write(counts), None, mm.Memlet(f"{counts}[block]"))

        entry, exit_node = state.add_map(
            "compaction_count", {"block": f"0:{sym2cpp(blocks)}"}, schedule=dtypes.ScheduleType.CPU_Multicore
        )
        nested = state.add_nested_sdfg(
            inner, set(outer_reads), {counts}, symbol_mapping={s: s for s in scope} | {"block": "block"}
        )
        for name, access in outer_reads.items():
            state.add_memlet_path(
                access, entry, nested, dst_conn=name, memlet=mm.Memlet.from_array(name, sdfg.arrays[name])
            )
        count_node = state.add_access(counts)
        state.add_memlet_path(nested, exit_node, count_node, src_conn=counts, memlet=mm.Memlet(f"{counts}[block]"))
        if not outer_reads:
            state.add_nedge(entry, nested, mm.Memlet())
        return count_node

    def offset_phase(self, state, sdfg, m, blocks, counts, starts, count_node):
        """Phase 2: every block's first rank and the total, in order."""
        # Plain integer arithmetic: tasklet text reaches every dialect as written, and CPF's C has no ``int_ceil``.
        nblocks = f"(({sym2cpp(m.length)}) + {BLOCK - 1}) / {BLOCK}"
        del blocks
        code = (
            f"long long running = 0;\n"
            f"for (long long b = 0; b < {nblocks}; ++b) {{ s[b] = running; running += c[b]; }}\n"
            f"t = running;"
        )
        expose_symbols(state, [sym2cpp(m.length)])
        tasklet = state.add_tasklet(
            "compaction_starts",
            {"c": dtypes.pointer(sdfg.arrays[counts].dtype)},
            {"s": dtypes.pointer(sdfg.arrays[starts].dtype), "t": sdfg.arrays[m.total_node.data].dtype},
            code,
            language=dtypes.Language.CPP,
        )
        state.add_edge(count_node, None, tasklet, "c", mm.Memlet.from_array(counts, sdfg.arrays[counts]))
        starts_node = state.add_access(starts)
        state.add_edge(tasklet, "s", starts_node, None, mm.Memlet.from_array(starts, sdfg.arrays[starts]))
        state.add_edge(tasklet, "t", m.total_node, None, mm.Memlet(f"{m.total_node.data}[0]"))
        return starts_node

    def write_phase(
        self,
        state,
        sdfg,
        m,
        blocks,
        starts,
        starts_node,
        mask_reads,
        mask_out_conn,
        scatter_reads,
        scatter_writes,
        rank_conn,
        outer_reads,
        outer_writes,
    ):
        """Phase 3: per block in order, the scatter body with the running rank for ``rank[i]``."""
        rank_dtype = sdfg.arrays[m.rank_node.data].dtype
        mask_dtype = sdfg.arrays[m.mask_node.data].dtype
        containers = sorted(set(outer_reads) | set(outer_writes) | {starts})
        scope = in_scope(state, (m.mask, m.scatter))
        inner = self.block_sdfg("compaction_write_block", sdfg, containers, scope)
        inner.add_scalar("taken", mask_dtype, transient=True)
        inner.add_scalar("run", rank_dtype, transient=True)
        start = inner.add_state("start", is_start_block=True)
        first = start.add_tasklet("first", {"s"}, {"r"}, "r = s")
        start.add_edge(start.add_read(starts), None, first, "s", mm.Memlet(f"{starts}[block]"))
        start.add_edge(first, "r", start.add_write("run"), None, mm.Memlet("run[0]"))
        loop = self.block_loop(inner, start, m.length, "it")
        body = loop.add_state("write", is_start_block=True)
        guard = clone_body(body, m.mask, "it")
        scalarize(guard, mask_out_conn)
        self.wire_reads(body, guard, mask_reads, m.mask.entry.map.params[0], "it")
        taken = body.add_access("taken")
        body.add_edge(guard, mask_out_conn, taken, None, mm.Memlet("taken[0]"))
        scatter = clone_body(body, m.scatter, "it")
        scalarize(scatter, rank_conn)
        self.wire_reads(body, scatter, scatter_reads, m.scatter.entry.map.params[0], "it", skip=rank_conn)
        run_read = body.add_read("run")
        body.add_edge(run_read, None, scatter, rank_conn, mm.Memlet("run[0]"))
        for conn, memlet in scatter_writes:
            body.add_edge(
                scatter, conn, body.add_write(memlet.data), None, renamed(memlet, m.scatter.entry.map.params[0], "it")
            )
        advance_state = loop.add_state("advance")
        loop.add_edge(body, advance_state, dace.InterstateEdge())
        add = advance_state.add_tasklet("advance", {"r", "t"}, {"o"}, "o = r + t")
        advance_state.add_edge(advance_state.add_read("run"), None, add, "r", mm.Memlet("run[0]"))
        advance_state.add_edge(advance_state.add_read("taken"), None, add, "t", mm.Memlet("taken[0]"))
        advance_state.add_edge(add, "o", advance_state.add_write("run"), None, mm.Memlet("run[0]"))

        entry, exit_node = state.add_map(
            "compaction_write", {"block": f"0:{sym2cpp(blocks)}"}, schedule=dtypes.ScheduleType.CPU_Multicore
        )
        nested = state.add_nested_sdfg(
            inner,
            set(outer_reads) | {starts},
            set(outer_writes),
            symbol_mapping={s: s for s in scope} | {"block": "block"},
        )
        for name, access in outer_reads.items():
            state.add_memlet_path(
                access, entry, nested, dst_conn=name, memlet=mm.Memlet.from_array(name, sdfg.arrays[name])
            )
        state.add_memlet_path(
            starts_node, entry, nested, dst_conn=starts, memlet=mm.Memlet.from_array(starts, sdfg.arrays[starts])
        )
        for name, access in outer_writes.items():
            write = mm.Memlet.from_array(name, sdfg.arrays[name])
            write.dynamic = True
            state.add_memlet_path(nested, exit_node, access, src_conn=name, memlet=write)
