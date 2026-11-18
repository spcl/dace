# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fuse an elementwise producer map into the ``Scan`` it feeds: one tiled two-pass scan, no buffer between them.

Canonicalization lifts ``b[i] = b[i-1] + c[i]*d[i] + c[i]*e[i]`` into a map that writes the deltas to a transient
and a ``Scan`` over that transient. On the CPU the buffer costs a full write and a full read: the scan is memory
bound, so it runs at the speed of the extra traffic. This pass replaces the producer map, the buffer and the scan
with one tasklet that runs the scan's tiled two-pass shape (:func:`dace.cpf_lowering.blocked_scan_body`) and
evaluates the producer's expression in both passes: per tile, each thread folds its block of produced values, one
barrier publishes the block totals, and each thread rescans its block from its offset while the block is still in
cache. Measured on s323's scan (32M doubles, four x86 cores): 42 ms with the buffer, 27 ms fused.

The producer must be a one-dimensional map whose body is a chain of single-assignment Python tasklets (through
scalar temporaries) writing one element of the buffer per iteration; every input is a point read of an array the
chain does not write. The buffer must have no reader but the scan and no writer but the map. The scan must be a
unit-stride, single-chain ``+``, ``*``, ``min`` or ``max`` scan outside any parallel scope (one that takes its
parallel expansion), and a ``min`` / ``max`` scan must carry a seed or an identity. Anything else is left as it is.
"""

import ast
from typing import Any, NamedTuple

from dace import SDFG, SDFGState, data, dtypes, symbolic
from dace import memlet as mm
from dace.codegen.common import sym2cpp
from dace.codegen.cppunparse import pyexpr2cpp
from dace.cpf_lowering import blocked_scan_body
from dace.libraries.standard.nodes.scan import (
    INIT_CONNECTOR_NAME,
    INPUT_CONNECTOR_NAME,
    Scan,
    ScanOp,
)
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl

#: The scan operations the fused tasklet supports, by the name ``blocked_scan_body`` takes.
OPERATIONS = {ScanOp.SUM: "sum", ScanOp.PRODUCT: "product", ScanOp.MIN: "min", ScanOp.MAX: "max"}

#: The seed of an unseeded inclusive scan, per operation; ``min`` and ``max`` have none without a type bound.
IDENTITY = {ScanOp.SUM: "0", ScanOp.PRODUCT: "1"}

#: The tasklet's output connector.
OUT = "scan_out"
#: The tasklet's seed connector.
SEED = "scan_seed"
#: The scan position while the producer's expression is built; ``{i}`` once it is C.
POSITION = "scan_position"


def connector(array: str) -> str:
    """The fused tasklet's connector for ``array``: connectors may not reuse an array's name."""
    return f"in_{array}"


class Producer(NamedTuple):
    """A fusable producer: its scope, its expression of the scan position ``{i}`` in C, and the arrays it reads."""

    entry: nodes.MapEntry
    exit: nodes.MapExit
    expression: str
    reads: dict[str, nodes.AccessNode]


def tasklet_expression(tasklet: nodes.Tasklet) -> ast.expr | None:
    """The right-hand side of a single-assignment Python tasklet writing its one output, or ``None``."""
    if tasklet.language is not dtypes.Language.Python or len(tasklet.out_connectors) != 1:
        return None
    body = tasklet.code.code if isinstance(tasklet.code.code, list) else []
    if len(body) != 1 or not isinstance(body[0], ast.Assign) or len(body[0].targets) != 1:
        return None
    target = body[0].targets[0]
    if not isinstance(target, ast.Name) or target.id not in tasklet.out_connectors:
        return None
    return body[0].value


def point_index(memlet: mm.Memlet, desc: data.Array, param: str, position: str) -> str | None:
    """The flat C index of ``memlet``'s single element with the map parameter ``param`` spelled ``position``."""
    if memlet.subset is None or memlet.subset.num_elements() != 1 or memlet.dynamic or memlet.wcr is not None:
        return None
    replace = {symbolic.pystr_to_symbolic(param): symbolic.pystr_to_symbolic(position)}
    flat = sum(
        (symbolic.pystr_to_symbolic(start).subs(replace) * stride)
        for (start, _, _), stride in zip(memlet.subset.ndrange(), desc.strides)
    )
    return sym2cpp(symbolic.simplify(flat))


def compose(
    state: SDFGState, tasklet: nodes.Tasklet, param: str, position: str, entry: nodes.MapEntry, sdfg: SDFG
) -> tuple[str, dict[str, str]] | None:
    """``tasklet``'s value as one Python expression over array element reads, with the map parameter spelled
    ``position``; the arrays it reads, by name. ``None`` when the chain is not a tree of single-assignment tasklets
    over point reads of arrays and scalar temporaries."""
    rhs = tasklet_expression(tasklet)
    if rhs is None:
        return None
    reads: dict[str, str] = {}
    replacement: dict[str, str] = {}
    for edge in state.in_edges(tasklet):
        source, conn = edge.src, edge.dst_conn
        if source is entry:
            name = edge.data.data
            desc = sdfg.arrays[name]
            if not isinstance(desc, data.Array):
                return None
            index = point_index(edge.data, desc, param, position)
            if index is None:
                return None
            replacement[conn] = f"{connector(name)}[{index}]"
            reads[name] = name
        elif isinstance(source, nodes.AccessNode) and sdfg.arrays[source.data].total_size == 1:
            writers = state.in_edges(source)
            if len(writers) != 1 or not isinstance(writers[0].src, nodes.Tasklet) or state.out_degree(source) != 1:
                return None
            inner = compose(state, writers[0].src, param, position, entry, sdfg)
            if inner is None:
                return None
            replacement[conn], more = inner
            reads.update(more)
        else:
            return None

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in replacement:
                return ast.parse(f"({replacement[node.id]})", mode="eval").body
            return node

    expression = ast.unparse(Substitute().visit(ast.parse(ast.unparse(rhs), mode="eval")))
    return expression, reads


def match_producer(state: SDFGState, buffer: nodes.AccessNode, sdfg: SDFG, start: str) -> Producer | None:
    """The producer of ``buffer`` the scan reads from offset ``start``, if it is fusable."""
    desc = sdfg.arrays[buffer.data]
    if not desc.transient or len(desc.shape) != 1 or state.in_degree(buffer) != 1 or state.out_degree(buffer) != 1:
        return None
    for other in sdfg.states():
        for node in other.data_nodes():
            if node.data == buffer.data and node is not buffer:
                return None
    write = state.in_edges(buffer)[0]
    exit_node = write.src
    if not isinstance(exit_node, nodes.MapExit):
        return None
    entry = state.entry_node(exit_node)
    if len(entry.map.params) != 1 or state.out_degree(exit_node) != 1 or state.entry_node(entry) is not None:
        return None
    param = entry.map.params[0]
    if entry.map.range[0][2] != 1:
        return None
    inner = list(state.in_edges_by_connector(exit_node, "IN_" + write.src_conn[len("OUT_") :]))
    if len(inner) != 1 or not isinstance(inner[0].src, nodes.Tasklet):
        return None
    if point_index(inner[0].data, desc, param, "p") is None:
        return None
    # The map writes buffer[p + k] for a constant k; the scan reads buffer[start + {i}] at its position {i}, which
    # the map produces at p = start + {i} - k.
    loop_var = symbolic.pystr_to_symbolic(param)
    offset = symbolic.pystr_to_symbolic(inner[0].data.subset.ndrange()[0][0]) - loop_var
    if loop_var in offset.free_symbols:
        return None
    position = f"{start} + {POSITION} - ({sym2cpp(offset)})"
    composed = compose(state, inner[0].src, param, position, entry, sdfg)
    if composed is None:
        return None
    expression, read_names = composed
    reads: dict[str, nodes.AccessNode] = {}
    for edge in state.in_edges(entry):
        if isinstance(edge.src, nodes.AccessNode) and edge.src.data in read_names:
            reads[edge.src.data] = edge.src
    if set(reads) != set(read_names) or buffer.data in reads:
        return None
    return Producer(entry, exit_node, pyexpr2cpp(expression).replace(POSITION, "{i}"), reads)


def expose_symbols(state: SDFGState, expressions: list[str]) -> None:
    """Make every symbol ``expressions`` name visible in ``state``'s SDFG.

    A C++ tasklet reports no free symbols (its code is not parsed), so a symbol written into its text is passed to
    no nested SDFG on the way down: declare it in each enclosing nested SDFG that lacks it and map it through, up to
    the first SDFG that already has it.
    """
    names: set[str] = set()
    for expression in expressions:
        names |= {str(s) for s in symbolic.pystr_to_symbolic(expression).free_symbols}
    sdfg = state.sdfg
    for name in sorted(names):
        current = sdfg
        while name not in current.symbols and current.parent_nsdfg_node is not None:
            outer = current.parent_sdfg
            stype = outer.symbols.get(name, dtypes.int64)
            current.add_symbol(name, stype)
            current.parent_nsdfg_node.symbol_mapping[name] = name
            current = outer


class FuseScanChains(ppl.Pass):
    """Fuse each fusable producer map into the ``Scan`` it feeds (module docstring)."""

    CATEGORY: str = "CPU Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Nodes | ppl.Modifies.Edges))

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        from dace.transformation.auto.auto_optimize import libnode_is_sequential

        fused = 0
        for node, state in list(sdfg.all_nodes_recursive()):
            if not isinstance(node, Scan) or not isinstance(state, SDFGState):
                continue
            if libnode_is_sequential(node, state, state.sdfg) or state.entry_node(node) is not None:
                continue
            if self.fuse(state, node, state.sdfg):
                fused += 1
        return fused or None

    def fuse(self, state: SDFGState, scan: Scan, sdfg: SDFG) -> bool:
        """Replace ``scan`` and its producer with one tasklet; whether it did."""
        if scan.op not in OPERATIONS or scan.chains != 1 or not symbolic.equal_valued(1, scan.stride):
            return False
        if scan.segments is not None and not symbolic.equal_valued(1, scan.segments):
            return False
        inputs = {e.dst_conn: e for e in state.in_edges(scan)}
        outputs = state.out_edges(scan)
        if INPUT_CONNECTOR_NAME not in inputs or len(outputs) != 1:
            return False
        source = inputs[INPUT_CONNECTOR_NAME]
        if not isinstance(source.src, nodes.AccessNode) or len(source.data.subset) != 1:
            return False
        start, stop, step = source.data.subset.ndrange()[0]
        if step != 1:
            return False
        out_edge = outputs[0]
        out_desc = sdfg.arrays[out_edge.data.data]
        if (
            not isinstance(out_desc, data.Array)
            or out_edge.data.subset.num_elements() != source.data.subset.num_elements()
        ):
            return False
        producer = match_producer(state, source.src, sdfg, sym2cpp(start))
        if producer is None:
            return False
        ctype = out_desc.dtype.ctype
        if INIT_CONNECTOR_NAME in inputs:
            seed = SEED
        elif scan.exclusive and scan.identity is not None:
            seed = f"({ctype})({scan.identity})"
        elif scan.op in IDENTITY:
            seed = f"({ctype}){IDENTITY[scan.op]}"
        else:
            return False

        out_start = sym2cpp(out_edge.data.subset.ndrange()[0][0]) if len(out_edge.data.subset) == 1 else None
        if out_start is None:
            return False
        length = sym2cpp(stop - start + 1)
        kind = "excl" if scan.exclusive else "incl"
        body = blocked_scan_body(
            kind,
            OPERATIONS[scan.op],
            ctype,
            c_dialect=True,
            element=producer.expression,
            store=f"{OUT}[({out_start}) + {{i}}] = acc;",
            # One position reads one element of every array the producer reads: the block holds 512 KB of those.
            element_bytes=str(sum(sdfg.arrays[name].dtype.bytes for name in producer.reads)),
        )
        code = "\n".join(
            [
                "{",
                "    const long lo = 0;",
                f"    const long hi = (long)({length});",
                f"    const {ctype} seed = {seed};",
                "    do {",
                *["    " + line.replace("return;", "break;") for line in body],
                "    } while (0);",
                "}",
            ]
        )
        expose_symbols(state, [length, out_start, producer.expression.replace("{i}", "0").replace("in_", "")])
        in_connectors = {connector(name): dtypes.pointer(sdfg.arrays[name].dtype) for name in producer.reads}
        if INIT_CONNECTOR_NAME in inputs:
            in_connectors[SEED] = sdfg.arrays[inputs[INIT_CONNECTOR_NAME].data.data].dtype
        tasklet = state.add_tasklet(
            scan.label + "_fused",
            set(in_connectors),
            {OUT},
            code,
            language=dtypes.Language.CPP,
        )
        for name, access in producer.reads.items():
            state.add_edge(access, None, tasklet, connector(name), mm.Memlet.from_array(name, sdfg.arrays[name]))
        if INIT_CONNECTOR_NAME in inputs:
            init = inputs[INIT_CONNECTOR_NAME]
            state.add_edge(
                init.src, init.src_conn, tasklet, SEED, mm.Memlet(data=init.data.data, subset=init.data.subset)
            )
        state.add_edge(
            tasklet, OUT, out_edge.dst, out_edge.dst_conn, mm.Memlet.from_array(out_edge.data.data, out_desc)
        )
        buffer = source.src
        scope = state.scope_subgraph(producer.entry, include_entry=True, include_exit=True)
        state.remove_nodes_from(list(scope.nodes()))
        state.remove_node(scan)
        state.remove_node(buffer)
        for access in producer.reads.values():
            if state.degree(access) == 0:
                state.remove_node(access)
        if not any(n.data == buffer.data for s in sdfg.states() for n in s.data_nodes()):
            sdfg.remove_data(buffer.data, validate=False)
        return True
