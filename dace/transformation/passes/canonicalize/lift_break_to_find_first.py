# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift an early-exit loop to a parallel find-first, the body below the hit and the exit effect once.

Pattern (TSVC ``s332`` / ``s481`` / ``s482``, llr40 ``ext_break_capture``)::

    for i in range(b, e):
        pre(i)                     # states and symbol assignments, before the test
        if p(i):
            effect(i)
            break
        post(i)

becomes, when ``pre`` or ``post`` does work, a sequential loop over chunks of the range::

    for c in range(b, e, C):
        first = FindFirst(p, c, min(c + C, e))   # parallel, cancelling: smallest hit in the chunk
        for i in range(c, first):               # an ordinary loop; LoopToMap makes it a map when it is one
            pre(i); post(i)
        if first < min(c + C, e):
            pre(first); effect(first)
            break

and, when the loop body is the test alone, one ``FindFirst`` over the whole range and the effect after it.
Chunking is what makes the body's second read of the predicate's arrays a cache hit and bounds the
search past the hit to one chunk; :data:`FIND_FIRST_HINT` records the alternatives on the node.

Sound when ``p`` reads nothing the loop writes: then every test sees the values the sequential loop
saw, all tests can run before any body, and the iterations below the hit run exactly ``pre; post``.
``effect`` runs once, after them, exactly where the sequential loop ran it. Refused when the loop
counter or a symbol the body assigns is read after the loop (the sequential loop leaks its last
value), when the test reads data the loop writes, or when the body holds another jump.

:class:`~dace.libraries.standard.nodes.find_first.FindFirst` is chunked and cancelling on both CPU and
GPU, so an early hit costs about one chunk instead of the whole range.
"""

import ast
import copy
from typing import Any

from dace import SDFG, InterstateEdge, data, dtypes, properties, subsets, symbolic
from dace.libraries.standard.nodes.find_first import INDEX_NAME, OUTPUT_CONNECTOR_NAME, FindFirst
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis.loop_analysis import (
    get_init_assignment,
    get_loop_end,
    get_loop_stride,
    loop_jumps,
    symbol_use_sites,
    symbol_used_outside,
)

#: Recorded on every lifted node: the alternative the lift did not take.
FIND_FIRST_HINT = (
    "find-first: a chunked, cancelling search -- chunks run in order of index and a hit stops every later chunk, "
    "so an early hit costs about one chunk instead of the whole range.\n"
    "Alternative: one full-range min-reduction over the firing indices, no cancellation.\n"
    "Chunked wins when the hit is early; the full scan wins when the hit is late or absent and the chunk "
    "bookkeeping is pure overhead.\n"
    "Both are correct. Measure before choosing."
)


class Match:
    """One liftable loop: its blocks split around the test, and the predicate in Python."""

    def __init__(self, loop: LoopRegion, chain: list[ControlFlowBlock], test: ConditionalBlock) -> None:
        self.loop = loop
        self.chain = chain
        self.test = test
        position = chain.index(test)
        self.pre = chain[:position]
        self.post = chain[position + 1 :]
        _, self.effect = test.branches[0]
        self.predicate = ""
        self.inputs: dict[str, str] = {}
        #: Per array the test subscripts: each read's per-dimension index, written over ``INDEX_NAME``.
        self.reads: dict[str, list[list[str]]] = {}
        self.symbols: set[str] = set()


def linear_chain(region: ControlFlowRegion) -> list[ControlFlowBlock] | None:
    """``region``'s blocks in order when they form one unconditional chain, else ``None``."""
    blocks = [region.start_block]
    while True:
        out = region.out_edges(blocks[-1])
        if not out:
            break
        if len(out) != 1 or not out[0].data.is_unconditional() or out[0].dst in blocks:
            return None
        blocks.append(out[0].dst)
    return blocks if len(blocks) == region.number_of_nodes() else None


def does_work(block: ControlFlowBlock) -> bool:
    """Whether running ``block`` does anything beyond its edges' symbol assignments."""
    return not isinstance(block, SDFGState) or block.number_of_nodes() > 0


def holds_jump(block: ControlFlowBlock) -> bool:
    """Whether ``block`` holds a break, continue or return that is not owned by a loop inside it."""
    if isinstance(block, (BreakBlock, ContinueBlock, ReturnBlock)):
        return True
    if isinstance(block, ConditionalBlock):
        return any(holds_jump(child) for _, branch in block.branches for child in branch.nodes())
    if isinstance(block, LoopRegion):
        return any(isinstance(child, ReturnBlock) for child in block.all_control_flow_blocks())
    if isinstance(block, ControlFlowRegion):
        return any(holds_jump(child) for child in block.nodes())
    return False


def written_data(blocks: list[ControlFlowBlock]) -> set[str]:
    """Every container a state in ``blocks`` writes, nested SDFGs included (by their outer name)."""
    written: set[str] = set()
    for block in blocks:
        states = (
            [block]
            if isinstance(block, SDFGState)
            else [s for s in block.all_control_flow_blocks() if isinstance(s, SDFGState)]
        )
        for state in states:
            for node in state.data_nodes():
                if state.in_degree(node) > 0:
                    written.add(node.data)
    return written


def searched(
    sdfg: SDFG, name: str, reads: list[list[str]] | None, lo: symbolic.SymbolicType, hi: symbolic.SymbolicType
) -> Memlet:
    """The memlet of a search over ``[lo, hi)`` on ``name``, its volume the span of the affine indices it reads.

    A dimension indexed by anything that is not affine in the index with a constant slope, and a
    container read whole (a scalar), keeps its full extent.
    """
    desc = sdfg.arrays[name]
    if not reads or isinstance(desc, data.Scalar):
        return Memlet.from_array(name, desc)
    ranges = []
    for dim, extent in enumerate(desc.shape):
        bounds: list[tuple[Any, Any]] | None = []
        for read in reads:
            expr = symbolic.pystr_to_symbolic(read[dim])
            # By name: the parsed index symbol need not carry the dtype a fresh one would.
            index = next((s for s in expr.free_symbols if str(s) == INDEX_NAME), None)
            slope = expr.diff(index) if index is not None else 0
            if bounds is None or not getattr(slope, "is_number", True):
                bounds = None
                continue
            first, last = (expr.subs(index, lo), expr.subs(index, hi - 1)) if index is not None else (expr, expr)
            bounds.append((first, last) if slope >= 0 else (last, first))
        if bounds is None:
            ranges.append((0, extent - 1, 1))
        else:
            low = symbolic.simplify(symbolic.sympy.Min(*(b for b, _ in bounds)))
            high = symbolic.simplify(symbolic.sympy.Max(*(e for _, e in bounds)))
            ranges.append((low, high, 1))
    # The subset stays the whole container: the predicate subscripts its connector with the absolute index,
    # so the pointer must be the array's base. What one search actually streams is the memlet's volume.
    memlet = Memlet.from_array(name, desc)
    memlet.volume = subsets.Range(ranges).num_elements()
    return memlet


class PredicateBuilder(ast.NodeTransformer):
    """Rewrites the test over the loop's symbols into a Python predicate over FindFirst's connectors."""

    def __init__(self, sdfg: SDFG, counter: str, bindings: dict[str, ast.expr], match: Match) -> None:
        self.sdfg = sdfg
        self.counter = counter
        self.bindings = bindings
        self.match = match

    def connector(self, name: str) -> str:
        return self.match.inputs.setdefault(name, f"_{name}")

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id in self.bindings:
            return copy.deepcopy(self.bindings[node.id])
        if node.id == self.counter:
            return ast.Name(INDEX_NAME, ast.Load())
        if node.id in self.sdfg.arrays:
            desc = self.sdfg.arrays[node.id]
            if not isinstance(desc, data.Scalar):
                raise ValueError(f"the test reads the array {node.id} whole")
            return ast.Name(self.connector(node.id), ast.Load())
        self.match.symbols.add(node.id)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        if not isinstance(node.value, ast.Name) or node.value.id not in self.sdfg.arrays:
            raise ValueError(f"the test subscripts {ast.unparse(node.value)}, which is not a container")
        desc = self.sdfg.arrays[node.value.id]
        if type(desc) is not data.Array:
            raise ValueError(f"the test reads {node.value.id}, which is not a plain array")
        index = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(index) != len(desc.shape) or any(isinstance(dim, ast.Slice) for dim in index):
            raise ValueError(f"the test reads {node.value.id} with a subscript that is not one element")
        for stride in desc.strides:
            self.match.symbols.update(str(s) for s in symbolic.pystr_to_symbolic(str(stride)).free_symbols)
        dims = [ast.unparse(self.visit(dim)) for dim in index]
        self.match.reads.setdefault(node.value.id, []).append(dims)
        # FindFirst's C++ expansions linearize the subscript by these strides; the symbols reach its scope
        return ast.parse(f"{self.connector(node.value.id)}[{', '.join(dims)}]", mode="eval").body


@properties.make_properties
@transformation.explicit_cf_compatible
class LiftBreakToFindFirst(ppl.Pass):
    """Lift every early-exit loop it can prove to :class:`FindFirst`, a loop below the hit and the effect."""

    CATEGORY: str = "Canonicalization"

    chunk = properties.Property(
        dtype=int,
        default=1 << 20,
        desc="Indices per chunk when a body runs below the hit; 0 searches the whole range in one go.",
    )

    def __init__(self, chunk: int = 1 << 20) -> None:
        super().__init__()
        self.chunk = chunk

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """:returns: how many loops were lifted, or ``None`` if none were."""
        lifted = 0
        for sd in list(sdfg.all_sdfgs_recursive()):
            loops = [r for r in sd.all_control_flow_regions() if isinstance(r, LoopRegion)]
            for loop in loops:
                match = self.match(sd, loop)
                if match is not None:
                    self.lift(sd, match)
                    lifted += 1
        if lifted:
            sdfg.reset_cfg_list()
        return lifted or None

    def match(self, sdfg: SDFG, loop: LoopRegion) -> Match | None:
        """The match for ``loop``, or ``None`` when it is not the pattern or the lift is not sound."""
        jumps = loop_jumps(loop)
        if len(jumps) != 1 or not isinstance(jumps[0], BreakBlock):
            return None
        if get_init_assignment(loop) is None or get_loop_end(loop) is None or get_loop_stride(loop) != 1:
            return None
        chain = linear_chain(loop)
        if chain is None:
            return None
        tests = [b for b in chain if isinstance(b, ConditionalBlock) and holds_jump(b)]
        if len(tests) != 1:
            return None
        test = tests[0]
        if len(test.branches) != 1 or test.branches[0][0] is None:
            return None
        match = Match(loop, chain, test)
        effect_chain = linear_chain(match.effect)
        if effect_chain is None or not isinstance(effect_chain[-1], BreakBlock):
            return None
        if any(holds_jump(b) for b in match.pre + match.post + effect_chain[:-1]):
            return None
        if not self.build_predicate(sdfg, match):
            return None
        if not self.private_to_loop(sdfg, match):
            return None
        return match

    def build_predicate(self, sdfg: SDFG, match: Match) -> bool:
        """Fill ``match.predicate`` from the test and the assignments that feed it; ``False`` if impossible."""
        counter = match.loop.loop_variable
        bindings: dict[str, ast.expr] = {}
        builder = PredicateBuilder(sdfg, counter, bindings, match)
        loop_written = written_data(match.pre + match.post)
        try:
            for block in match.pre:
                for name, rhs in match.loop.out_edges(block)[0].data.assignments.items():
                    bindings[name] = builder.visit(ast.parse(rhs, mode="eval").body)
            condition = ast.parse(match.test.branches[0][0].as_string, mode="eval").body
            predicate = builder.visit(condition)
        except (ValueError, SyntaxError):
            return False
        if set(match.inputs) & loop_written or counter in match.symbols:
            return False
        # A symbol the body assigns but the test reads unbound would be read at its value before the loop.
        body_assigned = {name for e in match.loop.edges() for name in e.data.assignments}
        if match.symbols & body_assigned:
            return False
        match.predicate = ast.unparse(predicate)
        return INDEX_NAME in match.predicate

    def private_to_loop(self, sdfg: SDFG, match: Match) -> bool:
        """Whether the counter and every symbol an iteration assigns stay inside the loop.

        Those leak their last iteration's value, which the lifted form does not reproduce when nothing
        hits. A symbol only the effect assigns may be read after the loop: the effect runs exactly once
        at the hit in both forms, and with no hit both leave it as it was before the loop.
        """
        use_sites, descriptor_symbols = symbol_use_sites(sdfg)
        inside = {id(match.loop)}
        inside.update(id(b) for b in match.loop.all_control_flow_blocks())
        inside.update(id(e) for e in match.loop.all_interstate_edges())
        effect_edges = {id(e) for e in match.effect.all_interstate_edges()}
        names = {match.loop.loop_variable}
        names.update(
            name for e in match.loop.all_interstate_edges() if id(e) not in effect_edges for name in e.data.assignments
        )
        return not any(symbol_used_outside(n, inside, use_sites, descriptor_symbols) for n in names)

    def lift(self, sdfg: SDFG, match: Match) -> None:
        """Rewrite the matched loop in place, chunked when there is a body to clip."""
        loop = match.loop
        parent = loop.parent_graph
        begin = get_init_assignment(loop)
        end = get_loop_end(loop) + 1
        chunked = any(does_work(b) for b in match.pre + match.post) and self.chunk > 0
        if chunked:
            # Chunks in order: search a chunk, run the body below its hit while the chunk is still in cache.
            chunk = sdfg.find_new_symbol(f"{loop.label}_chunk")
            container = LoopRegion(
                f"{loop.label}_chunks",
                condition_expr=f"{chunk} < {symbolic.symstr(end)}",
                loop_var=chunk,
                initialize_expr=f"{chunk} = {symbolic.symstr(begin)}",
                update_expr=f"{chunk} = {chunk} + {self.chunk}",
                sdfg=sdfg,
            )
            parent.add_node(container, is_start_block=parent.start_block is loop)
            for edge in list(parent.in_edges(loop)):
                parent.add_edge(edge.src, container, edge.data)
            for edge in list(parent.out_edges(loop)):
                parent.add_edge(container, edge.dst, edge.data)
            parent.remove_node(loop)
            lo = symbolic.pystr_to_symbolic(chunk)
            hi = symbolic.pystr_to_symbolic(f"min({chunk} + {self.chunk}, {symbolic.symstr(end)})")
            self.build(sdfg, match, container, lo, hi, chunked=True)
        else:
            self.build(sdfg, match, parent, begin, end, chunked=False)

    def build(
        self,
        sdfg: SDFG,
        match: Match,
        container: ControlFlowRegion,
        lo: symbolic.SymbolicType,
        hi: symbolic.SymbolicType,
        chunked: bool,
    ) -> None:
        """Search ``[lo, hi)``, run the body below the hit, then the effect at it, inside ``container``.

        Chunked, ``container`` is the new chunk loop and the effect ends in the original ``break``;
        otherwise the search replaces the loop in ``container`` and the effect simply ends.
        """
        loop = match.loop
        counter = loop.loop_variable
        first_data, _ = sdfg.add_scalar(f"{loop.label}_first", dtypes.int64, transient=True, find_new_name=True)
        first_sym = sdfg.find_new_symbol(f"{loop.label}_hit")
        has_body = any(does_work(b) for b in match.pre + match.post)

        search = container.add_state(
            f"{loop.label}_find_first", is_start_block=chunked or container.start_block is loop
        )
        node = FindFirst(f"{loop.label}_find_first", predicate=match.predicate, begin=lo, end=hi)
        node.symbols = sorted(match.symbols)
        node.specialization_hint = FIND_FIRST_HINT
        search.add_node(node)
        for name, conn in sorted(match.inputs.items()):
            node.add_in_connector(conn)
            search.add_edge(
                search.add_read(name), None, node, conn, searched(sdfg, name, match.reads.get(name), lo, hi)
            )
        search.add_edge(node, OUTPUT_CONNECTOR_NAME, search.add_write(first_data), None, Memlet(f"{first_data}[0]"))

        hit = ConditionalBlock(f"{loop.label}_hit_effect", sdfg=sdfg, parent=container)
        branch = ControlFlowRegion(f"{loop.label}_hit_branch", sdfg=sdfg, parent=hit)
        hit.add_branch(properties.CodeBlock(f"{first_sym} < {symbolic.symstr(hi)}"), branch)
        self.copy_pre_and_effect(sdfg, match, branch, counter, first_sym, keep_break=chunked)
        container.add_node(hit)
        if not chunked:
            for edge in list(container.in_edges(loop)):
                container.add_edge(edge.src, search, edge.data)
                container.remove_edge(edge)
            for edge in list(container.out_edges(loop)):
                container.add_edge(hit, edge.dst, edge.data)
                container.remove_edge(edge)

        bind = InterstateEdge(assignments={first_sym: first_data})
        if has_body:
            # The iterations below the hit: the original loop without its test.
            loop.init_statement = properties.CodeBlock(f"{counter} = {symbolic.symstr(lo)}")
            loop.loop_condition = properties.CodeBlock(f"{counter} < {first_sym}")
            stand_in = loop.add_state(f"{match.test.label}_lifted", is_start_block=loop.start_block is match.test)
            for edge in list(loop.in_edges(match.test)):
                loop.add_edge(edge.src, stand_in, edge.data)
            for edge in list(loop.out_edges(match.test)):
                loop.add_edge(stand_in, edge.dst, edge.data)
            loop.remove_node(match.test)
            if chunked:
                container.add_node(loop)
            container.add_edge(search, loop, bind)
            container.add_edge(loop, hit, InterstateEdge())
        else:
            container.remove_node(loop)
            container.add_edge(search, hit, bind)

    def copy_pre_and_effect(
        self, sdfg: SDFG, match: Match, branch: ControlFlowRegion, counter: str, first_sym: str, keep_break: bool
    ) -> None:
        """Fill ``branch`` with ``pre(first); effect(first)`` (and its ``break`` when kept), the counter renamed."""
        loop = match.loop
        previous = branch.add_state(f"{loop.label}_bind_hit", is_start_block=True)
        pending = InterstateEdge()
        for block in match.pre:
            twin = copy.deepcopy(block)
            branch.add_node(twin)
            branch.add_edge(previous, twin, pending)
            pending = copy.deepcopy(loop.out_edges(block)[0].data)
            previous = twin
        effect = match.effect
        effect_chain = linear_chain(effect)
        for block in effect_chain if keep_break else effect_chain[:-1]:
            twin = copy.deepcopy(block)
            branch.add_node(twin)
            branch.add_edge(previous, twin, pending)
            out = effect.out_edges(block)
            pending = copy.deepcopy(out[0].data) if out else InterstateEdge()
            previous = twin
        if pending.assignments and not keep_break:
            closing = branch.add_state(f"{loop.label}_hit_done")
            branch.add_edge(previous, closing, pending)
        branch.replace_dict({counter: first_sym})
        for block in branch.all_control_flow_blocks():
            block.sdfg = sdfg
        for state in [s for s in branch.all_control_flow_blocks() if isinstance(s, SDFGState)]:
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    node.sdfg.parent = state
                    node.sdfg.parent_sdfg = sdfg
