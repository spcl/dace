# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift an early-exit loop to a parallel find-first, the body below the hit and the exit effect once; and a loop that
records the first or the last index where a test holds, without an exit, to the same search (from the end for the
last), the record made once at the found index."""

import ast
import copy
from typing import Any

import sympy

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
    is_written,
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
from dace.transformation.passes.canonicalize.annotate_loop_kinds import EARLY_EXIT

#: Recorded on every lifted node: the alternative the lift did not take.
#: What the chunk loop of a lifted early exit adds to its class line: how its iterations run.
CHUNKED_EXIT = (
    "; blocks of {chunk} iterations, in order: each block is searched in parallel for the first exit (find-first), "
    "then its iterations below the exit run in parallel, and no block after the exit runs"
)
FIND_FIRST_HINT = (
    "parallel -- reduction: min over the indices where the test holds (find-first: chunked cancelling search); "
    "alt: one min-reduction (faster on a late or no hit)"
)
#: Recorded on the search of a lifted last-match loop: the find-first runs over the range reversed.
FIND_LAST_HINT = (
    "parallel -- reduction: max over the indices where the test holds (find-first from the end: chunked cancelling "
    "search); alt: one max-reduction (faster on an early or no hit)"
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
        #: A break-free record: the LAST firing index is the one kept (searched from the end).
        self.last = False
        #: A break-free first-match: the test's terms that read what the record assigns, ``"p < 0"``; they hold at
        #: entry or the loop records nothing, and once the record runs they never hold again.
        self.sentinel = ""
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
                if is_written(state, node):
                    written.add(node.data)
    return written


def searched(
    sdfg: SDFG, name: str, reads: list[list[str]] | None, lo: symbolic.SymbolicType, hi: symbolic.SymbolicType
) -> Memlet:
    """The memlet of a search over ``[lo, hi)`` on ``name``, its volume the span of the affine indices it reads."""
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
            if bounds is None or (isinstance(slope, sympy.Basic) and not slope.is_number):
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


def names_in(code: str) -> set[str]:
    """Every bare name ``code`` reads."""
    return {n.id for n in ast.walk(ast.parse(code, mode="eval")) if isinstance(n, ast.Name)}


def split_conjunction(code: str) -> list[str]:
    """The terms of ``code`` read as ``a and b and ...``; ``code`` alone when it is no conjunction."""
    tree = ast.parse(code, mode="eval").body
    if isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.And):
        return [ast.unparse(term) for term in tree.values]
    return [ast.unparse(tree)]


def substituted(code: str, values: dict[str, str]) -> str | None:
    """``code`` with each name of ``values`` replaced by its expression, or ``None`` if one reads data."""
    replacements = {name: ast.parse(rhs, mode="eval").body for name, rhs in values.items()}
    if any(isinstance(n, ast.Subscript) for rhs in replacements.values() for n in ast.walk(rhs)):
        return None

    class Replace(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            return copy.deepcopy(replacements[node.id]) if node.id in replacements else node

    return ast.unparse(Replace().visit(ast.parse(code, mode="eval")))


def never_holds(
    expr: Any, counter: Any, begin: symbolic.SymbolicType, end: symbolic.SymbolicType, facts: symbolic.Facts
) -> bool:
    """Whether ``expr`` is false at every ``counter`` in ``[begin, end]``: a literal false, a conjunction with a term
    that never holds, or a comparison of two sides that differ by an affine function of ``counter`` staying on the
    false side over the whole range."""
    if expr is sympy.false or expr is False:
        return True
    if isinstance(expr, sympy.And):
        return any(never_holds(term, counter, begin, end, facts) for term in expr.args)
    if not isinstance(expr, sympy.core.relational.Relational):
        return False
    difference = sympy.expand(expr.lhs - expr.rhs)
    slope = difference.diff(counter)
    if not slope.is_number:
        return False
    at_begin, at_end = difference.subs(counter, begin), difference.subs(counter, end)

    # An affine difference takes its least value at one end of the range, chosen by its slope
    def at_least(bound: int) -> bool:
        lowest = at_begin if slope >= 0 else at_end
        return symbolic.provably_le(sympy.Integer(bound), lowest, facts)

    def at_most(bound: int) -> bool:
        highest = at_end if slope >= 0 else at_begin
        return symbolic.provably_le(highest, sympy.Integer(bound), facts)

    if isinstance(expr, sympy.StrictLessThan):
        return at_least(0)
    if isinstance(expr, sympy.LessThan):
        return at_least(1)
    if isinstance(expr, sympy.StrictGreaterThan):
        return at_most(0)
    if isinstance(expr, sympy.GreaterThan):
        return at_most(-1)
    if isinstance(expr, sympy.Equality):
        return at_least(1) or at_most(-1)
    return False


class PredicateBuilder(ast.NodeTransformer):
    """Rewrites the test over the loop's symbols into a Python predicate over FindFirst's connectors."""

    def __init__(
        self, sdfg: SDFG, counter: str, bindings: dict[str, ast.expr], match: Match, index: ast.expr | None = None
    ) -> None:
        self.sdfg = sdfg
        self.counter = counter
        self.bindings = bindings
        self.match = match
        #: What the counter becomes: the search index, or for a last-match the index counted from the end.
        self.index = ast.Name(INDEX_NAME, ast.Load()) if index is None else index

    def connector(self, name: str) -> str:
        return self.match.inputs.setdefault(name, f"_{name}")

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id in self.bindings:
            return copy.deepcopy(self.bindings[node.id])
        if node.id == self.counter:
            return copy.deepcopy(self.index)
        if node.id in self.sdfg.arrays:
            desc = self.sdfg.arrays[node.id]
            if not isinstance(desc, data.Scalar):
                raise ValueError(f"the test reads the array {node.id} whole")
            return ast.Name(self.connector(node.id), ast.Load())
        self.match.symbols.add(node.id)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        # The search tests indices past the hit, where a divisor read from data may be zero and trap.
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and not isinstance(node.right, ast.Constant):
            raise ValueError("the test divides by a value that is not a constant")
        return self.generic_visit(node)

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
        # A do-while runs its first body before the test of the range, which the search does not.
        if loop.inverted:
            return None
        if not jumps:
            return self.match_record(sdfg, loop)
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
        if not self.reads_stay_in_bounds(sdfg, match):
            return None
        return match

    def match_record(self, sdfg: SDFG, loop: LoopRegion) -> Match | None:
        """A break-free loop whose one guard records where its test holds (``if p(i): last = i; v = a[i]``): the last
        firing index, or with a sentinel term (``if p(i) and pos < 0: pos = i``) the first one. ``None`` unless the
        record, made once at that index, leaves what all the records in order leave."""
        if get_init_assignment(loop) is None or get_loop_end(loop) is None or get_loop_stride(loop) != 1:
            return None
        chain = linear_chain(loop)
        if chain is None:
            return None
        tests = [b for b in chain if isinstance(b, ConditionalBlock)]
        # Nothing but the test does work: the search replaces every iteration but the found one.
        if len(tests) != 1 or any(does_work(b) for b in chain if b is not tests[0]):
            return None
        test = tests[0]
        if len(test.branches) != 1 or test.branches[0][0] is None:
            return None
        match = Match(loop, chain, test)
        effect_chain = linear_chain(match.effect)
        if effect_chain is None or not all(isinstance(b, SDFGState) for b in effect_chain):
            return None
        assigned = {name for e in match.effect.edges() for name in e.data.assignments}
        counter = loop.loop_variable
        reads_index = any(counter in names_in(rhs) for e in match.effect.edges() for rhs in e.data.assignments.values())
        if not reads_index and not any(counter in state.free_symbols for state in effect_chain):
            return None  # a flag records no index: an any-reduction (``ConstantStoreToLogicalReduction``)
        if not self.record_is_last_wins(sdfg, match, effect_chain, assigned):
            return None
        terms = split_conjunction(match.test.branches[0][0].as_string)
        sentinel = [t for t in terms if names_in(t) & assigned]
        predicate = [t for t in terms if t not in sentinel]
        if not predicate:
            return None
        if sentinel:
            match.sentinel = " and ".join(f"({t})" for t in sentinel)
            if not self.sentinel_closes(sdfg, match, assigned):
                return None
        else:
            match.last = True
        if not self.build_predicate(sdfg, match, " and ".join(f"({t})" for t in predicate)):
            return None
        if match.symbols & assigned or set(match.inputs) & written_data(effect_chain):
            return None  # the test reads what the record writes: the order of the records matters
        if not self.private_to_loop(sdfg, match) or not self.reads_stay_in_bounds(sdfg, match):
            return None
        return match

    def record_is_last_wins(
        self, sdfg: SDFG, match: Match, effect_chain: list[ControlFlowBlock], assigned: set[str]
    ) -> bool:
        """Whether running the record once, at the last index it runs at, leaves what every run in order leaves: it
        reads nothing it assigns or writes, and writes the same elements at every index."""
        counter = match.loop.loop_variable
        moving = {counter, *(name for e in match.loop.all_interstate_edges() for name in e.data.assignments)}
        written = written_data(effect_chain)
        for edge in match.effect.edges():
            for rhs in edge.data.assignments.values():
                if names_in(rhs) & (assigned | written):
                    return False  # ``count = count + 1``: each record builds on the last
        for state in effect_chain:
            if state.free_symbols & assigned:
                return False
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG) or (
                    isinstance(node, nodes.CodeNode) and node.has_side_effects(sdfg)
                ):
                    return False
                if (
                    isinstance(node, nodes.AccessNode)
                    and state.in_degree(node) == 0
                    and state.out_degree(node)
                    and node.data in written
                ):
                    return False  # reads what an earlier record wrote
            for edge in state.edges():
                if edge.data.is_empty() or not isinstance(edge.dst, nodes.AccessNode):
                    continue
                subset = edge.data.get_dst_subset(edge, state)
                if edge.data.wcr is not None or subset is None or {str(f) for f in subset.free_symbols} & moving:
                    return False  # a write that moves with the index keeps every record, not only the last
        return True

    def sentinel_closes(self, sdfg: SDFG, match: Match, assigned: set[str]) -> bool:
        """Whether the sentinel terms read only what the record assigns and what the loop leaves fixed, and the record
        makes them false at every index of the range: then only the first firing index records."""
        loop = match.loop
        moving = {loop.loop_variable, *(n for e in loop.all_interstate_edges() for n in e.data.assignments)} - assigned
        if names_in(match.sentinel) & (moving | set(sdfg.arrays)):
            return False
        values = {name: rhs for e in match.effect.edges() for name, rhs in e.data.assignments.items()}
        closed = substituted(match.sentinel, values)
        if closed is None:
            return False
        try:
            expr = symbolic.pystr_to_symbolic(closed)
        except (TypeError, ValueError, SyntaxError):
            return False
        counter = symbolic.pystr_to_symbolic(loop.loop_variable)
        expr, counter, begin, end = symbolic.equalize_symbols_across(
            expr, counter, get_init_assignment(loop), get_loop_end(loop)
        )
        return never_holds(expr, counter, begin, end, sdfg.facts())

    def reads_stay_in_bounds(self, sdfg: SDFG, match: Match) -> bool:
        """Whether every element the test reads through the counter exists at every index of the range."""
        facts = sdfg.facts()
        for name, reads in match.reads.items():
            for read in reads:
                for dim, extent in zip(read, sdfg.arrays[name].shape):
                    parts = list(ast.walk(ast.parse(dim, mode="eval")))
                    if not any(isinstance(n, ast.Name) and n.id == INDEX_NAME for n in parts):
                        continue
                    if any(isinstance(n, ast.Subscript) for n in parts):
                        return False
                    # One instance per name: the bounds are reparsed, the extent carries the declared symbol
                    expr, low, high, extent = symbolic.equalize_symbols_across(
                        symbolic.pystr_to_symbolic(dim),
                        get_init_assignment(match.loop),
                        get_loop_end(match.loop),
                        extent,
                    )
                    index = next(s for s in expr.free_symbols if str(s) == INDEX_NAME)
                    slope = expr.diff(index)
                    if not slope.is_number:
                        return False
                    first, last = (low, high) if slope >= 0 else (high, low)
                    if not symbolic.provably_le(0, expr.subs(index, first), facts) or not symbolic.provably_le(
                        expr.subs(index, last), extent - 1, facts
                    ):
                        return False
        return True

    def build_predicate(self, sdfg: SDFG, match: Match, test: str | None = None) -> bool:
        """Fill ``match.predicate`` from the test (``test``, else the guard's condition) and the assignments that feed
        it; ``False`` if impossible. A last-match reads the counter as ``begin + end - __i``: the search runs from the
        end."""
        counter = match.loop.loop_variable
        bindings: dict[str, ast.expr] = {}
        index = None
        if match.last:
            bounds = (
                f"({symbolic.symstr(get_init_assignment(match.loop))}) + ({symbolic.symstr(get_loop_end(match.loop))})"
            )
            index = ast.parse(f"({bounds} - {INDEX_NAME})", mode="eval").body
        builder = PredicateBuilder(sdfg, counter, bindings, match, index)
        loop_written = written_data(match.pre + match.post)
        try:
            for block in match.pre:
                for name, rhs in match.loop.out_edges(block)[0].data.assignments.items():
                    bindings[name] = builder.visit(ast.parse(rhs, mode="eval").body)
            condition = ast.parse(match.test.branches[0][0].as_string if test is None else test, mode="eval").body
            predicate = builder.visit(condition)
        except (ValueError, SyntaxError):
            return False
        if set(match.inputs) & loop_written or counter in match.symbols:
            return False
        # A symbol the body assigns but the test reads unbound would be read at its value before the loop.
        body_assigned = {name for e in match.loop.edges() for name in e.data.assignments}
        if match.symbols & body_assigned:
            return False
        if match.last:
            # The reversed index reads the loop bounds
            bounds = (get_init_assignment(match.loop), get_loop_end(match.loop))
            match.symbols.update(str(f) for bound in bounds for f in sympy.sympify(bound).free_symbols)
        match.predicate = ast.unparse(predicate)
        return INDEX_NAME in match.predicate

    def private_to_loop(self, sdfg: SDFG, match: Match) -> bool:
        """Whether the counter and every symbol an iteration assigns stay inside the loop."""
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
            container.specialization_hint = EARLY_EXIT.format(statement="break") + CHUNKED_EXIT.format(chunk=self.chunk)
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
        """Search ``[lo, hi)``, run the body below the hit, then the effect at it, inside ``container``."""
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
        node.specialization_hint = FIND_LAST_HINT if match.last else FIND_FIRST_HINT
        search.add_node(node)
        for name, conn in sorted(match.inputs.items()):
            node.add_in_connector(conn)
            search.add_edge(
                search.add_read(name), None, node, conn, searched(sdfg, name, match.reads.get(name), lo, hi)
            )
        search.add_edge(node, OUTPUT_CONNECTOR_NAME, search.add_write(first_data), None, Memlet(f"{first_data}[0]"))

        hit = ConditionalBlock(f"{loop.label}_hit_effect", sdfg=sdfg, parent=container)
        branch = ControlFlowRegion(f"{loop.label}_hit_branch", sdfg=sdfg, parent=hit)
        found = f"{first_sym} < {symbolic.symstr(hi)}"
        # The sentinel reads the values from before the loop: it held there, or no index records
        hit.add_branch(properties.CodeBlock(f"({found}) and ({match.sentinel})" if match.sentinel else found), branch)
        index = first_sym
        if match.last:
            # The search ran over the reversed range: its answer counts from the end
            index = sdfg.find_new_symbol(f"{loop.label}_last")
        self.copy_pre_and_effect(sdfg, match, branch, counter, first_sym, keep_break=chunked, index=index)
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
        self,
        sdfg: SDFG,
        match: Match,
        branch: ControlFlowRegion,
        counter: str,
        first_sym: str,
        keep_break: bool,
        index: str | None = None,
    ) -> None:
        """Fill ``branch`` with ``pre(i); effect(i)`` (and its ``break`` when kept), the counter renamed to ``index``:
        the search's answer ``first_sym``, or for a last-match the index it stands for, bound first."""
        loop = match.loop
        index = first_sym if index is None else index
        previous = branch.add_state(f"{loop.label}_bind_hit", is_start_block=True)
        pending = InterstateEdge()
        if index != first_sym:
            bounds = f"({symbolic.symstr(get_init_assignment(loop))}) + ({symbolic.symstr(get_loop_end(loop))})"
            pending = InterstateEdge(assignments={index: f"{bounds} - {first_sym}"})
        for block in match.pre:
            twin = copy.deepcopy(block)
            branch.add_node(twin)
            branch.add_edge(previous, twin, pending)
            pending = copy.deepcopy(loop.out_edges(block)[0].data)
            previous = twin
        effect = match.effect
        effect_chain = linear_chain(effect)
        # A record (no break) copies whole
        if not keep_break and isinstance(effect_chain[-1], BreakBlock):
            effect_chain = effect_chain[:-1]
        for block in effect_chain:
            twin = copy.deepcopy(block)
            branch.add_node(twin)
            branch.add_edge(previous, twin, pending)
            out = effect.out_edges(block)
            pending = copy.deepcopy(out[0].data) if out else InterstateEdge()
            previous = twin
        if pending.assignments and not keep_break:
            closing = branch.add_state(f"{loop.label}_hit_done")
            branch.add_edge(previous, closing, pending)
        branch.replace_dict({counter: index})
        for block in branch.all_control_flow_blocks():
            block.sdfg = sdfg
        for state in [s for s in branch.all_control_flow_blocks() if isinstance(s, SDFGState)]:
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    node.sdfg.parent = state
                    node.sdfg.parent_sdfg = sdfg
