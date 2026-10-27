# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""What the solver may assume in one loop's dependence queries: the symbol facts at the loop and the array facts of
the containers a query reads.

Every z3 query of the parallelization passes (LoopToMap's write and read/write proofs, the wavefront skew's
dependence distances, the anti-dependence offsets) takes a :class:`SolverFacts` and adds its :meth:`axioms` to the
antecedent. Dropping an axiom only weakens a proof, so a relation or a fact that does not hold everywhere in the loop
is left out rather than approximated.
"""

from collections import Counter
from typing import Any

from dace import symbolic
from dace.optionals import required
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.symbolic_facts import RelationKind
from dace.transformation.passes.analysis import array_facts, smt_dependence
from dace.transformation.passes.analysis.smt_dependence import z3

#: The relations a query may assume; a ``<<`` order holds only under a recorded runtime check, a divisibility needs
#: a quotient the encoding does not introduce.
ENCODED_KINDS = (RelationKind.LT, RelationKind.LE, RelationKind.EQ, RelationKind.NE)


def constant_names(terms: list[Any]) -> tuple[set[str], set[str]]:
    """The uninterpreted integer constants and the arrays the z3 ``terms`` name."""
    z = required(z3)
    names: set[str] = set()
    arrays: set[str] = set()
    seen: set[int] = set()
    stack = [t for t in terms if z.is_expr(t)]
    while stack:
        term = stack.pop()
        if term.get_id() in seen:
            continue
        seen.add(term.get_id())
        if z.is_select(term):
            arrays.add(str(term.arg(0)))
        if z.is_const(term) and term.decl().kind() == z.Z3_OP_UNINTERPRETED:
            names.add(str(term))
        stack.extend(term.children())
    return names, arrays


def iteration_bindings(loop: LoopRegion) -> dict[str, Any]:
    """The names ``loop`` assigns exactly once per iteration, with their values: on its own interstate edges or a
    branch's, not inside a nested loop, and assigned nowhere else in the loop."""
    counts: Counter = Counter()
    for edge in loop.all_interstate_edges():
        counts.update(edge.data.assignments.keys())
    bindings: dict[str, Any] = {}
    regions: list[ControlFlowRegion] = [loop]
    while regions:
        region = regions.pop()
        for edge in region.edges():
            for name, value in edge.data.assignments.items():
                if counts[name] == 1:
                    try:
                        bindings[name] = symbolic.pystr_to_symbolic(value)
                    except Exception:
                        continue
        for block in region.nodes():
            if isinstance(block, ConditionalBlock):
                regions.extend(branch for condition, branch in block.branches)
            elif isinstance(block, ControlFlowRegion) and not isinstance(block, LoopRegion):
                regions.append(block)
    return bindings


class SolverFacts:
    """The symbol facts at ``loop`` and its array facts, as z3 axioms over one query's terms.

    ``symbols`` are the facts on entry to ``loop``; a relation naming the loop's iterator or a symbol the body
    reassigns does not hold in every iteration and is not assumed. Array facts are derived on first use for the
    arrays a query reads, and only for those the loop does not write: a fact on entry to the loop holds inside it
    only while nothing in it stores to the array.

    A subset names an array element through a symbol the iteration binds (``row = perm[r]``); :meth:`expand`
    replaces such a symbol by its value, so the query reads the array and its facts apply.
    """

    __slots__ = ("loop", "symbols", "excluded", "written", "known", "bindings", "offered", "used")

    def __init__(self, loop: LoopRegion, symbols: symbolic.Facts, varying: set[str]) -> None:
        self.loop = loop
        self.symbols = symbols
        self.excluded = set(varying) | {loop.loop_variable}
        self.written: set[str] | None = None
        self.known: dict[str, list[array_facts.ArrayFact]] = {}
        self.bindings: dict[str, Any] | None = None
        #: The array facts the current query was offered, and those a proof relied on.
        self.offered: list[array_facts.ArrayFact] = []
        self.used: list[array_facts.ArrayFact] = []

    @property
    def key(self) -> tuple[Any, ...]:
        """What the axioms are a function of, for a verdict cache: the loop and the facts at it."""
        return (self.loop, self.symbols.key())

    def written_data(self) -> set[str]:
        if self.written is None:
            self.written = set(self.loop.read_and_write_sets()[1])
        return self.written

    def expand(self, expr: Any) -> Any:
        """``expr`` with every symbol the iteration binds once replaced by its value, while that value reads only
        arrays the loop does not write; ``expr`` itself when any binding it needs does not qualify."""
        if self.bindings is None:
            self.bindings = iteration_bindings(self.loop)
        if not self.bindings or not {str(s) for s in expr.free_symbols} & self.bindings.keys():
            return expr
        expanded = array_facts.bound_expression(expr, self.bindings, Counter(dict.fromkeys(self.bindings, 1)))
        if expanded is None:
            return expr
        reads = {str(a.args[0]) for a in expanded.atoms(symbolic.Subscript)}
        return expr if reads & self.written_data() else expanded

    def array_facts(self, names: set[str]) -> list[array_facts.ArrayFact]:
        self.written_data()
        found: list[array_facts.ArrayFact] = []
        for name in sorted(names - self.written):
            if name not in self.known:
                self.known[name] = array_facts.facts_at(self.loop.sdfg, self.loop, [name])
            found.extend(self.known[name])
        return found

    def symbol_axioms(self, names: set[str]) -> list[Any]:
        axioms = []
        for relation in self.symbols.about(names).relations:
            if relation.kind not in ENCODED_KINDS:
                continue
            if self.excluded.intersection(symbolic.free_symbol_names([relation.lhs, relation.rhs])):
                continue
            lhs = smt_dependence._sympy_to_z3(relation.lhs, {}, {})
            rhs = smt_dependence._sympy_to_z3(relation.rhs, {}, {})
            if lhs is None or rhs is None or not (required(z3).is_int(lhs) and required(z3).is_int(rhs)):
                continue
            if relation.kind is RelationKind.LT:
                axioms.append(lhs < rhs)
            elif relation.kind is RelationKind.LE:
                axioms.append(lhs <= rhs)
            elif relation.kind is RelationKind.EQ:
                axioms.append(lhs == rhs)
            else:
                axioms.append(lhs != rhs)
        return axioms

    def axioms(self, terms: list[Any]) -> list[Any]:
        """What may be assumed about the symbols and array reads ``terms`` name."""
        names, arrays = constant_names(terms)
        axioms = self.symbol_axioms(names - self.excluded)
        self.offered = self.array_facts(arrays) if arrays else []
        if self.offered:
            axioms += array_facts.fact_axioms(self.offered, {}, terms, {})
        return axioms

    def replay(self, used: tuple) -> None:
        """Note the array facts a cached proof relied on."""
        self.used += [f for f in used if f not in self.used]

    def confirm(self) -> None:
        """Note that the query just asked was proven, under the array facts it was offered."""
        self.used += [f for f in self.offered if f not in self.used]
