# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import copy

import sympy
from dace import sdfg as sd, properties, symbolic
from dace.properties import CodeBlock
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, ConditionalBlock, rehome_claimed_block
from dace.transformation import transformation as xf


def flatten_and(expr: sympy.Basic) -> list[sympy.Basic]:
    """All conjuncts of a (possibly nested) conjunction, flattened. Accepts dace's own ``AND`` as well as
    ``sympy.And``."""
    if isinstance(expr, (sympy.And, symbolic.AND)):
        conjuncts = []
        for arg in expr.args:
            conjuncts.extend(flatten_and(arg))
        return conjuncts
    return [expr]


def simplify_conjunction(cond_str: str) -> str:
    """Minimal equivalent of a fused-branch conjunction string built by the branch cartesian product:
    ``'False'`` if a conjunct and its negation both appear (unsatisfiable cross-term), the de-duplicated
    form if a conjunct repeats (identical fused guards), else ``cond_str`` unchanged."""
    try:
        expr = symbolic.pystr_to_symbolic(cond_str)
    except Exception:  # noqa: BLE001 -- an unparsable guard is simply not simplifiable
        return cond_str
    conjuncts = flatten_and(expr)
    if len(conjuncts) <= 1:
        return cond_str
    if unsatisfiable(expr):
        return 'False'
    uniq = []
    for c in conjuncts:
        if not any(c == u for u in uniq):
            uniq.append(c)
    if len(uniq) == len(conjuncts):
        return cond_str
    # ``symstr``, not ``str``: sympy prints a negation as ``~x``, which parses back as a BITWISE invert
    # (``~True`` is -2, which is true) and hides the negation from the next simplification.
    return ' and '.join(f'({symbolic.symstr(u, cpp_mode=False)})' for u in uniq)


def constant_equality(conjunct: sympy.Basic):
    """``(x, c)`` for a conjunct ``x == c`` with ``c`` a number, else ``None``."""
    if not isinstance(conjunct, sympy.Eq):
        return None
    lhs, rhs = conjunct.args
    if rhs.is_Number and not lhs.is_Number:
        return lhs, rhs
    if lhs.is_Number and not rhs.is_Number:
        return rhs, lhs
    return None


def propositional(expr: sympy.Basic, atoms: dict) -> sympy.Basic:
    """``expr`` over boolean atoms: every relation becomes a variable, ``!=`` / ``<`` / ``<=`` the negation
    of the ``==`` / ``>=`` / ``>`` it negates, and anything else opaque an independent variable."""
    if isinstance(expr, (sympy.And, symbolic.AND)):
        return sympy.And(*(propositional(a, atoms) for a in expr.args))
    if isinstance(expr, (sympy.Or, symbolic.OR)):
        return sympy.Or(*(propositional(a, atoms) for a in expr.args))
    if isinstance(expr, sympy.Not):
        return sympy.Not(propositional(expr.args[0], atoms))
    if expr in (sympy.true, sympy.false):
        return expr
    if isinstance(expr, (sympy.Ne, sympy.Lt, sympy.Le)):
        return sympy.Not(propositional(expr.negated, atoms))
    if expr not in atoms:
        atoms[expr] = sympy.Symbol(f'__atom{len(atoms)}')
    return atoms[expr]


def unsatisfiable(expr: sympy.Basic) -> bool:
    """Whether ``expr`` is provably false: unsatisfiable as a propositional formula over its relations,
    given only that one expression cannot equal two different numbers. Every relation is otherwise a free
    variable, so a satisfiable answer can be wrong but an unsatisfiable one cannot.

    The cartesian branch product of consecutive fusions builds these cross-terms, and one that survives
    unpruned is copied into every later product: over a chain of guards the branch count grows
    geometrically (warpx_field_gather: 9 -> 31 -> 98 -> ... -> 12573).
    """
    atoms: dict = {}
    formula = propositional(expr, atoms)
    pinned: dict = {}
    for relation, variable in atoms.items():
        equality = constant_equality(relation)
        if equality is not None:
            pinned.setdefault(equality[0], []).append(variable)
    exclusive = [
        sympy.Not(sympy.And(a, b)) for variables in pinned.values() for k, a in enumerate(variables)
        for b in variables[k + 1:]
    ]
    return sympy.satisfiable(sympy.And(formula, *exclusive)) is False


@properties.make_properties
@xf.explicit_cf_compatible
class ConditionFusion(xf.MultiStateTransformation):
    """
    Fuses conditional blocks that are either nested or consecutive.
    """

    cblck1 = xf.PatternNode(ConditionalBlock)
    cblck2 = xf.PatternNode(ConditionalBlock)

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(cls.cblck1, cls.cblck2),
            sdutil.node_path_graph(cls.cblck1),
        ]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Case 1: Consecutive conditional blocks
        if expr_index == 0:
            if len(graph.successors(self.cblck1)) != 1:
                return False
            if len(graph.predecessors(self.cblck2)) != 1:
                return False
            if len(graph.edges_between(self.cblck1, self.cblck2)) != 1:
                return False

            edge = graph.edges_between(self.cblck1, self.cblck2)[0]
            if edge.data.condition.as_string != "1":
                return False
            modified_symbols = edge.data.assignments.keys()
            for e in self.cblck1.all_interstate_edges():
                modified_symbols |= e.data.assignments.keys()

            if any([
                    cnd is not None and cnd.get_free_symbols() & modified_symbols != set()
                    for cnd, _ in self.cblck2.branches
            ]):
                return False

            return True

        # Case 2: Nested conditional blocks
        if expr_index == 1:
            if len(graph.predecessors(self.cblck1)) != 0:
                return False
            if len(graph.successors(self.cblck1)) != 0:
                return False

            parent_cfg = self.cblck1.parent_graph
            if parent_cfg is None:
                return False
            parent_cfg = parent_cfg.parent_graph
            if not isinstance(parent_cfg, ConditionalBlock):
                return False
            return True

        return False

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        if self.expr_index == 0:
            self.fuse_consecutive_conditions(sdfg, self.cblck1, self.cblck2)
        elif self.expr_index == 1:
            self.fuse_nested_conditions(sdfg, self.cblck1)

    def fuse_consecutive_conditions(self, sdfg: sd.SDFG, cblck1: ConditionalBlock, cblck2: ConditionalBlock):
        """Merge ``cblck2`` into ``cblck1``.

        Two guarded blocks with the same guard (``if c: A`` then ``if c: B``) become ``if c: A; B``,
        and opposite guards (``if c: A`` then ``if not c: B``) become ``if c: A else: B``. Anything
        else falls back to the cartesian product of the two branch sets below.
        """
        if self.merge_matching_guards(cblck1, cblck2):
            return

        # Check if cblck1 has a single sink node for each branch
        assert all([len(cfg.sink_nodes()) == 1 for _, cfg in cblck1.branches])

        # Check if it only has one successor and that successor is a conditional block
        outer_cfg = cblck1.parent_graph
        assert (len(outer_cfg.successors(cblck1)) == 1), "Conditional block has no or multiple successors"
        assert (outer_cfg.successors(cblck1)[0] == cblck2), "Consecutive conditional block is not a successor"

        # Check if cblck2 has a single predecessor
        assert (len(outer_cfg.predecessors(cblck2)) == 1), "Conditional block has no or multiple predecessors"

        # Edge between cblck1 and cblck2 should not have any conditions
        assert (len(outer_cfg.edges_between(cblck1, cblck2)) == 1), "Multiple edges between conditional blocks"

        cblck_edge = outer_cfg.edges_between(cblck1, cblck2)[0]
        assert (cblck_edge.data.condition.as_string == "1"), "Edge between conditional blocks has conditions"

        # Edge between cblck1 and cblck2 may have assignments, but only if none of the conditions in cblck2 depend on them
        assert all([
            cnd is None or cnd.get_free_symbols() & cblck_edge.data.assignments.keys() == set()
            for cnd, _ in cblck2.branches
        ]), "Assignments in edge are used in cblck2"

        # There should be exactly one or no else branches in each conditional block
        cblck1_elses = len([True for cnd, cfg in cblck1.branches if cnd is None])
        cblck2_elses = len([True for cnd, cfg in cblck2.branches if cnd is None])
        assert cblck1_elses <= 1, "Multiple else branches in cblck1"
        assert cblck2_elses <= 1, "Multiple else branches in cblck2"

        # Add an else branch if there is none
        if cblck1_elses == 0:
            cfg = ControlFlowRegion()
            cfg.add_state(is_start_block=True)
            cblck1.add_branch(None, cfg)
        if cblck2_elses == 0:
            cfg = ControlFlowRegion()
            cfg.add_state(is_start_block=True)
            cblck2.add_branch(None, cfg)

        # First any else branch conditons with not(cond_blck condition)
        for cblck in [cblck1, cblck2]:
            cond_string = ""
            for cnd, cfg in cblck.branches:
                if cnd is not None:
                    assert cnd.as_string != "1", "Branch condition is always true"
                    if cond_string == "":
                        cond_string = f"not({cnd.as_string})"
                    else:
                        cond_string = f"{cond_string} and not({cnd.as_string})"

            for i, (cnd, cfg) in enumerate(cblck.branches):
                if cnd is None:
                    cblck.branches[i] = (CodeBlock(cond_string), cfg)

        # Clone each ORIGINAL branch of cblck1 once per further branch of cblck2. Re-reading the grown
        # list doubled it per round instead: the branches past the product kept their bare cblck1
        # conditions, unreachable behind the product, and the next fusion copied them again.
        orig_blck1_branches = len(cblck1.branches)
        originals = list(cblck1.branches)
        for _ in range(len(cblck2.branches) - 1):
            for cnd, cfg in originals:
                cnd2 = copy.deepcopy(cnd)
                cfg2 = copy.deepcopy(cfg)
                cblck1.add_branch(cnd2, cfg2)

        # Add the conditons of cblck2 to cblck1 and copy the cfgs. ``cblck2`` is dropped below, so its
        # blocks move into the last product branch they join instead of being copied once more.
        for i, (cnd, cfg) in enumerate(cblck2.branches):
            for j in range(orig_blck1_branches):
                off = orig_blck1_branches * i + j
                cblck1.branches[off][0].as_string = (f"({cblck1.branches[off][0].as_string}) and ({cnd.as_string})")

                last_use = j == orig_blck1_branches - 1
                old_new_mapping = {}
                for node in cfg.nodes():
                    new_node = node if last_use else copy.deepcopy(node)
                    old_new_mapping[node] = new_node
                    cblck1.branches[off][1].add_node(new_node)

                for node in cfg.nodes():
                    new_node = old_new_mapping[node]
                    if node is cfg.start_block:
                        cblck1.branches[off][1].add_edge(
                            cblck1.branches[off][1].sink_nodes()[0],
                            new_node,
                            copy.deepcopy(cblck_edge.data),
                        )

                    for edge in cfg.in_edges(node):
                        cblck1.branches[off][1].add_edge(
                            old_new_mapping[edge.src],
                            new_node,
                            copy.deepcopy(edge.data),
                        )

        # Remove cblck2
        for e in outer_cfg.out_edges(cblck2):
            outer_cfg.add_edge(cblck1, e.dst, copy.deepcopy(e.data))
        outer_cfg.remove_node(cblck2)

        # Simplify the fused branch conditions: drop a branch whose condition is an unsatisfiable
        # cartesian cross-term, collapse a redundant ``(c) and (c)`` to the minimal predicate.
        for cnd, cfg in list(cblck1.branches):
            if cnd is None or len(cblck1.branches) <= 1:
                continue
            simplified = simplify_conjunction(cnd.as_string)
            if simplified == 'False':
                cblck1.remove_branch(cfg)
            elif simplified != cnd.as_string:
                cnd.as_string = simplified

        # If a branch is empty (single empty state), remove branch (implicit else)
        implicit_else = False
        for _, cfg in cblck1.branches:
            if len(list(cfg.all_nodes_recursive())) == 1:
                # Remove the branch
                cblck1.remove_branch(cfg)
                implicit_else = True
                break

        # Otherwise, make the last branch of cblck1 an else branch
        if not implicit_else:
            cblck1.branches[-1] = (None, cblck1.branches[-1][1])

        # Give each branch a unique label and nested nodes unique names
        for i, (cnd, cfg) in enumerate(cblck1.branches):
            cfg.label = f"{cblck1.label}_{i}"
            for j, node in enumerate(cfg.nodes()):
                node.label = f"{node.label}_{j}"

        # ``add_branch`` and ``add_node`` re-home every block they claim; only the fused block's own
        # subtree changed, so re-home that one instead of walking the whole SDFG.
        rehome_claimed_block(cblck1, cblck1.parent_graph.sdfg)

    def merge_matching_guards(self, cblck1: ConditionalBlock, cblck2: ConditionalBlock) -> bool:
        """Merge two single-guard blocks whose guards are equal or opposite. ``False`` if they are not."""
        if len(cblck1.branches) != 1 or len(cblck2.branches) != 1:
            return False
        condition, body = cblck1.branches[0]
        other_condition, other_body = cblck2.branches[0]
        if condition is None or other_condition is None or not body.sink_nodes():
            return False

        if self.conditions_are_equal(condition, other_condition):
            outer_cfg = cblck1.parent_graph
            self.splice_after(body, other_body, outer_cfg.edges_between(cblck1, cblck2)[0].data)
        elif self.conditions_are_complementary(condition, other_condition):
            cblck1.add_branch(None, other_body)  # ``cblck2`` is dropped below: move its body, do not copy it
        else:
            return False

        outer_cfg = cblck1.parent_graph
        for edge in outer_cfg.out_edges(cblck2):
            outer_cfg.add_edge(cblck1, edge.dst, copy.deepcopy(edge.data))
        outer_cfg.remove_node(cblck2)
        return True

    @staticmethod
    def splice_after(target: ControlFlowRegion, source: ControlFlowRegion, link: sd.InterstateEdge) -> None:
        """Move ``source``'s blocks after ``target``'s sink, keeping ``source``'s edges; ``source`` is spent."""
        sink = target.sink_nodes()[0]
        start = source.start_block
        for node in source.nodes():
            target.add_node(node, ensure_unique_name=True)
        target.add_edge(sink, start, copy.deepcopy(link))
        for node in source.nodes():
            for edge in source.in_edges(node):
                target.add_edge(edge.src, node, copy.deepcopy(edge.data))

    @staticmethod
    def conditions_are_equal(first: CodeBlock, second: CodeBlock) -> bool:
        """Whether two branch guards are the same predicate."""
        try:
            return bool(symbolic.pystr_to_symbolic(first.as_string) == symbolic.pystr_to_symbolic(second.as_string))
        except Exception:  # noqa: BLE001 -- an unparsable guard is simply not mergeable
            return False

    @staticmethod
    def conditions_are_complementary(first: CodeBlock, second: CodeBlock) -> bool:
        """Whether two branch guards are exact opposites, so the second becomes an ``else``."""
        try:
            a = symbolic.pystr_to_symbolic(first.as_string)
            b = symbolic.pystr_to_symbolic(second.as_string)
            return symbolic.simplify(sympy.Equivalent(sympy.Not(a), b)) == sympy.true
        except Exception:  # noqa: BLE001 -- an unparsable guard is simply not mergeable
            return False

    def fuse_nested_conditions(self, sdfg: sd.SDFG, cblck1: ConditionalBlock):
        nbranch = cblck1.parent_graph

        # Check if cblck1 has no predecessors and no successors
        assert len(nbranch.predecessors(cblck1)) == 0
        assert len(nbranch.successors(cblck1)) == 0

        # Check if cblck1 is nested in another conditional block
        assert nbranch is not None
        assert isinstance(nbranch.parent_graph, ConditionalBlock)
        cblckp = nbranch.parent_graph

        # There should be exactly one or no else branches in the parent conditional block
        cblck1_elses = len([True for cnd, cfg in cblck1.branches if cnd is None])
        cblckp_elses = len([True for cnd, cfg in cblckp.branches if cnd is None])
        assert cblck1_elses <= 1, "Multiple else branches in cblck1"
        assert cblckp_elses <= 1, "Multiple else branches in cblckp"

        # Add an else branch if there is none
        if cblck1_elses == 0:
            cfg = ControlFlowRegion()
            cfg.add_state(is_start_block=True)
            cblck1.add_branch(None, cfg)
        if cblckp_elses == 0:
            cfg = ControlFlowRegion()
            cfg.add_state(is_start_block=True)
            cblckp.add_branch(None, cfg)

        # First any else branch conditons with not(cond_blck condition)
        for cblck in [cblck1, cblckp]:
            cond_string = ""
            for cnd, cfg in cblck.branches:
                if cnd is not None:
                    assert cnd.as_string != "1", "Branch condition is always true"
                    if cond_string == "":
                        cond_string = f"not({cnd.as_string})"
                    else:
                        cond_string = f"{cond_string} and not({cnd.as_string})"

            for i, (cnd, cfg) in enumerate(cblck.branches):
                if cnd is None:
                    cblck.branches[i] = (CodeBlock(cond_string), cfg)

        # Find condition of cblck1 in cblckp
        cond = None
        for cnd, cfg in cblckp.branches:
            if cfg == nbranch:
                cond = cnd
                break
        assert cond is not None

        # For each branch of cblck1, add a branch to cblckp. ``cblck1`` leaves with ``nbranch`` below,
        # so its branch bodies move rather than being copied.
        for cnd1, cfg1 in list(cblck1.branches):
            cnd2 = copy.deepcopy(cnd1)
            cnd2.as_string = f"({cond.as_string}) and ({cnd2.as_string})"
            cblckp.add_branch(cnd2, cfg1)

        # Remove original branch from cblckp
        cblckp.remove_branch(nbranch)

        # If a branch is empty (single empty state or two empty states connected by an empty edge), remove branch (implicit else)
        implicit_else = False
        for _, cfg in cblckp.branches:
            if len(list(cfg.all_nodes_recursive())) == 1:
                # Remove the branch
                cblckp.remove_branch(cfg)
                implicit_else = True
                break

        # Otherwise, make the last branch of cblckp an else branch
        if not implicit_else:
            cblckp.branches[-1] = (None, cblckp.branches[-1][1])

        # Give each branch a unique label and nested nodes unique names
        for i, (cnd, cfg) in enumerate(cblckp.branches):
            cfg.label = f"{cblckp.label}_{i}"
            for j, node in enumerate(cfg.nodes()):
                node.label = f"{node.label}_{j}"

        # As in ``fuse_consecutive_conditions``: only the parent block's subtree changed.
        rehome_claimed_block(cblckp, cblckp.parent_graph.sdfg)
