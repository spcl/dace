# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Converting data-dependent branches in innermost loops to selects, and computing the arms of selects first."""
import ast
import copy
from typing import Dict, Optional

import sympy

from dace import data, dtypes
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.common import (AccessIndex, clone_subtree, condition_of, is_pure,
                                                            range_analysis)
from dace.sdfg.analysis.schedule_tree.passes.folding import (RangeFacts, facts_at)
from dace.sdfg.analysis.schedule_tree.passes.rerolling import (roll_key)

# Operators that may trap or have undefined behavior when evaluated speculatively
_UNSAFE_OPERATORS = (ast.FloorDiv, ast.Mod)


def _speculatable(statement: tn.TaskletNode) -> bool:
    """Whether a tasklet may run where it did not before: Python code without side effects, integer division or
    modulo, and calls only to pure functions."""
    node = statement.node
    if node.language != dtypes.Language.Python or getattr(node, 'side_effects', False):
        return False
    for n in ast.walk(ast.Module(body=list(node.code.code), type_ignores=[])):
        if isinstance(n, ast.BinOp) and isinstance(n.op, _UNSAFE_OPERATORS):
            return False
        if isinstance(n, ast.Call) and not (isinstance(n.func, (ast.Name, ast.Attribute))):
            return False
    return True


def _cost(statements: list) -> int:
    """A rough operation count of straight-line tasklets."""
    return sum(
        sum(
            isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.BoolOp, ast.IfExp))
            for n in ast.walk(ast.Module(body=list(s.node.code.code), type_ignores=[]))) + 1 for s in statements)


def _within_bounds(memlet: Memlet, desc: data.Data, facts: RangeFacts) -> bool:
    """Whether the single element ``memlet`` accesses lies within ``desc`` for every value the enclosing loops give
    their variables (``facts``); indices must be affine in the loop variables."""
    lrr = range_analysis()
    if isinstance(desc, data.Scalar):
        return True
    for dim, index in enumerate(memlet.subset.min_element()):
        expr = sympy.expand(sympy.sympify(index))
        lo = hi = expr
        for sym in expr.free_symbols:
            if str(sym) not in facts.known:
                continue  # Invariant symbol: compared symbolically with the shape
            coefficient = expr.coeff(sym)
            _, interval = facts.known[str(sym)]
            if not coefficient.is_number or interval.lo is None or interval.hi is None:
                return False
            lo = lo.subs(sym, interval.lo if coefficient > 0 else interval.hi)
            hi = hi.subs(sym, interval.hi if coefficient > 0 else interval.lo)
        if any(str(s) in facts.known for s in (lo.free_symbols | hi.free_symbols)):
            return False  # Not affine in the loop variables
        if not (lrr.provably_ge(lo, 0) and lrr.provably_ge(desc.shape[dim] - 1, hi)):
            return False
    return True


def _element(memlet: Memlet) -> tuple:
    return memlet.data, str(memlet.subset)


def convert_diamonds_to_selects(stree: tn.ScheduleTreeScope,
                                max_cost_ratio: float = 3.0,
                                max_enumeration: int = 1 << 20) -> int:
    """
    Replace if/else pairs in innermost loops, whose branches compute the same outputs, with straight-line code that
    computes both branches and selects the results (if-conversion of diamonds), so the loop can be vectorized.

    ``if c: X[i] = f(...) else: X[i] = g(...)`` becomes ``t = c; a = f(...); b = g(...); X[i] = a if t else b``. The
    condition is evaluated first, each branch runs into fresh temporaries (its reads of elements it wrote earlier
    read those), and each element written by both branches is selected into place. A pair is converted if
    * it is directly in the body of a loop or map that contains no other loop or map (innermost);
    * its condition is pure, and both branches are straight-line single-element Python tasklets that are safe to run
      speculatively: no side effects, integer division or modulo, and every element they read provably exists for
      all iterations of the enclosing loops (the condition is not assumed);
    * every element one branch writes that may be read later (it is not a transient read only within its branch
      after being written) is written by the other branch too (triangles are not converted);
    * the branches cost within ``max_cost_ratio`` of each other, so a cheap branch is not burdened with an expensive
      one.
    Run after splitting and unswitching, which remove the conditions that ranges decide or that do not change within
    a loop; what remains in hot loops is data-dependent.

    :param stree: The schedule tree (or subtree) to transform in place.
    :param max_cost_ratio: Do not convert if one branch costs more than this many times the other.
    :param max_enumeration: Upper bound on the iterates evaluated for range facts over compile-time constant data.
    :return: The number of pairs converted.
    """
    root = stree.get_root()
    containers = root.containers
    index = AccessIndex(root)

    def fresh(base: str, desc: data.Data) -> str:
        name = data.find_new_name(f'__sel_{base}', containers)
        containers[name] = data.Scalar(desc.dtype, transient=True)
        return name

    def reads_inside(scope: tn.ScheduleTreeScope) -> Dict[str, int]:
        reads: Dict[str, int] = {}
        index._count(scope, reads, {}, set())
        return reads

    def convert(first: tn.IfScope, second: tn.ElseScope, facts: RangeFacts) -> Optional[list]:
        condition = condition_of(first)
        if condition is None or not is_pure(condition):
            return None
        branches = [list(first.children), list(second.children)]
        if not all(
                isinstance(s, tn.TaskletNode) and roll_key(s) is not None and _speculatable(s) for body in branches
                for s in body):
            return None
        costs = [_cost(body) for body in branches]
        if max(costs) > max_cost_ratio * max(min(costs), 1):
            return None
        # Every read must be safe to perform unconditionally
        for body in branches:
            for statement in body:
                for memlet in statement.in_memlets.values():
                    if not _within_bounds(memlet, containers[memlet.data], facts):
                        return None
        # Elements written, and containers that must be selected (live after the pair or read before written)
        written = [[_element(m) for s in body for m in s.out_memlets.values()] for body in branches]
        chain_written = {e[0] for w in written for e in w}
        inside = [reads_inside(first), reads_inside(second)]
        live = set()
        for body in branches:
            seen = set()
            for statement in body:
                for memlet in statement.in_memlets.values():
                    if memlet.data in chain_written and _element(memlet) not in seen:
                        live.add(memlet.data)  # Reads a value from before the pair (possibly of another branch)
                for memlet in statement.out_memlets.values():
                    seen.add(_element(memlet))
        for name in chain_written:
            desc = containers[name]
            outside = index.reads.get(name, 0) > inside[0].get(name, 0) + inside[1].get(name, 0)
            if not desc.transient or outside:
                live.add(name)
        selected = [e for e in dict.fromkeys(written[0]) if e[0] in live]
        if sorted(selected) != sorted(e for e in dict.fromkeys(written[1]) if e[0] in live):
            return None  # Some live element is written by one branch only

        from dace.frontend.python import astutils  # Avoid import loops
        result = []
        # The condition, evaluated first
        inputs = {}

        class Accesses(ast.NodeTransformer):

            def visit_Subscript(self, node: ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id in containers:
                    connector = f'__in{len(inputs)}'
                    indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                    inputs[connector] = Memlet(f'{node.value.id}[{", ".join(ast.unparse(i) for i in indices)}]')
                    return ast.copy_location(ast.Name(id=connector, ctx=ast.Load()), node)
                return self.generic_visit(node)

            def visit_Name(self, node: ast.Name):
                if node.id in containers:
                    if not isinstance(containers[node.id], data.Scalar):
                        raise ValueError('whole array in condition')
                    connector = f'__in{len(inputs)}'
                    inputs[connector] = Memlet(f'{node.id}[0]')
                    return ast.copy_location(ast.Name(id=connector, ctx=ast.Load()), node)
                return node

        try:
            expression = Accesses().visit(astutils.copy_tree(condition))
        except ValueError:
            return None
        predicate = fresh('cond', data.Scalar(dtypes.bool_))
        tasklet = nodes.Tasklet('select_condition', {c: None
                                                     for c in inputs}, {'__out': None},
                                f'__out = {ast.unparse(expression)}')
        result.append(tn.TaskletNode(node=tasklet, in_memlets=inputs, out_memlets={'__out': Memlet(f'{predicate}[0]')}))
        # Each branch into temporaries
        temporaries = []
        for body in branches:
            renamed: Dict[tuple, str] = {}
            for statement in body:
                clone = clone_subtree(statement)
                for connector, memlet in clone.in_memlets.items():
                    if _element(memlet) in renamed:
                        clone.in_memlets[connector] = Memlet(f'{renamed[_element(memlet)]}[0]')
                for connector, memlet in clone.out_memlets.items():
                    element = _element(memlet)
                    if element not in renamed:
                        renamed[element] = fresh(memlet.data, containers[memlet.data])
                    clone.out_memlets[connector] = Memlet(f'{renamed[element]}[0]')
                result.append(clone)
            temporaries.append(renamed)
        # Select the live elements into place
        for element in dict.fromkeys(selected):
            data_name, subset = element
            tasklet = nodes.Tasklet('select', {
                '__c': None,
                '__a': None,
                '__b': None
            }, {'__out': None}, '__out = __a if __c else __b')
            result.append(
                tn.TaskletNode(node=tasklet,
                               in_memlets={
                                   '__c': Memlet(f'{predicate}[0]'),
                                   '__a': Memlet(f'{temporaries[0][element]}[0]'),
                                   '__b': Memlet(f'{temporaries[1][element]}[0]')
                               },
                               out_memlets={'__out': Memlet(f'{data_name}[{subset}]')}))
        return result

    converted = 0

    def visit(scope: tn.ScheduleTreeScope, facts: RangeFacts):
        nonlocal converted
        innermost = isinstance(scope, (tn.ForScope, tn.MapScope)) and not any(
            isinstance(n, (tn.ForScope, tn.MapScope)) for n in scope.preorder_traversal() if n is not scope)
        if innermost:
            children, result, k, changed = scope.children, [], 0, False
            while k < len(children):
                node = children[k]
                following = children[k + 1] if k + 1 < len(children) else None
                after = children[k + 2] if k + 2 < len(children) else None
                if (isinstance(node, tn.IfScope) and not isinstance(node, tn.StateIfScope)
                        and isinstance(following, tn.ElseScope) and not isinstance(after, tn.ElifScope)):
                    replacement = convert(node, following, facts)
                    if replacement is not None:
                        result += replacement
                        converted += 1
                        changed = True
                        k += 2
                        continue
                result.append(node)
                k += 1
            if changed:
                scope.children = []
                scope.add_children(result)
            return
        previous = None
        for child in scope.children:
            if isinstance(child, tn.ScheduleTreeScope):
                visit(child, facts.within(child, previous))
            previous = child

    visit(stree, facts_at(stree, max_enumeration))
    return converted


def _operations(expression: ast.expr) -> int:
    return sum(
        isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.BoolOp, ast.IfExp))
        for n in ast.walk(expression))


def _speculatable_expression(expression: ast.expr) -> bool:
    """Whether evaluating ``expression`` unconditionally is safe: no integer division or modulo, no subscripts (which
    may index out of bounds where the condition does not hold), no conversions of computed values to integers (which
    are undefined for values out of range), and calls only to named functions."""
    for n in ast.walk(expression):
        if isinstance(n, ast.BinOp) and isinstance(n.op, _UNSAFE_OPERATORS):
            return False
        if isinstance(n, ast.Subscript):
            return False
        if isinstance(n, ast.Call):
            if not isinstance(n.func, (ast.Name, ast.Attribute)):
                return False
            if (isinstance(n.func, ast.Name) and n.func.id in ('int', 'round')
                    and not all(isinstance(a, ast.Constant) for a in n.args)):
                return False
    return True


def hoist_select_arms(stree: tn.ScheduleTreeScope, min_operations: int = 1) -> int:
    """
    Compute the arms of conditional expressions in tasklets before selecting between them.

    ``x = (a / b) if c else (a / d)`` becomes ``__arm0 = a / b; __arm1 = a / d; x = __arm0 if c else __arm1``. Both
    arms are then computed unconditionally, and the conditional expression only selects between two values. This
    keeps compilers from sinking the common operations of the arms below the selection, which some (e.g., LLVM) turn
    into gathers of the selected operands in vectorized loops. Arms are hoisted in place, within the tasklet, and are
    declared with the type of the expression (``auto``), so the results do not change.

    This helps vectorized loops, where both arms are computed for all lanes anyway, and may cost time in scalar code
    where one arm is expensive and rarely taken; it is therefore not part of the default pipelines. Only arms that are
    safe to evaluate where their condition does not hold are hoisted (see :func:`_speculatable_expression`).

    :param stree: The schedule tree (or subtree) to transform in place.
    :param min_operations: Only hoist arms with at least this many operations (names and constants are never hoisted).
    :return: The number of arms hoisted.
    """
    hoisted = 0
    for node in stree.preorder_traversal():
        if not isinstance(node, tn.TaskletNode) or node.node.code.language != dtypes.Language.Python:
            continue
        if getattr(node.node, 'side_effects', False):
            continue
        body = list(node.node.code.code)
        taken = {n.id for s in body for n in ast.walk(s) if isinstance(n, ast.Name)}
        taken |= set(node.node.in_connectors) | set(node.node.out_connectors)
        counter = [0]

        def fresh() -> str:
            while f'__arm{counter[0]}' in taken:
                counter[0] += 1
            name = f'__arm{counter[0]}'
            taken.add(name)
            return name

        class Hoist(ast.NodeTransformer):
            """Replaces nontrivial arms of conditional expressions with fresh names, collecting their definitions
            (innermost first, so an arm may use the arms of the conditional expressions nested in it)."""

            def __init__(self):
                self.definitions = []

            def visit_Lambda(self, node: ast.Lambda):
                return node  # Arms may refer to the arguments

            def visit_IfExp(self, node: ast.IfExp):
                self.generic_visit(node)
                for field in ('body', 'orelse'):
                    arm = getattr(node, field)
                    if _operations(arm) < max(min_operations, 1) or not _speculatable_expression(arm):
                        continue
                    name = fresh()
                    self.definitions.append(
                        ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=arm, lineno=0, col_offset=0))
                    setattr(node, field, ast.copy_location(ast.Name(id=name, ctx=ast.Load()), arm))
                return node

        result, changed = [], False
        for statement in body:
            if not isinstance(statement, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr)):
                result.append(statement)  # Control flow in tasklets: leave its conditional parts alone
                continue
            hoist = Hoist()
            statement = hoist.visit(copy.deepcopy(statement))
            result += hoist.definitions + [statement]
            hoisted += len(hoist.definitions)
            changed |= bool(hoist.definitions)
        if changed:
            module = ast.fix_missing_locations(ast.Module(body=result, type_ignores=[]))
            node.node.code = CodeBlock(module.body, dtypes.Language.Python)
    return hoisted
