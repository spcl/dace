# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Passes that bring conditions into a canonical form: pairing complementary guards, substituting the values conditions
read, and reading containers before conditions."""
import ast
from typing import Dict, Optional, Set, Tuple

from dace import data, dtypes, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.common import (AccessIndex, ReplaceAccesses, bound_names, condition_of,
                                                            condition_element_accesses, in_bounds, is_pure, loop_ranges,
                                                            names_in_subtrees, names_written, range_analysis,
                                                            repository_of)


def _access(memlet: Memlet, containers: Dict[str, data.Data]) -> ast.expr:
    """How a condition reads the data a memlet points to: scalars by name, other containers by subscript."""
    if isinstance(containers.get(memlet.data, None), data.Scalar):
        return ast.Name(id=memlet.data, ctx=ast.Load())
    return ast.parse(f'{memlet.data}[{memlet.subset}]', mode='eval').body


def _assignment(node: tn.ScheduleTreeNode, containers: Dict[str, data.Data]) -> Optional[Tuple[str, ast.expr]]:
    """A node as ``(target, value)`` if it assigns a single value that a condition may use in its place: a symbol
    assignment, a single-element copy, or a Python tasklet ``out = <expression of single-element inputs>``."""
    from dace.frontend.python import astutils  # Avoid import loops
    if isinstance(node, tn.AssignNode):
        code = node.value.code[0]
        return node.name, astutils.copy_tree(getattr(code, 'value', code))
    if isinstance(node, tn.CopyNode) and node.memlet.subset is not None and node.memlet.subset.num_elements() == 1:
        return node.target, _access(node.memlet, containers)
    if isinstance(node, tn.TaskletNode) and node.node.language == dtypes.Language.Python:
        code, outputs = node.node.code.code, list(node.out_memlets.items())
        if (len(outputs) == 1 and len(code) == 1 and isinstance(code[0], ast.Assign) and len(code[0].targets) == 1
                and isinstance(code[0].targets[0], ast.Name) and code[0].targets[0].id == outputs[0][0]
                and all(m.subset.num_elements() == 1 for _, m in outputs + list(node.in_memlets.items()))):
            value = astutils.copy_tree(code[0].value)
            value = astutils.ASTFindReplace({
                c: _access(m, containers)
                for c, m in node.in_memlets.items()
            }).visit(value)
            return outputs[0][1].data, value
    return None


# Largest expression (in AST nodes) substituted for a tasklet output
_MAX_SUBSTITUTED_NODES = 200


class _ReplaceNames(ast.NodeTransformer):
    """Replaces names by (copies of) expressions."""

    def __init__(self, values: Dict[str, ast.expr]):
        self.values = values

    def visit_Name(self, node: ast.Name):
        if node.id in self.values:
            from dace.frontend.python import astutils  # Avoid import loops
            return astutils.copy_tree(self.values[node.id])
        return node


def _tasklet_values(node: tn.TaskletNode, containers: Dict[str, data.Data]) -> Dict[str, ast.expr]:
    """The values a straight-line Python tasklet writes, by written container, as expressions over what it reads
    (``{'mask': <expr>}`` for ``t = a * 2; mask = t > 1`` writing ``mask``). Empty if the tasklet is not straight-line
    single assignments over single-element accesses. Outputs whose value reads a container the tasklet also writes are
    left out: the expression would read the new value where it is substituted."""
    from dace.frontend.python import astutils  # Avoid import loops
    if (node.node.language != dtypes.Language.Python or getattr(node.node, 'side_effects', False)
            or not isinstance(node.in_memlets, dict) or not isinstance(node.out_memlets, dict)):
        return {}
    memlets = list(node.in_memlets.values()) + list(node.out_memlets.values())
    if any(m.subset is None or m.subset.num_elements() != 1 for m in memlets):
        return {}
    env: Dict[str, ast.expr] = {c: _access(m, containers) for c, m in node.in_memlets.items()}
    for statement in node.node.code.code:
        if isinstance(statement, ast.AnnAssign) and statement.value is None:
            continue  # A declaration
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            target, value = statement.target.id, statement.value
        elif (isinstance(statement, ast.Assign) and len(statement.targets) == 1
              and isinstance(statement.targets[0], ast.Name)):
            target, value = statement.targets[0].id, statement.value
        else:
            return {}
        env[target] = _ReplaceNames(env).visit(astutils.copy_tree(value))
    written = {m.data for m in node.out_memlets.values()}
    result = {}
    for connector, memlet in node.out_memlets.items():
        value = env.get(connector)
        if value is None or sum(1 for _ in ast.walk(value)) > _MAX_SUBSTITUTED_NODES:
            continue
        if set(symbolic.symbols_in_ast(value)) & written:
            continue
        result[memlet.data] = value
    return result


def _assigned_values(node: tn.ScheduleTreeNode, containers: Dict[str, data.Data]) -> Dict[str, ast.expr]:
    """The values ``node`` assigns that a condition may use in their place, by target: see :func:`_assignment`, and
    every output of a straight-line tasklet (:func:`_tasklet_values`)."""
    if isinstance(node, tn.TaskletNode):
        return _tasklet_values(node, containers)
    assigned = _assignment(node, containers)
    return {} if assigned is None else {assigned[0]: assigned[1]}


def _substitutable(target: str, value: ast.expr, node: tn.ScheduleTreeNode, containers: Dict[str, data.Data]) -> bool:
    """Whether a condition may read ``value`` instead of ``target`` (as assigned by ``node``) with the same result:
    symbols hold values as they are computed, boolean containers hold truth values, an element of a container of the
    same type is copied exactly, and literals are exact in the type of their container."""
    if not is_pure(value):
        return False
    if isinstance(node, tn.AssignNode):
        return True
    desc = containers.get(target, None)
    if desc is None or desc.total_size != 1:
        return False
    if desc.dtype == dtypes.bool_:
        return True
    source = value.value if isinstance(value, ast.Subscript) else value
    if isinstance(source, ast.Name) and source.id in containers and containers[source.id].dtype == desc.dtype:
        return True  # A copy of an element of the same type
    literal = value.operand if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub) else value
    if isinstance(literal, ast.Constant) and isinstance(literal.value, (bool, int, float)):
        try:
            return desc.dtype.type(literal.value) == literal.value
        except (TypeError, ValueError, OverflowError):
            return False
    return False


class _ReplaceReads(ast.NodeTransformer):
    """Replaces reads of a single-element container (``t`` or ``t[0]``) with an expression."""

    def __init__(self, target: str, value: ast.expr):
        self.target, self.value, self.replaced = target, value, 0

    def _replacement(self) -> ast.expr:
        from dace.frontend.python import astutils  # Avoid import loops
        self.replaced += 1
        return astutils.copy_tree(self.value)

    def visit_Name(self, node: ast.Name):
        return self._replacement() if node.id == self.target else node

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == self.target:
            index = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if all(isinstance(i, ast.Constant) and i.value == 0 for i in index):
                return self._replacement()
            return node  # Some other element: leave alone
        return self.generic_visit(node)


def forward_substitute_conditions(stree: tn.ScheduleTreeScope) -> int:
    """
    Replace the values a condition reads by the expressions that compute them.

    A condition that reads a symbol or a single-element container (``if mask:``) is rewritten in terms of the value
    assigned to it (``mask = (order[k] == 0)`` makes it ``if order[k] == 0:``) when the assignment reaches the
    condition unchanged: it is the last write of that name before the condition, in the same scope or an enclosing
    one, and nothing in between writes the name or anything the value reads. Loops and maps between the assignment
    and the condition count as "in between" in their entirety (their later iterations run before the condition is
    evaluated again), and must not rebind a name the value reads. Assignments are symbol assignments, single-element
    copies, and straight-line Python tasklets over single-element accesses, each of whose outputs is the expression
    computing it from the tasklet's inputs (``t = a * 2; mask = t > 1`` gives ``mask`` the value ``a * 2 > 1``); a
    container's value is only used if the substitution cannot change the outcome (a boolean container, or a literal
    exact in the container's type).
    The assignments themselves are kept (see :func:`remove_dead_assignments`).

    :param stree: The schedule tree (or subtree) to transform in place.
    :return: The number of substitutions made.
    """
    root = stree.get_root()
    containers = root.containers
    written_in: Dict[int, tuple] = {}  # Subtree writes, by node (the node is held so that ids are not reused)

    def writes(node: tn.ScheduleTreeNode) -> Set[str]:
        entry = written_in.get(id(node))
        if entry is None:
            entry = written_in[id(node)] = (node, names_in_subtrees([node], names_written))
        return entry[1]

    values_of: Dict[int, tuple] = {}  # Assigned values, by node (held so that ids are not reused)

    def values(node: tn.ScheduleTreeNode) -> Dict[str, ast.expr]:
        entry = values_of.get(id(node))
        if entry is None:
            entry = values_of[id(node)] = (node, _assigned_values(node, containers))
        return entry[1]

    def reaching(branch: tn.ScheduleTreeScope, name: str) -> Optional[ast.expr]:
        """The value assigned to ``name`` that reaches the condition of ``branch``, or ``None``."""
        between: Set[str] = set()
        rebound: Set[str] = set()
        current = branch
        if isinstance(branch, tn.ElifScope):
            # Evaluated where its chain starts: the conditions before it have no effects, and their bodies do not run
            siblings = branch.parent.children
            position = next(k for k, c in enumerate(siblings) if c is branch)
            while not isinstance(siblings[position], tn.IfScope):
                position -= 1
            current = siblings[position]
        while current.parent is not None:
            parent = current.parent
            if isinstance(parent, tn.GBlock):
                return None  # Unstructured control flow: the preceding siblings need not run first
            siblings = parent.children
            position = next(k for k, c in enumerate(siblings) if c is current)
            for sibling in reversed(siblings[:position]):
                value = values(sibling).get(name)
                if value is not None:
                    depends = set(symbolic.symbols_in_ast(value))
                    if depends & (between | rebound) or not _substitutable(name, value, sibling, containers):
                        return None
                    return value
                if name in writes(sibling):
                    return None  # Written in a way that is not analyzed
                between |= writes(sibling)
            if isinstance(parent, (tn.LoopScope, tn.MapScope)):
                between |= writes(parent)
                rebound |= bound_names(parent)
                if name in between:
                    return None
            current = parent
        return None

    substitutions = 0
    for node in list(stree.preorder_traversal()):
        condition = condition_of(node) if isinstance(node, tn.ScheduleTreeScope) else None
        if condition is None:
            continue
        changed = True
        while changed:  # Substituted values may read further substitutable names
            changed = False
            for name in sorted(set(symbolic.symbols_in_ast(condition))):
                value = reaching(node, name)
                if value is None:
                    continue
                replacer = _ReplaceReads(name, value)
                condition = replacer.visit(condition)
                if replacer.replaced:
                    substitutions += replacer.replaced
                    changed = True
        node.condition = CodeBlock([ast.fix_missing_locations(ast.Expr(value=condition))])
    return substitutions


def remove_dead_assignments(stree: tn.ScheduleTreeScope) -> int:
    """
    Remove assignments whose result nothing reads: symbol assignments, and single-element copies or pure Python
    tasklets writing a transient scalar (as :func:`forward_substitute_conditions` leaves behind).

    :param stree: The schedule tree (or subtree) to transform in place.
    :return: The number of assignments removed.
    """
    root = stree.get_root()
    containers = root.containers
    in_descriptors = set().union(*(map(str, desc.free_symbols) for desc in containers.values()))
    removed = 0
    while True:
        index = AccessIndex(root)

        def dead(node: tn.ScheduleTreeNode) -> bool:
            assigned = _assignment(node, containers)
            if assigned is None or index.reads.get(assigned[0], 0) > 0 or not is_pure(assigned[1]):
                return False
            if isinstance(node, tn.AssignNode):
                return assigned[0] not in in_descriptors
            desc = containers.get(assigned[0], None)
            return desc is not None and desc.transient and desc.total_size == 1

        count = 0
        for scope in [n for n in stree.preorder_traversal() if isinstance(n, tn.ScheduleTreeScope)]:
            kept = [c for c in scope.children if not dead(c)]
            if len(kept) < len(scope.children):
                count += len(scope.children) - len(kept)
                scope.children = []
                scope.add_children(kept)
        removed += count
        if count == 0:
            return removed


def _negation_of(condition: ast.expr, other: ast.expr) -> bool:
    """Whether ``other`` is syntactically the negation of ``condition``, up to normalizing both (negations pushed into
    comparisons, disjunctive normal form)."""
    from dace.frontend.python import astutils  # Avoid import loops
    if ast.dump(astutils.negate_expr(condition).value) == ast.dump(other):
        return True
    lrr = range_analysis()

    def normalized(clauses) -> Optional[frozenset]:
        return None if clauses is None else frozenset(
            frozenset(ast.dump(atom) for atom in clause) for clause in clauses)

    negated = normalized(lrr.disjunctive_normal_form(condition, negate=True))
    return negated is not None and negated == normalized(lrr.disjunctive_normal_form(other))


def _complementary(first: tn.ScheduleTreeNode, second: tn.ScheduleTreeNode, containers: Dict[str, data.Data]) -> bool:
    """Whether ``if c: A`` (``first``) directly followed by ``if not c: B`` (``second``) runs exactly one of the two
    bodies, i.e., may become ``if c: A else: B``. The conditions must be pure and negations of each other, and ``A``
    must not change what ``c`` reads: it writes none of its names, creates no aliases (views, references) and does not
    jump away (a ``goto`` could re-enter before ``second``)."""
    if not all(isinstance(n, tn.IfScope) and not isinstance(n, tn.StateIfScope) for n in (first, second)):
        return False
    condition, other = condition_of(first), condition_of(second)
    if condition is None or other is None or not (is_pure(condition) and is_pure(other)):
        return False
    if not _negation_of(condition, other):
        return False
    read = set(symbolic.symbols_in_ast(condition))
    written = names_in_subtrees([first], names_written)
    if read & written:
        return False
    for name in read | written:
        if isinstance(containers.get(name, None), (data.View, data.Reference)):
            return False
    for node in first.preorder_traversal():
        if isinstance(node, (tn.ViewNode, tn.RefSetNode, tn.GotoNode)):
            return False
        if isinstance(node, (tn.TaskletNode, tn.LibraryCall)) and getattr(node.node, 'side_effects', False):
            return False  # E.g., a callback that may write anything
    return True


def pair_complementary_guards(stree: tn.ScheduleTreeScope) -> int:
    """
    Turn a guard followed directly by its negation (``if c: A`` then ``if not c: B``) into ``if c: A else: B``.

    The two are only exclusive if running ``A`` cannot change the outcome of ``c``: ``A`` may not write anything
    ``c`` reads, nor create aliases or jump away (see :func:`_complementary`). The conditions are compared after
    normalization, so ``if i < 4`` / ``if i >= 4`` and ``if mask`` / ``if not mask`` are both recognized. The
    resulting ``else`` lets later passes use the condition's negation as a fact without re-deriving it.

    :param stree: The schedule tree (or subtree) to transform in place.
    :return: The number of pairs formed.
    """
    containers = stree.get_root().containers
    paired = 0
    # Innermost scopes first: pairing drops the second ``if``, which must not be visited (and rewritten) afterwards
    for scope in reversed([n for n in stree.preorder_traversal() if isinstance(n, tn.ScheduleTreeScope)]):
        children, result, k, before = scope.children, [], 0, paired
        while k < len(children):
            node = children[k]
            following = children[k + 1] if k + 1 < len(children) else None
            after = children[k + 2] if k + 2 < len(children) else None
            # ``node`` must end its chain (``following`` is an ``if``), and ``following`` must be a chain of its own
            if (following is not None and not isinstance(after, (tn.ElifScope, tn.ElseScope))
                    and _complementary(node, following, containers)):
                result += [node, tn.ElseScope(children=list(following.children))]
                paired += 1
                k += 2
            else:
                result.append(node)
                k += 1
        if paired > before:
            scope.children = []
            scope.add_children(result)
    return paired


def hoist_condition_reads(stree: tn.ScheduleTreeScope) -> int:
    """
    Read the transient arrays an ``if`` condition reads in a tasklet before it: ``if T[i, k] * U[i, k] < 0:`` becomes
    ``c = tasklet(T[i, k], U[i, k])`` followed by ``if c:``, with a new transient boolean scalar ``c``.

    The passes that change transients (refinement, reuse of memory, moving to the stack) analyze and rewrite memlets,
    and leave containers that conditions read by name alone; after this pass, those accesses are memlets too. Only
    conditions that read a transient array are rewritten, and only if every container access in them can be evaluated
    before the condition: an access the condition may skip (e.g., the second operand of ``and``) must read an element
    the condition reads anyway, or one within the shape of its container for every iteration of the enclosing loops
    (such accesses are then read even where the condition would have skipped them), and no index may read a container
    (see :func:`condition_element_accesses`). ``elif`` conditions stay as they are, since they are
    evaluated only if the conditions before them do not hold.
    :func:`forward_substitute_conditions` does the opposite (it substitutes the boolean back into the condition), so
    this pass runs after it, before the transient passes.

    :param stree: The schedule tree (or subtree) to transform in place.
    :return: The number of conditions rewritten.
    """
    root = stree.get_root()
    containers = root.containers
    repository = repository_of(root)
    hoisted = 0

    def hoist(scope: tn.ScheduleTreeScope) -> None:
        nonlocal hoisted
        result = []
        for child in scope.children:
            if isinstance(child, tn.ScheduleTreeScope):
                hoist(child)
            condition = condition_of(child) if type(child) is tn.IfScope else None
            accesses = None
            if condition is not None and is_pure(condition):
                accesses = condition_element_accesses(condition, containers)
            if accesses and any(containers[c].transient and containers[c].total_size != 1 for c, *_ in accesses):
                # Reading an access the condition may skip is safe if it reads an element read anyway, or one in bounds
                certain = {element for _, element, _, conditional in accesses if not conditional}
                conditional = [(c, n) for c, element, n, cond in accesses if cond and element not in certain]
                if conditional:
                    ranges = loop_ranges(child, repository)
                    if not all(in_bounds(n, containers[c], ranges) for c, n in conditional):
                        accesses = None
            if accesses and any(containers[c].transient and containers[c].total_size != 1 for c, *_ in accesses):
                connectors: Dict[str, str] = {}  # By element read
                for _, element, _, _ in accesses:
                    connectors.setdefault(element, f'__in{len(connectors)}')
                names = {id(node): connectors[element] for _, element, node, _ in accesses}
                expression = ast.unparse(ReplaceAccesses(names).visit(condition))  # The condition is replaced below
                name = data.find_new_name('__condition', containers)
                containers[name] = data.Scalar(dtypes.bool_, transient=True)
                tasklet = nodes.Tasklet('condition', {c: None
                                                      for c in connectors.values()}, {'__out': None},
                                        f'__out = {expression}')
                read = tn.TaskletNode(node=tasklet,
                                      in_memlets={
                                          c: Memlet(e)
                                          for e, c in connectors.items()
                                      },
                                      out_memlets={'__out': Memlet(f'{name}[0]')})
                read.parent = scope
                result.append(read)
                child.condition = CodeBlock(name)
                hoisted += 1
            result.append(child)
        if len(result) != len(scope.children):
            scope.children = result

    hoist(stree)
    return hoisted
