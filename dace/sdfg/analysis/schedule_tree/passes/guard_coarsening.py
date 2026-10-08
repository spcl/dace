# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Guards that rarely hold: moving statements under the guards of their only readers, and skipping loop nests whose
statements are all guarded when no guard holds."""
import ast
import copy
from typing import Dict, List, Optional, Set, Tuple

from dace import data, dtypes, symbolic
from dace.frontend.python import astutils
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.common import (AccessIndex, ReplaceAccesses, ancestors, bound_names,
                                                            clone_subtree, make_scope, range_analysis,
                                                            condition_element_accesses, condition_of, in_bounds,
                                                            is_pure, iteration_spaces, loop_ranges, names_in_subtrees,
                                                            names_read, names_written, repository_of, trip_count)
from dace.sdfg.analysis.schedule_tree.passes.transient_reuse import Liveness, Use
from dace.sdfg.state import LoopRegion

# ----------------------------------------------------------------------------------------------------------------------
# Coarsening guards
# ----------------------------------------------------------------------------------------------------------------------


def _perfect_nest(loop: tn.ForScope) -> List[tn.ForScope]:
    """The loops of a perfect nest starting at ``loop``, outermost first."""
    chain = [loop]
    while len(chain[-1].children) == 1 and type(chain[-1].children[0]) is tn.ForScope:
        chain.append(chain[-1].children[0])
    return chain


def _symbols(expression: ast.expr) -> Set[str]:
    return set(symbolic.symbols_in_ast(expression))


def _testable_guards(body: List[tn.ScheduleTreeNode], containers: Dict[str, data.Data], repository,
                     local: Set[str]) -> Optional[List[Tuple[ast.expr, list]]]:
    """Conditions implied by those of a loop body that consists only of ``if`` scopes (without ``elif``/``else``),
    with their container accesses (see ``condition_element_accesses``), if they can be evaluated in a separate test:
    pure, and every access a condition may skip either read anyway or within the shape of its container. Conjuncts
    that read ``local`` containers (values computed in each iteration, meaningless before the loop) are left out of
    each condition, which the remaining conjuncts are then implied by."""
    if not body or any(type(child) is not tn.IfScope for child in body):
        return None
    guards = []
    for child in body:
        condition = condition_of(child)
        if condition is None or not is_pure(condition):
            return None
        kept = [c for c in _conjuncts(condition) if not (_symbols(c) & local)]
        if not kept:
            return None
        condition = kept[0] if len(kept) == 1 else ast.BoolOp(op=ast.And(), values=kept)
        accesses = condition_element_accesses(condition, containers)
        if accesses is None:
            return None
        certain = {element for _, element, _, conditional in accesses if not conditional}
        skippable = [(c, n) for c, element, n, conditional in accesses if conditional and element not in certain]
        if skippable:
            ranges = loop_ranges(child, repository)
            if not all(in_bounds(n, containers[c], ranges) for c, n in skippable):
                return None
        guards.append((condition, accesses))
    return guards


def _loop_copy(loop: tn.ForScope, suffix: str, body: List[tn.ScheduleTreeNode]) -> tn.ForScope:
    """A loop with the header of ``loop`` and the given body."""
    old = loop.loop
    header = LoopRegion(f'{old.label}_{suffix}',
                        condition_expr=copy.deepcopy(old.loop_condition),
                        loop_var=old.loop_variable,
                        initialize_expr=copy.deepcopy(old.init_statement),
                        update_expr=copy.deepcopy(old.update_statement))
    return tn.ForScope(loop=header, children=body)


def _existence_test(flag: str, loops: List[tn.ForScope], guards: List[Tuple[ast.expr, list]],
                    suffix: str) -> List[tn.ScheduleTreeNode]:
    """``flag = False`` followed by loops (copies of ``loops``) that set ``flag`` if any guard holds."""
    connectors: Dict[str, str] = {}  # By element read
    for _, accesses in guards:
        for _, element, _, _ in accesses:
            connectors.setdefault(element, f'__in{len(connectors)}')
    terms = []
    for condition, accesses in guards:
        # The accesses are replaced in a copy (the condition belongs to the scope), found by position in the walk
        copied = astutils.copy_tree(condition)
        copies = {id(original): new for original, new in zip(ast.walk(condition), ast.walk(copied))}
        names = {id(copies[id(node)]): connectors[element] for _, element, node, _ in accesses}
        terms.append(f'bool({ast.unparse(ReplaceAccesses(names).visit(copied))})')
    init = tn.TaskletNode(node=nodes.Tasklet('any_init', {}, {'__out'}, '__out = False'),
                          in_memlets={},
                          out_memlets={'__out': Memlet(f'{flag}[0]')})
    in_memlets = {connector: Memlet(element) for element, connector in connectors.items()}
    in_memlets['__previous'] = Memlet(f'{flag}[0]')
    test: tn.ScheduleTreeNode = tn.TaskletNode(node=nodes.Tasklet('any_guard', set(in_memlets), {'__out'},
                                                                  f'__out = __previous | {" | ".join(terms)}'),
                                               in_memlets=in_memlets,
                                               out_memlets={'__out': Memlet(f'{flag}[0]')})
    for loop in reversed(loops):
        test = _loop_copy(loop, suffix, [test])
    return [init, test]


def coarsen_guards(stree: tn.ScheduleTreeScope,
                   levels: Tuple[str, ...] = ('nest', 'row'),
                   min_iterations: int = 32,
                   min_row_trip_count: int = 8) -> int:
    """
    Skip the iterations of loop nests whose statements are all guarded when no guard holds: a perfect nest of ``for``
    loops whose innermost body consists only of ``if`` scopes runs only if a test finds a guard that holds.

    The test evaluates the guards on the values the nest starts with. If none holds there, no statement runs, so no
    value changes and no guard holds later in the nest either; if one holds, the nest runs as before. This pays off
    for guards that rarely hold (e.g., fixing negative values), where the test is a reduction that compilers
    vectorize instead of a loop of branches. It never runs more statements than the nest; it evaluates the guards
    up to twice more.

    Levels (``levels``):

    * ``nest``: one test before the nest, over the loops whose variables the guards (or the bounds of those loops)
      read; a guard that depends only on ``i, j`` in ``for k: for j: for i:`` is tested once per ``(j, i)``. Not if
      the guards read all variables and there are row tests: such a test reads as much as the row tests do, without
      their contiguous accesses.
    * ``row``: a test before the innermost loop, in each iteration of the loop around it, if the guards read the
      innermost variable and the innermost loop runs at least ``min_row_trip_count`` times.

    Guards must be pure and every container access they may skip (the second operand of ``and``, for instance) must
    be read anyway or provably within bounds, since the test reads all of them. Nests whose loop variables are used
    after them and nests with fewer than ``min_iterations`` iterations (when known) are left alone.

    :param stree: The schedule tree (or subtree) to transform in place.
    :param levels: The levels at which to test, see above.
    :param min_iterations: Do not coarsen nests with fewer known iterations.
    :param min_row_trip_count: Smallest known trip count of the innermost loop for a row test.
    :return: The number of nests coarsened.
    """
    root = stree.get_root()
    containers = root.containers
    repository = repository_of(root)
    index = AccessIndex(root)
    coarsened = 0

    def trips(loop: tn.ForScope) -> Optional[int]:
        spaces = iteration_spaces(loop, repository)
        if len(spaces) != 1 or spaces[0][2] is None:
            return None
        return trip_count(spaces[0][2])

    def coarsen(nest: List[tn.ForScope]) -> Optional[List[tn.ScheduleTreeNode]]:
        nonlocal coarsened
        if any(not loop.loop.loop_variable or index.used_outside(loop, loop.loop.loop_variable) for loop in nest):
            return None
        written = names_in_subtrees([nest[0]], names_written)
        local = {n for n in written if n in containers and containers[n].transient and containers[n].total_size == 1}
        guards = _testable_guards(nest[-1].children, containers, repository, local)
        if guards is None:
            return None
        counts = [trips(loop) for loop in nest]
        if all(c is not None for c in counts) and _product(counts) < min_iterations:
            return None
        guard_symbols = set().union(*(_symbols(condition) for condition, _ in guards))
        variables = [loop.loop.loop_variable for loop in nest]
        # The loops the test iterates: those whose variables the guards read, and those their bounds read
        needed = {v for v in variables if v in guard_symbols}
        while True:
            more = {
                v
                for loop in nest if loop.loop.loop_variable in needed for v in variables if v in _header_symbols(loop)
            } - needed
            if not more:
                break
            needed |= more
        result: List[tn.ScheduleTreeNode] = [nest[0]]
        innermost = nest[-1]
        count = counts[-1]
        rows = ('row' in levels and len(nest) > 1 and innermost.loop.loop_variable in guard_symbols
                and (count is None or count >= min_row_trip_count))
        tested = [loop for loop in nest if loop.loop.loop_variable in needed]
        if rows:
            flag = data.find_new_name('__any_row', containers)
            containers[flag] = data.Scalar(dtypes.bool_, transient=True)
            parent = nest[-2]
            test = _existence_test(flag, [innermost], guards, 'any_row')
            parent.children = []
            parent.add_children(test + [tn.IfScope(condition=CodeBlock(flag), children=[innermost])])
            index.add(parent)
        # A test over every loop of the nest would read as much as the row tests, without their contiguous accesses
        # (compilers unroll the short innermost loop and vectorize across rows), so it is left to them
        if 'nest' in levels and not (rows and len(tested) == len(nest)):
            flag = data.find_new_name('__any', containers)
            containers[flag] = data.Scalar(dtypes.bool_, transient=True)
            test = _existence_test(flag, tested, guards, 'any')
            result = test + [tn.IfScope(condition=CodeBlock(flag), children=[nest[0]])]
        elif not rows:
            return None
        coarsened += 1
        return result

    def visit(scope: tn.ScheduleTreeScope):
        changed, result = False, []
        for child in scope.children:
            replacement = coarsen(_perfect_nest(child)) if type(child) is tn.ForScope else None
            if replacement is None:
                if isinstance(child, tn.ScheduleTreeScope):
                    visit(child)
                result.append(child)
            else:
                changed = True
                result += replacement
        if changed:
            scope.children = []
            scope.add_children(result)
            index.changed()

    visit(stree)
    return coarsened


def _product(values: List[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _header_symbols(loop: tn.ForScope) -> Set[str]:
    return {s for code in loop.loop.get_meta_codeblocks() for s in code.get_free_symbols()} - {loop.loop.loop_variable}


# ----------------------------------------------------------------------------------------------------------------------
# Sinking statements into guards
# ----------------------------------------------------------------------------------------------------------------------


def _conjuncts(condition: ast.expr) -> List[ast.expr]:
    if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.And):
        return list(condition.values)
    return [condition]


def _normalized(indices: List[str]) -> Tuple[str, ...]:
    return tuple(str(symbolic.pystr_to_symbolic(i)) for i in indices)


def _memlet_element(memlet: Memlet) -> Optional[Tuple[str, ...]]:
    """The indices of the single element a memlet accesses, or ``None`` if it accesses more."""
    indices = []
    for start, end, _ in memlet.subset.ndrange():
        if symbolic.simplify(end - start) != 0:
            return None
        indices.append(str(start))
    return _normalized(indices)


class _Read:
    """A read of a container: by the memlets of a tasklet, or in the condition of an ``if`` scope (then with the
    conjuncts of the condition before the first one that reads it)."""

    def __init__(self,
                 node: tn.ScheduleTreeNode,
                 elements: List[Optional[Tuple[str, ...]]],
                 preceding: Optional[List[ast.expr]] = None):
        self.node = node
        self.elements = elements
        self.preceding = preceding or []


def _reads(root: tn.ScheduleTreeRoot) -> Tuple[Dict[str, List[_Read]], Set[str]]:
    """The reads of every container by tasklets and ``if`` conditions, and the containers read in other ways (which
    are left alone)."""
    containers = root.containers
    reads: Dict[str, List[_Read]] = {}
    other: Set[str] = set()
    for node in root.preorder_traversal():
        if isinstance(node, tn.TaskletNode):
            by_name: Dict[str, list] = {}
            for memlet in node.in_memlets.values():
                by_name.setdefault(memlet.data, []).append(_memlet_element(memlet))
            for name, elements in by_name.items():
                reads.setdefault(name, []).append(_Read(node, elements))
            other |= set(map(str, node.node.free_symbols)) & containers.keys()
            continue
        condition = condition_of(node) if type(node) is tn.IfScope else None
        accesses = None if condition is None else condition_element_accesses(condition, containers)
        if accesses is None:
            other |= names_read(node) & containers.keys()
            continue
        conjuncts = _conjuncts(condition)
        found: Dict[str, Tuple[list, int]] = {}
        for name, _, access, _ in accesses:
            position = next(k for k, c in enumerate(conjuncts) if any(n is access for n in ast.walk(c)))
            if isinstance(access, ast.Subscript):
                indices = access.slice.elts if isinstance(access.slice, ast.Tuple) else [access.slice]
                element = _normalized([ast.unparse(i) for i in indices])
            else:  # A single-element container by name
                element = ('0', )
            elements, first = found.get(name, ([], position))
            found[name] = (elements + [element], min(first, position))
        for name, (elements, position) in found.items():
            reads.setdefault(name, []).append(_Read(node, elements, conjuncts[:position]))
    return reads, other


def _path_to(node: tn.ScheduleTreeNode, scope: tn.ScheduleTreeScope) -> List[tn.ScheduleTreeNode]:
    """``node`` and its ancestors below ``scope``, innermost first."""
    path = []
    while node is not scope:
        path.append(node)
        node = node.parent
    return path


def _child_containing(scope: tn.ScheduleTreeScope, node: tn.ScheduleTreeNode) -> tn.ScheduleTreeNode:
    return _path_to(node, scope)[-1]


def _inside(node: tn.ScheduleTreeNode, scope: tn.ScheduleTreeScope) -> bool:
    while node is not None:
        if node is scope:
            return True
        node = node.parent
    return False


def _bound_between(node: tn.ScheduleTreeNode, scope: tn.ScheduleTreeScope) -> Set[str]:
    return set().union(*(bound_names(n) for n in _path_to(node, scope)))


def _guard_candidates(read: _Read, scope: tn.ScheduleTreeScope) -> List[ast.expr]:
    """The conditions that hold whenever ``read`` happens, from the ``if`` scopes between it and ``scope``."""
    candidates = list(read.preceding)
    for node in _path_to(read.node, scope)[1:]:
        if type(node) is tn.IfScope:
            condition = condition_of(node)
            if condition is not None and is_pure(condition):
                candidates += _conjuncts(condition)
    return candidates


def _guarded_anywhere(read: _Read) -> bool:
    """Whether some ``if`` condition holds whenever ``read`` happens (a quick filter before the full analysis)."""
    return bool(read.preceding) or any(type(n) is tn.IfScope for n in ancestors(read.node))


def _some_read_guarded(statement: tn.ScheduleTreeNode, reads: List[_Read]) -> bool:
    """Whether a container that ``statement`` writes is never read by others, or some read is under a guard."""
    others = [r for r in reads if r.node is not statement]
    return not others or any(_guarded_anywhere(r) for r in others)


def _memlet_uses(root: tn.ScheduleTreeRoot) -> Dict[str, List[Use]]:
    uses: Dict[str, List[Use]] = {}
    for node in root.preorder_traversal():
        for attr, write in (('in_memlets', False), ('out_memlets', True)):
            memlets = getattr(node, attr, None)
            if isinstance(memlets, dict):
                for connector, memlet in memlets.items():
                    uses.setdefault(memlet.data, []).append(Use(node, memlets, connector, write))
    return uses


def _value_reads(statement: tn.TaskletNode, name: str, reads: List[_Read], liveness: Liveness,
                 memlet_uses: Dict[str, List[Use]],
                 cache: Dict[str, tuple]) -> Tuple[List[_Read], Optional[tn.ScheduleTreeScope]]:
    """The reads that may see a value ``statement`` writes to ``name`` (those in its live segment), and the lowest
    scope above the statement that contains every access in the segment and, if the statement reads ``name`` too (as
    an accumulation does), in which ``name`` is written before it is read: no value flows into one execution of the
    scope from outside or from a previous one. ``(reads, None)`` if there is no such scope."""
    uses = list(memlet_uses.get(name, []))
    for read in reads:
        if not isinstance(read.node, tn.TaskletNode):  # Reads in conditions
            for element in read.elements:
                memlet = Memlet(f'{name}[{", ".join(element)}]')
                uses.append(Use(read.node, {'__cond': memlet}, '__cond', False))
    if name not in cache:  # The same for every statement writing ``name`` (until the tree changes)
        cache[name] = liveness.segments(uses)[0]
    segments = cache[name]
    position = liveness.position[id(statement)]
    segment = next(((first, last) for first, last in segments if first <= position <= last), None)
    if segment is None:
        return [], statement.parent  # The values are never read

    def in_segment(node: tn.ScheduleTreeNode) -> bool:
        return segment[0] <= liveness.position[id(node)] <= segment[1]

    found = [r for r in reads if r.node is not statement and in_segment(r.node)]
    segment_uses = [u for u in uses if in_segment(u.node)]
    accumulates = any(m.data == name for m in statement.in_memlets.values())
    scope = statement.parent
    while scope is not None:
        if all(_inside(u.node, scope) for u in segment_uses):
            if not accumulates or not liveness.exposed_in(scope, [u for u in uses if _inside(u.node, scope)]):
                return found, scope
        scope = scope.parent
    return found, None


def _map_to_element(condition: ast.expr, bound: Set[str], elements: List[Optional[Tuple[str, ...]]],
                    written: Tuple[str, ...]) -> Optional[ast.expr]:
    """``condition`` with the loop variables in ``bound`` replaced by the indices the statement writes at the
    positions where they index every element read, or ``None`` if a variable does not."""
    replacements: Dict[str, str] = {}
    for variable in sorted(_symbols(condition) & bound):
        positions = None
        for element in elements:
            if element is None:
                return None
            here = {k for k, index in enumerate(element) if index == variable}
            positions = here if positions is None else positions & here
        if not positions:
            return None
        replacements[variable] = f'({written[min(positions)]})'
    mapped = astutils.copy_tree(condition)
    if replacements:
        mapped = astutils.ASTFindReplace(replacements).visit(mapped)
    return mapped


def _assigned_in(nodes_: List[tn.ScheduleTreeNode]) -> Set[str]:
    """Containers and symbols assigned in subtrees, other than loop and map variables."""
    assigned: Set[str] = set()
    for root in nodes_:
        for node in root.preorder_traversal():
            if not isinstance(node, (tn.LoopScope, tn.MapScope)):
                assigned |= names_written(node)
    return assigned


def _unchanged_between(condition: ast.expr, statement: tn.ScheduleTreeNode, read: tn.ScheduleTreeNode,
                       scope: tn.ScheduleTreeScope) -> bool:
    """Whether nothing in ``scope``, from the child that contains ``statement`` to the one that contains ``read``,
    assigns the containers or symbols ``condition`` reads."""
    children = scope.children
    first = next(k for k, c in enumerate(children) if c is _child_containing(scope, statement))
    last = next(k for k, c in enumerate(children) if c is _child_containing(scope, read))
    if last < first:
        return False
    return not (_assigned_in(children[first:last + 1]) & _symbols(condition))


def _sinking_guard(statement: tn.TaskletNode, reads: Dict[str, List[_Read]], other: Set[str],
                   containers: Dict[str, data.Data], liveness: Liveness, memlet_uses: Dict[str, List[Use]],
                   cache: Dict[str, tuple]) -> Optional[ast.expr]:
    """The guard under which ``statement`` can run (see :func:`sink_into_guards`), or ``None``."""
    tasklet = statement.node
    if tasklet.language != dtypes.Language.Python or getattr(tasklet, 'side_effects', False):
        return None
    if any(
            isinstance(n, ast.Call) and not isinstance(n.func, (ast.Name, ast.Attribute))
            for n in ast.walk(ast.Module(body=list(tasklet.code.code), type_ignores=[]))):
        return None
    outputs: Dict[str, Tuple[str, ...]] = {}
    for memlet in statement.out_memlets.values():
        element = _memlet_element(memlet)
        desc = containers.get(memlet.data)
        if (element is None or desc is None or not desc.transient or isinstance(desc, data.View) or memlet.data in other
                or memlet.wcr is not None or outputs.get(memlet.data, element) != element):
            return None
        outputs[memlet.data] = element
    if any(m.data in outputs and _memlet_element(m) != outputs[m.data] for m in statement.in_memlets.values()):
        return None  # Reads an output at another element than it writes
    # Quick filter before the liveness analysis: some read of every output that is read must be under a guard at all
    if not all(_some_read_guarded(statement, reads.get(name, [])) for name in outputs):
        return None
    uses: Dict[str, List[_Read]] = {}
    scopes: Dict[str, tn.ScheduleTreeScope] = {}
    for name in outputs:
        uses[name], scope = _value_reads(statement, name, reads.get(name, []), liveness, memlet_uses, cache)
        if scope is None:
            return None
        scopes[name] = scope
    if not any(uses.values()) or not all(_guarded_anywhere(r) for rs in uses.values() for r in rs):
        return None

    # Guards that already hold where the statement runs
    present = {
        ast.dump(c)
        for n in ancestors(statement) if type(n) is tn.IfScope for c in _conjuncts(condition_of(n))
        if condition_of(n) is not None
    }
    choice: Optional[Tuple[str, ast.expr]] = None
    for name, written in outputs.items():
        if not uses[name]:
            continue  # Never read: whether it runs does not matter
        scope = scopes[name]
        statement_bound = _bound_between(statement, scope)
        options: Optional[Dict[str, ast.expr]] = None
        for read in uses[name]:
            bound = _bound_between(read.node, scope)
            found: Dict[str, ast.expr] = {}
            for condition in _guard_candidates(read, scope):
                if (_symbols(condition) - bound) & statement_bound:
                    continue  # Would read a variable of a loop around the statement under the same name
                mapped = _map_to_element(condition, bound, read.elements, written)
                if mapped is not None and _unchanged_between(condition, statement, read.node, scope):
                    found.setdefault(ast.dump(mapped), mapped)
            options = found if options is None else {k: v for k, v in options.items() if k in found}
            if not options:
                return None
        if choice is None:
            choice = next(iter(options.items()))
        elif choice[0] not in options:
            return None
    if choice is None or choice[0] in present:
        return None
    return choice[1]


def sink_into_guards(stree: tn.ScheduleTreeScope) -> int:
    """
    Put statements whose results are only used under a guard under that guard: a tasklet ``t = f(...)`` whose outputs
    are read only where ``g`` holds becomes ``if g: t = f(...)``, so that it does not run where its results would be
    discarded (partial dead code elimination). :func:`coarsen_guards` can then skip whole nests of such statements
    when ``g`` rarely holds.

    A tasklet without side effects is guarded by a condition ``g`` if, for each of its outputs (transient containers,
    which it reads, if at all, only at the elements it writes, as in ``s[i] = s[i] + x``), every read that may see a
    value it writes (in the same live segment, see ``Liveness``; containers shared by unrelated values are fine):

    * is by a tasklet or an ``if`` condition, inside an ``if`` scope with ``g`` as a conjunct of its condition (or in
      a later conjunct of such a condition), below the lowest scope ``P`` above the tasklet that contains all accesses
      of the segment (for outputs the tasklet also reads: in which the output is written before it is read, so that
      no value flows into an execution of ``P`` from before it);
    * sees ``g`` read the variables of the loops between ``P`` and the read only as indices of the element read; they
      are replaced by the indices of the element the tasklet writes, so that ``g`` has one value per element;
    * follows the tasklet without anything in ``P`` assigning what ``g`` reads in between.

    Each element of an output is then read only where ``g`` holds for it, and where it does, the tasklet ran for it
    exactly as before.

    :param stree: The schedule tree (or subtree) to transform in place.
    :return: The number of tasklets guarded.
    """
    root = stree.get_root()
    containers = root.containers
    sunk = 0
    while True:
        reads, other = _reads(root)
        liveness = Liveness(root, trust_reads=False)
        memlet_uses = _memlet_uses(root)
        cache: Dict[str, tuple] = {}
        for node in stree.preorder_traversal():
            if not isinstance(node, tn.TaskletNode):
                continue
            guard = _sinking_guard(node, reads, other, containers, liveness, memlet_uses, cache)
            if guard is None:
                continue
            parent = node.parent
            position = next(k for k, c in enumerate(parent.children) if c is node)
            wrapper = tn.IfScope(condition=CodeBlock([ast.fix_missing_locations(ast.Expr(value=guard))]),
                                 children=[node])
            children = list(parent.children)
            children[position] = wrapper
            parent.children = []
            parent.add_children(children)
            sunk += 1
            break  # The analyses describe the tree before the change
        else:
            return sunk


# ----------------------------------------------------------------------------------------------------------------------
# Moving statements to their readers
# ----------------------------------------------------------------------------------------------------------------------


def _nest_around(statement: tn.ScheduleTreeNode) -> Optional[List[tn.ForScope]]:
    """The loops of the perfect nest whose innermost body contains ``statement``, outermost first."""
    loops = []
    scope = statement.parent
    while type(scope) is tn.ForScope:
        loops.append(scope)
        if type(scope.parent) is not tn.ForScope or len(scope.parent.children) != 1:
            break
        scope = scope.parent
    return loops[::-1] or None


def _injective(element: Tuple[str, ...], variables: List[str]) -> bool:
    """Whether every variable is the index of some dimension, up to a constant (so iterations access distinct
    elements)."""
    for variable in variables:
        if not any(
                symbolic.simplify(symbolic.pystr_to_symbolic(index) - symbolic.pystr_to_symbolic(variable)).is_number
                for index in element):
            return False
    return True


def _outer_offset(element: Tuple[str, ...], variable: str) -> Optional[Tuple[int, int]]:
    """``(dimension, c)`` of the first dimension indexed by ``variable + c``, or ``None``."""
    for dim, index in enumerate(element):
        difference = symbolic.simplify(symbolic.pystr_to_symbolic(index) - symbolic.pystr_to_symbolic(variable))
        if difference.is_Integer:
            return dim, int(difference)
    return None


def _overlap(a, b) -> bool:
    return all(alo <= bhi and blo <= ahi for (alo, ahi), (blo, bhi) in zip(a, b))


def move_statements_to_readers(stree: tn.ScheduleTreeScope) -> int:
    """
    Move statements past what prevents putting them under the guard of their readers: an effect-free tasklet in the
    body of a perfect nest of ``for`` loops, whose values are read only under ``if`` scopes in a later nest of the same
    scope, is split off into a nest of its own (loop fission) and placed right after the last nest in between that
    assigns what those guards read. :func:`sink_into_guards` can then put it under the guard of its readers.

    The split is valid if the other statements of the nest neither read nor write the tasklet's outputs, and write
    its inputs only at the elements it reads, before it in the body, with every loop variable indexing those
    elements (each iteration then reads what it read before). The move is valid past nests that access none of the
    elements the tasklet writes and write none of those it reads, as regions over the loop ranges. Where they do so
    only in the first or last iterations of the outermost loop, those iterations stay in place in a nest of their own.

    :param stree: The schedule tree (or subtree) to transform in place.
    :return: The number of statements moved.
    """
    root = stree.get_root()
    containers = root.containers
    lrr = range_analysis()
    repository = repository_of(root)
    moved = 0
    while True:
        reads, other = _reads(root)
        liveness = Liveness(root, trust_reads=False)
        memlet_uses = _memlet_uses(root)
        cache: Dict[str, tuple] = {}
        for node in stree.preorder_traversal():
            if isinstance(node, tn.TaskletNode):
                plan = _move_plan(node, reads, other, containers, liveness, memlet_uses, repository, cache)
                if plan is not None:
                    _apply_move(node, *plan, lrr)
                    moved += 1
                    break  # The analyses describe the tree before the change
        else:
            return moved


def _move_plan(statement: tn.TaskletNode, reads: Dict[str, List[_Read]], other: Set[str],
               containers: Dict[str, data.Data], liveness: Liveness, memlet_uses: Dict[str, List[Use]], repository,
               cache: Dict[str, tuple]) -> Optional[tuple]:
    """``(nest, target, space, stay)`` to move ``statement`` out of ``nest`` to before the sibling ``target``, keeping
    the iterations of the outermost loop (iteration space ``space``) in the interval ``stay`` in place (or ``None``),
    or ``None`` if it cannot or need not be moved."""
    tasklet = statement.node
    if tasklet.language != dtypes.Language.Python or getattr(tasklet, 'side_effects', False):
        return None
    nest = _nest_around(statement)
    if nest is None or any(not loop.loop.loop_variable for loop in nest):
        return None
    scope = nest[0].parent
    if scope is None:
        return None
    variables = [loop.loop.loop_variable for loop in nest]
    outputs: Dict[str, Tuple[str, ...]] = {}
    for memlet in statement.out_memlets.values():
        element = _memlet_element(memlet)
        desc = containers.get(memlet.data)
        if (element is None or desc is None or not desc.transient or memlet.data in other or memlet.wcr is not None
                or outputs.get(memlet.data, element) != element):
            return None
        outputs[memlet.data] = element
    inputs: Dict[str, Tuple[str, ...]] = {}
    for memlet in statement.in_memlets.values():
        element = _memlet_element(memlet)
        if element is None or memlet.data in outputs or inputs.get(memlet.data, element) != element:
            return None
        inputs[memlet.data] = element

    if not all(_some_read_guarded(statement, reads.get(name, [])) for name in outputs):
        return None  # Quick filter before the liveness analysis
    # The readers: under guards, all in one later sibling nest, with nests in between (else there is no reason to move)
    siblings = scope.children
    position = next(k for k, c in enumerate(siblings) if c is nest[0])
    target = None
    for name in outputs:
        found, _ = _value_reads(statement, name, reads.get(name, []), liveness, memlet_uses, cache)
        for read in found:
            if not _guarded_anywhere(read) or not _inside(read.node, scope) or _inside(read.node, nest[0]):
                return None
            container = _child_containing(scope, read.node)
            if target is not None and container is not target:
                return None
            target = container
    if target is None:
        return None
    # Move only past the nests that assign what the readers' guards read, i.e., that prevent sinking it, and no further
    guard_names = set()
    for name in outputs:
        for read in _value_reads(statement, name, reads.get(name, []), liveness, memlet_uses, cache)[0]:
            guard_names |= set().union(*(_symbols(c) for c in _guard_candidates(read, scope)))
    readers_position = next(k for k, c in enumerate(siblings) if c is target)
    blockers = [k for k in range(position + 1, readers_position) if _assigned_in([siblings[k]]) & guard_names]
    if not blockers:
        return None
    target_position = blockers[-1] + 1
    target = siblings[target_position]

    # The split: the rest of the body does not use the outputs and writes the inputs only where the statement reads
    body = nest[-1].children
    index = next(k for k, c in enumerate(body) if c is statement)
    rest = body[:index] + body[index + 1:]
    if (names_in_subtrees(rest, names_read) | names_in_subtrees(rest, names_written)) & outputs.keys():
        return None
    for k, child in enumerate(body):
        if child is statement:
            continue
        for node in child.preorder_traversal():
            written = names_written(node) & inputs.keys()
            if not written:
                continue
            if k > index or not isinstance(node, tn.TaskletNode):
                return None
            for memlet in node.out_memlets.values():
                if memlet.data in inputs and (_memlet_element(memlet) != inputs[memlet.data]
                                              or not _injective(inputs[memlet.data], variables)):
                    return None

    # The move: conflicts with the nests in between, as intervals of the outermost loop
    spaces = iteration_spaces(nest[0], repository)
    if len(spaces) != 1 or spaces[0][2] is None:
        return None
    space = spaces[0][2]
    first, last = (int(space.start), int(space.end)) if space.ascending else (int(space.end), int(space.start))
    conflicts: List[Tuple[int, int]] = []
    accesses = [(Use(statement, statement.in_memlets, c, False), m) for c, m in statement.in_memlets.items()]
    accesses += [(Use(statement, statement.out_memlets, c, True), m) for c, m in statement.out_memlets.items()]
    free = set(map(str, tasklet.free_symbols)) | {
        str(sym)
        for m in list(statement.in_memlets.values()) + list(statement.out_memlets.values())
        for sym in m.free_symbols
    }
    for between in siblings[position + 1:target_position]:
        if _assigned_in([between]) & (free - set(variables)):
            return None  # Changes a symbol the statement reads
        through_memlets = {
            name: [u for u in memlet_uses.get(name, []) if _inside(u.node, between)]
            for name in list(inputs) + list(outputs)
        }
        accessed = names_in_subtrees([between], names_read) | names_in_subtrees([between], names_written)
        for name in outputs:
            if name in accessed and not through_memlets[name]:
                return None  # Accessed other than through memlets (e.g., in a condition)
        for name in inputs:
            if name in _assigned_in([between]) and not any(u.write for u in through_memlets[name]):
                return None  # Written other than through memlets (e.g., by a copy)
        for use, memlet in accesses:
            mine = liveness._box(use, scope)
            for theirs_use in through_memlets[memlet.data]:
                if not (use.write or theirs_use.write):
                    continue  # Two reads do not conflict
                theirs = liveness._box(theirs_use, scope)
                if mine is None or theirs is None:
                    return None
                if not _overlap(mine, theirs):
                    continue
                offset = _outer_offset(_memlet_element(memlet), variables[0])
                if offset is None:
                    return None
                dim, c = offset
                lo, hi = max(theirs[dim][0] - c, first), min(theirs[dim][1] - c, last)
                if lo <= hi:
                    conflicts.append((int(lo), int(hi)))
    stay = None
    if conflicts:
        lo, hi = min(c[0] for c in conflicts), max(c[1] for c in conflicts)
        if lo == first and hi < last:
            stay = (first, hi)
        elif hi == last and lo > first:
            stay = (lo, last)
        else:
            return None
    return nest, target, space, stay


def _apply_move(statement: tn.TaskletNode, nest: List[tn.ForScope], target: tn.ScheduleTreeNode, space, stay,
                lrr) -> None:
    scope = nest[0].parent
    innermost = nest[-1]
    innermost.children = [c for c in innermost.children if c is not statement]

    def copy_nest(interval, k: int) -> tn.ForScope:
        body: List[tn.ScheduleTreeNode] = [clone_subtree(statement)]
        for loop in reversed(nest[1:]):
            body = [make_scope(loop, None, None, None, body, k)]
        return make_scope(nest[0], None, space, interval, body, k)

    first, last = (space.start, space.end) if space.ascending else (space.end, space.start)
    moved_interval = None
    if stay is not None:
        lo, hi = stay
        moved_interval = lrr.Interval(hi + 1, last) if lo == first else lrr.Interval(first, lo - 1)
    children = list(scope.children)
    if stay is not None:
        children.insert(next(k for k, c in enumerate(children) if c is nest[0]) + 1, copy_nest(lrr.Interval(*stay), 1))
    children.insert(next(k for k, c in enumerate(children) if c is target), copy_nest(moved_interval, 2))
    if not innermost.children:
        children = [c for c in children if c is not nest[0]]
    scope.children = []
    scope.add_children(children)
