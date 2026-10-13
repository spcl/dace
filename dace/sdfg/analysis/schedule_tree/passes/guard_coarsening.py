# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Guards that rarely hold: putting statements under the guards of their only readers, and skipping loop nests whose
statements are all guarded when no guard holds."""

import ast
import copy

from dace import data, dtypes, utils
from dace.frontend.python import astutils
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.dataflow import ReachingAnalysis
from dace.sdfg.analysis.schedule_tree.passes.common import (
    AccessIndex,
    ReplaceAccesses,
    ancestors,
    assigned_in,
    bound_between,
    child_containing,
    condition_element_accesses,
    condition_of,
    conjuncts,
    expression_symbols,
    in_bounds,
    is_pure,
    iteration_spaces,
    loop_ranges,
    memlet_element,
    names_in_subtrees,
    names_written,
    repository_of,
    trip_count,
)
from dace.sdfg.analysis.schedule_tree.passes.readers import (
    Read,
    collect_memlet_uses,
    collect_reads,
    guard_candidates,
    guarded_anywhere,
    some_read_guarded,
    value_reads,
)
from dace.sdfg.analysis.schedule_tree.passes.transient_reuse import Liveness, Use
from dace.sdfg.state import LoopRegion

# ----------------------------------------------------------------------------------------------------------------------
# Coarsening guards
# ----------------------------------------------------------------------------------------------------------------------


def _perfect_nest(loop: tn.ForScope) -> list[tn.ForScope]:
    """The loops of a perfect nest starting at ``loop``, outermost first."""
    chain = [loop]
    while len(chain[-1].children) == 1 and type(chain[-1].children[0]) is tn.ForScope:
        chain.append(chain[-1].children[0])
    return chain


def _testable_guards(
    body: list[tn.ScheduleTreeNode], containers: dict[str, data.Data], repository, local: set[str]
) -> list[tuple[ast.expr, list]] | None:
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
        kept = [c for c in conjuncts(condition) if not (expression_symbols(c) & local)]
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


def _loop_copy(loop: tn.ForScope, suffix: str, body: list[tn.ScheduleTreeNode]) -> tn.ForScope:
    """A loop with the header of ``loop`` and the given body."""
    old = loop.loop
    header = LoopRegion(
        f"{old.label}_{suffix}",
        condition_expr=copy.deepcopy(old.loop_condition),
        loop_var=old.loop_variable,
        initialize_expr=copy.deepcopy(old.init_statement),
        update_expr=copy.deepcopy(old.update_statement),
    )
    return tn.ForScope(loop=header, children=body)


def _existence_test(
    flag: str, loops: list[tn.ForScope], guards: list[tuple[ast.expr, list]], suffix: str
) -> list[tn.ScheduleTreeNode]:
    """``flag = False`` followed by loops (copies of ``loops``) that set ``flag`` if any guard holds."""
    connectors: dict[str, str] = {}  # By element read
    for _, accesses in guards:
        for _, element, _, _ in accesses:
            connectors.setdefault(element, f"__in{len(connectors)}")
    terms = []
    for condition, accesses in guards:
        # The accesses are replaced in a copy (the condition belongs to the scope), found by position in the walk
        copied = astutils.copy_tree(condition)
        copies = {id(original): new for original, new in zip(ast.walk(condition), ast.walk(copied))}
        names = {id(copies[id(node)]): connectors[element] for _, element, node, _ in accesses}
        terms.append(f"bool({ast.unparse(ReplaceAccesses(names).visit(copied))})")
    init = tn.TaskletNode(
        node=nodes.Tasklet("any_init", {}, {"__out"}, "__out = False"),
        in_memlets={},
        out_memlets={"__out": Memlet(f"{flag}[0]")},
    )
    in_memlets = {connector: Memlet(element) for element, connector in connectors.items()}
    in_memlets["__previous"] = Memlet(f"{flag}[0]")
    test: tn.ScheduleTreeNode = tn.TaskletNode(
        node=nodes.Tasklet("any_guard", set(in_memlets), {"__out"}, f"__out = __previous | {' | '.join(terms)}"),
        in_memlets=in_memlets,
        out_memlets={"__out": Memlet(f"{flag}[0]")},
    )
    for loop in reversed(loops):
        test = _loop_copy(loop, suffix, [test])
    return [init, test]


def coarsen_guards(
    stree: tn.ScheduleTreeScope,
    levels: tuple[str, ...] = ("nest", "row"),
    min_iterations: int = 32,
    min_row_trip_count: int = 8,
) -> int:
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

    def trips(loop: tn.ForScope) -> int | None:
        spaces = iteration_spaces(loop, repository)
        if len(spaces) != 1 or spaces[0][2] is None:
            return None
        return trip_count(spaces[0][2])

    def coarsen(nest: list[tn.ForScope]) -> list[tn.ScheduleTreeNode] | None:
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
        guard_symbols = set().union(*(expression_symbols(condition) for condition, _ in guards))
        variables = [loop.loop.loop_variable for loop in nest]
        # The loops the test iterates: those whose variables the guards read, and those their bounds read
        needed = {v for v in variables if v in guard_symbols}
        while True:
            more = {
                v for loop in nest if loop.loop.loop_variable in needed for v in variables if v in _header_symbols(loop)
            } - needed
            if not more:
                break
            needed |= more
        result: list[tn.ScheduleTreeNode] = [nest[0]]
        innermost = nest[-1]
        count = counts[-1]
        rows = (
            "row" in levels
            and len(nest) > 1
            and innermost.loop.loop_variable in guard_symbols
            and (count is None or count >= min_row_trip_count)
        )
        tested = [loop for loop in nest if loop.loop.loop_variable in needed]
        if rows:
            flag = utils.find_new_name("__any_row", containers)
            containers[flag] = data.Scalar(dtypes.bool_, transient=True)
            parent = nest[-2]
            test = _existence_test(flag, [innermost], guards, "any_row")
            parent.children = []
            parent.add_children(test + [tn.IfScope(condition=CodeBlock(flag), children=[innermost])])
            index.add(parent)
        # A test over every loop of the nest would read as much as the row tests, without their contiguous accesses
        # (compilers unroll the short innermost loop and vectorize across rows), so it is left to them
        if "nest" in levels and not (rows and len(tested) == len(nest)):
            flag = utils.find_new_name("__any", containers)
            containers[flag] = data.Scalar(dtypes.bool_, transient=True)
            test = _existence_test(flag, tested, guards, "any")
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


def _product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _header_symbols(loop: tn.ForScope) -> set[str]:
    return {s for code in loop.loop.get_meta_codeblocks() for s in code.get_free_symbols()} - {loop.loop.loop_variable}


# ----------------------------------------------------------------------------------------------------------------------
# Sinking statements into guards
# ----------------------------------------------------------------------------------------------------------------------


def _map_to_element(
    condition: ast.expr, bound: set[str], elements: list[tuple[str, ...] | None], written: tuple[str, ...]
) -> ast.expr | None:
    """``condition`` with the loop variables in ``bound`` replaced by the indices the statement writes at the
    positions where they index every element read, or ``None`` if a variable does not."""
    replacements: dict[str, str] = {}
    for variable in sorted(expression_symbols(condition) & bound):
        positions = None
        for element in elements:
            if element is None:
                return None
            here = {k for k, index in enumerate(element) if index == variable}
            positions = here if positions is None else positions & here
        if not positions:
            return None
        replacements[variable] = f"({written[min(positions)]})"
    mapped = astutils.copy_tree(condition)
    if replacements:
        mapped = astutils.ASTFindReplace(replacements).visit(mapped)
    return mapped


def _unchanged_between(
    condition: ast.expr, statement: tn.ScheduleTreeNode, read: tn.ScheduleTreeNode, scope: tn.ScheduleTreeScope
) -> bool:
    """Whether nothing in ``scope``, from the child that contains ``statement`` to the one that contains ``read``,
    assigns the containers or symbols ``condition`` reads."""
    children = scope.children
    first = next(k for k, c in enumerate(children) if c is child_containing(scope, statement))
    last = next(k for k, c in enumerate(children) if c is child_containing(scope, read))
    if last < first:
        return False
    return not (assigned_in(children[first : last + 1]) & expression_symbols(condition))


def _sinking_guard(
    statement: tn.TaskletNode,
    reads: dict[str, list[Read]],
    other: set[str],
    containers: dict[str, data.Data],
    liveness: Liveness,
    reaching: ReachingAnalysis,
    memlet_uses: dict[str, list[Use]],
    cache: dict[str, tuple],
) -> ast.expr | None:
    """The guard under which ``statement`` can run (see :func:`sink_into_guards`), or ``None``."""
    tasklet = statement.node
    if tasklet.language != dtypes.Language.Python or getattr(tasklet, "side_effects", False):
        return None
    if any(
        isinstance(n, ast.Call) and not isinstance(n.func, (ast.Name, ast.Attribute))
        for n in ast.walk(ast.Module(body=list(tasklet.code.code), type_ignores=[]))
    ):
        return None
    outputs: dict[str, tuple[str, ...]] = {}
    for memlet in statement.out_memlets.values():
        element = memlet_element(memlet)
        desc = containers.get(memlet.data)
        if (
            element is None
            or desc is None
            or not desc.transient
            or isinstance(desc, data.View)
            or memlet.data in other
            or memlet.wcr is not None
            or outputs.get(memlet.data, element) != element
        ):
            return None
        outputs[memlet.data] = element
    if any(m.data in outputs and memlet_element(m) != outputs[m.data] for m in statement.in_memlets.values()):
        return None  # Reads an output at another element than it writes
    # Quick filter before the liveness analysis: some read of every output that is read must be under a guard at all
    if not all(some_read_guarded(statement, reads.get(name, [])) for name in outputs):
        return None
    uses: dict[str, list[Read]] = {}
    scopes: dict[str, tn.ScheduleTreeScope] = {}
    for name in outputs:
        uses[name], scope = value_reads(statement, name, reads.get(name, []), liveness, reaching, memlet_uses, cache)
        if scope is None:
            return None
        scopes[name] = scope
    if not any(uses.values()) or not all(guarded_anywhere(r) for rs in uses.values() for r in rs):
        return None

    # Guards that already hold where the statement runs
    present = {
        ast.dump(c)
        for n in ancestors(statement)
        if type(n) is tn.IfScope
        for c in conjuncts(condition_of(n))
        if condition_of(n) is not None
    }
    choice: tuple[str, ast.expr] | None = None
    for name, written in outputs.items():
        if not uses[name]:
            continue  # Never read: whether it runs does not matter
        scope = scopes[name]
        statement_bound = bound_between(statement, scope)
        options: dict[str, ast.expr] | None = None
        for read in uses[name]:
            bound = bound_between(read.node, scope)
            found: dict[str, ast.expr] = {}
            for condition in guard_candidates(read, scope):
                if (expression_symbols(condition) - bound) & statement_bound:
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
        reads, other = collect_reads(root)
        liveness = Liveness(root, trust_reads=False)
        reaching = ReachingAnalysis(root)
        memlet_uses = collect_memlet_uses(root)
        cache: dict[str, tuple] = {}
        for node in stree.preorder_traversal():
            if not isinstance(node, tn.TaskletNode):
                continue
            guard = _sinking_guard(node, reads, other, containers, liveness, reaching, memlet_uses, cache)
            if guard is None:
                continue
            parent = node.parent
            position = next(k for k, c in enumerate(parent.children) if c is node)
            wrapper = tn.IfScope(
                condition=CodeBlock([ast.fix_missing_locations(ast.Expr(value=guard))]), children=[node]
            )
            children = list(parent.children)
            children[position] = wrapper
            parent.children = []
            parent.add_children(children)
            sunk += 1
            break  # The analyses describe the tree before the change
        else:
            return sunk
