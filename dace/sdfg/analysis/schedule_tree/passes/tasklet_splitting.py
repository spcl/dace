# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Splitting tasklets into a part that computes some of their values and a part that uses them.

A Python tasklet whose code is a sequence of single assignments is a small dataflow graph. Splitting it lets other
passes treat its parts separately: e.g., the part of a statement in a loop that does not depend on the loop can be
hoisted out of it while the rest stays (see :func:`~dace.sdfg.analysis.schedule_tree.passes.code_motion.
hoist_invariant_statements`). Values that the second part needs from the first pass through containers.
"""

import ast
import copy
from collections.abc import Callable
from dataclasses import dataclass

from dace import data, dtypes
from dace.frontend.python import astutils
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import type_inference
from dace.sdfg.analysis.schedule_tree import treenodes as tn


@dataclass
class TaskletCode:
    """
    The code of a Python tasklet as a dataflow graph of single assignments.

    :param statements: The assignments, in order (declarations without a value are kept in ``declarations``).
    :param targets: The name each statement assigns.
    :param reads: The names each statement reads (connectors, locals, symbols and builtins).
    :param declarations: Type annotations without values (``x: dace.float32``), by name.
    :param inputs: The input connectors.
    :param outputs: The output connectors.
    """

    statements: list[ast.stmt]
    targets: list[str]
    reads: list[set[str]]
    declarations: dict[str, ast.AnnAssign]
    inputs: set[str]
    outputs: set[str]

    def defining(self, name: str) -> int | None:
        """The index of the statement that assigns ``name``, if any."""
        return next((k for k, target in enumerate(self.targets) if target == name), None)

    def trivial(self, index: int) -> bool:
        """Whether a statement only copies an input connector or a constant (cheaper to repeat than to store)."""
        names = self.reads[index] - {"float", "int", "bool", "dace"}
        value = self.statements[index].value
        return not names or (isinstance(value, ast.Name) and value.id in self.inputs)


# Rough cost of operations relative to an addition: divisions, powers and calls (e.g., ``sqrt``, ``exp``) take many
# cycles or are not vectorized as cheaply
_EXPENSIVE = (ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_EXPENSIVE_WEIGHT = 8
_CHEAP_CALLS = {"float", "int", "bool", "abs", "min", "max"}


def operation_cost(statement: ast.stmt) -> int:
    """The rough cost of the operations of a statement, in additions (see ``_EXPENSIVE_WEIGHT``)."""
    cost = 0
    for node in ast.walk(statement):
        if isinstance(node, ast.BinOp):
            cost += _EXPENSIVE_WEIGHT if isinstance(node.op, _EXPENSIVE) else 1
        elif isinstance(node, (ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp)):
            cost += 1
        elif isinstance(node, ast.Call):
            name = astutils.rname(node.func)
            cost += 0 if name in _CHEAP_CALLS else _EXPENSIVE_WEIGHT
    return cost


def parse_tasklet(node: tn.TaskletNode) -> TaskletCode | None:
    """
    The code of a tasklet as single assignments, if it is a Python tasklet without side effects or global code whose
    statements each assign one name (a local or an output connector) exactly once, never an input connector.

    :param node: The tasklet.
    :return: The parsed code, or ``None`` if it has another form.
    """
    tasklet = node.node
    if (
        tasklet.language != dtypes.Language.Python
        or tasklet.side_effects
        or tasklet.code_global.as_string.strip()
        or tasklet.code_init.as_string.strip()
        or tasklet.code_exit.as_string.strip()
    ):
        return None
    inputs, outputs = set(tasklet.in_connectors), set(tasklet.out_connectors)
    code = TaskletCode([], [], [], {}, inputs, outputs)
    for statement in tasklet.code.code:
        if isinstance(statement, ast.AnnAssign) and statement.value is None and isinstance(statement.target, ast.Name):
            code.declarations[statement.target.id] = statement
            continue
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target = statement.target
        else:
            return None
        if not isinstance(target, ast.Name) or target.id in inputs or target.id in code.targets:
            return None
        code.statements.append(statement)
        code.targets.append(target.id)
        code.reads.append({n.id for n in ast.walk(statement.value) if isinstance(n, ast.Name)})
    if set(code.targets) & outputs != outputs:
        return None  # An output is never assigned
    return code


def dependencies(code: TaskletCode, indices: set[int]) -> set[int]:
    """The statements that ``indices`` depend on (transitively), including them."""
    result, pending = set(), list(indices)
    while pending:
        index = pending.pop()
        if index in result:
            continue
        result.add(index)
        for name in code.reads[index]:
            defining = code.defining(name)
            if defining is not None:
                pending.append(defining)
    return result


def passed_values(code: TaskletCode, first: set[int]) -> list[str]:
    """The values computed by the statements ``first`` that :func:`split_tasklet` passes to the rest (those it reads,
    other than copies of inputs or constants)."""
    second = [k for k in range(len(code.statements)) if k not in first]
    needed = set().union(*(code.reads[k] for k in second)) if second else set()
    return [code.targets[k] for k in sorted(first) if code.targets[k] in needed and not code.trivial(k)]


def split_saving(code: TaskletCode, first: set[int]) -> int:
    """
    The rough saving (in additions, see :func:`operation_cost`) per execution of a tasklet when the statements
    ``first`` no longer run there (e.g., hoisted out of a loop): their operations, minus a load for each value passed
    to the rest, plus a load for each input that only they read. Copies of inputs or constants that the rest needs are
    computed there again, so they save nothing.

    :param code: The parsed tasklet.
    :param first: The statements that no longer run.
    :return: The saving; not positive if the split does not pay off.
    """
    second = [k for k in range(len(code.statements)) if k not in first]
    needed = set().union(*(code.reads[k] for k in second)) if second else set()
    # Copies the rest needs are computed there again
    kept = second + [k for k in first if code.trivial(k) and code.targets[k] in needed]
    removed = [k for k in first if k not in kept]
    if not removed:
        return 0
    still_read = set().union(*(code.reads[k] for k in kept)) if kept else set()
    only_first = {name for k in removed for name in code.reads[k] if name in code.inputs} - still_read
    operations = sum(operation_cost(code.statements[k]) for k in removed)
    return operations - len(passed_values(code, first)) + len(only_first)


def split_tasklet(
    node: tn.TaskletNode,
    code: TaskletCode,
    first: set[int],
    store: Callable[[str, dtypes.typeclass], Memlet],
    containers: dict[str, data.Data],
) -> tuple[tn.TaskletNode, tn.TaskletNode]:
    """
    Split a tasklet in two that run one after the other: the first computes the statements ``first`` (which must not
    depend on the others), the second the rest. Values of the first part that the second reads are passed through
    containers: output connectors through their own memlets, locals through ``store``; locals that only copy an input
    or a constant are computed again in the second part instead.

    :param node: The tasklet (unchanged).
    :param code: Its parsed code (see :func:`parse_tasklet`).
    :param first: Indices of the statements of the first part (closed under :func:`dependencies`).
    :param store: Called with the name and type of each local to pass; returns the memlet through which the first part
                  writes it and the second reads it. The type is ``None`` if it cannot be inferred.
    :param containers: The containers of the tree (for the types of connectors that have none).
    :return: The two tasklets.
    """
    second = [k for k in range(len(code.statements)) if k not in first]
    needed = set().union(*(code.reads[k] for k in second)) if second else set()
    passed = passed_values(code, first)
    repeated = [k for k in sorted(first) if code.targets[k] in needed and code.trivial(k)]
    second = sorted(second + repeated)

    tasklet = node.node
    types: dict[str, dtypes.typeclass] = {}
    for connectors, memlets in ((tasklet.in_connectors, node.in_memlets), (tasklet.out_connectors, node.out_memlets)):
        for connector, dtype in connectors.items():
            if (dtype is None or dtype == dtypes.typeclass(None)) and connector in memlets:
                dtype = containers[memlets[connector].data].dtype
            types[connector] = dtype
    # Over the whole code with its declarations: a local declared as ``float32`` stays one even if computed from doubles
    inferred = type_inference.infer_types(list(tasklet.code.code), {k: v for k, v in types.items() if v is not None})
    inferred = {k: v for k, v in inferred.items() if v != dtypes.typeclass(None)}

    def part(indices: list[int], suffix: str, extra_inputs: list[str], extra_outputs: list[str]) -> tn.TaskletNode:
        statements = [astutils.copy_tree(code.statements[k]) for k in indices]
        assigned = {code.targets[k] for k in indices}
        declared = [astutils.copy_tree(d) for n, d in code.declarations.items() if n in assigned]
        read = set().union(*(code.reads[k] for k in indices)) if indices else set()
        inputs = sorted((code.inputs & read) | set(extra_inputs))
        outputs = sorted((code.outputs & assigned) | set(extra_outputs))
        new = copy.deepcopy(tasklet)
        new.label = f"{tasklet.label}_{suffix}"
        new.code = CodeBlock(declared + statements, dtypes.Language.Python)
        for connector in list(new.in_connectors):
            new.remove_in_connector(connector)
        for connector in list(new.out_connectors):
            new.remove_out_connector(connector)
        for connector in inputs:
            new.add_in_connector(connector, types.get(connector) or inferred.get(connector))
        for connector in outputs:
            new.add_out_connector(connector, types.get(connector) or inferred.get(connector))
        in_memlets = {c: copy.deepcopy(node.in_memlets[c]) for c in inputs if c in node.in_memlets}
        out_memlets = {c: copy.deepcopy(node.out_memlets[c]) for c in outputs if c in node.out_memlets}
        return tn.TaskletNode(node=new, in_memlets=in_memlets, out_memlets=out_memlets)

    a = part(sorted(first), "a", [], [n for n in passed if n not in code.outputs])
    b = part(second, "b", passed, [])
    for name in passed:
        memlet = copy.deepcopy(node.out_memlets[name]) if name in code.outputs else store(name, inferred.get(name))
        if name not in code.outputs:
            a.out_memlets[name] = memlet
        b.in_memlets[name] = copy.deepcopy(memlet)
    return a, b
