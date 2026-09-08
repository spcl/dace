# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Removal of the tasklets that materialize a compile-time scalar nothing reads.

A temporary holding a symbolic value is both *recorded* and *materialized*: the
record lets a consumer's subset carry the arithmetic, while the scalar is there
for a consumer that reads the value back as data (see
:func:`~..rules.assign.lower_name_assignment` for why both). When every
consumer took the record instead -- ``range(2 * N)`` folds its bound
symbolically into the loop header -- what is left is a tasklet computing a
number no one looks at.

Simplification removes such a tasklet, but an SDFG is also analyzed and
compiled unsimplified, and there it counts as real work: the work/depth
analysis charged ``3 * N**2 + 1`` for a program that does ``3 * N**2``.
"""
import ast
from typing import Any, Dict, List, Set

from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def drop_unread_symbolic_temporaries(root: tn.ScheduleTreeRoot, symbolic_values: Dict[str, Any]) -> None:
    """
    Delete the materialization of every recorded compile-time scalar that no
    node in the tree reads.

    :param root: The lowered tree, modified in place.
    :param symbolic_values: The compile-time value recorded per container name
                            (``ProgramContext.symbolic_scalar_values``); only
                            these containers are considered, since only they
                            have a record a consumer could have used instead.
    """
    candidates = {name for name in symbolic_values if name in root.containers}
    candidates -= set(root.arg_names)
    candidates = {name for name in candidates if root.containers[name].transient}
    if not candidates:
        return

    read = _read_names(root, candidates)
    for name in candidates - read:
        writers = _sole_writers(root, name)
        if writers is None:
            continue
        for scope, node in writers:
            scope.children.remove(node)
        del root.containers[name]


def _read_names(root: tn.ScheduleTreeRoot, candidates: Set[str]) -> Set[str]:
    """
    Which of ``candidates`` the tree reads anywhere: as the data of a memlet, as
    a symbol inside one's subset, or by name in a code block a scope evaluates
    (a loop header, a branch condition).

    Subsets matter as much as memlet data. An indirect write spells its index
    with the container's own name (``a(dyn) [0:20, k] = ...``), so a scalar can
    be "read" without ever being an input memlet -- and deleting it there left
    a subset naming something no longer in the program.

    :param root: The lowered tree.
    :param candidates: The container names to look for.
    :return: The subset that is read.
    """
    read: Set[str] = set()
    for node in root.preorder_traversal():
        if not isinstance(node, tn.ScheduleTreeScope):
            # A scope's memlets are its children's, which the traversal reaches
            # in turn; asking the scope would propagate them for nothing.
            read.update(memlet.data for memlet in node.input_memlets())
            for memlet in (*node.input_memlets(), *node.output_memlets()):
                # The name a memlet WRITES is not a read of it; the symbols in
                # its subset are, whichever direction the memlet goes.
                symbols = {str(symbol) for symbol in memlet.free_symbols} - {memlet.data}
                read.update(symbols & candidates)
        for text in _code_strings(node):
            read.update(_names_in(text) & candidates)
    return read


def _code_strings(node: tn.ScheduleTreeNode) -> List[str]:
    """Every source string ``node`` carries that names things by identifier."""
    texts: List[str] = []
    for value in vars(node).values():
        if isinstance(value, CodeBlock):
            texts.append(value.as_string)
    loop = getattr(node, 'loop', None)
    if loop is not None:
        for attribute in ('init_statement', 'loop_condition', 'update_statement'):
            block = getattr(loop, attribute, None)
            if isinstance(block, CodeBlock):
                texts.append(block.as_string)
    return texts


def _names_in(text: str) -> Set[str]:
    """The identifiers appearing in a source string, or nothing if it will not parse."""
    try:
        tree = ast.parse(text, mode='exec')
    except SyntaxError:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _sole_writers(root: tn.ScheduleTreeRoot, name: str):
    """
    The (scope, node) pairs writing ``name``, or None when removing them would
    not be safe.

    Only a tasklet that writes nothing else qualifies: anything with another
    output, or a node whose meaning is not "compute this value" (a callback, a
    library call, a nested SDFG call), has to stay.

    :param root: The lowered tree.
    :param name: The container to find the writers of.
    :return: A list of (parent scope, node) pairs, or None.
    """
    writers = []
    for node in root.preorder_traversal():
        if isinstance(node, tn.ScheduleTreeScope):
            continue
        written = {memlet.data for memlet in node.output_memlets()}
        if name not in written:
            continue
        if not isinstance(node, tn.TaskletNode) or written != {name}:
            return None
        tasklet = node.node
        attached = (tasklet.code_global, tasklet.code_init, tasklet.code_exit)
        if tasklet.side_effects or any(getattr(block, 'as_string', block) for block in attached):
            return None
        scope = node.parent
        if scope is None or node not in scope.children:
            return None
        writers.append((scope, node))
    return writers or None
