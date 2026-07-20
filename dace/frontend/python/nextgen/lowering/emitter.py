# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Closed schedule-tree emitter for the next-generation Python frontend.

The emitter is the only way lowering rules add nodes to the tree, and it only
accepts the *frontend-legal* subset of schedule tree node types. In
particular, :class:`~dace.sdfg.analysis.schedule_tree.treenodes.StatementNode`
is not legal: a statement either lowers to a semantic node or becomes a fully
specified :class:`PythonCallbackNode`. Attempting to emit anything else is an
immediate frontend bug, not a runtime warning.
"""
from contextlib import contextmanager
from typing import FrozenSet, Iterator, List, Type

from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python.nextgen.common import TreeVerificationError

#: Schedule tree node types the frontend is allowed to produce.
FRONTEND_LEGAL_NODES: FrozenSet[Type[tn.ScheduleTreeNode]] = frozenset({
    tn.TaskletNode,
    tn.LibraryCall,
    tn.ReplacementCallNode,
    tn.CopyNode,
    tn.ViewNode,
    tn.RefSetNode,
    tn.AssignNode,
    tn.DynScopeCopyNode,
    tn.MapScope,
    tn.ConsumeScope,
    tn.LoopScope,
    tn.ForScope,
    tn.WhileScope,
    tn.IfScope,
    tn.ElifScope,
    tn.ElseScope,
    tn.BreakNode,
    tn.ContinueNode,
    tn.ReturnNode,
    tn.RaiseNode,
    tn.PythonCallbackNode,
    tn.ReassignExternalNode,
    tn.FunctionCallScope,
    tn.SDFGCallNode,
})


class TreeEmitter:
    """Appends frontend-legal nodes to the current scope of a schedule tree."""

    def __init__(self, root: tn.ScheduleTreeRoot):
        self.root = root
        self._scope_stack: List[tn.ScheduleTreeScope] = [root]

    @property
    def current_scope(self) -> tn.ScheduleTreeScope:
        return self._scope_stack[-1]

    @property
    def in_dataflow_scope(self) -> bool:
        """Whether emission currently targets (a descendant of) a dataflow
        scope (map/consume), where interstate mechanisms like scalar-to-symbol
        promotion cannot apply."""
        return any(isinstance(scope, tn.DataflowScope) for scope in self._scope_stack)

    def emit(self, node: tn.ScheduleTreeNode) -> tn.ScheduleTreeNode:
        """
        Append a node to the current scope.

        :raises TreeVerificationError: If the node type is not frontend-legal.
        """
        if type(node) not in FRONTEND_LEGAL_NODES:
            raise TreeVerificationError(f'Attempted to emit non-frontend-legal node type {type(node).__name__}. '
                                        'This is a frontend bug: statements must lower to semantic nodes or '
                                        'explicit Python callbacks.')
        self.current_scope.add_child(node)
        return node

    @contextmanager
    def scope(self, scope_node: tn.ScheduleTreeScope) -> Iterator[tn.ScheduleTreeScope]:
        """Emit a scope node and make it the current emission target."""
        self.emit(scope_node)
        self._scope_stack.append(scope_node)
        try:
            yield scope_node
        finally:
            self._scope_stack.pop()

    def checkpoint(self) -> int:
        """A rollback mark for the current scope (see :meth:`rollback`)."""
        return len(self.current_scope.children)

    def rollback(self, mark: int) -> None:
        """
        Discard every node emitted into the current scope since ``mark`` was
        taken. Used by rules that must retract partially emitted structures
        (e.g., an if-chain whose join cannot be merged) before falling back to
        a callback. Containers registered meanwhile stay in the repository;
        orphaned containers are harmless.
        """
        del self.current_scope.children[mark:]
