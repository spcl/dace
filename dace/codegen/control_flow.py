# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Various classes to facilitate the detection of control flow elements (e.g.,
    `for`, `if`, `while`) from state machines in SDFGs. """

from dace.sdfg.graph import Edge
from dace.properties import Property, make_properties

###############################################################################


@make_properties
class ControlFlowScope:

    nodes_in_scope = Property(
        dtype=set,
        desc="Nodes contained in this scope, "
        "including entry and exit nodes, in topological order.")

    def __init__(self, nodes_in_scope):
        self.nodes_in_scope = nodes_in_scope

    def __contains__(self, node):
        return node in self.nodes_in_scope

    def __iter__(self):
        return iter(self.nodes_in_scope)


# make_properties will be called after adding cyclic class reference members
class LoopScope(ControlFlowScope):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assignment = None
        self.entry = None
        self.back = None
        self.exit = None


class ControlFlow:
    pass


@make_properties
class LoopAssignment(ControlFlow):

    scope = Property(dtype=LoopScope, allow_none=True)
    edge = Property(dtype=Edge, allow_none=True)

    def __init__(self, scope, edge, *args, **kwargs):
        self.scope = scope
        self.edge = edge
        scope.assignment = self
        super().__init__(*args, **kwargs)


@make_properties
class LoopEntry(ControlFlow):

    scope = Property(dtype=LoopScope, allow_none=True)
    edge = Property(dtype=Edge, allow_none=True)

    def __init__(self, scope, edge, *args, **kwargs):
        self.scope = scope
        self.edge = edge
        scope.entry = self
        super().__init__(*args, **kwargs)


@make_properties
class LoopExit(ControlFlow):

    scope = Property(dtype=LoopScope, allow_none=True)
    edge = Property(dtype=Edge, allow_none=True)

    def __init__(self, scope, edge, *args, **kwargs):
        self.scope = scope
        self.edge = edge
        scope.exit = self
        super().__init__(*args, **kwargs)


@make_properties
class LoopBack(ControlFlow):

    scope = Property(dtype=LoopScope, allow_none=True)
    edge = Property(dtype=Edge, allow_none=True)

    def __init__(self, scope, edge, *args, **kwargs):
        self.scope = scope
        self.edge = edge
        scope.back = self
        super().__init__(*args, **kwargs)


# These will be assigned when the various control flow objects are created
LoopScope.assignment = Property(dtype=LoopAssignment, allow_none=True)
LoopScope.entry = Property(dtype=LoopEntry, allow_none=True)
LoopScope.back = Property(dtype=LoopBack, allow_none=True)
LoopScope.exit = Property(dtype=LoopExit, allow_none=True)
LoopScope = make_properties(LoopScope)


# Extra meta-object binding together then and else scopes.
# make_properties will be called after adding cyclic class reference members
class IfThenElse:

    entry = Property(allow_none=True)
    exit = Property(allow_none=True)

    def __init__(self, entry, exit):
        self.entry = entry
        self.exit = exit
        self.then_scope = None
        self.else_scope = None


@make_properties
class IfEntry(ControlFlow):

    scope = Property(dtype=ControlFlowScope, allow_none=True)
    edge = Property(dtype=Edge, allow_none=True)

    def __init__(self, scope, edge, *args, **kwargs):
        self.scope = scope
        self.edge = edge
        scope.entry = self
        super().__init__(*args, **kwargs)


@make_properties
class IfExit(ControlFlow):

    scope = Property(dtype=ControlFlowScope, allow_none=True)
    edge = Property(dtype=Edge, allow_none=True)

    def __init__(self, scope, edge, *args, **kwargs):
        self.scope = scope
        self.edge = edge
        scope.exit = self
        super().__init__(*args, **kwargs)


@make_properties
class IfThenScope(ControlFlowScope):

    if_then_else = Property(dtype=IfThenElse, allow_none=True)
    entry = Property(dtype=IfEntry, allow_none=True)
    exit = Property(dtype=IfExit, allow_none=True)

    def __init__(self, if_then_else, *args, **kwargs):
        self.if_then_else = if_then_else
        if_then_else.then_scope = self
        self.entry = None
        self.exit = None
        super().__init__(*args, **kwargs)


@make_properties
class IfElseScope(ControlFlowScope):

    if_then_else = Property(dtype=IfThenElse, allow_none=True)
    entry = Property(dtype=IfEntry, allow_none=True)
    exit = Property(dtype=IfExit, allow_none=True)

    def __init__(self, if_then_else, *args, **kwargs):
        self.if_then_else = if_then_else
        if_then_else.else_scope = self
        self.entry = None
        self.exit = None
        super().__init__(*args, **kwargs)


# Cyclic class reference
IfThenElse.then_scope = Property(dtype=IfThenScope, allow_none=True)
IfThenElse.else_scope = Property(dtype=IfElseScope, allow_none=True)
IfThenElse = make_properties(IfThenElse)
