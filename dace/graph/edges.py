import ast
import copy
import enum
import re

import dace
from dace import types
from dace.graph.graph import Edge
from dace.frontend.python import astutils
from dace.properties import Property, CodeProperty, make_properties


def assignments_from_string(astr):
    """ Returns a dictionary of assignments from a semicolon-delimited 
        string of expressions. """

    result = {}
    for aitem in astr.split(';'):
        aitem = aitem.strip()
        m = re.search(r'([^=\s]+)\s*=\s*([^=]+)', aitem)
        result[m.group(1)] = m.group(2)

    return result


def assignments_to_string(assdict):
    """ Returns a semicolon-delimited string from a dictionary of assignment 
        expressions. """
    return '; '.join(['%s=%s' % (k, v) for k, v in assdict.items()])


@make_properties
class InterstateEdge(object):
    """ An SDFG state machine edge. These edges can contain a condition     
        (which may include data accesses for data-dependent decisions) and
        zero or more assignments of values to inter-state variables (e.g.,
        loop iterates).
    """

    assignments = Property(
        dtype=dict,
        desc="Assignments to perform upon transition (e.g., 'x=x+1; y = 0')",
        from_string=assignments_from_string,
        to_string=assignments_to_string)
    condition = CodeProperty(desc="Transition condition")
    language = Property(enum=types.Language, default=types.Language.Python)

    def __init__(self, condition=None, assignments=None):

        if condition is None:
            condition = ast.parse("1").body[0]

        if assignments is None:
            assignments = {}

        self.condition = condition
        self.assignments = assignments

        self._dotOpts = {"minlen": 3, "color": "blue", "fontcolor": "blue"}

    def is_unconditional(self):
        """ Returns True if the state transition is unconditional. """
        return (self.condition == None or InterstateEdge.condition.to_string(
            self.condition).strip() == "1")

    def condition_sympy(self):
        cond_ast = self.condition
        return symbolic.pystr_to_symbolic(astutils.unparse(cond_ast))

    def condition_symbols(self):
        return dace.symbolic.symbols_in_ast(self.condition[0])

    def toJSON(self, indent=0):
        json = str(self.label)
        # get rid of newlines (why are they there in the first place?)
        json = re.sub(r"\n", " ", json)
        return "\"" + json + "\""

    @property
    def label(self):
        assignments = ','.join(
            ['%s=%s' % (k, v) for k, v in self.assignments.items()])

        # Edge with assigment only (no condition)
        if astutils.unparse(self.condition) == '1':
            # Edge without conditions or assignments
            if len(self.assignments) == 0:
                return ''
            return assignments

        # Edge with condition only (no assignment)
        if len(self.assignments) == 0:
            return astutils.unparse(self.condition)

        # Edges with assigments and conditions
        return assignments + '; ' + astutils.unparse(self.condition)

    @property
    def dotOpts(self):
        result = {}
        result.update(self._dotOpts)
        result.update({'label': self.label})
        return result


class RedirectEdge(InterstateEdge):
    """ An inter-state edge type used for rendering self-looping edges
        on graph clusters in GraphViz. """

    def __init__(self):
        super(RedirectEdge, self).__init__()
        self._dotOpts["arrowhead"] = "none"


###############################################################################
# Various classes to facilitate the detection of control flow elements (e.g.,
# `for`, `if`, `while`) from state machines in SDFGs.


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

    scope = Property(dtype=LoopScope)
    edge = Property(dtype=Edge)

    def __init__(self, scope, edge, *args, **kwargs):
        self.scope = scope
        self.edge = edge
        scope.assignment = self
        super().__init__(*args, **kwargs)


@make_properties
class LoopEntry(ControlFlow):

    scope = Property(dtype=LoopScope)
    edge = Property(dtype=Edge)

    def __init__(self, scope, edge, *args, **kwargs):
        self.scope = scope
        self.edge = edge
        scope.entry = self
        super().__init__(*args, **kwargs)


@make_properties
class LoopExit(ControlFlow):

    scope = Property(dtype=LoopScope)
    edge = Property(dtype=Edge)

    def __init__(self, scope, edge, *args, **kwargs):
        self.scope = scope
        self.edge = edge
        scope.exit = self
        super().__init__(*args, **kwargs)


@make_properties
class LoopBack(ControlFlow):

    scope = Property(dtype=LoopScope)
    edge = Property(dtype=Edge)

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

    entry = Property()
    exit = Property()

    def __init__(self, entry, exit):
        self.entry = entry
        self.exit = exit
        self.then_scope = None
        self.else_scope = None


@make_properties
class IfEntry(ControlFlow):

    scope = Property(dtype=ControlFlowScope)
    edge = Property(dtype=Edge)

    def __init__(self, scope, edge, *args, **kwargs):
        self.scope = scope
        self.edge = edge
        scope.entry = self
        super().__init__(*args, **kwargs)


@make_properties
class IfExit(ControlFlow):

    scope = Property(dtype=ControlFlowScope)
    edge = Property(dtype=Edge)

    def __init__(self, scope, edge, *args, **kwargs):
        self.scope = scope
        self.edge = edge
        scope.exit = self
        super().__init__(*args, **kwargs)


@make_properties
class IfThenScope(ControlFlowScope):

    if_then_else = Property(dtype=IfThenElse)
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

    if_then_else = Property(dtype=IfThenElse)
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
