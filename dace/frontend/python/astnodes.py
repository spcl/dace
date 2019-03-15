""" Support classes for the DaCe Python AST parser. """

from collections import OrderedDict
from copy import deepcopy as dcpy

from dace import data, types
from dace.frontend.python import astutils


class _Node(object):
    """ SDFG AST node class, generated from the DaCe Python AST parser. """

    def __init__(self, name, node_ast):
        self.name = name

        # Maps: {local variable name: array subscript expression (AST node)}
        self.inputs = OrderedDict()

        # Maps: {local variable name: array subscript expression (AST node)}
        self.outputs = OrderedDict()

        # All variables in the parent scope + current scope
        # Maps: {variable name: value AST node}
        self.globals = OrderedDict()

        # All local variables defined in this scope
        # Maps: {local variable name: value AST node}
        self.locals = OrderedDict()

        # Maps: {transient array name: data.Data}
        self.transients = OrderedDict()

        # List of parameter names
        self.params = []

        # Parent _Node object
        self.parent = None

        # List of children _Node objects
        self.children = []

        # Is asynchronous
        self.is_async = False

        # Node AST
        self.ast = node_ast

    def __deepcopy__(self, memo):
        n = object.__new__(type(self))

        n.name = dcpy(self.name)
        n.inputs = dcpy(self.inputs)
        n.outputs = dcpy(self.outputs)
        n.globals = self.globals
        n.locals = dcpy(self.locals)
        n.transients = dcpy(self.transients)
        n.params = dcpy(self.params)
        n.parent = None
        n.children = []
        n.is_async = dcpy(self.is_async)

        return n

    # Returns the arrays local to this node's context
    def arrays(self):
        return OrderedDict([(k, v) for k, v in self.globals.items()
                            if isinstance(v, data.Data)])

    # Returns all arrays (children included)
    def all_arrays(self):
        result = self.arrays()
        for c in self.children:
            result.update(c.all_arrays())
        return result

    def dump(self, indent=0):
        print('    ' * indent + self.__class__.__name__ + ': ' + self.name)
        for c in self.children:
            c.dump(indent + 1)


class _ProgramNode(_Node):
    """ SDFG AST node class. """
    pass


# Dataflow nodes
class _DataFlowNode(_Node):
    """ Dataflow AST node superclass. """
    pass


class _ScopeNode(_DataFlowNode):
    """ Scope (map/consume) AST node superclass. """
    pass


class _MapNode(_ScopeNode):
    """ Map AST node type. """
    #def __init__(self, name, node_ast, range, )
    pass


class _ConsumeNode(_ScopeNode):
    """ Consume AST node type. """
    #def __init(self, name, node_ast, stream, ...)
    pass


class _TaskletNode(_DataFlowNode):
    """ Tasklet AST node type. """

    def __init__(self,
                 name,
                 node_ast,
                 language=types.Language.Python,
                 global_code=''):
        super(_TaskletNode, self).__init__(name, node_ast)
        self.language = language
        self.extcode = None
        self.gcode = global_code


class _EmptyTaskletNode(_TaskletNode):
    """ Empty Tasklet AST node type. """
    pass


class _NestedSDFGNode(_DataFlowNode):
    """ Nested SDFG AST node type. """

    def __init__(self, name, node_ast, sdfg):
        super(_NestedSDFGNode, self).__init__(name, node_ast)
        self.sdfg = sdfg


# Operation nodes
class _ReduceNode(_DataFlowNode):
    """ Reduce AST node type. """
    pass


# Control flow nodes
class _ControlFlowNode(_Node):
    """ Control-flow AST node superclass. """
    pass


class _IterateNode(_ControlFlowNode):
    """ Iteration (for-loop) AST node type. """
    pass


class _LoopNode(_ControlFlowNode):
    """ Loop (while-loop) AST node type. """
    pass


class _ConditionalNode(_ControlFlowNode):
    """ Conditional (if/else) AST node superclass. """
    pass


class _IfNode(_ConditionalNode):
    """ If conditional AST node type. """
    pass


class _ElseNode(_ConditionalNode):
    """ Else conditional AST node type. """
    pass


class _Memlet(object):
    """ AST Memlet type. Becomes an SDFG edge. """

    def __init__(self, data, data_name, attribute, num_accesses,
                 write_conflict_resolution, wcr_identity, subset,
                 vector_length, local_name, ast, array_dependencies):
        self.data = data  # type: Data
        self.dataname = data_name  # type: str
        self.attribute = attribute  # type: str
        self.num_accesses = num_accesses  # type: sympy math
        self.wcr = write_conflict_resolution  # type: ast._Lambda
        self.wcr_identity = wcr_identity  # type: memlet type or None
        self.subset = subset  # type: subsets.Subset
        self.veclen = vector_length  # type: int (in elements, default 1)
        self.local_name = local_name  # type: str
        self.ast = ast  # type: ast._AST
        self.otherdeps = array_dependencies  # type: dict(str, data.Data)

    def wcr_name(self):
        label = astutils.unparse(self.wcr.body)
        if self.wcr_identity is not None:
            label += ', id: ' + str(self.wcr_identity)
        return label
