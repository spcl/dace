from ast_components import *


def iter_fields(node):
    """
    Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
    that is present on *node*.
    """
    if not hasattr(node, "_fields"):
        a = 1
    for field in node._fields:
        try:
            yield field, getattr(node, field)
        except AttributeError:
            pass


def iter_child_nodes(node):
    """
    Yield all direct child nodes of *node*, that is, all fields that are nodes
    and all items of fields that are lists of nodes.
    """
    #print("CLASS: ",node.__class__)
    #if isinstance(node,DeclRefExpr):
    #print("NAME: ", node.name)

    for name, field in iter_fields(node):
        #print("NASME:",name)
        if isinstance(field, Node):
            yield field
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, Node):
                    yield item


class NodeVisitor(object):
    def visit(self, node):
        # print(node.__class__.__name__)
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, Node):
                        self.visit(item)
            elif isinstance(value, Node):
                self.visit(value)


class NodeTransformer(NodeVisitor):
    def as_list(self, x):
        if isinstance(x, list):
            return x
        if x is None:
            return []
        return [x]

    def generic_visit(self, node):
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, Node):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, Node):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, Node):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class FindFunctionAndSubroutines(NodeVisitor):
    def __init__(self):
        self.nodes: List[Name_Node] = []

    def visit_Subroutine_Subprogram_Node(self,
                                         node: Subroutine_Subprogram_Node):
        self.nodes.append(node.name)

    def visit_Function_Subprogram_Node(self, node: Function_Subprogram_Node):
        self.nodes.append(node.name)


class FindInputs(NodeVisitor):
    def __init__(self):
        self.nodes: List[Name_Node] = []

    def visit_Name_Node(self, node: Name_Node):
        self.nodes.append(node)

    def visit_Array_Subscript_Node(self, node: Array_Subscript_Node):
        self.nodes.append(node.name)

    def visit_BinOp_Node(self, node: BinOp_Node):
        if node.op == "=":
            if isinstance(node.lval, Name_Node):
                pass
            elif isinstance(node.lval, Array_Subscript_Node):
                for i in node.lval.indices:
                    self.visit(i)

        else:
            self.visit(node.lval)
        self.visit(node.rval)


class FindOutputs(NodeVisitor):
    def __init__(self):
        self.nodes: List[Name_Node] = []

    def visit_BinOp_Node(self, node: BinOp_Node):
        if node.op == "=":
            if isinstance(node.lval, Name_Node):
                self.nodes.append(node.lval)
            elif isinstance(node.lval, Array_Subscript_Node):
                self.nodes.append(node.lval.name)
            self.visit(node.rval)


class CallToArray(NodeTransformer):
    def __init__(self, funcs=None):
        if funcs is None:
            funcs = []
        self.funcs = funcs
        self.excepted_funcs = [
            "malloc", "exp", "pow", "sqrt", "cbrt", "max", "abs", "min",
            "dace_sum", "dace_sign", "tanh"
        ]

    def visit_Call_Expr_Node(self, node: Call_Expr_Node):

        if node.name.name in self.excepted_funcs or node.name in self.funcs:
            processed_args = []
            for i in node.args:
                arg = CallToArray(self.funcs).visit(i)
                processed_args.append(arg)
            node.args = processed_args
            return node
        indices = [CallToArray(self.funcs).visit(i) for i in node.args]
        return Array_Subscript_Node(name=node.name, indices=indices)
