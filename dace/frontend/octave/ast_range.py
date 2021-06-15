# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

from .ast_node import AST_Node


class AST_RangeExpression(AST_Node):
    def __init__(self, context, lhs, rhs):
        AST_Node.__init__(self, context)
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return "AST_RangeExpression(" + str(self.lhs) + ", " + str(
            self.rhs) + ")"

    def get_children(self):
        L = [self.lhs, self.rhs]
        return [x for x in L if x is not None]

    def get_dims(self):
        from .ast_values import AST_Constant
        if isinstance(self.lhs, AST_Constant) and isinstance(
                self.rhs, AST_Constant):
            l = self.rhs.get_value() - self.lhs.get_value() + 1
            return [1, l]
        else:
            print("Dimensionality of " + str(self) + " cannot be inferred")
            return [1, 1]

    def get_basetype(self):
        return dace.dtypes.float64

    def replace_child(self, old, new):
        if old == self.lhs:
            self.lhs = new
            return
        if old == self.rhs:
            self.rhs = new
            return
        raise ValueError("The child " + str(old) + " is not a child of " +
                         str(self))

    def specialize(self):
        return None

    def generate_code(self, sdfg, state):
        # If lhs and rhs are constant, generate a matrix
        from .ast_values import AST_Constant
        from .ast_matrix import AST_Matrix_Row, AST_Matrix
        if isinstance(self.lhs, AST_Constant) and isinstance(
                self.rhs, AST_Constant):
            lval = self.lhs.get_value()
            rval = self.rhs.get_value()
            vals = [
                AST_Constant(self.context, v)
                for v in list(range(lval, rval + 1))
            ]
            new = AST_Matrix(self.context,
                             [AST_Matrix_Row(self.context, vals)])
            new.parent = self.parent
            new.prev = self.prev
            new.next = self.next
            new.generate_code(sdfg, state)
        else:
            raise NotImplementedError(
                "Code generation for Range with non-constant bounds not "
                "implemented")

    __str__ = __repr__
