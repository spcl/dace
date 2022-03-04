import ast
import astunparse
import dace

class FLOPCounter(ast.NodeVisitor):
    def __init__(self):
        self.count = 0

    @staticmethod
    def count(tasklet: dace.nodes.Tasklet) -> int:
        code = tasklet.code.code

        ctr = FLOPCounter()
        if isinstance(code, (tuple, list)):
            for stmt in code:
                ctr.visit(stmt)
        elif isinstance(code, str):
            ctr.visit(ast.parse(code))
        else:
            ctr.visit(code)
        return ctr.count

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            raise NotImplementedError("MatMult op count requires shape " "inference")
        self.count += 1
        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        self.count += 1
        return self.generic_visit(node)

    def visit_Call(self, node):
        fname = astunparse.unparse(node.func)[:-1]
        if fname not in PYFUNC_TO_ARITHMETICS:
            print('WARNING: Unrecognized python function "%s"' % fname)
            return self.generic_visit(node)
        self.count += PYFUNC_TO_ARITHMETICS[fname]
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        return self.visit_BinOp(node)

    def visit_For(self, node):
        raise NotImplementedError

    def visit_While(self, node):
        raise NotImplementedError

# TODO: Fill with reasonable values
PYFUNC_TO_ARITHMETICS = {
    "math.sqrt": 3 * 3 + 5, # Newton's method: iter_num * update_formula + initial_guess
    "math.square": 1,
    "math.exp": 6,
    "math.log": 6,
    "math.sin": 6,
    "math.cos": 6,
    "math.tan": 6,
    "math.atan": 6,
    "math.atan2": 6,
    "math.sinh": 6,
    "math.cosh": 6,
    "math.tanh": 6,
    "math.ceil": 1,
    "math.floor": 1,
    "math.absolute": 1,
}
