import ast
import astunparse
import dace

class FLOPCounter(ast.NodeVisitor):
    def __init__(self):
        self.add = 0
        self.mul = 0

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

        return {"+": ctr.add, "*": ctr.mul}

    def visit_Subscript(self, node: ast.Subscript):
        # Not counting index accesses as FLOP
        pass

    def visit_BinOp(self, node):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            self.add += 1
        elif isinstance(node.op, (ast.Mult, ast.Div)):
            self.mul += 1
        else:
            raise NotImplementedError()

        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        return self.generic_visit(node)

    def visit_Call(self, node):
        fname = astunparse.unparse(node.func)[:-1]
        if fname not in PYFUNC_TO_ARITHMETICS:
            print('WARNING: Unrecognized python function "%s"' % fname)
            return self.generic_visit(node)

        flop = PYFUNC_TO_ARITHMETICS[fname]
        self.add += flop["+"]
        self.mul += flop["*"]
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        return self.visit_BinOp(node)

    def visit_For(self, node):
        raise NotImplementedError

    def visit_While(self, node):
        raise NotImplementedError

# TODO: Fill with reasonable values
PYFUNC_TO_ARITHMETICS = {
    "dace.float64": {"+": 0, "*": 0},
    "math.sqrt": {"+": 5 * 1, "*": 5 * 1}, # Method: Newton's method with 5 iterations
    "math.square": {"+": 0, "*": 1},
    "math.exp": {"+": 0, "*": 2 * 5 + 1}, # Method: Exponentiation by squaring for pow=5
    "math.log": {"+": 0, "*": 2 * 5 + 1},
    "math.sin": {"+": 0, "*": 2 * 5 + 1},
    "math.cos": {"+": 0, "*": 2 * 5 + 1},
    "math.tan": {"+": 0, "*": 2 * 5 + 1},
    "math.atan": {"+": 0, "*": 2 * 5 + 1},
    "math.atan2": {"+": 0, "*": 2 * 5 + 1},
    "math.sinh": {"+": 0, "*": 2 * 5 + 1},
    "math.cosh": {"+": 0, "*": 2 * 5 + 1},
    "math.tanh": {"+": 0, "*": 2 * 5 + 1},
    "math.ceil": {"+": 0, "*": 1},
    "math.floor": {"+": 0, "*": 1},
    "math.absolute": {"+": 0, "*": 1},
}
