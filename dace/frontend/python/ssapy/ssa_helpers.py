

from ast import dump, parse


def print_stmt(code: str) -> None:
    full_ast = parse(code)
    for s in full_ast.body:
        print(dump(s))


def print_exp(code: str) -> None:
    full_ast = parse(code)
    exp_stmt = full_ast.body[0]
    exp = exp_stmt.value
    print(dump(exp))


class Counting_UID:

    def __init__(self) -> None:

        self.counter = 0

    def __call__(self, name: str) -> str:

        self.counter += 1
        uid = name + "_" + str(self.counter)

        return uid


class EnclosingLoop:

    def __init__(self, ast_visitor, loop_node):

        self.ast_visitor = ast_visitor
        self.loop_node = loop_node
    
    def __enter__(self):

        ast_visitor = self.ast_visitor
        loop_node = self.loop_node

        # save old state
        self.prev_loop = getattr(ast_visitor, 'current_loop', None)
        self.prev_break = getattr(ast_visitor, 'has_break', False)

        # create new state
        ast_visitor.current_loop = loop_node
        ast_visitor.has_break = False
    
    def __exit__(self, exc_type,exc_value, exc_traceback):

        # restore old state
        self.ast_visitor.current_loop = self.prev_loop
        self.ast_visitor.has_break = self.prev_break
