# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from .ast_node import AST_Node


class AST_NullStmt(AST_Node):
    def __init__(self, context):
        AST_Node.__init__(self, context)

    def get_children(self):
        return []

    def replace_child(self, old, new):
        raise ValueError("AST_NullStmt has no children")

    def generate_code(self, sdfg, state):
        pass

    def __repr__(self):
        return "AST_NullStmt()"


class AST_EndStmt(AST_Node):
    def __init__(self, context):
        AST_Node.__init__(self, context)

    def __repr__(self):
        return "AST_End()"

    def get_children(self):
        return []

    def replace_child(self, old, new):
        raise ValueError("This class does not have children")


class AST_Comment(AST_Node):
    def __init__(self, context, text):
        AST_Node.__init__(self, context)
        self.text = text

    def get_children(self):
        return []

    def replace_child(self, old, new):
        raise ValueError("AST_Comment has no children")

    def generate_code(self, sdfg, state):
        pass

    def __repr__(self):
        text = self.text
        text = text.encode("unicode_escape").decode("utf-8")
        return "AST_Comment(\"" + text + "\")"
