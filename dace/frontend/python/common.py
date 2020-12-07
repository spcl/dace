# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from typing import Dict, Any

class DaceSyntaxError(Exception):
    def __init__(self, visitor, node: ast.AST, message: str):
        self.visitor = visitor
        self.node = node
        self.message = message

    def __str__(self):
        # Try to recover line and column
        try:
            line = self.node.lineno
            col = self.node.col_offset
        except AttributeError:
            line = 0
            col = 0

        return (self.message + "\n  in File " + str(self.visitor.filename) +
                ", line " + str(line) + ":" + str(col))


def inverse_dict_lookup(dict: Dict[str, Any], value: Any):
    """ Finds the first key in a dictionary with the input value. """
    for k, v in dict.items():
        if v == value:
            return k
    return None
