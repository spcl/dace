# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import astunparse
from collections import defaultdict


class SubscriptConverter(ast.NodeTransformer):
    def __init__(self, offset=None, dtype=None):
        self.names = defaultdict(dict)
        self.offset = offset
        self.dtype = dtype

    def convert(self, varname, index_tuple):

        # Add offset to last index
        if self.offset:
            index_tuple = tuple(i + o for i, o in zip(index_tuple, self.offset))

        # Remove extraneous symbols
        index_str = ''.join(c for c in str(index_tuple) if c not in '( )')

        # Replace tuple and negative symbols
        index_str = index_str.replace(',', '_')
        index_str = index_str.replace('-', 'm')

        # Add variable name
        index_str = varname + '_' + index_str

        self.names[varname][index_tuple] = index_str

        return index_str

    def visit_Subscript(self, node: ast.Subscript):
        if not isinstance(node.value, ast.Name):
            raise TypeError('Only subscripts of variables are supported')

        varname = node.value.id
        index_tuple = ast.literal_eval(node.slice.value)
        try:
            len(index_tuple)
        except TypeError:
            # Turn into a tuple
            index_tuple = (index_tuple, )

        index_str = self.convert(varname, index_tuple)

        return ast.copy_location(ast.Name(id=index_str), node)

    def visit_Constant(self, node: ast.Constant):
        if self.dtype is not None:
            return ast.copy_location(
                ast.Name(id=f"dace.{self.dtype.type.__name__}({node.value})"),
                node)
        else:
            return self.generic_visit(node)
