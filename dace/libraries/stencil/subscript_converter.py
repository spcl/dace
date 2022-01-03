# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import astunparse
from collections import defaultdict
from typing import Tuple


class SubscriptConverter(ast.NodeTransformer):
    """
    Finds all subscript accesses using constant indices in the given code, and
    renames them to a connector name that is not indexed, e.g.:

    a[0, 1, 0] is renamed to a_0_1_0
    b[-1, 0] is renamed to b_m1_0

    These mappings can be accessed in the `mapping` property, which returns a
    dictionary mapping the index tuple to the connector name, e.g.:

    {
      "a": {
        (0, 1, 0): "a_0_1_0"
      },
      "b": {
        (-1, 0): "b_m1_0
      }
    }
    """
    def __init__(self, offset: Tuple[int] = None, dtype=None):
        """
        :param offset: Apply the given offset tuple to every index found.
        :param dtype: Data type of constants found to enforce that the right
                      type is used (e.g., that 1.0 is interpreted as float32).
        """
        self._mapping = defaultdict(dict)
        self.offset = offset
        self.dtype = dtype

    @property
    def mapping(self):
        return self._mapping

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

        self._mapping[varname][index_tuple] = index_str

        return index_str

    def visit_Subscript(self, node: ast.Subscript):
        if not isinstance(node.value, ast.Name):
            raise TypeError('Only subscripts of variables are supported')

        varname = node.value.id

        # This can be a bunch of different things, varying between Python 3.8
        # and Python 3.9, so try hard to unpack it into an index we can use.
        index_tuple = node.slice
        if isinstance(index_tuple, (ast.Subscript, ast.Index)):
            index_tuple = index_tuple.value
        if isinstance(index_tuple, (ast.Constant, ast.Num)):
            index_tuple = (index_tuple, )
        if isinstance(index_tuple, ast.Tuple):
            index_tuple = index_tuple.elts
        index_tuple = tuple(ast.literal_eval(t) for t in index_tuple)

        index_str = self.convert(varname, index_tuple)

        return ast.copy_location(ast.Name(id=index_str), node)

    def visit_Constant(self, node: ast.Constant):
        if self.dtype is not None:
            return ast.copy_location(ast.Name(id=f"dace.{self.dtype.type.__name__}({node.value})"), node)
        else:
            return self.generic_visit(node)
