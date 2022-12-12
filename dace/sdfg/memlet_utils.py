# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import ast
from dace.frontend.python import astutils, memlet_parser
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import graph as gr
from dace.sdfg import utils as sdutil
from dace.properties import CodeBlock
from dace import data, subsets, Memlet
from typing import Callable, Dict, Optional, Set, Union


class MemletReplacer(ast.NodeTransformer):
    """
    Iterates over all memlet expressions (name or subscript with matching array in SDFG) in a code block.
    The callable can also return another memlet to replace the current one.
    """

    def __init__(self,
                 arrays: Dict[str, data.Data],
                 process: Callable[[Memlet], Union[Memlet, None]],
                 array_filter: Optional[Set[str]] = None) -> None:
        """
        Create a new memlet replacer.

        :param arrays: A mapping from array names to data descriptors.
        :param process: A callable that takes a memlet and returns a memlet or None.
        :param array_filter: An optional subset of array names to process.
        """
        self.process = process
        self.arrays = arrays
        self.array_filter = array_filter or self.arrays.keys()

    def _parse_memlet(self, node: Union[ast.Name, ast.Subscript]) -> Memlet:
        """
        Parses a memlet from a subscript or name node.

        :param node: The node to parse.
        :return: The parsed memlet.
        """
        # Get array name
        if isinstance(node, ast.Name):
            data = node.id
        elif isinstance(node, ast.Subscript):
            data = node.value.id
        else:
            raise TypeError('Expected Name or Subscript')

        # Parse memlet subset
        array = self.arrays[data]
        subset, newaxes, _ = memlet_parser.parse_memlet_subset(array, node, self.arrays)
        if newaxes:
            raise NotImplementedError('Adding new axes to memlets is not supported')

        return Memlet(data=data, subset=subset)

    def _memlet_to_ast(self, memlet: Memlet) -> ast.Subscript:
        """
        Converts a memlet to a subscript node.

        :param memlet: The memlet to convert.
        :return: The converted node.
        """
        return ast.parse(f'{memlet.data}[{memlet.subset}]').body[0].value

    def _replace(self, node: Union[ast.Name, ast.Subscript]) -> ast.Subscript:
        cur_memlet = self._parse_memlet(node)
        new_memlet = self.process(cur_memlet)
        if new_memlet is None:
            return node

        new_node = self._memlet_to_ast(new_memlet)
        return ast.copy_location(new_node, node)

    def visit_Name(self, node: ast.Name):
        if node.id in self.array_filter:
            return self._replace(node)
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in self.array_filter:
            return self._replace(node)
        return self.generic_visit(node)
