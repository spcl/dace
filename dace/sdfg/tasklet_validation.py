# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from typing import TYPE_CHECKING
from dace import data

if TYPE_CHECKING:
    from dace.sdfg.sdfg import SDFG
    from dace.memlet import Memlet


class ConnectorDimensionalityValidator(ast.NodeVisitor):
    """
    Checks whether a connector accessed with a subscript is accessed with the correct dimensionality.
    The SDFG IR specifies that:
      * A tasklet may access data containers only through their associated connectors (the
        same data container may have more than one connector that corresponds to it).
      * The connector is a view of the memlet that is connected to it, where all the scalar indices are collapsed
        (squeezed). This means that the access subscript must only use the range dimensions. For example:
        array ``A`` of size MxNxKxL has a memlet ``A[i, j:j+k, 0, 0:L]`` connected to connector ``a``; in a tasklet,
        accesses to the array must have two dimensions, and ``a[m, n]`` in the tasklet corresponds to accessing
        ``A[i, j+m, 0, n]``.
    """

    def __init__(self, in_edges: dict[str, 'Memlet'], out_edges: dict[str, 'Memlet'], sdfg: 'SDFG') -> None:
        self.edges = {**in_edges, **out_edges}
        self.arrays = {
            k: sdfg.arrays[v.data]
            for k, v in in_edges.items() if k is not None and isinstance(v, data.Array)
        }
        self.arrays.update({
            k: sdfg.arrays[v.data]
            for k, v in out_edges.items() if k is not None and isinstance(v, data.Array)
        })

    def visit_Subscript(self, node: ast.Subscript):
        # A connector we should check
        if isinstance(node.value, ast.Name) and node.value.id in self.arrays:
            # Decode slice
            if isinstance(node.slice, ast.Tuple):
                slices = node.slice.elts
            elif isinstance(node.slice, tuple):
                slices = node.slice
            else:
                slices = [node.slice]
            # Compute number of non-scalar dimensions in memlet
            nonscalar_dims = sum(1 if end != start else 0
                                 for start, end, _ in self.edges[node.value.id].subset.ndrange())
            # Validate length
            if len(slices) != nonscalar_dims:
                raise IndexError(f'Subscript expression "{ast.unparse(node)}" contains an invalid number of dimensions.'
                                 f' Expected {nonscalar_dims} non-scalar dimensions, got {len(slices)}')

        return self.generic_visit(node)
