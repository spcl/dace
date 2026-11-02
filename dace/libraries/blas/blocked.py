# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Matrix operands stored in blocks of columns, for the matrix-product library nodes ``MatMul`` and ``Gemm``.

A matrix ``X[R, C]`` stored in blocks of ``b`` columns is the array ``S[C / b, R, b]``, with ``S[c // b, r, c % b] =
X[r, c]``: block ``j`` holds columns ``j * b`` to ``j * b + b - 1`` of all rows, one row after the other. A batch of
matrices ``X[..., R, C]`` is stored as ``S[..., C / b, R, b]``. Rows ``r0:r1`` and whole blocks of columns ``c0:c1`` of
``X`` are the subset ``S[..., c0 / b:c1 / b, r0:r1, 0:b]``, in which the ``b`` columns of consecutive rows are
contiguous. A product that reads ``b`` columns of many rows at a time, as the thread-block products do (see
``gemm_thread_block``), then reads contiguous memory rather than a short piece of every row: inference engines pack
their weights so.

An operand connector of a node in the node's ``blocked`` set is such an operand: its memlet addresses the three innermost
dimensions of its array, ``[C / b, R, b]`` (single indices in any others), and the node uses the matrix ``X[R, C]``.
Implementations that read blocked operands do so in their own expansion; the others copy them into transients in the
usual layout first (see ``dace.libraries.blas.indexed.expand_by_copies``).

``dace.transformation.passes.BlockMatrixOperands`` stores arguments of an SDFG in this layout, and the ``Pack`` library
node stores a matrix in it.
"""

from dace import subsets
from dace.libraries.blas.blas_helpers import matrix_view


def blocked_shape(shape, block) -> list:
    """The shape of a matrix (or a batch of them) of shape ``shape``, stored in blocks of ``block`` columns."""
    *batch, rows, columns = shape
    return [*batch, columns // block, rows, block]


def blocked_view(subset: subsets.Subset) -> tuple[list, list[int]]:
    """The view of a blocked operand: the size of its memlet's three innermost dimensions (blocks, rows and the
    columns of a block), and those dimensions."""
    size = subset.size()
    if len(size) < 3:
        raise ValueError(f"A blocked operand addresses three dimensions (blocks, rows, columns), not {subset}")
    dims = list(range(len(size) - 3, len(size)))
    return [size[d] for d in dims], dims


def operand_view(subset: subsets.Subset, blocked: bool) -> tuple[list, list[int]]:
    """The view of an operand that a node works with: its blocked view (see ``blocked_view``), or its matrix view."""
    return blocked_view(subset) if blocked else matrix_view(subset)


def matrix_size(subset: subsets.Subset, blocked: bool) -> list:
    """The size of the matrix of an operand: ``[R, C]`` for a blocked operand, otherwise its matrix view."""
    if not blocked:
        return matrix_view(subset)[0]
    (blocks, rows, columns), _ = blocked_view(subset)
    return [rows, blocks * columns]


__all__ = ["blocked_shape", "blocked_view", "operand_view", "matrix_size"]
