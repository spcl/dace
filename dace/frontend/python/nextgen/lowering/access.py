# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared helpers for resolving canonical data accesses into repository
containers, subsets, and connector-substituted tasklet code.
"""
import ast
import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

from dace import data, dtypes, subsets, symbolic
from dace.frontend.python import astutils
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.registry import LoweringState


@dataclass
class DataAccess:
    """A resolved read or write of a repository container."""
    container: str  #: Repository container name
    subset: subsets.Range  #: Accessed subset (in container index space)
    descriptor: data.Data
    #: Per-subset-dimension flags: True when the source index was slice-formed
    #: (the dimension survives in the NumPy result shape even at size 1),
    #: False when integer-indexed (dropped). None when the index form could
    #: not be analyzed (fall back to squeezing all size-1 dimensions).
    kept_dims: Optional[List[bool]] = None
    #: True when this access is a full-array pointer connector synthesized for
    #: an indirect (data-dependent-index) read: its subset must not be
    #: broadcast/indexed against an iteration space — the tasklet code indexes
    #: it directly (e.g. ``__in0[__ind1]``).
    indirect: bool = False

    @property
    def is_scalar_access(self) -> bool:
        return self.subset.num_elements() == 1

    @property
    def numpy_shape(self) -> List:
        """
        The NumPy-semantic result shape of this access: slice-formed
        dimensions are kept (``a[0:20, 1:2]`` → (20, 1)), integer-indexed
        dimensions are dropped (``a[0:20, 1]`` → (20,)). Empty for scalar
        element accesses.
        """
        if self.kept_dims is None:
            return nondegenerate_shape(self.subset)
        return [size for size, kept in zip(self.subset.size(), self.kept_dims) if kept]


def nondegenerate_shape(subset: subsets.Range) -> List:
    """The shape of a subset with size-1 dimensions squeezed out."""
    return [s for s in subset.size() if s != 1]


def _kept_dimensions(slice_node: ast.expr, ndim: int) -> Optional[List[bool]]:
    """
    Which subset dimensions a subscript keeps in its NumPy result shape:
    slice-formed indices keep their dimension, integer indices drop it, and
    unindexed trailing dimensions are kept. Returns None (unknown) for index
    forms this analysis does not model (``...``, ``None``/newaxis).
    """
    elements = list(slice_node.elts) if isinstance(slice_node, ast.Tuple) else [slice_node]
    if len(elements) > ndim:
        return None
    kept: List[bool] = []
    for element in elements:
        if isinstance(element, ast.Constant) and (element.value is Ellipsis or element.value is None):
            return None
        kept.append(isinstance(element, ast.Slice))
    kept.extend([True] * (ndim - len(kept)))
    return kept


def resolve_access(node: ast.expr, state: LoweringState) -> Optional[DataAccess]:
    """
    Resolve a canonical data reference (``Name``, structure member
    ``Attribute(Name)``, or a ``Subscript`` over either) to a repository data
    access. Returns None if the expression does not refer to a data container
    (e.g., a symbol or constant).
    """
    if isinstance(node, ast.Name):
        binding = state.context.resolve(node.id)
        if binding is None or binding.kind != 'container':
            return None
        descriptor = state.context.containers[binding.container]
        return DataAccess(binding.container, subsets.Range.from_array(descriptor), descriptor)
    if isinstance(node, ast.Attribute):
        return _member_access(node, state)
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name):
            binding = state.context.resolve(node.value.id)
            if binding is None or binding.kind != 'container':
                return None
            base_container = binding.container
            base_descriptor = state.context.containers[base_container]
        elif isinstance(node.value, ast.Attribute):
            member = _member_access(node.value, state)
            if member is None:
                return None
            base_container = member.container
            base_descriptor = member.descriptor
        else:
            return None
        expr = state.inference.parse_access(node)
        subset = expr.subset
        if isinstance(subset, subsets.Indices):
            subset = subsets.Range.from_indices(subset)
        if expr.arrdims:
            raise UnsupportedFeatureError('Advanced (array-valued) indexing is not supported yet',
                                          state.context.filename,
                                          node,
                                          category='data-dependent-subscript')
        scalar_reads, array_reads = data_dependent_container_names(subset, state.context)
        if array_reads:
            # An array element read inside the index (x[A_col[j]]): genuine
            # indirection, only supported by the explicit-tasklet rule.
            raise UnsupportedFeatureError(
                f'Data-dependent subscript index (reads {", ".join(sorted(array_reads))}) is not supported here',
                state.context.filename,
                node,
                category='data-dependent-subscript')
        if scalar_reads and state.emitter.in_dataflow_scope:
            # Scalar containers in subsets are legitimate at control-flow
            # level (scalar-to-symbol promotion turns them into symbols), but
            # inside a dataflow scope no such mechanism exists — the memlet
            # would reference a runtime scalar as if it were a symbol.
            raise UnsupportedFeatureError(
                f'Subscript index reads scalar container(s) {", ".join(sorted(scalar_reads))} inside a '
                'dataflow scope',
                state.context.filename,
                node,
                category='data-dependent-subscript')
        kept = None if expr.new_axes else _kept_dimensions(node.slice, len(subset.ranges))
        return DataAccess(base_container, subset, base_descriptor, kept_dims=kept)
    return None


def data_dependent_container_names(subset: subsets.Range, context) -> Tuple[set, set]:
    """
    Names referenced by a subset's index expressions that are themselves
    bound to data containers (rather than symbols), split into scalar
    containers and array reads -- e.g. the ``A_col`` in ``x[A_col[j]]``.

    The shared memlet parser (``dace.frontend.python.memlet_parser``)
    represents an inner subscript it cannot evaluate as a compile-time
    constant as an un-evaluated applied sympy function (e.g. ``A_col(j)``),
    so the true free symbols of the subset (``subsets.Range.free_symbols``,
    built from ``sympy.Expr.free_symbols``) do not include ``A_col`` -- only
    ``dace.symbolic.free_symbols_and_functions`` reports both the real
    symbols and any function heads.

    Used as a soundness guard: array reads in an index are genuine
    indirection and only the explicit-tasklet rule implements them (see
    ``lowering/rules/dataflow_explicit.py``); scalar containers are
    legitimate at control-flow level (scalar-to-symbol promotion applies)
    but not inside dataflow scopes.

    :return: A 2-tuple of (scalar container names, array read names).
    """
    names = set()
    for dim in subset.ranges:
        for component in dim:
            names |= symbolic.free_symbols_and_functions(component)
    scalar_reads = set()
    array_reads = set()
    for name in names:
        binding = context.resolve(name)
        if binding is None or binding.kind != 'container':
            continue
        descriptor = context.containers[binding.container]
        if isinstance(descriptor, data.Scalar):
            scalar_reads.add(name)
        else:
            array_reads.add(name)
    return scalar_reads, array_reads


def _member_access(node: ast.Attribute, state: LoweringState) -> Optional[DataAccess]:
    """Resolve a whole structure-member access (``tracers.data``) to its
    dotted repository data path."""
    if not isinstance(node.value, ast.Name):
        return None
    member = state.context.member_access_of(node.value.id, node.attr)
    if member is None:
        return None
    path, descriptor = member
    return DataAccess(path, subsets.Range.from_array(descriptor), descriptor)


def indirect_index_reads(array_expression: ast.expr, state: LoweringState) -> List[ast.expr]:
    """
    The outermost data-access subexpressions inside a subscript's index (e.g.
    the ``A_col[j]`` in ``x[A_col[j]]``), or an empty list when the index is
    data-independent. Detection is AST-based and must run BEFORE
    :func:`resolve_access`: the shared memlet parser silently represents an
    inner data subscript it cannot evaluate as an applied sympy function, so a
    parse would not fail — it would produce a subset that references runtime
    data as if it were symbolic.

    Used by both the explicit-tasklet memlet rule and the elementwise
    computation mechanism to detect and lower genuine indirection
    (``x[A_col[j]]``) as a full-array connector plus synthetic index
    connectors, rather than falling back to the interpreter.

    Only *scalar* index reads count. A whole-array index (``A[indices]``) is
    NumPy advanced indexing — a different feature, with its own broadcasting
    and result-shape rules — and is left to :func:`resolve_access`, which
    rejects it as ``arrdims``. Treating it as indirection produces a tree that
    passes the callback-discrepancy check but cannot be converted to a valid
    SDFG: the index array inherits the *base* array's subset, so ``A[indices]``
    with ``A: float64[20]`` and ``indices: int32[3]`` emits ``indices[0:20]``,
    an out-of-bounds memlet. Verified for ``A[indices, 4]``,
    ``A[rows, columns]`` and ``A[indices, 2:7:2, [15, 10, 1]]`` — all three
    fail SDFG construction, two of them while scoring as successes.
    """
    if not isinstance(array_expression, ast.Subscript):
        return []
    reads: List[ast.expr] = []
    seen: set = set()

    class _Collector(ast.NodeVisitor):

        def _try_collect(self, node: ast.expr) -> bool:
            if not isinstance(getattr(node, 'ctx', ast.Load()), ast.Load):
                return False
            access = resolve_access(node, state)  # UnsupportedFeatureError propagates (nested indirection)
            if access is None or not access.is_scalar_access:
                return False
            key = astutils.unparse(node)
            if key not in seen:
                seen.add(key)
                reads.append(node)
            return True

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if not self._try_collect(node):
                self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if not self._try_collect(node):
                self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            self._try_collect(node)

    _Collector().visit(array_expression.slice)
    return reads


def resolve_symbol_names(node: ast.expr, state: LoweringState) -> ast.expr:
    """
    Return a copy of an expression with source-level names replaced by their
    repository names, so emitted code blocks reference tree containers and
    symbols directly.
    """
    result = copy.deepcopy(node)

    class _Renamer(ast.NodeTransformer):

        def visit_Name(self, name_node: ast.Name) -> ast.Name:
            binding = state.context.resolve(name_node.id)
            if binding is not None and binding.kind == 'container':
                name_node.id = binding.container
            return name_node

    return ast.fix_missing_locations(_Renamer().visit(result))


def substitute_data_operands(expr: ast.expr,
                             state: LoweringState,
                             connector_prefix: str = '__in') -> Tuple[str, List[Tuple[str, DataAccess]]]:
    """
    Replace every data access in a canonical (flat) expression with a fresh
    tasklet connector name.

    Indirect (data-dependent-index) reads such as ``x[A_col[j]]`` lower as a
    full-array pointer connector plus synthetic index connectors, with the
    element access moved into the rewritten code (``__in0[__in1]``) — the same
    scheme as the explicit-tasklet memlet rule
    (:mod:`~dace.frontend.python.nextgen.lowering.rules.dataflow_explicit`),
    applied here to ordinary (non-tasklet-syntax) computations.

    :return: A 2-tuple of (rewritten expression source, list of
             (connector, access) pairs in order of first appearance).
    """
    operands: List[Tuple[str, DataAccess]] = []
    seen: dict = {}
    rewritten = copy.deepcopy(expr)

    class _Substituter(ast.NodeTransformer):

        def visit_Subscript(self, subscript_node: ast.Subscript) -> ast.AST:
            if isinstance(subscript_node.ctx, ast.Load):
                if isinstance(subscript_node.value, (ast.Name, ast.Attribute)):
                    reads = indirect_index_reads(subscript_node, state)
                    if reads:
                        return self._indirect_subscript(subscript_node)
                access = resolve_access(subscript_node, state)
                if access is not None:
                    return self._connector_for(subscript_node, access)
            return self.generic_visit(subscript_node)

        def visit_Attribute(self, attribute_node: ast.Attribute) -> ast.AST:
            if isinstance(attribute_node.ctx, ast.Load):
                access = resolve_access(attribute_node, state)
                if access is not None:
                    return self._connector_for(attribute_node, access)
            return self.generic_visit(attribute_node)

        def visit_Name(self, name_node: ast.Name) -> ast.AST:
            if isinstance(name_node.ctx, ast.Load):
                access = resolve_access(name_node, state)
                if access is not None:
                    return self._connector_for(name_node, access)
            return name_node

        def _indirect_subscript(self, node: ast.Subscript) -> ast.AST:
            base_access = resolve_access(node.value, state)
            if base_access is None or isinstance(base_access.descriptor.dtype, dtypes.pyobject):
                raise UnsupportedFeatureError(
                    f'Indirect access references unknown or interpreter-only container '
                    f'"{astutils.unparse(node.value)}"',
                    state.context.filename,
                    node,
                    category='indirect-memlet')
            base_key = astutils.unparse(node.value) + '\0indirect'
            if base_key not in seen:
                connector = f'{connector_prefix}{len(operands)}'
                operands.append((connector,
                                 DataAccess(base_access.container,
                                            base_access.subset,
                                            base_access.descriptor,
                                            indirect=True)))
                seen[base_key] = connector
            connector = seen[base_key]
            index_node = self.visit(copy.deepcopy(node.slice))
            # Built as an AST rather than round-tripped through source: an
            # unparsed index tuple carries parentheses, and ``base[(i, 1:3)]``
            # is not valid Python even though ``base[i, 1:3]`` is.
            replacement = ast.Subscript(value=ast.Name(id=connector, ctx=ast.Load()), slice=index_node, ctx=ast.Load())
            return ast.fix_missing_locations(ast.copy_location(replacement, node))

        def _connector_for(self, original: ast.expr, access: DataAccess) -> ast.Name:
            key = astutils.unparse(original)
            if key not in seen:
                connector = f'{connector_prefix}{len(operands)}'
                operands.append((connector, access))
                seen[key] = connector
            return ast.copy_location(ast.Name(id=seen[key], ctx=ast.Load()), original)

    code = astutils.unparse(ast.fix_missing_locations(_Substituter().visit(rewritten)))
    return code, operands


def indexed_subset(access: DataAccess, params: List[str], result_shape: List) -> subsets.Range:
    """
    Compute the per-element subset of an operand access inside a map with the
    given parameters, applying NumPy-style broadcasting: dimensions of size 1
    are pinned, and missing leading dimensions are dropped (right-aligned).

    :param access: The operand access (its subset defines extents and offsets).
    :param params: Map parameter names, one per result dimension.
    :param result_shape: The broadcast result shape.
    """
    operand_size = access.subset.size()
    nondegenerate = [dim for dim, size in enumerate(operand_size) if size != 1]
    if len(nondegenerate) > len(params):
        raise UnsupportedFeatureError(
            'Operand has more nondegenerate dimensions than the broadcast result '
            f'({len(nondegenerate)} > {len(params)})',
            category='broadcast')
    # Right-align operand dims against result dims. When the operand carries
    # more raw dimensions than the result (integer-indexed dimensions kept as
    # size-1 subset entries), align its nondegenerate dims instead.
    param_offset = len(result_shape) - len(operand_size)
    if param_offset < 0:
        aligned_params = {dim: len(params) - 1 - position for position, dim in enumerate(reversed(nondegenerate))}
    else:
        aligned_params = {dim: dim + param_offset for dim in nondegenerate}
    ranges = []
    for dim_index, (dim_size, (start, _, step)) in enumerate(zip(operand_size, access.subset.ranges)):
        if dim_size == 1:
            index = start
        else:
            param = symbolic.pystr_to_symbolic(params[aligned_params[dim_index]])
            index = start + param * step
        ranges.append((index, index, 1))
    return subsets.Range(ranges)
