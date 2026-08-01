# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared helpers for resolving canonical data accesses into repository
containers, subsets, and connector-substituted tasklet code.
"""
import ast
import copy
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from dace import data, dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.registry import LoweringState
from dace.frontend.python.nextgen.semantics.indexing import (SOURCE, IndexPlan, build_plan, index_slots,
                                                             reverse_normalized, whole_container_plan)


@dataclass
class DataAccess:
    """A resolved read or write of a repository container."""
    container: str  #: Repository container name
    subset: subsets.Range  #: Accessed subset (in container index space)
    descriptor: data.Data
    #: The resolved NumPy index model of the subscript this access came from
    #: (see :mod:`~dace.frontend.python.nextgen.semantics.indexing`). Left off
    #: for a whole-container access, where :meth:`__post_init__` fills in the
    #: plan that says so -- every axis survives.
    plan: Optional[IndexPlan] = None
    #: True when this access is a full-array pointer connector synthesized for
    #: an indirect (data-dependent-index) read: its subset must not be
    #: broadcast/indexed against an iteration space — the tasklet code indexes
    #: it directly (e.g. ``__in0[__ind1]``).
    indirect: bool = False
    #: Source axes the subscript traverses from the top down (``A[::-1]``). A
    #: memlet subset cannot express a reversal, so the subset stays forward and
    #: the direction is applied where the index is formed
    #: (:func:`indexed_subset`).
    reversed_axes: FrozenSet[int] = frozenset()
    #: True when the index form could not be modeled against the container's
    #: rank (more elements than axes, several ``Ellipsis``). Shape questions
    #: then fall back to squeezing every size-1 dimension, which is a guess --
    #: hence :attr:`unmodeled_new_axes`.
    unmodeled: bool = False
    #: True when the shared parser reported ``newaxis`` insertions on an
    #: :attr:`unmodeled` index. Squeezing and inserting against the same
    #: assumed shape can silently produce a wrong result, so consumers must
    #: refuse rather than guess (see ``rules.assign._lower_name_assign``).
    unmodeled_new_axes: bool = False

    def __post_init__(self) -> None:
        if self.plan is None and not self.unmodeled:
            # A whole-container access: every axis of the descriptor survives.
            self.plan = container_plan(self.descriptor)

    @property
    def is_scalar_access(self) -> bool:
        return self.subset.num_elements() == 1

    @property
    def numpy_shape(self) -> List:
        """
        The NumPy-semantic result shape of this access: slice-formed
        dimensions survive at their size **including size 1** (``a[0:1, :]``
        is ``(1, 10)``), integer-indexed dimensions are dropped (``a[0:20, 1]``
        is ``(20,)``), and ``newaxis`` positions are inserted as size-1
        dimensions. Empty for scalar element accesses with no newaxes.
        """
        if self.plan is None or self.plan.ndim != len(self.subset.ranges):
            # Unmodeled, or a subset whose rank does not line up with the plan:
            # all that can be said is which dimensions are degenerate.
            return nondegenerate_shape(self.subset)
        return self.plan.result_shape(self.subset.size())


def container_plan(descriptor: data.Data) -> IndexPlan:
    """
    The index plan of a whole-container access.

    A dace ``Scalar`` is a rank-0 value in NumPy terms even though its
    descriptor carries a ``(1,)`` shape, so it contributes no result axis --
    otherwise assigning a scalar into an element target reads as a rank-1
    operand broadcast into a rank-0 result.
    """
    return whole_container_plan(0 if isinstance(descriptor, data.Scalar) else len(descriptor.shape))


def nondegenerate_shape(subset: subsets.Range) -> List:
    """
    The shape of a subset with size-1 dimensions squeezed out.

    This is the *fallback* for an index form that could not be modeled, and
    the rank-matching helper for broadcasting — not the NumPy result shape.
    For that, use :attr:`DataAccess.numpy_shape`, which keeps a size-1
    dimension that the subscript wrote as a slice.
    """
    return [s for s in subset.size() if s != 1]


def kept_dimensions(slice_node: ast.expr, ndim: int) -> Optional[List[bool]]:
    """
    Which subset dimensions a subscript keeps in its NumPy result shape:
    slice-formed indices keep their dimension, integer indices drop it, and
    unindexed trailing dimensions are kept.

    A thin view over :class:`~...semantics.indexing.IndexPlan` for callers that
    only need the per-dimension flags. Returns None (unknown -- caller must not
    blindly squeeze) when the index cannot be modeled against ``ndim``.
    """
    plan = build_plan(slice_node, ndim)
    return None if plan is None else plan.kept_dims


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
        return DataAccess(binding.container,
                          subsets.Range.from_array(descriptor),
                          descriptor,
                          plan=container_plan(descriptor))
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
        node = promote_index_reads(node, state, ndim=len(base_descriptor.shape))
        node, reversed_axes = reverse_normalized(node, base_descriptor.shape, state.inference.constant_int)
        member_reads = _member_index_reads(node, state)
        if member_reads:
            # ``B[i, A.indices[idx]]``: indirection through a STRUCTURE MEMBER.
            # The plain-array spelling reaches the same conclusion below (the
            # parser renders ``indices[idx]`` as an applied function and
            # ``data_dependent_container_names`` recognizes it), but a member
            # read parses as an attribute of a symbol, which sympy then cannot
            # apply at all. Reporting it here keeps both spellings on the
            # tasklet indirection path instead of ending one of them in a
            # memlet-parse failure.
            raise UnsupportedFeatureError(
                f'Data-dependent subscript index (reads {", ".join(sorted(member_reads))}) is not supported here',
                state.context.filename,
                node,
                category='data-dependent-subscript')
        expr = state.inference.parse_access(node)
        subset = substitute_symbolic_scalars(expr.subset, state.context)
        if isinstance(subset, subsets.Indices):
            subset = subsets.Range.from_indices(subset)
        if expr.arrdims:
            raise UnsupportedFeatureError('Advanced (array-valued) indexing is not supported yet',
                                          state.context.filename,
                                          node,
                                          category='data-dependent-subscript')
        _, array_reads = data_dependent_container_names(subset, state.context)
        if array_reads:
            # An array element read left in an ELEMENT position of an
            # otherwise element-only subscript (x[A_col[j]]): genuine
            # indirection, lowered by the tasklet paths, not here.
            raise UnsupportedFeatureError(
                f'Data-dependent subscript index (reads {", ".join(sorted(array_reads))}) is not supported here',
                state.context.filename,
                node,
                category='data-dependent-subscript')
        plan = build_plan(node.slice, len(subset.ranges))
        return DataAccess(base_container,
                          subset,
                          base_descriptor,
                          plan=plan,
                          reversed_axes=reversed_axes,
                          unmodeled=plan is None,
                          unmodeled_new_axes=plan is None and bool(expr.new_axes))
    return None


def _member_index_reads(node: ast.Subscript, state: LoweringState) -> List[str]:
    """
    The source texts of structure-member data reads sitting in this
    subscript's index (the ``A.indices[idx]`` in ``B[i, A.indices[idx]]``).

    Detection is AST-based and deliberately runs before the memlet parse: a
    member read reaches sympy as an attribute of a symbol, which raises rather
    than producing the applied function a plain array read would.
    """
    return [
        astutils.unparse(read) for read in indirect_index_reads(node, state) if any(
            isinstance(inner, ast.Attribute) for inner in ast.walk(read))
    ]


def keeps_range(node: ast.Subscript, state: LoweringState, ndim: Optional[int] = None) -> bool:
    """
    Whether a subscript's result still spans a range in some dimension, which
    is what decides whether indirection can express it at all (indirection
    lowers a whole-ELEMENT read).

    A range need not be written as a slice: indexing fewer dimensions than the
    array has leaves the rest whole (``G[k, E, neigh[a, b]]`` on a 5-D ``G``),
    and ``Ellipsis`` stands for however many of those there are.

    :param ndim: The base container's rank, when the caller already knows it.
    """
    elements = index_elements(node.slice)
    if any(isinstance(element, ast.Slice) for element in elements):
        return True
    if any(isinstance(element, ast.Constant) and element.value is Ellipsis for element in elements):
        return True
    if ndim is None:
        base = resolve_access(node.value, state) if isinstance(node.value, (ast.Name, ast.Attribute)) else None
        if base is None:
            return False
        ndim = len(base.subset.ranges)
    # ``newaxis`` consumes no source dimension (see ``kept_dimensions``).
    indexed = [element for element in elements if not (isinstance(element, ast.Constant) and element.value is None)]
    return len(indexed) < ndim


def promote_index_reads(node: ast.Subscript,
                        state: LoweringState,
                        force_elements: bool = False,
                        ndim: Optional[int] = None) -> ast.Subscript:
    """
    Rewrite the data reads inside a subscript's index that belong on the SYMBOL
    side into symbols defined from them, each by an interstate assignment
    (:class:`~...treenodes.AssignNode`) emitted just before the access -- the
    same mechanism dynamic map bounds use.

    ``doc/extensions/symbolic.rst`` puts a quantity consumed in a range on the
    symbol side, and that is decided by the index FORM, not by the extent it
    happens to produce:

    - every SLICE BOUND is a range bound, including one that spans a single
      element (``A[n:n + 1]``). Left as data, a container name in a range
      prints where the code generator expects a symbol, and lowering it as
      indirection instead is worse still: the source goes behind a pointer
      connector and the slice stays in the tasklet code
      (``__out = __in0[0:__in1] + 1.0``), from which the code generator drops
      the subscript entirely and emits pointer arithmetic;
    - an ELEMENT position too, but only when the subscript still spans a range
      somewhere (``A[col[i], 0:5]``, and equally ``G[k, E, neigh[a, b]]`` on a
      5-D ``G``, where the unindexed dimensions stay whole -- see
      :func:`keeps_range`). Indirection lowers a whole-element read, so a
      subscript that keeps a range cannot use it at all -- that shape reached
      SDFG construction as ``__in0[__in1, 0:5]`` and failed there.

    An element position in an otherwise element-only subscript (``A[n]``,
    ``x[A_col[j]]``) is left alone: that IS indirection, which the tasklet
    paths lower by reading the index as data (with the conflict resolution an
    indirect write needs).

    Symbols are cached for the duration of one statement
    (``LoweringState.index_symbols``), keyed by the read they are defined from,
    so a target and the operands of one statement share a single definition --
    ``O[0:n] = A[0:n] + 1.0`` needs its two subsets to agree on one symbol, not
    on two symbols that merely hold equal values. The cache is dropped between
    statements, so a scalar reassigned in between cannot leave a stale
    definition behind.

    :param force_elements: Promote element positions unconditionally. Set when
        the subscript is itself becoming an interstate assignment's right-hand
        side (``O[0:ends[k]]`` defines a symbol from ``ends[k]``, so ``k`` has
        to be a symbol there too -- an interstate edge cannot read data).
    :raises UnsupportedFeatureError: If a read that has to be promoted is not
        of integer type, which is a genuine feature gap.
    """
    elements = index_elements(node.slice)
    ranged = force_elements or keeps_range(node, state, ndim)
    sources: List[ast.expr] = []
    for element in elements:
        if isinstance(element, ast.Slice):
            sources.extend(bound for bound in (element.lower, element.upper, element.step) if bound is not None)
        elif ranged:
            sources.append(element)

    replacements: Dict[str, str] = {}
    for source in sources:
        for read in scalar_data_reads(source, state):
            key = astutils.unparse(read)
            if key not in replacements:
                replacements[key] = _index_symbol(read, node, state)
    if not replacements:
        return node
    promoted = astutils.copy_tree(node)
    promoted.slice = substitute_index_reads(promoted.slice, replacements)
    return ast.fix_missing_locations(ast.copy_location(promoted, node))


def _index_symbol(read: ast.expr, node: ast.AST, state: LoweringState) -> str:
    """
    The symbol defined from a single data read in an index position, reusing
    this statement's definition of the same read if there is one.

    The read becomes an interstate assignment's right-hand side, which can name
    symbols but not data, so its own indices are promoted first (``ends[k]``
    with a scalar ``k`` defines its symbol from ``ends[__idx_k_0]``).
    """
    if isinstance(read, ast.Subscript):
        read = promote_index_reads(read, state, force_elements=True)
    access = resolve_access(read, state)
    expression = scalar_read_expression(access) if access is not None else None
    if expression is None or data_dependent_container_names(access.subset, state.context) != (set(), set()):
        raise UnsupportedFeatureError(
            f'Subscript index reads "{astutils.unparse(read)}", which is not a single element of known position',
            state.context.filename,
            node,
            category='data-dependent-subscript')
    existing = state.index_symbols.get(expression)
    if existing is not None:
        return existing
    dtype = access.descriptor.dtype
    if not isinstance(dtype, dtypes.typeclass) or dtype not in dtypes.INTEGER_TYPES:
        raise UnsupportedFeatureError(f'Subscript index depends on non-integer data "{access.container}"',
                                      state.context.filename,
                                      node,
                                      category='data-dependent-subscript')
    # A structure member arrives as a dotted repository path, which is not a
    # legal symbol name.
    symbol_name = state.context.fresh_name(f'__idx_{access.container.lstrip("_").replace(".", "_")}_')
    state.context.symbols[symbol_name] = symbolic.symbol(symbol_name, dtype)
    state.emitter.emit(
        tn.AssignNode(name=symbol_name,
                      value=CodeBlock(expression),
                      edge=InterstateEdge(assignments={symbol_name: expression})))
    state.index_symbols[expression] = symbol_name
    return symbol_name


def scalar_read_expression(access: DataAccess) -> Optional[str]:
    """A single-element access rendered for an interstate assignment
    (``bounds[0]``), or None if the access is not one element."""
    if isinstance(access.descriptor, data.Scalar):
        return f'{access.container}[0]'
    try:
        if data._prod(access.subset.size()) != 1:
            return None
    except TypeError:
        return None
    return f'{access.container}[{", ".join(str(begin) for begin, _, _ in access.subset.ranges)}]'


def substitute_symbolic_scalars(subset, context):
    """
    Put back the symbolic VALUE of any ANF temporary a subset references.

    Canonicalization hoists a compound index expression into a temporary
    (``O[i * 96:(i + 1) * 96]`` becomes ``__anf0 = i + 1; O[i * 96:__anf0 *
    96]``), and that temporary is materialized as a scalar container. In a
    subset the container NAME is meaningless -- it reads as a data-dependent
    index, which inside a dataflow scope is refused outright, and even outside
    one it keeps sizes like ``__anf0 * 96 - i * 96`` from folding to 96.

    The temporaries that hold a compile-time symbolic value record it
    (``ProgramContext.symbolic_scalar_values``, written for single-assignment
    ANF temps precisely so the value survives materialization), so substituting
    it back recovers the expression the source wrote.
    """
    if not context.symbolic_scalar_values:
        return subset
    replacements = {symbolic.pystr_to_symbolic(name): value for name, value in context.symbolic_scalar_values.items()}
    if not replacements:
        return subset
    try:
        substituted = copy.deepcopy(subset)
        substituted.replace(replacements)
        return substituted
    except Exception:
        return subset  # A subset shape the substitution cannot rewrite: leave it alone


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
    return DataAccess(path, subsets.Range.from_array(descriptor), descriptor, plan=container_plan(descriptor))


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

    Indirection lowers a whole-ELEMENT read, so a subscript that keeps a range
    (``A[0:n]``, ``A[col[i], 0:5]``) has none: its data-dependent indices
    belong on the symbol side and are promoted there by
    :func:`promote_index_reads`. Reporting them here instead put the source
    behind a pointer connector and left the slice in the tasklet code
    (``__out = __in0[0:__in1] + 1.0``), from which the code generator drops the
    subscript entirely and emits ``__in0_0 + 1.0`` -- pointer arithmetic where
    an element was meant, caught only by the C++ compiler.

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
    if keeps_range(array_expression, state):
        return []
    elements = index_elements(array_expression.slice)
    reads: List[ast.expr] = []
    seen: set = set()
    for element in elements:
        for read in scalar_data_reads(element, state):
            key = astutils.unparse(read)
            if key not in seen:
                seen.add(key)
                reads.append(read)
    return reads


def scalar_data_reads(expression: ast.expr, state: LoweringState) -> List[ast.expr]:
    """
    The outermost single-element data reads inside an expression, in order of
    first appearance and deduplicated by source text. Shared by
    :func:`indirect_index_reads` (which lowers them as tasklet connectors) and
    :func:`promote_index_reads` (which defines symbols from them).
    """
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

    _Collector().visit(expression)
    return reads


def index_elements(slice_node: ast.expr) -> List[ast.expr]:
    """The per-dimension elements of a subscript index, as a flat list.

    Kept as the lowering-side spelling of
    :func:`~dace.frontend.python.nextgen.semantics.indexing.index_slots`, which
    owns the definition."""
    return index_slots(slice_node)


def resolve_symbol_names(node: ast.expr, state: LoweringState) -> ast.expr:
    """
    Return a copy of an expression with source-level names replaced by their
    repository names, so emitted code blocks reference tree containers and
    symbols directly.
    """
    result = astutils.copy_tree(node)

    class _Renamer(ast.NodeTransformer):

        def visit_Name(self, name_node: ast.Name) -> ast.Name:
            binding = state.context.resolve(name_node.id)
            if binding is not None and binding.kind == 'container':
                name_node.id = binding.container
            elif binding is not None and binding.kind == 'symbol':
                # A symbol's repository name may differ from the source name
                # (see ``ProgramContext.bind_symbol``).
                name_node.id = state.context.symbol_of(name_node.id).name
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
    rewritten = astutils.copy_tree(expr)

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
            index_node = self.visit(astutils.copy_tree(node.slice))
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


def substitute_index_reads(index_expression: ast.expr, index_names: Dict[str, str]) -> ast.expr:
    """
    Replace collected data reads in an index expression with their synthetic
    connector names (matched by unparsed source text). Shared between the
    explicit-tasklet indirect-memlet rule
    (:mod:`~dace.frontend.python.nextgen.lowering.rules.dataflow_explicit`)
    and :func:`lower_indirect_write` below -- both move a data-dependent
    index's element access into tasklet code the same way.
    """
    result = astutils.copy_tree(index_expression)

    class _Replacer(ast.NodeTransformer):

        def visit(self, node: ast.AST) -> ast.AST:
            key = astutils.unparse(node) if isinstance(node, ast.expr) else None
            if key in index_names:
                return ast.copy_location(ast.Name(id=index_names[key], ctx=ast.Load()), node)
            return super().visit(node)

    return ast.fix_missing_locations(_Replacer().visit(result))


def lower_indirect_write(target: ast.Subscript,
                         value: ast.expr,
                         statement: ast.stmt,
                         state: LoweringState,
                         wcr: Optional[str] = None) -> bool:
    """
    Lower a write into a subscript whose index is data-dependent
    (``hist[bin] += 1``, ``hist[bin] = x``) inside a dataflow scope, as a
    genuine indirect write: the base array becomes a full-array OUTPUT
    connector, each data-dependent index subexpression becomes a synthetic
    input connector, and the element write moves into the tasklet code
    (``__arr[__ind0, ...] = <value>``) -- the mirror of this module's own
    READ-side indirection (:func:`substitute_data_operands`'s
    ``_indirect_subscript``), and the same shape
    ``rules/dataflow_explicit.py::_lower_indirect_memlet`` already implements
    for explicit-tasklet syntax (``local >> A[A_col[j]]``), ported here for
    ordinary assignment statements, which have no such syntax to hook into.

    Returns False when the target's index is not actually data-dependent (no
    indirect reads found), so the caller falls through to the ordinary
    paths -- which, for that case, correctly resolve the target directly.

    :param wcr: Conflict-resolution lambda for an accumulating write
                (``hist[bin] += 1``). An indirect write takes WCR
                UNCONDITIONALLY when accumulating -- a stronger rule than
                ``conflict.accumulation_wcr`` applies to a plain (direct)
                target: the index expression may name the same element from
                two different map iterations, and no subset inspection can
                rule that out (the same reasoning
                ``advanced_indexing.emit_scatter`` already applies to
                accumulating advanced-index writes). A plain (non-accumulating)
                overwrite needs no WCR: like NumPy's own fancy-index
                assignment, which of several colliding writes wins is
                unspecified, not a correctness hazard the frontend must guard.
    :raises UnsupportedFeatureError: If the base container or an index read is
        unknown or interpreter-only (a pyobject scalar from an earlier
        callback fallback, which cannot participate in dataflow).
    """
    if not isinstance(target.value, (ast.Name, ast.Attribute)):
        return False
    reads = indirect_index_reads(target, state)
    if not reads:
        return False
    base_access = resolve_access(target.value, state)
    if base_access is None or isinstance(base_access.descriptor.dtype, dtypes.pyobject):
        raise UnsupportedFeatureError(
            f'Indirect write references unknown or interpreter-only container "{astutils.unparse(target.value)}"',
            state.context.filename,
            target,
            category='indirect-memlet')

    index_names: Dict[str, str] = {}
    in_memlets: Dict[str, Memlet] = {}
    for read in reads:
        access = resolve_access(read, state)
        if access is None or isinstance(access.descriptor.dtype, dtypes.pyobject):
            raise UnsupportedFeatureError('Indirect write index references an unknown or interpreter-only container',
                                          state.context.filename,
                                          target,
                                          category='indirect-memlet')
        connector = f'__ind{len(index_names)}'
        in_memlets[connector] = Memlet(data=access.container, subset=access.subset)
        index_names[astutils.unparse(read)] = connector

    substituted_index = substitute_index_reads(target.slice, index_names)
    index_code = astutils.unparse(
        ast.Subscript(value=ast.Name(id='__arr', ctx=ast.Load()), slice=substituted_index, ctx=ast.Load()))

    code, operands = substitute_data_operands(value, state, connector_prefix='__val')
    for connector, operand in operands:
        in_memlets[connector] = Memlet(data=operand.container, subset=operand.subset)

    out_memlets = {'__arr': Memlet(data=base_access.container, subset=base_access.subset, wcr=wcr)}
    line = getattr(statement, 'lineno', 0)
    tasklet = nodes.Tasklet(f'indirect_write_{line}', set(in_memlets), {'__arr'}, f'{index_code} = {code}')
    state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))
    return True


def indexed_subset(access: DataAccess, params: List[str], result_shape: List) -> subsets.Range:
    """
    Compute the per-element subset of an access inside a map over the broadcast
    result, applying NumPy broadcasting: the access's NumPy shape is
    right-aligned against the result, a size-1 dimension is pinned (the same
    element is read across the broadcast extent), and a dimension the subscript
    dropped is pinned at its own start.

    Alignment goes through the access's :class:`~...semantics.indexing.IndexPlan`
    rather than through its subset, because the subset cannot express a
    ``newaxis``: ``A[:, None]`` and ``A[None, :]`` have the *same* rank-1
    subset over ``A`` and differ only in where their axes land in the result.
    Aligning by subset shape made both read ``A[__i1]``, which produced the
    wrong values for every outer-product-shaped expression.

    :param access: The access (its subset defines extents and offsets).
    :param params: Map parameter names, one per result dimension.
    :param result_shape: The broadcast result shape.
    """
    shape = access.numpy_shape
    excess = len(shape) - len(params)
    if excess > 0 and any(size != 1 for size in shape[:excess]):
        raise UnsupportedFeatureError(
            f'Operand shape {tuple(shape)} has higher rank than the broadcast result '
            f'({len(shape)} > {len(params)})',
            category='broadcast')

    ranges: List[Optional[Tuple]] = [None] * len(access.subset.ranges)
    offset = len(params) - len(shape)  # Right-alignment against the result
    if access.plan is None:
        # Unmodeled index form: fall back to aligning nondegenerate dimensions,
        # which is what the squeezed shape supports.
        nondegenerate = [dim for dim, size in enumerate(access.subset.size()) if size != 1]
        placement = {dim: offset + position for position, dim in enumerate(nondegenerate)}
    else:
        placement = {
            entry.axis: offset + position
            for position, entry in enumerate(access.plan.result_layout()) if entry.kind == SOURCE
        }

    sizes = access.subset.size()
    for axis, (start, end, step) in enumerate(access.subset.ranges):
        slot = placement.get(axis)
        if slot is None or slot < 0 or sizes[axis] == 1 or params[slot] is None:
            # Dropped by the subscript, a leading singleton beyond the result
            # rank, a broadcast singleton, or a result dimension that does not
            # iterate (``params`` entry is None): pinned.
            index = start
        elif axis in access.reversed_axes:
            # ``A[::-1]``: the subset is the forward range, walked from the top.
            param = symbolic.pystr_to_symbolic(params[slot])
            index = end - param * step
        else:
            param = symbolic.pystr_to_symbolic(params[slot])
            index = start + param * step
        ranges[axis] = (index, index, 1)
    return subsets.Range(ranges)
