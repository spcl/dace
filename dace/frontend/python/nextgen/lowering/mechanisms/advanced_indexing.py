# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
NumPy advanced (array-valued) indexing.

``A[indices]``, ``A[rows, columns]``, ``A[ind, 2:7:2, [15, 10, 1]]`` — a
subscript in which one or more dimensions are indexed by an *array* rather than
by a scalar or a slice. This is a distinct feature from scalar indirection
(``x[A_col[j]]``, see :func:`~...lowering.access.indirect_index_reads`), which
reads a single element to index with; here the index is a whole array, the
index arrays broadcast against each other, and the result shape follows NumPy's
own rules rather than the subset's.

The shared memlet parser already recognizes the form and reports it as
``MemletExpr.arrdims`` (a ``{dimension: index array}`` mapping); everything
below turns that into dataflow.

Lowering shape, matching the classic frontend
(``newast.py::_array_indirection_subgraph``): one map over the *result* index
space, with

- a pointer connector on the base array whose subset pins every basic-indexed
  dimension to one element and leaves every advanced-indexed dimension whole,
- one element-read connector per index array, subset by the broadcast index
  parameters, and
- a tasklet ``__out = __arr[__inp0, __inp1, ...]`` that performs the gather.

Result-shape rules implemented here (all from NumPy, verified against classic):

- advanced-indexed dimensions collapse to the *broadcast* shape of the index
  arrays, inserted at the position of the advanced chunk;
- if the advanced indices are **not contiguous** in the subscript, the
  broadcast shape moves to the *front* of the result instead;
- scalar-indexed dimensions count as advanced once any advanced index is
  present, which is why ``A[ind, 4]`` differs in shape from ``A[ind, 4:5]``;
- ``None``/``newaxis`` inserts a singleton, before or after the advanced chunk
  depending on the same contiguity rule.

Boolean-mask *writes* (``A[mask] = ...``) lower as a guarded update over the
full array -- no allocation needed, since positions are static even though the
written *count* is data-dependent (see :func:`emit_masked_write`).

Boolean-mask *reads* (``B = A[mask]``) are different: the result itself is
data-dependent in size. :func:`emit_boolean_gather` supports the bare
top-level form (``B = A[mask]``, the whole assignment) by minting a fresh SDFG
symbol from a runtime-computed element count and using it to size the result
-- see ``tests/sdfg/deferred_symbol_boolean_filter_test.py`` for the
underlying mechanism, proven independently of this frontend, including the
two configurable strategies (``frontend.boolean_index_strategy``) and a
load-bearing pitfall (the WCR accumulator needs an explicit zero-init) found
while building it. A boolean-mask read nested inside a larger expression, or
combined with another index, still falls back to a callback exactly as before
-- matching classic (``newast.py::_add_read_slice``: "Boolean array indexing
is only supported for assignment targets") for everything except this one
new, narrowly-scoped case.
"""
import ast
import copy
from dataclasses import dataclass
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy

from dace import data, dtypes, subsets, symbolic
from dace.config import Config
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import InterstateEdge, nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python.memlet_parser import MemletExpr
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import DataAccess
from dace.frontend.python.nextgen.lowering.mechanisms.conflict import WCR_OPERATORS
from dace.frontend.python.nextgen.lowering.registry import LoweringState
from dace.frontend.python.nextgen.semantics.indexing import SOURCE, IndexPlan, build_plan


@dataclass(frozen=True)
class AdvancedIndex:
    """
    A resolved advanced-indexing access, ready to emit.

    :param container: The indexed container.
    :param output_shape: The NumPy result shape of the access.
    :param map_ranges: Map parameter to its ``(start, end, step)`` range,
                       covering the whole result index space.
    :param base_subset: Subset of the base array read (or written) by one
                        iteration: basic dimensions pinned to one element,
                        advanced dimensions left whole so the connector is a
                        pointer the tasklet can index.
    :param output_subset: The matching subset of the result container.
    :param index_memlets: One element-read memlet per index array, in the order
                          their connectors appear in the tasklet code.
    :param index_extents: The extent of the base dimension each index array
                          indexes into, in the same order -- what a negative
                          index entry wraps against (see
                          :func:`index_expressions`).
    """
    container: str
    descriptor: data.Data
    output_shape: List[Any]
    map_ranges: Dict[str, Tuple[Any, Any, Any]]
    base_subset: subsets.Range
    output_subset: subsets.Range
    index_memlets: Tuple[Memlet, ...]
    index_extents: Tuple[Any, ...] = ()


def index_arrays(expr: MemletExpr) -> Dict[int, Any]:
    """The ``{dimension: index array}`` mapping of an access, empty if the
    access uses no advanced indexing."""
    return expr.arrdims or {}


def has_boolean_index(expr: MemletExpr, context, index_types: Optional[Dict[str, data.Data]] = None) -> bool:
    """Whether any index array is a boolean mask, which selects elements by
    predicate and therefore has a data-dependent result size.

    :param index_types: Descriptors of index-expression placeholders, for a
                        parse that has not been materialized yet (see
                        :meth:`InferenceService.parse_access_typed`).
    """
    for index in index_arrays(expr).values():
        if _index_dtype(index, context, index_types) == dtypes.bool:
            return True
    return False


def _plan_for(expr: MemletExpr, node: ast.expr) -> IndexPlan:
    """
    The index model of an advanced-indexing access.

    :raises UnsupportedFeatureError: If the index cannot be modeled against the
        base rank -- previously this fell back to a range-based heuristic that
        could not tell a naturally size-1 dimension from an integer-indexed
        one, and silently produced the wrong shape.
    """
    plan = build_plan(node.slice, len(expr.subset.ranges), frozenset(index_arrays(expr)))
    if plan is None:
        raise UnsupportedFeatureError(
            f'Cannot model the index form of "{astutils.unparse(node)}" '
            'against the indexed container',
            node=node,
            category='advanced-indexing')
    return plan


def output_shape(expr: MemletExpr,
                 context,
                 inference,
                 node: ast.expr,
                 index_types: Optional[Dict[str, data.Data]] = None) -> List[Any]:
    """
    The NumPy result shape of an advanced-indexing access.

    Kept separate from :func:`analyze` because inference needs the shape to
    allocate the result container before any node is emitted.
    """
    broadcast = _broadcast_index_shape(expr, context, inference, node, index_types)
    return _plan_for(expr, node).result_shape(expr.subset.size(), broadcast)


def analyze(node: ast.Subscript, expr: MemletExpr, container: str, descriptor: data.Data, context,
            inference) -> AdvancedIndex:
    """
    Resolve an advanced-indexing access into the map ranges, memlets and result
    shape needed to emit it.

    :raises UnsupportedFeatureError: If an index array is a boolean mask, or
                                     cannot be resolved to a container.
    """
    if has_boolean_index(expr, context):
        raise UnsupportedFeatureError(
            'Boolean-mask indexing produces a data-dependent result size and is only '
            'supported on assignment targets',
            context.filename,
            node,
            category='advanced-indexing')

    ndrange = expr.subset.ndrange()
    # One map parameter per non-degenerate basic dimension; the advanced
    # dimensions get their parameters from the index broadcast instead.
    iteration = [(symbolic.symbol(f'__i{i}'), symbolic.symbol(f'__i{i}'), 1) if start != end else (0, 0, 1)
                 for i, (start, end, _) in enumerate(ndrange)]
    base_subset = subsets.Range([(rb + index * rs, rb + index * rs, 1)
                                 for (rb, _, rs), (index, _, _) in zip(ndrange, iteration)])
    map_ranges = {
        f'__i{i}': (0, size - 1, 1)
        for i, (size, (start, end, _)) in enumerate(zip(expr.subset.size(), ndrange)) if start != end
    }

    plan = _plan_for(expr, node)
    broadcast = _broadcast_index_shape(expr, context, inference, node)
    parameters = [f'__ind{i}' for i in range(len(broadcast))]
    for parameter, size in zip(parameters, broadcast):
        map_ranges[parameter] = (0, size - 1, 1)
    advanced_ndrange = [(symbolic.symbol(parameter), symbolic.symbol(parameter), 1) for parameter in parameters]

    # Result shape and result subset are laid out from the SAME plan, so the
    # subset written into the result cannot disagree with the shape allocated
    # for it -- the class of mismatch that made ``A[ind, None, 4]`` crash.
    result_shape = plan.result_shape(expr.subset.size(), broadcast)
    output_ndrange = plan.place(iteration, advanced_ndrange, newaxis=(0, 0, 1))

    index_memlets: List[Memlet] = []
    index_extents: List[Any] = []
    sizes = expr.subset.size()
    for dimension, index in index_arrays(expr).items():
        index_container = _index_container(index, dimension, container, context, inference, node)
        index_shape = context.containers[index_container].shape
        index_memlets.append(Memlet(data=index_container, subset=_broadcast_subset(index_shape, parameters), volume=1))
        index_extents.append(sizes[dimension])
        map_ranges.pop(f'__i{dimension}', None)
        base_subset[dimension] = ndrange[dimension]

    return AdvancedIndex(container=container,
                         descriptor=descriptor,
                         output_shape=result_shape,
                         map_ranges=map_ranges,
                         base_subset=base_subset,
                         output_subset=subsets.Range(output_ndrange),
                         index_memlets=tuple(index_memlets),
                         index_extents=tuple(index_extents))


def index_expressions(access: AdvancedIndex) -> List[str]:
    """
    The tasklet index expression per index connector.

    NumPy wraps a negative index entry (``ind = [-1]`` reads the last element);
    the emitted tasklet indexes the base pointer directly, so without a wrap a
    negative entry reads out of bounds. Whether to pay for the wrap follows the
    same configuration entry scalar indices already use
    (``frontend.runtime_negative_indices``, default off), so the default path
    stays byte-identical.

    An index array's entries are runtime data, so unlike
    ``memlet_parser._wrap_scalar_index`` there is no sign to prove and no
    symbolic form to emit: the wrap goes into the tasklet body, written as a
    conditional expression because ``%`` in tasklet code becomes C++'s
    truncating modulo rather than Python's.
    """
    connectors = [f'__inp{position}' for position in range(len(access.index_memlets))]
    if not Config.get_bool('frontend', 'runtime_negative_indices'):
        return connectors
    return [
        f'({connector} + {extent} if {connector} < 0 else {connector})'
        for connector, extent in zip(connectors, access.index_extents)
    ]


def emit_gather(target: DataAccess, access: AdvancedIndex, statement: ast.stmt, state: LoweringState) -> None:
    """Emit the map-with-tasklet that gathers an advanced-indexing read into a
    target container."""
    line = getattr(statement, 'lineno', 0)
    connectors = [f'__inp{i}' for i in range(len(access.index_memlets))]
    code = f'__out = __arr[{", ".join(index_expressions(access))}]'

    in_memlets = {'__arr': Memlet(data=access.container, subset=access.base_subset, volume=1)}
    for connector, memlet in zip(connectors, access.index_memlets):
        in_memlets[connector] = memlet

    parameters = list(access.map_ranges)
    map_node = nodes.MapEntry(
        nodes.Map(f'advanced_index_{line}', parameters,
                  subsets.Range([access.map_ranges[parameter] for parameter in parameters])))
    tasklet = nodes.Tasklet(f'advanced_index_{line}', set(in_memlets), {'__out'}, code)
    out_memlet = Memlet(data=target.container, subset=_target_subset(target, access), volume=1)

    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets={'__out': out_memlet}))


def emit_scatter(access: AdvancedIndex,
                 value: ast.expr,
                 statement: ast.stmt,
                 state: LoweringState,
                 wcr: Optional[str] = None) -> None:
    """
    Emit the map-with-tasklet that scatters a value into an advanced-indexing
    write target (``A[indices] = B``, ``A[1:2, indices, 3:4] = 2``).

    The mirror of :func:`emit_gather`: the base array becomes an *output*
    pointer connector and the element access moves into the tasklet code, the
    same shape the explicit-tasklet rule uses for indirect writes
    (``rules/dataflow_explicit.py``).

    :param wcr: Conflict-resolution lambda for the write. Required whenever two
                index entries can name the same element -- an accumulation
                (``A[indices] += 1``) with a repeated index would otherwise lose
                updates.
    """
    from dace.frontend.python.nextgen.lowering.access import substitute_data_operands

    line = getattr(statement, 'lineno', 0)
    index_connectors = [f'__inp{i}' for i in range(len(access.index_memlets))]

    code, operands = substitute_data_operands(value, state, connector_prefix='__val')
    in_memlets: Dict[str, Memlet] = dict(zip(index_connectors, access.index_memlets))
    for connector, operand in operands:
        in_memlets[connector] = Memlet(data=operand.container, subset=_value_subset(operand, access), volume=1)

    out_memlets = {'__arr': Memlet(data=access.container, subset=access.base_subset, volume=1, wcr=wcr)}
    tasklet_code = f'__arr[{", ".join(index_expressions(access))}] = {code}'

    parameters = list(access.map_ranges)
    map_node = nodes.MapEntry(
        nodes.Map(f'advanced_scatter_{line}', parameters,
                  subsets.Range([access.map_ranges[parameter] for parameter in parameters])))
    tasklet = nodes.Tasklet(f'advanced_scatter_{line}', set(in_memlets), set(out_memlets), tasklet_code)

    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))


def emit_masked_write(node: ast.Subscript, expr: MemletExpr, container: str, descriptor: data.Data, value: ast.expr,
                      statement: ast.stmt, state: LoweringState) -> None:
    """
    Emit a boolean-mask write (``A[mask] = 2``, ``A[A > 15] += 5``).

    A mask selects elements by predicate, so the *count* of written elements is
    data-dependent — but their *positions* are not: every candidate element is
    visited exactly once. The write therefore lowers as a map over the full
    array with a guarded update, ``__out = <new value> if __mask else __in``,
    with no conflict resolution needed (distinct iterations write distinct
    elements) and no dynamic allocation (nothing is gathered).

    Unlike the mask *read* path (:func:`emit_boolean_gather`), no allocation
    is needed here even though the count is data-dependent: positions are
    static (every candidate element is visited exactly once), only the
    number that end up written varies.

    :raises UnsupportedFeatureError: If the mask is combined with an integer
        index array, if the mask shape does not cover the target, or if the
        assigned value is not elementwise-uniform over the mask.
    """
    mask_container = resolve_single_boolean_mask(node, expr, container, state.context, state.inference)
    target_shape = list(expr.subset.size())

    line = getattr(statement, 'lineno', 0)
    parameters = [f'__i{i}' for i in range(len(target_shape))]
    element = subsets.Range([(start + symbolic.symbol(parameter) * step, start + symbolic.symbol(parameter) * step, 1)
                             for parameter, (start, _, step) in zip(parameters, expr.subset.ranges)])
    mask_subset = subsets.Range([(symbolic.symbol(parameter), symbolic.symbol(parameter), 1)
                                 for parameter in parameters])

    code, operands = substitute_masked_operands(value, state)
    in_memlets: Dict[str, Memlet] = {
        '__mask': Memlet(data=mask_container, subset=mask_subset, volume=1),
        # Copied, not shared: the read and the write of the updated element are
        # distinct memlets and SDFG validation rejects a shared subset object.
        '__in': Memlet(data=container, subset=copy.deepcopy(element), volume=1),
    }
    for connector, operand in operands:
        if not operand.is_scalar_access:
            raise UnsupportedFeatureError(
                'Boolean-mask assignment from an array requires as many values as the mask '
                'selects, which is only known at runtime',
                state.context.filename,
                node,
                category='advanced-indexing')
        in_memlets[connector] = Memlet(data=operand.container, subset=operand.subset, volume=1)

    operator = getattr(statement, 'augmented_op', None)
    symbol = WCR_OPERATORS.get(type(operator)) if operator is not None else None
    updated = f'__in {symbol} ({code})' if symbol is not None else f'({code})'
    tasklet = nodes.Tasklet(f'masked_write_{line}', set(in_memlets), {'__out'},
                            f'__out = {updated} if __mask else __in')

    map_node = nodes.MapEntry(
        nodes.Map(f'masked_write_{line}', parameters, subsets.Range([(0, size - 1, 1) for size in target_shape])))
    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.emitter.emit(
            tn.TaskletNode(node=tasklet,
                           in_memlets=in_memlets,
                           out_memlets={'__out': Memlet(data=container, subset=element, volume=1)}))


def substitute_masked_operands(value: ast.expr, state: LoweringState):
    """The assigned expression with its data reads replaced by connectors, for
    a masked write."""
    from dace.frontend.python.nextgen.lowering.access import substitute_data_operands
    return substitute_data_operands(value, state, connector_prefix='__val')


def resolve_single_boolean_mask(node: ast.Subscript, expr: MemletExpr, container: str, context, inference) -> str:
    """
    The single boolean-mask container an access uses, when the *whole* index
    is exactly one full-coverage boolean mask -- no other index arrays, no
    partial coverage. Shared by mask writes (:func:`emit_masked_write`) and
    mask reads (:func:`emit_boolean_gather`).

    :raises UnsupportedFeatureError: If combined with an integer index array,
        if more than one boolean mask is present, or if the mask's shape does
        not cover the indexed region.
    """
    index_map = index_arrays(expr)
    masks = [index for index in index_map.values() if _index_dtype(index, context) == dtypes.bool]
    if len(masks) != len(index_map):
        raise UnsupportedFeatureError('Boolean-mask indexing cannot be combined with integer array indexing',
                                      context.filename,
                                      node,
                                      category='advanced-indexing')
    if len(masks) != 1:
        raise UnsupportedFeatureError('Only a single boolean mask index is supported',
                                      context.filename,
                                      node,
                                      category='advanced-indexing')
    mask_container = _index_container(masks[0], next(iter(index_map)), container, context, inference, node)
    mask_shape = list(context.containers[mask_container].shape)
    target_shape = list(expr.subset.size())
    # Dimensions the subscript pinned to one element (``A[0, mask]``) select a
    # region the mask still covers exactly, element for element; what must
    # agree is the region the mask spans, not the rank of the subscript. A mask
    # that covers only PART of the selected region (a 1-D mask against
    # ``A[:, mask]``, whose NumPy result is 2-D) does not, and is still refused.
    if [size for size in mask_shape if size != 1] != [size for size in target_shape if size != 1]:
        raise UnsupportedFeatureError(
            f'Boolean mask of shape {tuple(mask_shape)} does not cover the indexed '
            f'region of shape {tuple(target_shape)}',
            context.filename,
            node,
            category='advanced-indexing')
    return mask_container


def emit_boolean_gather(target_name: str,
                        base_container: str,
                        base_descriptor: data.Data,
                        mask_container: str,
                        statement: ast.stmt,
                        state: LoweringState,
                        base_subset: Optional[subsets.Range] = None) -> DataAccess:
    """
    Lower ``B = A[mask]`` (a full-coverage boolean-mask read): a
    data-dependent-size gather.

    Only the bare top-level assignment reaches here (see
    ``rules.assign._lower_boolean_gather_assign``); the general advanced-index
    read path (:func:`analyze`/:func:`emit_gather`) still rejects a boolean
    mask everywhere else (nested in an expression, combined with another
    index), matching classic.

    Two configurable strategies (``frontend.boolean_index_strategy``), both
    verified independently of this frontend in
    ``tests/sdfg/deferred_symbol_boolean_filter_test.py``:

    - ``'view'`` (default) -- ONE pass over ``A``/``mask``: a stream push,
      fused with a WCR sum computing the count, into an upper-bound
      (source-sized) backing buffer; the result is a :class:`~...treenodes.ViewNode`
      over the buffer's first ``M`` elements. Cheaper in compute, wastes
      ``N - M`` elements of backing memory.
    - ``'exact'`` -- TWO passes: a pure count-only reduction resolves ``M``
      first (no per-element writes at all), then a compaction pass whose
      stream drains directly into an exactly ``M``-sized array. Costs a
      second read of ``mask``, never over-allocates.

    Both mint a fresh SDFG symbol from the runtime-computed count via a plain
    interstate-edge assignment (:class:`~...treenodes.AssignNode`) -- the same
    "frontend mints a symbol, registers it directly in the repository's symbol
    table" shape ``lowering.rules.control_flow._dynamic_bound`` already uses
    for dynamic map bounds, generalized here to an ordinary (non-map-range)
    symbol. The WCR accumulator is explicitly zeroed first: its initial memory
    is uninitialized garbage, not zero -- omitting this produces a build that
    compiles cleanly and often *appears* to work, then segfaults
    non-deterministically (see the referenced test's module docstring).

    ``B`` is registered as an ordinary (transient) local container in both
    strategies, exactly like any other computed intermediate -- NOT forced
    non-transient, even though that would make it directly callable from
    outside the compiled program (as the referenced test does by hand): a
    compiled SDFG's calling convention needs every argument's memory to
    already exist before the call, which is impossible for a size only
    known mid-call, so an always-external ``B`` would silently demand a
    caller-supplied ``M``/``B`` for a plain local variable that may never
    leave the program. If ``B`` is genuinely the program's *return* value,
    the same boundary limitation applies regardless of this function --
    unsupported for now, tracked as future work (a two-kernel count-then-fill
    dispatch pattern), not specific to either strategy here.
    """
    from dace.config import Config

    dtype = base_descriptor.dtype
    # The region the mask selects from: the whole container by default, or the
    # sub-region a partially-indexed subscript pinned (``A[0, mask]``).
    base_region = base_subset if base_subset is not None else subsets.Range.from_array(base_descriptor)
    mask_region = subsets.Range.from_array(state.context.containers[mask_container])
    source_size = data._prod(base_region.size())
    strategy = Config.get('frontend', 'boolean_index_strategy')
    line = getattr(statement, 'lineno', 0)

    nnz_container = state.context.add_container(f'__nnz_{line}', data.Array(dtypes.uint32, [1]), transient=True)
    _emit_zero_init(nnz_container, state)
    # What the count IS, for diagnostics about containers this symbol sizes:
    # the mask is itself usually a canonicalization temporary, so resolve it
    # back to the expression the user wrote.
    count_description = f'the number of elements "{state.context.describe_expression(mask_container)}" selects'

    if strategy == 'exact':
        _emit_boolean_count(base_container, mask_container, nnz_container, base_region, mask_region, source_size, line,
                            state)
        count_symbol = _resolve_symbol_from_scalar(nnz_container, f'nnz{line}', count_description, state)
        result_container = state.context.add_container(target_name, data.Array(dtype, [count_symbol]))
        _emit_boolean_compact(base_container, mask_container, result_container, base_region, mask_region, source_size,
                              count_symbol, line, state)
    else:
        buf_container = state.context.add_container(f'__buf_{line}', data.Array(dtype, [source_size]), transient=True)
        _emit_boolean_count_and_fill(base_container, mask_container, buf_container, nnz_container, base_region,
                                     mask_region, source_size, line, state)
        count_symbol = _resolve_symbol_from_scalar(nnz_container, f'nnz{line}', count_description, state)
        result_container = state.context.add_container(target_name,
                                                       data.ArrayView(dtype, [count_symbol]),
                                                       transient=True)
        view_descriptor = state.context.containers[result_container]
        buf_descriptor = state.context.containers[buf_container]
        state.emitter.emit(
            tn.ViewNode(target=result_container,
                        source=buf_container,
                        memlet=Memlet(data=buf_container, subset=subsets.Range([(0, count_symbol - 1, 1)])),
                        src_desc=buf_descriptor,
                        view_desc=view_descriptor))

    descriptor = state.context.containers[result_container]
    return DataAccess(result_container, subsets.Range.from_array(descriptor), descriptor)


def _emit_zero_init(container: str, state: LoweringState) -> None:
    """
    Zero a freshly-allocated scalar-shaped accumulator before a WCR reduction
    runs into it. Its allocated memory is NOT zero-initialized by default --
    skipping this produces a build that compiles cleanly and typically
    *appears* to work, then segfaults non-deterministically depending on what
    happened to already be on the heap (found the hard way, see
    ``tests/sdfg/deferred_symbol_boolean_filter_test.py``'s module docstring).
    """
    tasklet = nodes.Tasklet('zero_init', set(), {'z'}, 'z = 0')
    state.emitter.emit(
        tn.TaskletNode(node=tasklet,
                       in_memlets={},
                       out_memlets={'z': Memlet(data=container, subset=subsets.Range([(0, 0, 1)]), volume=1)}))


def _resolve_symbol_from_scalar(scalar_container: str, hint: str, description: str,
                                state: LoweringState) -> symbolic.symbol:
    """
    Mint a fresh SDFG symbol and resolve its value from a scalar container
    computed earlier in this same lowering, via a plain interstate-edge
    assignment. Registered directly in the repository's symbol table, the
    same shape ``lowering.rules.control_flow._dynamic_bound`` uses for
    dynamic map bounds (``__dynN``) -- this is an ordinary, non-map-range
    symbol instead, usable in any later container's shape.

    :param scalar_container: Container holding the runtime value.
    :param hint: Name hint for the generated symbol.
    :param description: Human-readable description of the quantity the symbol
                        holds, shown by diagnostics about containers sized by
                        it (the generated name means nothing to a reader).
    :param state: The active lowering state.
    :return: The freshly minted symbol.
    """
    name = state.context.fresh_name(f'__{hint}')
    symbol = symbolic.symbol(name, dtypes.int64)
    state.context.symbols[name] = symbol
    # Its value exists only once the program is running, which is what makes a
    # container sized by it unusable across the program boundary.
    state.context.deferred_symbols[name] = description
    assignment = f'{scalar_container}[0]'
    state.emitter.emit(
        tn.AssignNode(name=name, value=CodeBlock(assignment), edge=InterstateEdge(assignments={name: assignment})))
    return symbol


def _flat_index_subset(region: subsets.Range, parameter: symbolic.symbol) -> subsets.Range:
    """
    Index a region of any rank by a single FLAT counter, in C order.

    ``A[mask]`` reads the selected elements in C order regardless of ``A``'s
    rank, so the filter below walks one counter over the region's element count
    and unravels it here. Keeping the map one-dimensional whatever the rank
    means the traversal -- and therefore the order elements reach the stream,
    which IS the result order -- is the same for every input.

    Unravelling against a REGION rather than a whole container is what lets a
    mask cover a sub-region (``A[0, mask]``): each pinned dimension contributes
    its fixed offset and a single position.

    The slowest axis needs no modulus (the counter never reaches the region's
    element count), which makes this exactly ``[(i, i, 1)]`` again for a whole
    1-D source.
    """
    sizes = list(region.size())
    ranges: List[Tuple[Any, Any, int]] = []
    stride = 1
    for position, (size, (start, _, step)) in enumerate(zip(reversed(sizes), reversed(list(region.ranges)))):
        index = parameter if stride == 1 else symbolic.int_floor(parameter, stride)
        if position < len(sizes) - 1:
            index = index % size
        offset = start + index * step
        ranges.append((offset, offset, 1))
        stride = stride * size
    ranges.reverse()
    return subsets.Range(ranges)


def _boolean_filter_map(base_container: str, mask_container: str, base_region: subsets.Range,
                        mask_region: subsets.Range, source_size, name: str):
    """The shared map header (a flat index ``i`` over ``[0, source_size)``) and
    its two source-read memlets, for both the count-only and count+fill maps."""
    parameter = symbolic.symbol('i')
    # Two distinct subset objects -- SDFG validation rejects two memlets
    # sharing one (same footgun documented on ``emit_masked_write``'s ``__in``
    # memlet above).
    # Underscore-prefixed connector names: a bare ``a``/``m`` collides with a
    # program parameter of the same name, which SDFG validation rejects.
    in_memlets = {
        '__a': Memlet(data=base_container, subset=_flat_index_subset(base_region, parameter), volume=1),
        '__m': Memlet(data=mask_container, subset=_flat_index_subset(mask_region, parameter), volume=1),
    }
    map_node = nodes.MapEntry(nodes.Map(name, ['i'], subsets.Range([(0, source_size - 1, 1)])))
    return map_node, in_memlets


def _emit_boolean_count(base_container: str, mask_container: str, nnz_container: str, base_region: subsets.Range,
                        mask_region: subsets.Range, source_size, line: int, state: LoweringState) -> None:
    """'exact' strategy, pass 1: a pure count, no per-element writes at all."""
    map_node, in_memlets = _boolean_filter_map(base_container, mask_container, base_region, mask_region, source_size,
                                               f'boolean_count_{line}')
    in_memlets = {'__m': in_memlets['__m']}  # the count doesn't need to read A at all
    tasklet = nodes.Tasklet(f'boolean_count_{line}', {'__m'}, {'__osz'}, '__osz = 1 if __m else 0')
    out_memlets = {
        '__osz':
        Memlet(data=nnz_container, subset=subsets.Range([(0, 0, 1)]), dynamic=True, volume=0, wcr='lambda x, y: x + y')
    }
    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))


def _emit_boolean_count_and_fill(base_container: str, mask_container: str, buf_container: str, nnz_container: str,
                                 base_region: subsets.Range, mask_region: subsets.Range, source_size, line: int,
                                 state: LoweringState) -> None:
    """'view' strategy: one pass, fused count + compaction push into an
    upper-bound-sized backing buffer."""
    dtype = state.context.containers[base_container].dtype
    stream_container = state.context.add_container(f'__stream_{line}', data.Stream(dtype, 1), transient=True)
    map_node, in_memlets = _boolean_filter_map(base_container, mask_container, base_region, mask_region, source_size,
                                               f'boolean_filter_{line}')
    tasklet = nodes.Tasklet(f'boolean_filter_{line}', {'__a', '__m'}, {'__b', '__osz'},
                            'if __m:\n    __b = __a\n__osz = 1 if __m else 0')
    out_memlets = {
        '__b':
        Memlet(data=stream_container, subset=subsets.Range([(0, 0, 1)]), dynamic=True, volume=0),
        '__osz':
        Memlet(data=nnz_container, subset=subsets.Range([(0, 0, 1)]), dynamic=True, volume=0, wcr='lambda x, y: x + y'),
    }
    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))
    state.emitter.emit(
        tn.CopyNode(target=buf_container,
                    memlet=Memlet(data=stream_container,
                                  subset=subsets.Range([(0, 0, 1)]),
                                  other_subset=subsets.Range([(0, source_size - 1, 1)]))))


def _emit_boolean_compact(base_container: str, mask_container: str, result_container: str, base_region: subsets.Range,
                          mask_region: subsets.Range, source_size, count_symbol: symbolic.symbol, line: int,
                          state: LoweringState) -> None:
    """'exact' strategy, pass 2: compact directly into the exactly-sized
    result array (its stream never needs to hold more than ``count_symbol``
    elements, since that many is exactly how many the mask selects)."""
    dtype = state.context.containers[base_container].dtype
    stream_container = state.context.add_container(f'__stream_{line}', data.Stream(dtype, 1), transient=True)
    map_node, in_memlets = _boolean_filter_map(base_container, mask_container, base_region, mask_region, source_size,
                                               f'boolean_compact_{line}')
    tasklet = nodes.Tasklet(f'boolean_compact_{line}', {'__a', '__m'}, {'__b'}, 'if __m:\n    __b = __a')
    out_memlets = {'__b': Memlet(data=stream_container, subset=subsets.Range([(0, 0, 1)]), dynamic=True, volume=0)}
    with state.emitter.scope(tn.MapScope(node=map_node, children=[])):
        state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))
    state.emitter.emit(
        tn.CopyNode(target=result_container,
                    memlet=Memlet(data=stream_container,
                                  subset=subsets.Range([(0, 0, 1)]),
                                  other_subset=subsets.Range([(0, count_symbol - 1, 1)]))))


def _value_subset(operand: DataAccess, access: AdvancedIndex) -> subsets.Range:
    """
    Index a right-hand-side operand by the result index space, NumPy-broadcast
    against it (``A[indices] = B`` with ``B`` one rank short repeats ``B`` over
    the indexed axis).
    """
    parameters = [symbolic.symbol(parameter) for parameter in access.output_subset.free_symbols]
    result = list(access.output_subset.ranges)
    ranges = []
    offset = len(result) - len(operand.subset.ranges)
    for dimension, (start, end, step) in enumerate(operand.subset.ranges):
        if start == end:
            ranges.append((start, end, 1))
            continue
        position = dimension + offset
        if position < 0 or position >= len(result):
            ranges.append((start, end, step))
            continue
        index, _, _ = result[position]
        ranges.append((start + index * step, start + index * step, 1))
    return subsets.Range(ranges)


def _target_subset(target: DataAccess, access: AdvancedIndex) -> subsets.Range:
    """
    Map the access's result subset onto the write target.

    The target's own subset is authoritative for placement (``B[0:3] = A[ind]``
    writes into ``B`` at an offset), so the result ranges are offset by the
    target's start in each surviving dimension.

    Both sides are laid out by an :class:`~...semantics.indexing.IndexPlan`, so
    result axis *i* of the access lands on result axis *i* of the target.
    Matching them by walking the target's subset and skipping degenerate
    dimensions instead got this wrong as soon as the result contained a
    ``newaxis``: the size-1 entry was skipped on the target side but still
    consumed a position on the access side, shifting every axis after it.
    """
    result = list(access.output_subset.ranges)
    if target.plan is None:
        # Unmodeled target: the previous positional matching over
        # non-degenerate dimensions is all that is available.
        ranges, result_index = [], 0
        for start, end, step in target.subset.ranges:
            if start == end:
                ranges.append((start, end, 1))
            elif result_index >= len(result):
                ranges.append((start, end, step))
            else:
                offset = result[result_index][0]
                ranges.append((start + offset * step, start + offset * step, 1))
                result_index += 1
        return subsets.Range(ranges)

    layout = target.plan.result_layout()
    # Right-align, as NumPy does when the value has fewer axes than the target.
    alignment = len(layout) - len(result)
    ranges: List[Optional[Tuple]] = [None] * len(target.subset.ranges)
    for position, entry in enumerate(layout):
        if entry.kind != SOURCE:
            continue
        start, end, step = target.subset.ranges[entry.axis]
        source = position - alignment
        if source < 0:
            ranges[entry.axis] = (start, end, step)  # Broadcast across this axis
            continue
        offset = result[source][0]
        index = start + offset * step
        ranges[entry.axis] = (index, index, 1)
    for axis, (start, _, _) in enumerate(target.subset.ranges):
        if ranges[axis] is None:
            ranges[axis] = (start, start, 1)  # Dropped by the target's subscript
    return subsets.Range(ranges)


def _broadcast_index_shape(expr: MemletExpr,
                           context,
                           inference,
                           node: ast.expr,
                           index_types: Optional[Dict[str, data.Data]] = None) -> Tuple[Any, ...]:
    """The shape all index arrays broadcast to, which becomes the advanced
    chunk of the result."""
    from dace.frontend.python.replacements.utils import broadcast_together

    result: Optional[Tuple[Any, ...]] = None
    for index in index_arrays(expr).values():
        shape = _index_shape(index, context, inference, node, index_types)
        result = tuple(shape) if result is None else broadcast_together(shape, result)[0]
    if result is None:
        raise UnsupportedFeatureError('Advanced indexing with no index arrays',
                                      context.filename,
                                      node,
                                      category='advanced-indexing')
    return result


def _broadcast_subset(shape: Sequence[Any], parameters: Sequence[str]) -> subsets.Range:
    """
    Subset an index array by the broadcast map parameters, right-aligned per
    NumPy broadcasting: a missing or singleton dimension is pinned to 0 so the
    same element is read across the broadcast extent.
    """
    ranges = []
    offset = len(parameters) - len(shape)
    for dimension, size in enumerate(shape):
        if size == 1:
            ranges.append((0, 0, 1))
            continue
        parameter = symbolic.symbol(parameters[offset + dimension])
        ranges.append((parameter, parameter, 1))
    return subsets.Range(ranges)


def _index_shape(index: Any,
                 context,
                 inference,
                 node: ast.expr,
                 index_types: Optional[Dict[str, data.Data]] = None) -> Sequence[Any]:
    """The shape of an index array, whether it is a registered container or a
    literal sequence in the subscript."""
    if isinstance(index, str):
        descriptor = _index_descriptor(index, context, index_types)
        if descriptor is not None:
            return descriptor.shape
        static = _static_index_sequence(index, context, inference, node)
        if static is not None:
            return static.shape
        raise UnsupportedFeatureError(f'Index array "{index}" is not a known container',
                                      context.filename,
                                      node,
                                      category='advanced-indexing')
    return _literal_index_array(index, inference, node).shape


def _index_dtype(index: Any, context, index_types: Optional[Dict[str, data.Data]] = None) -> Optional[dtypes.typeclass]:
    if isinstance(index, str):
        descriptor = _index_descriptor(index, context, index_types)
        return None if descriptor is None else descriptor.dtype
    try:
        values = [value for value in index]
    except TypeError:
        return None
    return dtypes.bool if values and all(isinstance(value, (bool, numpy.bool_)) for value in values) else None


def _index_descriptor(name: str, context, index_types: Optional[Dict[str, data.Data]] = None) -> Optional[data.Data]:
    binding = context.resolve(name)
    if binding is not None and binding.kind == 'container':
        return context.containers.get(binding.container)
    if name in context.containers:
        return context.containers[name]
    # An array-valued index EXPRESSION inference gave a placeholder name for
    # typing. It has a shape and a dtype but no storage, which is all the shape
    # rules here need; the emitting paths only ever see the real container
    # lowering materializes for it. The placeholders belong to the single parse
    # that introduced them and are passed in, not held on the context.
    return (index_types or {}).get(name)


def _index_container(index: Any, dimension: int, container: str, context, inference, node: ast.expr) -> str:
    """
    The registered container holding an index array, materializing a literal
    sequence (``A[[1, 10, 15]]``, ``A[:, (1, 2, 3)]``) into a constant one.
    """
    if isinstance(index, str):
        binding = context.resolve(index)
        if binding is not None and binding.kind == 'container':
            return binding.container
        if index in context.containers:
            return index
        # A literal index sequence in the subscript (``A[:, (1, 2, 3)]``) is
        # hoisted by ANF into a name bound to a compile-time Python sequence,
        # so the parser reports the temporary's *name* rather than the values.
        literal = _static_index_sequence(index, context, inference, node)
        if literal is None:
            raise UnsupportedFeatureError(f'Index array "{index}" is not a known container',
                                          context.filename,
                                          node,
                                          category='advanced-indexing')
        return _constant_index_container(literal, dimension, container, context)

    literal = _literal_index_array(index, inference, node)
    return _constant_index_container(literal, dimension, container, context)


def _constant_index_container(literal: numpy.ndarray, dimension: int, container: str, context) -> str:
    """Register a compile-time index array as a constant container."""
    descriptor = data.Array(dtypes.dtype_to_typeclass(literal.dtype.type), list(literal.shape), transient=True)
    return context.add_constant_container(f'__ind{dimension}_{container}', descriptor, literal)


def _static_index_sequence(name: str, context, inference, node: ast.expr) -> Optional[numpy.ndarray]:
    """The constant array behind a name bound to a compile-time Python
    sequence, or None if the name is not such a binding."""
    binding = context.resolve(name)
    if binding is None or binding.kind != 'static':
        return None
    sequence = context.static_values.get(name)
    if sequence is None:
        return None
    try:
        values = inference.sequence_constants(sequence)
    except UnsupportedFeatureError:
        return None
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return None
    return numpy.array(values, dtype=dtypes.typeclass(int).type)


def _literal_index_array(index: Any, inference, node: ast.expr) -> numpy.ndarray:
    """A literal index sequence in the subscript, resolved to a constant array."""
    values = []
    for element in index:
        if isinstance(element, Number):
            values.append(element)
            continue
        resolved = inference.constant_int(element) if isinstance(element, ast.AST) else None
        if resolved is None:
            raise UnsupportedFeatureError('Index sequence element is not a compile-time integer',
                                          inference.context.filename,
                                          node,
                                          category='advanced-indexing')
        values.append(resolved)
    return numpy.array(values, dtype=dtypes.typeclass(int).type)
