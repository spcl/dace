# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering rule for canonical assignments.

The rule owns *binding semantics* — the decisions that follow Python and NumPy
assignment rules rather than computation:

- compile-time Python sequences (``a = [1, 2, x]``, concatenation, repetition,
  static indexing/slicing) stay in the value domain and emit no nodes,
- whole-array aliasing (``b = A``) is a binding update (arrays are mutable),
- slice reads (``b = A[1:4]``) bind as :class:`ViewNode` (NumPy basic-indexing
  view semantics),
- subset-to-subset copies (``B[2:4] = A[0:2]``) become :class:`CopyNode`,
- names declared as :class:`~dace.data.Reference` (via annotation) are runtime
  aliases: every assignment to them emits :class:`RefSetNode` instead of a
  binding update, and assigning *from* a reference-bound name materializes the
  target as a fresh reference (a pointer copy — a compile-time binding to a
  reference container would break when that reference is later re-set, e.g.
  in a double-buffering swap).

Everything computational is delegated to the type-directed dispatch seam
(:mod:`~dace.frontend.python.nextgen.lowering.dispatch`), which selects a
mechanism by operand types (elementwise, materialization, callback).

Scalar Python names are deliberately materialized as scalar containers with
in-place writes (not interstate symbols), which keeps loop-carried updates
correct without SSA φ-resolution. Symbol promotion of compile-time scalars is
a planned optimization pass, not a correctness requirement.
"""
import ast
import copy
import numbers
from typing import Any, Optional, Tuple

import numpy

from dace import data, dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python.nextgen.semantics import structures as structure_support
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering import dispatch
from dace.frontend.python.nextgen.lowering.access import (DataAccess, lower_indirect_write, nondegenerate_shape,
                                                          resolve_access)
from dace.frontend.python.nextgen.lowering.mechanisms import conflict, static_values
from dace.frontend.python.nextgen.lowering.registry import LoweringState, rule
from dace.frontend.python.nextgen.semantics.inference import Inferred


def _aliasing_required(statement: ast.stmt) -> bool:
    """
    Whether this binding was hoisted out of an assignment TARGET's base by
    canonicalization (``ANFTransform._flatten_target``) and therefore has to
    alias what it names -- a write through the outer subscript must reach the
    source, so a binding that would only copy is refused instead.
    """
    return getattr(statement, 'aliasing_required', False)


@rule(ast.Assign)
def lower_assign(statement: ast.Assign, state: LoweringState) -> None:
    target = statement.targets[0]
    value = statement.value

    # Declared-type hint from a desugared annotated assignment: pre-bind the
    # target container so every lowering path (including callbacks whose
    # results inference cannot see) types it.
    if isinstance(target, ast.Name):
        apply_annotation_hint(target.id, statement, state)
        if isinstance(value, ast.Subscript) and _lower_boolean_gather_assign(target, value, statement, state):
            return

    if isinstance(value, ast.Call):
        from dace.frontend.python.nextgen.lowering.rules import calls
        calls.lower_call_assign(statement, state)
        return

    # Accumulation into a DATA-DEPENDENT target (``hist[bin] += 1``): the
    # written element is chosen at runtime, so no analysis of the emitted
    # dataflow can tell whether iterations collide, and the syntactic fact that
    # this is an accumulation is the only evidence there is. Every other
    # accumulation is left to ``ResolveWriteConflicts``, which sees the whole
    # region rather than one statement -- including the self-read ANF hoists
    # out of it, which is what made deciding it here unsound.
    wcr = conflict.accumulation_wcr(statement, state)
    if wcr is not None and _lower_indirect_accumulation(target, statement, wcr, state):
        return

    # Descriptor PROPERTY reads (``A.shape[0]``, ``A.ndim``) are compile-time
    # values, but the computation path would resolve ``A.shape`` as data and
    # see all of ``A``. Substitute the value they denote before anything tries
    # to lower them.
    value = statement.value = static_values.fold_descriptor_properties(value, state)

    # Value-domain handling: sequence literals and operations on them that
    # fold at compile time bind statically without emitting nodes.
    try:
        inferred = state.inference.infer(value)
    except UnsupportedFeatureError as reason:
        # ``b = A.flat[10:15]`` / ``b = A.T[0:2]``: inference types the base
        # attribute, but the shared memlet parser then sees "A.flat" as a data
        # NAME and rejects it. Materializing the attribute read gives the
        # subscript a real container to index, after which both inference and
        # lowering proceed normally. Attempted only after a failure, so no
        # attribute is materialized speculatively.
        rewritten = (dispatch.rewrite_attribute_subscript_base(value, state, writable=_aliasing_required(statement))
                     if isinstance(value, ast.Subscript) else value)
        if rewritten is value:
            dispatch.fallback_to_callback(statement, state, reason)
            return
        value = statement.value = rewritten
        try:
            inferred = state.inference.infer(value)
        except UnsupportedFeatureError as retried:
            dispatch.fallback_to_callback(statement, state, retried)
            return
    if isinstance(target, ast.Name) and _index_temporary(statement) and _bind_index_symbol(
            target.id, value, inferred, state):
        return

    if inferred.kind == 'static':
        if isinstance(target, ast.Name) and _reference_binding(target.id, state) is None:
            state.context.bind_static(target.id, inferred.value)
            return
        # Static value written into a container subset: materialize first
        access = static_values.materialize(inferred.value, state)
        value = ast.copy_location(ast.Name(id=access.container, ctx=ast.Load()), value)

    if isinstance(target, ast.Name):
        _lower_name_assign(target, value, inferred, statement, state)
    elif isinstance(target, ast.Attribute):
        _lower_member_assign(target, value, inferred, statement, state)
    elif isinstance(target, ast.Subscript):
        _lower_subscript_assign(target, value, statement, state)
    else:
        raise UnsupportedFeatureError(f'Unsupported assignment target: {astutils.unparse(target)}',
                                      state.context.filename,
                                      statement,
                                      category='assign-target')


def _index_temporary(statement: ast.stmt) -> bool:
    """Whether canonicalization hoisted this binding out of a subscript INDEX
    (``ANFTransform._flatten_index``)."""
    return getattr(statement, 'index_temporary', False)


def _bind_index_symbol(name: str, value: ast.expr, inferred: Inferred, state: LoweringState) -> bool:
    """
    Bind a temporary hoisted out of an index position as a SYMBOL rather than
    materializing it as a scalar transient.

    ``doc/extensions/symbolic.rst`` ("Symbolic types vs. scalars") puts a
    quantity that is set once and consumed in a range on the symbol side, and a
    subscript bound is exactly that. Materializing it as a container instead
    put a data name where a memlet expects a symbol -- which is refused
    outright inside a dataflow scope, and even outside one keeps sizes like
    ``__anf0 * 96 - i * 96`` from folding to 96.

    Two ways to get there, both of which must work (an index is an index):

    - a compile-time SYMBOLIC value (``i + 1`` over map parameters and symbols)
      IS the expression; the name binds to it and nothing is emitted, so the
      subset carries the arithmetic and the symbolic machinery folds it;
    - a value read from DATA (``n = bounds[0]`` … ``A[:n]``) defines a real
      symbol through an interstate assignment
      (:class:`~...treenodes.AssignNode`), the same mechanism dynamic map
      bounds and the advanced-indexing gather use.

    :return: True when the name was bound as a symbol (nothing else to lower).
    """
    if inferred.kind in ('symbolic', 'constant') and (symbolic.issymbolic(inferred.value)
                                                      or isinstance(inferred.value, numbers.Integral)):
        state.context.bind_constant(name, inferred.value)
        return True
    if not inferred.is_data or not isinstance(inferred.descriptor.dtype, (dtypes.typeclass, )):
        return False
    if not numpy.issubdtype(inferred.descriptor.dtype.type, numpy.integer):
        return False  # A non-integer index is not a symbol; let the ordinary paths report it
    try:
        access = resolve_access(value, state)
    except UnsupportedFeatureError:
        return False
    if access is None:
        return False
    read = _scalar_read_expression(access)
    if read is None:
        return False
    symbol_name = state.context.fresh_name(f'__idx_{name.lstrip("_")}_')
    state.context.symbols[symbol_name] = symbolic.symbol(symbol_name, access.descriptor.dtype)
    state.emitter.emit(
        tn.AssignNode(name=symbol_name, value=CodeBlock(read), edge=InterstateEdge(assignments={symbol_name: read})))
    state.context.bind_symbol(name, access.descriptor.dtype, symbol_name=symbol_name)
    return True


def _scalar_read_expression(access: DataAccess) -> Optional[str]:
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


def _lower_indirect_accumulation(target: ast.expr, statement: ast.Assign, wcr: str, state: LoweringState) -> bool:
    """
    Lower an accumulation into a data-dependent target (``hist[bin] += 1``) as
    a conflict-resolved indirect write of the accumulated value alone, dropping
    the self-read that the conflict resolution subsumes.

    This is the ONE accumulation the frontend still decides. The written
    element is chosen at runtime, so ``ResolveWriteConflicts`` classifies such
    a write as undecidable — it can neither prove nor rule out a collision —
    and the syntactic fact that the statement is an accumulation is the only
    evidence available. Every accumulation into a plain access returns False
    here and is left to that pass, which reads the emitted dataflow instead of
    one statement.

    Returning False for anything else also keeps the commuted form
    (``b = x + b``) safe for non-numeric values: swapping the operands of ``+``
    would reverse a Python sequence concatenation, and a compile-time list or
    string never reaches an indirect write.
    """
    if not isinstance(target, ast.Subscript) or not isinstance(statement.value, ast.BinOp):
        return False
    # The accumulated value is whichever operand is not the self-read. A
    # desugared ``AugAssign`` carries no side marker and is always left-folded.
    accumulated = (statement.value.left
                   if getattr(statement, 'accumulator_side', 'left') == 'right' else statement.value.right)
    try:
        resolve_access(target, state)
    except UnsupportedFeatureError:
        # Not a plain access, but possibly an indirect one -- see
        # lowering.access.lower_indirect_write.
        return lower_indirect_write(target, accumulated, statement, state, wcr=wcr)
    return False


def _lower_name_assign(target: ast.Name, value: ast.expr, inferred: Inferred, statement: ast.Assign,
                       state: LoweringState) -> None:
    # Names bound to Reference containers re-point at runtime: assignments are
    # reference sets, never binding updates.
    reference = _reference_binding(target.id, state)
    if reference is not None:
        _lower_reference_set(target.id, reference, value, inferred, statement, state)
        return

    # Registry ATTRIBUTE-family reads (``b = A.T``/``.real``/``.imag``/
    # ``.flat``) that need an actual data operation: tried before the
    # generic Name/Attribute aliasing path below (which only recognizes
    # structure members), the same placement principle as
    # ``dispatch._lower_reshape_call`` being tried before generic call
    # dispatch.
    if isinstance(value, ast.Attribute) and dispatch.lower_attribute_assign(target, value, state):
        return

    if isinstance(value, (ast.Name, ast.Attribute)):
        source_access = resolve_access(value, state)
        # Assigning from a reference-bound name materializes a fresh reference
        # (pointer copy): the source reference may later be re-set, so a
        # compile-time alias to its container would be unsound.
        if source_access is not None and isinstance(source_access.descriptor, data.Reference):
            reference_descriptor = copy.deepcopy(source_access.descriptor)
            container = state.context.add_container(target.id, reference_descriptor)
            state.context.bind(target.id, container)
            state.emitter.emit(
                tn.RefSetNode(target=container,
                              memlet=Memlet(data=source_access.container, subset=source_access.subset),
                              src_desc=source_access.descriptor,
                              ref_desc=copy.deepcopy(reference_descriptor)))
            return
        # Whole-structure aliasing (x = outer.inner, including the ANF temps
        # that reduce nested member chains to single-level datarefs): a
        # compile-time binding to the dotted member path. Sound because
        # structure members are embedded, never re-pointed.
        if source_access is not None and structure_support.supports_member_access(source_access.descriptor):
            state.context.bind(target.id, source_access.container)
            return
        # Whole-array aliasing: rebind the name. Arrays are mutable in Python,
        # so both names must observe subsequent writes; a binding update
        # models that.
        if source_access is not None and isinstance(source_access.descriptor, data.Array):
            state.context.bind(target.id, source_access.container)
            return

    # Slice reads bind as views so writes through the new name reach the
    # original container (NumPy basic-indexing semantics). The result shape
    # follows NumPy indexing: slice-formed dimensions survive (a[0:20, 1:2]
    # is (20, 1)), integer-indexed dimensions are dropped (a[0:20, 1] is
    # (20,)); a fully integer-indexed access is a scalar element read and
    # lowers as a computation instead.
    if isinstance(value, ast.Subscript):
        try:
            access = resolve_access(value, state)
        except UnsupportedFeatureError:
            # Not a directly resolvable access (e.g. an indirect index like
            # ``x[A_col[j]]``): falls through to the computation path below,
            # which lowers indirection through the elementwise mechanism.
            access = None
        if access is not None and access.kept_dims is None and access.subset.size() and access.new_axes:
            # ``DataAccess.numpy_shape``'s fallback for a genuinely unmodeled
            # index form (more elements than dimensions, more than one
            # ``Ellipsis``, ...) blindly squeezes every size-1 subset
            # dimension via ``nondegenerate_shape`` -- fine on its own since
            # there is nothing to disambiguate, but combined with ``new_axes``
            # insertions (positions computed against the SAME assumed shape)
            # it can silently produce a wrong result. ``kept_dimensions``
            # already handles the ordinary Ellipsis/newaxis cases correctly
            # (see its docstring); this only remains reachable for whatever
            # it still can't model, so fail loudly instead of guessing.
            raise UnsupportedFeatureError('Cannot determine the exact NumPy result shape of this subscript',
                                          state.context.filename,
                                          value,
                                          category='data-dependent-subscript')
        if access is not None and access.numpy_shape:
            _lower_view_binding(target, access, state)
            return

    if _aliasing_required(statement):
        # Canonicalization hoisted this binding out of an assignment TARGET's
        # base (``A[:, 1:5][:, 0:2] = ...``), so it must alias what it names.
        # Every path below computes a fresh container, through which a write
        # would be silently discarded.
        raise UnsupportedFeatureError(f'Assignment target base "{astutils.unparse(value)}" cannot bind a view',
                                      state.context.filename,
                                      statement,
                                      category='assign-target')

    # Constants with no C representation (enum classes, type objects, ...)
    # cannot materialize as containers; they bind as compile-time values so
    # downstream attribute reads and call arguments resolve at compile time.
    if inferred.kind == 'constant' and not _representable_constant(inferred.value):
        state.context.bind_constant(target.id, inferred.value)
        return

    target_access = prepare_name_target(target, inferred, state, statement)
    # Pure-symbolic values of ANF temporaries stay visible to inference (e.g.
    # as computed shape arguments) alongside the materialized scalar. ANF
    # temps are single-assignment, so the recorded value cannot go stale.
    if inferred.kind == 'symbolic' and target.id.startswith('__anf'):
        state.context.symbolic_scalar_values[target_access.container] = inferred.value
    dispatch.lower_computation(target_access, value, statement, state)


def _lower_boolean_gather_assign(target: ast.Name, value: ast.Subscript, statement: ast.Assign,
                                 state: LoweringState) -> bool:
    """
    Lower ``B = A[mask]`` (NumPy boolean-mask indexing on the RHS). Returns
    False when ``value`` is not exactly this shape, so the caller falls
    through to the ordinary assignment paths -- this includes any OTHER
    boolean-mask use (nested in an expression, combined with other indices):
    only the bare top-level assignment of the whole access is implemented,
    everything else still reaches the existing ``analyze()``/
    ``has_boolean_index`` rejection unchanged.

    Tried BEFORE generic inference (``state.inference.infer(value)`` in
    :func:`lower_assign`) rather than after it, deliberately: inference has no
    correct way to type ``B`` before the element count is computed (it would
    otherwise silently answer with the mask's OWN shape, treating it like an
    ordinary same-shape index array), so this performs typing and lowering
    together instead, in :func:`advanced_indexing.emit_boolean_gather`.
    """
    from dace.frontend.python.nextgen.lowering.mechanisms import advanced_indexing

    if not isinstance(value.value, (ast.Name, ast.Attribute)):
        return False
    try:
        base = resolve_access(value.value, state)
        if base is None:
            return False
        expr = state.inference.parse_access(value)
        if not expr.arrdims or not advanced_indexing.has_boolean_index(expr, state.context):
            return False
        mask_container = advanced_indexing.resolve_single_boolean_mask(value, expr, base.container, state.context,
                                                                       state.inference)
        access = advanced_indexing.emit_boolean_gather(target.id, base.container, base.descriptor, mask_container,
                                                       statement, state)
        state.context.bind(target.id, access.container)
        return True
    except UnsupportedFeatureError as reason:
        dispatch.fallback_to_callback(statement, state, reason)
        return True


def _lower_member_assign(target: ast.Attribute, value: ast.expr, inferred: Inferred, statement: ast.Assign,
                         state: LoweringState) -> None:
    """
    Lower an assignment to a whole structure member (``tracers.vapor = ...``).

    An EXISTING :class:`~dace.data.Reference` member can be assigned as a
    whole — the assignment re-points the member (a :class:`RefSetNode`),
    matching SDFG reference semantics. Any other existing member is embedded
    STORAGE, so assigning to it writes into that storage (``B.sum = tmp`` is a
    write of ``tmp`` into the member's element, ``B.y = A.y`` a copy) — the
    same treatment a newly created member gets below. Re-POINTING a
    non-reference member is what has no dataflow equivalent, and a value that
    does not fit its storage surfaces as a feature gap from the computation
    path. A member that does not exist yet on a
    :class:`~dace.data.PythonClass` base is created dynamically (mirroring
    the classic frontend's ``_ensure_pythonclass_member``, see
    :func:`_create_pythonclass_member`); attributes that are neither an
    existing member nor a creatable one are a feature gap.
    """
    label = astutils.unparse(target)
    access = resolve_access(target, state)
    if access is not None:
        if not isinstance(access.descriptor, data.Reference):
            dispatch.lower_computation(access, value, statement, state)
            return
        _lower_reference_set(label, (access.container, access.descriptor), value, inferred, statement, state)
        return

    created = _create_pythonclass_member(target, inferred, state)
    if created is None:
        raise UnsupportedFeatureError(f'Assignment to unsupported attribute "{label}"',
                                      state.context.filename,
                                      statement,
                                      category='structure-member')
    path, member_descriptor = created
    if isinstance(member_descriptor, data.Reference):
        _lower_reference_set(label, (path, member_descriptor), value, inferred, statement, state)
        return
    dispatch.lower_computation(DataAccess(path, subsets.Range.from_array(member_descriptor), member_descriptor), value,
                               statement, state)


def _create_pythonclass_member(target: ast.Attribute, inferred: Inferred,
                               state: LoweringState) -> Optional[Tuple[str, data.Data]]:
    """
    Dynamically create a new field on a :class:`~dace.data.PythonClass`
    structure, mirroring the classic frontend's dynamic attribute creation
    (``_ensure_pythonclass_member``): an array-valued assignment creates a
    :class:`~dace.data.Reference` member (set via :class:`RefSetNode`, a
    pointer copy on every subsequent assignment); any other value creates a
    plain member of the value's own type (written by a normal
    copy/computation). Members are non-transient (their storage belongs to
    the enclosing object, not an SDFG-scoped temporary). Other structure
    kinds (plain :class:`~dace.data.Structure`) have a fixed member set and
    do not support this.

    :return: The (dotted member path, registered descriptor) pair, or None if
             the target's base is not a name bound to a ``PythonClass`` or the
             value's descriptor cannot be determined.
    """
    if not isinstance(target.value, ast.Name):
        return None
    binding = state.context.resolve(target.value.id)
    if binding is None or binding.kind != 'container':
        return None
    base_container = binding.container
    base_descriptor = state.context.containers.get(base_container)
    if not isinstance(base_descriptor, data.PythonClass):
        return None
    member = target.attr
    if member in base_descriptor.members:
        return None  # Already a member: the normal resolve_access path handles it

    if inferred.is_data:
        member_descriptor = (data.Reference.view(inferred.descriptor)
                             if isinstance(inferred.descriptor, data.Array) else copy.deepcopy(inferred.descriptor))
    else:
        dtype = state.inference.dtype_of(inferred)
        if dtype is None:
            try:
                dtype = dtypes.typeclass(type(inferred.value))
            except (KeyError, TypeError):
                return None
        member_descriptor = data.Scalar(dtype)
    member_descriptor.transient = False
    base_descriptor.members[member] = member_descriptor
    return structure_support.structure_member_path(base_container, member), member_descriptor


def _lower_subscript_assign(target: ast.Subscript, value: ast.expr, statement: ast.Assign,
                            state: LoweringState) -> None:
    # ``A.flat[idx] = ...`` on a contiguous array: rewrite the base to the
    # materialized flat-view container (see
    # ``dispatch.rewrite_attribute_subscript_base``) so the ordinary machinery
    # below, which only resolves a plain-name or structure-member base,
    # can write through it -- NumPy's flatiter aliases the source, so this
    # must reach ``A``, not a disposable copy.
    target = dispatch.rewrite_attribute_subscript_base(target, state, writable=True)
    if _lower_advanced_index_write(target, value, statement, state):
        return
    try:
        target_access = resolve_access(target, state)
    except UnsupportedFeatureError as reason:
        # A data-dependent target (``hist[bin] = x``): not a plain access,
        # but possibly an indirect one. No WCR needed for a plain overwrite --
        # like NumPy's own fancy-index assignment, which of several colliding
        # writes wins is unspecified, not a hazard the frontend must guard.
        if lower_indirect_write(target, value, statement, state):
            return
        dispatch.fallback_to_callback(statement, state, reason)
        return
    if target_access is None:
        raise UnsupportedFeatureError(f'Assignment to unknown container "{astutils.unparse(target)}"',
                                      state.context.filename,
                                      statement,
                                      category='undefined-name')
    if isinstance(target_access.descriptor.dtype, dtypes.pyobject):
        # Writing INTO an opaque Python object (``__anf0[0] = 5`` where an
        # earlier fallback left ``__anf0`` a pyobject): the write has to run in
        # the interpreter too, or it lands in a container unrelated to whatever
        # the object refers to. The read side of the same propagation is
        # ``dispatch._consumes_pyobject``.
        raise UnsupportedFeatureError(f'Assignment into an opaque Python object "{astutils.unparse(target)}"',
                                      state.context.filename,
                                      statement,
                                      category='pyobject-propagation')

    # Subset-to-subset copy
    if isinstance(value, (ast.Name, ast.Subscript)):
        try:
            source_access = resolve_access(value, state)
        except UnsupportedFeatureError:
            # Not a directly resolvable access (e.g. a data-dependent index
            # like ``hist[bin]``): falls through to the computation path
            # below, which lowers indirection through the elementwise
            # mechanism -- the same guard already applied to the analogous
            # probe in ``_lower_name_assign``.
            source_access = None
        if (source_access is not None and not source_access.is_scalar_access and not target_access.is_scalar_access
                and nondegenerate_shape(source_access.subset) == nondegenerate_shape(target_access.subset)):
            state.emitter.emit(
                tn.CopyNode(target=target_access.container,
                            memlet=Memlet(data=source_access.container,
                                          subset=source_access.subset,
                                          other_subset=target_access.subset)))
            return

    dispatch.lower_computation(target_access, value, statement, state)


def _lower_advanced_index_write(target: ast.Subscript, value: ast.expr, statement: ast.Assign,
                                state: LoweringState) -> bool:
    """
    Lower a write through NumPy advanced indexing (``A[indices] = B``) as a
    scatter. Returns False when the target uses no array-valued index, so the
    caller falls through to the ordinary subscript-assignment paths.

    An accumulation always takes conflict resolution here regardless of scope:
    the index array may name the same element twice (``A[[0, 0]] += 1``), so the
    map iterations collide on data the frontend cannot inspect. That is a
    stronger rule than :func:`conflict.accumulation_wcr` applies elsewhere,
    where collision is decided from the subset.
    """
    from dace.frontend.python.nextgen.lowering.mechanisms import advanced_indexing

    if not isinstance(target.value, (ast.Name, ast.Attribute)):
        return False
    try:
        base = resolve_access(target.value, state)
        if base is None:
            return False
        expr = state.inference.parse_access(target)
        if not expr.arrdims:
            return False
        # An accumulation reaches lowering desugared into ``t = <t read> op v``;
        # both write paths re-apply the operator themselves, so strip the
        # self-read down to the accumulated operand first.
        operator = getattr(statement, 'augmented_op', None)
        symbol = conflict.WCR_OPERATORS.get(type(operator)) if operator is not None else None
        accumulated = value
        if symbol is not None and isinstance(value, ast.BinOp):
            accumulated = (value.left if getattr(statement, 'accumulator_side', 'left') == 'right' else value.right)

        if advanced_indexing.has_boolean_index(expr, state.context):
            # A mask writes in place through a guarded update, so it needs
            # neither a gather nor conflict resolution.
            advanced_indexing.emit_masked_write(target, expr, base.container, base.descriptor, accumulated, statement,
                                                state)
            return True
        access = advanced_indexing.analyze(target, expr, base.container, base.descriptor, state.context,
                                           state.inference)
        wcr = f'lambda x, y: x {symbol} y' if symbol is not None else None
        advanced_indexing.emit_scatter(access, accumulated, statement, state, wcr=wcr)
        return True
    except UnsupportedFeatureError as reason:
        dispatch.fallback_to_callback(statement, state, reason)
        return True


def _lower_view_binding(target: ast.Name, access: DataAccess, state: LoweringState) -> None:
    """Bind a slice read as a view container with its NumPy result shape."""
    shape = access.numpy_shape
    selected = access.kept_dims if access.kept_dims is not None else [size != 1 for size in access.subset.size()]
    strides = [
        access.descriptor.strides[i] * step
        for i, (keep, (_, _, step)) in enumerate(zip(selected, access.subset.ranges)) if keep
    ] if isinstance(access.descriptor, data.Array) else None
    if strides is not None:
        # A newaxis-inserted dimension has no corresponding source stride --
        # it only ever has one valid index (0), so any value is safe; 0
        # matches NumPy's own convention for a size-1 broadcast/newaxis axis.
        for position in sorted(access.new_axes):
            strides.insert(position, 0)
    view_descriptor = data.ArrayView(access.descriptor.dtype, shape, strides=strides)
    view_name = state.context.add_container(target.id, view_descriptor)
    state.context.bind(target.id, view_name)
    state.emitter.emit(
        tn.ViewNode(target=view_name,
                    source=access.container,
                    memlet=Memlet(data=access.container, subset=access.subset),
                    src_desc=access.descriptor,
                    view_desc=view_descriptor))


@rule(ast.AnnAssign)
def lower_declaration(statement: ast.AnnAssign, state: LoweringState) -> None:
    """
    Lower a bare declaration (``x: T`` without a value): register the declared
    descriptor so the first assignment to the name adopts it (the primary use
    is Reference declarations, e.g. ``tmp: dace.data.ArrayReference(...)``).
    Unresolvable annotations are no-ops, matching the classic frontend.
    """
    apply_annotation_hint(statement.target.id, statement, state)


def _reference_binding(name: str, state: LoweringState) -> Optional[Tuple[str, data.Data]]:
    """The (container, descriptor) pair of a name bound to a Reference
    container, or None."""
    binding = state.context.resolve(name)
    if binding is None or binding.kind != 'container':
        return None
    descriptor = state.context.containers[binding.container]
    if isinstance(descriptor, data.Reference):
        return binding.container, descriptor
    return None


def _lower_reference_set(label: str, reference: Tuple[str, data.Data], value: ast.expr, inferred: Inferred,
                         statement: ast.Assign, state: LoweringState) -> None:
    """
    Emit a :class:`RefSetNode` re-pointing a reference container (a
    reference-bound name or a reference-typed structure member). Direct
    container sources re-point in place; computed values materialize into a
    fresh container first and the reference is set to it.

    :param label: Source-level label of the reference (for diagnostics and
                  value-container name hints).
    """
    container, reference_descriptor = reference
    hint = f"__{label.replace('.', '_')}_value"

    access = None
    if isinstance(value, (ast.Name, ast.Attribute, ast.Subscript)):
        access = resolve_access(value, state)
    if access is None:
        if inferred.kind == 'static':
            access = static_values.materialize(inferred.value, state, name_hint=hint)
        elif inferred.is_data:
            descriptor = _result_descriptor(inferred, state, statement)
            if not isinstance(descriptor, data.Array):
                raise UnsupportedFeatureError(f'Cannot set reference "{label}" to a scalar value',
                                              state.context.filename,
                                              statement,
                                              category='reference-set')
            value_container = state.context.add_container(hint, descriptor)
            value_access = DataAccess(value_container, subsets.Range.from_array(descriptor), descriptor)
            dispatch.lower_computation(value_access, value, statement, state)
            access = value_access
        else:
            raise UnsupportedFeatureError(
                f'Cannot set reference "{label}" to non-data value '
                f'"{astutils.unparse(value)}"',
                state.context.filename,
                statement,
                category='reference-set')

    if access.is_scalar_access:
        raise UnsupportedFeatureError(f'Cannot set reference "{label}" to a scalar element',
                                      state.context.filename,
                                      statement,
                                      category='reference-set')
    if list(nondegenerate_shape(access.subset)) != [s for s in reference_descriptor.shape if s != 1]:
        raise UnsupportedFeatureError(
            f'Reference "{label}" of shape {tuple(reference_descriptor.shape)} cannot be set '
            f'to source of shape {tuple(nondegenerate_shape(access.subset))}',
            state.context.filename,
            statement,
            category='reference-set')

    state.emitter.emit(
        tn.RefSetNode(target=container,
                      memlet=Memlet(data=access.container, subset=access.subset),
                      src_desc=access.descriptor,
                      ref_desc=copy.deepcopy(reference_descriptor)))


def apply_annotation_hint(target_name: str, statement: ast.stmt, state: LoweringState) -> None:
    """
    Pre-register an annotated assignment target (``y: dace.float64 = ...``)
    with its declared descriptor. Applies only to names with no container
    binding yet; unresolvable annotations are ignored (inference decides).
    """
    descriptor = annotation_descriptor(getattr(statement, 'annotation', None), state)
    if descriptor is None:
        return
    binding = state.context.resolve(target_name)
    if binding is not None and binding.kind == 'container':
        return
    container = state.context.add_container(target_name, descriptor)
    state.context.bind(target_name, container, declared=True)


def annotation_descriptor(annotation: Optional[ast.expr], state: LoweringState) -> Optional[data.Data]:
    """The data descriptor a type annotation declares, or None if the
    annotation is absent or does not resolve to a typed descriptor."""
    if annotation is None:
        return None
    if isinstance(annotation, ast.Constant):
        value = annotation.value
    else:
        try:
            value = astutils.evalnode(annotation, _annotation_environment(state))
        except Exception:
            return None
    try:
        descriptor = data.create_datadescriptor(value)
    except Exception:
        return None
    if not isinstance(descriptor, data.Data) or descriptor.dtype is None or descriptor.dtype.type is None:
        return None
    if isinstance(descriptor.dtype, dtypes.pyobject):
        return None
    descriptor = copy.deepcopy(descriptor)
    descriptor.transient = True
    return descriptor


def _annotation_environment(state: LoweringState) -> dict:
    """The evaluation environment for type annotations: program globals plus
    the descriptors of bound containers, so annotations may reference argument
    properties (``dace.data.ArrayReference(A.dtype, A.shape)``)."""
    environment = dict(state.context.globals)
    for name, binding in state.context.bindings.items():
        if binding.kind == 'container' and binding.container in state.context.containers:
            environment[name] = state.context.containers[binding.container]
    return environment


def prepare_name_target(target: ast.Name, inferred: Inferred, state: LoweringState,
                        statement: ast.Assign) -> DataAccess:
    """
    Return the target access for a name assignment, reusing the currently
    bound container when it is compatible (in-place update) or registering a
    new one otherwise.
    """
    binding = state.context.resolve(target.id)
    result_descriptor = _result_descriptor(inferred, state, statement)

    # A declared type outranks the values later assigned to the name:
    # ``ytmp: float32 = 0`` is a float32 accumulator holding zero, not an int64
    # one, and a subsequent ``ytmp = t`` converts into float32 rather than
    # re-typing the name. Without this the declaration would be found
    # "incompatible" with the value and immediately rebound away -- and inside
    # a loop that rebinding is loop-carried, which forces a callback.
    if binding is not None and binding.kind == 'container' and binding.declared:
        declared = state.context.containers[binding.container]
        if _writable_as(declared, result_descriptor):
            return DataAccess(binding.container, subsets.Range.from_array(declared), declared)

    if binding is not None and binding.kind == 'container':
        existing = state.context.containers[binding.container]
        if _compatible(existing, result_descriptor):
            return DataAccess(binding.container, subsets.Range.from_array(existing), existing)
        if inferred.kind in ('constant', 'symbolic') and _writable_as(existing, result_descriptor):
            # A compile-time SCALAR written into a name that already holds a
            # container converts into it (``b = dace.ndarray(...)`` … ``b = 0``
            # fills ``b``), which is what the classic frontend does and what
            # makes such an assignment mergeable across branches -- rebinding
            # the name to a fresh scalar of the literal's own type instead left
            # the branches of an if/elif/else disagreeing on both the container
            # and its dtype, and the whole chain fell back.
            return DataAccess(binding.container, subsets.Range.from_array(existing), existing)

    container_name = state.context.add_container(target.id, result_descriptor)
    state.context.bind(target.id, container_name)
    return DataAccess(container_name, subsets.Range.from_array(result_descriptor), result_descriptor)


def _result_descriptor(inferred: Inferred, state: LoweringState, statement: ast.Assign) -> data.Data:
    if inferred.kind == 'static':
        return state.inference.sequence_descriptor(inferred.value)
    if inferred.is_data:
        descriptor = inferred.descriptor
        if isinstance(descriptor, data.Stream):
            # Streams keep their full descriptor (buffer size etc.)
            return copy.deepcopy(descriptor)
        if not isinstance(descriptor, (data.Array, data.Scalar)):
            # Not storage this can re-allocate: a communicator
            # (``ProcessGrid``, from ``MPI.COMM_WORLD.Create_cart``), a
            # structure, ... -- the descriptor IS the value, and reducing it to
            # a scalar of its dtype loses the type that later method calls on
            # the name resolve through (``pgrid.Bcast(A)`` is registered on
            # ``ProcessGrid``, not on a scalar).
            return copy.deepcopy(descriptor)
        if isinstance(descriptor, data.Array):
            # A fresh array of the same shape, dropping any VIEW-ness of the
            # source (the result is its own storage) but keeping the layout:
            # ``dace.ndarray(..., strides=(1, 2))`` means those strides, and
            # rebuilding a default-layout array here would silently ignore
            # them while the call still looks lowered.
            return data.Array(descriptor.dtype,
                              list(descriptor.shape),
                              strides=list(descriptor.strides),
                              storage=descriptor.storage,
                              lifetime=descriptor.lifetime)
        return data.Scalar(descriptor.dtype, storage=descriptor.storage, lifetime=descriptor.lifetime)
    dtype = state.inference.dtype_of(inferred)
    if dtype is None:
        if inferred.kind == 'symbolic':
            dtype = dtypes.int64
        else:
            try:
                dtype = dtypes.typeclass(type(inferred.value))
            except (KeyError, TypeError):
                # Constants of non-C-representable types (enum classes, ...)
                raise UnsupportedFeatureError(
                    f'Cannot represent constant of type {type(inferred.value).__name__} in a container',
                    node=statement,
                    category='type-inference')
    return data.Scalar(dtype)


def _representable_constant(value) -> bool:
    """Whether a constant value's type has a C container representation."""
    try:
        dtypes.typeclass(type(value))
        return True
    except (KeyError, TypeError):
        return False


def _writable_as(declared: data.Data, new: data.Data) -> bool:
    """
    Whether a value described by ``new`` can be written into a container
    declared as ``declared``, converting on the way in.

    Unlike :func:`_compatible` this ignores the dtype -- a declaration exists
    precisely to override it -- and only asks whether the value fits the
    declared storage: same shape, or a scalar broadcast into it.
    """
    if isinstance(declared, (data.View, data.Reference, data.Stream)) or isinstance(new, data.Stream):
        return False
    if isinstance(new, data.Scalar) or not new.shape:
        return True
    return tuple(declared.shape) == tuple(new.shape)


def _compatible(existing: data.Data, new: data.Data) -> bool:
    """
    In-place updates require same dtype and shape (scalars always match on
    dtype).

    A :class:`~dace.data.View` qualifies like any other array of its shape: a
    view IS the window it names, so writing into it writes through to the
    viewed container, which is what ``y = a[0:10]; y += 1`` has to do.
    Refusing reuse here allocated a fresh container instead and rebound ``y``
    to it BEFORE the right-hand side was lowered, so the update read the new,
    uninitialized container and the original was never touched -- a silent
    miscompilation, not a fallback.
    """
    if existing.dtype != new.dtype:
        return False
    if isinstance(existing, data.Scalar) and isinstance(new, data.Scalar):
        return True
    if isinstance(existing, data.Array) and isinstance(new, data.Array):
        return tuple(existing.shape) == tuple(new.shape)
    # A scalar container holds a one-element result, and vice versa: the two
    # descriptors name the same single element, so the write goes in place
    # rather than rebinding the name (``i = 1`` … ``i += j`` with ``j`` a
    # ``float64[1]`` is one accumulator, not two containers -- and inside a
    # loop the rebinding would be loop-carried and force a callback).
    if isinstance(existing, (data.Scalar, data.Array)) and isinstance(new, (data.Scalar, data.Array)):
        return _element_count(existing) == 1 and _element_count(new) == 1
    return False


def _element_count(descriptor: data.Data) -> Any:
    """The number of elements a scalar/array descriptor holds."""
    return data._prod(descriptor.shape) if isinstance(descriptor, data.Array) else 1
