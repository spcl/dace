# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Demand-driven inference for canonical (post-ANF) expressions.

Because lowering only ever sees depth-1 ("flat") expressions, inference here
is intentionally small: it classifies an expression as a container access, a
symbolic expression, or a compile-time constant, and computes the result
descriptor for flat operator expressions. There is no separate whole-program
inference pass — rules ask on demand.

Descriptor inference for library calls (NumPy and friends) is added by the
call-lowering rules through the replacement registry; this module only covers
the operator core.
"""
import ast
import copy
import numbers
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy
from dace import data, dtypes, symbolic
from dace.frontend.python import astutils
from dace.frontend.python.memlet_parser import ParseMemlet, MemletExpr
from dace.frontend.python.nextgen.common import (UnsupportedFeatureError, normalize_qualname, supported_data_attribute)
from dace.frontend.python.nextgen.semantics import values
from dace.frontend.python.nextgen.semantics.context import ProgramContext
from dace.frontend.python.nextgen.semantics.indexing import array_index_slots, reverse_normalized, substitute_slots
from dace.frontend.python.nextgen.semantics.values import StaticSequence

#: Comparison and boolean operators always produce booleans.
_BOOLEAN_OPS = (ast.Compare, ast.BoolOp)

#: Array properties readable at compile time straight off the data descriptor,
#: as the NumPy array interface spells them.
#:
#: Enumerated rather than forwarded to the descriptor object generically
#: (``getattr(descriptor, attr)``): a data descriptor is a DaCe object, not an
#: ``ndarray``, and most of its properties either mean something else under the
#: same name (``strides`` counts elements, not bytes) or are internal
#: (``transient``, ``storage``, ``lifetime``). A property is listed here when
#: reading it from the descriptor is what the program asked for; each query
#: returns None when the descriptor cannot answer it.
DESCRIPTOR_PROPERTIES: Dict[str, Any] = {
    'dtype': lambda descriptor: descriptor.dtype,
    'shape': lambda descriptor: tuple(descriptor.shape) if isinstance(descriptor, data.Array) else None,
    'ndim': lambda descriptor: len(descriptor.shape) if isinstance(descriptor, data.Array) else None,
}


def _attribute_value(descriptor: data.Data, attr_name: str) -> Any:
    """
    Evaluate an ATTRIBUTE-family replacement that computes a compile-time
    value rather than data (``A.size``), returning None when there is no such
    entry or it produces something else.

    The replacement is run on a scratch SDFG, the same trial-before-commit
    shape ``lowering.dispatch._run_attribute_trial`` uses for the attributes
    that do produce data; here the trial is what distinguishes the two, since
    a value-valued entry is exactly one that returns no data name and builds
    no dataflow. Everything downstream is already generic over compile-time
    values (``mechanisms.static_values.fold_descriptor_properties`` folds
    whatever inference reports), so registering a new such attribute needs no
    frontend change.
    """
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population
    from dace.sdfg.analysis.schedule_tree.tree_to_sdfg import ReplacementVisitorShim
    from dace.sdfg.sdfg import SDFG

    function = oprepo.Replacements.get_attribute(type(descriptor), attr_name)
    if function is None:
        return None
    scratch = SDFG('__attribute_value')
    scratch_descriptor = copy.deepcopy(descriptor)
    scratch_descriptor.transient = False
    scratch.add_datadesc('__attr_base', scratch_descriptor)
    scratch_state = scratch.add_state()
    shim = ReplacementVisitorShim(scratch, scratch_state, '__attr_value')
    try:
        result = function(shim, scratch, scratch_state, '__attr_base')
    except Exception:
        return None
    if shim.views or scratch_state.number_of_nodes() > 0 or len(scratch.nodes()) > 1:
        return None  # Produced dataflow: a data attribute, handled above
    if isinstance(result, (int, float, complex)) or symbolic.issymbolic(result):
        return result
    return None


def _apply_unary_operator(operator: ast.unaryop, value: Any) -> Any:
    """Apply a unary AST operator to a compile-time constant."""
    if isinstance(operator, ast.USub):
        return -value
    if isinstance(operator, ast.UAdd):
        return +value
    if isinstance(operator, ast.Invert):
        return ~value
    raise TypeError(f'Cannot constant-fold unary operator {type(operator).__name__}')


@dataclass
class Inferred:
    """
    Classification of a canonical expression.

    :param kind: ``'data'`` (container access), ``'symbolic'`` (symbol
                 expression), ``'constant'`` (compile-time value),
                 ``'static'`` (compile-time Python sequence, see
                 :class:`~dace.frontend.python.nextgen.semantics.values.StaticSequence`),
                 ``'data-tuple'`` (a call producing SEVERAL containers, e.g.
                 ``numpy.split``/``numpy.divmod``; the descriptors are in
                 ``descriptors``), or ``'none'`` (a call whose registry
                 descriptor inference confirmed a zero-output/pure-side-effect
                 signature, e.g. ``A.fill(0)`` — distinct from
                 :func:`infer_call` returning Python ``None`` for "no registry
                 entry matched").
    :param descriptor: Result data descriptor for ``'data'`` expressions.
    :param value: The symbolic expression, constant, or static sequence otherwise.
    :param descriptors: Result descriptors, in result order, for
                        ``'data-tuple'`` expressions.
    """
    kind: str
    descriptor: Optional[data.Data] = None
    value: Any = None
    descriptors: Optional[Tuple[data.Data, ...]] = None

    @property
    def is_data(self) -> bool:
        return self.kind == 'data'

    @property
    def is_data_tuple(self) -> bool:
        """Whether this is a call result made of several containers. Kept
        distinct from ``is_data`` on purpose: every single-result path would be
        wrong for it, so it must opt in rather than opt out."""
        return self.kind == 'data-tuple'

    @property
    def is_none_output(self) -> bool:
        return self.kind == 'none'

    @property
    def is_pyobject(self) -> bool:
        return self.kind == 'data' and isinstance(self.descriptor.dtype, dtypes.pyobject)

    @property
    def dtype(self) -> Optional[dtypes.typeclass]:
        if self.descriptor is not None:
            return self.descriptor.dtype
        if self.kind == 'symbolic':
            try:
                return symbolic.symtype(self.value)
            except (TypeError, AttributeError) as error:
                # Mixed or missing symbol dtypes in the expression
                raise UnsupportedFeatureError(f'Cannot infer symbolic expression type of "{self.value}": {error}',
                                              category='type-inference')
        if self.kind == 'constant' and isinstance(self.value, tuple(dtypes.dtype_to_typeclass().keys())):
            return dtypes.dtype_to_typeclass(type(self.value))
        return None


class _LocationShim:
    """Minimal visitor stand-in for the shared memlet parser's error reports."""

    def __init__(self, filename: str):
        self.filename = filename


def is_literal_constant(value: Any) -> bool:
    """Whether a constant node's value is a plain Python literal (as opposed
    to an arbitrary object embedded by preprocessing's global resolution)."""
    if value is None or value is Ellipsis:
        return True
    if isinstance(value, (bool, int, float, complex, str, bytes)):
        return True
    if isinstance(value, (tuple, frozenset)):
        return all(is_literal_constant(element) for element in value)
    return False


def _qualified_object_name(obj: Any, fallback: Optional[str]) -> Optional[str]:
    """The registry-facing qualified name of a resolved Python object."""
    module_name = getattr(obj, '__module__', None)
    object_name = getattr(obj, '__name__', None)
    if module_name and object_name and module_name != 'builtins':
        return f'{module_name}.{object_name}'
    if object_name:
        return object_name
    return fallback


#: Kind ranks for NEP 50 weak promotion (see
#: :meth:`InferenceService._result_dtype`): a literal only widens a typed
#: operand when its own kind ranks higher.
_WEAK_SCALAR_RANKS = {bool: 0, int: 1, float: 2, complex: 3}
_WEAK_KIND_DEFAULTS = {0: dtypes.bool_, 1: dtypes.int64, 2: dtypes.float64, 3: dtypes.complex128}
_DTYPE_KIND_RANKS = {'b': 0, 'i': 1, 'u': 1, 'f': 2, 'c': 3}

#: Function names the SYMBOLIC parser resolves on its own
#: (``dace.symbolic.pystr_to_symbolic``), which a program may therefore call
#: without ever defining them (``dace.mapscope(_[0:ceiling(N / 32)])``). Over
#: compile-time operands such a call is a symbolic expression rather than a
#: function call — see :meth:`InferenceService._symbolic_intrinsic`.
SYMBOLIC_INTRINSICS = frozenset(
    {'ceiling', 'ceil', 'floor', 'int_ceil', 'int_floor', 'sqrt', 'Min', 'Max', 'Abs', 'Mod', 'round'})


def _weak_scalar_rank(operand: 'Inferred') -> Optional[int]:
    """
    The weak-promotion rank of an operand that is a Python scalar LITERAL, or
    None for anything typed.

    Exact types only: ``numpy.int32(1)`` is a typed value that promotes
    normally, while the ``1`` in ``i + 1`` is weak. ``bool`` is checked before
    ``int`` because it is a subclass of it.
    """
    if operand.kind != 'constant':
        return None
    return _WEAK_SCALAR_RANKS.get(type(operand.value))


def _dtype_kind_rank(dtype: dtypes.typeclass) -> int:
    """The kind rank of a dace dtype, on the same scale as
    :data:`_WEAK_SCALAR_RANKS`. Unknown kinds rank highest, so a literal never
    widens them."""
    try:
        return _DTYPE_KIND_RANKS.get(numpy.dtype(dtype.type).kind, len(_WEAK_KIND_DEFAULTS))
    except TypeError:
        return len(_WEAK_KIND_DEFAULTS)


def _registered_qualname(name: Optional[str]) -> bool:
    """Whether the replacement registry has an entry under this exact name,
    in either the implementation or the descriptor-inference keyspace."""
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
    if name is None:
        return False
    return (oprepo.Replacements.get(name) is not None or oprepo.Replacements.get_descriptor_inference(name) is not None)


def broadcast_shapes(first: Sequence[Any], second: Sequence[Any]) -> Tuple[Any, ...]:
    """
    NumPy-style shape broadcasting for symbolic shapes.

    :raises UnsupportedFeatureError: If the shapes cannot be broadcast.
    """
    result: List[Any] = []
    for dim_a, dim_b in zip(_padded(first, second), _padded(second, first)):
        if dim_a is None:
            result.append(dim_b)
        elif dim_b is None:
            result.append(dim_a)
        elif dim_a == dim_b or dim_b == 1:
            result.append(dim_a)
        elif dim_a == 1:
            result.append(dim_b)
        else:
            # Symbolically unequal dimensions: assume equality (matches the
            # stable frontend, which defers mismatches to runtime).
            result.append(dim_a)
    return tuple(result)


def _padded(shape: Sequence[Any], other: Sequence[Any]) -> List[Any]:
    pad = max(len(other) - len(shape), 0)
    return [None] * pad + list(shape)


def registry_operator_operands(node: ast.expr, operands: Sequence['Inferred']) -> Optional[Tuple[str, List[Any]]]:
    """
    The (AST operator name, operand values) an operator expression presents to
    the replacement registry, or None when the expression is not one the
    registry can be asked about.

    Registry operator rules are keyed on the AST operator name and the operand
    classes, and receive descriptors for data operands and plain values for
    compile-time ones. Comparison chains and boolean operators are excluded:
    they have no single operand pair to key on.

    Shared by inference and lowering so both stages agree on which expressions
    the registry owns.
    """
    if isinstance(node, ast.BinOp) and len(operands) == 2:
        optype = type(node.op).__name__
    elif isinstance(node, ast.UnaryOp) and len(operands) == 1:
        optype = type(node.op).__name__
    else:
        return None
    values = []
    for operand in operands:
        if operand.is_data:
            values.append(operand.descriptor)
        elif operand.kind in ('constant', 'symbolic'):
            values.append(operand.value)
        else:
            return None  # A compile-time sequence or an opaque object
    return optype, values


def operator_lookup_arguments(optype: str, values: Sequence[Any]) -> Tuple[Any, ...]:
    """The arguments to :meth:`Replacements.getop` for an operator over
    ``values`` — (left, optype[, right]), each operand reduced to the key the
    registry rules are registered under."""
    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population
    keys = [oprepo.operand_lookup_key(value) for value in values]
    if len(keys) == 1:
        return (keys[0], optype)
    return (keys[0], optype, keys[1])


class InferenceService:
    """Classifies canonical expressions against a :class:`ProgramContext`."""

    def __init__(self, context: ProgramContext):
        self.context = context
        self._shim = _LocationShim(context.filename)

    def infer(self, node: ast.expr) -> Inferred:
        """
        Infer the classification and result descriptor of a canonical
        expression.

        :raises UnsupportedFeatureError: If the expression cannot be inferred.
        """
        if isinstance(node, ast.Constant):
            return Inferred(kind='constant', value=node.value)
        if isinstance(node, (ast.List, ast.Tuple)):
            sequence_kind = 'list' if isinstance(node, ast.List) else 'tuple'
            return Inferred(kind='static', value=StaticSequence(elements=list(node.elts), kind=sequence_kind))
        if isinstance(node, ast.Name):
            return self._infer_name(node)
        if isinstance(node, ast.Attribute):
            return self._infer_attribute(node)
        if isinstance(node, ast.UnaryOp):
            operand = self.infer(node.operand)
            if isinstance(node.op, ast.Not):
                return self._demote_to_bool(operand)
            if operand.kind == 'constant':
                try:
                    return Inferred(kind='constant', value=_apply_unary_operator(node.op, operand.value))
                except TypeError:
                    pass
            if operand.kind == 'symbolic':
                return Inferred(kind='symbolic', value=self._symbolic_expression(node))
            # Data operands: sign/inversion preserves descriptor shape and dtype
            return operand
        if isinstance(node, ast.Subscript):
            return self._infer_subscript(node)
        if isinstance(node, (ast.BinOp, ast.Compare, ast.BoolOp)):
            return self._infer_operator(node)
        if isinstance(node, ast.Call):
            inferred = self.infer_call(node)
            # A zero-output ('none') call has no value to use as an
            # expression operand; only a genuine result is usable here.
            if inferred is not None and inferred.kind != 'none':
                return inferred
        raise UnsupportedFeatureError(f'Cannot infer type of expression: {astutils.unparse(node)}',
                                      self.context.filename,
                                      node,
                                      category='type-inference')

    def resolve_callee(self, func: ast.expr) -> Tuple[str, Optional[Any]]:
        """
        Resolve a canonical callee expression (a name or attribute chain) to a
        qualified name and, when possible, the Python object it refers to.

        The qualified name is normalized to ``module.__name__``-based form
        (e.g. ``numpy.zeros`` even for ``np.zeros``), matching the keys of the
        replacement registry. It is then passed through
        :func:`~dace.frontend.python.nextgen.common.normalize_qualname`, since
        ``module.__name__`` reports a callee's REAL defining module -- e.g.
        ``dace.ndarray`` is really ``dace.frontend.python.wrappers.ndarray``,
        re-exported at the top-level ``dace`` package -- which needs
        collapsing to the registry's shorter key whenever a callee is reached
        other than through a literal, unaliased ``dace.<name>`` attribute
        chain (an aliased module import, a name bound to the function object,
        the fully-qualified path written out directly, ...). This is the
        SINGLE normalization point: every qualname this method returns,
        regardless of which branch produced it, is already registry-facing --
        callers (:meth:`infer_call`, ``lowering.dispatch``) must not re-derive
        or re-normalize it themselves.

        :return: A 2-tuple of (qualified name, resolved object or None).
        """
        # Preprocessing embeds resolved global objects (dace programs, SDFGs,
        # constants) directly into the AST as constant nodes with a qualname.
        if isinstance(func, ast.Constant) and not is_literal_constant(func.value):
            resolved = func.value
            return normalize_qualname(_qualified_object_name(resolved, getattr(func, 'qualname', None))), resolved

        parts: List[str] = []
        node = func
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return normalize_qualname(astutils.rname(func)), None
        parts.append(node.id)
        parts.reverse()

        root = self._global_value(parts[0])
        resolved = root
        for attribute in parts[1:]:
            if resolved is None:
                break
            resolved = getattr(resolved, attribute, None)

        candidates: List[str] = []
        if resolved is not None:
            qualified = _qualified_object_name(resolved, None)
            if qualified is not None:
                candidates.append(normalize_qualname(qualified))
        if len(parts) > 1 and isinstance(root, types.ModuleType):
            # The call-site path with the module alias resolved
            # (``donnx.ONNXGather`` -> ``dace.libraries.onnx.ONNXGather``).
            candidates.append(normalize_qualname(f'{root.__name__}.{".".join(parts[1:])}'))
        candidates.append(normalize_qualname('.'.join(parts)))
        # Prefer whichever spelling the registry actually knows. A library that
        # registers its replacements under a package's PUBLIC path while the
        # objects themselves live in a private submodule (the ONNX op registry
        # keys ``dace.libraries.onnx.ONNXGather`` for a class whose
        # ``__module__.__name__`` is ``dace.libraries.onnx.nodes.
        # onnx_op_registry.ONNXGather_13``) is reachable only through the
        # module-path candidate. Asking the registry keeps this from being one
        # more entry in :data:`_QUALNAME_MODULE_REWRITES`, which can only
        # describe rewrites known in advance.
        for candidate in candidates:
            if _registered_qualname(candidate):
                return candidate, resolved
        return candidates[0], resolved

    #: Ufunc methods with dedicated registry entries (``get_ufunc(method)``).
    _UFUNC_METHODS = ('reduce', 'accumulate', 'outer')

    def resolve_ufunc_call(self, node: ast.Call) -> Optional[Tuple[Any, Optional[str]]]:
        """
        Resolve a canonical call as a NumPy universal function invocation,
        either direct (``numpy.add(...)``) or through one of its
        ``reduce``/``accumulate``/``outer`` methods (``numpy.add.reduce(...)``
        — an ``ast.Attribute`` call whose *base* resolves to a
        ``numpy.ufunc``, not the callee itself, which resolves to a bound
        method object instead).

        Shared by inference (:meth:`infer_call`) and lowering
        (``dispatch._lower_registry_call``) so both stages agree on which
        calls are ufunc invocations and which method they use.

        :return: A 2-tuple of (ufunc object, method name, or ``None`` for a
                 direct call), or ``None`` if the call is not a ufunc
                 invocation.
        """
        _, callee = self.resolve_callee(node.func)
        if isinstance(callee, numpy.ufunc):
            return callee, None
        if isinstance(node.func, ast.Attribute) and node.func.attr in self._UFUNC_METHODS:
            _, base = self.resolve_callee(node.func.value)
            if isinstance(base, numpy.ufunc):
                return base, node.func.attr
        return None

    def infer_call(self, node: ast.Call) -> Optional[Inferred]:
        """
        Descriptor inference for a canonical flat call through the
        descriptor-inference families of the replacement registry
        (:class:`dace.frontend.common.op_repository.Replacements`).

        Queried in order: method inference for calls on data-bound objects,
        ufunc inference for NumPy universal functions, then free-function
        inference by qualified name.

        :return: The inferred result, or None if no registry entry matched
                 (the caller decides how to fall back).
        """
        from dace.frontend.common import op_repository as oprepo  # Deferred: registry population needs replacements
        arguments = self.call_arguments(node)
        if arguments is None:
            return None
        input_descs, args, kwargs = arguments

        # Method calls on data containers (a.sum(), a.copy(), ...)
        if isinstance(node.func, ast.Attribute):
            base = self.bound_descriptor_of(node.func.value)
            if base is not None:
                infer_fn = oprepo.Replacements.get_method_descriptor_inference(type(base), node.func.attr)
                if infer_fn is not None:
                    return self._registry_inference(infer_fn, base, *args, **kwargs)
                # No method-family inference for this (type, method) pair:
                # fall through to ufunc/free-function inference below rather
                # than hard-aborting the whole call (the base being data
                # doesn't rule out `node.func` resolving some other way, e.g.
                # a qualified free function embedded via a constant callee).
            receiver_object = self.constant_object(node.func.value)
            if receiver_object is not None:
                infer_fn = oprepo.Replacements.get_method_descriptor_inference(type(receiver_object), node.func.attr)
                if infer_fn is not None:
                    return self._registry_inference(infer_fn, receiver_object, *args, **kwargs)

        qualname, callee = self.resolve_callee(node.func)

        # Symbolic intrinsics (``ceiling(N / 32)`` as a map bound). These names
        # are resolved by the symbolic parser, not by the program: nothing
        # defines them, so the callee is unresolvable and no registry entry
        # exists. Over purely symbolic arguments such a call IS a symbolic
        # expression, and typing it as one is what keeps it out of the
        # interpreter -- canonicalization hoists a compound map bound into its
        # own statement, which would otherwise become a callback and poison
        # the loop it bounds.
        symbolic_intrinsic = self._symbolic_intrinsic(node, qualname, callee)
        if symbolic_intrinsic is not None:
            return symbolic_intrinsic

        # NumPy universal functions (np.add, np.sin, ...), direct or through
        # one of their reduce/accumulate/outer methods (np.add.reduce(...)).
        ufunc_form = self.resolve_ufunc_call(node)
        if ufunc_form is not None:
            ufunc, ufunc_method = ufunc_form
            infer_fn = oprepo.Replacements.get_ufunc_descriptor_inference(ufunc_method)
            if infer_fn is None:
                return None
            return self._registry_inference(infer_fn, input_descs, ufunc.__name__, *args, **kwargs)

        # Free functions by qualified name (numpy.zeros, numpy.sum, ...).
        # Fall back to the source-level name: the qualname preprocessing
        # attaches to embedded callee constants, or the textual call name.
        infer_fn = oprepo.Replacements.get_descriptor_inference(qualname)
        if infer_fn is None:
            textual_name = getattr(node.func, 'qualname', None) or astutils.rname(node.func)
            if textual_name != qualname:
                infer_fn = oprepo.Replacements.get_descriptor_inference(textual_name)
        if infer_fn is None:
            return None
        return self._registry_inference(infer_fn, input_descs, *args, **kwargs)

    def _is_replacement_handle(self, node: ast.expr) -> bool:
        """Whether an argument names a container holding an opaque handle a
        registry replacement produced (see
        :attr:`ProgramContext.replacement_handles`), which the next replacement
        consumes by name even though it is Python-object-typed."""
        if not isinstance(node, ast.Name):
            return False
        binding = self.context.resolve(node.id)
        return binding is not None and binding.container in self.context.replacement_handles

    def constant_object(self, node: ast.expr) -> Optional[Any]:
        """
        The compile-time Python OBJECT a name refers to (an MPI communicator
        from the closure, ``commworld`` in ``commworld.Bcast(A)``), or None
        when the expression is not a name bound to one.

        Numbers, strings and symbols are excluded: what this identifies is a
        receiver whose *class* the replacement registry may have method
        entries for (``replaces_method('Intracomm', 'Bcast')``), which the
        value domains handle themselves.
        """
        if not isinstance(node, ast.Name):
            return None
        try:
            inferred = self.infer(node)
        except UnsupportedFeatureError:
            return None
        if inferred.kind != 'constant':
            return None
        value = inferred.value
        if value is None or isinstance(value, (numbers.Number, str, bytes, bool, type, tuple, list, dict, set)):
            return None
        return value

    def symbolic_intrinsic_value(self, node: ast.Call) -> Optional[Any]:
        """
        The compile-time symbolic value of a call to a symbolic intrinsic
        (:data:`SYMBOLIC_INTRINSICS`), or None when the call is not one.

        Lowering asks this before treating a call as dataflow: such a call has
        no runtime implementation to emit — the symbolic parser is what
        resolves it — so it binds as a value instead.
        """
        qualname, callee = self.resolve_callee(node.func)
        inferred = self._symbolic_intrinsic(node, qualname, callee)
        return None if inferred is None else inferred.value

    def _symbolic_intrinsic(self, node: ast.Call, qualname: str, callee: Optional[Any]) -> Optional[Inferred]:
        """
        The symbolic value of a call to a symbolic intrinsic
        (:data:`SYMBOLIC_INTRINSICS`) over compile-time operands, or None when
        the call is not one.

        Restricted to an UNRESOLVABLE callee: a name the program actually
        binds (``numpy.floor``, the builtin ``min``) is a real function with
        its own registry lowering, and only the names left for the symbolic
        parser to resolve belong here.
        """
        if callee is not None or qualname not in SYMBOLIC_INTRINSICS or node.keywords:
            return None
        try:
            operands = [self.infer(argument) for argument in node.args]
        except UnsupportedFeatureError:
            return None
        if not operands or any(operand.kind not in ('constant', 'symbolic') for operand in operands):
            return None
        # Built from the operands' VALUES rather than by re-parsing the source
        # text: an argument hoisted into an ANF temporary (``__anf0 = N / 32``)
        # is a name the symbolic parser would take for an integer symbol, and
        # ``ceiling`` of an integer folds away before anything substitutes the
        # value back.
        arguments = ', '.join(str(operand.value) for operand in operands)
        try:
            return Inferred(kind='symbolic', value=symbolic.pystr_to_symbolic(f'{qualname}({arguments})'))
        except Exception:
            return None

    def call_arguments(self, node: ast.Call) -> Optional[Tuple[Dict[str, data.Data], List[Any], Dict[str, Any]]]:
        """
        Resolve canonical call arguments to the replacement registry's
        inference convention: data operands are passed by name with their
        descriptors collected separately; constants, symbols, and static
        sequences are passed by value.

        :return: A 3-tuple of (input descriptors by name, positional argument
                 values, keyword argument values), or None if any argument
                 cannot be represented (e.g., references an opaque object).
        """
        input_descs: Dict[str, data.Data] = {}

        def convert(argument: ast.expr) -> Tuple[bool, Any]:
            if isinstance(argument, ast.Lambda):
                # The combiner of a reduction intrinsic (``dace.reduce(lambda
                # a, b: a + b, ...)``). Both frontends hand these to the
                # registry as SOURCE TEXT -- the classic frontend's
                # ``visit_Lambda`` unparses the node, and the implementations
                # (and the WCR properties they build) parse the string back --
                # so a lambda has no descriptor and never reaches ``infer``.
                return True, astutils.unparse(argument)
            try:
                inferred = self.infer(argument)
            except UnsupportedFeatureError:
                return False, None
            if inferred.is_pyobject and not self._is_replacement_handle(argument):
                return False, None
            if inferred.is_data:
                try:
                    name = astutils.rname(argument)
                except TypeError:
                    # Data-valued compound expression (e.g. UnaryOp over an
                    # array) — cannot be passed to the registry by name
                    return False, None
                input_descs[name] = inferred.descriptor
                return True, name
            if inferred.kind in ('constant', 'symbolic'):
                return True, inferred.value
            if inferred.kind == 'static':
                # A static sequence's elements may themselves be data
                # containers (e.g. the ``(A, B)`` in ``numpy.concatenate((A,
                # B))``): each is passed through by name, exactly like a
                # top-level data argument, matching the classic frontend's
                # convention of passing lists of container names here.
                elements = []
                for element in inferred.value.elements:
                    ok, value = convert(element)
                    if not ok:
                        return False, None
                    elements.append(value)
                return True, tuple(elements) if inferred.value.kind == 'tuple' else elements
            return False, None

        args: List[Any] = []
        for argument in node.args:
            ok, value = convert(argument)
            if not ok:
                return None
            args.append(value)
        kwargs: Dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                return None
            ok, value = convert(keyword.value)
            if not ok:
                return None
            kwargs[keyword.arg] = value
        return input_descs, args, kwargs

    def _global_value(self, name: str) -> Optional[Any]:
        """Resolve a root name against the program globals, tolerating the
        module-name rewriting done by preprocessing (aliased imports appear
        under their real module names in the AST)."""
        if name in self.context.globals:
            return self.context.globals[name]
        for value in self.context.globals.values():
            if isinstance(value, types.ModuleType) and value.__name__ == name:
                return value
        return None

    def bound_descriptor_of(self, node: ast.expr) -> Optional[data.Data]:
        """
        The container descriptor a canonical expression is bound to, if any.

        Beyond a bare ``Name`` (``a.sum()``), one more level is admitted --
        an indexed or attribute expression that itself infers to a data
        result, e.g. ``A[0].sum()`` or a structure member access -- so a
        method call on such an expression is at least *typed* (whether it
        can subsequently be *lowered* through the method-replacement family
        is a separate, stricter question decided at dispatch time, which
        only accepts a whole-container ``Name``/member receiver).
        """
        if isinstance(node, ast.Name):
            binding = self.context.resolve(node.id)
            if binding is not None and binding.kind == 'container':
                return self.context.containers[binding.container]
            return None
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            try:
                inferred = self.infer(node)
            except UnsupportedFeatureError:
                return None
            if inferred.is_data:
                return inferred.descriptor
        return None

    def _registry_inference(self, infer_fn: Any, *args: Any, **kwargs: Any) -> Optional[Inferred]:
        """Invoke a registry inference function defensively and normalize its
        result. An empty tuple/list is the registry convention for a confirmed
        zero-output/pure-side-effect signature (e.g.
        ``infers_method_descriptor('Array', 'fill')`` — see
        ``array_creation.py``), distinguished here (``kind='none'``) from
        "no registry entry matched" (``None``), which callers must not
        conflate: one is a typed call with nothing to assign, the other is an
        untyped call that should fall back to a callback. Several descriptors
        are a multi-output call (``numpy.split``, ``numpy.divmod``) and come
        back as ``kind='data-tuple'``."""
        try:
            result = infer_fn(*args, **kwargs)
        except Exception:
            return None
        if isinstance(result, data.Data):
            return Inferred(kind='data', descriptor=result)
        if isinstance(result, (tuple, list)):
            if len(result) == 1 and isinstance(result[0], data.Data):
                return Inferred(kind='data', descriptor=result[0])
            if len(result) == 0:
                return Inferred(kind='none')
            if all(isinstance(element, data.Data) for element in result):
                return Inferred(kind='data-tuple', descriptors=tuple(result))
        return None

    def parse_access(self, node: Union[ast.Name, ast.Subscript]) -> MemletExpr:
        """
        Parse a canonical data access (name or subscript of a name) into a
        memlet expression with an explicit subset, using the shared memlet
        parser.

        :raises UnsupportedFeatureError: If the shared parser cannot handle
            the access form. The parser fails in assorted ways on exotic
            indexing (advanced indexing, ``.flat``, long index tuples, ...);
            every failure at this boundary is a feature gap, not a crash.
        """
        return self.parse_access_typed(node)[0]

    def parse_access_typed(self, node: Union[ast.Name, ast.Subscript]) -> Tuple[MemletExpr, Dict[str, data.Data]]:
        """
        :func:`parse_access`, additionally reporting the descriptors of the
        placeholder names it introduced for array-valued index EXPRESSIONS.

        Those placeholders are types without storage and belong to this parse
        alone, so they travel with its result rather than accumulating on the
        program context. Only the shape rules need them, and only until
        lowering materializes the same expressions into real containers.

        :raises UnsupportedFeatureError: As :func:`parse_access`.
        """
        try:
            defined = self.context.defined_view()
            node = self._forward_reversed_slices(node)
            node, index_types = self._name_array_index_expressions(self._restore_index_sequences(node), defined)
            # ``partial_boolean_index``: this frontend can lower a mask over a
            # single dimension (``A[0, mask]``), so it asks the shared parser
            # to report one instead of refusing it.
            return ParseMemlet(self._shim, defined, node, partial_boolean_index=True), index_types
        except UnsupportedFeatureError:
            raise
        except Exception as error:
            raise UnsupportedFeatureError(f'Unsupported access expression "{astutils.unparse(node)}": {error}',
                                          self.context.filename,
                                          node,
                                          category='memlet-parse')

    def _forward_reversed_slices(self, node: Union[ast.Name, ast.Subscript]) -> Union[ast.Name, ast.Subscript]:
        """
        Rewrite a negative-step slice into the equivalent forward slice before
        parsing (see ``semantics.indexing.reverse_normalized``).

        The shared parser applies forward conventions to a negative step, so
        ``A[::-1]`` parsed as written yields a NEGATIVE extent and the shape
        derived from it fails descriptor validation. Lowering applies the same
        rewrite and additionally keeps the direction; here only the extent
        matters, since a result shape is the same either way round.
        """
        if not isinstance(node, ast.Subscript):
            return node
        shape = self._subscript_base_shape(node)
        if shape is None:
            return node
        rewritten, _ = reverse_normalized(node, shape, self.constant_int)
        return rewritten

    def _subscript_base_shape(self, node: ast.Subscript):
        """The shape of the container a subscript reads, or None when its base
        is not a plain registered container."""
        base = node.value
        if not isinstance(base, ast.Name):
            return None
        binding = self.context.resolve(base.id)
        if binding is None or binding.kind != 'container':
            return None
        return self.context.containers[binding.container].shape

    def _name_array_index_expressions(
            self, node: Union[ast.Name, ast.Subscript],
            defined: Dict[str, Any]) -> Tuple[Union[ast.Name, ast.Subscript], Dict[str, data.Data]]:
        """
        Give every array-valued index EXPRESSION a name the memlet parser can
        look up, so the access types as the advanced indexing it is.

        The parser recognizes an array index only as a bare name
        (``memlet_parser._fill_missing_slices``); written any other way
        (``A[ind[0]]``, ``A[2:4, ind[1], 3]``) the index becomes an applied
        symbolic function, and the access types as a single ELEMENT. That is
        the damaging failure: the shape is wrong rather than absent, so an ANF
        temporary holding the result is allocated as a scalar and the real
        error surfaces later as a broadcast-rank complaint about an unrelated
        statement.

        Which slots need this is decided once, by
        :func:`~dace.frontend.python.nextgen.semantics.indexing.array_index_slots`,
        the same classification lowering uses to materialize the very same
        expressions (``lowering.dispatch.materialize_array_indices``) -- the
        two must agree exactly, so they no longer derive it separately. The
        substitution is positional for the same reason.

        The names introduced here are for typing only: they carry a descriptor,
        not storage, never reach the tree, and are scoped to this parse, which
        is why they are returned rather than recorded on the context. They are
        added to ``defined`` (a per-parse view) so the parser can look them up.
        """
        if not isinstance(node, ast.Subscript):
            return node, {}
        described = array_index_slots(node.slice, self._array_index_descriptor)
        if not described:
            return node, {}
        replacements: Dict[int, ast.expr] = {}
        index_types: Dict[str, data.Data] = {}
        for position, descriptor in described.items():
            # Named after the slot it occupies, which is unique within the
            # access and needs no counter shared across accesses.
            name = f'__idxexpr{position}'
            index_types[name] = descriptor
            defined[name] = descriptor
            replacements[position] = ast.Name(id=name, ctx=ast.Load())
        return substitute_slots(node, replacements), index_types

    def _array_index_descriptor(self, element: ast.expr) -> Optional[data.Data]:
        """The descriptor of an index expression that is an integer or boolean
        ARRAY, or None when it is anything else (a scalar, a symbol, or an
        expression this service cannot type)."""
        try:
            inferred = self.infer(element)
        except Exception:
            return None
        if not inferred.is_data or inferred.descriptor is None:
            return None
        descriptor = inferred.descriptor
        shape = [size for size in descriptor.shape if size != 1]
        if not shape:
            return None
        dtype = descriptor.dtype
        if not isinstance(dtype, dtypes.typeclass):
            return None
        if dtype not in dtypes.INTEGER_TYPES and dtype not in (dtypes.bool, dtypes.bool_):
            return None
        return data.Array(dtype, shape)

    def _restore_index_sequences(self, node: Union[ast.Name, ast.Subscript]) -> Union[ast.Name, ast.Subscript]:
        """
        Put literal index sequences back into a subscript before parsing it.

        ``A[:, (1, 2, 3)]`` is an advanced-indexing access, but ANF hoists the
        tuple into a temporary bound to a compile-time sequence, leaving
        ``A[:, __anf0]``. The shared memlet parser recognizes an array index
        only from a literal or a registered container, so it would classify the
        temporary as a scalar symbol and silently produce a subset referring to
        a name that has no runtime value. Substituting the literal back keeps
        the two spellings equivalent, and is a no-op for every other access.

        The literal is re-encoded the way the parser expects it
        (``memlet_parser.py::_fill_missing_slices``): an ``ast.Name`` whose
        ``id`` is the Python list itself, which is what the classic frontend's
        global resolver leaves in the AST.
        """
        if not isinstance(node, ast.Subscript):
            return node
        node = self._restore_slice_objects(node)
        replacements: Dict[str, ast.expr] = {}
        for name in {n.id for n in ast.walk(node.slice) if isinstance(n, ast.Name)}:
            binding = self.context.resolve(name)
            if binding is None or binding.kind != 'static':
                continue
            sequence = self.context.static_values.get(name)
            if sequence is None:
                continue
            try:
                values = self.sequence_constants(sequence)
            except UnsupportedFeatureError:
                continue
            if not values or not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
                continue
            replacements[name] = list(values)
        if not replacements:
            return node

        class _Substituter(ast.NodeTransformer):

            def visit_Name(self, name_node: ast.Name) -> ast.AST:
                if name_node.id not in replacements:
                    return name_node
                return ast.Name(id=list(replacements[name_node.id]), ctx=ast.Load())

        restored = astutils.copy_tree(node)
        restored.slice = _Substituter().visit(restored.slice)
        return ast.copy_location(restored, node)

    def _restore_slice_objects(self, node: ast.Subscript) -> ast.Subscript:
        """
        Expand names bound to a compile-time Python ``slice`` into real slice
        syntax before parsing a subscript.

        ``A[i, j, kslice]`` where ``kslice`` is a ``dace.compiletime`` argument
        (or the result of a ``slice(...)`` call) means exactly ``A[i, j,
        start:stop:step]``, but the shared memlet parser sees only a name and
        classifies it as a scalar index — which then looks like an
        advanced-indexing array to the frontend. Rewriting it here keeps the two
        spellings equivalent.
        """

        def as_slice_node(value: Any) -> Optional[ast.Slice]:
            if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], slice):
                # ``A[i, j, (slice(2, None),)]``: a one-element tuple around a
                # slice indexes exactly like the slice itself (Python's own
                # ``A[(s,)]`` == ``A[s]``). Nested-call argument binding can
                # produce this wrapper.
                value = value[0]
            if not isinstance(value, slice):
                return None
            return ast.Slice(lower=None if value.start is None else ast.Constant(value=value.start),
                             upper=None if value.stop is None else ast.Constant(value=value.stop),
                             step=None if value.step is None else ast.Constant(value=value.step))

        names = {n.id for n in ast.walk(node.slice) if isinstance(n, ast.Name)}
        replacements: Dict[str, ast.expr] = {}
        for name in names:
            replaced = as_slice_node(self.constant_value(ast.Name(id=name, ctx=ast.Load())))
            if replaced is not None:
                replacements[name] = replaced
        has_constant_slice = any(
            isinstance(n, ast.Constant) and as_slice_node(n.value) is not None for n in ast.walk(node.slice))
        if not replacements and not has_constant_slice:
            return node

        class _SliceSubstituter(ast.NodeTransformer):

            def visit_Name(self, name_node: ast.Name) -> ast.AST:
                if name_node.id not in replacements:
                    return name_node
                return astutils.copy_tree(replacements[name_node.id])

            def visit_Constant(self, constant_node: ast.Constant) -> ast.AST:
                # Preprocessing embeds a resolved compile-time argument as a
                # Constant carrying the Python object itself.
                replaced = as_slice_node(constant_node.value)
                return constant_node if replaced is None else replaced

            def visit_Tuple(self, tuple_node: ast.Tuple) -> ast.AST:
                tuple_node = self.generic_visit(tuple_node)
                if len(tuple_node.elts) == 1 and isinstance(tuple_node.elts[0], ast.Slice):
                    # A dimension slot holding a one-element tuple around a
                    # slice (``A[i, j, (kslice,)]``, which nested-call argument
                    # binding can produce) indexes as the slice itself.
                    return tuple_node.elts[0]
                return tuple_node

        restored = astutils.copy_tree(node)
        restored.slice = _SliceSubstituter().visit(restored.slice)
        return ast.fix_missing_locations(ast.copy_location(restored, node))

    def constant_int(self, node: ast.expr) -> Optional[int]:
        """Resolve a canonical atom to a compile-time integer, or None."""
        value = self.constant_value(node)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    def constant_value(self, node: ast.expr) -> Any:
        """Resolve a canonical atom to a compile-time value, or None."""
        try:
            inferred = self.infer(node)
        except UnsupportedFeatureError:
            return None
        if inferred.kind == 'constant':
            return inferred.value
        return None

    def sequence_constants(self, sequence: StaticSequence) -> List[Any]:
        """
        Resolve all elements of a static sequence to compile-time values.

        :raises UnsupportedFeatureError: If any element is not a compile-time
            constant (e.g., references runtime data).
        """
        result = []
        for element in sequence.elements:
            value = self.constant_value(element)
            if value is None:
                # A NESTED sequence (``((4, 5, 6), [1, 2, 3])``, which ANF
                # hoists element-first) is still compile-time: resolve it the
                # same way, one dimension down.
                nested = self.static_sequence(element)
                if nested is not None:
                    result.append(self.sequence_constants(nested))
                    continue
                raise UnsupportedFeatureError(
                    f'Python sequence element "{astutils.unparse(element)}" is not a compile-time constant',
                    self.context.filename,
                    element,
                    category='static-sequence')
            result.append(value)
        return result

    def static_sequence(self, node: ast.expr) -> Optional[StaticSequence]:
        """The compile-time Python sequence a canonical atom resolves to, or
        None when it is not one."""
        try:
            inferred = self.infer(node)
        except UnsupportedFeatureError:
            return None
        return inferred.value if inferred.kind == 'static' else None

    def sequence_descriptor(self, sequence: StaticSequence) -> data.Array:
        """The constant-array descriptor a static sequence materializes to."""
        array = numpy.array(self.sequence_constants(sequence))
        return data.Array(dtypes.dtype_to_typeclass(array.dtype.type), list(array.shape))

    # ------------------------------------------------------------------ #

    def _infer_name(self, node: ast.Name) -> Inferred:
        binding = self.context.resolve(node.id)
        if binding is not None:
            if binding.kind == 'container':
                # Materialized ANF scalar temps with a known pure-symbolic
                # value stay symbolic to inference, so computed shape
                # expressions and derived temps keep compile-time values.
                symbolic_value = self.context.symbolic_scalar_values.get(binding.container)
                if symbolic_value is not None:
                    return Inferred(kind='symbolic', value=symbolic_value)
                return Inferred(kind='data', descriptor=self.context.containers[binding.container])
            if binding.kind == 'symbol':
                return Inferred(kind='symbolic', value=self.context.symbol_of(node.id))
            if binding.kind == 'static':
                return Inferred(kind='static', value=self.context.static_values[node.id])
            if binding.kind == 'constant':
                value = self.context.constant_values[node.id]
                if symbolic.issymbolic(value):
                    # A compile-time value that is a symbolic expression (e.g.
                    # ``len(A)`` on a symbolically-shaped array) belongs to the
                    # symbolic domain: it has a dtype and composes with other
                    # symbolic expressions, which the constant domain does not.
                    return Inferred(kind='symbolic', value=value)
                return Inferred(kind='constant', value=value)
        if node.id in self.context.symbols:
            return Inferred(kind='symbolic', value=self.context.symbols[node.id])
        if node.id in self.context.constants:
            return Inferred(kind='constant', value=self.context.constants[node.id][1])
        if node.id in self.context.globals:
            value = self.context.globals[node.id]
            if isinstance(value, symbolic.symbol):
                return Inferred(kind='symbolic', value=value)
            return Inferred(kind='constant', value=value)
        raise UnsupportedFeatureError(f'Use of undefined name "{node.id}"',
                                      self.context.filename,
                                      node,
                                      category='undefined-name')

    def _infer_attribute(self, node: ast.Attribute) -> Inferred:
        """Infer a structure member access (``tracers.data``) or an attribute
        chain resolving through the program globals to a compile-time value
        (``dace.int32``). Any other attribute read is a feature gap that
        degrades to the interpreter."""
        if isinstance(node.value, ast.Name):
            member = self.context.member_access_of(node.value.id, node.attr)
            if member is not None:
                return Inferred(kind='data', descriptor=member[1])
            binding = self.context.resolve(node.value.id)
            # Attribute on a compile-time constant value (e.g. an enum member
            # of a constant-bound enum class)
            if binding is not None and binding.kind == 'constant':
                base_value = self.context.constant_values[node.value.id]
                if hasattr(base_value, node.attr):
                    return Inferred(kind='constant', value=getattr(base_value, node.attr))
            # Compile-time descriptor properties of data-bound names
            # (``A.dtype``, ``A.shape``, ``A.ndim``)
            if binding is not None and binding.kind == 'container':
                descriptor = self.context.containers[binding.container]
                query = DESCRIPTOR_PROPERTIES.get(node.attr)
                if query is not None:
                    value = query(descriptor)
                    if value is not None:
                        return Inferred(kind='constant', value=value)
                # Registry-backed attributes needing an actual data operation
                # (``.T``/``.real``/``.imag``/``.flat``): typed through the
                # ATTRIBUTE family of the replacement registry so
                # ``lowering.dispatch``'s dedicated frontend paths (mirroring
                # the reshape view precedent, ``dispatch._lower_reshape_call``)
                # can materialize them. Scoped to the attributes that HAVE
                # such a path (``common.supported_data_attribute``) -- see its
                # docstring for why a registered attribute outside that set
                # must keep failing inference here rather than typing
                # successfully with no matching lowering path.
                if supported_data_attribute(node.attr):
                    from dace.frontend.common import op_repository as oprepo  # Deferred: registry population
                    infer_fn = oprepo.Replacements.get_attribute_descriptor_inference(type(descriptor), node.attr)
                    if infer_fn is not None:
                        try:
                            result = infer_fn(descriptor)
                        except Exception:
                            result = None
                        if isinstance(result, (tuple, list)) and len(result) == 1:
                            result = result[0]
                        if isinstance(result, data.Data):
                            return Inferred(kind='data', descriptor=result)
                # Registry-backed attributes computing a compile-time *value*
                # rather than data (``A.size``): asked of the registry entry
                # itself, so a newly registered one needs no change here.
                value = _attribute_value(descriptor, node.attr)
                if value is not None:
                    return Inferred(kind='symbolic' if symbolic.issymbolic(value) else 'constant', value=value)
        _, resolved = self.resolve_callee(node)
        if resolved is not None:
            if isinstance(resolved, symbolic.symbol):
                return Inferred(kind='symbolic', value=resolved)
            return Inferred(kind='constant', value=resolved)
        raise UnsupportedFeatureError(f'Cannot infer type of expression: {astutils.unparse(node)}',
                                      self.context.filename,
                                      node,
                                      category='type-inference')

    def _infer_constant_element(self, sequence, node: ast.Subscript) -> Inferred:
        """
        Index a compile-time Python sequence value (``A.shape[0]``) with a
        compile-time index.

        :raises UnsupportedFeatureError: If the index is not compile-time or
            out of range -- both are things only the interpreter can settle.
        """
        index = self.constant_int(node.slice)
        if index is None or not -len(sequence) <= index < len(sequence):
            raise UnsupportedFeatureError(f'Cannot index "{astutils.unparse(node.value)}" at compile time',
                                          self.context.filename,
                                          node,
                                          category='type-inference')
        element = sequence[index]
        if symbolic.issymbolic(element):
            return Inferred(kind='symbolic', value=element)
        return Inferred(kind='constant', value=element)

    def _basic_result_shape(self, node: ast.Subscript, expr) -> List:
        """
        The NumPy result shape of a basic-indexing access, from the shared
        index model.

        The model keeps a slice-formed dimension at size 1 (``A[0:1, :]`` is
        ``(1, 10)``) and inserts ``newaxis`` dimensions -- both of which the
        previous ``[s for s in expr.subset.size() if s != 1]`` got wrong,
        silently, wherever a value was consumed as an operand rather than
        bound to a name.
        """
        from dace.frontend.python.nextgen.semantics.indexing import build_plan

        plan = build_plan(node.slice, len(expr.subset.ranges))
        if plan is None:
            return [size for size in expr.subset.size() if size != 1]
        return plan.result_shape(expr.subset.size())

    def _infer_subscript(self, node: ast.Subscript) -> Inferred:
        base = self.infer(node.value)
        if base.kind == 'static':
            element = values.fold_subscript(base.value, node, self.constant_int)
            if isinstance(element, ast.expr):
                return self.infer(element)
            return Inferred(kind='static', value=element)
        if base.kind == 'constant' and isinstance(base.value, (tuple, list)):
            # A compile-time Python sequence that is a VALUE rather than a
            # source-level literal -- ``A.shape``, most of all, whose elements
            # may be symbols. Indexing it folds here; the alternative is a
            # callback around ``A.shape[0]``, which then poisons the loop bound
            # that reads it.
            return self._infer_constant_element(base.value, node)
        if not base.is_data:
            raise UnsupportedFeatureError('Subscript of a non-container value',
                                          self.context.filename,
                                          node,
                                          category='type-inference')
        expr, index_types = self.parse_access_typed(node)
        # Array-read indices (``x[A_col[j]]``) are NOT rejected here: the
        # shared memlet parser represents them as applied sympy functions,
        # whose ``.size()`` still computes a definite shape, and the
        # elementwise computation mechanism now lowers this pattern as
        # indirection (see ``lowering.access.indirect_index_reads``). A
        # consumer that cannot handle indirection (e.g. an assignment target,
        # or a ufunc/creation-call argument) re-resolves the same expression
        # through ``resolve_access``, which keeps rejecting it there.
        if expr.arrdims:
            # Advanced (array-valued) indexing: the result shape follows NumPy's
            # own rules, not the subset's -- index arrays broadcast together and
            # collapse the indexed dimensions into one chunk.
            from dace.frontend.python.nextgen.lowering.mechanisms import advanced_indexing
            if advanced_indexing.has_boolean_index(expr, self.context, index_types):
                # A boolean mask's result size depends on how many elements it
                # selects, which is not known until it is actually read --
                # genuinely undefined at this (pure, side-effect-free) point,
                # not merely unhandled. Only the bare top-level assignment
                # form (``B = A[mask]``) resolves this deferred dimension,
                # in ``rules.assign._lower_boolean_gather_assign``, which is
                # tried BEFORE this function ever runs for that case; this
                # answer is for every other use (nested in an expression,
                # combined with another index), which still cannot lower and
                # will fall back to a callback downstream.
                return Inferred(kind='data', descriptor=data.Array(base.descriptor.dtype, [symbolic.UndefinedSymbol()]))
            shape = advanced_indexing.output_shape(expr, self.context, self, node, index_types)
            if not shape:
                return Inferred(kind='data', descriptor=data.Scalar(base.descriptor.dtype))
            return Inferred(kind='data', descriptor=data.Array(base.descriptor.dtype, shape))
        try:
            shape = self._basic_result_shape(node, expr)
            if not shape:
                return Inferred(kind='data', descriptor=data.Scalar(base.descriptor.dtype))
            return Inferred(kind='data', descriptor=data.Array(base.descriptor.dtype, shape))
        except UnsupportedFeatureError:
            raise
        except Exception as error:
            # Exotic subsets produce non-real symbolic sizes (e.g. sympy zoo)
            # that crash descriptor validation; treat them as feature gaps.
            raise UnsupportedFeatureError(f'Cannot infer subscript shape of "{astutils.unparse(node)}": {error}',
                                          self.context.filename,
                                          node,
                                          category='data-dependent-subscript')

    def dtype_of(self, inferred: Inferred) -> Optional[dtypes.typeclass]:
        """
        Context-aware dtype of an inferred value. Symbolic expressions resolve
        their symbols' dtypes by name through the context: sympy symbol
        identity ignores the dace dtype attribute, so the objects embedded in
        an expression may carry stale defaults from the process-wide cache.
        """
        if inferred.kind == 'symbolic':
            return self.symbolic_dtype(inferred.value)
        return inferred.dtype

    def symbolic_dtype(self, expression: Any) -> dtypes.typeclass:
        """The result dtype of a pure symbolic expression, resolving each free
        symbol by name through the context (with NumPy promotion when the
        symbol dtypes differ)."""
        found: List[dtypes.typeclass] = []
        for free_symbol in getattr(expression, 'free_symbols', ()):
            name = str(free_symbol)
            registered = self.context.symbols.get(name)
            if registered is None:
                global_value = self.context.globals.get(name)
                if isinstance(global_value, symbolic.symbol):
                    registered = global_value
            candidate = registered if registered is not None else free_symbol
            found.append(getattr(candidate, 'dtype', symbolic.DEFAULT_SYMBOL_TYPE))
        if not found:
            return symbolic.DEFAULT_SYMBOL_TYPE
        return dtypes.result_type_of(found[0], *found[1:]) if len(found) > 1 else found[0]

    def _symbolic_expression(self, node: ast.expr) -> Any:
        """
        Build the symbolic value of a purely symbolic/constant expression.

        Parsing the source text mints fresh default-typed symbols for every
        name, so free symbols are substituted with their known context values:
        the recorded symbolic values of materialized ANF scalar temps, and the
        registered (correctly typed) symbol objects for program symbols.
        """
        expr = symbolic.pystr_to_symbolic(astutils.unparse(node))
        if not hasattr(expr, 'free_symbols'):
            return expr
        replacements = {}
        for free_symbol in expr.free_symbols:
            name = str(free_symbol)
            binding = self.context.resolve(name)
            if binding is not None and binding.kind == 'container':
                value = self.context.symbolic_scalar_values.get(binding.container)
                if value is not None:
                    replacements[free_symbol] = value
                    continue
            registered = self.context.symbols.get(name)
            if registered is None:
                # Symbols of inlined callees resolve through their globals
                global_value = self.context.globals.get(name)
                if isinstance(global_value, symbolic.symbol):
                    registered = global_value
            if registered is not None and registered is not free_symbol:
                replacements[free_symbol] = registered
        return expr.subs(replacements) if replacements else expr

    def _infer_operator(self, node: ast.expr) -> Inferred:
        if isinstance(node, ast.BinOp):
            operands = [self.infer(node.left), self.infer(node.right)]
        elif isinstance(node, ast.Compare):
            operands = [self.infer(node.left)] + [self.infer(c) for c in node.comparators]
        else:  # BoolOp
            operands = [self.infer(v) for v in node.values]

        # Operators over Python sequences follow Python semantics at compile
        # time; sequences mixed with data operands materialize as constant
        # arrays and participate in broadcasting instead.
        if isinstance(node, ast.BinOp) and any(op.kind == 'static' for op in operands):
            if not any(op.is_data for op in operands):
                left_sequence = operands[0].value if operands[0].kind == 'static' else None
                right_sequence = operands[1].value if operands[1].kind == 'static' else None
                return Inferred(kind='static',
                                value=values.fold_binop(node, left_sequence, right_sequence, self.constant_int))
            operands = [
                Inferred(kind='data', descriptor=self.sequence_descriptor(op.value)) if op.kind == 'static' else op
                for op in operands
            ]

        # Opaque Python objects poison the expression: consuming them requires
        # the interpreter, which the dispatch seam turns into a callback.
        if any(op.is_pyobject for op in operands):
            return Inferred(kind='data', descriptor=data.Scalar(dtypes.pyobject()))

        boolean_result = isinstance(node, _BOOLEAN_OPS)
        data_operands = [op for op in operands if op.is_data]
        if not data_operands:
            # Purely symbolic/constant expression
            return Inferred(kind='symbolic', value=self._symbolic_expression(node))

        # Operators with their own registry inference (``@``, whose result is a
        # contraction rather than a broadcast) answer for themselves. Without
        # this, ``(24, 12) @ (12, 48)`` broadcasts to (24, 12) and lowers as an
        # elementwise multiply -- the wrong answer, silently.
        registry_result = self._infer_registry_operator(node, operands)
        if registry_result is not None:
            return registry_result

        result_dtype = self._result_dtype(operands, boolean_result)
        shape: Tuple[Any, ...] = ()
        for operand in data_operands:
            operand_shape = tuple(operand.descriptor.shape) if isinstance(operand.descriptor, data.Array) else ()
            shape = broadcast_shapes(shape, operand_shape)
        if not shape:
            # No array operand contributed a dimension: a genuine scalar
            # expression (``A[0] + 1``). A size-1 array operand is NOT one --
            # NumPy keeps the dimension (``numpy.zeros(1) + 3`` has shape
            # (1,)), and collapsing it here made ``total += x[i]`` on a
            # ``float64[1]`` parameter incompatible with its own container, so
            # the name was rebound to a fresh scalar and the parameter was
            # never written.
            return Inferred(kind='data', descriptor=data.Scalar(result_dtype))
        return Inferred(kind='data', descriptor=data.Array(result_dtype, list(shape)))

    def _infer_registry_operator(self, node: ast.expr, operands: List['Inferred']) -> Optional['Inferred']:
        """
        The result of an operator the replacement registry OVERRIDES for these
        operand types, or None when it does not — the broadcast rule below
        answers those.

        An operator counts as overridden when the implementation registered for
        its operand classes is not the stock elementwise one (see
        ``op_repository.ELEMENTWISE_OPERATOR_ATTRIBUTE``). Python operators are
        dunder methods, so an override may do anything: contract (``@``), move
        storage (``A @ StorageType.GPU_Global``), reduce, reshape. Nothing here
        may assume the result has the broadcast shape of the operands, so the
        registry's own inference is asked for it.
        """
        from dace.frontend.common import op_repository as oprepo  # Deferred: registry population
        resolved = registry_operator_operands(node, operands)
        if resolved is None:
            return None
        optype, values = resolved
        implementation = oprepo.Replacements.getop(*operator_lookup_arguments(optype, values))
        if implementation is None or oprepo.is_elementwise_operator(implementation):
            return None
        infer_fn = oprepo.Replacements.get_operator_descriptor_inference(optype, *values)
        result = None
        if infer_fn is not None:
            try:
                result = infer_fn(*values)
            except Exception:
                result = None
        if not isinstance(result, data.Data):
            # The registry owns this operator's meaning here and could not type
            # it, so nothing else may guess: the statement goes to a callback.
            raise UnsupportedFeatureError(f'Cannot determine the result of "{optype}" on these operands',
                                          self.context.filename,
                                          node,
                                          category='type-inference')
        return Inferred(kind='data', descriptor=result)

    def _result_dtype(self, operands: List[Inferred], boolean_result: bool) -> dtypes.typeclass:
        """
        The dtype of an operator's result, with WEAK promotion for Python
        scalar literals (NumPy's NEP 50, the rule NumPy 2 implements): a plain
        ``1`` or ``1.5`` written in the source takes the dtype of the typed
        operands instead of widening it, so ``i + 1`` on an ``int32`` stays
        ``int32``. A literal of a HIGHER kind still promotes, to the default
        dtype of its own kind (``int32_array + 1.5`` is float64).

        Widening here is not merely imprecise: it made every ``i += 1`` counter
        rebind its name to a differently-typed container, which inside a loop
        is a loop-carried rebinding and forced the whole loop into a callback.
        """
        if boolean_result:
            return dtypes.bool_
        strong: List[dtypes.typeclass] = []
        weak_rank: Optional[int] = None
        for operand in operands:
            rank = _weak_scalar_rank(operand)
            if rank is not None:
                weak_rank = rank if weak_rank is None else max(weak_rank, rank)
                continue
            dtype = self.dtype_of(operand)
            if dtype is not None:
                strong.append(dtype)
        if not strong:
            if weak_rank is None:
                raise UnsupportedFeatureError('Cannot determine operator result type',
                                              self.context.filename,
                                              category='type-inference')
            return _WEAK_KIND_DEFAULTS[weak_rank]
        result = dtypes.result_type_of(strong[0], *strong[1:]) if len(strong) > 1 else strong[0]
        if weak_rank is not None and weak_rank > _dtype_kind_rank(result):
            result = dtypes.result_type_of(result, _WEAK_KIND_DEFAULTS[weak_rank])
        return result

    def _demote_to_bool(self, operand: Inferred) -> Inferred:
        if operand.is_data:
            descriptor = operand.descriptor
            if isinstance(descriptor, data.Array):
                return Inferred(kind='data', descriptor=data.Array(dtypes.bool_, list(descriptor.shape)))
            return Inferred(kind='data', descriptor=data.Scalar(dtypes.bool_))
        return operand
