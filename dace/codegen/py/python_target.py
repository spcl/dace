# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import copy
import re
from typing import TYPE_CHECKING, Optional
import warnings

from dace import Config, data, dtypes, memlet as mmlt, registry, subsets, symbolic
from dace.utils import prod
import dace.codegen.dispatcher as dispatcher_mod
from dace.codegen.common import update_persistent_desc
from dace.codegen.py import utils as pyutils
from dace.codegen.py.framecode import (_collect_nested_runtime_defined_names, _collect_runtime_defined_names,
                                       _collect_runtime_used_names, codeblock_to_python)
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.target import PythonTargetCodeGenerator
from dace.frontend.python import astutils
from dace.sdfg import NodeNotExpandedError, SDFG, ScopeSubgraphView, dynamic_map_inputs, nodes
from dace.sdfg.scope import scope_contains_scope
from dace.sdfg.state import ControlFlowRegion

if TYPE_CHECKING:
    from dace.codegen.dispatcher import TargetDispatcher
    from dace.codegen.py.framecode import DaCePythonCodeGenerator


def _python_expr(expr) -> str:
    if isinstance(expr, ast.AST):
        return astutils.unparse(expr) or ''
    return symbolic.symstr(expr, cpp_mode=False)


#: Dtype names a kept cast can carry, built from the dtype registry.
_PY_CAST_DTYPES = {s.split('::')[-1] for s in dtypes.TYPECLASS_TO_STRING.values()}
_DACE_DTYPE_CAST_RE = re.compile(r'\bdace\.(' + '|'.join(sorted(_PY_CAST_DTYPES, key=len, reverse=True)) + r')\b')


def _rewrite_dace_dtype_casts(body: str) -> str:
    """Repoint a kept ``dace.<dtype>(x)`` cast at ``numpy``.

    VTI keeps dtype casts rather than stripping them, so a host Python tasklet can
    carry ``dace.float64(x)``; the generated module imports ``numpy`` but not ``dace``.
    Only the ``dace.<dtype>`` token is rewritten -- the rest of the body is untouched.
    """
    return _DACE_DTYPE_CAST_RE.sub(r'numpy.\1', body)


def _python_view_component(start, end, step) -> str:
    try:
        start_int = int(start)
        end_int = int(end)
        step_int = int(step)
    except (TypeError, ValueError) as e:
        # We likely have a symbolic expression
        start_int = end_int = step_int = None

    if step_int is not None:
        stop_expr = str(end_int + 1) if step_int > 0 else str(end_int - 1)
        if step_int == 1:
            return f'{start_int}:{stop_expr}'
        return f'{start_int}:{stop_expr}:{step_int}'

    start_expr = _python_expr(start)
    end_expr = _python_expr(end)
    step_expr = _python_expr(step)
    stop_expr = f'(({end_expr}) + (1 if ({step_expr}) > 0 else -1))'
    if step_expr == '1':
        return f'{start_expr}:{stop_expr}'
    return f'{start_expr}:{stop_expr}:{step_expr}'


def _python_type(dtype: dtypes.typeclass) -> str:
    return dtypes.PYTHON_TYPES.get(dtype.type, str(dtype))


def _numpy_dtype(dtype: dtypes.typeclass) -> str:
    return dtypes.NUMPY_TYPES[dtype.type]


def _structure_type_name(desc: data.Structure) -> str:
    return desc.name


def _defined_type_for(desc: data.Data):
    if isinstance(desc, data.Scalar):
        return dispatcher_mod.DefinedType.Scalar
    if isinstance(desc, data.Structure):
        return dispatcher_mod.DefinedType.Object
    return dispatcher_mod.DefinedType.Pointer


def _is_inside_cutile_scope(cfg: ControlFlowRegion, state_id: int, node: nodes.Node) -> bool:
    """True if *node* lies inside a CuTile-scheduled map scope.

    cuTile kernels allocate tile transients as Python locals via ct.load /
    tasklet writes — there should be no module-level numpy.zeros allocation
    for them.
    """
    if state_id is None or state_id < 0:
        return False
    try:
        state = cfg.state(state_id)
    except Exception:
        return False
    scope = state.scope_dict()
    cur = scope.get(node)
    while cur is not None:
        if (isinstance(cur, nodes.MapEntry) and cur.map.schedule == dtypes.ScheduleType.CuTile):
            return True
        cur = scope.get(cur)
    return False


def _sdfg_uses_cutile(sdfg: SDFG) -> bool:
    """True if *sdfg* (or any nested SDFG) contains a CuTile-scheduled map."""
    found = False
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.CuTile:
            found = True
            break
    return found


def _sdfg_needs_cupy(sdfg: SDFG) -> bool:
    """True if the generated module touches GPU (cupy) arrays.

    Mirrors the allocation rule in :meth:`PythonCodeGen.allocate_array`: a
    ``GPU_Global`` Array (or any non-Register Array transient of a cuTile
    SDFG) allocates with ``cupy.empty``, so the frame needs ``import cupy``
    even when the cuTile target itself is never dispatched (e.g. an SDFG with
    ``GPU_Global`` transients but no host<->device copies and no CuTile map).
    Scalars never allocate with cupy, so only Array descriptors are scanned.
    """
    if _sdfg_uses_cutile(sdfg):
        return True
    return any(
        isinstance(desc, data.Array) and desc.storage == dtypes.StorageType.GPU_Global
        for _, _, desc in sdfg.arrays_recursive())


def _defined_ptype_for(desc: data.Data) -> str:
    if isinstance(desc, data.Scalar):
        return _python_type(desc.dtype)
    if isinstance(desc, data.Structure):
        return _structure_type_name(desc)
    if isinstance(desc, data.Array):
        return 'numpy.ndarray'
    return 'object'


@registry.autoregister_params(name='python')
class PythonCodeGen(PythonTargetCodeGenerator):
    title = 'Python'
    target_name = 'python'
    language = 'python'

    def __init__(self, frame_codegen: 'DaCePythonCodeGenerator', sdfg: SDFG):
        self._frame = frame_codegen
        self._dispatcher: 'TargetDispatcher' = frame_codegen.dispatcher
        self._sdfg = sdfg
        self.calling_codegen = self
        self._toplevel_schedule = None
        self._generated_nodes = set()
        self._generated_nested_sdfg: dict[str, str] = {}

        frame_arglist = getattr(self._frame, 'arglist', None)
        if frame_arglist is None:
            frame_arglist = sdfg.arglist(scalars_only=False)
        if hasattr(self._dispatcher, 'defined_vars'):
            self._define_sdfg_arguments(sdfg, dict(frame_arglist))

        dispatcher = self._dispatcher
        dispatcher.register_node_dispatcher(self)

        SUPPORTED_SCHEDULES = [
            # This is kind of a hack. In practice all of these are treated as sequential schedules
            # TODO: Refactor this somehow, for example by not defaulting to CPU_Multicore for maps in infer_types
            dtypes.ScheduleType.CPU_Multicore,
            dtypes.ScheduleType.CPU_Persistent,
            dtypes.ScheduleType.Sequential,
        ]
        COPY_SCHEDULES = [*SUPPORTED_SCHEDULES]
        dispatcher.register_map_dispatcher(SUPPORTED_SCHEDULES, self)

        # Is GPU_Global correct?
        supported_storage = [dtypes.StorageType.CPU_Heap, dtypes.StorageType.Register, dtypes.StorageType.GPU_Global]
        for storage in supported_storage:
            dispatcher.register_array_dispatcher(storage, self)
        for src_storage in supported_storage:
            for dst_storage in supported_storage:
                dispatcher.register_copy_dispatcher(src_storage, dst_storage, None, self)
                for schedule in COPY_SCHEDULES:
                    dispatcher.register_copy_dispatcher(src_storage, dst_storage, schedule, self)

    def get_generated_codeobjects(self):
        # Unfortunately we need to redefine some sympy functions so we just load that file as a code object
        from pathlib import Path

        HERE = Path(__file__).parent
        file_path = HERE / "sympy_function_redefinitions.py"
        content = file_path.read_text()

        from dace.codegen.codeobject import CodeObject

        code = CodeObject(
            name="sympy_function_redefinitions",
            code=content,
            language="py",
            target=type(self),
            title="Sympy Function Redefinitions",
        )
        return [code]

    def get_includes(self) -> dict[str, list[str]]:
        includes = [
            'import numpy',
            'from dataclasses import dataclass',
            "from sympy_function_redefinitions import *",
        ]
        if _sdfg_needs_cupy(self._sdfg):
            # GPU_Global allocations emit cupy.empty; do not rely on the cuTile
            # target being dispatch-'used' to provide the import.
            includes.insert(1, 'import cupy')
        return {'frame': includes}

    def preprocess(self, sdfg: SDFG) -> None:
        """Strip remaining View access nodes before Python backend codegen.

        The Python backend does not support View access nodes and raises
        ``NotImplementedError`` when one is encountered during code generation.
        Therefore we hopefully remove them here
        """
        from dace.transformation.passes.remove_views import RemoveViews
        remove_views = RemoveViews()
        for nsdfg in sdfg.all_sdfgs_recursive():
            remove_views.apply_pass(nsdfg, {})

    @property
    def has_initializer(self):
        return False

    @property
    def has_finalizer(self):
        return False

    def _define_sdfg_arguments(self, sdfg: SDFG, arglist):

        def _visit_structure(struct: data.Structure, prefix: str) -> None:
            for field_name, field_desc in struct.members.items():
                member_name = f'{prefix}.{field_name}'
                if isinstance(field_desc, data.Structure):
                    self._dispatcher.defined_vars.add(member_name, dispatcher_mod.DefinedType.Object,
                                                      _structure_type_name(field_desc))
                    _visit_structure(field_desc, member_name)
                elif isinstance(field_desc, data.Array):
                    self._dispatcher.defined_vars.add(member_name, dispatcher_mod.DefinedType.Pointer, 'numpy.ndarray')
                elif isinstance(field_desc, data.Scalar):
                    self._dispatcher.defined_vars.add(member_name, dispatcher_mod.DefinedType.Scalar,
                                                      _python_type(field_desc.dtype))
                else:
                    raise NotImplementedError(
                        f'Python backend only supports Scalars, Arrays, and nested Structures in Structures. '
                        f'Unsupported member type: {type(field_desc).__name__}')

        for name, arg_type in arglist.items():
            if isinstance(arg_type, data.Stream):
                raise NotImplementedError('Streams are not supported for the Python backend.')
            if isinstance(arg_type, data.View):
                raise NotImplementedError('Views are not supported for the Python backend.')
            if isinstance(arg_type, data.Reference):
                raise NotImplementedError('References are not supported for the Python backend.')
            if isinstance(arg_type, data.Structure):
                self._dispatcher.defined_vars.add(name, dispatcher_mod.DefinedType.Object,
                                                  _structure_type_name(arg_type))
                _visit_structure(arg_type, name)
            elif isinstance(arg_type, data.Array):
                self._dispatcher.defined_vars.add(name, dispatcher_mod.DefinedType.Pointer, 'numpy.ndarray')
            elif isinstance(arg_type, data.Scalar):
                self._dispatcher.defined_vars.add(name, dispatcher_mod.DefinedType.Scalar, _python_type(arg_type.dtype))
            else:
                raise TypeError(f'Unrecognized argument type: {type(arg_type).__name__}')

    def _is_scalar_buffer(self, name: str, desc: data.Data) -> bool:
        return isinstance(desc, data.Scalar) and not desc.transient and '.' not in name

    def _shape_expression(self, shape) -> str:
        dims = [_python_expr(dim) for dim in shape]
        if len(dims) == 1:
            return f'({dims[0]},)'
        return f'({", ".join(dims)})'

    def _scalar_default(self, desc: data.Scalar) -> str:
        if desc.dtype.type is bool:
            return 'False'
        return '0'

    def _structure_default_expression(self, desc: data.Structure) -> str:
        members = []
        for field_name, field_desc in desc.members.items():
            if isinstance(field_desc, data.Structure):
                value = self._structure_default_expression(field_desc)
            elif isinstance(field_desc, data.Array):
                value = self._alloc_expr(field_desc)
            elif isinstance(field_desc, data.Scalar):
                value = self._scalar_default(field_desc)
            else:
                raise NotImplementedError(
                    f'Python backend only supports Scalars, Arrays, and nested Structures in Structures. '
                    f'Unsupported member {field_name}: {type(field_desc).__name__}')
            members.append(f'{field_name}={value}')
        return f'{_structure_type_name(desc)}({", ".join(members)})'

    def _default_expression(self, desc: data.Data, *, on_gpu: bool = False, setzero: bool = False) -> str:
        """Returns an expression that evaluates to a default-initialized value of the given descriptor's type.

        :param desc: The data descriptor to generate an expression for.
        :param on_gpu: Whether to use cupy (GPU) instead of numpy (CPU).
        :param setzero: Whether to zero-initialize the allocation (``zeros``) or leave it
            uninitialized (``empty``).  Mirrors ``AccessNode.setzero``.
        """
        if isinstance(desc, data.Structure):
            return self._structure_default_expression(desc)
        if isinstance(desc, data.Array):
            return self._alloc_expr(desc, on_gpu=on_gpu, setzero=setzero)
        if isinstance(desc, data.Scalar):
            return self._scalar_default(desc)
        raise NotImplementedError(f'Unsupported descriptor in Python backend: {type(desc).__name__}')

    def _alloc_expr(self, desc: data.Array, *, on_gpu: bool = False, setzero: bool = False) -> str:
        module = 'cupy' if on_gpu else 'numpy'
        func = 'zeros' if setzero else 'empty'
        return f'{module}.{func}({self._shape_expression(desc.shape)}, dtype={_numpy_dtype(desc.dtype)})'

    def _persistent_key(self, sdfg: SDFG, name: str) -> str:
        return f'{sdfg.cfg_id}:{name}'

    def _register_defined_name(self, name: str, desc: data.Data, *, is_global: bool = False) -> None:
        define = self._dispatcher.defined_vars.add_global if is_global else self._dispatcher.defined_vars.add
        define(name, _defined_type_for(desc), _defined_ptype_for(desc))

    def _register_structure_members(self, name: str, desc: data.Structure, *, is_global: bool = False) -> None:
        for field_name, field_desc in desc.members.items():
            member_name = f'{name}.{field_name}'
            self._register_defined_name(member_name, field_desc, is_global=is_global)
            if isinstance(field_desc, data.Structure):
                self._register_structure_members(member_name, field_desc, is_global=is_global)

    def _normalize_subset(self, subset):
        if isinstance(subset, subsets.Subset) or subset is None:
            return subset
        # TODO: Is this correct?
        return None

    def _runtime_data_name(self, sdfg: SDFG, name: str) -> str:
        root_name, separator, suffix = name.partition('.')
        root_desc = sdfg.arrays.get(root_name)
        if root_desc is None:
            return name
        if root_desc.lifetime not in (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.Persistent,
                                      dtypes.AllocationLifetime.External):
            return name
        base_name = f'globals()[{root_name!r}]'
        if not separator:
            return base_name
        return f'{base_name}.{suffix}'

    def _nested_view_expr(self, sdfg: SDFG, data_name: str, subset) -> str:
        runtime_name = self._runtime_data_name(sdfg, data_name)
        actual_subset = self._normalize_subset(subset)
        if actual_subset is None:
            return runtime_name
        if isinstance(actual_subset, subsets.Range):
            components = ', '.join(
                _python_view_component(start, end, step) for start, end, step in actual_subset.ranges)
            return f'{runtime_name}[{components}]'
        raise NotImplementedError(f'Unsupported subset type for nested SDFG connector: {type(actual_subset).__name__}')

    @staticmethod
    def _unify_symbols_by_name(expr: symbolic.SymbolicType) -> symbolic.SymbolicType:
        """Unify symbol names without changing typed literals."""
        expr = symbolic.pystr_to_symbolic(expr)
        return symbolic.replace_symbols(expr, {str(sym): symbolic.symbol(str(sym)) for sym in expr.free_symbols})

    @classmethod
    def _provably_equal(cls, a: symbolic.SymbolicType, b: symbolic.SymbolicType,
                        facts: Optional[symbolic.Facts] = None) -> bool:
        """Return whether the values are equal under the caller's facts."""
        try:
            facts = facts if facts is not None else symbolic.Facts.none()
            lhs = symbolic.simplify(cls._unify_symbols_by_name(a), facts)
            rhs = symbolic.simplify(cls._unify_symbols_by_name(b), facts)
            return symbolic.ask(symbolic.Relation(symbolic.RelationKind.EQ, lhs, rhs), facts) is symbolic.Truth.TRUE
        except (TypeError, ValueError):
            return False

    @classmethod
    def _is_c_contiguous_layout(cls, shape, strides, facts: Optional[symbolic.Facts] = None) -> bool:
        """Whether ``strides`` are provably the canonical C-contiguous strides
        for ``shape``. Strides of provably size-1 dimensions are ignored.

        :param shape: The shape (may contain symbolic dimensions).
        :param strides: The strides (may contain symbolic values).
        :returns: True when the layout is provably C-contiguous.
        """
        expected = 1
        for dim, stride in zip(reversed(list(shape)), reversed(list(strides))):
            if not cls._provably_equal(1, dim, facts) and not cls._provably_equal(stride, expected, facts):
                return False
            expected = expected * dim
        return True

    def _nested_shapes_match(self, memlet: mmlt.Memlet, desc: data.Array, facts: Optional[symbolic.Facts] = None) -> bool:
        """Whether the outer memlet subset and the nested connector have
        provably identical ordered shapes (including singleton positions).

        :param memlet: The connector's memlet.
        :param desc: The nested connector's array descriptor.
        :returns: True when no reconciliation is needed.
        """
        subset_size = list(memlet.subset.size())
        conn_shape = list(desc.shape)
        return len(subset_size) == len(conn_shape) and all(
            self._provably_equal(a, b, facts) for a, b in zip(subset_size, conn_shape))

    def _outer_view_dims(self, outer_desc: data.Array, subset) -> Optional[list]:
        """Per-dimension (size, stride, start, end, step) of the view the outer
        memlet subset selects from ``outer_desc``; strides in elements of the
        underlying array (subset step folded in).

        :param outer_desc: The outer array descriptor.
        :param subset: The (normalized) memlet subset, or None for the whole array.
        :returns: A list of (size, stride, start, end, step) tuples, or None
            when the subset type is unsupported.
        """
        if subset is None:
            return [(size, stride, 0, size - 1, 1) for size, stride in zip(outer_desc.shape, outer_desc.strides)]
        if isinstance(subset, subsets.Range):
            return [(size, stride * step, start, end, step)
                    for (start, end, step), size, stride in zip(subset.ranges, subset.size(), outer_desc.strides)]
        return None

    def _nested_strided_view_expr(self, sdfg: SDFG, memlet: mmlt.Memlet, desc: data.Array, facts: Optional[symbolic.Facts] = None) -> Optional[str]:
        """Expression for a genuine strided view of the outer array matching
        the nested connector's declared shape/strides, or None when no such
        view provably exists.

        The connector's non-singleton dims must map one-to-one (possibly
        permuted) onto the non-singleton dims of the outer subset view, with
        provably equal sizes and element strides. The view is then a slice
        (integer-indexing away singleton subset dims), an optional
        ``.transpose``, and optional ``None``-indexing to insert singleton
        connector dims -- all pure view operations for numpy and cupy alike,
        hence safe for inputs AND outputs (writes land in the outer array).

        :param sdfg: The parent SDFG.
        :param memlet: The connector's memlet.
        :param desc: The nested connector's array descriptor.
        :returns: A Python view expression, or None.
        """
        outer_desc = sdfg.arrays[memlet.data]
        if not isinstance(outer_desc, data.Array) or outer_desc.dtype != desc.dtype:
            return None
        outer_dims = self._outer_view_dims(outer_desc, self._normalize_subset(memlet.subset))
        if outer_dims is None:
            return None

        # Subscript: integer index drops singleton dims, slice keeps the rest.
        components = []
        kept = []  # (size, stride) of kept outer dims
        for size, stride, start, end, step in outer_dims:
            if self._provably_equal(1, size, facts):
                components.append(_python_expr(start))
            else:
                components.append(_python_view_component(start, end, step))
                kept.append((size, stride))

        conn_nonsingleton = [(size, stride) for size, stride in zip(desc.shape, desc.strides)
                             if not self._provably_equal(1, size, facts)]
        if len(kept) == 0 or len(conn_nonsingleton) != len(kept):
            return None

        # Match connector dims to kept outer dims: identity order first, then
        # a permutation (on provably equal size AND stride).
        if all(
                self._provably_equal(cs, os, facts) and self._provably_equal(ct, ot, facts)
                for (cs, ct), (os, ot) in zip(conn_nonsingleton, kept)):
            perm = list(range(len(kept)))
        else:
            perm = []
            unmatched = list(range(len(kept)))
            for conn_size, conn_stride in conn_nonsingleton:
                match = next(
                    (k for k in unmatched
                     if self._provably_equal(conn_size, kept[k][0], facts) and self._provably_equal(conn_stride, kept[k][1], facts)),
                    None)
                if match is None:
                    return None
                unmatched.remove(match)
                perm.append(match)

        expr = f'{self._runtime_data_name(sdfg, memlet.data)}[{", ".join(components)}]'
        if perm != list(range(len(kept))):
            expr = f'{expr}.transpose({tuple(perm)})'
        if len(conn_nonsingleton) != len(desc.shape):
            # Insert singleton connector dims via None-indexing (a pure view).
            inserts = ', '.join('None' if self._provably_equal(1, size, facts) else ':' for size in desc.shape)
            expr = f'{expr}[{inserts}]'
        return expr

    def _nested_desc_spans_full_outer(self, memlet: mmlt.Memlet, desc: data.Array, outer_desc: data.Data, facts: Optional[symbolic.Facts] = None) -> bool:
        """Whether the nested connector is declared over the WHOLE outer array
        while the memlet subset is only the (zero-based, step-1) accessed window.

        This is the widened-boundary shape (``ExpandNestedSDFGInputs``) and the
        unsimplified-frontend shape (``to_sdfg(simplify=False)`` with a
        loop-dependent window such as ``A[0:i+1, 0:M]`` into a connector declared
        ``(N, M)``): inner subscripts are outer-origin coordinates, so the whole
        outer array is the exact argument — no reshape or copy is involved.

        :param memlet: The connector's memlet (subset = accessed window).
        :param desc: The nested connector's array descriptor.
        :param outer_desc: The outer array descriptor.
        :returns: True when the whole outer array can be passed directly.
        """
        if not isinstance(outer_desc, data.Array) or isinstance(outer_desc, data.View):
            # A View is not the array itself; self-contained even though
            # ``preprocess`` normally strips Views beforehand.
            return False
        if outer_desc.dtype != desc.dtype:
            return False
        if memlet.other_subset is not None:
            # An explicit inner-side subset means inner subscripts are NOT
            # outer-origin coordinates; the whole-array shortcut is unsound.
            return False
        conn_shape = list(desc.shape)
        outer_shape = list(outer_desc.shape)
        if len(conn_shape) != len(outer_shape):
            return False
        if not all(self._provably_equal(a, b, facts) for a, b in zip(conn_shape, outer_shape)):
            return False
        subset = self._normalize_subset(memlet.subset)
        if subset is None:
            return True  # Whole-array memlet.
        if not isinstance(subset, subsets.Range):
            return False
        return all(
            self._provably_equal(start, 0, facts) and self._provably_equal(1, step, facts) for start, _end, step in subset.ranges)

    def _nested_arg_needs_flat_reshape(self, connector_name: str, memlet: mmlt.Memlet, desc: data.Array, facts: Optional[symbolic.Facts] = None) -> bool:
        """Whether a nested-SDFG array connector needs (and safely admits) a
        flat C-order ``.reshape`` from the outer memlet subset to its shape.

        The outer subset and the connector shape may legitimately differ when a
        reshape :class:`~dace.data.View` was rewritten (e.g. by
        :class:`~dace.transformation.passes.remove_views.RemoveViews`) into a
        differently-shaped slice of the underlying array -- a ``(NQ, 1, NP)``
        view becomes a ``(1, NQ, NP)`` slice. A flat reshape reconciles those
        only when the flat element order is preserved: the element counts must
        agree AND the connector's declared strides must be the canonical
        C-contiguous strides for its shape. Non-contiguous (e.g. permuted)
        connector strides indicate a transpose-like view, which a flat reshape
        would silently turn into transposed data.

        ``prod`` handles symbolic dims (unlike ``math.prod``).

        :param connector_name: The nested SDFG connector name (for diagnostics).
        :param memlet: The connector's memlet (its subset gives the outer shape).
        :param desc: The nested connector's array descriptor.
        :returns: True when a flat reshape is needed and provably safe, False
            when the shapes already match.
        :raises NotImplementedError: When the shapes mismatch but a flat
            reshape is not provably order-preserving.
        """
        subset_size = list(memlet.subset.size())
        conn_shape = list(desc.shape)
        # Identical ordered shapes (including singleton positions) need nothing.
        if self._nested_shapes_match(memlet, desc, facts):
            return False
        # A pure squeeze/unsqueeze mismatch -- only size-1 dims differ (e.g. a
        # strided column slice ``A[0:N, i]`` of subset size ``(N, 1)`` feeding
        # a 1-D ``(N,)`` connector) -- preserves flat element order regardless
        # of the connector strides: dropping/inserting size-1 axes never
        # reorders elements (NumPy realizes such a reshape as a view).  It
        # therefore needs the reshape but must BYPASS the C-contiguity guard
        # below, which would wrongly reject the non-contiguous column strides.
        sub_nontrivial = [s for s in subset_size if not self._provably_equal(s, 1, facts)]
        conn_nontrivial = [s for s in conn_shape if not self._provably_equal(s, 1, facts)]
        if len(sub_nontrivial) == len(conn_nontrivial) and all(
                self._provably_equal(a, b, facts) for a, b in zip(sub_nontrivial, conn_nontrivial)):
            return True
        if (not self._provably_equal(prod(subset_size), prod(conn_shape), facts)
                or not self._is_c_contiguous_layout(desc.shape, desc.strides, facts)):
            raise NotImplementedError(
                f'Cannot reconcile nested SDFG connector {connector_name!r} (shape {conn_shape}, '
                f'strides {list(desc.strides)}) with the outer subset of array {memlet.data!r} '
                f'(shape {subset_size}): the mismatch is neither a representable strided view of the '
                f'outer subset nor a provably flat-order-preserving reshape.')
        return True

    def _nested_scalar_bridge_name(self, cfg: ControlFlowRegion, state, node: nodes.NestedSDFG,
                                   connector_name: str) -> str:
        return f'__dace_nested_scalar_{cfg.cfg_id}_{state.block_id}_{state.node_id(node)}_{connector_name}'

    def _nested_reshape_bridge_name(self, cfg: ControlFlowRegion, state, node: nodes.NestedSDFG,
                                    connector_name: str) -> str:
        return f'__dace_nested_reshape_{cfg.cfg_id}_{state.block_id}_{state.node_id(node)}_{connector_name}'

    def _writeback_shape(self, subset) -> list:
        """Shape of the target expression a subset renders to.

        The subset renderer (:func:`pyutils._python_slice_component`) emits a
        step-1 ``start == end`` range as a plain index, which DROPS that
        dimension from the indexed target (``A[0:N, 0]`` has shape ``(N,)``,
        not ``(N, 1)``). A writeback value must be reshaped to the kept
        dimensions only, or the assignment raises a shape mismatch.

        :param subset: The target memlet subset.
        :returns: List of sizes of the dimensions the rendered target keeps.
        """
        if not isinstance(subset, subsets.Range):
            return list(subset.size())
        return [
            size for (start, end, step), size in zip(subset.ranges, subset.size())
            if ':' in pyutils._python_slice_component(start, end, step)
        ]

    def _nested_scalar_direct_expr(self, sdfg: SDFG, data_name: str, subset) -> Optional[str]:
        outer_desc = sdfg.arrays[data_name]
        if not isinstance(outer_desc, data.Scalar):
            return None
        if self._normalize_subset(subset) is not None:
            return None
        if not self._is_scalar_buffer(data_name, outer_desc):
            return None
        return self._runtime_data_name(sdfg, data_name)

    def _is_singleton_buffer_desc(self, desc: data.Data) -> bool:
        return isinstance(desc, data.Array) and desc.total_size == 1 and len(desc.shape) <= 1

    def _nested_bridge_source_expr(self, sdfg: SDFG, memlet: mmlt.Memlet) -> str:
        outer_desc = sdfg.arrays[memlet.data]
        if isinstance(outer_desc, data.Scalar):
            return self._runtime_data_name(sdfg, memlet.data)
        return self._read_expr(sdfg, memlet)

    def _nested_bridge_target_subset(self, desc: data.Data, subset):
        if isinstance(desc, data.Scalar):
            return None
        return self._normalize_subset(subset)

    def _nested_bridge_value_expr(self, name: str, desc: data.Data) -> str:
        if isinstance(desc, data.Scalar):
            return f'{name}[()]'
        if self._is_singleton_buffer_desc(desc):
            if len(desc.shape) == 0:
                return f'{name}[()]'
            return f'{name}[0]'
        raise NotImplementedError('Nested scalar bridge currently only supports scalar values or 1D size-1 buffers.')

    def _nsdfg_runtime_symbol_names(self, node: nodes.NestedSDFG) -> list[str]:
        runtime_defined_names = _collect_runtime_defined_names(node.sdfg)
        # ``free_symbols`` (not ``used_symbols(all_symbols=False)``) is the set of
        # symbols the nested SDFG needs from outside: the latter drops symbols
        # that appear only in data shapes / memlet subsets (e.g. an outer matmul
        # dimension ``M`` used as ``_a[0:M, 0:K]``), which the Python backend
        # still emits literally and must therefore receive as parameters.
        free_symbols = set(map(str, node.sdfg.free_symbols))
        return [
            symname for symname in sorted(node.symbol_mapping.keys())
            if symname in free_symbols and symname not in node.sdfg.constants and symname not in runtime_defined_names
        ]

    def _data_expr(self, name: str, desc: data.Data, subset=None, is_write: bool = False) -> str:
        # TODO: Pass sdfg as argument instead of relying on self._current_sdfg
        runtime_name = self._runtime_data_name(self._current_sdfg, name) if hasattr(self, '_current_sdfg') else name
        return pyutils.data_access_expression(runtime_name,
                                              desc,
                                              self._normalize_subset(subset),
                                              scalar_buffer=self._is_scalar_buffer(name, desc),
                                              is_write=is_write)

    def _read_expr(self, sdfg: SDFG, memlet: mmlt.Memlet, data_name: Optional[str] = None, subset=None) -> str:
        name = data_name or memlet.data
        if name is None:
            raise NotImplementedError('Code-to-code memlets not supported in the Python backend.')
        desc = sdfg.arrays[name]
        actual_subset = self._normalize_subset(subset if subset is not None else memlet.subset)
        previous_sdfg = getattr(self, '_current_sdfg', None)
        self._current_sdfg = sdfg
        # TODO: After changing self._data_expr to take sdfg as argument, remove the need to set self._current_sdfg here
        try:
            return self._data_expr(name, desc, actual_subset)
        finally:
            self._current_sdfg = previous_sdfg

    def _write_target_expr(self, sdfg: SDFG, data_name: str, subset=None) -> str:
        desc = sdfg.arrays[data_name]
        actual_subset = self._normalize_subset(subset)
        previous_sdfg = getattr(self, '_current_sdfg', None)
        self._current_sdfg = sdfg
        # TODO: After changing self._data_expr to take sdfg as argument, remove the need to set self._current_sdfg here
        try:
            return self._data_expr(data_name, desc, actual_subset, is_write=True)
        finally:
            self._current_sdfg = previous_sdfg

    def _emit_memlet_write(self,
                           sdfg: SDFG,
                           memlet: mmlt.Memlet,
                           value_expr: str,
                           stream: PythonCodeIOStream,
                           cfg: ControlFlowRegion,
                           state_id: int,
                           subset=None,
                           target_name: Optional[str] = None,
                           target_desc: Optional[data.Data] = None) -> None:
        name = target_name or memlet.data
        if name is None:
            raise NotImplementedError('Code-to-code memlets not supported in the Python backend.')
        desc = target_desc or sdfg.arrays[name]
        if isinstance(desc, data.Stream):
            raise NotImplementedError('Streams are not supported for the Python backend.')
        if isinstance(desc, data.View):
            raise NotImplementedError('Views are not supported for the Python backend.')
        if isinstance(desc, data.Reference):
            raise NotImplementedError('References are not supported for the Python backend.')
        if subset is None and target_name is not None:
            actual_subset = None
        else:
            actual_subset = self._normalize_subset(subset if subset is not None else memlet.subset)
        target_expr = self._write_target_expr(sdfg, name, actual_subset)
        if memlet.wcr is not None:
            current_expr = self._read_expr(sdfg, memlet, name, actual_subset)
            value_expr = self.write_and_resolve_expr(memlet, current_expr, value_expr)
            stream.write(f'{target_expr} = {value_expr}', cfg, state_id)
            return
        # TODO: Inconsistent handling of copies. What happens if we have something that is not scalar or array?
        # On the other hand is that even possible? I can only think of structures but not sure if those can be memlet targets
        # And we don't support views or references or streams
        if isinstance(desc, data.Array) and actual_subset is None:
            stream.write(f'numpy.copyto({self._runtime_data_name(sdfg, name)}, {value_expr})', cfg, state_id)
            return
        stream.write(f'{target_expr} = {value_expr}', cfg, state_id)

    def _source_subset(self, memlet: mmlt.Memlet):
        subset = getattr(memlet, 'src_subset', None)
        if subset is not None:
            return subset
        return memlet.subset

    def _destination_subset(self, memlet: mmlt.Memlet, src_node: nodes.Node):
        subset = getattr(memlet, 'dst_subset', None)
        if subset is not None:
            return subset
        if isinstance(src_node, nodes.AccessNode):
            return memlet.other_subset
        return memlet.subset

    def generate_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                       function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        # TODO: What if there are more than one source nodes?
        entry_node = dfg_scope.source_nodes()[0]
        self.generate_node(sdfg, cfg, dfg_scope, state_id, entry_node, function_stream, callsite_stream)
        self._dispatcher.dispatch_subgraph(sdfg,
                                           cfg,
                                           dfg_scope,
                                           state_id,
                                           function_stream,
                                           callsite_stream,
                                           skip_entry_node=True)

    def generate_node(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, node: nodes.Node,
                      function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        try:
            generator = getattr(self, f'_generate_{type(node).__name__}')
        except AttributeError:
            if isinstance(node, nodes.LibraryNode):
                raise NodeNotExpandedError(sdfg, state_id, dfg.node_id(node))
            raise
        generator(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)
        self._generated_nodes.add(node)

    def allocate_reference(self, *args, **kwargs) -> None:
        raise NotImplementedError('References are not supported for the Python backend.')

    def declare_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, node: nodes.Node,
                      nodedesc: data.Data, global_stream: PythonCodeIOStream,
                      declaration_stream: PythonCodeIOStream) -> None:
        if isinstance(nodedesc, (data.View, data.Reference, data.Stream)) or not isinstance(node, nodes.AccessNode):
            return
        name = node.data
        if self._dispatcher.declared_arrays.has(name):
            return
        declaration_stream.write(f'{name}: {_defined_ptype_for(nodedesc)} | None = None', cfg, state_id)
        self._dispatcher.declared_arrays.add(name, _defined_type_for(nodedesc), _defined_ptype_for(nodedesc))

    def allocate_array(self,
                       sdfg: SDFG,
                       cfg: ControlFlowRegion,
                       dfg,
                       state_id: int,
                       node: nodes.Node,
                       nodedesc: data.Data,
                       global_stream: PythonCodeIOStream,
                       declaration_stream: PythonCodeIOStream,
                       allocation_stream: PythonCodeIOStream,
                       allocate_nested_data: bool = True) -> None:
        if not isinstance(node, nodes.AccessNode):
            return
        if isinstance(nodedesc, data.View):
            raise NotImplementedError('Views are not supported for the Python backend.')
        if isinstance(nodedesc, data.Reference):
            raise NotImplementedError('References are not supported for the Python backend.')
        if isinstance(nodedesc, data.Stream):
            raise NotImplementedError('Stream descriptors are not supported for the Python backend.')
        if _is_inside_cutile_scope(cfg, state_id, node):
            return

        root_name = node.data.split('.')[0]
        root_desc = sdfg.arrays[root_name]
        if not root_desc.transient:
            return

        if '.' in node.data and isinstance(root_desc, data.Structure):
            self._register_defined_name(node.data, nodedesc)
            return

        name = node.data
        is_global = root_desc.lifetime in (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.Persistent,
                                           dtypes.AllocationLifetime.External)
        if self._dispatcher.defined_vars.has(name):
            return

        if root_desc.lifetime is dtypes.AllocationLifetime.External:
            raise NotImplementedError('External memory management is not supported in the Python backend.')

        desc = update_persistent_desc(nodedesc, sdfg) if is_global else nodedesc
        # A transient must live in GPU memory (cupy) rather than host memory
        # (numpy) when either it is explicitly GPU_Global (e.g. a matmul operand
        # or an apply_gpu_transformations copy target -- true even in a
        # matmul-only kernel with no cuTile map), or the SDFG uses cuTile
        # kernels and the transient is passed between kernel launches.
        # Register-storage transients stay numpy: they are never passed to a
        # kernel directly.
        on_gpu = (isinstance(desc, data.Array) and desc.storage != dtypes.StorageType.Register
                  and (desc.storage == dtypes.StorageType.GPU_Global or _sdfg_uses_cutile(sdfg)))
        init_expr = self._default_expression(desc, on_gpu=on_gpu, setzero=node.setzero)

        if is_global:
            allocation_stream.write(f'global {name}', cfg, state_id)
            allocation_stream.write(
                f'{name} = __dace_persistent_transients.setdefault({self._persistent_key(sdfg, node.data)!r}, {init_expr})',
                cfg,
                state_id,
            )
        else:
            allocation_stream.write(f'{name} = {init_expr}', cfg, state_id)

        self._register_defined_name(name, desc, is_global=is_global)
        if isinstance(desc, data.Structure):
            self._register_structure_members(name, desc, is_global=is_global)

    def deallocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, node: nodes.Node,
                         nodedesc: data.Data, function_stream: PythonCodeIOStream,
                         callsite_stream: PythonCodeIOStream) -> None:
        if not isinstance(node, nodes.AccessNode):
            return
        if isinstance(nodedesc, (data.Scalar, data.View, data.Stream, data.Reference)):
            return
        if nodedesc.lifetime in (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.Persistent,
                                 dtypes.AllocationLifetime.External):
            return
        if '.' in node.data:
            return
        if _is_inside_cutile_scope(cfg, state_id, node):
            return
        callsite_stream.write(f'del {node.data}', cfg, state_id)

    def copy_memory(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, src_node: nodes.Node,
                    dst_node: nodes.Node, edge, function_stream: PythonCodeIOStream,
                    callsite_stream: PythonCodeIOStream) -> None:
        self._emit_copy(sdfg, cfg, state_id, src_node, dst_node, edge, callsite_stream)

    def _emit_copy(self, sdfg: SDFG, cfg: ControlFlowRegion, state_id: int, src_node: nodes.Node, dst_node: nodes.Node,
                   edge, stream: PythonCodeIOStream) -> None:
        memlet = edge.data
        if isinstance(src_node, nodes.CodeNode) and isinstance(dst_node, nodes.CodeNode):
            raise NotImplementedError('Code-to-code memlets not supported in the Python backend.')

        src_desc = src_node.desc(sdfg) if isinstance(src_node, nodes.AccessNode) else None
        dst_desc = dst_node.desc(sdfg) if isinstance(dst_node, nodes.AccessNode) else None
        if isinstance(src_desc, data.Stream) or isinstance(dst_desc, data.Stream):
            raise NotImplementedError('Streams are not supported for the Python backend.')
        if isinstance(src_desc, data.View) or isinstance(dst_desc, data.View):
            raise NotImplementedError('Views are not supported for the Python backend.')
        if isinstance(src_desc, data.Reference) or isinstance(dst_desc, data.Reference):
            raise NotImplementedError('References are not supported for the Python backend.')

        if isinstance(src_node, nodes.AccessNode):
            src_expr = self._read_expr(sdfg, memlet, src_node.data, subset=self._source_subset(memlet))
        elif isinstance(src_node, nodes.CodeNode):
            if edge.src_conn is None:
                raise NotImplementedError('Code-to-code memlets not supported in the Python backend.')
            src_expr = edge.src_conn
        else:
            raise NotImplementedError(f'Unsupported copy source node: {type(src_node).__name__}')

        if isinstance(dst_node, nodes.CodeNode):
            if edge.dst_conn is None:
                raise NotImplementedError('Code-to-code memlets not supported in the Python backend.')
            stream.write(f'{edge.dst_conn} = {src_expr}', cfg, state_id)
            return

        # Device -> scalar copies need value semantics: Python-backend Scalars
        # are host Python values regardless of their stamped storage, and
        # without ``.item()`` the scalar would be bound to a cupy 0-d array,
        # which cupy later rejects when mixed with numpy operands on the host.
        if (isinstance(src_desc, data.Array) and src_desc.storage == dtypes.StorageType.GPU_Global
                and isinstance(dst_desc, data.Scalar)):
            src_expr = f'({src_expr}).item()'

        self._emit_memlet_write(sdfg,
                                memlet,
                                src_expr,
                                stream,
                                cfg,
                                state_id,
                                subset=self._destination_subset(memlet, src_node),
                                target_name=dst_node.data,
                                target_desc=dst_desc)

    def write_and_resolve_expr(self, memlet: mmlt.Memlet, current_expr: str, new_expr: str) -> str:
        reduction = memlet.wcr
        reduction_expr = _python_expr(reduction)
        return f'({reduction_expr})({current_expr}, {new_expr})'

    def process_out_memlets(self,
                            sdfg: SDFG,
                            cfg: ControlFlowRegion,
                            state_id: int,
                            node: nodes.Node,
                            dfg,
                            dispatcher: 'TargetDispatcher',
                            result: PythonCodeIOStream,
                            locals_defined: bool,
                            function_stream: PythonCodeIOStream,
                            skip_wcr: bool = False,
                            codegen=None):
        for edge in dfg.out_edges(node):
            if skip_wcr and edge.data.wcr is not None:
                continue
            if isinstance(edge.dst, nodes.AccessNode):
                self._emit_copy(sdfg, cfg, state_id, node, edge.dst, edge, result)
            elif isinstance(edge.dst, nodes.CodeNode):
                raise NotImplementedError('Code-to-code memlets not supported in the Python backend.')

    def make_ptr_assignment(self, src_expr, src_dtype, dst_expr, dst_dtype, codegen=None):
        del src_dtype, dst_dtype, codegen
        return f'{dst_expr} = {src_expr}'

    def memlet_view_ctor(self, sdfg: SDFG, memlet: mmlt.Memlet, dtype, is_output: bool) -> str:
        del dtype, is_output
        return self._read_expr(sdfg, memlet)

    def memlet_definition(self,
                          sdfg: SDFG,
                          memlet: mmlt.Memlet,
                          output: bool,
                          local_name: str,
                          conntype=None,
                          allow_shadowing: bool = False,
                          codegen=None):
        if output:
            return f'{local_name} = None'
        return f'{local_name} = {self._read_expr(sdfg, memlet)}'

    def memlet_stream_ctor(self, sdfg: SDFG, memlet: mmlt.Memlet) -> str:
        raise NotImplementedError('Streams are not supported for the Python backend.')

    def memlet_ctor(self, sdfg: SDFG, memlet: mmlt.Memlet, dtype, is_output: bool) -> str:
        return self._read_expr(sdfg, memlet)

    def _generate_Tasklet(self,
                          sdfg: SDFG,
                          cfg: ControlFlowRegion,
                          dfg,
                          state_id: int,
                          node: nodes.Tasklet,
                          function_stream: PythonCodeIOStream,
                          callsite_stream: PythonCodeIOStream,
                          codegen=None):
        if node.code.language != dtypes.Language.Python:
            raise NotImplementedError('Python backend only supports Python tasklets.')

        init_code = codeblock_to_python(node.code_init).strip()
        if init_code:
            self._frame._initcode.write(init_code, sdfg)
        exit_code = codeblock_to_python(node.code_exit).strip()
        if exit_code:
            self._frame._exitcode.write(exit_code, sdfg)

        state_dfg = cfg.state(state_id)
        self._dispatcher.defined_vars.enter_scope(node)

        # Instrumentation: node begin (before reading inputs). The node hooks are
        # intentionally emitted INSIDE the enclosing map-loop body, so an N-trip
        # map produces N node timing events -- matching the C++ TimerProvider,
        # where the tasklet body and its timing also live inside the loop.
        if node.instrument != dtypes.InstrumentationType.No_Instrumentation:
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_node_begin(sdfg, cfg, state_dfg, node, callsite_stream, callsite_stream, function_stream)

        for edge in state_dfg.in_edges(node):
            if not edge.dst_conn:
                continue
            src_node = state_dfg.memlet_path(edge)[0].src
            if isinstance(src_node, nodes.CodeNode):
                raise NotImplementedError('Code-to-code memlets not supported in the Python backend.')
            callsite_stream.write(f'{edge.dst_conn} = {self._read_expr(sdfg, edge.data)}', cfg, state_id)
            self._dispatcher.defined_vars.add(edge.dst_conn, dispatcher_mod.DefinedType.Scalar, 'object')

        tasklet_body = _rewrite_dace_dtype_casts(codeblock_to_python(node.code).strip())

        callsite_stream.write(f'\n####### Tasklet: {node.label}\n\n', cfg, state_id)

        callsite_stream.write(tasklet_body or 'pass')

        callsite_stream.write(f'\n####### End of tasklet: {node.label}\n\n', cfg, state_id)

        for edge in state_dfg.out_edges(node):
            if edge.src_conn is None:
                continue
            dst_node = state_dfg.memlet_path(edge)[-1].dst
            if isinstance(dst_node, nodes.CodeNode):
                raise NotImplementedError('Code-to-code memlets not supported in the Python backend.')
            # TODO: Involve self._dispatcher.dispatch_copy instead?
            self._emit_memlet_write(sdfg, edge.data, edge.src_conn, callsite_stream, cfg, state_id)

        # Instrumentation: node end (after writing outputs)
        if node.instrument != dtypes.InstrumentationType.No_Instrumentation:
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_node_end(sdfg, cfg, state_dfg, node, callsite_stream, callsite_stream, function_stream)

        self._dispatcher.defined_vars.exit_scope(node)

    def unparse_tasklet(self, sdfg, cfg, state_id, dfg, node, function_stream, inner_stream, locals, ldepth,
                        toplevel_schedule):
        if node.code.language != dtypes.Language.Python:
            raise NotImplementedError('Python backend only supports Python tasklets.')
        inner_stream.write(codeblock_to_python(node.code).strip() or 'pass')

    def define_out_memlet(self, sdfg: SDFG, cfg: ControlFlowRegion, state_dfg, state_id: int, src_node: nodes.Node,
                          dst_node: nodes.Node, edge, function_stream: PythonCodeIOStream,
                          callsite_stream: PythonCodeIOStream) -> None:
        # TODO: Is this correct?
        pass

    def generate_nsdfg_header(self, sdfg, cfg, state, state_id, node, memlet_references, sdfg_label, state_struct=True):
        arguments = [aname for _, aname, _ in memlet_references]
        arguments.extend(self._nsdfg_runtime_symbol_names(node))
        return f'def {sdfg_label}({", ".join(arguments)}):'

    def generate_nsdfg_call(self, sdfg, cfg, state, node, memlet_references, sdfg_label, state_struct=True):
        # TODO: state struct?
        args = [argval for _, _, argval in memlet_references]
        args.extend(_python_expr(node.symbol_mapping[symname]) for symname in self._nsdfg_runtime_symbol_names(node))
        return f'{sdfg_label}({", ".join(args)})'

    def _prepare_nsdfg_arguments(self, sdfg: SDFG, cfg: ControlFlowRegion, state, node: nodes.NestedSDFG):
        references = []
        pre_call_statements = []
        post_call_actions = []
        bindings = {}
        initialized_bridges = set()
        initialized_reshape_bridges = set()
        registered_reshape_writebacks = set()
        seen_inputs = set()
        seen_outputs = set()
        output_connector_names = {e.src_conn for e in state.out_edges(node) if e.src_conn is not None}
        facts = self._frame._symbol_resolver.facts_at(state, node)

        def _bind_bridge(connector_name: str, memlet: mmlt.Memlet, desc: data.Data, is_input: bool) -> str:
            bridge_name = self._nested_scalar_bridge_name(cfg, state, node, connector_name)
            if bridge_name not in initialized_bridges:
                pre_call_statements.append(self._nested_buffer_initialization(bridge_name, desc))
                initialized_bridges.add(bridge_name)
            if is_input:
                pre_call_statements.append(f'{bridge_name}[...] = {self._nested_bridge_source_expr(sdfg, memlet)}')
            else:
                target_desc = sdfg.arrays[memlet.data]
                post_call_actions.append({
                    'memlet': memlet,
                    'target_name': memlet.data,
                    'target_desc': target_desc,
                    'subset': self._nested_bridge_target_subset(target_desc, memlet.subset),
                    'value_expr': self._nested_bridge_value_expr(bridge_name, desc),
                })
            return bridge_name

        def _bind_reshape_bridge(connector_name: str, memlet: mmlt.Memlet, desc: data.Array, view_expr: str,
                                 is_input: bool) -> str:
            """Bind a flat-reshape-mismatched OUTPUT (or in/out) connector via a
            contiguous temporary, copied back after the call.

            A plain ``.reshape`` of a non-contiguous outer slice would silently
            return a copy, dropping the nested SDFG's writes. The bridge is a
            contiguous copy in the connector's shape; after the call it is
            reshaped back to the outer subset's shape and written into the
            outer array explicitly.
            """
            bridge_name = self._nested_reshape_bridge_name(cfg, state, node, connector_name)
            if bridge_name not in initialized_reshape_bridges:
                pre_call_statements.append(
                    f'{bridge_name} = ({view_expr}).copy().reshape({self._shape_expression(desc.shape)})')
                initialized_reshape_bridges.add(bridge_name)
            if not is_input and bridge_name not in registered_reshape_writebacks:
                target_desc = sdfg.arrays[memlet.data]
                outer_shape = self._writeback_shape(memlet.subset)
                post_call_actions.append({
                    'memlet': memlet,
                    'target_name': memlet.data,
                    'target_desc': target_desc,
                    'subset': self._normalize_subset(memlet.subset),
                    'value_expr': f'{bridge_name}.reshape({self._shape_expression(outer_shape)})',
                })
                registered_reshape_writebacks.add(bridge_name)
            return bridge_name

        def _register_reference(connector_name: str, memlet: mmlt.Memlet, is_input: bool) -> None:
            if connector_name is None or memlet.data is None:
                return

            desc = node.sdfg.arrays[connector_name]
            if isinstance(desc, data.Array):
                desc = copy.deepcopy(desc)
                for field in ('shape', 'strides', 'offset'):
                    setattr(desc, field, tuple(symbolic.replace_symbols(value, node.symbol_mapping)
                                              for value in getattr(desc, field)))
                desc.total_size = symbolic.replace_symbols(desc.total_size, node.symbol_mapping)
            outer_desc = sdfg.arrays[memlet.data]
            if isinstance(desc, data.Scalar):
                arg_expr = self._nested_scalar_direct_expr(sdfg, memlet.data, memlet.subset)
                if arg_expr is None:
                    arg_expr = _bind_bridge(connector_name, memlet, desc, is_input)
            elif self._is_singleton_buffer_desc(desc) and isinstance(outer_desc, data.Scalar):
                arg_expr = _bind_bridge(connector_name, memlet, desc, is_input)
            elif isinstance(desc, data.Array):
                # A connector declared over the whole outer array takes the array
                # itself (inner subscripts are outer-origin coordinates), whether
                # the subset is the explicit full array or a zero-based accessed
                # window. Consulted BEFORE the shapes-match shortcut so an in/out
                # connector with an explicit-full IN subset and a window OUT
                # subset canonicalizes to the same (bare-name) binding instead of
                # raising the inconsistent-binding error.
                if self._nested_desc_spans_full_outer(memlet, desc, outer_desc, facts):
                    arg_expr = self._runtime_data_name(sdfg, memlet.data)
                else:
                    arg_expr = self._nested_view_expr(sdfg, memlet.data, memlet.subset)
                    if not self._nested_shapes_match(memlet, desc, facts):
                        # Otherwise prefer a genuine strided view (exact layout,
                        # valid for reads and writes); otherwise fall back to a flat
                        # reshape, which raises when not provably order-preserving.
                        if (strided_expr := self._nested_strided_view_expr(sdfg, memlet, desc, facts)) is not None:
                            arg_expr = strided_expr
                        elif self._nested_arg_needs_flat_reshape(connector_name, memlet, desc, facts):
                            if connector_name in output_connector_names:
                                # Writes must land in the outer array: go through a
                                # contiguous bridge with an explicit copy-back.
                                arg_expr = _bind_reshape_bridge(connector_name, memlet, desc, arg_expr, is_input)
                            else:
                                # Read-only: a flat reshape view (or copy) suffices.
                                arg_expr = f'({arg_expr}).reshape({self._shape_expression(desc.shape)})'
            else:
                arg_expr = self._runtime_data_name(sdfg, memlet.data)

            existing_expr = bindings.get(connector_name)
            if existing_expr is not None and existing_expr != arg_expr:
                raise NotImplementedError(
                    f'Nested SDFG connector {connector_name!r} is bound inconsistently in the Python backend.')
            if existing_expr is None:
                bindings[connector_name] = arg_expr
                references.append((_defined_ptype_for(desc), connector_name, arg_expr))

        for _, _, _, dst_conn, in_memlet in sorted(state.in_edges(node), key=lambda e: e.dst_conn or ''):
            if dst_conn in seen_inputs:
                raise NotImplementedError(
                    f'Nested SDFG input connector {dst_conn!r} has multiple bindings in the Python backend.')
            if dst_conn is not None and in_memlet.data is not None:
                seen_inputs.add(dst_conn)
            _register_reference(dst_conn, in_memlet, True)

        for _, src_conn, _, _, out_memlet in sorted(state.out_edges(node), key=lambda e: e.src_conn or ''):
            if src_conn in seen_outputs:
                raise NotImplementedError(
                    f'Nested SDFG output connector {src_conn!r} has multiple bindings in the Python backend.')
            if src_conn is not None and out_memlet.data is not None:
                seen_outputs.add(src_conn)
            _register_reference(src_conn, out_memlet, False)

        return references, pre_call_statements, post_call_actions

    def generate_nsdfg_arguments(self, sdfg, cfg, dfg, state, node):
        references, _, _ = self._prepare_nsdfg_arguments(sdfg, cfg, state, node)
        return references

    def _nested_buffer_initialization(self, name: str, desc: data.Data) -> str:
        if isinstance(desc, data.Scalar):
            return f'{name} = numpy.zeros((), dtype={_numpy_dtype(desc.dtype)})'
        if self._is_singleton_buffer_desc(desc):
            return f'{name} = numpy.zeros({self._shape_expression(desc.shape)}, dtype={_numpy_dtype(desc.dtype)})'
        raise NotImplementedError('Nested scalar bridge currently only supports scalar values or 1D size-1 buffers.')

    def _generate_NestedSDFG(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: ScopeSubgraphView, state_id: int,
                             node: nodes.NestedSDFG, function_stream: PythonCodeIOStream,
                             callsite_stream: PythonCodeIOStream):
        state = cfg.state(state_id)
        self._dispatcher.defined_vars.enter_scope(node.sdfg, can_access_parent=False)
        self._dispatcher.declared_arrays.enter_scope(node.sdfg, can_access_parent=False)

        # We do not support function inlining for nested SDFGs
        if Config.get_bool('compiler', 'inline_sdfgs'):
            warnings.warn(
                'Function inlining for nested SDFGs is not supported in the Python backend. Ignoring inline_sdfgs=True.'
            )

        fsyms = self._frame.free_symbols(node.sdfg)
        self._define_sdfg_arguments(node.sdfg, node.sdfg.arglist(scalars_only=False, free_symbols=fsyms))

        # TODO: There are more options for the key here such as 'unique_name' or False or 'none'
        if Config.get('compiler', 'unique_functions') in (True, 'hash'):
            nested_key = str(node.sdfg.hash_sdfg())
        else:
            nested_key = f'{cfg.cfg_id}:{state_id}:{state.node_id(node)}'
        is_function_already_defined = nested_key in self._generated_nested_sdfg
        sdfg_label = self._generated_nested_sdfg.setdefault(
            nested_key, f'{node.sdfg.name}_{cfg.cfg_id}_{state_id}_{state.node_id(node)}')

        memlet_references, pre_call_statements, post_call_actions = self._prepare_nsdfg_arguments(
            sdfg, cfg, state, node)
        runtime_symbol_names = self._nsdfg_runtime_symbol_names(node)

        if not is_function_already_defined:
            preamble_stream = PythonCodeIOStream()
            finalizer_stream = PythonCodeIOStream()
            self._frame.generate_embedded_function_preamble(node.sdfg, preamble_stream)
            self._frame.generate_embedded_function_finalizer(node.sdfg, finalizer_stream)
            old_schedule = self._toplevel_schedule
            old_arglist = self._frame.arglist
            runtime_defined_names = _collect_runtime_defined_names(node.sdfg)
            nested_runtime_defined_names = _collect_nested_runtime_defined_names(node.sdfg)
            nested_only_runtime_names = {name for name in nested_runtime_defined_names if name not in node.sdfg.symbols}
            runtime_symbol_names = {
                name
                for name in _collect_runtime_used_names(node.sdfg) if name in node.sdfg.symbols
            }
            # Include the nested SDFG's ``free_symbols`` (shape/subset symbols
            # such as an outer matmul dimension ``M``) in addition to
            # ``used_symbols(all_symbols=False)``: the Python backend emits data
            # subsets literally, so those symbols must appear in the nested
            # function's arglist to match ``_nsdfg_runtime_symbol_names``.
            nested_free_symbols = ((self._frame.free_symbols(node.sdfg) | runtime_symbol_names
                                    | set(map(str, node.sdfg.free_symbols))) - runtime_defined_names -
                                   nested_only_runtime_names)
            nested_arglist = node.sdfg.arglist(scalars_only=False, free_symbols=nested_free_symbols)
            ordered_arglist = {name: nested_arglist[name] for _, name, _ in memlet_references}
            for symname in self._nsdfg_runtime_symbol_names(node):
                ordered_arglist[symname] = nested_arglist[symname]
            self._frame.arglist = ordered_arglist
            try:
                global_code, local_code, _, used_environments = self._frame.generate_code(
                    node.sdfg,
                    old_schedule,
                    function_name=sdfg_label,
                    include_lifecycle=False,
                    include_file_header=False,
                    function_body_preamble=preamble_stream.getvalue(),
                    function_body_finally=finalizer_stream.getvalue(),
                )
            finally:
                self._frame.arglist = old_arglist
            self._dispatcher._used_environments |= used_environments
            function_stream.write(global_code)
            function_stream.write(local_code)

        for statement in pre_call_statements:
            callsite_stream.write(statement, cfg, state_id)
        callsite_stream.write(self.generate_nsdfg_call(sdfg, cfg, state, node, memlet_references, sdfg_label), cfg,
                              state_id)
        for action in post_call_actions:
            self._emit_memlet_write(sdfg,
                                    action['memlet'],
                                    action['value_expr'],
                                    callsite_stream,
                                    cfg,
                                    state_id,
                                    subset=action['subset'],
                                    target_name=action['target_name'],
                                    target_desc=action['target_desc'])

        self._dispatcher.declared_arrays.exit_scope(node.sdfg)
        self._dispatcher.defined_vars.exit_scope(node.sdfg)

    def _map_range_statement(self, variable: str, begin: str, end: str, step: str) -> str:
        if step == '1':
            return f'for {variable} in range({begin}, ({end}) + 1):'
        return f'for {variable} in range({begin}, ({end}) + (1 if ({step}) > 0 else -1), {step}):'

    def _generate_MapEntry(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, node: nodes.MapEntry,
                           function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream):
        schedule = node.map.schedule
        if schedule == dtypes.ScheduleType.Default:
            schedule = dtypes.ScheduleType.Sequential
        # Technically we don't support other schedules, but let's just treat them all as sequential for now
        # if schedule != dtypes.ScheduleType.Sequential:
        #     raise NotImplementedError('Python backend only supports sequential maps.')
        if node.map.unroll:
            raise NotImplementedError('Map unrolling is not supported in the Python backend.')

        state_dfg = cfg.state(state_id)
        for edge in dynamic_map_inputs(state_dfg, node):
            if edge.dst_conn is None:
                continue
            callsite_stream.write(f'{edge.dst_conn} = {self._read_expr(sdfg, edge.data)}', cfg, state_id)

        # Instrumentation: Pre-scope (before for-loop headers)
        if node.map.instrument != dtypes.InstrumentationType.No_Instrumentation:
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_scope_entry(sdfg, cfg, state_dfg, node, callsite_stream, callsite_stream, function_stream)

        for current_range, variable in zip(node.map.range, node.map.params):
            begin, end, step = current_range
            callsite_stream.write(
                self._map_range_statement(str(variable), _python_expr(begin), _python_expr(end), _python_expr(step)),
                cfg, state_id)
            callsite_stream.indent()

        if hasattr(self._frame, 'allocate_arrays_in_scope'):
            self._frame.allocate_arrays_in_scope(sdfg, cfg, node, PythonCodeIOStream(), callsite_stream)
        if len(dfg.scope_children().get(node, [])) == 0:
            callsite_stream.write('pass', cfg, state_id)

    def _generate_MapExit(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, node: nodes.MapExit,
                          function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        map_node = dfg.scope_dict()[node]
        if map_node is None:
            raise NotImplementedError('MapExit generation requires an enclosing MapEntry scope.')
        if hasattr(self._frame, 'deallocate_arrays_in_scope'):
            self._frame.deallocate_arrays_in_scope(sdfg, cfg, map_node, function_stream, callsite_stream)
        callsite_stream.dedent(len(map_node.map.range))

        # Instrumentation: Post-scope (after all for-loop iterations complete)
        if map_node.map.instrument != dtypes.InstrumentationType.No_Instrumentation:
            state_dfg = cfg.state(state_id)
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_scope_exit(sdfg, cfg, state_dfg, node, callsite_stream, callsite_stream, function_stream)

    def _generate_ConsumeEntry(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, node: nodes.ConsumeEntry,
                               function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        raise NotImplementedError('Consume scopes are not supported in the Python backend.')

    def _generate_ConsumeExit(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, node: nodes.ConsumeExit,
                              function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        raise NotImplementedError('Consume scopes are not supported in the Python backend.')

    def _generate_AccessNode(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, node: nodes.Node,
                             function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        if not isinstance(node, nodes.AccessNode):
            return
        state_dfg = cfg.state(state_id)
        scope_dict = state_dfg.scope_dict()
        for edge in state_dfg.in_edges(node):
            memlet = edge.data
            if memlet.data is None:
                continue
            memlet_path = state_dfg.memlet_path(edge)
            if memlet_path[-1].dst != node:
                continue
            src_node = memlet_path[0].src
            if isinstance(src_node, nodes.CodeNode):
                continue
            nested_scope = scope_contains_scope(scope_dict, src_node, node) and scope_dict[src_node] != scope_dict[node]
            if nested_scope:
                self._dispatcher.dispatch_copy(src_node, node, edge, sdfg, cfg, state_dfg, state_id,
                                               PythonCodeIOStream(), callsite_stream)

        for edge in state_dfg.out_edges(node):
            memlet = edge.data
            if memlet.data is None:
                continue
            memlet_path = state_dfg.memlet_path(edge)
            dst_node = memlet_path[-1].dst
            if isinstance(dst_node, nodes.CodeNode) or dst_node == node:
                continue
            if not isinstance(dst_node, nodes.AccessNode):
                continue
            if scope_dict[node] != scope_dict[dst_node] and scope_contains_scope(scope_dict, node, dst_node):
                continue
            self._dispatcher.dispatch_copy(node, dst_node, edge, sdfg, cfg, state_dfg, state_id, PythonCodeIOStream(),
                                           callsite_stream)

    # TODO: docstrings
    # TODO: Incorporate methods into code so they can be overwritten by subclasses instead of having
    # to overwrite entire node generation methods

    def generate_scope_preamble(self, sdfg, dfg_scope, state_id, function_stream, outer_stream, inner_stream):
        pass

    def generate_scope_postamble(self, sdfg, dfg_scope, state_id, function_stream, outer_stream, inner_stream):
        pass

    def generate_tasklet_preamble(self, sdfg, cfg, dfg_scope, state_id, node, function_stream, before_memlets_stream,
                                  after_memlets_stream):
        pass

    def generate_tasklet_postamble(self, sdfg, cfg, dfg_scope, state_id, node, function_stream, before_memlets_stream,
                                   after_memlets_stream):
        pass

    def emit_interstate_variable_declaration(self, name: str, dtype: dtypes.typeclass,
                                             callsite_stream: PythonCodeIOStream, sdfg: SDFG):
        pass
