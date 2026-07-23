# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from copy import deepcopy
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import AbstractControlFlowRegion, ControlFlowRegion, LoopRegion, SDFGState, StateSubgraphView
import functools
import itertools
import warnings

import numpy as np

from dace import data, dtypes, registry, memlet as mmlt, subsets, symbolic, Config
from dace.codegen import compiler_family, cppunparse, exceptions as cgx
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets import cpp
from dace.codegen.common import codeblock_to_cpp, sym2cpp, update_persistent_desc
from dace.codegen.target import TargetCodeGenerator, make_absolute
from dace.codegen.dispatcher import DefinedType, TargetDispatcher
from dace.frontend import operations
from dace.sdfg import nodes, utils as sdutils
from dace.sdfg import (ScopeSubgraphView, SDFG, scope_contains_scope, is_array_stream_view, NodeNotExpandedError,
                       dynamic_map_inputs)
from dace.sdfg.scope import is_devicelevel_gpu, is_in_scope
from dace.sdfg.validation import validate_memlet_data
from dace.transformation.passes.analysis.loop_analysis import counter_used_outside_loop
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple, Union

import re

if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator

#: C++ spelling of each ``codegen_params.loop_index_type`` value. ``auto`` deduces from the lower
#: bound (for the usual ``0`` that is ``int``); the others state the width outright, as exact-width
#: ``<cstdint>`` types -- ``long long`` is only guaranteed to be AT LEAST 64 bits, so it does not
#: state the width the key names.
LOOP_INDEX_CTYPES = {'auto': 'auto', 'int64': 'int64_t', 'int32': 'int32_t'}


def loop_index_ctype() -> str:
    """Declared type of a map loop's induction variable, per ``codegen_params.loop_index_type``."""
    return LOOP_INDEX_CTYPES[Config.get('compiler', 'cpu', 'codegen_params', 'loop_index_type')]


def loop_region_index_ctype() -> Optional[str]:
    """Declared-type override for a sequential ``LoopRegion`` counter, per
    ``codegen_params.loop_index_type``. ``auto`` (the default) returns ``None`` -- keep the counter's
    inferred type, so the emitted declaration is byte-for-byte what it was before the knob existed.
    ``int32`` / ``int64`` return ``int32_t`` / ``int64_t`` (the same spellings the map-loop emitter uses
    via ``LOOP_INDEX_CTYPES``), applied to the hoisted declaration a LoopRegion counter is given ahead
    of its loop. A LoopRegion is inherently sequential, so there is no OpenMP gate here."""
    ctype = LOOP_INDEX_CTYPES[Config.get('compiler', 'cpu', 'codegen_params', 'loop_index_type')]
    return None if ctype == 'auto' else ctype


def is_loop_region_variable(name: str, sdfg: SDFG) -> bool:
    """Whether ``name`` is the loop counter of some ``LoopRegion`` in ``sdfg``. Only such a counter's
    declaration is retyped by ``loop_index_type``; every other interstate symbol keeps its inferred
    type (retyping an arbitrary interstate assignment target would change semantics, not spelling)."""
    return any(isinstance(cfr, LoopRegion) and cfr.loop_variable == name for cfr in sdfg.all_control_flow_regions())


def decl_placement() -> str:
    """Where a declaration is emitted relative to its first use, per ``codegen_params.decl_placement``:
    ``eager`` (the default -- every declaration at the top of its scope, the legacy placement) or
    ``late`` (each declaration moved as close to its first use as it is provably sound to move it)."""
    return Config.get('compiler', 'cpu', 'codegen_params', 'decl_placement')


def scalar_init_style() -> str:
    """Whether a mutable scalar's declaration and its first write are emitted as one binding, per
    ``codegen_params.scalar_init_style``: ``split`` (the default -- ``T x;`` then ``x = expr;``) or
    ``fused`` (``T x = expr;``, the declaration IS the first write)."""
    return Config.get('compiler', 'cpu', 'codegen_params', 'scalar_init_style')


def counter_init_assigns_only(loop: LoopRegion) -> bool:
    """Whether ``loop``'s init statement is exactly ``<loop_variable> = <expr>``, the one shape a
    declared-in-place counter can be spelled as (``T i = <expr>``). Any other init (a tuple target, a
    compound statement, an augmented assignment, or one that initialises a DIFFERENT name) has no such
    spelling, so the counter keeps its hoisted declaration."""
    code = loop.init_statement.code
    if not isinstance(code, list) or len(code) != 1:
        return False
    stmt = code[0]
    return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == loop.loop_variable)


def loop_local_counter_ctype(name: str, dtype: dtypes.typeclass, sdfg: SDFG) -> Optional[str]:
    """The C++ type to declare interstate symbol ``name`` with INSIDE its own loop's ``for``-init clause
    (``for (int64_t i = 0; ...)``), or ``None`` to keep the hoisted ``int64_t i;`` declaration that
    every LoopRegion counter gets today.

    ``late`` only: this is the ``decl_placement`` knob applied to a loop counter, whose hoisted
    declaration is the single furthest-from-use declaration the generator emits -- it sits at the top of
    the function no matter how deep the loop is. Moving it into the init clause is sound only when the
    counter is genuinely loop-local:

    - exactly ONE LoopRegion owns the name. Two loops sharing a counter share the one hoisted
      declaration; giving each its own is a bigger change than a placement knob should make.
    - the loop is not ``inverted``. An inverted loop emits its init BEFORE the ``do``/``while`` brace,
      so a declaration there is not scoped to the loop and could collide with a sibling.
    - the init statement is a plain ``i = <expr>`` (see :func:`counter_init_assigns_only`).
    - nothing outside the loop uses the counter (see :func:`counter_used_outside_loop`).

    Applies to both generators (the loop emitter is shared). ``eager`` returns ``None`` throughout, so
    output stays byte-for-byte what it is today.
    """
    if decl_placement() != 'late':
        return None
    owners = [
        cfr for cfr in sdfg.all_control_flow_regions()
        if isinstance(cfr, LoopRegion) and cfr.loop_variable == name and cfr.init_statement is not None
    ]
    if len(owners) != 1:
        return None
    loop = owners[0]
    if loop.inverted or not counter_init_assigns_only(loop) or counter_used_outside_loop(name, loop, sdfg):
        return None
    return loop_region_index_ctype() or dtype.ctype


def map_schedule_is_sequential(node: nodes.MapEntry) -> bool:
    """Whether this map emits a plain sequential loop rather than an OpenMP ``parallel for``. The gate
    shared by every knob that rewrites the loop into a non-canonical form (hoisted declaration, walking
    pointer): an OpenMP pragma must be immediately followed by a CANONICAL indexed loop, and a
    loop-carried rewrite is exactly what a ``parallel for`` forbids, so those knobs apply to sequential
    (non-``CPU_Multicore`` / non-``CPU_Persistent``) maps only."""
    return node.map.schedule not in (dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.CPU_Persistent)


def hoist_loop_decls(node: nodes.MapEntry) -> bool:
    """Whether this map's induction variables are declared ahead of their loops (``T i = begin; for (;
    ...)``) instead of in the for-statement's init clause, per ``codegen_params.loop_decl_style``.

    Never for an OpenMP-scheduled map: the pragma must be immediately followed by a CANONICAL loop
    whose init clause declares the induction variable, so hoisting leaves the pragma facing a
    declaration and the compiler rejects it ("loop nest expected"). The knob therefore applies to
    sequential maps only.
    """
    if Config.get('compiler', 'cpu', 'codegen_params', 'loop_decl_style') != 'hoisted':
        return False
    return map_schedule_is_sequential(node)


def loop_exit_test(begin, end, skip, node: nodes.MapEntry) -> Tuple[str, str]:
    """The ``(comparison, bound)`` of a map loop's exit test, per ``codegen_params.loop_bound_cmp``.

    Every spelling covers the identical iteration space ``[begin, end]`` at stride ``skip``.

    ``ne`` supports any stride on a SEQUENTIAL loop. A naive ``i != end + 1`` is only correct when the
    stride divides the range -- otherwise the counter steps OVER that bound, never compares equal, and
    the loop does not terminate. So for a non-unit stride the bound is normalised to the first value
    the counter actually LANDS on at or past the end, ``begin + int_ceil(end + 1 - begin, skip) *
    skip``, which the induction variable is guaranteed to hit exactly.

    On an OpenMP-scheduled map, ``ne`` is legal ONLY with a stride the compiler can see is +/-1: the
    canonical loop form the pragma requires rejects ``!=`` otherwise (``g++``: "increment is not
    constant 1 or -1 for '!=' condition"). A non-unit / symbolic stride there falls back to ``<``.
    """
    mode = Config.get('compiler', 'cpu', 'codegen_params', 'loop_bound_cmp')
    if mode == 'le':
        return '<=', sym2cpp(end)
    if mode == 'ne':
        if symbolic.pystr_to_symbolic(skip) == 1:
            return '!=', sym2cpp(end + 1)
        openmp = node.map.schedule in (dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.CPU_Persistent)
        if not openmp:
            return '!=', sym2cpp(begin + symbolic.int_ceil(end + 1 - begin, skip) * skip)
    return '<', sym2cpp(end + 1)


def gpu_block_reduction_write_slot(subset, base, length):
    """Register-partial slot for a write to a GPU thread-block tree-reduction accumulator.

    Returns the index into the per-thread register partial (``offset - base``) for a write the
    block fold can absorb, or ``None`` if it cannot -- in which case the caller keeps the plain
    atomic WCR. Only a single 1-D element inside the reduced span ``[base, base + length)`` is
    foldable; a multi-dimensional subset, a missing subset, or a constant offset outside the span
    (e.g. from a second reduction edge over the same array) falls back to the atomic.

    :param subset: the write memlet's subset.
    :param base: the reduced range base recorded when the accumulator was covered.
    :param length: the reduced span length ``m`` (the register partial has this many slots).
    :return: the (possibly symbolic) slot expression, or ``None`` to keep the atomic path.
    """
    if subset is None or len(subset.ranges) != 1:
        return None
    offset = subset.ranges[0][0] - base
    if not symbolic.issymbolic(offset):
        if int(offset) < 0 or int(offset) >= length:
            return None
    return offset


def collect_gpu_block_reductions(sdfg: SDFG, state: SDFGState, scope_entry: nodes.MapEntry, block_dims, frame) -> list:
    """Scalar/array map-exit WCR accumulators under ``scope_entry`` that fold via ``cub::BlockReduce``
    + one atomic per block -- the GPU mirror of an OpenMP ``reduction(op:var)`` clause. Shared by both
    the legacy and the experimental CUDA code generators so the two emit one tree reduction.

    Each thread accumulates into a private register partial and cub folds the partials, so thread 0
    commits a single atomic (versus one contended atomic per thread). A scalar accumulator is one slot
    (``m == 1``); a length-``m`` reduced subset is folded element-wise by the drain loop.

    The guard is deliberately narrow: a single, loop-invariant, compile-time-length 1-D span (a
    param-dependent subset is a scatter, not a reduction), a scalar (non-vector) built-in op with a
    known identity element, and compile-time-constant block dimensions (cub templates on them). Anything
    else keeps the per-thread atomic fallback.

    :return: one field dict per qualifying accumulator, consumed by :func:`register_gpu_block_reduction`
             and :func:`drain_gpu_block_reduction`.
    """
    out: list = []
    try:
        map_exit = state.exit_node(scope_entry)
    except (KeyError, StopIteration):
        return out
    # cub::BlockReduce templates on the block dimensions, which must be compile-time constants.
    if any(symbolic.issymbolic(b, sdfg.constants) for b in block_dims):
        return out
    # cub reduces over the whole block and must be told each dimension; the 1-D BlockReduce<T, N> form
    # mis-maps threads whenever the block is 2-D/3-D (it assumes threadIdx.y == threadIdx.z == 0).
    block_x, block_y, block_z = (int(block_dims[0]), int(block_dims[1]), int(block_dims[2]))
    map_params = set(scope_entry.map.params)
    seen_targets: Set[str] = set()
    for i, iedge in enumerate(state.in_edges(map_exit)):
        if iedge.data is None or iedge.data.wcr is None:
            continue
        acc_desc = sdfg.arrays.get(iedge.data.data)
        subset = iedge.data.subset
        if acc_desc is None or iedge.data.data in seen_targets or subset is None:
            continue
        # cub, the identity literal and ``_wcr_fixed::reduce_atomic`` all want a scalar ctype.
        if isinstance(acc_desc.dtype, dtypes.vector):
            continue
        if len(subset) != 1 or subset.ranges[0][2] != 1:
            continue
        base, end, _ = subset.ranges[0]
        try:
            m = int(end - base) + 1
        except (TypeError, ValueError):
            continue  # symbolic length: cannot size the register partial / drain loop
        # Loop-invariant target: not indexed by this map's iteration variables.
        if any(str(s) in map_params for s in subset.free_symbols):
            continue
        redtype = operations.detect_reduction_type(iedge.data.wcr)
        identity = dtypes.reduction_identity(acc_desc.dtype, redtype)
        if redtype == dtypes.ReductionType.Custom or identity is None:
            continue  # only built-in ops with a known identity element
        seen_targets.add(iedge.data.data)
        ctype = acc_desc.dtype.ctype
        partial = f'__bpart_{state.block_id}_{state.node_id(scope_entry)}_{i}'
        # Emit integers as integers: routing a 64-bit extreme (e.g. a Min identity of INT64_MAX)
        # through ``float`` rounds to 2**63 and overflows the cast.
        if np.issubdtype(acc_desc.dtype.type, np.integer):
            identity_literal = f'{ctype}({int(identity)})'
        else:
            identity_literal = f'{ctype}({float(identity)!r})'
        out.append({
            'acc_ptr': cpp.ptr(iedge.data.data, acc_desc, sdfg, frame),
            'partial': partial,
            'ctype': ctype,
            'credtype': 'dace::ReductionType::' + str(redtype).split('.')[-1],
            'identity': identity_literal,
            'block_x': block_x,
            'block_y': block_y,
            'block_z': block_z,
            'm': m,
            'base': base,
            'data': iedge.data.data,
        })
    return out


def register_gpu_block_reduction(red: dict, covered: dict) -> str:
    """Declare the per-thread register partial and identity-init it, and mark the accumulator
    ``covered`` so :meth:`CPUCodeGen.write_and_resolve_expr` redirects its per-thread WCR writes into
    the partial instead of emitting an atomic. Emit the returned C before the bounds guard so
    out-of-range threads still carry the identity into the barrier fold. Caveman: make partial, mark it.
    """
    covered[red['data']] = {
        'partial': red['partial'],
        'credtype': red['credtype'],
        'ctype': red['ctype'],
        'base': red['base'],
        'm': red['m'],
    }
    return (f"{red['ctype']} {red['partial']}[{red['m']}];\n"
            f"for (int __bi = 0; __bi < {red['m']}; ++__bi) {red['partial']}[__bi] = {red['identity']};")


def drain_gpu_block_reduction(red: dict, idstr: str, covered: dict) -> str:
    """For each of the ``m`` reduced elements, ``cub::BlockReduce`` over each thread's register partial,
    then one ``reduce_atomic`` from thread 0 into that accumulator element; then un-cover it. Emit the
    returned C after the bounds guard closes so every thread reaches the barrier-using cub call; the
    ``__syncthreads`` between iterations lets the single shared ``TempStorage`` be reused. Caveman: fold
    block, one atomic.
    """
    covered.pop(red['data'], None)
    functor = 'dace::_wcr_fixed<{credtype}, {ctype}>'.format(**red)
    base_cpp = sym2cpp(red['base'])
    return ('{{\n'
            'typedef cub::BlockReduce<{ctype}, {block_x}, cub::BLOCK_REDUCE_WARP_REDUCTIONS, {block_y}, {block_z}> '
            '__brt_{id};\n'
            '__shared__ typename __brt_{id}::TempStorage __brs_{id};\n'
            'for (int __bk_{id} = 0; __bk_{id} < {m}; ++__bk_{id}) {{\n'
            '    {ctype} __bres_{id} = __brt_{id}(__brs_{id}).Reduce({partial}[__bk_{id}], {functor}());\n'
            '    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {{\n'
            '        {functor}::reduce_atomic({acc_ptr} + (({base_cpp}) + __bk_{id}), __bres_{id});\n'
            '    }}\n'
            '    __syncthreads();\n'
            '}}\n'
            '}}'.format(id=idstr, functor=functor, base_cpp=base_cpp, **red))


def replace_float_literals(expr: str) -> str:
    """
    Replace floating-point literals like 1.0, 2.000, 3., .5 with integer literals.
    Keeps integers untouched.
    """
    float_literal = re.compile(
        r"""
        (?<![\w.])           # no letter before (avoid matching struct.field)
        (                    # start capture group
            \d+\.\d*         # 1.0, 2.   , 3.
        | \d*\.\d+         # .5, 0.25
        )
        (?![\w.])            # no letter after
        """, re.VERBOSE)

    def convert(m):
        val = float(m.group(0))
        return str(int(val))

    return float_literal.sub(convert, expr)


#: ReductionType -> OpenMP ``reduction(op:var)`` clause operator. Only the
#: ops OpenMP's reduction clause natively supports are listed. ``Custom`` and
#: ``Sub`` / ``Div`` fall through to the atomic emission path.
_REDUCTION_TO_OMP_OP = {
    dtypes.ReductionType.Sum: "+",
    dtypes.ReductionType.Product: "*",
    dtypes.ReductionType.Min: "min",
    dtypes.ReductionType.Max: "max",
    dtypes.ReductionType.Logical_And: "&&",
    dtypes.ReductionType.Logical_Or: "||",
    dtypes.ReductionType.Bitwise_And: "&",
    dtypes.ReductionType.Bitwise_Or: "|",
    dtypes.ReductionType.Bitwise_Xor: "^",
}

#: Complex element types an OpenMP array/scalar reduction can target -- only via a
#: ``#pragma omp declare reduction`` (OpenMP has no built-in reduction for complex).
_COMPLEX_TYPES = (dtypes.complex64, dtypes.complex128)


def _contiguous_element_count(desc):
    """Element count if ``desc`` is a plain, 0-offset, C-contiguous ``Array`` buffer.

    Only such a buffer can be reduced as one flat ``A[0:count]`` OpenMP array section:
    the section covers the whole allocation contiguously, so the runtime's per-thread
    private copy + final element-wise combine is a faithful whole-buffer reduction.
    Returns ``None`` (caller falls back to atomics) for views, strided/padded layouts,
    or anything whose contiguity can't be proven. ``count`` may be symbolic.
    """
    # Exact type only -- subclasses (View, Reference, Stream) are not plain buffers.
    if type(desc) is not data.Array:
        return None
    acc = 1
    exp = []
    for s in reversed(list(desc.shape)):
        exp.append(acc)
        acc = acc * s
    if list(desc.strides) != list(reversed(exp)):
        return None
    # No allocation padding beyond the logical element count.
    try:
        if bool(symbolic.simplify(desc.total_size - acc) != 0):
            return None
    except TypeError:
        return None
    return acc


def _complex_declare_reduction(op_str: str, ctype: str):
    """``#pragma omp declare reduction`` line for a complex element type, or ``None``.

    OpenMP has no native reduction over ``std::complex``; only ``+`` / ``*`` get a
    well-defined identity (0 / 1) and combiner. Everything else returns ``None`` so the
    caller falls back to the atomic path.
    """
    if op_str == "+":
        return (f"#pragma omp declare reduction(+ : {ctype} : omp_out += omp_in) "
                f"initializer(omp_priv = {ctype}(0))")
    if op_str == "*":
        return (f"#pragma omp declare reduction(* : {ctype} : omp_out *= omp_in) "
                f"initializer(omp_priv = {ctype}(1))")
    return None


@registry.autoregister_params(name='cpu')
class CPUCodeGen(TargetCodeGenerator):
    """ SDFG CPU code generator. """

    title = "CPU"
    target_name = "cpu"
    language = "cpp"

    def _define_sdfg_arguments(self, sdfg, arglist):
        # NOTE: Multi-nesting with container arrays must be further investigated.
        def _visit_structure(struct: data.Structure, args: dict, prefix: str = ''):
            for k, v in struct.members.items():
                if isinstance(v, data.Structure):
                    _visit_structure(v, args, f'{prefix}->{k}')
                elif isinstance(v, data.ContainerArray):
                    _visit_structure(v.stype, args, f'{prefix}->{k}')
                if isinstance(v, data.Data):
                    args[f'{prefix}->{k}'] = v

        # Keeps track of generated connectors, so we know how to access them in nested scopes
        args = dict(arglist)
        for name, arg_type in arglist.items():
            if isinstance(arg_type, data.Structure):
                desc = sdfg.arrays[name]
                _visit_structure(arg_type, args, name)
            elif isinstance(arg_type, data.ContainerArray):
                desc = sdfg.arrays[name]
                desc = desc.stype
                if isinstance(desc, data.Structure):
                    _visit_structure(desc, args, name)

        for name, arg_type in args.items():
            if isinstance(arg_type, data.Scalar):
                # A GPU_Global scalar is a device pointer on the host side: allocate_array
                # cudaMallocs it as ``T*`` and connector-type inference (infer_types) treats
                # GPU_Global data as pointer-typed, so it must be registered as a pointer to
                # match the allocation rather than as a value-typed CPU scalar. This branch is
                # reachable on the legacy CUDA target, which shares this codegen but never runs
                # PromoteGPUScalarsToArrays -- the pass that would otherwise widen such scalars
                # to 1-element arrays before codegen.
                if arg_type.storage is dtypes.StorageType.GPU_Global:
                    self._dispatcher.defined_vars.add(name, DefinedType.Pointer, dtypes.pointer(arg_type.dtype).ctype)
                    continue

                self._dispatcher.defined_vars.add(name, DefinedType.Scalar, arg_type.dtype.ctype)
            elif isinstance(arg_type, data.Array):
                self._dispatcher.defined_vars.add(name, DefinedType.Pointer, dtypes.pointer(arg_type.dtype).ctype)
            elif isinstance(arg_type, data.Stream):
                if arg_type.is_stream_array():
                    self._dispatcher.defined_vars.add(name, DefinedType.StreamArray, arg_type.as_arg(name=''))
                else:
                    self._dispatcher.defined_vars.add(name, DefinedType.Stream, arg_type.as_arg(name=''))
            elif isinstance(arg_type, data.Structure):
                self._dispatcher.defined_vars.add(name, DefinedType.Pointer, arg_type.dtype.ctype)
            else:
                raise TypeError("Unrecognized argument type: {t} (value {v})".format(t=type(arg_type).__name__,
                                                                                     v=str(arg_type)))

    def __init__(self, frame_codegen: 'DaCeCodeGenerator', sdfg: SDFG):
        self._frame = frame_codegen
        self._dispatcher: TargetDispatcher = frame_codegen.dispatcher
        self.calling_codegen = self
        # Root SDFG, kept for get_generated_codeobjects (which runs after frame codegen and has no
        # SDFG argument), mirroring the CUDA target's ``_global_sdfg``.
        self._global_sdfg: SDFG = sdfg
        dispatcher = self._dispatcher

        self._locals = cppunparse.CPPLocals()
        # Scope depth (for defining locals)
        self._ldepth = 0

        # Keep nested SDFG schedule when descending into it
        self._toplevel_schedule = None

        # Keep track of traversed nodes
        self._generated_nodes = set()

        # Stack of {memlet_data_name -> op_str} -- one frame per enclosing
        # OpenMP-reducible map. ``write_and_resolve_expr`` checks the union
        # of frames before emitting a WCR atomic: if the target is already
        # covered by an ``reduction(op:var)`` clause on an enclosing
        # ``#pragma omp parallel for``, the OMP runtime privatizes + tree-
        # reduces it, and an extra atomic add on the per-thread copy is
        # strictly wasted work (and would be incorrect in the rare case
        # the OMP runtime privatizes-by-value rather than by-pointer).
        self._omp_reduction_scope_stack = []

        # id(Map) -> whether its MapEntry opened an encapsulating C scope, so the matching MapExit
        # closes exactly the braces that were opened. Keyed on the Map, which the entry and exit
        # nodes share (they are reached through different subgraph views). See map_scope_needs_brace.
        self._map_scope_braced: Dict[int, bool] = {}

        # Keep track of generated NestedSDG, and the name of the assigned function
        self._generated_nested_sdfg = dict()

        # Buffered translation units for the per-nest split
        # (``compiler.cpu.codegen_params.split_nsdfg_translation_units``):
        # sdfg_label -> (full C++ text of that nest's function, environments it needs).
        # ``_generate_NestedSDFG`` routes a split nest's body here instead of into the frame's
        # function stream, and ``get_generated_codeobjects`` turns each entry into its own
        # CodeObject (= its own .cpp). Empty unless the flag is on, so the default path is untouched.
        self._nsdfg_translation_units: Dict[str, Tuple[str, Set[str]]] = {}

        # Top-level GPU nests lifted into their OWN standalone SDFG + translation unit
        # (``compiler.cpu.codegen_params.external_translation_units``, Model 2):
        # public-ABI child name -> the child SDFG. ``_generate_NestedSDFG`` emits a handle-ABI call
        # to the child instead of inlining its kernels here, and ``get_generated_codeobjects`` runs a
        # fresh ``generate_code`` per child (its own ``.cu``). Empty unless the flag is on.
        self.external_children: Dict[str, SDFG] = {}

        # Identifies the host OUTPUT FILE being generated right now. ``id(self)`` is the frame .cpp,
        # which is the only host file unless the per-nest split routes a top-level nest into its own
        # .cpp -- ``_generate_NestedSDFG`` then re-points this at that nest while generating its
        # subtree, using the nest's ``sdfg_label``. The label (not ``id()`` of the nest's code buffer)
        # is the key precisely because that buffer is a short-lived local: CPython reuses the address
        # of a freed object, so successive nests would collide on one key and the second nest's file
        # would skip a helper the first had already emitted -- the very bug this exists to prevent.
        # The label is the emitted function's name, so it is unique per TU by construction.
        # The base generator only maintains the value; the readable generator reads it to scope its
        # per-file ``<array>_idx`` / ``<array>_size`` dedup (see _flush_generated_functions).
        self._current_tu_key: Union[int, str] = id(self)

        # Accumulator data-names whose map-exit WCR is folded by an enclosing GPU thread-block
        # tree reduction. Maps ``data-name -> {'partial', 'credtype', 'ctype'}``: while a name is
        # present, ``write_and_resolve_expr`` redirects the per-thread atomic into a register
        # accumulate on the named partial (``partial = op(partial, value)``) instead, which the
        # CUDA codegen then folds across the block with ``cub::BlockReduce`` (one atomic/block).
        # The CUDA codegen adds/removes the entries around the thread-block body.
        self._gpu_block_reduction_covered = {}

        # Keeps track of generated connectors, so we know how to access them in nested scopes
        arglist = dict(self._frame.arglist)
        self._define_sdfg_arguments(sdfg, arglist)

        # Register dispatchers
        dispatcher.register_node_dispatcher(self)
        dispatcher.register_map_dispatcher(
            [dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.CPU_Persistent, dtypes.ScheduleType.Sequential],
            self)

        cpu_storage = [dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_ThreadLocal, dtypes.StorageType.Register]
        dispatcher.register_array_dispatcher(cpu_storage, self)

        # Register CPU copies (all internal pairs)
        for src_storage, dst_storage in itertools.product(cpu_storage, cpu_storage):
            dispatcher.register_copy_dispatcher(src_storage, dst_storage, None, self)

    @staticmethod
    def cmake_options():
        options = []

        # Always pinned, so the compiler the flags were chosen for is the one CMake uses. This wins
        # over a CMAKE_CXX_COMPILER set in a toolchain file passed through extra_cmake_args.
        options.append('-DCMAKE_CXX_COMPILER="{}"'.format(make_absolute(compiler_family.host_compiler())))

        flags = compiler_family.cpu_args()
        if flags:
            options.append('-DCMAKE_CXX_FLAGS="{}"'.format(flags))

        return options

    @staticmethod
    def _nsdfg_subtree_is_cpu_only(nsdfg: SDFG) -> bool:
        """Whether a nested SDFG's whole subtree is pure host code, i.e. safe to move into its own
        host ``.cpp``.

        A nest with a GPU-scheduled interior is generated through the GPU codegen, which emits its
        own device file and expects the launching host code in the frame TU it cooperates with;
        relocating that host code into a separate host TU is not a routing decision this pass can
        make safely. ``codegen is self`` already rejects the case where the GPU generator is the
        caller, but a CPU-called nest can still contain a GPU map deeper down, so check the subtree.
        Conservative on purpose: anything not clearly host-only stays in the frame TU.
        """
        gpu_schedules = set(dtypes.GPU_SCHEDULES) | set(dtypes.GPU_SCHEDULES_EXPERIMENTAL_CUDACODEGEN)
        gpu_storages = set(dtypes.GPU_STORAGES)
        for sd in nsdfg.all_sdfgs_recursive():
            for desc in sd.arrays.values():
                if desc.storage in gpu_storages:
                    return False
            for state in sd.states():
                for node in state.nodes():
                    if isinstance(node, nodes.MapEntry) and node.map.schedule in gpu_schedules:
                        return False
        return True

    @staticmethod
    def _rename_full_array_connectors_to_outer(sdfg, state, node):
        """Rename each full-array connector of ``node`` to its OUTER array's name where that binding is
        unambiguous, so the connector and the outer array share a name and codegen emits no alias at all
        (the body just uses the outer pointer, in scope via ``can_access_parent``). Skips a rename that
        would clash -- the same outer array bound through a SEPARATE in- and out-connector (an in/out name
        clash), or a target name already used by a distinct nested array/symbol -- leaving those to the
        ``__restrict__`` alias path. Mutates ``node.sdfg`` in place; only sound because a nest reaching the
        inline path is generated exactly once, here.
        """
        bindings = {}  # outer array name -> list of (edge, connector, is_input)
        for e in state.in_edges(node):
            if e.data is not None and e.data.data is not None and e.dst_conn:
                bindings.setdefault(e.data.data, []).append((e, e.dst_conn, True))
        for e in state.out_edges(node):
            if e.data is not None and e.data.data is not None and e.src_conn:
                bindings.setdefault(e.data.data, []).append((e, e.src_conn, False))
        for outer, binds in bindings.items():
            conns = {conn for _, conn, _ in binds}
            if len(conns) != 1:
                continue  # in/out name clash: in- and out-connectors are distinct nested arrays -> alias
            conn = next(iter(conns))
            if conn == outer:
                continue  # already the same name
            if outer in node.sdfg.arrays or outer in node.sdfg.symbols or outer in node.sdfg.constants:
                continue  # target name already taken inside the nest -> alias
            node.sdfg.replace(conn, outer)  # rename the nested array + every reference to it
            for e, _, is_input in binds:
                if is_input:
                    node.in_connectors[outer] = node.in_connectors.pop(conn)
                    e.dst_conn = outer
                else:
                    node.out_connectors[outer] = node.out_connectors.pop(conn)
                    e.src_conn = outer

    @staticmethod
    def _nsdfg_connectors_are_full_arrays(sdfg, state, node) -> bool:
        """True iff EVERY in/out connector of ``node`` binds a whole outer array (full range, offset 0)
        to a nested array of identical shape and strides -- the case where the two can be aliased with
        one ``T* __restrict__`` pointer assignment instead of passed through a function argument. Rejects
        scalars, sub-ranges, WCR and views, which each need real argument handling. Backs the
        ``inline_full_array_nsdfg`` knob.
        """
        edges = [(e, e.dst_conn) for e in state.in_edges(node)] + [(e, e.src_conn) for e in state.out_edges(node)]
        seen = False
        for e, conn in edges:
            if e.data is None or e.data.data is None:
                continue
            seen = True
            if conn is None or e.data.wcr is not None or conn not in node.sdfg.arrays:
                return False
            outer = sdfg.arrays.get(e.data.data)
            inner = node.sdfg.arrays[conn]
            if outer is None or isinstance(outer, data.Scalar) or isinstance(inner, data.Scalar):
                return False
            if isinstance(outer, data.View) or isinstance(inner, data.View):
                return False
            full = subsets.Range.from_array(outer)
            if e.data.subset is None or not (e.data.subset.covers(full) and full.covers(e.data.subset)):
                return False
            if list(inner.shape) != list(outer.shape) or list(inner.strides) != list(outer.strides):
                return False
        return seen

    def get_generated_codeobjects(self):
        objects = []

        # External-TU children (Model 2): each top-level GPU nest, lifted in ``_generate_NestedSDFG``,
        # is code-generated here as its OWN standalone program -- a fresh ``generate_code`` pass, so its
        # kernels get their own ``.cu``. The child's CodeObjects (its ``.cu``, frame ``.cpp``, etc.) are
        # flattened into this target's output, which feeds the one flat source list both builders
        # (cmake ``DACE_FILES`` and the native compiler) consume -- one project, no sub-libraries.
        if self.external_children:
            from dace.codegen.codegen import generate_code
            for name, child_sdfg in sorted(self.external_children.items()):
                child = deepcopy(child_sdfg)
                child.name = name  # the public ABI the parent forward-declared: __program_<name>, ...
                child.reset_cfg_list()  # deepcopy leaves _cfg_list empty; rebuild so it is a valid root
                # Depth 1: a child is generated with the split OFF, so a nest inside it is not re-lifted.
                with set_temporary('compiler', 'cpu', 'codegen_params', 'external_translation_units', value=False):
                    child_objects = generate_code(child, validate=False)
                # Namespace the child's target-level init/exit so N sub-programs share one binary.
                self._namespace_child_module_symbols(child_objects, name)
                objects.extend(child_objects)

        # The CPU target normally generates inline code (everything lands in the frame's .cpp), so
        # unless the per-nest split buffered something there is nothing further to emit here.
        if not self._nsdfg_translation_units:
            return objects

        top_sdfg = self._global_sdfg
        for label, (code, envs) in sorted(self._nsdfg_translation_units.items()):
            fileheader = CodeIOStream()
            # Re-emit the shared preamble into this TU: includes, custom type definitions and
            # constants. Same technique the CUDA target uses to make its .cu self-contained
            # (cuda.py, get_generated_codeobjects). ``include_hash=False``: the frame's hash.h
            # include is written relative to src/<target>/ and this file sits one level deeper
            # (src/cpu/nsdfg/); only frame code uses __HASH_*.
            #
            # The <sdfg>_state_t struct is DEFINED here, not forward-declared: dace Streams and
            # persistent-lifetime storage are state fields, so any nest may dereference __state,
            # and an incomplete type cannot be.
            #
            # This TU therefore carries a second, identical definition of everything
            # generate_fileheader emits at namespace scope. That is safe for a struct (a type), for
            # guarded includes, and for generate_constants' ``constexpr`` (internal linkage). It is
            # NOT safe for ``sdfg.global_code``, which the same function re-emits verbatim
            # (framecode.py, generate_fileheader) and which accepts arbitrary text: a non-inline
            # definition there multiply-defines across the frame and every nest TU. That predates
            # the split -- the frame/.cu pair has the same exposure -- but N nests widen it.
            self._frame.generate_fileheader(top_sdfg, fileheader, 'frame', include_hash=False)
            objects.append(
                CodeObject(f'{top_sdfg.name}.{label}',
                           '/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */\n#include <dace/dace.h>\n' +
                           fileheader.getvalue() + '\n' + code,
                           'cpp',
                           CPUCodeGen,
                           'NestedSDFG',
                           target_type='nsdfg',
                           environments=envs,
                           sdfg=top_sdfg))
        return objects

    @property
    def has_initializer(self):
        return False

    @property
    def has_finalizer(self):
        return False

    def generate_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                       function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        entry_node = dfg_scope.source_nodes()[0]
        cpp.presynchronize_streams(sdfg, cfg, dfg_scope, state_id, entry_node, callsite_stream)

        self.generate_node(sdfg, cfg, dfg_scope, state_id, entry_node, function_stream, callsite_stream)
        self._dispatcher.dispatch_subgraph(sdfg,
                                           cfg,
                                           dfg_scope,
                                           state_id,
                                           function_stream,
                                           callsite_stream,
                                           skip_entry_node=True)

    def generate_node(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: ScopeSubgraphView, state_id: int, node: nodes.Node,
                      function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        # Dynamically obtain node generator according to class name
        try:
            gen = getattr(self, "_generate_" + type(node).__name__)
        except AttributeError:
            if isinstance(node, nodes.LibraryNode):
                raise NodeNotExpandedError(sdfg, state_id, dfg.node_id(node))
            raise

        gen(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

        # Mark node as "generated"
        self._generated_nodes.add(node)
        self._locals.clear_scope(self._ldepth + 1)

    def _viewed_data_is_const(self, sdfg: SDFG, viewed_dnode: nodes.AccessNode) -> bool:
        """Whether the data viewed by a ``View`` is already declared ``const`` in the emitted code.

        A view aliasing ``const`` data must itself be ``const`` (mirroring its parent). The viewed
        node is allocated before the view (see :meth:`allocate_view`), so its registered ctype is
        available -- a leading ``const`` qualifier is the signal.
        """
        for key in (self.ptr(viewed_dnode.data, viewed_dnode.desc(sdfg), sdfg), viewed_dnode.data):
            try:
                _, ctype = self._dispatcher.defined_vars.get(key)
            except KeyError:
                continue
            return ctype.strip().startswith('const ')
        return False

    def allocate_view(self,
                      sdfg: SDFG,
                      cfg: ControlFlowRegion,
                      dfg: SDFGState,
                      state_id: int,
                      node: nodes.AccessNode,
                      global_stream: CodeIOStream,
                      declaration_stream: CodeIOStream,
                      allocation_stream: CodeIOStream,
                      decouple_array_interfaces: bool = False) -> None:
        """
        Allocates (creates pointer and refers to original) a view of an
        existing array, scalar, or view.
        """

        name = node.data
        nodedesc = node.desc(sdfg)
        ptrname = self.ptr(name, nodedesc, sdfg)

        # Check if array is already declared
        declared = self._dispatcher.declared_arrays.has(ptrname)

        # Check directionality of view (referencing dst or src)
        edge = sdutils.get_view_edge(dfg, node)

        if edge is None:
            return

        # We need to know if this is a read or a write variation
        is_write = edge.src is node

        # Allocate the viewed data before the view, if necessary
        mpath = dfg.memlet_path(edge)
        viewed_dnode: nodes.AccessNode = mpath[-1].dst if is_write else mpath[0].src
        self._dispatcher.dispatch_allocate(sdfg, cfg, dfg, state_id, viewed_dnode, viewed_dnode.desc(sdfg),
                                           global_stream, allocation_stream)

        # Memlet points to view, construct mirror memlet
        memlet = edge.data
        if memlet.data == node.data:
            memlet = deepcopy(memlet)
            memlet.data = viewed_dnode.data
            memlet.subset = memlet.dst_subset if is_write else memlet.src_subset
            if memlet.subset is None:
                memlet.subset = subsets.Range.from_array(viewed_dnode.desc(sdfg))

        # Emit memlet as a reference and register defined variable. A view must mirror the const
        # qualifier of the data it views: a ``const`` parent (e.g. a read-only nested-SDFG argument)
        # cannot be aliased by a non-const ``T*`` view (an illegal ``const T* -> T*`` conversion), so
        # the view is emitted pointer-to-const too. A non-const parent must keep non-const views --
        # the view edge's read/write *direction* does not imply the view's contents are never written
        # (reinterpret / same-name views are read-direction yet written), so const-ness is keyed off
        # the parent, not the direction. ``_mutated_descriptors`` guarantees a const parent is never
        # written through any view, so mirroring is always sound.
        const_view = (not isinstance(sdfg.arrays[viewed_dnode.data],
                                     (data.Structure, data.ContainerArray, data.ContainerView))
                      and self._viewed_data_is_const(sdfg, viewed_dnode))
        atype, aname, value = cpp.emit_memlet_reference(self._dispatcher,
                                                        sdfg,
                                                        memlet,
                                                        name,
                                                        dtypes.pointer(nodedesc.dtype),
                                                        codegen=self,
                                                        ancestor=0,
                                                        is_write=is_write,
                                                        const_read_only_array=const_view)

        # Test for views of container arrays and structs
        if isinstance(sdfg.arrays[viewed_dnode.data], (data.Structure, data.ContainerArray, data.ContainerView)):
            vdesc = sdfg.arrays[viewed_dnode.data]
            ptrname = self.ptr(memlet.data, vdesc, sdfg)
            field_name = None
            if is_write and mpath[-1].dst_conn:
                field_name = mpath[-1].dst_conn
            elif not is_write and mpath[0].src_conn:
                field_name = mpath[0].src_conn

            # Plain view into a container array
            if isinstance(vdesc, data.ContainerArray) and not isinstance(vdesc.stype, data.Structure):
                offset = cpp.cpp_offset_expr(vdesc, memlet.subset)
                value = f'{ptrname}[{offset}]'
            else:
                if field_name is not None:
                    if isinstance(vdesc, data.ContainerArray):
                        offset = cpp.cpp_offset_expr(vdesc, memlet.subset)
                        arrexpr = f'{ptrname}[{offset}]'
                        stype = vdesc.stype
                    else:
                        arrexpr = f'{ptrname}'
                        stype = vdesc

                    value = f'{arrexpr}->{field_name}'
                    if isinstance(stype.members[field_name], data.Scalar):
                        value = '&' + value

        if not declared:
            # Keep the registered ctype consistent with the emitted declaration: a read-only view
            # is declared as a pointer-to-const (see ``const_view`` above), so consumers that look
            # it up must see the same qualifier.
            ctypedef = dtypes.pointer(nodedesc.dtype).ctype
            if const_view:
                ctypedef = 'const ' + ctypedef
            self._dispatcher.declared_arrays.add(aname, DefinedType.Pointer, ctypedef)
            if isinstance(nodedesc, data.StructureView):
                for k, v in nodedesc.members.items():
                    if isinstance(v, data.Data):
                        ctypedef = dtypes.pointer(v.dtype).ctype if isinstance(v, data.Array) else v.dtype.ctype
                        defined_type = DefinedType.Scalar if isinstance(v, data.Scalar) else DefinedType.Pointer
                        self._dispatcher.declared_arrays.add(f"{name}->{k}", defined_type, ctypedef)
                        self._dispatcher.defined_vars.add(f"{name}->{k}", defined_type, ctypedef)
                # TODO: Find a better way to do this (the issue is with pointers of pointers)
                if atype.endswith('*'):
                    atype = atype[:-1]
                if value.startswith('&'):
                    value = value[1:]
            declaration_stream.write(f'{atype} {aname};', cfg, state_id, node)
        allocation_stream.write(f'{aname} = {value};', cfg, state_id, node)

    def allocate_reference(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: SDFGState, state_id: int,
                           node: nodes.AccessNode, global_stream: CodeIOStream, declaration_stream: CodeIOStream,
                           allocation_stream: CodeIOStream) -> None:
        name = node.data
        nodedesc = node.desc(sdfg)
        ptrname = self.ptr(name, nodedesc, sdfg)

        # Check if reference is already declared
        declared = self._dispatcher.declared_arrays.has(ptrname)

        if not declared:
            declaration_stream.write(f'{nodedesc.dtype.ctype} *{ptrname};', cfg, state_id, node)
            ctypedef = dtypes.pointer(nodedesc.dtype).ctype
            self._dispatcher.declared_arrays.add(ptrname, DefinedType.Pointer, ctypedef)
            self._dispatcher.defined_vars.add(ptrname, DefinedType.Pointer, ctypedef)

    def declare_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int, node: nodes.Node,
                      nodedesc: data.Data, function_stream: CodeIOStream, declaration_stream: CodeIOStream) -> None:
        fsymbols = self._frame.symbols_and_constants(sdfg)
        # NOTE: `dfg` (state) will be None iff `nodedesc` is non-free symbol dependent
        # (see `DaCeCodeGenerator.determine_allocation_lifetime` in `dace.codegen.targets.framecode`).
        # We add the `dfg is not None` check because the `sdutils.is_nonfree_sym_dependent` check will fail if
        # `nodedesc` is a View and `dfg` is None.
        if dfg and not sdutils.is_nonfree_sym_dependent(node, nodedesc, dfg, fsymbols):
            raise NotImplementedError("The declare_array method should only be used for variables "
                                      "that must have their declaration and allocation separate.")

        name = node.root_data
        ptrname = self.ptr(name, nodedesc, sdfg)

        if nodedesc.transient is False:
            return

        # Check if array is already declared
        if self._dispatcher.declared_arrays.has(ptrname):
            return

        # Compute array size
        arrsize = nodedesc.total_size
        if not isinstance(nodedesc.dtype, dtypes.opaque):
            arrsize_bytes = arrsize * nodedesc.dtype.bytes

        if (nodedesc.storage == dtypes.StorageType.CPU_Heap or nodedesc.storage == dtypes.StorageType.Register):

            ctypedef = dtypes.pointer(nodedesc.dtype).ctype

            declaration_stream.write(f'{nodedesc.dtype.ctype} *{name} = nullptr;\n', cfg, state_id, node)
            self._dispatcher.declared_arrays.add(name, DefinedType.Pointer, ctypedef)
            return
        elif nodedesc.storage is dtypes.StorageType.CPU_ThreadLocal:
            # Define pointer once
            # NOTE: OpenMP threadprivate storage MUST be declared globally.
            function_stream.write(
                "{ctype} *{name} = nullptr;\n"
                "#pragma omp threadprivate({name})".format(ctype=nodedesc.dtype.ctype, name=name),
                cfg,
                state_id,
                node,
            )
            self._dispatcher.declared_arrays.add_global(name, DefinedType.Pointer, '%s *' % nodedesc.dtype.ctype)
        else:
            raise NotImplementedError("Unimplemented storage type " + str(nodedesc.storage))

    def allocate_array(self,
                       sdfg: SDFG,
                       cfg: ControlFlowRegion,
                       dfg: StateSubgraphView,
                       state_id: int,
                       node: nodes.AccessNode,
                       nodedesc: data.Data,
                       function_stream: CodeIOStream,
                       declaration_stream: CodeIOStream,
                       allocation_stream: CodeIOStream,
                       allocate_nested_data: bool = True) -> None:
        alloc_name = self.ptr(node.data, nodedesc, sdfg)
        name = alloc_name

        tokens = node.data.split('.')
        top_desc = sdfg.arrays[tokens[0]]
        # NOTE: Assuming here that all Structure members share transient/storage/lifetime properties.
        # TODO: Study what is needed in the DaCe stack to ensure this assumption is correct.
        top_transient = top_desc.transient
        top_storage = top_desc.storage
        top_lifetime = top_desc.lifetime

        if top_transient is False:
            return

        # Check if array is already allocated
        if self._dispatcher.defined_vars.has(name):
            return

        if len(tokens) > 1:
            for i in range(len(tokens) - 1):
                tmp_name = '.'.join(tokens[:i + 1])
                tmp_alloc_name = self.ptr(tmp_name, sdfg.arrays[tmp_name], sdfg)
                if not self._dispatcher.defined_vars.has(tmp_alloc_name):
                    self.allocate_array(sdfg,
                                        cfg,
                                        dfg,
                                        state_id,
                                        nodes.AccessNode(tmp_name),
                                        sdfg.arrays[tmp_name],
                                        function_stream,
                                        declaration_stream,
                                        allocation_stream,
                                        allocate_nested_data=False)
            declared = True
        else:
            # Check if array is already declared
            declared = self._dispatcher.declared_arrays.has(name)

        define_var = self._dispatcher.defined_vars.add
        if top_lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
            define_var = self._dispatcher.defined_vars.add_global
            nodedesc = update_persistent_desc(nodedesc, sdfg)

        # Compute array size
        arrsize = nodedesc.total_size
        arrsize_bytes = None
        if not isinstance(nodedesc.dtype, dtypes.opaque):
            arrsize_bytes = arrsize * nodedesc.dtype.bytes

        if isinstance(nodedesc, data.Structure) and not isinstance(nodedesc, data.StructureView):
            declaration_stream.write(f"{nodedesc.ctype} {name} = new {nodedesc.dtype.base_type};\n")
            define_var(name, DefinedType.Pointer, nodedesc.ctype)
            if allocate_nested_data:
                for k, v in nodedesc.members.items():
                    if isinstance(v, data.Data):
                        ctypedef = dtypes.pointer(v.dtype).ctype if isinstance(v, data.Array) else v.dtype.ctype
                        defined_type = DefinedType.Scalar if isinstance(v, data.Scalar) else DefinedType.Pointer
                        self._dispatcher.declared_arrays.add(f"{name}->{k}", defined_type, ctypedef)
                        if isinstance(v, data.Scalar):
                            # NOTE: Scalar members are already defined in the struct definition.
                            self._dispatcher.defined_vars.add(f"{name}->{k}", defined_type, ctypedef)
                        else:
                            self.allocate_array(sdfg, cfg, dfg, state_id, nodes.AccessNode(f"{name}.{k}"), v,
                                                function_stream, declaration_stream, allocation_stream)
            return
        if isinstance(nodedesc, data.View):
            return self.allocate_view(sdfg, cfg, dfg, state_id, node, function_stream, declaration_stream,
                                      allocation_stream)
        if isinstance(nodedesc, data.Reference):
            return self.allocate_reference(sdfg, cfg, dfg, state_id, node, function_stream, declaration_stream,
                                           allocation_stream)
        if isinstance(nodedesc, data.Scalar):
            if node.setzero:
                declaration_stream.write("%s %s = 0;\n" % (nodedesc.dtype.ctype, name), cfg, state_id, node)
            else:
                declaration_stream.write("%s %s;\n" % (nodedesc.dtype.ctype, name), cfg, state_id, node)
            define_var(name, DefinedType.Scalar, nodedesc.dtype.ctype)
        elif isinstance(nodedesc, data.Stream):
            ###################################################################
            # Stream directly connected to an array

            if is_array_stream_view(sdfg, dfg, node):
                if state_id is None:
                    raise SyntaxError("Stream-view of array may not be defined in more than one state")

                arrnode = sdfg.arrays[nodedesc.sink]
                state: SDFGState = cfg.nodes()[state_id]
                edges = state.out_edges(node)
                if len(edges) > 1:
                    raise NotImplementedError("Cannot handle streams writing to multiple arrays.")

                memlet_path = state.memlet_path(edges[0])
                # Allocate the array before its stream view, if necessary
                self.allocate_array(sdfg, cfg, dfg, state_id, memlet_path[-1].dst, memlet_path[-1].dst.desc(sdfg),
                                    function_stream, declaration_stream, allocation_stream)

                array_expr = cpp.copy_expr(self._dispatcher, sdfg, nodedesc.sink, edges[0].data)
                threadlocal = ""
                threadlocal_stores = [dtypes.StorageType.CPU_ThreadLocal, dtypes.StorageType.Register]
                if (sdfg.arrays[nodedesc.sink].storage in threadlocal_stores or nodedesc.storage in threadlocal_stores):
                    threadlocal = "Threadlocal"
                ctype = 'dace::ArrayStreamView%s<%s>' % (threadlocal, arrnode.dtype.ctype)
                declaration_stream.write(
                    "%s %s (%s);\n" % (ctype, name, array_expr),
                    cfg,
                    state_id,
                    node,
                )
                define_var(name, DefinedType.Stream, ctype)
                return

            ###################################################################
            # Regular stream

            dtype = nodedesc.dtype.ctype
            ctypedef = 'dace::Stream<{}>'.format(dtype)
            if nodedesc.buffer_size != 0:
                definition = "{} {}({});".format(ctypedef, name, nodedesc.buffer_size)
            else:
                definition = "{} {};".format(ctypedef, name)

            declaration_stream.write(definition, cfg, state_id, node)
            define_var(name, DefinedType.Stream, ctypedef)

        elif (nodedesc.storage == dtypes.StorageType.CPU_Heap
              or (nodedesc.storage == dtypes.StorageType.Register and
                  ((symbolic.issymbolic(arrsize, sdfg.constants)) or
                   (arrsize_bytes and ((arrsize_bytes > Config.get("compiler", "max_stack_array_size")) == True))))):

            if nodedesc.storage == dtypes.StorageType.Register:

                if symbolic.issymbolic(arrsize, sdfg.constants):
                    warnings.warn('Variable-length array %s with size %s '
                                  'detected and was allocated on the heap instead of '
                                  '%s' % (name, cpp.sym2cpp(arrsize), nodedesc.storage))
                elif (arrsize_bytes > Config.get("compiler", "max_stack_array_size")) == True:
                    warnings.warn("Array {} with size {} detected and was allocated on the heap instead of "
                                  "{} since its size is greater than max_stack_array_size ({})".format(
                                      name, cpp.sym2cpp(arrsize_bytes), nodedesc.storage,
                                      Config.get("compiler", "max_stack_array_size")))

            ctypedef = dtypes.pointer(nodedesc.dtype).ctype

            # A generator may fuse the declaration into the allocation, turning the pair into one
            # definition statement; the base never does, so `declarator` is None here and the
            # declaration/allocation split below is unchanged.
            declarator = self.fused_heap_declarator(sdfg, name, nodedesc, arrsize, declared, declaration_stream,
                                                    allocation_stream)
            if not declared and declarator is None:
                declaration_stream.write(f'{nodedesc.dtype.ctype} *{name};\n', cfg, state_id, node)
            allocation_stream.write(
                self.heap_alloc_stmt(alloc_name if declarator is None else declarator,
                                     nodedesc.dtype.ctype,
                                     cpp.sym2cpp(arrsize),
                                     nodedesc.alignment,
                                     sdfg=sdfg,
                                     nodedesc=nodedesc,
                                     data_name=node.data), cfg, state_id, node)
            define_var(name, DefinedType.Pointer, ctypedef)

            if node.setzero:
                allocation_stream.write("memset(%s, 0, sizeof(%s)*%s);" %
                                        (alloc_name, nodedesc.dtype.ctype, cpp.sym2cpp(arrsize)))
            if nodedesc.start_offset != 0:
                allocation_stream.write(f'{alloc_name} += {cpp.sym2cpp(nodedesc.start_offset)};\n', cfg, state_id, node)

            return
        elif (nodedesc.storage == dtypes.StorageType.Register):
            # The assignment necessary to unify the explicit streams and streams declared through
            # the state of the SDFG.
            if nodedesc.dtype == dtypes.gpuStream_t:
                ctype = dtypes.gpuStream_t.ctype
                allocation_stream.write(f"{ctype}* {name} = __state->gpu_context->streams;")
                # Local is ``gpuStream_t* {name}`` -- register the matching
                # pointer ctype so consumers (``emit_memlet_reference``) emit
                # ``gpuStream_t* gpu_streams`` in nested-SDFG signatures
                # instead of ``gpuStream_t gpu_streams`` (1 vs. 2 pointer
                # levels).
                define_var(name, DefinedType.Pointer, dtypes.pointer(dtypes.gpuStream_t).ctype)
                return

            ctypedef = dtypes.pointer(nodedesc.dtype).ctype
            if nodedesc.start_offset != 0:
                raise NotImplementedError('Start offset unsupported for registers')
            if node.setzero:
                declaration_stream.write(
                    "%s %s[%s]  DACE_ALIGN(64) = {0};\n" % (nodedesc.dtype.ctype, name, cpp.sym2cpp(arrsize)),
                    cfg,
                    state_id,
                    node,
                )
                define_var(name, DefinedType.Pointer, ctypedef)
                return
            declaration_stream.write(
                "%s %s[%s]  DACE_ALIGN(64);\n" % (nodedesc.dtype.ctype, name, cpp.sym2cpp(arrsize)),
                cfg,
                state_id,
                node,
            )
            define_var(name, DefinedType.Pointer, ctypedef)
            return
        elif nodedesc.storage is dtypes.StorageType.CPU_ThreadLocal:
            # Define pointer once
            # NOTE: OpenMP threadprivate storage MUST be declared globally.
            if not declared:
                function_stream.write(
                    "{ctype} *{name};\n#pragma omp threadprivate({name})".format(ctype=nodedesc.dtype.ctype, name=name),
                    cfg,
                    state_id,
                    node,
                )
                self._dispatcher.declared_arrays.add_global(name, DefinedType.Pointer, '%s *' % nodedesc.dtype.ctype)

            # Allocate in each OpenMP thread
            allocation_stream.write(
                """
                #pragma omp parallel
                {{
                    {name} = new {ctype} DACE_ALIGN(64)[{arrsize}];""".format(ctype=nodedesc.dtype.ctype,
                                                                              name=alloc_name,
                                                                              arrsize=cpp.sym2cpp(arrsize)),
                cfg,
                state_id,
                node,
            )
            if node.setzero:
                allocation_stream.write("memset(%s, 0, sizeof(%s)*%s);" %
                                        (alloc_name, nodedesc.dtype.ctype, cpp.sym2cpp(arrsize)))
            if nodedesc.start_offset != 0:
                allocation_stream.write(f'{alloc_name} += {cpp.sym2cpp(nodedesc.start_offset)};\n', cfg, state_id, node)

            # Close OpenMP parallel section
            allocation_stream.write('}')
            self._dispatcher.defined_vars.add_global(name, DefinedType.Pointer, '%s *' % nodedesc.dtype.ctype)
        else:
            raise NotImplementedError("Unimplemented storage type " + str(nodedesc.storage))

    def deallocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                         node: nodes.AccessNode, nodedesc: data.Data, function_stream: CodeIOStream,
                         callsite_stream: CodeIOStream) -> None:
        arrsize = nodedesc.total_size
        arrsize_bytes = None
        if not isinstance(nodedesc.dtype, dtypes.opaque):
            arrsize_bytes = arrsize * nodedesc.dtype.bytes

        alloc_name = self.ptr(node.data, nodedesc, sdfg)
        if isinstance(nodedesc, data.Array) and nodedesc.start_offset != 0:
            alloc_name = f'({alloc_name} - {cpp.sym2cpp(nodedesc.start_offset)})'

        if self._dispatcher.declared_arrays.has(alloc_name):
            is_global = nodedesc.lifetime in (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.Persistent,
                                              dtypes.AllocationLifetime.External)
            self._dispatcher.declared_arrays.remove(alloc_name, is_global=is_global)

        if isinstance(nodedesc, (data.Scalar, data.View, data.Stream, data.Reference)):
            return
        elif nodedesc.dtype == dtypes.gpuStream_t:
            callsite_stream.write(f"{alloc_name} = nullptr;")
            return
        elif (nodedesc.storage == dtypes.StorageType.CPU_Heap
              or (nodedesc.storage == dtypes.StorageType.Register and
                  (symbolic.issymbolic(arrsize, sdfg.constants) or
                   (arrsize_bytes and ((arrsize_bytes > Config.get("compiler", "max_stack_array_size")) == True))))):
            callsite_stream.write(self.heap_free_stmt(alloc_name, isinstance(nodedesc, data.Array)), cfg, state_id,
                                  node)
        elif nodedesc.storage is dtypes.StorageType.CPU_ThreadLocal:
            # Deallocate in each OpenMP thread
            if isinstance(nodedesc, data.Array):
                deleteop = "delete[]"
            else:
                deleteop = "delete"
            callsite_stream.write(
                f"""#pragma omp parallel
                {{
                    {deleteop} {alloc_name};
                }}""",
                cfg,
                state_id,
                node,
            )
        else:
            return

    def copy_memory(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        src_node: Union[nodes.Tasklet, nodes.AccessNode],
        dst_node: Union[nodes.Tasklet, nodes.AccessNode],
        edge: Tuple[nodes.Node, Optional[str], nodes.Node, Optional[str], mmlt.Memlet],
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        if isinstance(src_node, nodes.Tasklet):
            src_storage = dtypes.StorageType.Register
            try:
                src_parent = dfg.entry_node(src_node)
            except KeyError:
                src_parent = None
            dst_schedule = None if src_parent is None else src_parent.map.schedule
        else:
            src_storage = src_node.desc(sdfg).storage

        if isinstance(dst_node, nodes.Tasklet):
            dst_storage = dtypes.StorageType.Register
        else:
            dst_storage = dst_node.desc(sdfg).storage

        try:
            dst_parent = dfg.entry_node(dst_node)
        except KeyError:
            dst_parent = None
        dst_schedule = None if dst_parent is None else dst_parent.map.schedule

        state_dfg = cfg.node(state_id)

        # Emit actual copy
        self._emit_copy(
            sdfg,
            cfg,
            state_id,
            src_node,
            src_storage,
            dst_node,
            dst_storage,
            dst_schedule,
            edge,
            state_dfg,
            callsite_stream,
        )

    def _emit_copy(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        state_id: int,
        src_node: nodes.Node,
        src_storage: dtypes.StorageType,
        dst_node: nodes.Node,
        dst_storage: dtypes.StorageType,
        dst_schedule: dtypes.ScheduleType,
        edge: Tuple[nodes.Node, Optional[str], nodes.Node, Optional[str], mmlt.Memlet],
        dfg: StateSubgraphView,
        stream: CodeIOStream,
    ) -> None:
        u, uconn, v, vconn, memlet = edge
        orig_vconn = vconn

        # Determine memlet directionality
        if isinstance(src_node, nodes.AccessNode) and validate_memlet_data(memlet.data, src_node.data):
            write = True
        elif isinstance(dst_node, nodes.AccessNode) and validate_memlet_data(memlet.data, dst_node.data):
            write = False
        elif isinstance(src_node, nodes.CodeNode) and isinstance(dst_node, nodes.CodeNode):
            # Code->Code copy (not read nor write)
            raise RuntimeError("Copying between code nodes is only supported as part of the participating nodes")
        elif uconn is None and vconn is None and memlet.data is None and dst_schedule == dtypes.ScheduleType.Sequential:
            # Sequential dependency edge
            return
        else:
            raise LookupError("Memlet does not point to any of the nodes")

        if isinstance(dst_node, nodes.Tasklet):
            # Copy into tasklet
            stream.write(
                "    " + self.memlet_definition(sdfg, memlet, False, vconn, dst_node.in_connectors[vconn]),
                cfg,
                state_id,
                [src_node, dst_node],
            )
            return
        elif isinstance(src_node, nodes.Tasklet):
            # Copy out of tasklet
            stream.write(
                "    " + self.memlet_definition(sdfg, memlet, True, uconn, src_node.out_connectors[uconn]),
                cfg,
                state_id,
                [src_node, dst_node],
            )
            return
        else:  # Copy array-to-array
            src_nodedesc = src_node.desc(sdfg)
            dst_nodedesc = dst_node.desc(sdfg)

            if write:
                vconn = self.ptr(dst_node.data, dst_nodedesc, sdfg)
            ctype = dst_nodedesc.dtype.ctype

            #############################################
            # Corner cases

            # Setting a reference
            if isinstance(dst_nodedesc, data.Reference) and orig_vconn == 'set':
                srcptr = self.ptr(src_node.data, src_nodedesc, sdfg)
                defined_type, _ = self._dispatcher.defined_vars.get(srcptr)
                stream.write(
                    "%s = %s;" % (vconn, cpp.cpp_ptr_expr(sdfg, memlet, defined_type, codegen=self)),
                    cfg,
                    state_id,
                    [src_node, dst_node],
                )
                return

            # Writing from/to a stream
            if isinstance(sdfg.arrays[memlet.data], data.Stream) or (isinstance(src_node, nodes.AccessNode)
                                                                     and isinstance(src_nodedesc, data.Stream)):
                # Identify whether a stream is writing to an array
                if isinstance(dst_nodedesc, (data.Scalar, data.Array)) and isinstance(src_nodedesc, data.Stream):
                    # Stream -> Array - pop bulk
                    if is_array_stream_view(sdfg, dfg, src_node):
                        return  # Do nothing (handled by ArrayStreamView)

                    array_subset = (memlet.subset if memlet.data == dst_node.data else memlet.other_subset)
                    if array_subset is None:  # Need to use entire array
                        array_subset = subsets.Range.from_array(dst_nodedesc)

                    # stream_subset = (memlet.subset
                    #                  if memlet.data == src_node.data else
                    #                  memlet.other_subset)
                    stream_subset = memlet.subset
                    if memlet.data != src_node.data and memlet.other_subset:
                        stream_subset = memlet.other_subset

                    stream_expr = cpp.cpp_offset_expr(src_nodedesc, stream_subset)
                    array_expr = cpp.cpp_offset_expr(dst_nodedesc, array_subset)
                    assert functools.reduce(lambda a, b: a * b, src_nodedesc.shape, 1) == 1
                    stream.write(
                        "{s}.pop(&{arr}[{aexpr}], {maxsize});".format(s=self.ptr(src_node.data, src_nodedesc, sdfg),
                                                                      arr=self.ptr(dst_node.data, dst_nodedesc, sdfg),
                                                                      aexpr=array_expr,
                                                                      maxsize=cpp.sym2cpp(array_subset.num_elements())),
                        cfg,
                        state_id,
                        [src_node, dst_node],
                    )
                    return
                # Array -> Stream - push bulk
                if isinstance(src_nodedesc, (data.Scalar, data.Array)) and isinstance(dst_nodedesc, data.Stream):
                    if isinstance(src_nodedesc, data.Scalar):
                        stream.write(
                            "{s}.push({arr});".format(s=self.ptr(dst_node.data, dst_nodedesc, sdfg),
                                                      arr=self.ptr(src_node.data, src_nodedesc, sdfg)),
                            cfg,
                            state_id,
                            [src_node, dst_node],
                        )
                    elif hasattr(src_nodedesc, "src"):  # Array-stream view, ``src`` set by is_array_stream_view
                        stream.write(
                            "{s}.push({arr});".format(s=self.ptr(dst_node.data, dst_nodedesc, sdfg),
                                                      arr=self.ptr(src_nodedesc.src, sdfg.arrays[src_nodedesc.src],
                                                                   sdfg)),
                            cfg,
                            state_id,
                            [src_node, dst_node],
                        )
                    else:
                        copysize = " * ".join([cpp.sym2cpp(s) for s in memlet.subset.size()])
                        stream.write(
                            "{s}.push({arr}, {size});".format(s=self.ptr(dst_node.data, dst_nodedesc, sdfg),
                                                              arr=self.ptr(src_node.data, src_nodedesc, sdfg),
                                                              size=copysize),
                            cfg,
                            state_id,
                            [src_node, dst_node],
                        )
                    return
                else:
                    # Unknown case
                    raise NotImplementedError

            #############################################

            state_dfg: SDFGState = cfg.nodes()[state_id]

            copy_shape, src_strides, dst_strides, src_expr, dst_expr = cpp.memlet_copy_to_absolute_strides(
                self._dispatcher, sdfg, state_dfg, edge, src_node, dst_node)

            # Which numbers to include in the variable argument part
            dynshape, dynsrc, dyndst = 1, 1, 1

            # Dynamic copy dimensions
            if any(symbolic.issymbolic(s, sdfg.constants) for s in copy_shape):
                copy_tmpl = "Dynamic<{type}, {veclen}, {aligned}, {dims}>".format(
                    type=ctype,
                    veclen=1,  # Taken care of in "type"
                    aligned="false",
                    dims=len(copy_shape),
                )
            else:  # Static copy dimensions
                copy_tmpl = "<{type}, {veclen}, {aligned}, {dims}>".format(
                    type=ctype,
                    veclen=1,  # Taken care of in "type"
                    aligned="false",
                    dims=", ".join(cpp.sym2cpp(copy_shape)),
                )
                dynshape = 0

            # Constant src/dst dimensions
            if not any(symbolic.issymbolic(s, sdfg.constants) for s in dst_strides):
                # Constant destination
                shape_tmpl = "template ConstDst<%s>" % ", ".join(cpp.sym2cpp(dst_strides))
                dyndst = 0
            elif not any(symbolic.issymbolic(s, sdfg.constants) for s in src_strides):
                # Constant source
                shape_tmpl = "template ConstSrc<%s>" % ", ".join(cpp.sym2cpp(src_strides))
                dynsrc = 0
            else:
                # Both dynamic
                shape_tmpl = "Dynamic"

            # Parameter pack handling
            stride_tmpl_args = [0] * (dynshape + dynsrc + dyndst) * len(copy_shape)
            j = 0
            for shape, src, dst in zip(copy_shape, src_strides, dst_strides):
                if dynshape > 0:
                    stride_tmpl_args[j] = shape
                    j += 1
                if dynsrc > 0:
                    stride_tmpl_args[j] = src
                    j += 1
                if dyndst > 0:
                    stride_tmpl_args[j] = dst
                    j += 1

            copy_args = ([src_expr, dst_expr] +
                         ([] if memlet.wcr is None else [cpp.unparse_cr(sdfg, memlet.wcr, dst_nodedesc.dtype)]) +
                         cpp.sym2cpp(stride_tmpl_args))

            # Instrumentation: Pre-copy
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_copy_begin(sdfg, cfg, state_dfg, src_node, dst_node, edge, stream, None, copy_shape,
                                        src_strides, dst_strides)

            nc = True
            if memlet.wcr is not None:
                nc = not cpp.is_write_conflicted(dfg, edge, sdfg_schedule=self._toplevel_schedule)
            if nc:
                stream.write(
                    """
                    dace::CopyND{copy_tmpl}::{shape_tmpl}::{copy_func}(
                        {copy_args});""".format(
                        copy_tmpl=copy_tmpl,
                        shape_tmpl=shape_tmpl,
                        copy_func="Copy" if memlet.wcr is None else "Accumulate",
                        copy_args=", ".join(copy_args),
                    ),
                    cfg,
                    state_id,
                    [src_node, dst_node],
                )
            else:  # Conflicted WCR
                if dynshape == 1:
                    warnings.warn('Performance warning: Emitting dynamically-'
                                  'shaped atomic write-conflict resolution of an array.')
                    stream.write(
                        """
                        dace::CopyND{copy_tmpl}::{shape_tmpl}::Accumulate_atomic(
                        {copy_args});""".format(
                            copy_tmpl=copy_tmpl,
                            shape_tmpl=shape_tmpl,
                            copy_args=", ".join(copy_args),
                        ),
                        cfg,
                        state_id,
                        [src_node, dst_node],
                    )
                elif copy_shape == [1]:  # Special case: accumulating one element
                    dst_expr = self.memlet_view_ctor(sdfg, memlet, dst_nodedesc.dtype, True)
                    stream.write(
                        self.write_and_resolve_expr(
                            sdfg, memlet, nc, dst_expr, '*(' + src_expr + ')', dtype=dst_nodedesc.dtype) + ';', cfg,
                        state_id, [src_node, dst_node])
                else:
                    warnings.warn('Minor performance warning: Emitting statically-'
                                  'shaped atomic write-conflict resolution of an array.')
                    stream.write(
                        """
                        dace::CopyND{copy_tmpl}::{shape_tmpl}::Accumulate_atomic(
                        {copy_args});""".format(
                            copy_tmpl=copy_tmpl,
                            shape_tmpl=shape_tmpl,
                            copy_args=", ".join(copy_args),
                        ),
                        cfg,
                        state_id,
                        [src_node, dst_node],
                    )

        #############################################################
        # Instrumentation: Post-copy
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_copy_end(sdfg, cfg, state_dfg, src_node, dst_node, edge, stream, None)
        #############################################################

    ###########################################################################
    # Memlet handling

    def write_and_resolve_expr(self,
                               sdfg: SDFG,
                               memlet: mmlt.Memlet,
                               nc: bool,
                               outname: str,
                               inname: str,
                               indices=None,
                               dtype=None):
        """
        Emits a conflict resolution call from a memlet.
        """

        redtype = operations.detect_reduction_type(memlet.wcr)
        # Enclosing GPU thread-block map folds this target via a tree reduction: accumulate the
        # value into this thread's private register partial (no atomic) and let ``cub::BlockReduce``
        # at the map exit drain it (one atomic per block). Emitting the per-thread atomic here would
        # both contend and, with the block fold, double-count. The partial is a register array whose
        # index is the accumulator offset relative to the reduced range base, so a single scalar
        # accumulator (``base``-offset 0, one slot) and a length-``m`` subset reduction share a path.
        cover = self._gpu_block_reduction_covered.get(memlet.data)
        if cover is not None:
            slot = gpu_block_reduction_write_slot(memlet.subset, cover['base'], cover['m'])
            if slot is not None:
                lhs = f"{cover['partial']}[{sym2cpp(slot)}]"
                # Vector value, scalar partial: horizontal fold first, as the atomic path below does.
                if isinstance(dtype, dtypes.vector):
                    return (f"dace::wcr_fixed<{cover['credtype']}, {cover['ctype']}>::"
                            f"vreduce<{dtype.veclen}>(&{lhs}, {inname})")
                return f"{lhs} = dace::_wcr_fixed<{cover['credtype']}, {cover['ctype']}>()({lhs}, {inname})"
        # Skip the atomic call entirely when an enclosing OMP map has put this
        # target in a ``reduction(...)`` clause -- the OMP runtime privatizes
        # the variable per thread and tree-reduces at the end, so adding an
        # atomic on top is strictly wasted work.
        _omp_covered = any(memlet.data in frame for frame in self._omp_reduction_scope_stack)
        atomic = "" if (nc or _omp_covered) else "_atomic"
        wcr_desc = sdfg.arrays[memlet.data]
        ptrname = self.ptr(memlet.data, wcr_desc, sdfg)
        # Readable path inlines pure-Reduce outputs out of their allocating scope, so the target may
        # be declared at a broader scope but absent from the live defined_vars stack at the write
        # site; resolve from declared_arrays first, falling back to defined_vars. Legacy never inlines
        # reductions, so it keeps the original defined_vars lookup -- byte-identical to before.
        if cpp.readable_cpu_codegen_active():
            is_global = wcr_desc.lifetime in (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.Persistent,
                                              dtypes.AllocationLifetime.External)
            try:
                defined_type, _ = self._dispatcher.declared_arrays.get(ptrname, is_global=is_global)
            except KeyError:
                defined_type, _ = self._dispatcher.defined_vars.get(ptrname, is_global=is_global)
        else:
            defined_type, _ = self._dispatcher.defined_vars.get(ptrname)
        if isinstance(indices, str):
            ptr = '%s + %s' % (cpp.cpp_ptr_expr(sdfg, memlet, defined_type, codegen=self), indices)
        else:
            ptr = cpp.cpp_ptr_expr(sdfg, memlet, defined_type, indices=indices, codegen=self)

        # An OMP ``reduction(op:var)`` clause target: the clause privatizes the scalar
        # per thread and tree-reduces at the end, so the body accumulates into the private
        # copy directly. For an op with a plain C++ infix operator (``+``/``*``/bitwise/
        # logical) emit that -- no ``wcr_fixed`` helper, which is redundant machinery on an
        # already-privatized scalar. ``min`` / ``max`` have no infix operator, so they fall
        # through to the ``wcr_fixed<...>::reduce`` emission below, which is already the
        # NON-atomic form here (``atomic`` was cleared to "" for an omp-covered target) and
        # matches the reduction runtime's ``::min`` / ``::max`` bit-for-bit.
        if _omp_covered:
            omp_op = _REDUCTION_TO_OMP_OP.get(redtype)
            if omp_op in ("+", "*", "&", "|", "^", "&&", "||"):
                return f'*({ptr}) = *({ptr}) {omp_op} ({inname})'

        if isinstance(dtype, dtypes.pointer):
            dtype = dtype.base_type

        # If there is a type mismatch and more than one element is used, cast
        # pointer (vector->vector WCR). Otherwise, generate vector->scalar
        # (horizontal) reduction.
        vec_prefix = ''
        vec_suffix = ''
        dst_dtype = sdfg.arrays[memlet.data].dtype
        if (isinstance(dtype, dtypes.vector) and not isinstance(dst_dtype, dtypes.vector)):
            if memlet.subset.num_elements() != 1:
                ptr = f'({dtype.ctype} *)({ptr})'
            else:
                vec_prefix = 'v'
                vec_suffix = f'<{dtype.veclen}>'
                dtype = dtype.base_type
        func = f'{vec_prefix}reduce{atomic}{vec_suffix}'

        # Special call for detected reduction types
        if redtype != dtypes.ReductionType.Custom:
            credtype = "dace::ReductionType::" + str(redtype)[str(redtype).find(".") + 1:]
            return (f'dace::wcr_fixed<{credtype}, {dtype.ctype}>::{func}({ptr}, {inname})')

        # General reduction
        custom_reduction = cpp.unparse_cr(sdfg, memlet.wcr, dtype)
        return (
            f'const auto __dace__reduction_lambda = {custom_reduction};\ndace::wcr_custom<{dtype.ctype}>::{func}<decltype(__dace__reduction_lambda)>(__dace__reduction_lambda, {ptr}, {inname})'
        )

    def process_out_memlets(self,
                            sdfg: SDFG,
                            cfg: ControlFlowRegion,
                            state_id: int,
                            node: nodes.Node,
                            dfg: StateSubgraphView,
                            dispatcher: TargetDispatcher,
                            result: CodeIOStream,
                            locals_defined: bool,
                            function_stream: CodeIOStream,
                            skip_wcr: bool = False,
                            codegen: Optional[TargetCodeGenerator] = None):
        codegen = codegen if codegen is not None else self
        state: SDFGState = cfg.nodes()[state_id]
        scope_dict = state.scope_dict()

        for edge in dfg.out_edges(node):
            _, uconn, v, _, memlet = edge
            if skip_wcr and memlet.wcr is not None:
                continue
            dst_edge = dfg.memlet_path(edge)[-1]
            dst_node = dst_edge.dst

            if isinstance(dst_node, nodes.AccessNode) and dst_node.desc(state).dtype == dtypes.gpuStream_t:
                # Special case: GPU Streams do not represent data flow - they assing GPU Streams to kernels/tasks
                # Thus, nothing needs to be written and out memlets of this kind should be ignored.
                continue

            # Target is neither a data nor a tasklet node
            if isinstance(node, nodes.AccessNode) and (not isinstance(dst_node, nodes.AccessNode)
                                                       and not isinstance(dst_node, nodes.CodeNode)):
                continue

            # Skip array->code (will be handled as a tasklet input)
            if isinstance(node, nodes.AccessNode) and isinstance(v, nodes.CodeNode):
                continue

            # code->code (e.g., tasklet to tasklet)
            if isinstance(dst_node, nodes.CodeNode) and edge.src_conn:
                shared_data_name = edge.data.data
                if not shared_data_name:
                    # Very unique name. TODO: Make more intuitive
                    shared_data_name = '__dace_%d_%d_%d_%d_%s' % (cfg.cfg_id, state_id, dfg.node_id(node),
                                                                  dfg.node_id(dst_node), edge.src_conn)

                result.write(
                    "%s = %s;" % (shared_data_name, edge.src_conn),
                    cfg,
                    state_id,
                    [edge.src, edge.dst],
                )
                continue

            # If the memlet is not pointing to a data node (e.g. tasklet), then
            # the tasklet will take care of the copy
            if not isinstance(dst_node, nodes.AccessNode):
                continue
            # If the memlet is pointing into an array in an inner scope, then
            # the inner scope (i.e., the output array) must handle it
            if scope_dict[node] != scope_dict[dst_node] and scope_contains_scope(scope_dict, node, dst_node):
                continue

            # Array to tasklet (path longer than 1, handled at tasklet entry)
            if node == dst_node:
                continue

            # Tasklet -> array with a memlet. Writing to array is emitted only if the memlet is not empty
            if isinstance(node, nodes.CodeNode) and not edge.data.is_empty():
                if uconn and not self._connector_needs_copy(node, uconn):
                    # Inlined by InlineTaskletConnectors: the body writes the array
                    # directly, so no copy-out is emitted.
                    continue
                if not uconn:
                    continue

                conntype = node.out_connectors[uconn]
                is_scalar = not isinstance(conntype, dtypes.pointer)
                if isinstance(conntype, dtypes.pointer) and sdfg.arrays[memlet.data].dtype == conntype:
                    is_scalar = True  # Pointer to pointer assignment
                is_stream = isinstance(sdfg.arrays[memlet.data], data.Stream)
                is_refset = isinstance(sdfg.arrays[memlet.data], data.Reference) and dst_edge.dst_conn == 'set'

                if (is_scalar and not memlet.dynamic and not is_stream) or is_refset:
                    out_local_name = "    __" + uconn
                    in_local_name = uconn
                    if not locals_defined:
                        out_local_name = codegen.memlet_ctor(sdfg, memlet, node.out_connectors[uconn], True)
                        in_memlets = [d for _, _, _, _, d in dfg.in_edges(node)]
                        assert len(in_memlets) == 1
                        in_local_name = codegen.memlet_ctor(sdfg, in_memlets[0], node.out_connectors[uconn], False)

                    if memlet.wcr is not None:
                        nc = not cpp.is_write_conflicted(dfg, edge, sdfg_schedule=self._toplevel_schedule)
                        write_expr = codegen.write_and_resolve_expr(
                            sdfg, memlet, nc, out_local_name, in_local_name, dtype=node.out_connectors[uconn]) + ";"
                    else:
                        if isinstance(node, nodes.NestedSDFG):
                            # This case happens with nested SDFG outputs,
                            # which we skip since the memlets are references
                            continue
                        desc = sdfg.arrays[memlet.data]
                        ptrname = codegen.ptr(memlet.data, desc, sdfg)
                        is_global = desc.lifetime in (dtypes.AllocationLifetime.Global,
                                                      dtypes.AllocationLifetime.Persistent,
                                                      dtypes.AllocationLifetime.External)
                        try:
                            defined_type, _ = self._dispatcher.declared_arrays.get(ptrname, is_global=is_global)
                        except KeyError:
                            defined_type, _ = self._dispatcher.defined_vars.get(ptrname, is_global=is_global)

                        if defined_type == DefinedType.Scalar:
                            mname = codegen.ptr(memlet.data, desc, sdfg)
                            write_expr = f"{mname} = {in_local_name};"
                        elif defined_type == DefinedType.Pointer and is_refset:
                            mname = codegen.ptr(memlet.data, desc, sdfg)
                            write_expr = f"{mname} = {in_local_name};"
                        else:
                            desc_dtype = desc.dtype
                            expr = cpp.cpp_array_expr(sdfg, memlet, codegen=codegen)
                            write_expr = codegen.make_ptr_assignment(in_local_name, conntype, expr, desc_dtype)

                    # Write out
                    result.write(write_expr, cfg, state_id, node)

            # Dispatch array-to-array outgoing copies here
            elif isinstance(node, nodes.AccessNode):
                if dst_node != node and not isinstance(dst_node, nodes.Tasklet):
                    dispatcher.dispatch_copy(
                        node,
                        dst_node,
                        edge,
                        sdfg,
                        cfg,
                        dfg,
                        state_id,
                        function_stream,
                        result,
                    )

    def make_ptr_assignment(self, src_expr, src_dtype, dst_expr, dst_dtype, codegen=None):
        """
        Write source to destination, where the source is a scalar, and the
        destination is a pointer.

        :return: String of C++ performing the write.
        """
        codegen = codegen or self
        # If there is a type mismatch, cast pointer
        dst_expr = codegen.make_ptr_vector_cast(dst_expr, dst_dtype, src_dtype, True, DefinedType.Pointer)
        return f"{dst_expr} = {src_expr};"

    def memlet_view_ctor(self, sdfg: SDFG, memlet: mmlt.Memlet, dtype, is_output: bool) -> str:
        memlet_params = []

        memlet_name = self.ptr(memlet.data, sdfg.arrays[memlet.data], sdfg)
        def_type, _ = self._dispatcher.defined_vars.get(memlet_name)

        if def_type == DefinedType.Pointer:
            memlet_expr = memlet_name  # Common case
        elif def_type == DefinedType.Scalar:
            memlet_expr = "&" + memlet_name
        else:
            raise TypeError("Unsupported connector type {}".format(def_type))

        if isinstance(memlet.subset, subsets.Indices):
            offset = cpp.cpp_array_expr(sdfg, memlet, False, codegen=self)

            # Compute address
            memlet_params.append(memlet_expr + " + " + offset)
            dims = 0

        else:

            if isinstance(memlet.subset, subsets.Range):

                dims = len(memlet.subset.ranges)
                offset = cpp.cpp_offset_expr(sdfg.arrays[memlet.data], memlet.subset)
                if offset == "0":
                    memlet_params.append(memlet_expr)
                else:
                    if def_type != DefinedType.Pointer:
                        raise cgx.CodegenError("Cannot offset address of connector {} of type {}".format(
                            memlet_name, def_type))
                    memlet_params.append(memlet_expr + " + " + offset)

                # Dimensions to remove from view (due to having one value)
                indexdims = []
                strides = sdfg.arrays[memlet.data].strides

                # Figure out dimensions for scalar version
                dimlen = dtype.veclen if isinstance(dtype, dtypes.vector) else 1
                for dim, (rb, re, rs) in enumerate(memlet.subset.ranges):
                    try:
                        # Check for number of elements in contiguous dimension
                        # (with respect to vector length)
                        if strides[dim] == 1 and (re - rb) == dimlen - 1:
                            indexdims.append(dim)
                        elif (re - rb) == 0:  # Elements in other dimensions
                            indexdims.append(dim)
                    except TypeError:
                        # Cannot determine truth value of Relational
                        pass

                # Remove index (one scalar) dimensions
                dims -= len(indexdims)

                if dims > 0:
                    strides = memlet.subset.absolute_strides(strides)
                    # Filter out index dims
                    strides = [s for i, s in enumerate(strides) if i not in indexdims]
                    # Use vector length to adapt strides
                    for i in range(len(strides) - 1):
                        strides[i] /= dimlen
                    memlet_params.extend(sym2cpp(strides))
                    dims = memlet.subset.data_dims()

            else:
                raise RuntimeError('Memlet type "%s" not implemented' % memlet.subset)

        # If there is a type mismatch, cast pointer (used in vector
        # packing/unpacking)
        if dtype != sdfg.arrays[memlet.data].dtype:
            memlet_params[0] = '(%s *)(%s)' % (dtype.ctype, memlet_params[0])

        return "dace::ArrayView%s<%s, %d, 1, 1> (%s)" % (
            "Out" if is_output else "In",
            dtype.ctype,
            dims,
            ", ".join(memlet_params),
        )

    def memlet_definition(self,
                          sdfg: SDFG,
                          memlet: mmlt.Memlet,
                          output: bool,
                          local_name: str,
                          conntype: Union[data.Data, dtypes.typeclass] = None,
                          allow_shadowing: bool = False,
                          codegen: Optional['CPUCodeGen'] = None):
        # TODO: Robust rule set
        if conntype is None:
            raise ValueError('Cannot define memlet for "%s" without connector type' % local_name)
        codegen = codegen or self
        # Convert from Data to typeclass
        if isinstance(conntype, data.Data):
            if isinstance(conntype, data.Array):
                conntype = dtypes.pointer(conntype.dtype)
            else:
                conntype = conntype.dtype

        desc = sdfg.arrays[memlet.data]

        is_scalar = not isinstance(conntype, dtypes.pointer) or desc.dtype == conntype
        is_pointer = isinstance(conntype, dtypes.pointer)

        # Allocate variable type
        memlet_type = conntype.dtype.ctype

        ptr = codegen.ptr(memlet.data, desc, sdfg)
        types = None
        # Non-free symbol dependent Arrays due to their shape
        dependent_shape = (isinstance(desc, data.Array) and not isinstance(desc, data.View) and any(
            str(s) not in self._frame.symbols_and_constants(sdfg) for s in self._frame.free_symbols(desc)))
        try:
            # NOTE: It is hard to get access to the view-edge here, so always
            # check the declared-arrays dictionary for Views.
            if dependent_shape or isinstance(desc, data.View):
                types = self._dispatcher.declared_arrays.get(ptr)
        except KeyError:
            pass
        if not types:
            types = self._dispatcher.defined_vars.get(ptr, is_global=True)
        var_type, ctypedef = types

        result = ''
        expr = (cpp.cpp_array_expr(sdfg, memlet, with_brackets=False, codegen=self)
                if var_type in (DefinedType.Pointer, DefinedType.StreamArray) else ptr)

        if expr != ptr:
            expr = '%s[%s]' % (ptr, expr)
        # If there is a type mismatch, cast pointer
        expr = codegen.make_ptr_vector_cast(expr, desc.dtype, conntype, is_scalar, var_type)

        defined = None

        if var_type in (DefinedType.Scalar, DefinedType.Pointer):
            if output:
                if not memlet.dynamic or (memlet.dynamic and memlet.wcr is not None):
                    # Dynamic WCR memlets start uninitialized
                    result += "{} {};".format(memlet_type, local_name)
                    defined = DefinedType.Scalar
            else:
                if not memlet.dynamic:
                    if is_scalar:
                        # We can pre-read the value
                        result += "{} {} = {};".format(memlet_type, local_name, expr)
                    else:
                        # constexpr arrays
                        if memlet.data in self._frame.symbols_and_constants(sdfg):
                            result += "const {} {} = {};".format(memlet_type, local_name, expr)
                        elif (var_type == DefinedType.Scalar and isinstance(conntype, dtypes.pointer)
                              and not isinstance(desc.dtype, dtypes.opaque)):
                            # Scalar source feeding a pointer-typed connector
                            # (e.g. CopyLibraryNode -> cudaMemcpyAsync from a host
                            # scalar argument). The connector's pointer type wins
                            # over the source's scalar ctypedef, and we have to
                            # take the address of the host variable. Skip for
                            # opaque dtypes (MPI_Comm / MPI_Request / cuda handles
                            # etc.) -- the value is already a pointer-like handle,
                            # so address-of would add an unwanted indirection
                            # that breaks the libnode call (e.g. ``MPI_Bcast``
                            # expects ``MPI_Comm``, not ``MPI_Comm *``).
                            result += "{} {} = &{};".format(conntype.ctype, local_name, expr)
                        else:
                            # Pointer reference. ``ctypedef`` may already include
                            # ``__restrict__`` (from a parent scope's Scalar->Pointer
                            # registration in :func:`emit_memlet_reference` ~line 337),
                            # so don't append a second one — gcc rejects duplicate
                            # cv-qualifiers.
                            pruned_expr = replace_float_literals(expr)
                            _restrict = "" if "__restrict__" in ctypedef else "__restrict__ "
                            result += "{} {}{} = {};".format(ctypedef, _restrict, local_name, pruned_expr)
                else:
                    # Variable number of reads: get a const reference that can
                    # be read if necessary.
                    memlet_type = 'const %s' % memlet_type
                    if is_pointer:
                        _restrict = "" if "__restrict__" in memlet_type else "__restrict__ "
                        result += "{} {}{} = {};".format(memlet_type, _restrict, local_name, expr)
                    else:
                        result += "{} &{} = {};".format(memlet_type, local_name, expr)
                defined = (DefinedType.Scalar if is_scalar else DefinedType.Pointer)
        elif var_type in [DefinedType.Stream, DefinedType.StreamArray]:
            if not memlet.dynamic and memlet.num_accesses == 1:
                if not output:
                    if isinstance(desc, data.Stream) and desc.is_stream_array():
                        index = cpp.cpp_offset_expr(desc, memlet.subset)
                        expr = f"{memlet.data}[{index}]"
                    result += f'{memlet_type} {local_name} = ({expr}).pop();'
                    defined = DefinedType.Scalar
            else:
                # Just forward actions to the underlying object
                memlet_type = ctypedef
                result += "{} &{} = {};".format(memlet_type, local_name, expr)
                defined = DefinedType.Stream

        # Set Defined Type for GPU Stream connectors
        # Shadowing for stream variable needs to be allowed
        if memlet_type == 'gpuStream_t':
            var_type = DefinedType.GPUStream
            defined = DefinedType.GPUStream

        if defined is not None:
            self._dispatcher.defined_vars.add(local_name, defined, memlet_type, allow_shadowing=allow_shadowing)

        return result

    def memlet_stream_ctor(self, sdfg: SDFG, memlet: mmlt.Memlet) -> str:
        stream = sdfg.arrays[memlet.data]
        return memlet.data + ("[{}]".format(cpp.cpp_offset_expr(stream, memlet.subset))
                              if isinstance(stream, data.Stream) and stream.is_stream_array() else "")

    def memlet_ctor(self, sdfg: SDFG, memlet: mmlt.Memlet, dtype, is_output: bool) -> str:
        ptrname = self.ptr(memlet.data, sdfg.arrays[memlet.data], sdfg)
        def_type, _ = self._dispatcher.defined_vars.get(ptrname)

        if def_type in [DefinedType.Stream, DefinedType.Object, DefinedType.StreamArray]:
            return self.memlet_stream_ctor(sdfg, memlet)

        elif def_type in [DefinedType.Pointer, DefinedType.Scalar]:
            return self.memlet_view_ctor(sdfg, memlet, dtype, is_output)

        else:
            raise NotImplementedError("Connector type {} not yet implemented".format(def_type))

    #########################################################################
    # Dynamically-called node dispatchers

    def tasklet_body_comment(self, node: nodes.Tasklet) -> str:
        """Comment above a tasklet's unparsed body (overridable; the readable generator drops it)."""
        return "// Tasklet code (%s)\n" % node.label

    def tasklet_body_open_marker(self, node: nodes.Tasklet) -> str:
        """Separator before a tasklet's unparsed body (overridable; the readable generator drops it)."""
        return "\n    ///////////////////\n"

    def tasklet_body_close_marker(self, node: nodes.Tasklet) -> str:
        """Separator after a tasklet's unparsed body."""
        return "    ///////////////////\n\n"

    def emit_tasklet_body_block(self, callsite_stream: CodeIOStream, cfg: ControlFlowRegion, state_id: int,
                                node: nodes.Tasklet, inner_body: str, postamble: str, has_locals: bool) -> None:
        """Emit a tasklet body in its own C++ scope block (overridable; the readable generator collapses
        a connector-free single-statement tasklet onto one brace-free line). ``has_locals`` is True when
        copy-in/out or code->code locals were declared, forcing the block."""
        callsite_stream.write('{', cfg, state_id, node)
        callsite_stream.write(inner_body, cfg, state_id, node)
        callsite_stream.write(postamble)
        callsite_stream.write('}', cfg, state_id, node)

    def _generate_Tasklet(self,
                          sdfg: SDFG,
                          cfg: ControlFlowRegion,
                          dfg: StateSubgraphView,
                          state_id: int,
                          node: nodes.Tasklet,
                          function_stream: CodeIOStream,
                          callsite_stream: CodeIOStream,
                          codegen=None):

        # Allow other code generators to call this with a callback
        codegen = codegen or self

        outer_stream_begin = CodeIOStream()
        outer_stream_end = CodeIOStream()
        inner_stream = CodeIOStream()

        # Add code to init and exit functions
        self._frame._initcode.write(codeblock_to_cpp(node.code_init), sdfg)
        self._frame._exitcode.write(codeblock_to_cpp(node.code_exit), sdfg)

        state_dfg: SDFGState = cfg.nodes()[state_id]

        # Free tasklets need to be presynchronized (e.g., CPU tasklet after
        # GPU->CPU copy)
        if state_dfg.entry_node(node) is None:
            cpp.presynchronize_streams(sdfg, cfg, state_dfg, state_id, node, callsite_stream)

        # Prepare preamble and code for after memlets
        after_memlets_stream = CodeIOStream()
        codegen.generate_tasklet_preamble(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream,
                                          after_memlets_stream)

        self._dispatcher.defined_vars.enter_scope(node)

        arrays = set()
        for edge in state_dfg.in_edges(node):
            u = edge.src
            memlet = edge.data
            src_node = state_dfg.memlet_path(edge)[0].src

            if edge.dst_conn:  # Not (None or "")
                if not self._connector_needs_copy(node, edge.dst_conn):
                    # Inlined by InlineTaskletConnectors: the body accesses the
                    # array directly, so no copy-in temporary is emitted.
                    continue
                if edge.dst_conn in arrays:  # Disallow duplicates
                    raise SyntaxError("Duplicates found in memlets")
                ctype = node.in_connectors[edge.dst_conn].ctype
                # Special case: code->code
                if isinstance(src_node, nodes.CodeNode):
                    shared_data_name = edge.data.data
                    if not shared_data_name:
                        # Very unique name. TODO: Make more intuitive
                        shared_data_name = '__dace_%d_%d_%d_%d_%s' % (cfg.cfg_id, state_id, dfg.node_id(src_node),
                                                                      dfg.node_id(node), edge.src_conn)

                    # Read variable from shared storage
                    defined_type, _ = self._dispatcher.defined_vars.get(shared_data_name)
                    if defined_type in (DefinedType.Scalar, DefinedType.Pointer):
                        assign_str = (f"const {ctype} {edge.dst_conn} = {shared_data_name};")
                    else:
                        assign_str = (f"const {ctype} &{edge.dst_conn} = {shared_data_name};")
                    inner_stream.write(assign_str, cfg, state_id, [edge.src, edge.dst])
                    self._dispatcher.defined_vars.add(edge.dst_conn, defined_type, f"const {ctype}")

                else:
                    self._dispatcher.dispatch_copy(
                        src_node,
                        node,
                        edge,
                        sdfg,
                        cfg,
                        dfg,
                        state_id,
                        function_stream,
                        inner_stream,
                    )

                # Also define variables in the C++ unparser scope
                self._locals.define(edge.dst_conn, -1, self._ldepth + 1, ctype)
                arrays.add(edge.dst_conn)

        # Use outgoing edges to preallocate output local vars
        # in two stages: first we preallocate for data<->code cases,
        # followed by code<->code
        tasklet_out_connectors = set()
        locals_defined = False
        for edge in state_dfg.out_edges(node):
            dst_node = state_dfg.memlet_path(edge)[-1].dst
            if isinstance(dst_node, nodes.CodeNode):
                # Handling this in a separate pass just below
                continue

            if edge.src_conn:
                if not self._connector_needs_copy(node, edge.src_conn):
                    # Inlined by InlineTaskletConnectors: the body writes the
                    # array directly, so no out-connector temporary is declared.
                    continue
                if edge.src_conn in tasklet_out_connectors:  # Disallow duplicates
                    continue

                self._dispatcher.dispatch_output_definition(node, dst_node, edge, sdfg, cfg, dfg, state_id,
                                                            function_stream, inner_stream)

                # Also define variables in the C++ unparser scope
                self._locals.define(edge.src_conn, -1, self._ldepth + 1, node.out_connectors[edge.src_conn].ctype)
                tasklet_out_connectors.add(edge.src_conn)

        for edge in state_dfg.out_edges(node):
            # Special case: code->code
            dst_node = state_dfg.memlet_path(edge)[-1].dst
            if edge.src_conn is None:
                continue
            if not self._connector_needs_copy(node, edge.src_conn):
                continue
            cdtype = node.out_connectors[edge.src_conn]
            ctype = cdtype.ctype
            # Convert dtype to data descriptor
            if isinstance(cdtype, dtypes.pointer):
                arg_type = data.Array(cdtype._typeclass, [1])
            else:
                arg_type = data.Scalar(cdtype)

            if (isinstance(dst_node, nodes.CodeNode) and edge.src_conn not in tasklet_out_connectors):
                memlet = edge.data

                # Generate register definitions for inter-tasklet memlets
                local_name = edge.data.data
                if not local_name:
                    # Very unique name. TODO: Make more intuitive
                    local_name = '__dace_%d_%d_%d_%d_%s' % (cfg.cfg_id, state_id, dfg.node_id(node),
                                                            dfg.node_id(dst_node), edge.src_conn)

                # Allocate variable type
                code = "%s %s;" % (ctype, local_name)
                outer_stream_begin.write(code, cfg, state_id, [edge.src, dst_node])
                if (isinstance(arg_type, data.Scalar) or isinstance(arg_type, dtypes.typeclass)):
                    self._dispatcher.defined_vars.add(local_name, DefinedType.Scalar, ctype, ancestor=1)
                elif isinstance(arg_type, data.Array):
                    self._dispatcher.defined_vars.add(local_name, DefinedType.Pointer, ctype, ancestor=1)
                elif isinstance(arg_type, data.Stream):
                    if arg_type.is_stream_array():
                        self._dispatcher.defined_vars.add(local_name, DefinedType.StreamArray, ctype, ancestor=1)
                    else:
                        self._dispatcher.defined_vars.add(local_name, DefinedType.Stream, ctype, ancestor=1)
                else:
                    raise TypeError("Unrecognized argument type: {}".format(type(arg_type).__name__))

                inner_stream.write("%s %s;" % (ctype, edge.src_conn), cfg, state_id, [edge.src, edge.dst])
                tasklet_out_connectors.add(edge.src_conn)
                self._dispatcher.defined_vars.add(edge.src_conn, DefinedType.Scalar, ctype)
                self._locals.define(edge.src_conn, -1, self._ldepth + 1, ctype)
                locals_defined = True

        # Emit post-memlet tasklet preamble code
        callsite_stream.write(after_memlets_stream.getvalue())

        # Instrumentation: Pre-tasklet. Fall back to the enclosing state's
        # ``instrument`` flag if the node itself wasn't tagged -- this makes
        # state-level annotations (e.g. ``GPU_TX_MARKERS`` on a copyin
        # state) surface for tasklets generated by library-node expansions
        # (CopyLibraryNode -> cudaMemcpyAsync) which don't carry their own
        # instrument attribute. The provider's hook can still filter by
        # node identity / label.
        instr_type = node.instrument
        if (instr_type == dtypes.InstrumentationType.No_Instrumentation
                and getattr(state_dfg, 'instrument', dtypes.InstrumentationType.No_Instrumentation)
                != dtypes.InstrumentationType.No_Instrumentation):
            instr_type = state_dfg.instrument
        instr = self._dispatcher.instrumentation.get(instr_type)
        if instr is not None:
            instr.on_node_begin(sdfg, cfg, state_dfg, node, outer_stream_begin, inner_stream, function_stream)

        inner_stream.write(codegen.tasklet_body_open_marker(node), cfg, state_id, node)

        codegen.unparse_tasklet(sdfg, cfg, state_id, dfg, node, function_stream, inner_stream, self._locals,
                                self._ldepth, self._toplevel_schedule)

        inner_stream.write(codegen.tasklet_body_close_marker(node), cfg, state_id, node)

        # Generate pre-memlet tasklet postamble
        after_memlets_stream = CodeIOStream()
        codegen.generate_tasklet_postamble(sdfg, cfg, dfg, state_id, node, function_stream, inner_stream,
                                           after_memlets_stream)

        # Process outgoing memlets
        codegen.process_out_memlets(
            sdfg,
            cfg,
            state_id,
            node,
            dfg,
            self._dispatcher,
            inner_stream,
            True,
            function_stream,
        )

        # Instrumentation: Post-tasklet
        if instr is not None:
            instr.on_node_end(sdfg, cfg, state_dfg, node, outer_stream_end, inner_stream, function_stream)

        callsite_stream.write(outer_stream_begin.getvalue(), cfg, state_id, node)
        # A tasklet with no copy-in/out temporaries and no code->code locals can be
        # emitted without its own scope block (used by the readable code generator to
        # collapse a connector-free element-wise tasklet onto a single line).
        has_locals = (bool(arrays) or bool(tasklet_out_connectors) or locals_defined
                      or bool(after_memlets_stream.getvalue().strip()))
        codegen.emit_tasklet_body_block(callsite_stream, cfg, state_id, node, inner_stream.getvalue(),
                                        after_memlets_stream.getvalue(), has_locals)
        callsite_stream.write(outer_stream_end.getvalue(), cfg, state_id, node)

        self._locals.clear_scope(self._ldepth + 1)
        self._dispatcher.defined_vars.exit_scope(node)

    def unparse_tasklet(self, sdfg, cfg, state_id, dfg, node, function_stream, inner_stream, locals, ldepth,
                        toplevel_schedule):
        # Call the generic CPP unparse_tasklet method
        cpp.unparse_tasklet(sdfg, cfg, state_id, dfg, node, function_stream, inner_stream, locals, ldepth,
                            toplevel_schedule, self)

    def make_keyword_remover(self, sdfg, memlets):
        """AST transformer used to lower a Python tasklet body to C++. A hook so ``cpp.unparse_tasklet``
        does not hard-code the class; the readable generator overrides it to also inline array accesses."""
        return cpp.DaCeKeywordRemover(sdfg, memlets, sdfg.constants, self)

    def _connector_needs_copy(self, node, conn):
        """Whether a tasklet connector needs a copy-in/out temporary. Always True here; the readable
        generator returns False for connectors InlineTaskletConnectors rewrote into direct accesses."""
        return True

    def fused_heap_declarator(self, sdfg: SDFG, name: str, nodedesc: data.Data, arrsize, declared: bool,
                              declaration_stream: CodeIOStream, allocation_stream: CodeIOStream) -> Optional[str]:
        """Declarator that fuses the heap declaration into the allocation statement, or None to keep
        the classic split ``T *name;`` + ``name = new T[...];`` pair.

        Returning a declarator (e.g. ``T* __restrict__ name``) makes :meth:`heap_alloc_stmt` emit a
        single definition instead of an assignment to a previously-declared pointer. The base
        generator never fuses -- its output is the reference the readable generator is compared
        against -- so this returns None; :class:`~dace.codegen.targets.experimental_cpu.
        ExperimentalCPUCodeGen` overrides it and documents when fusing is sound."""
        return None

    def heap_alloc_stmt(self,
                        alloc_name: str,
                        ctype: str,
                        arrsize: str,
                        alignment: int = 0,
                        sdfg: Optional[SDFG] = None,
                        nodedesc: Optional[data.Data] = None,
                        data_name: Optional[str] = None) -> str:
        """C++ statement allocating a CPU heap array with aligned ``new[]`` (paired with the
        ``delete[]`` in heap_free_stmt). ``alloc_name`` is the assignment target: either a plain
        pointer name (the classic split form) or a full declarator (see fused_heap_declarator), in
        which case the emitted statement is a definition. The trailing ``sdfg``/``nodedesc``/
        ``data_name`` are unused here; the readable generator overrides this to route the count
        through an ``<array>_size`` helper."""
        return "%s = new %s DACE_ALIGN(64)[%s];\n" % (alloc_name, ctype, arrsize)

    def heap_free_stmt(self, alloc_name: str, is_array: bool) -> str:
        """ C++ statement freeing a CPU heap array (paired with heap_alloc_stmt). """
        return ("delete[] %s;\n" if is_array else "delete %s;\n") % alloc_name

    def rewrite_cpp_tasklet_body(self, node, sdfg, state_dfg):
        """C++ body of a native (C++/library) tasklet as it should be emitted. Verbatim here; the
        readable generator overrides this to inline connector accesses (direct array/base-pointer)."""
        return type(node).__properties__["code"].to_string(node.code)

    def define_out_memlet(self, sdfg: SDFG, cfg: ControlFlowRegion, state_dfg: StateSubgraphView, state_id: int,
                          src_node: nodes.Node, dst_node: nodes.Node, edge: MultiConnectorEdge[mmlt.Memlet],
                          function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        cdtype = src_node.out_connectors[edge.src_conn]
        if isinstance(sdfg.arrays[edge.data.data], data.Stream):
            pass
        elif isinstance(dst_node, nodes.AccessNode) and dst_node.desc(state_dfg).dtype == dtypes.gpuStream_t:
            # Special case: GPU Streams do not represent data flow - they assing GPU Streams to kernels/tasks
            # Thus, nothing needs to be written.
            pass
        elif isinstance(cdtype, dtypes.pointer):  # If pointer, also point to output
            desc = sdfg.arrays[edge.data.data]

            # If reference set, do not emit initial assignment
            is_refset = isinstance(desc, data.Reference) and state_dfg.memlet_path(edge)[-1].dst_conn == 'set'

            if not is_refset and not isinstance(desc.dtype, dtypes.pointer):
                ptrname = self.ptr(edge.data.data, desc, sdfg)
                is_global = desc.lifetime in (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.Persistent,
                                              dtypes.AllocationLifetime.External)
                # A shared transient is declared at SDFG scope (in ``declared_arrays``) but its
                # ``define_var`` runs in the allocating state's scope, which is popped before a
                # pointer-write in a later state / nested control-flow region. Resolve via
                # ``declared_arrays`` first -- mirroring the normal write path in
                # ``process_out_memlets`` -- and only fall back to ``defined_vars`` so the
                # SDFG-scope pointer still resolves.
                try:
                    defined_type, _ = self._dispatcher.declared_arrays.get(ptrname, is_global=is_global)
                except KeyError:
                    defined_type, _ = self._dispatcher.defined_vars.get(ptrname, is_global=is_global)
                base_ptr = cpp.cpp_ptr_expr(sdfg, edge.data, defined_type, codegen=self)
                base_ptr = replace_float_literals(base_ptr)
                callsite_stream.write(f'{cdtype.ctype} __restrict__ {edge.src_conn} = {base_ptr};', cfg, state_id,
                                      src_node)
            else:
                callsite_stream.write(f'{cdtype.as_arg(edge.src_conn)};', cfg, state_id, src_node)
        else:
            callsite_stream.write(f'{cdtype.ctype} {edge.src_conn};', cfg, state_id, src_node)

    def generate_nsdfg_header(self, sdfg, cfg, state, state_id, node, memlet_references, sdfg_label, state_struct=True):
        arguments = []

        if state_struct:
            toplevel_sdfg: SDFG = sdfg.cfg_list[0]
            arguments.append(f'{cpp.mangle_dace_state_struct_name(toplevel_sdfg)} *__state')

        # Add "__restrict__" keywords to arguments that do not alias with others in the context of this SDFG
        restrict_args = []
        for atype, aname, _ in memlet_references:

            def make_restrict(expr: str) -> str:
                # Check whether "restrict" has already been added before and can be added
                if expr.strip().endswith('*'):
                    return '__restrict__'
                else:
                    return ''

            if aname in node.sdfg.arrays and not node.sdfg.arrays[aname].may_alias:
                restrict_args.append(make_restrict(atype))
            else:
                restrict_args.append('')

        arguments += [
            f'{atype} {restrict} {aname}' for (atype, aname, _), restrict in zip(memlet_references, restrict_args)
        ]
        fsyms = node.sdfg.used_symbols(all_symbols=False, keep_defined_in_mapping=True)
        arguments += [
            f'{node.sdfg.symbols[aname].as_arg(aname)}' for aname in sorted(node.symbol_mapping.keys())
            if aname in fsyms and aname not in sdfg.constants
        ]
        arguments = ', '.join(arguments)
        return f'void {sdfg_label}({arguments}) {{'

    def generate_nsdfg_call(self, sdfg, cfg, state, node, memlet_references, sdfg_label, state_struct=True):
        prepend = []
        if state_struct:
            prepend = ['__state']
        fsyms = node.sdfg.used_symbols(all_symbols=False, keep_defined_in_mapping=True)
        args = ', '.join(prepend + [argval for _, _, argval in memlet_references] + [
            cpp.sym2cpp(symval) for symname, symval in sorted(node.symbol_mapping.items())
            if symname in fsyms and symname not in sdfg.constants
        ])
        return f'{sdfg_label}({args});'

    @staticmethod
    def _mutated_descriptors(nsdfg: SDFG) -> Set[str]:
        """Names of descriptors in ``nsdfg`` that may be mutated -- the set that must *not* be
        ``const``-qualified as a device-function argument.

        Beyond data written directly, this propagates non-const *up* a ``View`` chain: a write
        through a view is recorded by :func:`read_and_write_sets` against the *view's* name, so the
        underlying parent it aliases would otherwise look read-only. A written view exposes its
        parent through a non-const pointer, which is the C++ rule made explicit -- a non-const view
        of const data is illegal, while a const (read-only) view of non-const data is fine, so only
        *written* views taint their parent. Propagation runs to a fixpoint to cover view-of-view.
        """
        mutated: Set[str] = set()
        # Underlying (parent) descriptor of each *write-direction* view -- the data it aliases and
        # writes through. A view's direction is read off its view edge exactly as ``allocate_view``
        # does (``is_write = view_edge.src is view_node``); a read-direction view never writes its
        # parent and so does not taint it. Note: ``read_and_write_sets`` counts the view-*linking*
        # edge as a write to the view itself, so the view's own name being in the write set is not a
        # reliable signal -- the edge direction is.
        write_view_parent: Dict[str, str] = {}
        for nstate in nsdfg.states():
            mutated |= nstate.read_and_write_sets()[1]
            for vn in nstate.nodes():
                if not (isinstance(vn, nodes.AccessNode) and isinstance(nsdfg.arrays[vn.data], data.View)):
                    continue
                view_edge = sdutils.get_view_edge(nstate, vn)
                if view_edge is None or view_edge.src is not vn:
                    continue  # read-direction view (or no view edge) -> does not write its parent
                parent = view_edge.dst
                if isinstance(parent, nodes.AccessNode):
                    write_view_parent[vn.data] = parent.data

        # A write-direction view writes its parent through a non-const pointer, so the parent is
        # mutated. Propagate to a fixpoint so a view-of-view write chain taints the deepest parent.
        changed = True
        while changed:
            changed = False
            for parent in write_view_parent.values():
                if parent not in mutated:
                    mutated.add(parent)
                    changed = True
        return mutated

    def generate_nsdfg_arguments(self, sdfg, cfg, dfg, state, node):
        # Connectors that are both input and output share the same name
        inout = set(node.in_connectors.keys() & node.out_connectors.keys())

        # An input array argument is ``const``-qualifiable only if the callee never mutates its
        # data. ``read_and_write_sets`` records a write against the *written access node's* name,
        # so a write through a ``View`` registers against the view -- not the underlying array.
        # ``_mutated_descriptors`` therefore propagates non-const *up* a view chain (a written
        # view exposes its parent through a non-const pointer), mirroring the C++ rule that a
        # non-const view of const data is illegal while a const view of non-const data is fine.
        written_inside = self._mutated_descriptors(node.sdfg)

        memlet_references = []
        for _, _, _, vconn, in_memlet in sorted(state.in_edges(node), key=lambda e: e.dst_conn or ''):
            if vconn in inout or in_memlet.data is None:
                continue
            const_read_only = vconn not in written_inside
            memlet_references.append(
                cpp.emit_memlet_reference(self._dispatcher,
                                          sdfg,
                                          in_memlet,
                                          vconn,
                                          codegen=self,
                                          is_write=vconn in node.out_connectors,
                                          const_read_only_array=const_read_only,
                                          conntype=node.in_connectors[vconn]))

        for _, uconn, _, _, out_memlet in sorted(state.out_edges(node), key=lambda e: e.src_conn or ''):
            if out_memlet.data is not None:
                memlet_references.append(
                    cpp.emit_memlet_reference(self._dispatcher,
                                              sdfg,
                                              out_memlet,
                                              uconn,
                                              codegen=self,
                                              conntype=node.out_connectors[uconn]))
        return memlet_references

    def _generate_NestedSDFG(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: ScopeSubgraphView,
        state_id: int,
        node: nodes.NestedSDFG,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):
        inline = Config.get_bool('compiler', 'inline_sdfgs')
        state_dfg = cfg.nodes()[state_id]
        # inline_full_array_nsdfg: a CPU-only nest whose connectors are ALL whole outer arrays can be
        # emitted inline via __restrict__ pointer aliases (see the inline branch below) instead of a
        # function. Take the inline path -- which also enters the scopes with can_access_parent=True so
        # the aliased outer pointers resolve. GPU nests (not cpu-only) fall through to do_external.
        do_alias_inline = (not inline and self.calling_codegen is self
                           and Config.get_bool('compiler', 'cpu', 'codegen_params', 'inline_full_array_nsdfg')
                           and self._nsdfg_subtree_is_cpu_only(node.sdfg)
                           and self._nsdfg_connectors_are_full_arrays(sdfg, state_dfg, node))
        if do_alias_inline:
            # Prefer renaming each connector to its outer array's name (no alias needed at all); only a
            # connector that cannot be renamed without a clash keeps its __restrict__ alias below.
            self._rename_full_array_connectors_to_outer(sdfg, state_dfg, node)
            inline = True
        self._dispatcher.defined_vars.enter_scope(sdfg, can_access_parent=inline)
        self._dispatcher.declared_arrays.enter_scope(sdfg, can_access_parent=inline)

        fsyms = self._frame.free_symbols(node.sdfg)
        arglist = node.sdfg.arglist(scalars_only=False, free_symbols=fsyms)
        self._define_sdfg_arguments(node.sdfg, arglist)

        # Quick sanity check.
        # TODO(later): Is this necessary or "can_access_parent" should always be False?
        if inline:
            for nestedarr, ndesc in node.sdfg.arrays.items():
                if (self._dispatcher.defined_vars.has(nestedarr) and ndesc.transient):
                    raise NameError(f'Data name "{nestedarr}" in SDFG "{node.sdfg.name}" '
                                    'already defined in higher scopes and will be shadowed. '
                                    'Please rename or disable inline_sdfgs in the DaCe '
                                    'configuration to compile.')

        # Emit nested SDFG as a separate function
        nested_stream = CodeIOStream()
        nested_global_stream = CodeIOStream()

        unique_functions_conf = Config.get('compiler', 'unique_functions')

        # Backwards compatibility
        if unique_functions_conf is True:
            unique_functions_conf = 'hash'
        elif unique_functions_conf is False:
            unique_functions_conf = 'none'

        if unique_functions_conf == 'hash':
            unique_functions = True
            unique_functions_hash = True
        elif unique_functions_conf == 'unique_name':
            unique_functions = True
            unique_functions_hash = False
        elif unique_functions_conf == 'none':
            unique_functions = False
        else:
            raise ValueError(f'Unknown unique_functions configuration: {unique_functions_conf}')

        if unique_functions and not unique_functions_hash and node.unique_name != "":
            # If the SDFG has a unique name, use it
            sdfg_label = node.unique_name
        else:
            sdfg_label = "%s_%d_%d_%d" % (node.sdfg.name, cfg.cfg_id, state_id, dfg.node_id(node))

        code_already_generated = False
        if unique_functions and not inline:
            hash = node.sdfg.hash_sdfg()
            # Dedup is per OUTPUT FILE, not per whole build: _current_tu_key names the TU being emitted
            # right now (id(self) -- the frame .cpp -- unless split_nsdfg_translation_units re-points it
            # at a nest's own .cpp). Keying on it means an inner nest shared by two split top-level nests
            # is RE-EMITTED into each of their TUs (as an ``inline`` definition, which is ODR-legal across
            # TUs) instead of being emitted into the first and only CALLED -- with no definition -- from
            # the second, which would not link. With the flag off _current_tu_key is invariantly
            # id(self), so this reduces to the old single-key behaviour and the output stays byte-identical.
            if unique_functions_hash:
                # Use hashing to check whether this Nested SDFG has been already generated. If that is the case,
                # use the saved name to call it, otherwise save the hash and the associated name
                if (self._current_tu_key, hash) in self._generated_nested_sdfg:
                    code_already_generated = True
                    sdfg_label = self._generated_nested_sdfg[(self._current_tu_key, hash)]
                else:
                    self._generated_nested_sdfg[(self._current_tu_key, hash)] = sdfg_label
            else:
                # Use the SDFG label to check if this has been already code generated.
                # Check the hash of the formerly generated SDFG to check that we are not
                # generating different SDFGs with the same name
                if (self._current_tu_key, sdfg_label) in self._generated_nested_sdfg:
                    code_already_generated = True
                    if hash != self._generated_nested_sdfg[(self._current_tu_key, sdfg_label)]:
                        raise ValueError(f'Different Nested SDFGs have the same unique name: {sdfg_label}')
                else:
                    self._generated_nested_sdfg[(self._current_tu_key, sdfg_label)] = hash

        #########################################
        # Take care of nested SDFG I/O (arguments)
        # Arguments are input connectors, output connectors, and symbols
        codegen = self.calling_codegen
        memlet_references = codegen.generate_nsdfg_arguments(sdfg, cfg, dfg, state_dfg, node)

        # Emit this nest into its OWN translation unit? Only for a nest that is already a standalone
        # function and whose CONTAINING SDFG is the root (``sdfg.parent is None``) -- so each top-level
        # nest becomes exactly one .cpp carrying everything nested inside it, rather than every nesting
        # level spraying its own file. ``node.no_inline`` ties the split to OutlineTopLevelNests (which
        # marks precisely the nests we want split) and ``codegen is self`` keeps a delegating GPU
        # codegen -- whose ``calling_codegen`` is the GPU generator writing a .cu -- from triggering a
        # host-side split. ``not inline`` because an inlined nest has no function to move.
        do_split = (Config.get_bool('compiler', 'cpu', 'codegen_params', 'split_nsdfg_translation_units')
                    and codegen is self and sdfg.parent is None and not inline and node.no_inline
                    and self._nsdfg_subtree_is_cpu_only(node.sdfg))

        # Model 2 (``external_translation_units``): the COMPLEMENT of do_split -- a top-level nest that
        # DOES contain GPU work is lifted into its own standalone SDFG and called through that SDFG's
        # public handle ABI, so its kernels land in their own ``.cu`` (one generate_code pass per nest)
        # rather than being folded into the parent's. Same enabling conditions as do_split (top-level,
        # standalone no_inline function, host-side codegen), only the CPU-only test is inverted.
        do_external = (Config.get_bool('compiler', 'cpu', 'codegen_params', 'external_translation_units')
                       and codegen is self and sdfg.parent is None and not inline and node.no_inline
                       and not self._nsdfg_subtree_is_cpu_only(node.sdfg))
        if do_external:
            self._emit_external_translation_unit_call(sdfg, node, memlet_references, sdfg_label,
                                                      function_stream, callsite_stream, cfg, state_id)
            self._dispatcher.declared_arrays.exit_scope(sdfg)
            self._dispatcher.defined_vars.exit_scope(sdfg)
            return

        if not inline and (not unique_functions or not code_already_generated):
            # A split nest is DEFINED in its own TU and only DECLARED in the frame's, so it must not be
            # ``inline``: an inline function used in a TU that lacks its definition is ODR-ill-formed
            # (and ``static`` would be worse -- unresolvable across TUs). DACE_HIDDEN gives it external
            # linkage with hidden visibility: the static linker resolves the cross-object call, the
            # symbol stays out of the .so's public ABI, and ThinLTO may still re-inline it.
            qualifier = 'DACE_HIDDEN ' if do_split else ('inline ' if codegen is self else '')
            nested_stream.write(
                qualifier +
                codegen.generate_nsdfg_header(sdfg, cfg, state_dfg, state_id, node, memlet_references, sdfg_label), cfg,
                state_id, node)

        #############################
        # Generate function contents

        if inline:
            callsite_stream.write('{', cfg, state_id, node)
            # inline_full_array_nsdfg: a connector that already shares its outer array's NAME needs no
            # binding at all (the outer pointer is in scope, can_access_parent=True); every other
            # full-array connector is aliased with a single __restrict__ pointer assignment.
            alias_same_name = set()
            if do_alias_inline:
                for e in list(state_dfg.in_edges(node)) + list(state_dfg.out_edges(node)):
                    conn = e.dst_conn if e.dst is node else e.src_conn
                    if conn is not None and e.data is not None and e.data.data == conn:
                        alias_same_name.add(conn)
            for atype, aname, argval in memlet_references:
                if do_alias_inline:
                    if aname in alias_same_name:
                        continue
                    callsite_stream.write('%s __restrict__ %s = %s;' % (atype, aname, argval), cfg, state_id, node)
                else:
                    callsite_stream.write('%s %s = %s;' % (atype, aname, argval), cfg, state_id, node)
            # Emit symbol mappings
            # We first emit variables of the form __dacesym_X = Y to avoid
            # overriding symbolic expressions when the symbol names match
            for symname, symval in sorted(node.symbol_mapping.items()):
                if symname in sdfg.constants:
                    continue
                callsite_stream.write(
                    '{dtype} __dacesym_{symname} = {symval};\n'.format(dtype=node.sdfg.symbols[symname],
                                                                       symname=symname,
                                                                       symval=cpp.sym2cpp(symval)), cfg, state_id, node)
            for symname in sorted(node.symbol_mapping.keys()):
                if symname in sdfg.constants:
                    continue
                callsite_stream.write(
                    '{dtype} {symname} = __dacesym_{symname};\n'.format(symname=symname,
                                                                        dtype=node.sdfg.symbols[symname]), cfg,
                    state_id, node)
            ## End of symbol mappings
            #############################
            nested_stream = callsite_stream
            nested_global_stream = function_stream

        # While generating THIS nest, any per-file bookkeeping belongs to the file the nest is being
        # written into. Only meaningful when the nest is split into its own TU; otherwise the key is
        # unchanged and the subtree keeps writing to the frame TU. Saved/restored around the recursion
        # (rather than reset to a constant) because nests can be generated within nests, and in
        # `finally` so a failure mid-nest cannot strand the key on an abandoned TU.
        outer_tu_key = self._current_tu_key
        if do_split:
            self._current_tu_key = sdfg_label
        try:
            if not unique_functions or not code_already_generated:
                if not inline:
                    self._frame.generate_constants(node.sdfg, nested_stream)

                old_schedule = self._toplevel_schedule

                # Generate code for internal SDFG
                global_code, local_code, used_targets, used_environments = self._frame.generate_code(
                    node.sdfg, old_schedule, sdfg_label)
                self._dispatcher._used_environments |= used_environments

                self._toplevel_schedule = old_schedule

                nested_stream.write(local_code)

                # Process outgoing memlets with the internal SDFG
                codegen.process_out_memlets(sdfg,
                                            cfg,
                                            state_id,
                                            node,
                                            state_dfg,
                                            self._dispatcher,
                                            nested_stream,
                                            True,
                                            nested_global_stream,
                                            skip_wcr=True)

                nested_stream.write('}\n\n', cfg, state_id, node)

            ########################
            if not inline:
                # Generate function call
                callsite_stream.write(
                    codegen.generate_nsdfg_call(sdfg, cfg, state_dfg, node, memlet_references, sdfg_label), cfg,
                    state_id, node)

                ###############################################################
                # Write generated code in the proper places (nested SDFG writes
                # location info)
                if do_split and (not unique_functions or not code_already_generated):
                    # Route the body to this nest's own TU instead of the frame's function stream.
                    unit = CodeIOStream()
                    unit.write(global_code)
                    unit.write(nested_global_stream.getvalue())
                    unit.write(nested_stream.getvalue())
                    self._nsdfg_translation_units[sdfg_label] = (unit.getvalue(), set(used_environments))

                if do_split and sdfg_label in self._nsdfg_translation_units:
                    # The body lives in another TU, so this one gets a forward declaration. Reusing
                    # ``generate_nsdfg_header`` for the prototype makes it match the definition by
                    # construction (same arguments, same order, same __restrict__ qualifiers) -- neither
                    # __restrict__ nor visibility participates in C++ mangling, so the call below resolves
                    # to the definition emitted in the other object.
                    decl = codegen.generate_nsdfg_header(sdfg, cfg, state_dfg, state_id, node, memlet_references,
                                                         sdfg_label)
                    function_stream.write('DACE_HIDDEN ' + decl.rstrip().removesuffix('{').rstrip() + ';\n', cfg,
                                          state_id, node)
                else:
                    # Not split (or ``unique_functions`` deduplicated this label onto a body already
                    # emitted into THIS TU, in which case the streams below are empty and no declaration
                    # is needed): write the body inline, exactly as before.
                    if not unique_functions or not code_already_generated:
                        function_stream.write(global_code)
                    function_stream.write(nested_global_stream.getvalue())
                    function_stream.write(nested_stream.getvalue())
        finally:
            self._current_tu_key = outer_tu_key

        self._dispatcher.declared_arrays.exit_scope(sdfg)
        self._dispatcher.defined_vars.exit_scope(sdfg)

    def _emit_external_translation_unit_call(self, sdfg, node, memlet_references, child_name, function_stream,
                                             callsite_stream, cfg, state_id):
        """Call a top-level GPU nest that is code-generated as its OWN standalone SDFG (Model 2).

        The nest is not inlined here; the parent calls the child SDFG's public extern-C handle ABI --
        ``__dace_init_<name>`` (allocate the child's state, return an opaque handle), ``__program_<name>``
        (run it), ``__dace_exit_<name>`` (free it) -- forward-declared and resolved in-binary by the
        static linker. Device pointers pass straight through: parent and child share the process CUDA
        context. The child is registered for a separate ``generate_code`` pass in
        ``get_generated_codeobjects`` (its kernels land in its own ``.cu``).
        """
        child_sdfg = node.sdfg
        # Match the child's generated signature EXACTLY: framecode derives both its arglist and its init
        # parameters from ``used_symbols(all_symbols=False)``, so use the same here, and render the
        # prototype from the child's own ``signature`` / ``init_signature`` -- that makes the parent's
        # declaration identical to the child's definition by construction (extern "C", nothing to mangle).
        child_fsyms = child_sdfg.used_symbols(all_symbols=False)
        child_arglist = child_sdfg.arglist(scalars_only=False, free_symbols=child_fsyms)

        # Parent-side value for each child argument NAME: array pointers from the memlet references,
        # symbols from the nest's symbol mapping. Emitted in the child's ARGLIST order -- not the
        # memlet-reference order, which sorts inputs-then-outputs and would pass swapped pointers when
        # in/out arrays interleave alphabetically.
        argval_by_name = {aname: argval for _, aname, argval in memlet_references}
        for symname, symval in node.symbol_mapping.items():
            if symname not in sdfg.constants:
                argval_by_name[symname] = cpp.sym2cpp(symval)

        init_symbols = [s for s in sorted(str(sym) for sym in child_fsyms) if not s.startswith('__dace')]
        program_args = [argval_by_name[name] for name in child_arglist.keys()]
        init_args = [argval_by_name[name] for name in init_symbols]

        # Forward declarations. ``void *`` stands in for the child's ``<name>Handle_t`` typedef so the
        # parent needs none of the child's generated headers.
        program_sig = child_sdfg.signature(with_types=True, for_call=False, arglist=child_arglist)
        program_params = 'void *__handle' + (f', {program_sig}' if program_sig else '')
        function_stream.write(
            f'extern "C" void *__dace_init_{child_name}({child_sdfg.init_signature(free_symbols=child_fsyms)});\n'
            f'extern "C" int __dace_exit_{child_name}(void *__handle);\n'
            f'extern "C" void __program_{child_name}({program_params});\n', cfg, state_id, node)

        # Call site: init -> run -> exit, braced so the handle stays local.
        handle_var = f'__exttu_h_{child_name}'
        callsite_stream.write(
            f'{{\nvoid *{handle_var} = __dace_init_{child_name}({", ".join(init_args)});\n'
            f'__program_{child_name}({", ".join([handle_var] + program_args)});\n'
            f'__dace_exit_{child_name}({handle_var});\n}}\n', cfg, state_id, node)

        self.external_children[child_name] = child_sdfg

    def _namespace_child_module_symbols(self, objects, child_name):
        """Namespace a child sub-program's TARGET-level ``__dace_init``/``__dace_exit`` symbols.

        Each standalone child is a full program, so its generate_code pass emits target-level init/exit
        named by the TARGET, not the SDFG -- e.g. ``__dace_init_experimental_cuda`` (and the same for
        every other target it uses). Linking several children plus the parent into one binary would give
        each such symbol multiple definitions. Suffix every one with the child name so the copies stay
        distinct, in lockstep across the child's own translation units (definition and every internal
        caller). The child's PUBLIC ``__dace_init_<child>`` -- the only entry the parent calls -- is left
        untouched, so the cross-program handle call still resolves.
        """
        module_targets = set()
        finder = re.compile(r'__dace_(?:init|exit)_(\w+)')
        for obj in objects:
            module_targets.update(m.group(1) for m in finder.finditer(obj.code) if m.group(1) != child_name)
        if not module_targets:
            return
        for obj in objects:
            code = obj.code
            for target in module_targets:
                for kind in ('init', 'exit'):
                    code = re.sub(r'\b__dace_%s_%s\b' % (kind, re.escape(target)),
                                  '__dace_%s_%s_%s' % (kind, target, child_name), code)
            obj.code = code

    def _collect_omp_reductions(self, sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry):
        """Walk the map's WCR-write edges that target an accumulator outside the scope and
        return ``(op_str, clause_target, data_name, declare_line)`` tuples for OpenMP
        ``reduction(...)`` clauses on the ``parallel for`` pragma.

        The WCR target must be (a) an ``AccessNode`` outside the map scope, (b) either a
        true ``Scalar`` (clause ``reduction(op:var)``, always eligible) or -- only when
        ``sdfg.openmp_array_reductions`` is on -- a plain 0-offset C-contiguous ``Array``
        buffer (clause ``reduction(op:A[0:n])`` over the whole buffer), (c) non-persistent
        with a subset independent of THIS map's iter variables, and (d) a WCR operator
        OpenMP supports (see ``_REDUCTION_TO_OMP_OP``); complex element types additionally
        need a ``#pragma omp declare reduction`` (returned as ``declare_line``, ``+``/``*``
        only). Anything else falls through to the atomic path -- correct but contended.

        Each returned tuple yields one clause; the caller emits any unique ``declare_line``
        ahead of the pragma and pushes ``data_name`` onto one scope frame so the nested
        ``write_and_resolve_expr`` skips the now-redundant atomic and accumulates into the
        OMP runtime's per-thread private copy (combined at the region barrier). Whole-buffer
        array reduction is safe even when the body touches only a sub-region: untouched
        elements keep their value (reduction identity + combine is a no-op).
        """
        out = []
        seen = set()
        try:
            map_exit = state.exit_node(map_entry)
        except (KeyError, StopIteration):
            return out
        # The reduction targets reachable from the map scope.
        for iedge in state.in_edges(map_exit):
            if iedge.data is None or iedge.data.wcr is None:
                continue
            in_conn = iedge.dst_conn
            if not in_conn or not in_conn.startswith("IN_"):
                continue
            out_conn = "OUT_" + in_conn[3:]
            out_edges = [e for e in state.out_edges(map_exit) if e.src_conn == out_conn]
            if len(out_edges) != 1:
                continue
            oedge = out_edges[0]
            if not isinstance(oedge.dst, nodes.AccessNode):
                continue
            desc = sdfg.arrays.get(oedge.dst.data)
            if desc is None:
                continue
            # A persistent / external accumulator is emitted as a state-struct member
            # (``__state->x``), which is not a valid ``reduction(...)`` lvalue. Only a
            # non-persistent (locally-declared) target can be privatized by the OMP
            # runtime; persistent targets fall through to the atomic path.
            if desc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
                continue
            # Loop-invariant subset (independent of THIS map's iter variables): the WCR
            # accumulates into the same region every iteration -- a genuine reduction over
            # this map. A param-dependent subset is a scatter, not a reduction: atomic path.
            if iedge.data.subset is not None:
                map_param_set = set(map_entry.map.params)
                if any(s in map_param_set for s in (str(x) for x in iedge.data.subset.free_symbols)):
                    continue
            redtype = operations.detect_reduction_type(iedge.data.wcr)
            op_str = _REDUCTION_TO_OMP_OP.get(redtype)
            if op_str is None:
                continue
            var_name = self.ptr(oedge.dst.data, desc, sdfg)

            # A true Scalar is always eligible for a plain ``reduction(op:var)`` clause
            # (the pre-existing, flag-independent behavior). A whole contiguous Array
            # buffer is eligible for an array-section clause ``reduction(op:A[0:n])`` ONLY
            # under ``openmp_array_reductions`` (canonicalize turns it on for its output):
            # the runtime privatizes the whole buffer per thread and combines element-wise,
            # so the body's plain ``+=`` into the private copy is race-free and elements the
            # body never touches stay unchanged (identity + combine is a no-op). Anything
            # not provably a plain contiguous buffer falls through to the atomic path.
            is_scalar = isinstance(desc, data.Scalar)
            declare = None
            if is_scalar:
                clause_target = var_name
            elif sdfg.openmp_array_reductions:
                # An array-section reduction ``reduction(op:A[0:n])`` privatizes the WHOLE
                # buffer per thread (identity-initialized) for the region. If the map body
                # also READS ``A`` (a non-WCR input flowing through the map entry), those
                # reads see the private identity copy instead of the shared values -- silently
                # wrong. polybench ``trisolv``: the forward-substitution map reduces into
                # ``x[i]`` but also reads ``x[j]`` (j < i); a whole-``x`` reduction makes every
                # ``x[j]`` read ``0``, dropping the sum (``x[i]=b[i]/L[i,i]``). A sound array
                # reduction must be a WRITE-ONLY accumulate; if the array is also read, fall
                # through to the (correct, contended) atomic path.
                if any(
                        isinstance(e.src, nodes.AccessNode) and e.src.data == oedge.dst.data
                        for e in state.in_edges(map_entry)):
                    continue
                count = _contiguous_element_count(desc)
                if count is None:
                    continue
                clause_target = "%s[0:%s]" % (var_name, cpp.sym2cpp(count))
            else:
                continue

            # Complex element types have no built-in OpenMP reduction; under the flag emit
            # a ``#pragma omp declare reduction`` (only ``+`` / ``*`` have a defined identity
            # + combiner). Off the flag, preserve the exact prior scalar behavior (no dtype
            # special-casing) so a disabled flag is a codegen no-op.
            if sdfg.openmp_array_reductions and desc.dtype in _COMPLEX_TYPES:
                declare = _complex_declare_reduction(op_str, desc.dtype.ctype)
                if declare is None:
                    continue

            key = (op_str, clause_target)
            if key in seen:
                continue
            seen.add(key)
            out.append((op_str, clause_target, oedge.dst.data, declare))
        return out

    def _generate_MapEntry(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.MapEntry,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):
        state_dfg = cfg.state(state_id)
        map_params = node.map.params

        result = callsite_stream
        map_header = ""

        # Encapsulate map with a C scope
        needs_brace = self.map_scope_needs_brace(sdfg, state_dfg, node)
        self._map_scope_braced[id(node.map)] = needs_brace
        if needs_brace:
            callsite_stream.write('{', cfg, state_id, node)

        # Define all input connectors of this map entry
        for e in dynamic_map_inputs(state_dfg, node):
            if self.ptr(e.data.data, sdfg.arrays[e.data.data], sdfg) != e.dst_conn:
                callsite_stream.write(
                    self.memlet_definition(sdfg, e.data, False, e.dst_conn, e.dst.in_connectors[e.dst_conn]), cfg,
                    state_id, node)

        inner_stream = CodeIOStream()
        self.generate_scope_preamble(sdfg, dfg, state_id, function_stream, callsite_stream, inner_stream)

        # Instrumentation: Pre-scope
        instr = self._dispatcher.instrumentation[node.map.instrument]
        if instr is not None:
            instr.on_scope_entry(sdfg, cfg, state_dfg, node, callsite_stream, inner_stream, function_stream)

        # TODO: Refactor to generate_scope_preamble once a general code
        #  generator (that CPU inherits from) is implemented
        if node.map.schedule in (dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.CPU_Persistent):
            # OpenMP header
            in_persistent = False
            if node.map.schedule == dtypes.ScheduleType.CPU_Multicore:
                in_persistent = is_in_scope(sdfg, state_dfg, node, [dtypes.ScheduleType.CPU_Persistent])
                if in_persistent:
                    # If already in a #pragma omp parallel, no need to use it twice
                    map_header += "#pragma omp for"
                    # TODO(later): barriers and map_header += " nowait"
                else:
                    map_header += "#pragma omp parallel for"

            elif node.map.schedule == dtypes.ScheduleType.CPU_Persistent:
                map_header += "#pragma omp parallel"

            # OpenMP schedule properties
            if not in_persistent:
                if node.map.omp_schedule != dtypes.OMPScheduleType.Default:
                    schedule = " schedule("
                    if node.map.omp_schedule == dtypes.OMPScheduleType.Static:
                        schedule += "static"
                    elif node.map.omp_schedule == dtypes.OMPScheduleType.Dynamic:
                        schedule += "dynamic"
                    elif node.map.omp_schedule == dtypes.OMPScheduleType.Guided:
                        schedule += "guided"
                    else:
                        raise ValueError("Unknown OpenMP schedule type")
                    if node.map.omp_chunk_size > 0:
                        schedule += f", {node.map.omp_chunk_size}"
                    schedule += ")"
                    map_header += schedule

                if node.map.omp_num_threads > 0:
                    map_header += f" num_threads({node.map.omp_num_threads})"

            # OpenMP nested loop properties
            if node.map.schedule == dtypes.ScheduleType.CPU_Multicore and node.map.collapse > 1:
                map_header += ' collapse(%d)' % node.map.collapse

            # OpenMP reduction clauses for WCR writes to scalar accumulators outside
            # the map. Atomic-add via wcr_fixed::reduce_atomic is correct but
            # contended -- replacing the (implicit) per-iter atomic with the OMP
            # runtime's per-thread privatization + final tree-reduce is what makes
            # scalar reductions in parallel maps fast. The covered ``(var, op)``
            # pairs are pushed onto ``_omp_reduction_scope_stack`` so the
            # downstream ``write_and_resolve_expr`` skips the now-redundant
            # ``reduce_atomic`` for them.
            # Gated by compiler.emit_tree_reductions: OFF leaves omp_reductions empty, so no
            # reduction(op:var) clause is emitted and the WCR write below takes the plain
            # atomic path (correct but contended) instead of privatize-and-tree-reduce.
            omp_reductions = []
            if (node.map.schedule == dtypes.ScheduleType.CPU_Multicore
                    and Config.get_bool('compiler', 'emit_tree_reductions')):
                omp_reductions = self._collect_omp_reductions(sdfg, state_dfg, node)
                declares = []
                for op_str, clause_target, _dname, declare in omp_reductions:
                    map_header += f' reduction({op_str}:{clause_target})'
                    if declare is not None and declare not in declares:
                        declares.append(declare)
                # ``declare reduction`` directives must be in scope before the pragma; emit
                # each unique one on its own line ahead of the ``parallel for``.
                for declare in declares:
                    map_header = declare + '\n' + map_header
            # Push scope frame even if empty -- ``_generate_MapExit`` always pops. Keyed by
            # target data name so the nested WCR write's covered-check (``memlet.data in
            # frame``) skips the now-redundant atomic and accumulates into the private copy.
            self._omp_reduction_scope_stack.append({dname: op for op, _ct, dname, _dec in omp_reductions})

        if node.map.unroll:
            if node.map.schedule in (dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.CPU_Persistent):
                raise ValueError("An OpenMP map cannot be unrolled (" + node.map.label + ")")

        # A symbolic step whose sign is not statically known is guarded by
        # an assert() (a statically-negative step is already rejected by
        # SDFG validation). Placed before the OpenMP pragma, which must be
        # immediately followed by its loop.
        for _, _, _skip in node.map.range:
            if (_skip > 0) != True:
                result.write(
                    'assert((%s) > 0 && "Map %s requires a positive step");\n' % (cpp.sym2cpp(_skip), node.map.label),
                    cfg, state_id, node)

        result.write(map_header, cfg, state_id, node)

        if node.map.schedule == dtypes.ScheduleType.CPU_Persistent:
            result.write('{\n', cfg, state_id, node)

            # Find if bounds are used within the scope
            scope = state_dfg.scope_subgraph(node, False, False)
            fsyms = self._frame.free_symbols(scope)
            # Include external edges
            for n in scope.nodes():
                for e in state_dfg.all_edges(n):
                    fsyms |= e.data.used_symbols(False, e)
            fsyms = set(map(str, fsyms))

            ntid_is_used = '__omp_num_threads' in fsyms
            tid_is_used = node.map.params[0] in fsyms
            if tid_is_used or ntid_is_used:
                function_stream.write('#include <omp.h>', cfg, state_id, node)
            if tid_is_used:
                result.write(f'auto {node.map.params[0]} = omp_get_thread_num();', cfg, state_id, node)
            if ntid_is_used:
                result.write(f'auto __omp_num_threads = omp_get_num_threads();', cfg, state_id, node)
        else:
            # Emit nested loops
            for i, r in enumerate(node.map.range):
                var = map_params[i]
                begin, end, skip = r

                if node.map.unroll:
                    unroll_pragma = "#pragma unroll"
                    if node.map.unroll_factor:
                        unroll_pragma += f" {node.map.unroll_factor}"
                    result.write(unroll_pragma, cfg, state_id, node)

                comparison, bound = loop_exit_test(begin, end, skip, node)
                init = '%s %s = %s' % (loop_index_ctype(), var, cpp.sym2cpp(begin))
                if hoist_loop_decls(node):
                    # Declared ahead of the loop, so it outlives it -- the map's encapsulating scope is
                    # what bounds it (experimental keeps that brace when hoisting).
                    result.write('%s;\n' % init, cfg, state_id, node)
                    init = ''
                result.write(
                    "for (%s; %s %s %s; %s += %s) {\n" % (init, var, comparison, bound, var, cpp.sym2cpp(skip)),
                    cfg,
                    state_id,
                    node,
                )

        callsite_stream.write(inner_stream.getvalue())

        # Emit internal transient array allocation
        self._frame.allocate_arrays_in_scope(sdfg, cfg, node, function_stream, result)

    def _generate_MapExit(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                          node: nodes.MapExit, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        result = callsite_stream

        # Obtain start of map
        scope_dict = dfg.scope_dict()
        map_node = scope_dict[node]
        state_dfg = cfg.state(state_id)

        if map_node is None:
            raise ValueError("Exit node " + str(node.map.label) + " is not dominated by a scope entry node")

        # Emit internal transient array deallocation
        self._frame.deallocate_arrays_in_scope(sdfg, cfg, map_node, function_stream, result)

        outer_stream = CodeIOStream()

        # Instrumentation: Post-scope
        instr = self._dispatcher.instrumentation[node.map.instrument]
        if instr is not None and not is_devicelevel_gpu(sdfg, state_dfg, node):
            instr.on_scope_exit(sdfg, cfg, state_dfg, node, outer_stream, callsite_stream, function_stream)

        self.generate_scope_postamble(sdfg, dfg, state_id, function_stream, outer_stream, callsite_stream)

        if map_node.map.schedule == dtypes.ScheduleType.CPU_Persistent:
            result.write("}", cfg, state_id, node)
        else:
            for _ in map_node.map.range:
                result.write("}", cfg, state_id, node)

        result.write(outer_stream.getvalue())

        # Close the encapsulating C scope only if the matching MapEntry opened one.
        if self._map_scope_braced.pop(id(node.map), True):
            callsite_stream.write('}', cfg, state_id, node)

        # Pop the OMP-reduction scope frame pushed by the matching MapEntry.
        if self._omp_reduction_scope_stack:
            self._omp_reduction_scope_stack.pop()

    def _generate_ConsumeEntry(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.ConsumeEntry,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        result = callsite_stream

        state_dfg: SDFGState = cfg.nodes()[state_id]

        input_sedge = next(e for e in state_dfg.in_edges(node) if e.dst_conn == "IN_stream")
        output_sedge = next(e for e in state_dfg.out_edges(node) if e.src_conn == "OUT_stream")
        input_stream = state_dfg.memlet_path(input_sedge)[0].src
        input_streamdesc = input_stream.desc(sdfg)

        # Take chunks into account
        if node.consume.chunksize == 1:
            ctype = 'const %s' % input_streamdesc.dtype.ctype
            chunk = "%s& %s" % (ctype, "__dace_" + node.consume.label + "_element")
            self._dispatcher.defined_vars.add("__dace_" + node.consume.label + "_element", DefinedType.Scalar, ctype)
        else:
            ctype = 'const %s *' % input_streamdesc.dtype.ctype
            chunk = "%s %s, size_t %s" % (ctype, "__dace_" + node.consume.label + "_elements",
                                          "__dace_" + node.consume.label + "_numelems")
            self._dispatcher.defined_vars.add("__dace_" + node.consume.label + "_elements", DefinedType.Pointer, ctype)
            self._dispatcher.defined_vars.add("__dace_" + node.consume.label + "_numelems", DefinedType.Scalar,
                                              'size_t')

        # Take quiescence condition into account
        if node.consume.condition is not None:
            condition_string = "[&]() { return %s; }, " % cppunparse.cppunparse(node.consume.condition.code, False)
        else:
            condition_string = ""

        inner_stream = CodeIOStream()

        self.generate_scope_preamble(sdfg, dfg, state_id, function_stream, callsite_stream, inner_stream)

        # Instrumentation: Post-scope
        instr = self._dispatcher.instrumentation[node.consume.instrument]
        if instr is not None:
            instr.on_scope_entry(sdfg, state_dfg, node, callsite_stream, inner_stream, function_stream)

        result.write(
            "dace::Consume<{chunksz}>::template consume{cond}({stream_in}, "
            "{num_pes}, {condition}"
            "[&](int {pe_index}, {element_or_chunk}) {{".format(
                chunksz=node.consume.chunksize,
                cond="" if node.consume.condition is None else "_cond",
                condition=condition_string,
                stream_in=input_stream.data,  # TODO: stream arrays
                element_or_chunk=chunk,
                num_pes=cpp.sym2cpp(node.consume.num_pes),
                pe_index=node.consume.pe_index,
            ),
            cfg,
            state_id,
            node,
        )

        # Since consume is an alias node, we create an actual array for the
        # consumed element and modify the outgoing memlet path ("OUT_stream")
        # TODO: do this before getting to the codegen (preprocess)
        if node.consume.chunksize == 1:
            newname, _ = sdfg.add_scalar("__dace_" + node.consume.label + "_element",
                                         input_streamdesc.dtype,
                                         transient=True,
                                         storage=dtypes.StorageType.Register,
                                         find_new_name=True)
            ce_node = nodes.AccessNode(newname)
        else:
            newname, _ = sdfg.add_array("__dace_" + node.consume.label + '_elements', [node.consume.chunksize],
                                        input_streamdesc.dtype,
                                        transient=True,
                                        storage=dtypes.StorageType.Register,
                                        find_new_name=True)
            ce_node = nodes.AccessNode(newname)
        state_dfg.add_node(ce_node)
        out_memlet_path = state_dfg.memlet_path(output_sedge)
        state_dfg.remove_edge(out_memlet_path[0])
        state_dfg.add_edge(
            out_memlet_path[0].src,
            out_memlet_path[0].src_conn,
            ce_node,
            None,
            mmlt.Memlet.from_array(ce_node.data, ce_node.desc(sdfg)),
        )
        state_dfg.add_edge(
            ce_node,
            None,
            out_memlet_path[0].dst,
            out_memlet_path[0].dst_conn,
            mmlt.Memlet.from_array(ce_node.data, ce_node.desc(sdfg)),
        )
        for e in out_memlet_path[1:]:
            e.data.data = ce_node.data
        # END of SDFG-rewriting code

        result.write(inner_stream.getvalue())

        # Emit internal transient array allocation
        self._frame.allocate_arrays_in_scope(sdfg, cfg, node, function_stream, result)

        # Generate register definitions for inter-tasklet memlets
        scope_dict = dfg.scope_dict()
        for child in dfg.scope_children()[node]:
            if not isinstance(child, nodes.AccessNode):
                continue

            for edge in dfg.edges():
                # Only interested in edges within current scope
                if scope_dict[edge.src] != node or scope_dict[edge.dst] != node:
                    continue
                # code->code edges
                if (isinstance(edge.src, nodes.CodeNode) and isinstance(edge.dst, nodes.CodeNode)):
                    local_name = edge.data.data
                    ctype = node.out_connectors[edge.src_conn].ctype
                    if not local_name:
                        # Very unique name. TODO: Make more intuitive
                        local_name = '__dace_%d_%d_%d_%d_%s' % (cfg.cfg_id, state_id, dfg.node_id(
                            edge.src), dfg.node_id(edge.dst), edge.src_conn)

                    # Allocate variable type
                    code = '%s %s;' % (ctype, local_name)
                    result.write(code, cfg, state_id, [edge.src, edge.dst])
                    self._dispatcher.defined_vars.add(local_name, DefinedType.Scalar, ctype)

    def _generate_ConsumeExit(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                              node: nodes.ConsumeExit, function_stream: CodeIOStream,
                              callsite_stream: CodeIOStream) -> None:
        result = callsite_stream

        # Obtain start of map
        scope_dict = dfg.scope_dict()
        entry_node = scope_dict[node]
        state_dfg: SDFGState = cfg.node(state_id)

        if entry_node is None:
            raise ValueError("Exit node " + str(node.consume.label) + " is not dominated by a scope entry node")

        # Emit internal transient array deallocation
        self._frame.deallocate_arrays_in_scope(sdfg, cfg, entry_node, function_stream, result)

        outer_stream = CodeIOStream()

        # Instrumentation: Post-scope
        instr = self._dispatcher.instrumentation[node.consume.instrument]
        if instr is not None:
            instr.on_scope_exit(sdfg, state_dfg, node, outer_stream, callsite_stream, function_stream)

        self.generate_scope_postamble(sdfg, dfg, state_id, function_stream, outer_stream, callsite_stream)

        result.write("});", cfg, state_id, node)

        result.write(outer_stream.getvalue())

    def _generate_AccessNode(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                             node: nodes.Node, function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        state_dfg: SDFGState = cfg.nodes()[state_id]

        if node not in state_dfg.sink_nodes():
            # NOTE: sink nodes are synchronized at the end of a state
            cpp.presynchronize_streams(sdfg, cfg, state_dfg, state_id, node, callsite_stream)

        # Instrumentation: Pre-node
        instr = self._dispatcher.instrumentation[node.instrument]
        if instr is not None:
            instr.on_node_begin(sdfg, cfg, state_dfg, node, callsite_stream, callsite_stream, function_stream)

        sdict = state_dfg.scope_dict()
        for edge in state_dfg.in_edges(node):
            predecessor, _, _, _, memlet = edge
            if memlet.data is None:
                continue  # If the edge has to be skipped

            # Determines if this path ends here or has a definite source (array) node
            memlet_path = state_dfg.memlet_path(edge)
            if memlet_path[-1].dst == node:
                src_node = memlet_path[0].src
                # Only generate code in case this is the innermost scope
                # (copies are generated at the inner scope, where both arrays exist)
                if (scope_contains_scope(sdict, src_node, node) and sdict[src_node] != sdict[node]):
                    self._dispatcher.dispatch_copy(
                        src_node,
                        node,
                        edge,
                        sdfg,
                        cfg,
                        dfg,
                        state_id,
                        function_stream,
                        callsite_stream,
                    )

        # Process outgoing memlets (array-to-array write should be emitted
        # from the first leading edge out of the array)
        self.process_out_memlets(
            sdfg,
            cfg,
            state_id,
            node,
            dfg,
            self._dispatcher,
            callsite_stream,
            False,
            function_stream,
        )

        # Instrumentation: Post-node
        if instr is not None:
            instr.on_node_end(sdfg, cfg, state_dfg, node, callsite_stream, callsite_stream, function_stream)

    # Methods for subclasses to override

    def map_scope_needs_brace(self, sdfg: SDFG, state_dfg: SDFGState, node: nodes.MapEntry) -> bool:
        """Whether the map's encapsulating C scope (``{ ... }``) must be emitted. It bounds only what is
        declared ahead of the loop headers (dynamic map inputs, scope preamble, instrumentation locals,
        OpenMP ``declare reduction``), not scope-lifetime transients (scoped by the innermost loop body).
        Always True here; the readable generator overrides it to drop braces that bound nothing."""
        return True

    def generate_scope_preamble(self, sdfg, dfg_scope, state_id, function_stream, outer_stream, inner_stream):
        """
        Generates code for the beginning of an SDFG scope, outputting it to
        the given code streams.

        :param sdfg: The SDFG to generate code from.
        :param dfg_scope: The `ScopeSubgraphView` to generate code from.
        :param state_id: The node ID of the state in the given SDFG.
        :param function_stream: A `CodeIOStream` object that will be
                                generated outside the calling code, for
                                use when generating global functions.
        :param outer_stream: A `CodeIOStream` object that points
                             to the code before the scope generation (e.g.,
                             before for-loops or kernel invocations).
        :param inner_stream: A `CodeIOStream` object that points
                             to the beginning of the scope code (e.g.,
                             inside for-loops or beginning of kernel).
        """
        pass

    def generate_scope_postamble(self, sdfg, dfg_scope, state_id, function_stream, outer_stream, inner_stream):
        """
        Generates code for the end of an SDFG scope, outputting it to
        the given code streams.

        :param sdfg: The SDFG to generate code from.
        :param dfg_scope: The `ScopeSubgraphView` to generate code from.
        :param state_id: The node ID of the state in the given SDFG.
        :param function_stream: A `CodeIOStream` object that will be
                                generated outside the calling code, for
                                use when generating global functions.
        :param outer_stream: A `CodeIOStream` object that points
                             to the code after the scope (e.g., after
                             for-loop closing braces or kernel invocations).
        :param inner_stream: A `CodeIOStream` object that points
                             to the end of the inner scope code (e.g.,
                             before for-loop closing braces or end of
                             kernel).
        """
        pass

    def generate_tasklet_preamble(self, sdfg, cfg, dfg_scope, state_id, node, function_stream, before_memlets_stream,
                                  after_memlets_stream):
        """
        Generates code for the beginning of a tasklet. This method is
        intended to be overloaded by subclasses.

        :param sdfg: The SDFG to generate code from.
        :param dfg_scope: The `ScopeSubgraphView` to generate code from.
        :param state_id: The node ID of the state in the given SDFG.
        :param node: The tasklet node in the state.
        :param function_stream: A `CodeIOStream` object that will be
                                generated outside the calling code, for
                                use when generating global functions.
        :param before_memlets_stream: A `CodeIOStream` object that will emit
                                      code before input memlets are generated.
        :param after_memlets_stream: A `CodeIOStream` object that will emit code
                                     after input memlets are generated.
        """
        pass

    def generate_tasklet_postamble(self, sdfg, cfg, dfg_scope, state_id, node, function_stream, before_memlets_stream,
                                   after_memlets_stream):
        """
        Generates code for the end of a tasklet. This method is intended to be
        overloaded by subclasses.

        :param sdfg: The SDFG to generate code from.
        :param dfg_scope: The `ScopeSubgraphView` to generate code from.
        :param state_id: The node ID of the state in the given SDFG.
        :param node: The tasklet node in the state.
        :param function_stream: A `CodeIOStream` object that will be
                                generated outside the calling code, for
                                use when generating global functions.
        :param before_memlets_stream: A `CodeIOStream` object that will emit
                                      code before output memlets are generated.
        :param after_memlets_stream: A `CodeIOStream` object that will emit code
                                     after output memlets are generated.
        """
        pass

    def make_ptr_vector_cast(self, *args, **kwargs):
        return cpp.make_ptr_vector_cast(*args, **kwargs)

    def ptr(self,
            name: str,
            desc: data.Data,
            sdfg: SDFG = None,
            subset: Optional[subsets.Subset] = None,
            is_write: Optional[bool] = None,
            ancestor: int = 0) -> str:
        """
        Returns a string that points to the data based on its name and descriptor.

        :param name: Data name.
        :param desc: Data descriptor.
        :param sdfg: SDFG in which the data resides.
        :param subset: Optional subset associated with the data.
        :param is_write: Whether the access is a write access.
        :param ancestor: Scope ancestor level.
        :return: C-compatible name that can be used to access the data.
        """
        return cpp.ptr(name, desc, sdfg, self._frame)

    def emit_interstate_variable_declaration(self, name: str, dtype: dtypes.typeclass, callsite_stream: CodeIOStream,
                                             sdfg: SDFG):
        # ``loop_index_type`` (int32/int64) retypes ONLY a LoopRegion counter's hoisted declaration --
        # ``int32_t i;`` / ``int64_t i;`` in place of the inferred type. The registered defined-type
        # follows the same spelling so the counter's other uses (condition, body indexing) name that
        # type without introducing a cast. ``auto`` (the default) leaves the declaration untouched, so
        # legacy output stays byte-identical. Every non-counter interstate symbol is unaffected.
        # ``decl_placement = late`` moves a loop-local counter's declaration into its own loop's
        # for-init clause -- the loop emitter reads the recorded ctype and spells ``for (T i = ...)``.
        # Nothing is written here in that case: this hoisted declaration IS what the knob removes. The
        # defined type is still registered, exactly as the hoisted declaration would, so the counter's
        # uses inside the loop resolve identically (by the gate, it has no uses outside).
        local_ctype = loop_local_counter_ctype(name, dtype, sdfg)
        if local_ctype is not None:
            self._frame.loop_local_counters[(sdfg.cfg_id, name)] = local_ctype
            self._frame.dispatcher.defined_vars.add(name, DefinedType.Scalar, local_ctype)
            return
        override = loop_region_index_ctype()
        if override is not None and is_loop_region_variable(name, sdfg):
            callsite_stream.write('%s %s;\n' % (override, name), sdfg)
            self._frame.dispatcher.defined_vars.add(name, DefinedType.Scalar, override)
            return
        isvar = data.Scalar(dtype)
        callsite_stream.write('%s;\n' % (isvar.as_arg(with_types=True, name=name)), sdfg)
        self._frame.dispatcher.defined_vars.add(name, DefinedType.Scalar, dtype.ctype)
