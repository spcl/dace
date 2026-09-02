# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ArgReduce``: argmax / argmin over a 1-D input slice -> (value, index).

A dedicated reduction library node (separate from :class:`Reduce`, which is a
single-output *value* fold) for the argmax / argmin pattern that
:class:`~dace.transformation.passes.canonicalize.arg_max_lift.ArgMaxLift` lifts.
It has **two scalar outputs** -- ``_out_val`` (the extreme value; the input's
dtype) and ``_out_idx`` (its position; ``int64`` by default) -- and reduces over
the full input slice presented on ``_in``.

Why a new node rather than extending :class:`Reduce`: ``Reduce`` hard-asserts a
single input / single output and all three of its expansions index
``out_edges[0]``; threading a second (index) output through it would touch a
node every existing reduction depends on. A standalone node emits CUB (GPU) or
plain CPU code directly and leaves ``Reduce`` alone.

The node reads its own operand: the ``_in`` memlet may carry a STRIDE, and
:attr:`ArgReduce.transform` names a unary function applied per element as it is
read. Both exist so a caller never has to materialise the sequence it wants
arg-reduced. TSVC ``s318`` asks for ``argmax |a[k]|`` with ``k = inc*i``; spelled
as a strided ``_in`` plus ``transform='abs'`` that is one streaming pass over
``a``, where staging ``buf[j] = |a[inc*j]|`` first would write and re-read a
whole extra copy of the array.

Expansions:

* ``pure`` (CPU default): a CPP tasklet with a sequential scan over the
  input -- correctness-first, no external dependency.
* ``OpenMP``: the same scan split across threads under a ``declare reduction``
  over the (value, index) pair (see :class:`ExpandArgReduceOpenMP`).
* ``CUDA`` (GPU): ``gpucub::DeviceReduce::ArgMax`` / ``ArgMin`` through
  ``dace::cub::arg_reduce``, which answers both scalar outputs. Unit-stride,
  untransformed input only; anything else needs an input iterator CUB does not
  take for free.

Tie-breaking matches the TSVC sequential source ``if a[i] OP best: best = a[i];
idx = i`` -- a STRICT comparison, so the FIRST occurrence of the extreme value
wins (a strictly-greater/lesser test never updates on a tie). Every parallel
combine below therefore also breaks ties toward the LOWER index, which is what
makes the answer independent of how the range was split. The ``_in`` index
is slice-local (``0 .. N-1``); the lift adds the slice base to recover the
original-array position.
"""
from typing import Callable, Optional, Tuple

import dace
from dace import library, properties, symbolic
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation
from dace.ordered import OrderedSet

_OP_CPP = {'max': '>', 'min': '<'}
_OP_CUB = {'max': 'ArgMax', 'min': 'ArgMin'}
#: The ``dace/cub_compat.cuh`` tag that picks the CUB routine, and with it the spelling that
#: is not deprecated on the toolkit in front of us.
_OP_TAG = {'max': 'ArgMaxOp', 'min': 'ArgMinOp'}
#: Unary element transforms, spelled for C++. A closed map rather than a passthrough: the name
#: reaches the generated source verbatim, so an unknown one must be refused, not pasted.
#: ``std::abs``, not ``dace::math::abs``: the latter namespace holds only the ``typeless_nan``
#: overload, so a real operand does not match it.
_TRANSFORM_CPP = {'': None, 'abs': 'std::abs'}

# The (value, index) pair below spells its fields ``__ar_v`` / ``__ar_i`` rather than ``v`` / ``i``
# because a CPP tasklet's free symbols come from an identifier scan of its code: the member access
# ``pair.i`` reads as a reference to a SYMBOL named ``i``, which then joins the SDFG's signature --
# and ``i`` is what a lifted loop calls its iteration variable.


def _count(in_edge) -> symbolic.SymbolicType:
    """How many elements the scan visits.

    The memlet's VOLUME, not its subset's element count: a subset states its count as
    ``ceiling((hi - lo + 1) / step)``, which a SYMBOLIC step leaves as ``n - 1 + ceiling(1/step)``
    -- neither the count nor anything codegen can spell. ``volume`` is defined as the exact number
    of elements the edge moves and defaults to the subset's count, so a plain slice is unaffected
    and a strided gather whose stride is a runtime symbol still names its own length.
    """
    return in_edge.data.volume


def _scan_context(node: "ArgReduce", parent_state: dace.SDFGState,
                  parent_sdfg: dace.SDFG) -> Tuple[dace.dtypes.typeclass, str, str, Callable[[str], str], bool]:
    """The operand facts both CPU expansions read off the wired edges.

    :returns: ``(value_dtype, index_ctype, element_count, read, has_val)``, where ``read(expr)``
        is the C++ that yields element ``expr`` of the input slice with the transform applied.
    """
    from dace.codegen.targets.cpp import sym2cpp
    node.validate(parent_sdfg, parent_state)
    in_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == '_in')
    val_edge = next((e for e in parent_state.out_edges(node) if e.src_conn == '_out_val'), None)
    idx_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == '_out_idx')

    in_dtype = parent_sdfg.arrays[in_edge.data.data].dtype
    idx_dtype = parent_sdfg.arrays[idx_edge.data.data].dtype
    sub = in_edge.data.subset

    # Stride of the (1-D) input slice. ``_in`` points at the slice base, so a strided slice
    # ``a[lo:hi:s]`` reads element ``j`` at ``_in[j*s]``. A unit-stride slice gets the bare
    # subscript rather than a multiply by one, so the common case reads as what it is; a
    # compile-time-constant stride folds away, a symbolic one stays a runtime multiply.
    step = sub.ranges[0][2] if len(sub.ranges) == 1 else 1
    try:
        unit_stride = (int(symbolic.simplify(step)) == 1)
    except (TypeError, ValueError):
        unit_stride = False
    step_str = sym2cpp(step)
    fn = _TRANSFORM_CPP[node.transform]

    def read(expr: str) -> str:
        raw = f'_in[{expr}]' if unit_stride else f'_in[({expr}) * ({step_str})]'
        return raw if fn is None else f'{fn}({raw})'

    return in_dtype, idx_dtype.ctype, sym2cpp(_count(in_edge)), read, val_edge is not None


def _beats(op: str, cand_v: str, cand_i: str, held_v: str, held_i: str) -> str:
    """C++ for "the candidate (value, index) pair replaces the held one".

    Strictly better on the value, or equal on the value at a LOWER index. That is a total order on
    pairs, which is what makes every combine below -- across lanes, across blocks, across threads --
    agree with the sequential first-wins scan however the range was split.
    """
    return f'{cand_v} {op} {held_v} || ({cand_v} == {held_v} && {cand_i} < {held_i})'


def _writeback(has_val: bool) -> str:
    """Copy the scan's answer out through whichever result connectors are wired."""
    return (f'_out_val = __ar_best.__ar_v;\n' if has_val else '') + f'_out_idx = __ar_best.__ar_i;'


def _connectors(has_val: bool):
    return {c: None for c in (('_out_val', '_out_idx') if has_val else ('_out_idx', ))}


@library.expansion
class ExpandArgReducePure(ExpandTransformation):
    """Correctness-only CPU lowering: a sequential argmax/argmin scan."""

    environments = []

    @staticmethod
    def expansion(node: "ArgReduce", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        in_dtype, it, n_str, read, has_val = _scan_context(node, parent_state, parent_sdfg)
        vt, op = in_dtype.ctype, _OP_CPP[node.op]

        # A strict comparison keeps the FIRST extreme element (matches the sequential source).
        # ``_out_idx`` is the SLICE-LOCAL position ``0 .. n-1``.
        code = (f"struct {{ {vt} __ar_v; {it} __ar_i; }} __ar_best;\n"
                f"__ar_best.__ar_v = {read('0')}; __ar_best.__ar_i = 0;\n"
                f"for ({it} __i = 1; __i < {n_str}; ++__i) {{\n"
                f"    const {vt} __v = {read('__i')};\n"
                f"    if (__v {op} __ar_best.__ar_v) {{ __ar_best.__ar_v = __v; __ar_best.__ar_i = __i; }}\n"
                f"}}\n" + _writeback(has_val))
        return nodes.Tasklet(label=f"{node.label}_pure",
                             inputs={'_in': dace.pointer(in_dtype)},
                             outputs=_connectors(has_val),
                             code=code,
                             language=dace.dtypes.Language.CPP)


@library.expansion
class ExpandArgReduceOpenMP(ExpandTransformation):
    """Parallel CPU lowering: an OpenMP ``declare reduction`` over a (value, index) pair.

    argmax is associative on the PAIR, not on the value alone -- combining two partial results has
    to carry the index that produced the winning value, and break ties toward the LOWER index so
    the result matches the sequential scan element-for-element. A plain ``reduction(max:val)``
    cannot express that, hence the custom combiner.

    The per-thread scan stays a plain branchy loop. Restructuring it into several independent
    accumulators would let a compiler vectorise the comparison, and that was measured on the four
    argmax kernels of the corpus at their largest rung: 1.12x geomean, contended and uncontended
    alike. The scan is bound by how fast its thread can pull the operand from memory, not by the
    compare, so the unroll factor and block size such a form needs buy a tuning constant rather
    than a different order of work -- which is not what this lowering is for.
    """

    environments = []

    @staticmethod
    def expansion(node: "ArgReduce", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        in_dtype, it, n_str, read, has_val = _scan_context(node, parent_state, parent_sdfg)
        vt, op = in_dtype.ctype, _OP_CPP[node.op]
        thread_beats = _beats(op, 'omp_in.__ar_v', 'omp_in.__ar_i', 'omp_out.__ar_v', 'omp_out.__ar_i')
        # ``__ar_best`` is seeded from element 0 rather than an identity: the input slice is never
        # empty, and ``initializer(omp_priv = omp_orig)`` then hands every thread a real element,
        # so no dtype-dependent infinity has to be spelled.
        code = (f"struct __ar_pair {{ {vt} __ar_v; {it} __ar_i; }};\n"
                f"#pragma omp declare reduction(__ar_best_op : struct __ar_pair : \\\n"
                f"    omp_out = ({thread_beats}) ? omp_in : omp_out) \\\n"
                f"    initializer(omp_priv = omp_orig)\n"
                f"struct __ar_pair __ar_best;\n"
                f"__ar_best.__ar_v = {read('0')}; __ar_best.__ar_i = 0;\n"
                f"#pragma omp parallel for reduction(__ar_best_op : __ar_best)\n"
                f"for ({it} __i = 1; __i < ({n_str}); ++__i) {{\n"
                f"    const {vt} __v = {read('__i')};\n"
                f"    if (__v {op} __ar_best.__ar_v) {{ __ar_best.__ar_v = __v; __ar_best.__ar_i = __i; }}\n"
                f"}}\n" + _writeback(has_val))
        return nodes.Tasklet(label=f"{node.label}_openmp",
                             inputs={'_in': dace.pointer(in_dtype)},
                             outputs=_connectors(has_val),
                             code=code,
                             language=dace.dtypes.Language.CPP)


def cuda_refusal(node: "ArgReduce", state: dace.SDFGState) -> Optional[str]:
    """Why :class:`ExpandArgReduceCUDA` cannot lower ``node`` as wired in ``state``, else ``None``.

    ``gpucub::DeviceReduce::ArgMax`` reads a plain contiguous pointer, so neither a strided ``_in``
    nor a per-element :attr:`ArgReduce.transform` has a CUB form here. The expansion raises this
    text, and a caller CHOOSING an implementation asks the same question first -- one rule, so a
    selector cannot pick a lowering the expansion then refuses.
    """
    in_edge = next(e for e in state.in_edges(node) if e.dst_conn == '_in')
    sub = in_edge.data.subset
    step = sub.ranges[0][2] if len(sub.ranges) == 1 else 1
    if symbolic.equal(step, 1) is not True:
        return (f"ArgReduce CUDA reads a slice of stride {step}; gpucub::DeviceReduce::{_OP_CUB[node.op]} takes a "
                "contiguous pointer. Lower this one through 'pure' or 'OpenMP', or wrap the input in a "
                "gpucub::TransformInputIterator over a CountingInputIterator first.")
    if node.transform:
        return (f"ArgReduce CUDA reads through the transform {node.transform!r}; "
                f"gpucub::DeviceReduce::{_OP_CUB[node.op]} takes a plain pointer. Lower this one through 'pure' or "
                "'OpenMP', or wrap the input in a gpucub::TransformInputIterator first.")
    return None


@library.expansion
class ExpandArgReduceCUDA(ExpandTransformation):
    """Device lowering: ``gpucub::DeviceReduce::ArgMax`` / ``ArgMin``, split into the two outputs.

    The wrapper emitted here is one call to ``dace::cub::arg_reduce``
    (:file:`dace/runtime/include/dace/cub_compat.cuh`), which owns the scratch block, the device
    result buffer, and the copy back to the two host scalars. That is also where the toolkit split
    lives: CCCL 2.8 / hipCUB 4.0 deprecated CUB's ``KeyValuePair`` output in favour of two separate
    output iterators, and warnings are errors. ``ArgMax`` breaks ties toward the LOWER index, which
    is the first-occurrence rule the sequential source has.
    """

    # Filled in on first expansion to dodge the sort<->standard import cycle.
    environments = []

    @staticmethod
    def expansion(node: "ArgReduce", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        from dace.codegen.targets.cpp import sym2cpp
        if not ExpandArgReduceCUDA.environments:
            from dace.libraries.sort.environments.cub import DetectScratch
            ExpandArgReduceCUDA.environments = [DetectScratch]
        node.validate(parent_sdfg, parent_state)

        in_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == '_in')
        val_edge = next((e for e in parent_state.out_edges(node) if e.src_conn == '_out_val'), None)
        idx_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == '_out_idx')
        in_dtype = parent_sdfg.arrays[in_edge.data.data].dtype
        idx_dtype = parent_sdfg.arrays[idx_edge.data.data].dtype

        refusal = cuda_refusal(node, parent_state)
        if refusal is not None:
            raise NotImplementedError(refusal)

        state_id = parent_state.parent_graph.node_id(parent_state)
        idstr = f'{parent_sdfg.name}_{state_id}_{parent_state.node_id(node)}'
        vt, it = in_dtype.ctype, idx_dtype.ctype
        prototype = (f'DACE_EXPORTED gpuError_t __dace_argreduce_{idstr}(const {vt} *__ar_in, {vt} *__ar_val, '
                     f'long long *__ar_idx, long long __ar_items, gpuStream_t __ar_stream);')

        parent_sdfg.append_global_code(prototype + '\n')
        parent_sdfg.append_global_code(
            f'{prototype}\n'
            f'gpuError_t __dace_argreduce_{idstr}(const {vt} *__ar_in, {vt} *__ar_val, long long *__ar_idx, '
            f'long long __ar_items, gpuStream_t __ar_stream) {{\n'
            # No ``DACE_GPU_CHECK`` in this body: the macro reports through ``__state``, which a
            # free function in the CUDA unit does not have. The status is returned to the host
            # tasklet instead, and the ``DACE_GPU_CHECK`` around the call there reports it.
            f'    return ::dace::cub::arg_reduce<::dace::cub::{_OP_TAG[node.op]}>('
            f'__ar_in, __ar_val, __ar_idx, __ar_items, __ar_stream);\n'
            f'}}\n',
            'cuda')

        items = sym2cpp(_count(in_edge))
        val_out = '&__ar_val' if val_edge is not None else 'nullptr'
        code = ((f'{vt} __ar_val;\n' if val_edge is not None else '') + f'long long __ar_idx;\n'
                f'DACE_GPU_CHECK(__dace_argreduce_{idstr}(_in, {val_out}, &__ar_idx, (long long)({items}), '
                f'__dace_current_stream));\n' + (f'_out_val = __ar_val;\n' if val_edge is not None else '') +
                f'_out_idx = ({it})__ar_idx;')
        return nodes.Tasklet(
            label=f'{node.label}_cuda',
            inputs={'_in': dace.pointer(in_dtype)},
            outputs={c: None
                     for c in (('_out_val', '_out_idx') if val_edge is not None else ('_out_idx', ))},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


@library.node
class ArgReduce(nodes.LibraryNode):
    """Argmax / argmin over ``_in`` -> ``_out_val`` (value) + ``_out_idx`` (index).

    :cvar implementations: ``"pure"`` (CPU sequential scan), ``"OpenMP"`` (parallel lane-blocked
        pair reduction) and ``"CUDA"`` (CUB ArgMax/ArgMin).
        ``default_implementation = "pure"``.
    """

    implementations = {
        'pure': ExpandArgReducePure,
        'OpenMP': ExpandArgReduceOpenMP,
        'CUDA': ExpandArgReduceCUDA,
    }
    default_implementation = 'pure'

    #: Both answers are HOST scalars in every expansion, the CUDA one included: CUB leaves its
    #: result in device scratch and the wrapper copies it back before writing them.
    host_connectors = frozenset({'_out_val', '_out_idx'})

    op = properties.Property(dtype=str,
                             default='max',
                             choices={'max', 'min'},
                             desc="Reduction kind: 'max' (argmax) or 'min' (argmin).")

    transform = properties.Property(
        dtype=str,
        default='',
        choices=set(_TRANSFORM_CPP),
        desc="Unary function applied to each element as it is read ('' = none). Lets a caller "
        "arg-reduce over f(a[...]) without materialising f(a[...]) first.")

    def __init__(self, name: str, op: str = 'max', transform: str = '', location: Optional[str] = None):
        if op not in _OP_CPP:
            raise ValueError(f"ArgReduce: op must be 'max' or 'min', got {op!r}")
        if transform not in _TRANSFORM_CPP:
            raise ValueError(f"ArgReduce: transform must be one of {sorted(_TRANSFORM_CPP)}, got {transform!r}")
        super().__init__(name, location=location, inputs={'_in'}, outputs=OrderedSet(('_out_val', '_out_idx')))
        self.op = op
        self.transform = transform

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Require ``_in`` and ``_out_idx``; ``_out_val`` is optional.

        An arg-reduce is asked for its INDEX -- ``np.argmax`` returns only that, and a lifted
        ``if a[i] > best`` loop whose value carrier is dead afterwards produces only ``_out_idx``.
        Demanding both would force such a caller to materialize a value nothing reads.
        """
        in_conns = {e.dst_conn for e in state.in_edges(self) if e.dst_conn is not None}
        out_conns = {e.src_conn for e in state.out_edges(self) if e.src_conn is not None}
        if in_conns != {'_in'}:
            raise ValueError(f"{self.label}: ArgReduce requires exactly one input '_in', got {sorted(in_conns)}")
        if '_out_idx' not in out_conns or not out_conns <= {'_out_val', '_out_idx'}:
            raise ValueError(f"{self.label}: ArgReduce requires output '_out_idx' and allows '_out_val', "
                             f"got {sorted(out_conns)}")
        # The scan's trip count is the input memlet's volume, and a dynamic memlet's volume is an
        # upper bound (0 when unknown) rather than a count -- reject it here instead of silently
        # scanning the wrong number of elements.
        in_edge = next(e for e in state.in_edges(self) if e.dst_conn == '_in')
        if in_edge.data.dynamic:
            raise ValueError(f"{self.label}: ArgReduce reduces a fixed slice, so '_in' cannot be a dynamic memlet")
