# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ArgReduce``: argmax / argmin over a contiguous input -> (value, index).

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

Expansions:

* ``pure`` (CPU default): a CPP tasklet with a sequential scan over the
  flattened input -- correctness-first, no external dependency.
* ``CUDA`` (GPU): ``gpucub::DeviceReduce::ArgMax`` / ``ArgMin`` through
  ``dace::cub::arg_reduce``, which answers both scalar outputs. Unit-stride input only;
  a strided slice needs an input iterator CUB does not take for free.

Tie-breaking matches the TSVC sequential source ``if a[i] OP best: best = a[i];
idx = i`` -- a STRICT comparison, so the FIRST occurrence of the extreme value
wins (a strictly-greater/lesser test never updates on a tie). The ``_in`` index
is slice-local (``0 .. N-1``); the lift adds the slice base to recover the
original-array position.
"""
from typing import Optional

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


@library.expansion
class ExpandArgReducePure(ExpandTransformation):
    """Correctness-only CPU lowering: a sequential argmax/argmin scan."""

    environments = []

    @staticmethod
    def expansion(node: "ArgReduce", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        node.validate(parent_sdfg, parent_state)
        in_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == '_in')
        val_edge = next((e for e in parent_state.out_edges(node) if e.src_conn == '_out_val'), None)
        idx_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == '_out_idx')

        in_dtype = parent_sdfg.arrays[in_edge.data.data].dtype
        idx_dtype = parent_sdfg.arrays[idx_edge.data.data].dtype
        from dace.codegen.targets.cpp import sym2cpp
        sub = in_edge.data.subset
        n = sub.num_elements()
        n_str = sym2cpp(n)
        op = _OP_CPP[node.op]

        # Stride of the (1-D) input slice. ``_in`` points at the slice base, so a
        # strided slice ``a[lo:hi:s]`` reads element ``j`` at ``_in[j*s]``. A
        # unit-stride slice gets the tight contiguous loop (separate code path,
        # so the compiler can fully vectorise the hot case); a non-unit stride
        # multiplies the lane index by the stride -- a compile-time-constant
        # stride folds away, a symbolic stride stays a runtime multiply.
        step = sub.ranges[0][2] if len(sub.ranges) == 1 else 1
        try:
            unit_stride = (int(symbolic.simplify(step)) == 1)
        except (TypeError, ValueError):
            unit_stride = False
        access = "__i" if unit_stride else f"(__i * ({sym2cpp(step)}))"

        # ``_out_val`` / ``_out_idx`` are scalar (by-value) connectors. A strict
        # comparison keeps the FIRST extreme element (matches the sequential
        # source). ``_out_idx`` is the SLICE-LOCAL position ``0 .. n-1``.
        code = (f"{idx_dtype.ctype} __bidx = 0;\n"
                f"{in_dtype.ctype} __best = _in[0];\n"
                f"for ({idx_dtype.ctype} __i = 1; __i < {n_str}; ++__i) {{\n"
                f"    if (_in[{access}] {op} __best) {{ __best = _in[{access}]; __bidx = __i; }}\n"
                f"}}\n" + (f"_out_val = __best;\n" if val_edge is not None else "") + f"_out_idx = __bidx;")
        return nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs={'_in': dace.pointer(in_dtype)},
            outputs={c: None
                     for c in (('_out_val', '_out_idx') if val_edge is not None else ('_out_idx', ))},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandArgReduceOpenMP(ExpandTransformation):
    """Parallel CPU lowering: an OpenMP ``declare reduction`` over a (value, index) pair.

    argmax is associative on the PAIR, not on the value alone -- combining two partial results has
    to carry the index that produced the winning value, and break ties toward the LOWER index so
    the result matches the sequential scan element-for-element. A plain
    ``reduction(max:val)`` cannot express that, hence the custom combiner.
    """

    environments = []

    @staticmethod
    def expansion(node: "ArgReduce", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        node.validate(parent_sdfg, parent_state)
        in_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == '_in')
        val_edge = next((e for e in parent_state.out_edges(node) if e.src_conn == '_out_val'), None)
        idx_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == '_out_idx')

        in_dtype = parent_sdfg.arrays[in_edge.data.data].dtype
        idx_dtype = parent_sdfg.arrays[idx_edge.data.data].dtype
        from dace.codegen.targets.cpp import sym2cpp
        sub = in_edge.data.subset
        n_str = sym2cpp(sub.num_elements())
        op = _OP_CPP[node.op]

        step = sub.ranges[0][2] if len(sub.ranges) == 1 else 1
        try:
            unit_stride = (int(symbolic.simplify(step)) == 1)
        except (TypeError, ValueError):
            unit_stride = False
        access = "__i" if unit_stride else f"(__i * ({sym2cpp(step)}))"

        vt, it = in_dtype.ctype, idx_dtype.ctype
        # Tie-break on the lower index so a parallel combine reproduces the sequential answer
        # exactly; without it the result depends on how the range was split across threads.
        code = (f"struct __ar_pair {{ {vt} v; {it} i; }};\n"
                f"#pragma omp declare reduction(__ar_best : struct __ar_pair : \\\n"
                f"    omp_out = (omp_in.v {op} omp_out.v || "
                f"(omp_in.v == omp_out.v && omp_in.i < omp_out.i)) ? omp_in : omp_out) \\\n"
                f"    initializer(omp_priv = omp_orig)\n"
                f"struct __ar_pair __best;\n"
                f"__best.v = _in[0]; __best.i = 0;\n"
                f"#pragma omp parallel for reduction(__ar_best : __best)\n"
                f"for ({it} __i = 1; __i < {n_str}; ++__i) {{\n"
                f"    if (_in[{access}] {op} __best.v) {{ __best.v = _in[{access}]; __best.i = __i; }}\n"
                f"}}\n" + (f"_out_val = __best.v;\n" if val_edge is not None else "") + f"_out_idx = __best.i;")
        return nodes.Tasklet(
            label=f"{node.label}_openmp",
            inputs={'_in': dace.pointer(in_dtype)},
            outputs={c: None
                     for c in (('_out_val', '_out_idx') if val_edge is not None else ('_out_idx', ))},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


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

        sub = in_edge.data.subset
        step = sub.ranges[0][2] if len(sub.ranges) == 1 else 1
        if symbolic.equal(step, 1) is not True:
            raise NotImplementedError(
                f"ArgReduce CUDA reads a slice of stride {step}; gpucub::DeviceReduce::{_OP_CUB[node.op]} takes a "
                "contiguous pointer. Lower this one through 'pure' or 'OpenMP', or wrap the input in a "
                "gpucub::TransformInputIterator over a CountingInputIterator first.")

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

        items = sym2cpp(sub.num_elements())
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

    :cvar implementations: ``"pure"`` (CPU sequential scan), ``"OpenMP"`` (parallel pair
        reduction) and ``"CUDA"`` (CUB ArgMax/ArgMin, stubbed).
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

    def __init__(self, name: str, op: str = 'max', location: Optional[str] = None):
        if op not in _OP_CPP:
            raise ValueError(f"ArgReduce: op must be 'max' or 'min', got {op!r}")
        super().__init__(name, location=location, inputs={'_in'}, outputs=OrderedSet(('_out_val', '_out_idx')))
        self.op = op

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
