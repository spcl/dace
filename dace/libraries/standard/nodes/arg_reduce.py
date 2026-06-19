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
* ``CUDA`` (GPU): ``cub::DeviceReduce::ArgMax`` / ``ArgMin``, splitting the
  returned ``KeyValuePair`` into the two scalar outputs (stubbed until the GPU
  path is exercised; pin ``pure`` meanwhile).

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

_OP_CPP = {'max': '>', 'min': '<'}
_OP_CUB = {'max': 'ArgMax', 'min': 'ArgMin'}


@library.expansion
class ExpandArgReducePure(ExpandTransformation):
    """Correctness-only CPU lowering: a sequential argmax/argmin scan."""

    environments = []

    @staticmethod
    def expansion(node: "ArgReduce", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        node.validate(parent_sdfg, parent_state)
        in_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == '_in')
        val_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == '_out_val')
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
                f"}}\n"
                f"_out_val = __best;\n"
                f"_out_idx = __bidx;")
        return nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs={'_in': dace.pointer(in_dtype)},
            outputs={
                '_out_val': None,
                '_out_idx': None
            },
            code=code,
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandArgReduceCUDA(ExpandTransformation):
    """GPU lowering via ``cub::DeviceReduce::ArgMax`` / ``ArgMin`` (stub)."""

    environments = []

    @staticmethod
    def expansion(node: "ArgReduce", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        raise NotImplementedError(
            f"ArgReduce CUDA expansion (cub::DeviceReduce::{_OP_CUB[node.op]}) is not yet wired up; "
            "the CUB ArgMax/ArgMin call returns a KeyValuePair<int, T> to split into _out_val / _out_idx. "
            "CUB ArgMax/ArgMin take a contiguous input pointer; a non-unit-stride slice needs a strided "
            "input iterator (e.g. cub::CountingInputIterator + a TransformInputIterator computing base+j*stride) "
            "before the call. Pin the 'pure' expansion (expand_library_nodes(implementation='pure')) for CPU.")


@library.node
class ArgReduce(nodes.LibraryNode):
    """Argmax / argmin over ``_in`` -> ``_out_val`` (value) + ``_out_idx`` (index).

    :cvar implementations: ``"pure"`` (CPU sequential scan) and ``"CUDA"``
        (CUB ArgMax/ArgMin, stubbed). ``default_implementation = "pure"``.
    """

    implementations = {
        'pure': ExpandArgReducePure,
        'CUDA': ExpandArgReduceCUDA,
    }
    default_implementation = 'pure'

    op = properties.Property(dtype=str,
                             default='max',
                             choices={'max', 'min'},
                             desc="Reduction kind: 'max' (argmax) or 'min' (argmin).")

    def __init__(self, name: str, op: str = 'max', location: Optional[str] = None):
        if op not in _OP_CPP:
            raise ValueError(f"ArgReduce: op must be 'max' or 'min', got {op!r}")
        super().__init__(name, location=location, inputs={'_in'}, outputs={'_out_val', '_out_idx'})
        self.op = op

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Require exactly ``_in`` and both ``_out_val`` / ``_out_idx`` connected."""
        in_conns = {e.dst_conn for e in state.in_edges(self) if e.dst_conn is not None}
        out_conns = {e.src_conn for e in state.out_edges(self) if e.src_conn is not None}
        if in_conns != {'_in'}:
            raise ValueError(f"{self.label}: ArgReduce requires exactly one input '_in', got {sorted(in_conns)}")
        if out_conns != {'_out_val', '_out_idx'}:
            raise ValueError(f"{self.label}: ArgReduce requires outputs '_out_val' and '_out_idx', "
                             f"got {sorted(out_conns)}")
