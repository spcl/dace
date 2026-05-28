# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Scan`` library node: in-place / out-of-place prefix scan over a 1-D array.

A *scan* (also "prefix-reduction") computes per-position partial reductions of an
input array: for an inclusive scan with op ``+``::

    out[k] = in[0] + in[1] + ... + in[k]    for k in 0..N-1

For an exclusive scan with identity ``e``::

    out[k] = e + in[0] + ... + in[k-1]      for k in 0..N-1  (out[0] == e)

This is the canonical *parallel-prefix* primitive: a true scan keeps every
partial sum visible, so a downstream consumer can read any ``out[k]``. It is
*not* equivalent to a single ``Reduce`` -- that collapses to one scalar and
loses the per-position values.

Why a library node, not a transformation: GPU implementations (Blelloch upsweep
+ downsweep) have O(N log N) work / O(log N) depth and are non-trivial to
emit from generic SDFG primitives; the CPU implementation is a tight
memory-bandwidth-bound sequential loop. Both already exist as battle-tested
library functions (``cub::DeviceScan`` on GPU, ``std::partial_sum`` /
``std::inclusive_scan`` / ``std::exclusive_scan`` on CPU). The libnode is the
right level of abstraction.

Implementations:

- ``CPU`` -- ``std::inclusive_scan`` / ``std::exclusive_scan`` (C++17 ``<numeric>``,
  sequential). Memory-bandwidth-bound on modern CPUs; the right choice for the
  cloudsc-style vertical-flux pattern (N ~ 137) where parallel scan overhead
  dwarfs its speedup.
- ``CUDA`` -- ``cub::DeviceScan::InclusiveScan`` / ``ExclusiveScan`` (Blelloch
  upsweep + downsweep; memory-bandwidth-bound on modern NVIDIA GPUs at
  GKeys/s rates).
- ``pure`` -- portable single-loop fallback (used when neither CPU nor CUDA
  expansion applies, e.g. for FPGA backends in v1).

For supported binary ops, see :data:`_OP_TO_STD_CPP` and :data:`_OP_TO_CUB`. The
op must be associative -- ``+``, ``*``, ``min``, ``max`` -- so the order of the
partial reductions does not change the result.
"""
from typing import Tuple

import dace
from dace import library, nodes
from dace.codegen.common import sym2cpp
from dace.properties import Property, EnumProperty
from dace.transformation.transformation import ExpandTransformation
import enum

from dace.libraries.sort.environments.cub import CUB
from dace.libraries.standard.environments.cpu import CPU as CPUEnv
from dace.libraries.standard.environments.scan import ScanCPU


# Connector names exposed for library-node builders.
INPUT_CONNECTOR_NAME = "_scan_in"
OUTPUT_CONNECTOR_NAME = "_scan_out"


class ScanOp(enum.Enum):
    """Associative binary operations supported by the :class:`Scan` libnode."""
    SUM = 'sum'
    PRODUCT = 'product'
    MIN = 'min'
    MAX = 'max'


#: Map op enum to the C++ binary-op functor for ``std::inclusive_scan`` / ``std::exclusive_scan``.
#: These are functor *values* (constructed via ``Op{}``) usable as a callable expression --
#: stripping the parens avoids being mis-parsed as a function-style cast in C++.
_OP_TO_STD_CPP = {
    ScanOp.SUM: 'std::plus<>{}',
    ScanOp.PRODUCT: 'std::multiplies<>{}',
    ScanOp.MIN: '[](auto a, auto b){ return std::min(a, b); }',
    ScanOp.MAX: '[](auto a, auto b){ return std::max(a, b); }',
}

#: Map op enum to the CUB-side binary functor for ``cub::DeviceScan::InclusiveScan``.
_OP_TO_CUB = {
    ScanOp.SUM: 'cub::Sum()',
    ScanOp.PRODUCT: '[] __device__(auto a, auto b){ return a * b; }',
    ScanOp.MIN: 'cub::Min()',
    ScanOp.MAX: 'cub::Max()',
}

#: Default identity literal for ``exclusive`` scans, per op.
_OP_TO_IDENTITY_CPP = {
    ScanOp.SUM: '0',
    ScanOp.PRODUCT: '1',
    # ``min``/``max`` have no universal identity in C++ literal form -- callers must
    # supply ``identity`` explicitly for exclusive ``min``/``max`` scans.
    ScanOp.MIN: None,
    ScanOp.MAX: None,
}


def _validate_inputs_and_outputs(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG):
    """Resolve and validate the in/out edges; raise on any wiring/shape/dtype mismatch."""
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    out_edges = [e for e in state.out_edges(node) if e.src_conn == OUTPUT_CONNECTOR_NAME]
    if len(in_edges) != 1 or len(out_edges) != 1:
        raise ValueError(f"Scan node {node.label} expects exactly one ``{INPUT_CONNECTOR_NAME}`` "
                         f"in-edge and one ``{OUTPUT_CONNECTOR_NAME}`` out-edge.")
    in_desc = sdfg.arrays[in_edges[0].data.data]
    out_desc = sdfg.arrays[out_edges[0].data.data]
    if not isinstance(in_desc, dace.data.Array) or not isinstance(out_desc, dace.data.Array):
        raise ValueError(f"Scan requires Array inputs/outputs; got {type(in_desc).__name__} -> "
                         f"{type(out_desc).__name__}.")
    if in_desc.dtype != out_desc.dtype:
        raise ValueError(f"Scan input/output dtype mismatch: {in_desc.dtype} vs {out_desc.dtype}.")
    return in_desc, out_desc, in_edges[0], out_edges[0]


def _resolve_length(node: "Scan", state: dace.SDFGState, _sdfg: dace.SDFG) -> str:
    """C++ expression for the number of elements ``N`` in the input edge."""
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    return sym2cpp(in_edges[0].data.subset.num_elements())


def _identity_expr(node: "Scan", in_desc) -> str:
    """C++ expression for the exclusive-scan identity element.

    The user-supplied ``identity`` property wins. Otherwise the per-op default
    from :data:`_OP_TO_IDENTITY_CPP` is used; if the op has no universal
    identity (``min``/``max``) the user *must* set ``identity``.
    """
    if node.identity is not None:
        return str(node.identity)
    default = _OP_TO_IDENTITY_CPP[node.op]
    if default is None:
        raise ValueError(f"Scan op {node.op.value!r} has no universal identity in C++ literal form; "
                         f"set ``identity`` explicitly when using ``exclusive=True``.")
    # Cast to the element type for completeness (avoids signed/unsigned warnings on integer dtypes).
    return f"static_cast<{in_desc.dtype.ctype}>({default})"


@library.expansion
class ExpandPure(ExpandTransformation):
    """Portable fallback: a hand-written single-loop scan."""

    environments = [ScanCPU]

    @staticmethod
    def expansion(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        in_desc, _out_desc, in_edge, _out_edge = _validate_inputs_and_outputs(node, state, sdfg)
        n_expr = _resolve_length(node, state, sdfg)
        op_cpp = _OP_TO_STD_CPP[node.op]
        if node.exclusive:
            seed = _identity_expr(node, in_desc)
            body = (f"{{ auto _acc = {seed};\n"
                    f"  for (decltype({n_expr}) _i = 0; _i < ({n_expr}); ++_i) {{\n"
                    f"      auto _v = {INPUT_CONNECTOR_NAME}[_i];\n"
                    f"      {OUTPUT_CONNECTOR_NAME}[_i] = _acc;\n"
                    f"      _acc = ({op_cpp})(_acc, _v);\n"
                    f"  }}\n"
                    f"}}")
        else:
            body = (f"{{ auto _acc = {INPUT_CONNECTOR_NAME}[0];\n"
                    f"  {OUTPUT_CONNECTOR_NAME}[0] = _acc;\n"
                    f"  for (decltype({n_expr}) _i = 1; _i < ({n_expr}); ++_i) {{\n"
                    f"      _acc = ({op_cpp})(_acc, {INPUT_CONNECTOR_NAME}[_i]);\n"
                    f"      {OUTPUT_CONNECTOR_NAME}[_i] = _acc;\n"
                    f"  }}\n"
                    f"}}")
        return nodes.Tasklet(
            node.name,
            inputs={INPUT_CONNECTOR_NAME: dace.dtypes.pointer(in_desc.dtype)},
            outputs={OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(in_desc.dtype)},
            code=body,
            language=dace.Language.CPP,
        )


@library.expansion
class ExpandCPU(ExpandTransformation):
    """C++17 ``std::inclusive_scan`` / ``std::exclusive_scan`` (sequential, vectorisable)."""

    environments = [CPUEnv, ScanCPU]

    @staticmethod
    def expansion(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        in_desc, _out_desc, _in_edge, _out_edge = _validate_inputs_and_outputs(node, state, sdfg)
        n_expr = _resolve_length(node, state, sdfg)
        op_cpp = _OP_TO_STD_CPP[node.op]
        if node.exclusive:
            seed = _identity_expr(node, in_desc)
            call = (f"std::exclusive_scan({INPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME} + ({n_expr}), "
                    f"{OUTPUT_CONNECTOR_NAME}, {seed}, {op_cpp});")
        else:
            call = (f"std::inclusive_scan({INPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME} + ({n_expr}), "
                    f"{OUTPUT_CONNECTOR_NAME}, {op_cpp});")
        return nodes.Tasklet(
            node.name,
            inputs={INPUT_CONNECTOR_NAME: dace.dtypes.pointer(in_desc.dtype)},
            outputs={OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(in_desc.dtype)},
            code=call,
            language=dace.Language.CPP,
        )


@library.expansion
class ExpandCUDA(ExpandTransformation):
    """``cub::DeviceScan::InclusiveScan`` / ``ExclusiveScan`` over device-global memory."""

    environments = [CUB]

    @staticmethod
    def expansion(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        in_desc, _out_desc, _in_edge, _out_edge = _validate_inputs_and_outputs(node, state, sdfg)
        n_expr = _resolve_length(node, state, sdfg)
        op_cub = _OP_TO_CUB[node.op]
        if node.exclusive:
            seed = _identity_expr(node, in_desc)
            scan_call = (f"::cub::DeviceScan::ExclusiveScan(_sc_temp, _sc_temp_bytes, "
                         f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, {op_cub}, {seed}, ({n_expr}));")
            query_call = (f"::cub::DeviceScan::ExclusiveScan(nullptr, _sc_temp_bytes, "
                          f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, {op_cub}, {seed}, ({n_expr}));")
        else:
            scan_call = (f"::cub::DeviceScan::InclusiveScan(_sc_temp, _sc_temp_bytes, "
                         f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, {op_cub}, ({n_expr}));")
            query_call = (f"::cub::DeviceScan::InclusiveScan(nullptr, _sc_temp_bytes, "
                          f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, {op_cub}, ({n_expr}));")
        code = (f"size_t _sc_temp_bytes = 0;\n"
                f"{query_call}\n"
                f"void* _sc_temp = nullptr;\n"
                f"cudaMalloc(&_sc_temp, _sc_temp_bytes);\n"
                f"{scan_call}\n"
                f"cudaFree(_sc_temp);")
        return nodes.Tasklet(
            node.name,
            inputs={INPUT_CONNECTOR_NAME: dace.dtypes.pointer(in_desc.dtype)},
            outputs={OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(in_desc.dtype)},
            code=code,
            language=dace.Language.CPP,
        )


@library.node
class Scan(nodes.LibraryNode):
    """Per-position prefix reduction over a 1-D array.

    Inputs / outputs:

    - ``_scan_in``:  input 1-D contiguous array of length ``N``.
    - ``_scan_out``: output 1-D contiguous array, same dtype, same shape.

    Properties:

    - ``op``: one of :class:`ScanOp` (``SUM`` / ``PRODUCT`` / ``MIN`` / ``MAX``).
    - ``exclusive``: ``False`` (inclusive: ``out[k] = in[0] OP ... OP in[k]``);
      ``True`` (exclusive: ``out[0] = identity``, ``out[k] = identity OP in[0] OP ... OP in[k-1]``).
    - ``identity``: the exclusive-scan seed. Defaults to ``0`` for ``SUM`` and ``1`` for
      ``PRODUCT``; ``MIN``/``MAX`` exclusive scans require this to be set explicitly.

    Implementations:

    - ``'CPU'`` (default) -- ``std::inclusive_scan`` / ``std::exclusive_scan`` (C++17 ``<numeric>``).
    - ``'CUDA'``           -- ``cub::DeviceScan::InclusiveScan`` / ``ExclusiveScan``.
    - ``'pure'``           -- portable single-loop fallback.

    The libnode is contractually pure: no aliasing between ``in`` and ``out`` is required
    (and not assumed), and no other state is read or written.
    """

    INPUT_CONNECTOR_NAME = INPUT_CONNECTOR_NAME
    OUTPUT_CONNECTOR_NAME = OUTPUT_CONNECTOR_NAME

    op = EnumProperty(dtype=ScanOp, default=ScanOp.SUM, desc="Associative binary op for the scan.")
    exclusive = Property(dtype=bool, default=False,
                         desc="If True, output an exclusive scan (out[0] = identity).")
    identity = Property(dtype=object, default=None, allow_none=True,
                        desc="Exclusive-scan identity element. Required for MIN/MAX exclusive scans.")

    implementations = {"CPU": ExpandCPU, "CUDA": ExpandCUDA, "pure": ExpandPure}
    default_implementation = 'CPU'

    def __init__(self, name: str = 'Scan', op: ScanOp = ScanOp.SUM, exclusive: bool = False,
                 identity=None, *args, **kwargs):
        super().__init__(name, *args, inputs={INPUT_CONNECTOR_NAME}, outputs={OUTPUT_CONNECTOR_NAME}, **kwargs)
        self.op = op
        self.exclusive = exclusive
        self.identity = identity

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState):
        _validate_inputs_and_outputs(self, state, sdfg)
