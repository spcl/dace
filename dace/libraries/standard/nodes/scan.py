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

import dace
from dace import library, nodes, symbolic
from dace.codegen.common import sym2cpp
from dace.properties import Property, EnumProperty
from dace.transformation.transformation import ExpandTransformation
import enum

# CUB env is imported lazily inside ``ExpandCUDA.expansion`` to break the
# ``dace.libraries.standard.nodes.scan`` ↔ ``dace.libraries.sort.environments.cub``
# circular import (cub.py pulls in standard.environments, which loads this module).
from dace.libraries.standard.environments.cpu import CPU as CPUEnv

# Connector names exposed for library-node builders.
INPUT_CONNECTOR_NAME = "_scan_in"
OUTPUT_CONNECTOR_NAME = "_scan_out"
#: Optional scalar input. When wired, the expansion emits an inclusive scan
#: with an initial accumulator value (``out[k] = init OP in[0] OP ... OP in[k]``),
#: which lets the LoopToScan rewrite skip its separate seed-add Map.
INIT_CONNECTOR_NAME = "_scan_init"


class ScanOp(enum.Enum):
    """Associative binary operations supported by the :class:`Scan` libnode."""
    SUM = 'sum'
    PRODUCT = 'product'
    MIN = 'min'
    MAX = 'max'


#: Map op enum to the C++ binary-op functor for ``std::inclusive_scan`` / ``std::exclusive_scan``
#: (used by the ``pure`` expansion). These are functor *values* (constructed via ``Op{}``).
_OP_TO_STD_CPP = {
    ScanOp.SUM: 'std::plus<>{}',
    ScanOp.PRODUCT: 'std::multiplies<>{}',
    ScanOp.MIN: '[](auto a, auto b){ return std::min(a, b); }',
    ScanOp.MAX: '[](auto a, auto b){ return std::max(a, b); }',
}

#: Map op enum to the suffix of the OpenMP-scan function in ``dace/scan.hpp``.
#: The ``CPU`` expansion emits ``dace::scan::inclusive_<suffix>`` / ``exclusive_<suffix>``.
_OP_TO_OMP_SUFFIX = {
    ScanOp.SUM: 'sum',
    ScanOp.PRODUCT: 'product',
    ScanOp.MIN: 'min',
    ScanOp.MAX: 'max',
}

#: Map op enum to the CUB-side binary functor for ``cub::DeviceScan::InclusiveScan``.
#: Routed through the ``DACE_CUB_*_OP`` macros from ``dace/cub_compat.cuh`` so the
#: same SDFG source builds against CUDA Toolkit 12 (CUB <= 2.x has ``cub::Sum`` /
#: ``cub::Min`` / ``cub::Max``) and 13 (CCCL 3.x dropped those in favour of
#: ``cuda::std::plus`` + device lambdas).
_OP_TO_CUB = {
    ScanOp.SUM: 'DACE_CUB_SUM_OP',
    ScanOp.PRODUCT: 'DACE_CUB_MUL_OP',
    ScanOp.MIN: 'DACE_CUB_MIN_OP',
    ScanOp.MAX: 'DACE_CUB_MAX_OP',
}

#: Default identity literal for ``exclusive`` scans, per op.
_OP_TO_IDENTITY_CPP = {
    ScanOp.SUM:
    '0',
    ScanOp.PRODUCT:
    '1',
    # ``min``/``max`` have no universal identity in C++ literal form -- callers must
    # supply ``identity`` explicitly for exclusive ``min``/``max`` scans.
    ScanOp.MIN:
    None,
    ScanOp.MAX:
    None,
}


def _validate_inputs_and_outputs(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG):
    """Resolve and validate the in/out edges; raise on any wiring/shape/dtype mismatch.

    ``_scan_init`` is optional; when present it must be a single scalar / length-1
    edge whose dtype matches the input array's element type.
    """
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    out_edges = [e for e in state.out_edges(node) if e.src_conn == OUTPUT_CONNECTOR_NAME]
    init_edges = [e for e in state.in_edges(node) if e.dst_conn == INIT_CONNECTOR_NAME]
    if len(in_edges) != 1 or len(out_edges) != 1:
        raise ValueError(f"Scan node {node.label} expects exactly one ``{INPUT_CONNECTOR_NAME}`` "
                         f"in-edge and one ``{OUTPUT_CONNECTOR_NAME}`` out-edge.")
    if len(init_edges) > 1:
        raise ValueError(f"Scan node {node.label}: ``{INIT_CONNECTOR_NAME}`` is optional but at "
                         f"most one in-edge is allowed; got {len(init_edges)}.")
    in_desc = sdfg.arrays[in_edges[0].data.data]
    out_desc = sdfg.arrays[out_edges[0].data.data]
    if not isinstance(in_desc, dace.data.Array) or not isinstance(out_desc, dace.data.Array):
        raise ValueError(f"Scan requires Array inputs/outputs; got {type(in_desc).__name__} -> "
                         f"{type(out_desc).__name__}.")
    if in_desc.dtype != out_desc.dtype:
        raise ValueError(f"Scan input/output dtype mismatch: {in_desc.dtype} vs {out_desc.dtype}.")
    if init_edges:
        init_desc = sdfg.arrays[init_edges[0].data.data]
        if init_desc.dtype != in_desc.dtype:
            raise ValueError(f"Scan node {node.label}: ``{INIT_CONNECTOR_NAME}`` dtype "
                             f"{init_desc.dtype} must match input dtype {in_desc.dtype}.")
    return in_desc, out_desc, in_edges[0], out_edges[0]


def _has_init(node: "Scan") -> bool:
    """``True`` iff this Scan instance has the optional ``_scan_init`` connector wired."""
    return INIT_CONNECTOR_NAME in node.in_connectors


def _resolve_length(node: "Scan", state: dace.SDFGState, _sdfg: dace.SDFG) -> str:
    """C++ expression for the number of elements ``N`` in the input edge."""
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    return sym2cpp(in_edges[0].data.subset.num_elements())


def _is_length_one(node: "Scan", state: dace.SDFGState) -> bool:
    """``True`` if the input subset is statically a single element. Single-element scans
    degenerate to a trivial copy (inclusive) or identity write (exclusive) -- no array
    iteration, no iterator-based template instantiation that would conflict with the
    codegen's scalar-typing of single-element subsets."""
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    n = symbolic.simplify(in_edges[0].data.subset.num_elements())
    return getattr(n, 'is_Integer', False) and int(n) == 1


def _degenerate_single_element_tasklet(node: "Scan", in_desc) -> nodes.Tasklet:
    """Return the single-element degenerate scan tasklet.

    For an inclusive single-element scan the result is just the input itself; for an
    exclusive single-element scan the result is the user-supplied identity. Both are
    expressed as Python tasklets so the codegen handles scalar connector typing
    naturally (no array indexing, no iterator templates).
    """
    if node.exclusive:
        # Treat the identity as a Python literal; the codegen casts via the connector
        # type (a scalar of ``in_desc.dtype``).
        seed_py = node.identity if node.identity is not None else (0 if node.op is ScanOp.SUM else
                                                                   (1 if node.op is ScanOp.PRODUCT else None))
        if seed_py is None:
            raise ValueError(f"Scan op {node.op.value!r} has no universal identity; set ``identity`` explicitly.")
        code = f"{OUTPUT_CONNECTOR_NAME} = {seed_py}"
    else:
        code = f"{OUTPUT_CONNECTOR_NAME} = {INPUT_CONNECTOR_NAME}"
    return nodes.Tasklet(node.name,
                         inputs={INPUT_CONNECTOR_NAME},
                         outputs={OUTPUT_CONNECTOR_NAME},
                         code=code,
                         language=dace.Language.Python)


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

    environments = [CPUEnv]

    @staticmethod
    def expansion(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        in_desc, _out_desc, in_edge, _out_edge = _validate_inputs_and_outputs(node, state, sdfg)
        if _is_length_one(node, state):
            return _degenerate_single_element_tasklet(node, in_desc)
        n_expr = _resolve_length(node, state, sdfg)
        op_cpp = _OP_TO_STD_CPP[node.op]
        stride_expr = sym2cpp(node.stride)
        is_stride_one = (symbolic.pystr_to_symbolic(stride_expr) == 1)

        if not is_stride_one:
            if node.exclusive:
                raise NotImplementedError("Scan(pure): exclusive with stride > 1 is not supported.")
            # Outer loop over residue classes ``_k in [0, s)``; inner sequential scan.
            # Initialise the accumulator from the first valid input in each class so we
            # don't need a per-op identity literal (matches the residue-class oracle).
            body = (f"{{ long _s = (long)({stride_expr}); long _n = (long)({n_expr});\n"
                    f"  if (_s <= 0) std::abort();\n"
                    f"  for (long _k = 0; _k < _s; ++_k) {{\n"
                    f"      if (_k >= _n) continue;\n"
                    f"      auto _acc = {INPUT_CONNECTOR_NAME}[_k];\n"
                    f"      {OUTPUT_CONNECTOR_NAME}[_k] = _acc;\n"
                    f"      for (long _j = _k + _s; _j < _n; _j += _s) {{\n"
                    f"          _acc = ({op_cpp})(_acc, {INPUT_CONNECTOR_NAME}[_j]);\n"
                    f"          {OUTPUT_CONNECTOR_NAME}[_j] = _acc;\n"
                    f"      }}\n"
                    f"  }}\n"
                    f"}}")
        elif node.exclusive:
            seed = _identity_expr(node, in_desc)
            body = (f"{{ auto _acc = {seed};\n"
                    f"  for (decltype({n_expr}) _i = 0; _i < ({n_expr}); ++_i) {{\n"
                    f"      auto _v = {INPUT_CONNECTOR_NAME}[_i];\n"
                    f"      {OUTPUT_CONNECTOR_NAME}[_i] = _acc;\n"
                    f"      _acc = ({op_cpp})(_acc, _v);\n"
                    f"  }}\n"
                    f"}}")
        elif _has_init(node):
            # Inclusive scan with explicit init: ``out[k] = init OP in[0] OP ... OP in[k]``.
            # The connector materialises a scalar; dereference to get the seed value.
            body = (f"{{ auto _acc = {INIT_CONNECTOR_NAME};\n"
                    f"  for (decltype({n_expr}) _i = 0; _i < ({n_expr}); ++_i) {{\n"
                    f"      _acc = ({op_cpp})(_acc, {INPUT_CONNECTOR_NAME}[_i]);\n"
                    f"      {OUTPUT_CONNECTOR_NAME}[_i] = _acc;\n"
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
        inputs = {INPUT_CONNECTOR_NAME}
        if _has_init(node):
            inputs.add(INIT_CONNECTOR_NAME)
        return nodes.Tasklet(
            node.name,
            inputs=inputs,
            outputs={OUTPUT_CONNECTOR_NAME},
            code=body,
            language=dace.Language.CPP,
        )


@library.expansion
class ExpandCPU(ExpandTransformation):
    """OpenMP 5.0 parallel scan via the vendored ``dace::scan`` runtime header.

    Emits a single call into ``dace::scan::{inclusive,exclusive}_{sum,product,min,max}``
    -- templated header-only routines that wrap ``#pragma omp parallel for simd
    reduction(inscan, op:acc)`` + ``#pragma omp scan {inclusive,exclusive}(acc)``.
    GCC 10+ / Clang 11+ / ICX 2021+ implement OpenMP 5.0's ``scan`` directive as a
    two-level chunked scan: phase-1 per-thread sequential scans, phase-2 small
    parallel-prefix over per-chunk totals, phase-3 parallel offset adjustment.
    Work O(2N), depth O(N/P + log P) -- the canonical multi-core CPU prefix scan.
    """

    environments = [CPUEnv]

    @staticmethod
    def expansion(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        in_desc, _out_desc, _in_edge, _out_edge = _validate_inputs_and_outputs(node, state, sdfg)
        if _is_length_one(node, state):
            return _degenerate_single_element_tasklet(node, in_desc)
        n_expr = _resolve_length(node, state, sdfg)
        suffix = _OP_TO_OMP_SUFFIX[node.op]
        stride_expr = sym2cpp(node.stride)
        is_stride_one = (symbolic.pystr_to_symbolic(stride_expr) == 1)

        op_cpp = _OP_TO_STD_CPP[node.op]
        if not is_stride_one:
            if node.exclusive:
                raise NotImplementedError("Scan: ``exclusive=True`` with ``stride > 1`` is not yet supported.")
            if _has_init(node):
                raise NotImplementedError("Scan: ``_scan_init`` with ``stride > 1`` is not yet supported.")
            call = (f"::dace::scan::strided_inclusive_{suffix}("
                    f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, ({n_expr}), ({stride_expr}));")
        elif node.exclusive:
            seed = _identity_expr(node, in_desc)
            call = (f"::dace::scan::exclusive_{suffix}("
                    f"{INPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME} + ({n_expr}), "
                    f"{OUTPUT_CONNECTOR_NAME}, {seed});")
        elif _has_init(node):
            # Inclusive scan with init: use C++17 ``std::inclusive_scan`` 5-arg overload.
            # Sequential; parallel-with-init would require extending the runtime header.
            call = (f"#include <numeric>\n"
                    f"std::inclusive_scan({INPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME} + ({n_expr}), "
                    f"{OUTPUT_CONNECTOR_NAME}, {op_cpp}, {INIT_CONNECTOR_NAME});")
        else:
            call = (f"::dace::scan::inclusive_{suffix}("
                    f"{INPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME} + ({n_expr}), "
                    f"{OUTPUT_CONNECTOR_NAME});")
        inputs = {INPUT_CONNECTOR_NAME}
        if _has_init(node):
            inputs.add(INIT_CONNECTOR_NAME)
        return nodes.Tasklet(
            node.name,
            inputs=inputs,
            outputs={OUTPUT_CONNECTOR_NAME},
            code=call,
            language=dace.Language.CPP,
        )


@library.expansion
class ExpandCUDA(ExpandTransformation):
    """``cub::DeviceScan::InclusiveScan`` / ``ExclusiveScan`` over device-global memory.

    Temporary storage is obtained from the per-libnode-class, per-stream CUB scratch pool
    tagged ``ScanTag`` (see :file:`dace/runtime/include/dace/cub_scratch.cuh` and the
    :class:`ScanScratch` environment): the default-stream entry is pre-allocated to 128 MB
    at SDFG init; additional streams allocate lazily on first use. Each per-stream entry is
    reused across every ``Scan`` call on that stream, grown in place if a request exceeds
    the current allocation, and released at SDFG exit. The libnode threads
    ``__dace_current_stream`` to both the scratch lookup and the underlying ``cub::DeviceScan``
    call, so concurrent launches on different streams cannot race on the pool.
    """

    # Populated lazily in :meth:`expansion` (and below) to dodge the sort↔standard cycle.
    environments = []

    @staticmethod
    def expansion(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        if not ExpandCUDA.environments:
            from dace.libraries.sort.environments.cub import ScanScratch
            ExpandCUDA.environments = [ScanScratch]
        in_desc, _out_desc, _in_edge, _out_edge = _validate_inputs_and_outputs(node, state, sdfg)
        if _is_length_one(node, state):
            return _degenerate_single_element_tasklet(node, in_desc)
        n_expr = _resolve_length(node, state, sdfg)
        op_cub = _OP_TO_CUB[node.op]
        stride_expr = sym2cpp(node.stride)
        is_stride_one = (symbolic.pystr_to_symbolic(stride_expr) == 1)

        if not is_stride_one:
            # ``cub::DeviceScan`` only handles a single contiguous scan; the
            # strided / residue-class shape has its own implementation
            # (``ExpandCUDAStrided``). Direct the user to the right knob
            # rather than silently mis-dispatch through a unit-stride cub
            # call that would walk past each residue's boundary.
            raise NotImplementedError("Scan(CUDA, unit-stride only): set ``implementation = 'CUDA_strided'`` on this "
                                      "Scan libnode (or use the AUTO selector in LoopToScan); stride > 1 dispatches "
                                      "to a separate expansion that calls ``dace::cuda_scan::strided_inclusive_<op>`` "
                                      "via the ``dace/cuda/scan_strided.cu`` auxiliary translation unit.")

        if node.exclusive:
            seed = _identity_expr(node, in_desc)
            scan_call = (f"::cub::DeviceScan::ExclusiveScan(_sc_scratch, _sc_needed, "
                         f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, {op_cub}, {seed}, "
                         f"({n_expr}), __dace_current_stream);")
            query_call = (f"::cub::DeviceScan::ExclusiveScan(nullptr, _sc_needed, "
                          f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, {op_cub}, {seed}, "
                          f"({n_expr}), __dace_current_stream);")
        elif _has_init(node):
            # Inclusive scan with init. ``cub::DeviceScan::InclusiveScanInit`` is the
            # direct API (CUB >= 2.0 / CUDA 12+); on older CUB it'd need an
            # ``ExclusiveScan`` + tail-add fallback, which can be added when
            # supporting CUDA 11 becomes a requirement.
            scan_call = (f"::cub::DeviceScan::InclusiveScanInit(_sc_scratch, _sc_needed, "
                         f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, {op_cub}, "
                         f"{INIT_CONNECTOR_NAME}, ({n_expr}), __dace_current_stream);")
            query_call = (f"::cub::DeviceScan::InclusiveScanInit(nullptr, _sc_needed, "
                          f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, {op_cub}, "
                          f"{INIT_CONNECTOR_NAME}, ({n_expr}), __dace_current_stream);")
        else:
            scan_call = (f"::cub::DeviceScan::InclusiveScan(_sc_scratch, _sc_needed, "
                         f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, {op_cub}, "
                         f"({n_expr}), __dace_current_stream);")
            query_call = (f"::cub::DeviceScan::InclusiveScan(nullptr, _sc_needed, "
                          f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, {op_cub}, "
                          f"({n_expr}), __dace_current_stream);")
        code = (f"size_t _sc_needed = 0;\n"
                f"{query_call}\n"
                f"void* _sc_scratch = ::dace::cub::get_scratch<::dace::cub::ScanTag>("
                f"_sc_needed, __dace_current_stream);\n"
                f"{scan_call}")
        inputs = {INPUT_CONNECTOR_NAME}
        if _has_init(node):
            inputs.add(INIT_CONNECTOR_NAME)
        return nodes.Tasklet(
            node.name,
            inputs=inputs,
            outputs={OUTPUT_CONNECTOR_NAME},
            code=code,
            language=dace.Language.CPP,
        )


@library.expansion
class ExpandCUDAStrided(ExpandTransformation):
    """Strided GPU scan: ``s`` independent residue-class scans, one device
    thread per class.

    Uses the ``::dace::cuda_scan::strided_inclusive_<op>`` kernels declared in
    :file:`dace/runtime/include/dace/cuda/scan.cuh` and called via the
    ``extern "C"`` wrappers in
    :file:`dace/runtime/include/dace/cuda/scan_strided.cu`. The wrappers are
    nvcc-compiled and linked into the SDFG library through the new
    ``library.environment`` ``auxiliary_sources`` field on
    :class:`ScanStrided`. The host ``.cpp`` translation unit therefore only
    sees a regular C function call -- no ``<<<>>>`` syntax, no ``__global__``
    symbols, no ``cub/cub.cuh`` dependency.

    Only inclusive scans are supported (mirroring the runtime header). Use
    ``ExpandCUDA`` (cub-based) for unit-stride scans.
    """

    # Populated lazily to avoid a load-order cycle with the env module.
    environments = []

    @staticmethod
    def expansion(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        if not ExpandCUDAStrided.environments:
            from dace.libraries.standard.environments.scan_strided import ScanStrided
            ExpandCUDAStrided.environments = [ScanStrided]
        in_desc, _out_desc, _in_edge, _out_edge = _validate_inputs_and_outputs(node, state, sdfg)
        if _is_length_one(node, state):
            return _degenerate_single_element_tasklet(node, in_desc)
        n_expr = _resolve_length(node, state, sdfg)
        stride_expr = sym2cpp(node.stride)
        if node.exclusive:
            raise NotImplementedError("Scan(CUDA_strided): ``exclusive=True`` is not yet supported.")
        if _has_init(node):
            raise NotImplementedError("Scan(CUDA_strided): ``_scan_init`` is not yet supported.")
        dtype = in_desc.dtype
        # The wrapper set in ``scan_strided.cu`` is pre-instantiated for these
        # dtypes. Extending it is mechanical -- add a ``_DACE_DEFINE_STRIDED_SCAN``
        # macro instantiation in the .cu and a matching ``_DACE_DECL_STRIDED_SCAN``
        # in the .h header.
        if dtype == dace.float64:
            dtype_suffix = 'f64'
        elif dtype == dace.float32:
            dtype_suffix = 'f32'
        elif dtype == dace.int64:
            dtype_suffix = 'i64'
        elif dtype == dace.int32:
            dtype_suffix = 'i32'
        else:
            raise NotImplementedError(
                f"Scan(CUDA_strided): dtype {dtype} not in the pre-instantiated wrapper set "
                f"(f64 / f32 / i64 / i32). Extend ``dace/runtime/include/dace/cuda/scan_strided.cu`` "
                f"and ``...decls.h``.")
        suffix = _OP_TO_OMP_SUFFIX[node.op]
        code = (f"dace_cuda_strided_inclusive_{suffix}_{dtype_suffix}("
                f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, "
                f"(long)({n_expr}), (long)({stride_expr}), __dace_current_stream);")
        return nodes.Tasklet(
            node.name,
            inputs={INPUT_CONNECTOR_NAME},
            outputs={OUTPUT_CONNECTOR_NAME},
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
    #: Optional scalar input connector; wire to a length-1 / scalar read to make
    #: the inclusive scan fold an explicit init value in (``out[k] = init OP in[0]
    #: OP ... OP in[k]``). Lets LoopToScan skip its seed-add Map.
    INIT_CONNECTOR_NAME = INIT_CONNECTOR_NAME

    op = EnumProperty(dtype=ScanOp, default=ScanOp.SUM, desc="Associative binary op for the scan.")
    exclusive = Property(dtype=bool, default=False, desc="If True, output an exclusive scan (out[0] = identity).")
    identity = Property(dtype=object,
                        default=None,
                        allow_none=True,
                        desc="Exclusive-scan identity element. Required for MIN/MAX exclusive scans.")
    stride = Property(dtype=object,
                      default=1,
                      allow_none=False,
                      desc="Per-element stride for the scan recurrence. Default ``1`` is the "
                      "contiguous case (``out[i+1] = out[i] OP in[i]``). Values ``s > 1`` express "
                      "``out[i+s] = out[i] OP in[i]``: the ``s`` residue classes mod ``s`` form "
                      "independent scans (CPU expansion runs them in parallel via OpenMP). "
                      "The expansion emits a runtime ``s > 0`` ``std::abort()`` check; passing a "
                      "non-positive stride at runtime terminates the program before the scan "
                      "starts. Exclusive strided scans (``exclusive=True`` with ``stride > 1``) "
                      "are not yet supported.")

    implementations = {
        "CPU": ExpandCPU,
        "CUDA": ExpandCUDA,
        "CUDA_strided": ExpandCUDAStrided,
        "pure": ExpandPure,
    }
    default_implementation = 'CPU'

    def __init__(self,
                 name: str = 'Scan',
                 op: ScanOp = ScanOp.SUM,
                 exclusive: bool = False,
                 identity=None,
                 *args,
                 **kwargs):
        super().__init__(name, *args, inputs={INPUT_CONNECTOR_NAME}, outputs={OUTPUT_CONNECTOR_NAME}, **kwargs)
        self.op = op
        self.exclusive = exclusive
        self.identity = identity

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState):
        _validate_inputs_and_outputs(self, state, sdfg)
