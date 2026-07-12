# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared helpers for CopyLibraryNode and MemsetLibraryNode expansions.
"""
from typing import Callable, List, Optional, Tuple

import sympy

import dace
from dace import dtypes, symbolic
from dace.codegen.common import sym2cpp
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.subsets import Range

# Ambient GPU stream symbol the libnode CUDA expansions reference; both the
# legacy and experimental codegens consume this exact name for stream wiring.
CURRENT_STREAM_NAME = "__dace_current_stream"

#: Byte threshold at/above which a contiguous CPU<->CPU ``memcpy`` / ``memset`` is
#: split across OpenMP threads (see ``ExpandMemcpyParallelCPU`` / ``ExpandMemsetParallelCPU``).
#: Below it a single ``std::memcpy`` / ``std::memset`` wins (the OpenMP fork/join is
#: ~2-10us and the source usually stays cache-resident); above it a chunked parallel
#: copy saturates the memory controllers (on a 72-core Grace die a lone memcpy reaches
#: ~30 GB/s vs ~450 GB/s aggregate). The emitted parallel code also carries a runtime
#: guard on this value, so a symbolic size that turns out small still takes the serial path.
PARALLEL_COPY_MIN_BYTES = 1 << 20


def is_parallel_cpu_transfer_size(num_bytes: dace.symbolic.SymbolicType) -> bool:
    """Whether a contiguous CPU transfer of ``num_bytes`` should take the parallel OpenMP path.

    ``True`` when the size is symbolic (unknown at compile time -- the emitted code carries a
    runtime threshold guard so a small symbolic size still runs serial) or statically
    ``>= PARALLEL_COPY_MIN_BYTES``; ``False`` only when statically below the threshold.

    :param num_bytes: total contiguous byte count (constant or symbolic).
    :returns: ``True`` to route to the parallel expansion, ``False`` to keep the serial call.
    """
    try:
        return int(dace.symbolic.simplify(num_bytes)) >= PARALLEL_COPY_MIN_BYTES
    except (TypeError, ValueError):
        return True


#: Per-chunk byte granularity of the parallel CPU transfer map. Each ``CPU_Multicore``
#: map iteration copies/zeroes one contiguous chunk of this many bytes via a single
#: ``std::memcpy`` / ``std::memset`` (preserving SIMD + streaming stores). The map's
#: ``schedule(static)`` hands each thread a contiguous run of chunks, so a thread's overall
#: region stays contiguous while there are ``ceil(total / chunk)`` chunks to balance across a
#: 72-core die. 256 KiB amortizes per-call overhead and exceeds a core's working set; glibc's
#: aarch64 ``memcpy`` uses temporal ``ldp``/``stp`` (no non-temporal path) and ``memset`` uses
#: ``dc zva`` streaming-zero, so plain per-chunk calls are near-optimal and portable -- the
#: parallelism across cores (~30 -> ~450 GB/s on Grace), not non-temporal stores, is the win.
PARALLEL_COPY_CHUNK_BYTES = 1 << 18

#: Map-parameter name of the parallel transfer chunk index. Distinct from the libnode
#: connector names (``_cpy_in`` / ``_cpy_out`` / ``_mset_out``) and the inner tasklet
#: connectors (``_in`` / ``_out``) so it never collides with a wrapper-SDFG array name.
_CHUNK_INDEX_NAME = "__pchunk"


def make_parallel_cpu_transfer_sdfg(label: str,
                                    dtype: dtypes.typeclass,
                                    storage: dtypes.StorageType,
                                    num_elements: symbolic.SymbolicType,
                                    out_array: str,
                                    in_array: Optional[str] = None) -> dace.SDFG:
    """Wrapper SDFG performing a contiguous CPU transfer as an OpenMP-parallel chunk map.

    Both operands are modelled as flat 1-D arrays of ``num_elements`` (valid because the
    transfer is contiguous); a ``CPU_Multicore`` map (which codegen lowers to
    ``#pragma omp parallel for``) iterates ``ceil(num_elements / chunk)`` chunks, and each
    iteration runs one contiguous ``std::memcpy`` / ``std::memset`` over its chunk of up to
    :data:`PARALLEL_COPY_CHUNK_BYTES` bytes (the last chunk clamped via a ``Min``). The
    OpenMP parallelism comes entirely from the map schedule, not a hand-written pragma.

    :param label: base name for the wrapper SDFG / state / map.
    :param dtype: element type (provides ``ctype`` and ``bytes``).
    :param storage: storage type for both operand arrays.
    :param num_elements: total contiguous element count (constant or symbolic).
    :param out_array: destination array name (must equal the libnode output connector).
    :param in_array: source array name (libnode input connector) for a copy, or ``None`` for a memset.
    :returns: the wrapper SDFG.
    """
    chunk_elems = max(1, PARALLEL_COPY_CHUNK_BYTES // dtype.bytes)

    sdfg = dace.SDFG(f"{label}_sdfg")
    sdfg.add_array(out_array, [num_elements], dtype, storage, strides=[1])
    if in_array is not None:
        sdfg.add_array(in_array, [num_elements], dtype, storage, strides=[1])
    state = sdfg.add_state(f"{label}_state", is_start_block=True)

    c = symbolic.symbol(_CHUNK_INDEX_NAME, nonnegative=True)
    nchunks = symbolic.int_ceil(num_elements, chunk_elems)
    begin = c * chunk_elems
    end_incl = sympy.Min((c + 1) * chunk_elems, num_elements) - 1
    length = end_incl - begin + 1  # min((c+1)*chunk, total) - c*chunk

    # Inner-tasklet connectors, distinct from the wrapper-SDFG parameter arrays.
    inner_out, inner_in = "_out", "_in"
    outputs = {inner_out: Memlet(data=out_array, subset=Range([(begin, end_incl, 1)]))}
    if in_array is None:
        inputs = {}
        code = f"std::memset({inner_out}, 0, ({sym2cpp(length)}) * sizeof({dtype.ctype}));"
    else:
        inputs = {inner_in: Memlet(data=in_array, subset=Range([(begin, end_incl, 1)]))}
        code = f"std::memcpy({inner_out}, {inner_in}, ({sym2cpp(length)}) * sizeof({dtype.ctype}));"

    # Pass the chunk-count range as a symbolic ``(begin, end, step)`` tuple, NOT a string: a size
    # expression containing a function call (``int_ceil``, ``ipow`` for a 2**k FFT length) stringifies
    # with commas / ``::`` that ``SubsetProperty.from_string`` mis-splits into an invalid range. A tuple
    # is consumed verbatim by ``_make_iterators`` with no string parsing.
    tasklet, _me, _mx = state.add_mapped_tasklet(f"{label}_chunk", {_CHUNK_INDEX_NAME: (0, nchunks - 1, 1)},
                                                 inputs,
                                                 code,
                                                 outputs,
                                                 schedule=dtypes.ScheduleType.CPU_Multicore,
                                                 language=dace.Language.CPP,
                                                 external_edges=True)
    # Pointer connectors so codegen offsets them to the chunk start (T*) rather than
    # dereferencing a scalar -- matching the memcpy/memset signatures.
    tasklet.out_connectors[inner_out] = dtypes.pointer(dtype)
    if in_array is not None:
        tasklet.in_connectors[inner_in] = dtypes.pointer(dtype)
    return sdfg


def collapse_shape_and_strides(
        subset: dace.subsets.Range,
        strides: List[dace.symbolic.SymExpr]) -> Tuple[List[dace.symbolic.SymExpr], List[dace.symbolic.SymExpr]]:
    """Drop length-1 dimensions from a (subset, strides) pair.

    Surviving strides are scaled by the subset step (``stride * s``) so they describe the access
    pattern as a view into the parent array -- a no-op for unit-step subsets, and the effective
    per-element distance for strided ones.

    :param subset: The access range, one ``(begin, end, step)`` per dimension.
    :param strides: The parent array strides, aligned with ``subset``.
    :returns: ``(collapsed_shape, collapsed_strides)`` with singletons removed.
    """
    collapsed_shape = []
    collapsed_strides = []
    for (b, e, s), stride in zip(subset, strides):
        length = (e + 1 - b) // s
        if length != 1:
            collapsed_shape.append(length)
            collapsed_strides.append(stride * s)
    return collapsed_shape, collapsed_strides


def auto_dispatch(node: nodes.LibraryNode, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG,
                  select_fn: Callable[[nodes.LibraryNode, dace.SDFGState, dace.SDFG], str], library_cls: type):
    """Dispatch a library node's ``'Auto'`` implementation to the one picked by ``select_fn``.

    Sets ``node.implementation`` to the resolved name so introspection
    (debug output, downstream passes) reflects what was actually picked.

    :param node: the library node being expanded.
    :param parent_state: state containing ``node``.
    :param parent_sdfg: SDFG containing ``parent_state``.
    :param select_fn: callable returning a concrete implementation name (not ``'Auto'``).
    :param library_cls: the library node class with the ``implementations`` dict.
    :returns: whatever the resolved expansion returns.
    """
    impl_name = select_fn(node, parent_state, parent_sdfg)
    assert impl_name != 'Auto', f"{select_fn.__name__} must not return 'Auto'."
    node.implementation = impl_name
    return library_cls.implementations[impl_name].expansion(node, parent_state, parent_sdfg)
