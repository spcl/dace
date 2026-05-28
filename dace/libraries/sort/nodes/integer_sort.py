# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``IntegerSort`` library node: sort a 1-D integer array ascending.

The node is deliberately specialised: integer keys, ascending order, no values, no
custom comparator, no stability requirement. This matches the radix-sort sweet spot
on both CPU (`ska_sort`) and GPU (`cub::DeviceRadixSort`), and keeps the libnode
surface minimal so it can serve as the sort primitive behind the scatter-conflict
guard (and any future DaCe pass that needs to sort integer indices).

Implementations:

- ``CPU`` -- vendored Boost-licensed ``ska_sort`` (single header at
  :file:`dace/runtime/include/dace/ska_sort.hpp`). The clear winner among MSD radix
  sorts in the Probably Dance benchmarks, and on the same order as `vqsort` on
  non-AVX-512 CPUs. Zero added external dependency: ``#include <dace/ska_sort.hpp>``.
- ``CUDA`` -- ``cub::DeviceRadixSort::SortKeys``. Uses scratch allocated via
  ``cudaMalloc`` once per call; for the scatter-guard use case the sort runs at
  most once per scatter so the allocation cost is acceptable.
- ``pure`` -- a ``std::sort`` C++ tasklet. Used as a portable fallback when neither
  CPU nor CUDA expansion is selectable (e.g. FPGA backends), and as the default
  ``Auto`` choice if no GPU storage is detected.

The output buffer must have the same shape and dtype as the input buffer; both must
be 1-D contiguous integer arrays.
"""
from typing import Tuple

import dace
from dace import library, nodes
from dace.codegen.common import sym2cpp
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.transformation import ExpandTransformation
from . import _helpers  # local helper functions kept out of this file for readability
from .. import environments


# Connector names exposed for library-node builders.
INPUT_CONNECTOR_NAME = "_keys_in"
OUTPUT_CONNECTOR_NAME = "_keys_out"


def _validate_inputs_and_outputs(
    node: "IntegerSort", state: dace.SDFGState, sdfg: dace.SDFG
) -> Tuple[dace.data.Array, dace.data.Array, str, str]:
    """Resolve and validate the in/out edges; return ``(in_desc, out_desc, in_name, out_name)``.

    :param node: The IntegerSort node being expanded.
    :param state: The state containing the node.
    :param sdfg: The SDFG containing the state.
    :returns: ``(in_desc, out_desc, in_name, out_name)``.
    :raises ValueError: On any wiring/shape/dtype mismatch.
    """
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    out_edges = [e for e in state.out_edges(node) if e.src_conn == OUTPUT_CONNECTOR_NAME]
    if len(in_edges) != 1 or len(out_edges) != 1:
        raise ValueError(f"IntegerSort node {node.label} expects exactly one ``{INPUT_CONNECTOR_NAME}`` "
                         f"in-edge and one ``{OUTPUT_CONNECTOR_NAME}`` out-edge.")
    in_name = in_edges[0].data.data
    out_name = out_edges[0].data.data
    in_desc = sdfg.arrays[in_name]
    out_desc = sdfg.arrays[out_name]
    if not isinstance(in_desc, dace.data.Array) or not isinstance(out_desc, dace.data.Array):
        raise ValueError(f"IntegerSort requires Array inputs/outputs; got {type(in_desc).__name__} -> "
                         f"{type(out_desc).__name__}.")
    if in_desc.dtype != out_desc.dtype:
        raise ValueError(f"IntegerSort input/output dtype mismatch: {in_desc.dtype} vs {out_desc.dtype}.")
    if not _helpers.is_integer_dtype(in_desc.dtype):
        raise ValueError(f"IntegerSort requires an integer dtype; got {in_desc.dtype}.")
    return in_desc, out_desc, in_name, out_name


def _resolve_length(node: "IntegerSort", state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """Return a C++ expression for the input length ``N`` from the in-edge memlet."""
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    return sym2cpp(in_edges[0].data.subset.num_elements())


def _is_length_one(node: "IntegerSort", state: dace.SDFGState) -> bool:
    """``True`` if the input subset is statically a single element. Sorts of length 1
    degenerate to a copy: the sole element is trivially "sorted"."""
    from dace import symbolic as _sym
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    n = _sym.simplify(in_edges[0].data.subset.num_elements())
    return getattr(n, 'is_Integer', False) and int(n) == 1


def _degenerate_single_element_tasklet(node: "IntegerSort") -> nodes.Tasklet:
    """Single-element ``IntegerSort`` is a no-op copy. Python-language scalar tasklet so
    the codegen handles the connector typing naturally (no array indexing, no iterator
    templates that would mis-type a single-element subset)."""
    return nodes.Tasklet(
        node.name,
        inputs={INPUT_CONNECTOR_NAME},
        outputs={OUTPUT_CONNECTOR_NAME},
        code=f"{OUTPUT_CONNECTOR_NAME} = {INPUT_CONNECTOR_NAME}",
        language=dace.Language.Python,
    )


@library.expansion
class ExpandPure(ExpandTransformation):
    """Portable fallback: a ``std::sort`` C++ tasklet."""

    environments = []

    @staticmethod
    def expansion(node: "IntegerSort", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        _in_desc, _out_desc, _in_name, _out_name = _validate_inputs_and_outputs(node, state, sdfg)
        if _is_length_one(node, state):
            return _degenerate_single_element_tasklet(node)
        n_expr = _resolve_length(node, state, sdfg)
        code = (f"std::copy({INPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME} + ({n_expr}), {OUTPUT_CONNECTOR_NAME});\n"
                f"std::sort({OUTPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME} + ({n_expr}));")
        return nodes.Tasklet(
            node.name,
            inputs={INPUT_CONNECTOR_NAME},
            outputs={OUTPUT_CONNECTOR_NAME},
            code=f"{{\n#include <algorithm>\n{code}\n}}",
            language=dace.Language.CPP,
        )


@library.expansion
class ExpandCPU(ExpandTransformation):
    """``ska_sort`` MSD radix sort over the input array."""

    environments = [environments.SkaSort]

    @staticmethod
    def expansion(node: "IntegerSort", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        _in_desc, _out_desc, _in_name, _out_name = _validate_inputs_and_outputs(node, state, sdfg)
        if _is_length_one(node, state):
            return _degenerate_single_element_tasklet(node)
        n_expr = _resolve_length(node, state, sdfg)
        # ``ska_sort`` lives in the global namespace (see the vendored header). It sorts the
        # range in place, so we first copy ``_keys_in`` into ``_keys_out`` and then sort the
        # output range. The copy keeps the input array untouched (which matters because the
        # libnode contract is "produce a sorted copy"; some callers may read ``_keys_in``
        # elsewhere in the SDFG).
        code = (f"std::copy({INPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME} + ({n_expr}), {OUTPUT_CONNECTOR_NAME});\n"
                f"::ska_sort({OUTPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME} + ({n_expr}));")
        return nodes.Tasklet(
            node.name,
            inputs={INPUT_CONNECTOR_NAME},
            outputs={OUTPUT_CONNECTOR_NAME},
            code=code,
            language=dace.Language.CPP,
        )


@library.expansion
class ExpandCUDA(ExpandTransformation):
    """``cub::DeviceRadixSort::SortKeys`` over the input array (device-global memory).

    Temporary storage is obtained from the per-libnode-class, per-stream CUB scratch pool
    tagged ``SortTag`` (see :file:`dace/runtime/include/dace/cub_scratch.cuh` and the
    :class:`SortScratch` environment): the default-stream entry is pre-allocated to 128 MB
    at SDFG init; additional streams allocate lazily on first use. Each per-stream entry is
    reused across every ``IntegerSort`` call on that stream, grown in place if a request
    exceeds the current allocation, and released at SDFG exit. The libnode threads
    ``__dace_current_stream`` to both the scratch lookup and the underlying
    ``cub::DeviceRadixSort`` call, so concurrent launches on different streams cannot race
    on the pool.
    """

    environments = [environments.SortScratch]

    @staticmethod
    def expansion(node: "IntegerSort", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        _in_desc, _out_desc, _in_name, _out_name = _validate_inputs_and_outputs(node, state, sdfg)
        if _is_length_one(node, state):
            return _degenerate_single_element_tasklet(node)
        n_expr = _resolve_length(node, state, sdfg)
        # ``cub::DeviceRadixSort::SortKeys`` accepts the stream as its last (default-0)
        # parameter; positional preceding args (``begin_bit``, ``end_bit``) take the
        # natural defaults for full-range key sort.
        in_dtype = _in_desc.dtype.ctype
        bit_args = f"0, sizeof({in_dtype}) * 8"
        code = (f"size_t _ks_needed = 0;\n"
                f"::cub::DeviceRadixSort::SortKeys(nullptr, _ks_needed, "
                f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, ({n_expr}), "
                f"{bit_args}, __dace_current_stream);\n"
                f"void* _ks_scratch = ::dace::cub::get_scratch<::dace::cub::SortTag>("
                f"_ks_needed, __dace_current_stream);\n"
                f"::cub::DeviceRadixSort::SortKeys(_ks_scratch, _ks_needed, "
                f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, ({n_expr}), "
                f"{bit_args}, __dace_current_stream);")
        return nodes.Tasklet(
            node.name,
            inputs={INPUT_CONNECTOR_NAME},
            outputs={OUTPUT_CONNECTOR_NAME},
            code=code,
            language=dace.Language.CPP,
        )


@library.node
class IntegerSort(nodes.LibraryNode):
    """Sort a 1-D integer array ascending.

    Inputs / outputs:

    - ``_keys_in``: input 1-D contiguous integer array of length ``N``.
    - ``_keys_out``: output 1-D contiguous integer array of length ``N`` and same dtype.

    Implementations:

    - ``'CPU'`` -- ska_sort (vendored, fast MSD radix). Default on host.
    - ``'CUDA'`` -- ``cub::DeviceRadixSort::SortKeys`` (memory-bandwidth bound on GPU).
    - ``'pure'`` -- ``std::sort`` portable fallback.

    The libnode is contractually pure: it neither aliases the input/output buffers
    nor reads/writes any other state. A caller may pass the same buffer for input
    and output -- in that case ``std::copy`` collapses to a self-copy then sort.
    """

    INPUT_CONNECTOR_NAME = INPUT_CONNECTOR_NAME
    OUTPUT_CONNECTOR_NAME = OUTPUT_CONNECTOR_NAME

    implementations = {"CPU": ExpandCPU, "CUDA": ExpandCUDA, "pure": ExpandPure}
    default_implementation = 'CPU'

    def __init__(self, name: str = 'IntegerSort', *args, **kwargs):
        super().__init__(name, *args, inputs={INPUT_CONNECTOR_NAME}, outputs={OUTPUT_CONNECTOR_NAME}, **kwargs)

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState):
        """Verify wiring + dtypes; called from each expansion's helper."""
        _validate_inputs_and_outputs(self, state, sdfg)
