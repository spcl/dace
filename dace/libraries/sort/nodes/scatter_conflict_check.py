# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ScatterConflictCheck`` libnode: count duplicate values in a 1-D integer array.

Runtime proof that a scatter index ``ip`` is a permutation -- no duplicates means no
write-write races, so ``a[ip[i]] = ...`` may run as a parallel Map. Opaque: the tile
vectorizer never looks through a library node, so the scatter guard never perturbs
tiling (the compute Map it guards vectorizes normally).

``_count_out`` is a length-1 ``int64`` **host** scalar in every backend, including CUDA,
so the downstream ``trap_sym`` interstate binding and the trap read stay on the host
regardless of where ``ip`` lives. ``count == 0`` iff the index is a permutation.

Implementations:

- ``pure`` -- ``std::sort`` a copy + adjacent-equal scan.
- ``CPU``  -- ``ska_sort`` (vendored radix) + adjacent-equal scan.
- ``CUDA`` -- copy ``ip`` device->host, then the same host sort + scan (a one-shot
  guard, so a host round-trip is cheaper than staging a device reduction; still
  returns a host scalar). Device-side sort/reduce is a future optimization.
"""
import dace
from dace import dtypes, library, nodes
from dace.codegen.common import sym2cpp
from dace.libraries.standard.environments.cuda import CUDA
from dace.transformation.transformation import ExpandTransformation
from . import _helpers
from .. import environments

INPUT_CONNECTOR_NAME = "_idx_in"
OUTPUT_CONNECTOR_NAME = "_count_out"


def _validate(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG):
    """Resolve + check the in/out edges; return ``(in_desc, in_name, out_name)``."""
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    out_edges = [e for e in state.out_edges(node) if e.src_conn == OUTPUT_CONNECTOR_NAME]
    if len(in_edges) != 1 or len(out_edges) != 1:
        raise ValueError(f"ScatterConflictCheck {node.label}: one '{INPUT_CONNECTOR_NAME}' in-edge + "
                         f"one '{OUTPUT_CONNECTOR_NAME}' out-edge required.")
    in_desc = sdfg.arrays[in_edges[0].data.data]
    out_desc = sdfg.arrays[out_edges[0].data.data]
    if not isinstance(in_desc, dace.data.Array) or not _helpers.is_integer_dtype(in_desc.dtype):
        raise ValueError(f"ScatterConflictCheck input must be an integer Array; got {in_desc}.")
    if out_desc.dtype != dtypes.int64:
        raise ValueError(f"ScatterConflictCheck output must be int64; got {out_desc.dtype}.")
    return in_desc, in_edges[0].data.data, out_edges[0].data.data


def _length(node: "ScatterConflictCheck", state: dace.SDFGState) -> str:
    """C++ expression for the input length from the in-edge memlet."""
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    return sym2cpp(in_edges[0].data.subset.num_elements())


def _tagcount_body(ct: str, n: str, src: str, omp: bool) -> str:
    """Tagged-write + verify duplicate detection: O(n), NO sort.

    Pass 1 ``owner[idx[i]] = i``; Pass 2 OR-reduce ``owner[idx[i]] != i``. Sorting is unnecessary --
    if two i's collide on a position, only one wins Pass 1, so the loser reads back a different i in
    Pass 2 and the OR trips. Pass 2 reads only positions Pass 1 wrote, so ``owner`` needs no init. The
    Pass-1 last-writer-wins race is benign (any winner is fine; the OR is monotonic). ``owner`` is sized
    max(idx)+1. ``omp`` toggles the OpenMP pragmas (ignored -> serial O(n), still far below any sort)."""
    par = (lambda clause: f"#pragma omp parallel for{clause}\n") if omp else (lambda clause: "")
    return (f"const long long _N = ({n});\n"
            f"{ct} _mx = 0;\n"
            f"{par(' reduction(max:_mx)')}for (long long _i = 0; _i < _N; ++_i) if ({src}[_i] > _mx) _mx = {src}[_i];\n"
            f"std::unique_ptr<long long[]> _own(new long long[(size_t)_mx + 1]);\n"
            f"{par('')}for (long long _i = 0; _i < _N; ++_i) _own[{src}[_i]] = _i;\n"
            f"long long _c = 0;\n"
            f"{par(' reduction(|:_c)')}for (long long _i = 0; _i < _N; ++_i) _c |= (_own[{src}[_i]] != _i);\n"
            f"{OUTPUT_CONNECTOR_NAME} = _c;\n")


@library.expansion
class ExpandPure(ExpandTransformation):
    """Tagged-write + verify (serial O(n))."""

    environments = []

    @staticmethod
    def expansion(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        in_desc, _in, _out = _validate(node, state, sdfg)
        n, ct = _length(node, state), in_desc.dtype.ctype
        code = "{\n#include <memory>\n" + _tagcount_body(ct, n, INPUT_CONNECTOR_NAME, omp=False) + "}"
        return nodes.Tasklet(node.name, {INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, code, language=dace.Language.CPP)


@library.expansion
class ExpandCPU(ExpandTransformation):
    """Tagged-write + verify, OpenMP-parallel (2 passes ~= 2x the scatter's own cost)."""

    environments = []

    @staticmethod
    def expansion(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        in_desc, _in, _out = _validate(node, state, sdfg)
        n, ct = _length(node, state), in_desc.dtype.ctype
        code = "{\n#include <memory>\n" + _tagcount_body(ct, n, INPUT_CONNECTOR_NAME, omp=True) + "}"
        return nodes.Tasklet(node.name, {INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, code, language=dace.Language.CPP)


@library.expansion
class ExpandCUDA(ExpandTransformation):
    """Copy the device index to host, then tagged-write + verify (host OpenMP).

    A one-shot guard, so a device->host round-trip of the index (O(n)) beats staging a device pass;
    ``_count_out`` lands on the host with no extra copy. ``cudaMemcpy`` is host-callable, so this
    compiles in the host (g++) translation unit like the CPU expansions.
    """

    environments = [CUDA]

    @staticmethod
    def expansion(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        in_desc, _in, _out = _validate(node, state, sdfg)
        n, ct = _length(node, state), in_desc.dtype.ctype
        code = ("{\n#include <memory>\n#include <vector>\n"
                f"std::vector<{ct}> _t(({n}));\n"
                f"cudaMemcpy(_t.data(), {INPUT_CONNECTOR_NAME}, ({n}) * sizeof({ct}), cudaMemcpyDeviceToHost);\n"
                + _tagcount_body(ct, n, "_t.data()", omp=True) + "}")
        return nodes.Tasklet(node.name, {INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, code, language=dace.Language.CPP)


@library.node
class ScatterConflictCheck(nodes.LibraryNode):
    """Count duplicate values in a 1-D integer index array (scatter no-conflict proof).

    - ``_idx_in``: input 1-D integer index array of length ``N``.
    - ``_count_out``: output length-1 ``int64`` **host** scalar = number of duplicate
      values (``0`` iff ``_idx_in`` is a permutation). Host in every backend so the
      guard's ``trap_sym`` binding + trap execute on the host.
    """

    INPUT_CONNECTOR_NAME = INPUT_CONNECTOR_NAME
    OUTPUT_CONNECTOR_NAME = OUTPUT_CONNECTOR_NAME

    implementations = {"CPU": ExpandCPU, "CUDA": ExpandCUDA, "pure": ExpandPure}
    default_implementation = "CPU"

    def __init__(self, name: str = "ScatterConflictCheck", *args, **kwargs):
        super().__init__(name, *args, inputs={INPUT_CONNECTOR_NAME}, outputs={OUTPUT_CONNECTOR_NAME}, **kwargs)

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState):
        _validate(self, state, sdfg)
