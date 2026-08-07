# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ScatterConflictCheck`` libnode: count duplicate values in a 1-D integer array.

Runtime proof that a scatter index ``ip`` is a permutation -- no duplicates means no
write-write races, so ``a[ip[i]] = ...`` may run as a parallel Map. Opaque: the tile
vectorizer never looks through a library node, so the scatter guard never perturbs
tiling (the compute Map it guards vectorizes normally).

``_count_out`` is a length-1 ``int64`` **host** scalar in every backend, including CUDA,
so the downstream ``trap_sym`` interstate binding and the trap read stay on the host
regardless of where ``ip`` lives. ``count == 0`` iff the index is a permutation.

The tag array the check indexes by index-*value* is an optional ``_owner_out`` connector: an
``int64`` host Array the caller sizes by the scattered array's domain (see
:func:`~dace.transformation.passes.scatter_conflict_guard.insert_scatter_guard`) and owns as a
regular SDFG transient, so DaCe allocates it -- persistently, outside the timed program body.
Left unconnected, the node falls back to sizing a heap buffer from a runtime ``max(idx)``
sweep, which costs one extra full pass over ``idx`` plus an allocation per call.

Implementations (all host code; the tag array must live in host memory on every backend):

- ``pure`` -- tagged-write + verify, serial.
- ``CPU``  -- tagged-write + verify, OpenMP-parallel.
- ``CUDA`` -- copy ``ip`` device->host, then the same host tagged-write + verify (a one-shot
  guard, so a host round-trip is cheaper than staging a device reduction; still returns a
  host scalar). A device-side check is a future optimization.
"""
from typing import Optional, Tuple

import dace
from dace import dtypes, library, nodes
from dace.codegen.common import sym2cpp
from dace.libraries.standard.environments.cuda import CUDA
from dace.transformation.transformation import ExpandTransformation
from . import _helpers

INPUT_CONNECTOR_NAME = "_idx_in"
OUTPUT_CONNECTOR_NAME = "_count_out"
SCRATCH_CONNECTOR_NAME = "_owner_out"

#: Storage the tag array may use: every expansion runs the check in host code.
HOST_STORAGE = (dtypes.StorageType.Default, dtypes.StorageType.Register, dtypes.StorageType.CPU_Heap,
                dtypes.StorageType.CPU_ThreadLocal, dtypes.StorageType.CPU_Pinned)


def _validate(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG):
    """Resolve + check the edges; return ``(in_desc, in_name, out_name, owner_desc)``.

    ``owner_desc`` is the tag-array descriptor when the optional ``_owner_out`` connector is
    wired, else ``None``.
    """
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

    owner_edges = [e for e in state.out_edges(node) if e.src_conn == SCRATCH_CONNECTOR_NAME]
    if len(owner_edges) > 1:
        raise ValueError(f"ScatterConflictCheck {node.label}: at most one '{SCRATCH_CONNECTOR_NAME}' out-edge.")
    owner_desc = None
    if owner_edges:
        owner_desc = sdfg.arrays[owner_edges[0].data.data]
        if not isinstance(owner_desc, dace.data.Array) or owner_desc.dtype != dtypes.int64:
            raise ValueError(f"ScatterConflictCheck '{SCRATCH_CONNECTOR_NAME}' must be an int64 Array; "
                             f"got {owner_desc}.")
        # The check is host code in every expansion, so a device-resident tag array would be
        # dereferenced from the host. Refuse loudly rather than corrupt memory.
        if owner_desc.storage not in HOST_STORAGE:
            raise ValueError(f"ScatterConflictCheck '{SCRATCH_CONNECTOR_NAME}' must live in host memory; "
                             f"got storage {owner_desc.storage}.")
    return in_desc, in_edges[0].data.data, out_edges[0].data.data, owner_desc


def _length(node: "ScatterConflictCheck", state: dace.SDFGState) -> str:
    """C++ expression for the input length from the in-edge memlet."""
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    return sym2cpp(in_edges[0].data.subset.num_elements())


def _outputs(owner_desc: Optional[dace.data.Array]) -> set:
    """Output connectors of the expanded tasklet (the tag array only when it is wired)."""
    return {OUTPUT_CONNECTOR_NAME} if owner_desc is None else {OUTPUT_CONNECTOR_NAME, SCRATCH_CONNECTOR_NAME}


def _owner(node: "ScatterConflictCheck", state: dace.SDFGState,
           owner_desc: Optional[dace.data.Array]) -> Optional[Tuple[str, str]]:
    """``(connector, capacity)`` for the wired tag array, or ``None``. Capacity comes from the
    edge's own subset, so it tracks whatever the caller actually sized the descriptor by."""
    if owner_desc is None:
        return None
    edge = next(e for e in state.out_edges(node) if e.src_conn == SCRATCH_CONNECTOR_NAME)
    return SCRATCH_CONNECTOR_NAME, sym2cpp(edge.data.subset.num_elements())


def _tagcount_body(ct: str, n: str, src: str, omp: bool, owner: Optional[Tuple[str, str]]) -> Tuple[str, str]:
    """Tagged-write + verify duplicate detection: O(n), NO sort. Returns ``(body, global_code)``.

    Pass 1 ``owner[idx[i]] = i``; Pass 2 OR-reduce ``owner[idx[i]] != i``. Sorting is unnecessary --
    if two i's collide on a position, only one wins Pass 1, so the loser reads back a different i in
    Pass 2 and the OR trips. Pass 2 reads only positions Pass 1 wrote, so ``owner`` needs no init (nor
    any clearing between calls, which is what lets a caller hand in one persistent buffer). The Pass-1
    last-writer-wins race is benign (any winner is fine; the OR is monotonic).

    Both passes skip values outside ``[0, capacity)``. Such a value can never be a slot the guarded
    scatter writes -- the scatter's own array is what bounds ``capacity`` -- so it can neither collide
    with nor mask a real conflict, and skipping it keeps the tag write in bounds when ``idx`` carries
    entries the scatter does not use (a strided scatter reads only part of ``idx``).

    ``owner`` is ``(connector, capacity)`` for the caller-provided tag array, which must span the
    scattered array's domain. Without one, a third pass computes ``max(idx)`` and sizes a heap buffer
    from it. ``omp`` toggles the OpenMP pragmas (ignored -> serial O(n)).

    Two passes is the floor: verify can only read a slot once every writer has had its turn, so
    folding the check into Pass 1 would compare against a half-written tag array. A single-pass form
    needs an atomic read-modify-write per element plus a pre-cleared (or epoch-stamped) tag array --
    an extra full sweep over the domain, which is at least as wide as ``idx``."""
    par = (lambda clause: f"#pragma omp parallel for{clause}\n") if omp else (lambda clause: "")
    #: Bitwise-or is simd-safe (see dace/runtime/include/dace/reduction.h), so the OR-reduce
    #: pass below carries simd -- unlike ``par``, no bare ``if()`` clause is ever appended here.
    par_simd_reduction = (lambda clause: f"#pragma omp parallel for simd{clause}\n") if omp else (lambda clause: "")
    if owner is None:
        tag, cap, global_code = "_own", "_mx + 1", "#include <memory>"
        alloc = (f"{ct} _mx = 0;\n"
                 f"{par(' reduction(max:_mx)')}"
                 f"for (long long _i = 0; _i < _N; ++_i) if ({src}[_i] > _mx) _mx = {src}[_i];\n"
                 f"std::unique_ptr<long long[]> {tag}(new long long[(size_t)_mx + 1]);\n")
    else:
        (tag, cap), global_code, alloc = owner, "", ""
    body = (f"const long long _N = ({n});\n" + alloc + f"const long long _M = ({cap});\n"
            f"{par('')}for (long long _i = 0; _i < _N; ++_i) "
            f"{{ const long long _v = {src}[_i]; if (_v >= 0 && _v < _M) {tag}[_v] = _i; }}\n"
            f"long long _c = 0;\n"
            f"{par_simd_reduction(' reduction(|:_c)')}for (long long _i = 0; _i < _N; ++_i) "
            f"{{ const long long _v = {src}[_i]; if (_v >= 0 && _v < _M) _c |= ({tag}[_v] != _i); }}\n"
            f"{OUTPUT_CONNECTOR_NAME} = _c;\n")
    return body, global_code


@library.expansion
class ExpandPure(ExpandTransformation):
    """Tagged-write + verify (serial O(n))."""

    environments = []

    @staticmethod
    def expansion(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        in_desc, _in, _out, owner_desc = _validate(node, state, sdfg)
        n, ct = _length(node, state), in_desc.dtype.ctype
        owner = _owner(node, state, owner_desc)
        body, global_code = _tagcount_body(ct, n, INPUT_CONNECTOR_NAME, omp=False, owner=owner)
        return nodes.Tasklet(node.name, {INPUT_CONNECTOR_NAME},
                             _outputs(owner_desc),
                             "{\n" + body + "}",
                             language=dace.Language.CPP,
                             code_global=global_code)


@library.expansion
class ExpandCPU(ExpandTransformation):
    """Tagged-write + verify, OpenMP-parallel (2 passes ~= 2x the scatter's own cost)."""

    environments = []

    @staticmethod
    def expansion(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        in_desc, _in, _out, owner_desc = _validate(node, state, sdfg)
        n, ct = _length(node, state), in_desc.dtype.ctype
        owner = _owner(node, state, owner_desc)
        body, global_code = _tagcount_body(ct, n, INPUT_CONNECTOR_NAME, omp=True, owner=owner)
        return nodes.Tasklet(node.name, {INPUT_CONNECTOR_NAME},
                             _outputs(owner_desc),
                             "{\n" + body + "}",
                             language=dace.Language.CPP,
                             code_global=global_code)


@library.expansion
class ExpandCUDA(ExpandTransformation):
    """Copy the device index to host, then tagged-write + verify (host OpenMP).

    A one-shot guard, so a device->host round-trip of the index (O(n)) beats staging a device pass;
    ``_count_out`` lands on the host with no extra copy. ``cudaMemcpy`` is host-callable, so this
    compiles in the host (g++) translation unit like the CPU expansions. The tag array, when wired,
    must therefore be host memory too -- :func:`_validate` refuses anything else.
    """

    environments = [CUDA]

    @staticmethod
    def expansion(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        in_desc, _in, _out, owner_desc = _validate(node, state, sdfg)
        n, ct = _length(node, state), in_desc.dtype.ctype
        owner = _owner(node, state, owner_desc)
        body, global_code = _tagcount_body(ct, n, "_t.data()", omp=True, owner=owner)
        code = ("{\n"
                f"std::vector<{ct}> _t(({n}));\n"
                f"cudaMemcpy(_t.data(), {INPUT_CONNECTOR_NAME}, ({n}) * sizeof({ct}), cudaMemcpyDeviceToHost);\n" +
                body + "}")
        return nodes.Tasklet(node.name, {INPUT_CONNECTOR_NAME},
                             _outputs(owner_desc),
                             code,
                             language=dace.Language.CPP,
                             code_global=(global_code + "\n#include <vector>").strip())


@library.node
class ScatterConflictCheck(nodes.LibraryNode):
    """Count duplicate values in a 1-D integer index array (scatter no-conflict proof).

    - ``_idx_in``: input 1-D integer index array of length ``N``.
    - ``_count_out``: output length-1 ``int64`` **host** scalar = number of duplicate
      values (``0`` iff ``_idx_in`` is a permutation). Host in every backend so the
      guard's ``trap_sym`` binding + trap execute on the host.
    - ``_owner_out`` (optional): scratch tag array, an ``int64`` **host** Array spanning
      every value ``_idx_in`` can hold. Wiring it lets DaCe own the allocation (so it can
      be persistent, outside the program body) and drops the ``max(idx)`` sizing pass.
      Contents are never read before being written, so it needs no initialization and may
      carry stale values from a previous call.
    """

    INPUT_CONNECTOR_NAME = INPUT_CONNECTOR_NAME
    OUTPUT_CONNECTOR_NAME = OUTPUT_CONNECTOR_NAME
    SCRATCH_CONNECTOR_NAME = SCRATCH_CONNECTOR_NAME

    implementations = {"CPU": ExpandCPU, "CUDA": ExpandCUDA, "pure": ExpandPure}
    default_implementation = "CPU"

    def __init__(self, name: str = "ScatterConflictCheck", *args, **kwargs):
        super().__init__(name, *args, inputs={INPUT_CONNECTOR_NAME}, outputs={OUTPUT_CONNECTOR_NAME}, **kwargs)

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState):
        _validate(self, state, sdfg)
