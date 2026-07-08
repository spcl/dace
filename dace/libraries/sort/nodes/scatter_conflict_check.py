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


def _scan_body(ct: str) -> str:
    """Adjacent-equal count over the sorted ``std::vector<ct> _t`` -> host ``_count_out``."""
    return ("long long _d = 0;\n"
            "for (size_t _i = 0; _i + 1 < _t.size(); ++_i) if (_t[_i] == _t[_i + 1]) ++_d;\n"
            f"{OUTPUT_CONNECTOR_NAME} = _d;\n")


@library.expansion
class ExpandPure(ExpandTransformation):
    """``std::sort`` a copy + adjacent-equal scan."""

    environments = []

    @staticmethod
    def expansion(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        in_desc, _in, _out = _validate(node, state, sdfg)
        n, ct = _length(node, state), in_desc.dtype.ctype
        code = ("{\n#include <algorithm>\n#include <vector>\n"
                f"std::vector<{ct}> _t({INPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME} + ({n}));\n"
                "std::sort(_t.begin(), _t.end());\n" + _scan_body(ct) + "}")
        return nodes.Tasklet(node.name, {INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, code, language=dace.Language.CPP)


@library.expansion
class ExpandCPU(ExpandTransformation):
    """``ska_sort`` a copy + adjacent-equal scan."""

    environments = [environments.SkaSort]

    @staticmethod
    def expansion(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        in_desc, _in, _out = _validate(node, state, sdfg)
        n, ct = _length(node, state), in_desc.dtype.ctype
        code = ("{\n#include <vector>\n"
                f"std::vector<{ct}> _t({INPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME} + ({n}));\n"
                "::ska_sort(_t.begin(), _t.end());\n" + _scan_body(ct) + "}")
        return nodes.Tasklet(node.name, {INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, code, language=dace.Language.CPP)


@library.expansion
class ExpandCUDA(ExpandTransformation):
    """Copy the device index to host, then host sort + adjacent-equal scan.

    A permutation guard runs once per scatter, so a device->host round-trip is cheaper
    than staging a device sort + reduction, and ``_count_out`` lands on the host with no
    extra copy. ``cudaMemcpy`` is a host-callable runtime call, so this tasklet compiles
    in the host (g++) translation unit like the CPU expansions.
    """

    environments = [CUDA]

    @staticmethod
    def expansion(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        in_desc, _in, _out = _validate(node, state, sdfg)
        n, ct = _length(node, state), in_desc.dtype.ctype
        code = ("{\n#include <algorithm>\n#include <vector>\n"
                f"std::vector<{ct}> _t(({n}));\n"
                f"cudaMemcpy(_t.data(), {INPUT_CONNECTOR_NAME}, ({n}) * sizeof({ct}), cudaMemcpyDeviceToHost);\n"
                "std::sort(_t.begin(), _t.end());\n" + _scan_body(ct) + "}")
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
