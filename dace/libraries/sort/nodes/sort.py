# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Sort`` library node: ``numpy.sort`` / ``numpy.argsort`` along one axis of an N-D array.

Any arithmetic key dtype, ascending and stable (equal keys keep their order, NumPy's ``kind='stable'``), NaN last
as NumPy orders it. ``argsort`` writes the ``int64`` positions instead of the keys. Every row along ``axis`` is
independent: the expansion is a map over the other axes whose body sorts one row with ``std::stable_sort``, so it
is parallel at top level and single-core (``'sequential'``) inside a parallel region, as the other library nodes
lower by scope.
"""

import dace
from dace import dtypes, library, nodes, properties
from dace.codegen.common import sym2cpp
from dace.libraries.standard.helper import single_core_expansion
from dace.transformation.transformation import ExpandTransformation

INPUT_CONNECTOR_NAME = "_keys_in"
OUTPUT_CONNECTOR_NAME = "_out"
#: The row tasklet's connectors (a connector may not share a name with the nested SDFG's arrays).
ROW_IN, ROW_OUT = "__keys", "__row_out"
#: What a row's ``std::stable_sort`` needs.
ROW_HEADERS = "#include <algorithm>\n#include <cmath>\n#include <numeric>\n#include <vector>\n"


def operand_descriptors(
    node: "Sort", state: dace.SDFGState, sdfg: dace.SDFG
) -> tuple[dace.data.Array, dace.data.Array]:
    """The input and output descriptors, validated: same shape, the output typed as the keys (or ``int64`` for
    ``argsort``), and ``axis`` within the rank."""
    in_edge = next(e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME)
    out_edge = next(e for e in state.out_edges(node) if e.src_conn == OUTPUT_CONNECTOR_NAME)
    keys = sdfg.arrays[in_edge.data.data]
    out = sdfg.arrays[out_edge.data.data]
    if len(keys.shape) != len(out.shape) or not 0 <= node.axis < len(keys.shape):
        raise ValueError(
            f"Sort {node.label}: axis {node.axis} of a rank-{len(keys.shape)} input into rank {len(out.shape)}"
        )
    want = dtypes.int64 if node.argsort else keys.dtype
    if out.dtype != want:
        raise ValueError(f"Sort {node.label}: output dtype {out.dtype}, expected {want}")
    return keys, out


def row_code(dtype: dtypes.typeclass, n: str, in_stride: str, out_stride: str, argsort: bool) -> str:
    """C++ that stably sorts the row ``__keys[0:n:in_stride]`` into ``__row_out[0:n:out_stride]``."""
    ctype = dtype.ctype
    if dtype in dtypes.FLOAT_TYPES:
        less = "return a < b || (std::isnan(b) && !std::isnan(a));"
    else:
        less = "return a < b;"
    head = f"const long long __n = {n};\nauto __less = [](const {ctype}& a, const {ctype}& b) {{ {less} }};\n"
    if argsort:
        return head + (
            "std::vector<long long> __order(__n);\n"
            "std::iota(__order.begin(), __order.end(), 0LL);\n"
            "std::stable_sort(__order.begin(), __order.end(), [&](long long a, long long b) {\n"
            f"    return __less({ROW_IN}[a * ({in_stride})], {ROW_IN}[b * ({in_stride})]);\n"
            "});\n"
            f"for (long long __k = 0; __k < __n; ++__k) {ROW_OUT}[__k * ({out_stride})] = __order[__k];"
        )
    return head + (
        f"std::vector<{ctype}> __row(__n);\n"
        f"for (long long __k = 0; __k < __n; ++__k) __row[__k] = {ROW_IN}[__k * ({in_stride})];\n"
        "std::stable_sort(__row.begin(), __row.end(), __less);\n"
        f"for (long long __k = 0; __k < __n; ++__k) {ROW_OUT}[__k * ({out_stride})] = __row[__k];"
    )


@library.expansion
class ExpandCPU(ExpandTransformation):
    """A map over every axis but ``axis``; each iteration ``std::stable_sort``\\ s one row."""

    environments = []

    @staticmethod
    def expansion(node: "Sort", state: dace.SDFGState, sdfg: dace.SDFG) -> dace.SDFG:
        keys, out = operand_descriptors(node, state, sdfg)
        nsdfg = dace.SDFG(f"{node.label}_sdfg")
        for name, desc in ((INPUT_CONNECTOR_NAME, keys), (OUTPUT_CONNECTOR_NAME, out)):
            inner = desc.clone()
            inner.transient = False
            nsdfg.add_datadesc(name, inner)
        body = nsdfg.add_state(f"{node.label}_rows")
        rows = {f"__s{d}": f"0:{sym2cpp(extent)}" for d, extent in enumerate(keys.shape) if d != node.axis}
        n = keys.shape[node.axis]
        row = ", ".join(f"0:{n}" if d == node.axis else f"__s{d}" for d in range(len(keys.shape)))
        code = row_code(
            keys.dtype, sym2cpp(n), sym2cpp(keys.strides[node.axis]), sym2cpp(out.strides[node.axis]), node.argsort
        )
        if rows:
            tasklet, _, _ = body.add_mapped_tasklet(
                f"{node.label}_row",
                rows,
                {ROW_IN: dace.Memlet(f"{INPUT_CONNECTOR_NAME}[{row}]")},
                code,
                {ROW_OUT: dace.Memlet(f"{OUTPUT_CONNECTOR_NAME}[{row}]")},
                language=dtypes.Language.CPP,
                external_edges=True,
            )
        else:
            # A 1-D input is one row: no map around it
            tasklet = body.add_tasklet(node.label, {ROW_IN}, {ROW_OUT}, code, language=dtypes.Language.CPP)
            body.add_edge(
                body.add_read(INPUT_CONNECTOR_NAME),
                None,
                tasklet,
                ROW_IN,
                dace.Memlet(f"{INPUT_CONNECTOR_NAME}[{row}]"),
            )
            body.add_edge(
                tasklet,
                ROW_OUT,
                body.add_write(OUTPUT_CONNECTOR_NAME),
                None,
                dace.Memlet(f"{OUTPUT_CONNECTOR_NAME}[{row}]"),
            )
        tasklet.code_global = dace.properties.CodeBlock(ROW_HEADERS, dtypes.Language.CPP)
        return nsdfg


@library.node
class Sort(nodes.LibraryNode):
    """Sort (or argsort) ``_keys_in`` along ``axis`` into ``_out``, stably, NaN last.

    - ``_keys_in``: the keys, any arithmetic dtype, any rank.
    - ``_out``: same shape; the keys' dtype, or ``int64`` positions along ``axis`` when ``argsort``.

    Implementations: ``'CPU'`` (rows in parallel, ``std::stable_sort`` per row) and ``'sequential'`` (the same, one
    core, for a node a parallel region encloses).
    """

    implementations = {"CPU": ExpandCPU, "sequential": single_core_expansion(ExpandCPU)}
    default_implementation = "CPU"

    axis = properties.Property(dtype=int, default=0, desc="The axis sorted along (nonnegative)")
    argsort = properties.Property(dtype=bool, default=False, desc="Write int64 positions instead of the keys")

    def __init__(self, name: str = "Sort", axis: int = 0, argsort: bool = False, *args, **kwargs):
        super().__init__(name, *args, inputs={INPUT_CONNECTOR_NAME}, outputs={OUTPUT_CONNECTOR_NAME}, **kwargs)
        self.axis = axis
        self.argsort = argsort

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState):
        operand_descriptors(self, state, sdfg)
