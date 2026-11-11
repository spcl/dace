# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""In-kernel CSR products by ONE thread block (CSR-scalar): output elements strided over the lanes, each a row's dot
product with the dense operand, so no lane waits on another until the closing barrier."""

from typing import Any

from dace import SDFG, Memlet, SDFGState, dtypes, nodes
from dace.codegen.common import sym2cpp
from dace.libraries.blas.blas_helpers import connector_sdfg
from dace.libraries.standard.block_reduce import BLOCK_COLLECTIVE_THREADS, add_block_lane_map


def scalar_text(value: Any, dtype: dtypes.typeclass) -> str:
    """``value`` (``alpha`` or ``beta``) as a C++ value of ``dtype``."""
    if isinstance(value, complex):
        return f"{dtype.ctype}({value.real}, {value.imag})"
    return f"{dtype.ctype}({sym2cpp(value)})"


def csr_block_code(node: nodes.LibraryNode, sdfg: SDFG, transposed: bool | None) -> str:
    """``_c = alpha * A @ _b + beta * C`` over the connector arrays of ``sdfg``; ``transposed`` is ``None`` for a
    vector ``_b`` (CSRMV), else whether ``_b`` is read transposed (CSRMM)."""
    rows, cols, vals, b, c = (sdfg.arrays[n] for n in ("_a_rows", "_a_cols", "_a_vals", "_b", "_c"))
    width = "1" if transposed is None else sym2cpp(c.shape[1])

    def at(desc, *indices: str) -> str:
        return " + ".join(f"({index}) * ({sym2cpp(stride)})" for index, stride in zip(indices, desc.strides))

    column = "(long long)__a_cols[__j * " + sym2cpp(cols.strides[0]) + "]"
    if transposed is None:
        dense = at(b, column)
    else:
        dense = at(b, "__k", column) if transposed else at(b, column, "__k")
    out = at(c, "__r") if transposed is None else at(c, "__r", "__k")
    ctype = c.dtype.ctype
    beta = ""
    if "_cin" in node.in_connectors:
        cin = sdfg.arrays["_cin"]
        initial = at(cin, "__r") if transposed is None else at(cin, "__r", "__k")
        beta = f" + {scalar_text(node.beta, c.dtype)} * __cin[{initial}]"
    elif node.beta != 0:
        beta = f" + {scalar_text(node.beta, c.dtype)} * __c[{out}]"
    row = sym2cpp(rows.strides[0])
    return f"""
const long long __m = {sym2cpp(rows.shape[0] - 1)}, __n = {width};
for (long long __t = threadIdx.x; __t < __m * __n; __t += {BLOCK_COLLECTIVE_THREADS}) {{
    const long long __r = __t / __n, __k = __t % __n;
    (void)__k;
    {ctype} __acc = {ctype}(0);
    for (long long __j = __a_rows[__r * {row}]; __j < __a_rows[(__r + 1) * {row}]; ++__j)
        __acc += __a_vals[__j * {sym2cpp(vals.strides[0])}] * __b[{dense}];
    __c[{out}] = {scalar_text(node.alpha, c.dtype)} * __acc{beta};
}}
__syncthreads();
"""


def csr_block_sdfg(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG, transposed: bool | None) -> SDFG:
    """A ``GPU_ThreadBlock`` lane map around the CSR product of :func:`csr_block_code`."""
    nsdfg = connector_sdfg(node, state, sdfg)
    code = csr_block_code(node, nsdfg, transposed)
    nstate = nsdfg.add_state(f"{node.label}_block_state")
    tasklet = nstate.add_tasklet(
        f"{node.label}_block_csr",
        {f"_{name}": dtypes.pointer(nsdfg.arrays[name].dtype) for name in node.in_connectors},
        {"__c": dtypes.pointer(nsdfg.arrays["_c"].dtype)},
        code,
        language=dtypes.Language.CPP,
    )
    entry, exit_node = add_block_lane_map(nstate, f"{node.label}_block_lanes")
    for name in node.in_connectors:
        nstate.add_memlet_path(
            nstate.add_read(name),
            entry,
            tasklet,
            dst_conn=f"_{name}",
            memlet=Memlet.from_array(name, nsdfg.arrays[name]),
        )
    nstate.add_memlet_path(
        tasklet, exit_node, nstate.add_write("_c"), src_conn="__c", memlet=Memlet.from_array("_c", nsdfg.arrays["_c"])
    )
    return nsdfg
