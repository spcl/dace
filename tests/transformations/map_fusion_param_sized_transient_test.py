# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fusing two maps renames the second map's parameter in the descriptor of a buffer that parameter sizes.

The second map of ``a[i] = x[i] * 2; for j: buf[0:M - j] = a[j] + y[j:M]; b[j] = sum(buf)`` holds a per-iteration
buffer of ``M - j`` elements. Vertical fusion names the fused map's parameter ``i``; the buffer must then be sized by
``i`` too, or ``j`` is left a free symbol no one defines.
"""

import numpy as np

import dace
from dace.transformation.dataflow import MapFusionVertical

M = dace.symbol("M")


def build() -> dace.SDFG:
    sdfg = dace.SDFG("param_sized_buffer")
    sdfg.add_array("x", [M], dace.float64)
    sdfg.add_array("y", [M], dace.float64)
    sdfg.add_array("b", [M], dace.float64)
    sdfg.add_transient("a", [M], dace.float64)
    sdfg.add_transient("buf", ["M - j"], dace.float64, lifetime=dace.AllocationLifetime.Scope)
    state = sdfg.add_state()
    a = state.add_access("a")
    state.add_mapped_tasklet(
        "double",
        {"i": "0:M"},
        {"v": dace.Memlet("x[i]")},
        "o = 2 * v",
        {"o": dace.Memlet("a[i]")},
        external_edges=True,
        output_nodes={"a": a},
    )
    entry, exit_ = state.add_map("rows", {"j": "0:M"})
    fill_entry, fill_exit = state.add_map("fill", {"k": "0:M - j"})
    fill = state.add_tasklet("fill", {"s", "t"}, {"o"}, "o = s + t")
    buf = state.add_access("buf")
    state.add_memlet_path(a, entry, fill_entry, fill, dst_conn="s", memlet=dace.Memlet("a[j]"))
    yy = state.add_read("y")
    state.add_memlet_path(yy, entry, fill_entry, fill, dst_conn="t", memlet=dace.Memlet("y[j + k]"))
    state.add_memlet_path(fill, fill_exit, buf, src_conn="o", memlet=dace.Memlet("buf[k]"))
    sum_entry, sum_exit = state.add_map("sum", {"k": "0:M - j"})
    add = state.add_tasklet("add", {"v"}, {"o"}, "o = v")
    b = state.add_write("b")
    state.add_memlet_path(buf, sum_entry, add, dst_conn="v", memlet=dace.Memlet("buf[k]"))
    state.add_memlet_path(add, sum_exit, exit_, b, src_conn="o", memlet=dace.Memlet("b[j]", wcr="lambda p, q: p + q"))
    return sdfg


# Reproducer: polybench covariance failed codegen with "Free symbols ['_loop_it_4'] are not declared": FuseMaps renamed
# the mirror map's iterator everywhere but in the size of its per-iteration row buffer
def test_fusing_renames_the_parameter_in_a_buffer_it_sizes():
    sdfg = build()
    sdfg.validate()
    assert sdfg.apply_transformations(MapFusionVertical) == 1
    assert "j" not in {str(s) for s in sdfg.free_symbols}
    assert str(sdfg.arrays["buf"].shape[0]) == "M - i"
    m = 7
    x, y = np.random.default_rng(0).random(m), np.random.default_rng(1).random(m)
    b = np.zeros(m)
    sdfg(x=x, y=y, b=b, M=m)
    want = np.array([(m - j) * 2 * x[j] + y[j:].sum() for j in range(m)])
    np.testing.assert_allclose(b, want, rtol=1e-12)


if __name__ == "__main__":
    test_fusing_renames_the_parameter_in_a_buffer_it_sizes()
