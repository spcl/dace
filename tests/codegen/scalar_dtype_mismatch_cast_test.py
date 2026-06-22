# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that a scalar tasklet output connector whose dtype differs from its
    destination array is cast on assignment (``B[i] = (float)y``) rather than
    stored by reinterpreting the destination pointer. """
import numpy as np
import dace


def _build(out_conn_dtype: dace.dtypes.typeclass) -> dace.SDFG:
    N = 8
    sdfg = dace.SDFG("scalar_dtype_mismatch_cast")
    sdfg.add_array("A", [N], dace.float64)  # input
    sdfg.add_array("B", [N], dace.float32)  # output: narrower than the computation

    st = sdfg.add_state()
    me, mx = st.add_map("m", {"i": f"0:{N}"})
    t = st.add_tasklet("t", {"x"}, {"y"}, "y = x * 2.0;", language=dace.Language.CPP)
    me.add_in_connector("IN_A")
    me.add_out_connector("OUT_A")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")
    st.add_edge(st.add_read("A"), None, me, "IN_A", dace.Memlet(f"A[0:{N}]"))
    st.add_edge(me, "OUT_A", t, "x", dace.Memlet("A[i]"))
    st.add_edge(t, "y", mx, "IN_B", dace.Memlet("B[i]"))
    st.add_edge(mx, "OUT_B", st.add_write("B"), None, dace.Memlet(f"B[0:{N}]"))

    # Compute in double, store into the f32 array -- the shape a precision-lowering
    # transformation produces (output connector wider than its destination array).
    t.in_connectors["x"] = dace.float64
    t.out_connectors["y"] = out_conn_dtype
    return sdfg


def test_scalar_output_connector_wider_than_array_narrows():
    sdfg = _build(dace.float64)
    A = np.arange(1, 9, dtype=np.float64)
    B = np.zeros(8, dtype=np.float32)
    sdfg(A=A, B=B)
    np.testing.assert_allclose(B, (A * 2.0).astype(np.float32))


if __name__ == "__main__":
    test_scalar_output_connector_wider_than_array_narrows()
