# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``Scal`` BLAS library node (``dace.libraries.blas.nodes.scal``)."""
import pytest

import dace
import dace.libraries.blas as blas
from dace.memlet import Memlet


def test_validate_accepts_reparsed_symbol_instances():
    """Same-named ``N`` reaching Scal.validate at two dtypes (one array's dace.int32
    instance vs the other's dace.int64 instance) must compare equal by name -- and a
    genuine size mismatch must still be rejected."""
    N32 = dace.symbol("N", dace.int32)
    N64 = dace.symbol("N", dace.int64)
    sdfg = dace.SDFG("scal_validate_symbol_identity")
    sdfg.add_array("x", [N32], dace.float64)
    sdfg.add_array("res", [N64], dace.float64)
    state = sdfg.add_state()
    node = blas.Scal("scal")
    state.add_node(node)
    state.add_edge(state.add_read("x"), None, node, "_x", Memlet.from_array("x", sdfg.arrays["x"]))
    state.add_edge(node, "_res", state.add_write("res"), None, Memlet.from_array("res", sdfg.arrays["res"]))
    node.validate(sdfg, state)  # must not raise

    sdfg.arrays["res"].shape = (dace.symbol("P", dace.int32), )
    res_edge = next(e for e in state.out_edges(node) if e.src_conn == "_res")
    res_edge.data = Memlet.from_array("res", sdfg.arrays["res"])
    with pytest.raises(ValueError):
        node.validate(sdfg, state)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
