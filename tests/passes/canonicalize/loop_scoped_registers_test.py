# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.finalize import finalize_transient_storage

N, K = dace.symbol("N"), dace.symbol("K")


def per_iteration_scratch(name: str, extent: str, small: bool) -> dace.SDFG:
    """``for ip: c = d[ip]; tmp[0:extent] = w[ip] * j; out[ip] = sum(tmp)``, ``K`` declared SMALL when ``small``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("d", [N], dace.int64)
    sdfg.add_array("w", [N], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_symbol("K", dace.int64)
    sdfg.add_symbol("c", dace.int64)
    sdfg.add_transient("tmp", [extent], dace.float64)
    if small:
        sdfg.set_symbol_scale("K", dace.symbolic.Scale.SMALL)
    loop = LoopRegion("particles", "ip < N", "ip", "ip = 0", "ip = ip + 1")
    sdfg.add_node(loop, is_start_block=True)
    read = loop.add_state("read", is_start_block=True)
    body = loop.add_state("body")
    loop.add_edge(read, body, dace.InterstateEdge(assignments={"c": "d[ip]"}))
    body.add_mapped_tasklet(
        "fill",
        {"j": f"0:{extent}"},
        {"wi": dace.Memlet("w[ip]")},
        "t = wi * j",
        {"t": dace.Memlet("tmp[j]")},
        external_edges=True,
    )
    tmp = next(n for n in body.data_nodes() if n.data == "tmp")
    body.add_mapped_tasklet(
        "sum",
        {"j": f"0:{extent}"},
        {"t": dace.Memlet("tmp[j]")},
        "o = t",
        {"o": dace.Memlet("out[ip]", wcr="lambda a, b: a + b")},
        external_edges=True,
        input_nodes={"tmp": tmp},
    )
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("small", [True, False])
def test_a_scratch_sized_by_the_iteration_goes_on_the_stack(small):
    """``K + 2 - c`` with ``c`` read per iteration and ``K`` SMALL: a VLA, not a ``new``/``delete`` per iteration.
    With ``K`` of the default scale the size is unbounded and stays on the heap."""
    sdfg = per_iteration_scratch(f"loop_vla_{small}", "K + 2 - c", small)
    finalize_transient_storage(sdfg, dtypes.DeviceType.CPU)
    storage = sdfg.arrays["tmp"].storage
    assert (storage == dtypes.StorageType.Register and storage.force) is small
    n, k = 50, 3
    rng = np.random.default_rng(0)
    d, w = rng.integers(0, 2, n), rng.random(n)
    out = np.zeros(n)
    sdfg(d=d, w=w, out=out, N=n, K=k)
    ref = np.array([w[i] * sum(range(k + 2 - d[i])) for i in range(n)])
    assert np.allclose(out, ref)


def test_a_scratch_the_loop_does_not_resize_stays_on_the_heap():
    sdfg = per_iteration_scratch("loop_invariant_scratch", "N", small=True)
    finalize_transient_storage(sdfg, dtypes.DeviceType.CPU)
    assert sdfg.arrays["tmp"].storage != dtypes.StorageType.Register


if __name__ == "__main__":
    for small in [True, False]:
        test_a_scratch_sized_by_the_iteration_goes_on_the_stack(small)
    test_a_scratch_the_loop_does_not_resize_stays_on_the_heap()
