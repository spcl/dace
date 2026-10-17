# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map of row scans reaches the device as ONE segmented scan, not a host loop of per-row launches.

``for i: for j in 1..N: b[i, j] = b[i, j-1] + a[i, j]`` canonicalizes to a map over the rows around a
``Scan``. Kept on the host, it was 13482 trips of three launches and a ``DeviceScan`` each
(``safety_map_of_scans``: 296 ms on an MI300 against 23 ms on 16 CPU cores).
"""

import numpy as np

import dace
from dace.libraries.standard.nodes.scan import Scan
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N", dtype=dace.int64, positive=True)


@dace.program
def map_of_scans(a: dace.float64[N, N], b: dace.float64[N, N]):
    for i in range(N):
        for j in range(1, N):
            b[i, j] = b[i, j - 1] + a[i, j]


def test_the_row_scans_become_one_segmented_scan_between_two_kernels():
    sdfg = map_of_scans.to_sdfg()
    canonicalize(sdfg, target="gpu", validate_all=False)
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, target="gpu")

    scans = [(node, state) for node, state in sdfg.all_nodes_recursive() if isinstance(node, Scan)]
    assert len(scans) == 1, scans
    scan, state = scans[0]
    assert state.entry_node(scan) is None, "the scans must not stay inside a map, which runs as a host loop"
    assert scan.segments == N, f"one segment per row, not {scan.segments}"
    assert scan.implementation == "CUDA", scan.implementation
    kernels = [
        node
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry) and parent.entry_node(node) is None
    ]
    assert all(kernel.map.schedule == dace.ScheduleType.GPU_Device for kernel in kernels), kernels
    assert all(len(kernel.map.params) == 2 for kernel in kernels), "staging and apply are one 2-D kernel each"


def test_the_cpu_form_keeps_its_parallel_map_of_sequential_scans():
    """Batching is part of the device move: the CPU keeps one thread per row walking its scan."""
    sdfg = map_of_scans.to_sdfg()
    canonicalize(sdfg, target="cpu", validate_all=False)
    finalize_for_target(sdfg, target="cpu")
    scan, state = next((node, state) for node, state in sdfg.all_nodes_recursive() if isinstance(node, Scan))
    assert scan.segments == 1 and state.entry_node(scan) is not None

    rng = np.random.default_rng(20261006)
    a, b = rng.random((16, 16)), rng.random((16, 16))
    expected = b.copy()
    for j in range(1, 16):
        expected[:, j] = expected[:, j - 1] + a[:, j]
    sdfg(a=a, b=b, N=16)
    np.testing.assert_allclose(b, expected, rtol=1e-12)


if __name__ == "__main__":
    test_the_row_scans_become_one_segmented_scan_between_two_kernels()
    test_the_cpu_form_keeps_its_parallel_map_of_sequential_scans()
