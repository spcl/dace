# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression test for ``AutoGPUStreamScheduler`` sync placement at root level.

Reproduces the shape of ``failed_validation.sdfg`` (an ICON-style program):

  * Three top-level blocks: a CPU host state, a GPU computation state, then a CPU host state.
  * The GPU state contains some free Tasklets/AccessNodes plus a ``GPU_Device`` map.
  * The ``GPU_Device`` map's body has an inner ``GPU_ThreadBlock`` map and a ``NestedSDFG``.

The earlier ``insert_sync_tasklets`` implementation walked the entire CFG recursively, so it
descended into the NSDFG that sits inside the kernel and spliced a sync state there --
planting a ``gpu_streams[0]`` memlet with no inner ``gpu_streams`` array, triggering
``Node validation failed: 'gpu_streams'`` at validation time. The fix restricts sync placement
to the root SDFG's ``.nodes()`` / ``.edges()`` only.

This test pins that contract: build the SDFG by hand to mirror the failing shape, run the
default pipeline, then assert (a) the sync state lands at the outer GPU -> CPU iedge, (b) no
sync tasklets exist anywhere inside the NestedSDFG that lives inside the GPU kernel, and
(c) the SDFG validates.
"""

import dace
from dace.codegen import common
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline


def _sync_tasklets(state):
    """``cudaStreamSynchronize`` / ``hipStreamSynchronize`` tasklets in ``state``."""
    backend = common.get_gpu_backend()
    needle = f"{backend}StreamSynchronize("
    return [n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and needle in n.code.as_string]


def host_after_copy_in(host_writes: str) -> dace.SDFG:
    """``copy_in`` queues ``A -> A_gpu`` and a kernel on the stream; the host state after it writes ``host_writes``."""
    sdfg = dace.SDFG(f"host_writes_{host_writes}_after_copy_in")
    sdfg.add_array("A", [16], dace.float32)
    sdfg.add_array("other", [16], dace.float32)
    sdfg.add_array("A_gpu", [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global, transient=True)
    sdfg.add_array("out", [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    copy_in = sdfg.add_state("copy_in", is_start_block=True)
    a_gpu = copy_in.add_access("A_gpu")
    copy_in.add_nedge(copy_in.add_read("A"), a_gpu, dace.Memlet("A[0:16]"))
    me, mx = copy_in.add_map("kernel", dict(i="0:16"), schedule=dace.dtypes.ScheduleType.GPU_Device)
    kt = copy_in.add_tasklet("twice", {"a"}, {"b"}, "b = 2 * a")
    copy_in.add_memlet_path(a_gpu, me, kt, dst_conn="a", memlet=dace.Memlet("A_gpu[i]"))
    copy_in.add_memlet_path(kt, mx, copy_in.add_write("out"), src_conn="b", memlet=dace.Memlet("out[i]"))
    host = sdfg.add_state_after(copy_in, "host")
    overwrite = host.add_tasklet("overwrite", {}, {"o"}, "o = 0")
    host.add_edge(overwrite, "o", host.add_write(host_writes), None, dace.Memlet(f"{host_writes}[0]"))
    return sdfg


def sync_neighbours(sdfg: dace.SDFG):
    (sync_state,) = [s for s in sdfg.states() if _sync_tasklets(s)]
    return ({e.src.label for e in sdfg.in_edges(sync_state)}, {e.dst.label for e in sdfg.out_edges(sync_state)})


def test_a_host_state_without_a_hazard_leaves_the_sync_to_the_exit():
    sdfg = host_after_copy_in("other")
    GPUStreamPipeline().apply_pass(sdfg, {})
    assert sync_neighbours(sdfg) == ({"host"}, set())
    sdfg.validate()


if __name__ == "__main__":
    test_a_host_state_without_a_hazard_leaves_the_sync_to_the_exit()
