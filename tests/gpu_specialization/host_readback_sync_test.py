# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A host read of a device-to-host copy waits for the stream, wherever the host reads it.

The single-stream scheduler synced only on an edge leaving a GPU block. A branch condition reads on the host
too, and so does an interstate assignment behind a host-only block while the copy is still queued:
cloudsc_subsidence_forcing's ``if iphase_index_host > 0`` and ``llfall_index = llfall_host[0]`` read stale
values on the GPU canon column unless every kernel was serialized.
"""

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import is_pipeline_sync_tasklet

N = dace.symbol("N")


def kernel_state(region, name: str, out: str) -> dace.SDFGState:
    state = region.add_state(name)
    state.add_mapped_tasklet(
        f"{name}_kernel",
        dict(i="0:N"),
        {},
        "o = 1.0",
        {"o": dace.Memlet(f"{out}[i]")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    return state


def read_back(read_in_condition: bool) -> tuple[dace.SDFG, dace.SDFGState, object]:
    """A kernel writes ``g`` and copies ``g[0]`` to the host array ``h``; the host then reads ``h``, either in a
    branch condition or in an interstate assignment behind a host-only state."""
    sdfg = dace.SDFG("read_back_in_condition" if read_in_condition else "read_back_after_host_state")
    sdfg.add_array("g", [N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("out", [N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("h", [1], dace.float64, storage=dace.StorageType.CPU_Heap, transient=True)
    produce = kernel_state(sdfg, "produce", "g")
    produce.add_nedge(produce.sink_nodes()[0], produce.add_write("h"), dace.Memlet("g[0:1]"))
    if read_in_condition:
        reader = ConditionalBlock("on_h", sdfg=sdfg)
        sdfg.add_node(reader)
        branch = ControlFlowRegion("on_h_body", sdfg=sdfg)
        reader.add_branch(CodeBlock("h[0] > 0"), branch)
        kernel_state(branch, "consume", "out")
        sdfg.add_edge(produce, reader, dace.InterstateEdge())
    else:
        host = sdfg.add_state("host_only")
        host.add_tasklet("noop", {}, {}, "pass")
        sdfg.add_edge(produce, host, dace.InterstateEdge())
        sdfg.add_symbol("v", dace.float64)
        reader = kernel_state(sdfg, "consume", "out")
        sdfg.add_edge(host, reader, dace.InterstateEdge(assignments={"v": "h[0]"}))
    sdfg.validate()
    return sdfg, produce, reader


def reaches_without_sync(sdfg: dace.SDFG, src, dst) -> bool:
    syncs = {
        block
        for block in sdfg.nodes()
        if isinstance(block, dace.SDFGState) and any(is_pipeline_sync_tasklet(n) for n in block.nodes())
    }
    seen, pending = {src}, [src]
    while pending:
        for edge in sdfg.out_edges(pending.pop()):
            if edge.dst is dst:
                return True
            if edge.dst not in syncs and edge.dst not in seen:
                seen.add(edge.dst)
                pending.append(edge.dst)
    return False


def test_a_branch_condition_reading_a_copied_back_value_waits_for_the_stream():
    sdfg, produce, reader = read_back(read_in_condition=True)
    GPUStreamPipeline().apply_pass(sdfg, {})
    assert not reaches_without_sync(sdfg, produce, reader)


def test_a_read_back_behind_a_host_state_waits_for_the_stream():
    sdfg, produce, reader = read_back(read_in_condition=False)
    GPUStreamPipeline().apply_pass(sdfg, {})
    assert not reaches_without_sync(sdfg, produce, reader)


if __name__ == "__main__":
    test_a_branch_condition_reading_a_copied_back_value_waits_for_the_stream()
    test_a_read_back_behind_a_host_state_waits_for_the_stream()
