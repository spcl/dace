# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Host synchronization of device-to-host copies: the first host read of the destination blocks on the copy's stream.

The trisolv shape: scalars copied back and handed to a kernel wrapper, which packs them by value on the host.
Host-to-device copies need none: their consumer runs on the same stream. Codegen only.
"""

import re

import dace
from dace import dtypes


def _generated_code(sdfg: dace.SDFG) -> str:
    """The host frame file, where copies, synchronizations and launches are emitted, so positions compare."""
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        frame = [code for code in sdfg.generate_code() if code.title == "Frame"]
    assert len(frame) == 1, "expected exactly one frame code object"
    return frame[0].clean_code


def _add_kernel(state: dace.SDFGState, label: str, scalar, dst: str) -> None:
    """A GPU_Device map multiplying ``gA`` by the host scalar ``scalar``, writing ``dst``."""
    entry, exit_node = state.add_map(label, dict(i="0:16"), schedule=dtypes.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet(f"{label}_mul", {"a": None, "sc": None}, {"o": None}, "o = a * sc")
    state.add_memlet_path(state.add_read("gA"), entry, tasklet, dst_conn="a", memlet=dace.Memlet("gA[i]"))
    state.add_memlet_path(scalar, entry, tasklet, dst_conn="sc", memlet=dace.Memlet("s[0]"))
    state.add_memlet_path(tasklet, exit_node, state.add_write(dst), src_conn="o", memlet=dace.Memlet(f"{dst}[i]"))


def _scalar_readback_sdfg(name: str, num_kernels: int = 1) -> dace.SDFG:
    """``gA[0]`` copied back to a host scalar, which ``num_kernels`` kernels then consume by value."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("gA", [16], dace.float64, storage=dtypes.StorageType.GPU_Global)
    for idx in range(num_kernels):
        sdfg.add_array(f"gB{idx}", [16], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_scalar("s", dace.float64, storage=dtypes.StorageType.Register, transient=True)

    state = sdfg.add_state("readback")
    scalar = state.add_access("s")
    state.add_nedge(state.add_read("gA"), scalar, dace.Memlet("gA[0] -> [0]"))
    for idx in range(num_kernels):
        _add_kernel(state, f"k{idx}", scalar, f"gB{idx}")
    sdfg.validate()
    return sdfg


def _host_upload_sdfg(name: str) -> dace.SDFG:
    """A host array uploaded to ``gA``, consumed by a kernel on the same stream."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("h", [16], dace.float64)
    sdfg.add_array("gA", [16], dace.float64, storage=dtypes.StorageType.GPU_Global, transient=True)
    sdfg.add_array("gB", [16], dace.float64, storage=dtypes.StorageType.GPU_Global)

    state = sdfg.add_state("upload")
    device = state.add_access("gA")
    state.add_nedge(state.add_read("h"), device, dace.Memlet("h[0:16]"))
    entry, exit_node = state.add_map("k", dict(i="0:16"), schedule=dtypes.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet("double", {"a": None}, {"o": None}, "o = a * 2.0")
    state.add_memlet_path(device, entry, tasklet, dst_conn="a", memlet=dace.Memlet("gA[i]"))
    state.add_memlet_path(tasklet, exit_node, state.add_write("gB"), src_conn="o", memlet=dace.Memlet("gB[i]"))
    sdfg.validate()
    return sdfg


def _launch_call(label: str) -> str:
    """The wrapper's call site. Its forward declaration takes the state struct by type, not ``__state``.

    The experimental target prefixes the wrapper name with the SDFG name, so the label is matched
    after an arbitrary ``\\w*`` prefix; the trailing ``[\\d_]+`` pins it to the id-suffixed wrapper.
    """
    return rf"__dace_runkernel_\w*{label}_[\d_]+\(__state"


def _position(code: str, pattern: str) -> int:
    match = re.search(pattern, code)
    assert match, f"{pattern!r} was not emitted, so this test is anchored on nothing"
    return match.start()


def test_the_synchronization_is_lazy_rather_than_emitted_at_the_copy():
    """Nothing is emitted at the copy site: the sync belongs to the consumer, not the producer."""
    code = _generated_code(_scalar_readback_sdfg("d2h_lazy"))
    copy_block = code[_position(code, r"MemcpyAsync\([^;]*DeviceToHost") : _position(code, _launch_call("k0"))]
    assert copy_block.count("StreamSynchronize") == 1, "more than one synchronization guards a single copy"


def test_a_second_consumer_reuses_the_first_synchronization():
    """One sync before the first consumer suffices; the host's program order covers the rest."""
    code = _generated_code(_scalar_readback_sdfg("d2h_two_consumers", num_kernels=2))
    first = _position(code, _launch_call("k0"))
    second = _position(code, _launch_call("k1"))
    assert "StreamSynchronize" not in code[first:second], (
        "the second consumer of an already synchronized copy synchronizes again"
    )


def test_a_host_to_device_copy_is_not_synchronized_before_the_kernel():
    """The kernel is issued on the copy's stream, which orders it without blocking the host."""
    code = _generated_code(_host_upload_sdfg("h2d_no_sync"))
    copy = _position(code, r"MemcpyAsync\([^;]*HostToDevice")
    launch = _position(code, _launch_call("k"))
    assert "StreamSynchronize" not in code[copy:launch], (
        "a host-to-device copy blocks the host before a kernel that the stream already orders after it"
    )


if __name__ == "__main__":
    test_the_synchronization_is_lazy_rather_than_emitted_at_the_copy()
    test_a_second_consumer_reuses_the_first_synchronization()
    test_a_host_to_device_copy_is_not_synchronized_before_the_kernel()
