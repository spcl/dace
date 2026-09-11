# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A GPU-to-host copy has to be waited for before the host reads its destination.

The copy is issued as ``MemcpyAsync`` on a stream, so nothing orders it against the host. The
trisolv shape is the one that bites: a device scalar is copied into host memory and the very next
kernel launch takes that scalar *by value*, which means the host reads it while packing the launch
arguments. Pageable host memory hides this today because the driver blocks on it anyway; pinned
memory, HIP and graph capture do not.

These assert on emitted code, so they need a GPU for neither compilation nor a run.
"""

import re

import dace
from dace import dtypes

COPY = r'Memcpy(?:2D)?Async\('
SYNC = r'StreamSynchronize\('


def generated_code(sdfg: dace.SDFG) -> str:
    """Every generated file for ``sdfg``, joined."""
    return '\n'.join(code.clean_code for code in sdfg.generate_code())


def scalar_through_host(scalar_storage: dtypes.StorageType) -> dace.SDFG:
    """Device scalar copied into ``scalar_storage``, then consumed by value in a device kernel."""
    sdfg = dace.SDFG('d2h_then_kernel_' + scalar_storage.name)
    sdfg.add_array('A', [20], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [20], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_scalar('s', dace.float64, storage=scalar_storage, transient=True)

    state = sdfg.add_state('main')
    a = state.add_access('A')
    s = state.add_access('s')
    b = state.add_access('B')
    state.add_edge(a, None, s, None, dace.Memlet('A[0]'))

    entry, exit_ = state.add_map('kern', {'i': '0:20'}, schedule=dtypes.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet('scale', {'inp', 'sc'}, {'out'}, 'out = inp * sc')
    for conn in ('IN_s', 'IN_A'):
        entry.add_in_connector(conn)
    for conn in ('OUT_s', 'OUT_A'):
        entry.add_out_connector(conn)
    exit_.add_in_connector('IN_B')
    exit_.add_out_connector('OUT_B')
    state.add_edge(s, None, entry, 'IN_s', dace.Memlet('s[0]'))
    state.add_edge(a, None, entry, 'IN_A', dace.Memlet('A[0:20]'))
    state.add_edge(entry, 'OUT_s', tasklet, 'sc', dace.Memlet('s[0]'))
    state.add_edge(entry, 'OUT_A', tasklet, 'inp', dace.Memlet('A[i]'))
    state.add_edge(tasklet, 'out', exit_, 'IN_B', dace.Memlet('B[i]'))
    state.add_edge(exit_, 'OUT_B', b, None, dace.Memlet('B[0:20]'))
    sdfg.validate()
    return sdfg


def device_to_device() -> dace.SDFG:
    """The same shape with the intermediate left on the device, so no host ever reads it."""
    sdfg = dace.SDFG('d2d_then_kernel')
    sdfg.add_array('A', [20], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [20], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('C', [20], dace.float64, storage=dtypes.StorageType.GPU_Global, transient=True)

    state = sdfg.add_state('main')
    a = state.add_access('A')
    c = state.add_access('C')
    b = state.add_access('B')
    state.add_edge(a, None, c, None, dace.Memlet('A[0:20]'))

    entry, exit_ = state.add_map('kern', {'i': '0:20'}, schedule=dtypes.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 2')
    entry.add_in_connector('IN_C')
    entry.add_out_connector('OUT_C')
    exit_.add_in_connector('IN_B')
    exit_.add_out_connector('OUT_B')
    state.add_edge(c, None, entry, 'IN_C', dace.Memlet('C[0:20]'))
    state.add_edge(entry, 'OUT_C', tasklet, 'inp', dace.Memlet('C[i]'))
    state.add_edge(tasklet, 'out', exit_, 'IN_B', dace.Memlet('B[i]'))
    state.add_edge(exit_, 'OUT_B', b, None, dace.Memlet('B[0:20]'))
    sdfg.validate()
    return sdfg


def host_copy_and_launch(code: str):
    """The device-to-host copy and the kernel launch that reads its destination."""
    copy = re.search(COPY + r'[^;]*DeviceToHost[^;]*\)', code)
    assert copy, 'no device-to-host copy was emitted, so this test is anchored on nothing'
    launch = re.search(r'__dace_runkernel_\w+\(', code[copy.end() :])
    assert launch, 'no kernel launch follows the device-to-host copy, so this test is anchored on nothing'
    return copy, copy.end() + launch.start()


def test_a_device_to_host_copy_is_synchronized_before_its_host_reader():
    code = generated_code(scalar_through_host(dtypes.StorageType.CPU_Heap))
    copy, launch_at = host_copy_and_launch(code)
    between = code[copy.end() : launch_at]
    assert re.search(SYNC, between), (
        'the kernel launch reads the copied scalar by value on the host, but no stream synchronization '
        'separates it from the asynchronous device-to-host copy'
    )


def test_the_synchronization_names_the_stream_the_copy_was_issued_on():
    """Waiting on some other stream orders nothing."""
    code = generated_code(scalar_through_host(dtypes.StorageType.CPU_Heap))
    copy, launch_at = host_copy_and_launch(code)
    stream = copy.group(0).rsplit(',', 1)[1].strip().rstrip(')')
    sync = re.search(SYNC + re.escape(stream) + r'\)', code[copy.end() : launch_at])
    assert sync, f'the synchronization before the kernel launch does not wait on {stream}'


def test_pinned_destinations_are_synchronized_too():
    """Pinned memory is where the accidental blocking of pageable copies stops covering for this."""
    code = generated_code(scalar_through_host(dtypes.StorageType.CPU_Pinned))
    copy, launch_at = host_copy_and_launch(code)
    assert re.search(SYNC, code[copy.end() : launch_at]), (
        'a copy into pinned host memory is not waited for before the host reads the destination'
    )


def test_a_device_to_device_copy_does_not_wait_on_the_host():
    """No host reads the destination, so a host wait would only serialize the stream."""
    code = generated_code(device_to_device())
    copy = re.search(COPY + r'[^;]*DeviceToDevice[^;]*\)', code)
    assert copy, 'no device-to-device copy was emitted, so this test is anchored on nothing'
    launch = re.search(r'__dace_runkernel_\w+\(', code[copy.end() :])
    assert launch, 'no kernel launch follows the device-to-device copy, so this test is anchored on nothing'
    between = code[copy.end() : copy.end() + launch.start()]
    assert not re.search(SYNC, between), (
        'a device-to-device copy is followed by a host wait, which serializes the '
        'stream for a destination no host reads'
    )
    assert not re.search(r'DeviceToHost', code), 'the device-to-device fixture emitted a host copy after all'


if __name__ == '__main__':
    test_a_device_to_host_copy_is_synchronized_before_its_host_reader()
    test_the_synchronization_names_the_stream_the_copy_was_issued_on()
    test_pinned_destinations_are_synchronized_too()
    test_a_device_to_device_copy_does_not_wait_on_the_host()
