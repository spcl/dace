# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A GPU_Device map whose range bounds arrive through dynamic-range connectors (the CloudSC shape:
``for jl in range(kidia, kfdia + 1)`` over two scalar program arguments, turned into a map).

A dynamic map range names its bounds by CONNECTOR name, and ``used_symbols`` counts a connector as
locally defined, so such a bound is absent from the scope arglist -- it has to be added back or the
device translation unit names an identifier nothing declares. The connectors here reuse their
containers' names, which is what the frontend produces, so the launch site must not redeclare them
either: ``int kidia = kidia;`` shadows the program argument with itself.
"""
import re

import dace
from dace import dtypes

N = dace.symbol('N', dace.int64)


def build_dynamic_range_kernel_sdfg() -> dace.SDFG:
    """One GPU_Device map over ``kidia:kfdia + 1``, both bounds read from scalar arguments."""
    sdfg = dace.SDFG('dynamic_range_kernel')
    sdfg.add_scalar('kidia', dace.int32)
    sdfg.add_scalar('kfdia', dace.int32)
    sdfg.add_array('a', (N, ), dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('b', (N, ), dace.float64, storage=dtypes.StorageType.GPU_Global)

    state = sdfg.add_state()
    entry, exit_node = state.add_map('kernel', {'jl': 'kidia:kfdia + 1'}, schedule=dtypes.ScheduleType.GPU_Device)
    for bound in ('kidia', 'kfdia'):
        entry.add_in_connector(bound)
        state.add_edge(state.add_read(bound), None, entry, bound, dace.Memlet(f'{bound}[0]'))

    tasklet = state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_memlet_path(state.add_read('a'), entry, tasklet, dst_conn='inp', memlet=dace.Memlet('a[jl]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('b'), src_conn='out', memlet=dace.Memlet('b[jl]'))
    return sdfg


def generate_experimental_cuda(sdfg: dace.SDFG):
    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        return sdfg.generate_code()


def code_of(code_objects, language: str) -> str:
    return '\n'.join(obj.clean_code for obj in code_objects if obj.language == language)


def test_dynamic_range_bounds_reach_the_device_translation_unit():
    """Both bounds must be parameters of the ``__global__`` and of its launch wrapper -- the device
    TU has no other way to see a host scalar."""
    sdfg = build_dynamic_range_kernel_sdfg()
    sdfg.validate()
    device_code = code_of(generate_experimental_cuda(sdfg), 'cu')

    # The declaration line, not just its first parenthesis: ``__launch_bounds__(...)`` sits between
    # the return type and the parameter list.
    kernel_signature = re.search(r'^.*__global__ void.*$', device_code, re.MULTILINE)
    assert kernel_signature is not None, device_code
    for bound in ('kidia', 'kfdia'):
        assert re.search(rf'\bint {bound}\b', kernel_signature.group(0)), kernel_signature.group(0)

    wrapper_signature = re.search(r'void __dace_runkernel_\w+\(([^)]*)\)', device_code)
    assert wrapper_signature is not None, device_code
    for bound in ('kidia', 'kfdia'):
        assert re.search(rf'\bint {bound}\b', wrapper_signature.group(1)), wrapper_signature.group(1)


def test_launch_site_does_not_self_initialise_a_program_argument():
    """``T x = x;`` reads an indeterminate value; the enclosing declaration of that name is what the
    launch has to pass."""
    sdfg = build_dynamic_range_kernel_sdfg()
    sdfg.validate()
    code_objects = generate_experimental_cuda(sdfg)
    host_code = code_of(code_objects, 'cpp')

    self_initialised = re.findall(r'\b\w+[\w\s*&]*?\b(\w+)\s*=\s*\1\s*;', host_code)
    assert not self_initialised, f'self-initialised declarations in host code: {self_initialised}'

    launch = re.search(r'__dace_runkernel_\w+\(([^;]*)\);', host_code)
    assert launch is not None, host_code
    for bound in ('kidia', 'kfdia'):
        assert re.search(rf'\b{bound}\b', launch.group(1)), launch.group(1)


if __name__ == '__main__':
    test_dynamic_range_bounds_reach_the_device_translation_unit()
    test_launch_site_does_not_self_initialise_a_program_argument()
