# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single-element operands of a device library call, on either side of the bus.

A vendor call reads a coefficient or a seed through a host pointer as happily as a device one, so
placement prefers the host (nothing to copy before the launch) -- but a value a kernel already wrote
on the device must NOT be dragged back, and the expansion has to read it where it lies. These pin
both halves: the by-value path for host memory, the pointer path (``POINTER_MODE_DEVICE`` for
cuBLAS, ``cub::FutureValue`` for cub) for device memory.
"""
import dace
from dace import dtypes
from dace.libraries.blas.nodes.syrk import Syrk
from dace.libraries.standard.nodes.scan import Scan, ScanOp, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME

N = 32


def syrk_with_runtime_alpha(storage: dtypes.StorageType) -> tuple:
    """A ``Syrk`` whose alpha and beta arrive through connectors backed by ``storage``."""
    sdfg = dace.SDFG(f'syrk_alpha_{storage.name}')
    sdfg.add_array('A', [N, N], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('C', [N, N], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('alpha', [1], dace.float64, storage=storage)
    sdfg.add_array('beta', [1], dace.float64, storage=storage)
    state = sdfg.add_state()
    node = Syrk('syrk', alpha=1, beta=1)
    node.implementation = 'cuBLAS'
    for conn in ('_a', '_c', '_alpha', '_beta'):
        node.add_in_connector(conn)
    node.add_out_connector('_c')
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_a', dace.Memlet('A[0:N, 0:N]'.replace('N', str(N))))
    state.add_edge(state.add_read('C'), None, node, '_c', dace.Memlet(f'C[0:{N}, 0:{N}]'))
    state.add_edge(state.add_read('alpha'), None, node, '_alpha', dace.Memlet('alpha[0]'))
    state.add_edge(state.add_read('beta'), None, node, '_beta', dace.Memlet('beta[0]'))
    state.add_edge(node, '_c', state.add_write('C'), None, dace.Memlet(f'C[0:{N}, 0:{N}]'))
    sdfg.validate()
    return sdfg, state, node


def scan_with_seed(storage: dtypes.StorageType) -> tuple:
    """An inclusive ``Scan`` whose seed is backed by ``storage``."""
    sdfg = dace.SDFG(f'scan_seed_{storage.name}')
    sdfg.add_array('A', [N], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [N], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('seed', [1], dace.float64, storage=storage)
    state = sdfg.add_state()
    node = Scan('scan', op=ScanOp.SUM, exclusive=False)
    node.implementation = 'CUDA'
    for conn in (INPUT_CONNECTOR_NAME, '_scan_init'):
        node.add_in_connector(conn)
    node.add_out_connector(OUTPUT_CONNECTOR_NAME)
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, INPUT_CONNECTOR_NAME, dace.Memlet(f'A[0:{N}]'))
    state.add_edge(state.add_read('seed'), None, node, '_scan_init', dace.Memlet('seed[0]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write('B'), None, dace.Memlet(f'B[0:{N}]'))
    sdfg.validate()
    return sdfg, state, node


def expanded_code(sdfg: dace.SDFG) -> str:
    """Every CPP body the expansion produced -- tasklets AND global code, concatenated.

    The cub call does not live in the tasklet: the scan expansion emits it as a CUDA unit through
    ``append_global_code(..., 'cuda')`` and leaves the tasklet holding only the call to it. Reading
    tasklets alone would make every assertion here pass or fail for the wrong reason.
    """
    sdfg.expand_library_nodes()
    parts = [node.code.as_string for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.sdfg.nodes.Tasklet)]
    for nested in sdfg.all_sdfgs_recursive():
        parts.extend(block.as_string for block in nested.global_code.values())
    return '\n'.join(parts)


def test_a_host_coefficient_is_read_by_value():
    """Nothing to copy: the value is right there, so the call takes a host pointer to a local."""
    sdfg, _, _ = syrk_with_runtime_alpha(dtypes.StorageType.CPU_Heap)
    code = expanded_code(sdfg)
    assert 'CUBLAS_POINTER_MODE_HOST' in code
    assert '&__alpha' in code, 'the host path should point at its own local'


def test_a_device_coefficient_is_read_where_it_lies():
    """Copying it back would cost a transfer and a sync; cuBLAS can dereference it itself."""
    sdfg, _, _ = syrk_with_runtime_alpha(dtypes.StorageType.GPU_Global)
    code = expanded_code(sdfg)
    assert 'CUBLAS_POINTER_MODE_DEVICE' in code
    assert '&__alpha' not in code, 'a device coefficient must not be dereferenced by the host'


def test_a_device_coefficient_reaches_the_tasklet_as_a_pointer():
    """A scalar connector would be dereferenced host-side, which SDFG validation rejects outright."""
    sdfg, _, _ = syrk_with_runtime_alpha(dtypes.StorageType.GPU_Global)
    sdfg.expand_library_nodes()
    tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.sdfg.nodes.Tasklet)]
    typed = [t.in_connectors['__alpha_in'] for t in tasklets if '__alpha_in' in t.in_connectors]
    assert typed, 'the coefficient connector went missing'
    assert all(isinstance(t, dtypes.pointer) for t in typed), typed
    sdfg.validate()


def test_a_host_seed_goes_to_cub_by_value():
    sdfg, _, _ = scan_with_seed(dtypes.StorageType.CPU_Heap)
    code = expanded_code(sdfg)
    assert 'FutureValue' not in code, 'a host seed needs no future'


def test_a_device_seed_goes_to_cub_as_a_future():
    """``cub::FutureValue`` is the documented way to hand ``DeviceScan`` a seed it must read itself."""
    sdfg, _, _ = scan_with_seed(dtypes.StorageType.GPU_Global)
    code = expanded_code(sdfg)
    assert 'FutureValue' in code, code[:400]
    sdfg.validate()
