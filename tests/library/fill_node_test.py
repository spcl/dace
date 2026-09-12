# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`FillLibraryNode` and its pure / CPU / CUDA / tasklet expansions."""
import contextlib
import re
from typing import Optional, Sequence

import dace
from dace.libraries.standard.nodes.fill import FillLibraryNode, byte_pattern, select_fill_implementation

import numpy as np
import pytest


def make_fill_sdfg(implementation: Optional[str],
                   shape: Sequence[int],
                   subset: str,
                   gpu: bool = True,
                   name: str = "fill_sdfg",
                   dtype: dace.dtypes.typeclass = dace.dtypes.float64,
                   value=0) -> dace.SDFG:
    """Build an SDFG that fills a sub-region of a single array.

    :param implementation: ``FillLibraryNode.implementation`` (``None`` keeps ``'Auto'``).
    :param shape: array shape (sequence of dim extents).
    :param subset: memlet subset string for the fill's output edge.
    :param gpu: True for ``GPU_Global`` storage, False for ``CPU_Heap``.
    :param name: SDFG name.
    :param dtype: element type of the filled array.
    :param value: the constant the node writes.
    :returns: the constructed SDFG.
    """
    sdfg = dace.SDFG(name)
    arr_name = "gpuB" if gpu else "B"
    storage = dace.dtypes.StorageType.GPU_Global if gpu else dace.dtypes.StorageType.CPU_Heap
    sdfg.add_array(name=arr_name, shape=list(shape), dtype=dtype, storage=storage, transient=False)

    state = sdfg.add_state("main")
    out = state.add_access(arr_name)
    libnode = FillLibraryNode(name="fill_libnode", value=value)
    if implementation is not None:
        libnode.implementation = implementation
    state.add_edge(libnode, FillLibraryNode.OUTPUT_CONNECTOR_NAME, out, None,
                   dace.memlet.Memlet(f"{arr_name}[{subset}]"))
    return sdfg


def _get_sdfg(implementation: Optional[str], gpu: bool = True) -> dace.SDFG:
    """1-D slice fill."""
    return make_fill_sdfg(implementation, (200, ), "50:100", gpu=gpu, name="fill_sdfg")


def _get_multi_dim_sdfg(implementation: Optional[str], gpu: bool = True) -> dace.SDFG:
    """3-D sub-block fill."""
    return make_fill_sdfg(implementation, (50, 2, 2), "40:50, 0:2, 0:2", gpu=gpu, name="fill_sdfg2")


def test_fill_pure_1d_cpu():
    """``pure`` zeros the 1D CPU slice, leaving the rest unchanged."""
    sdfg = _get_sdfg("pure", gpu=False)
    sdfg.name += "_pure_cpu"
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = np.ones((200, ), dtype=np.float64)
    exe(B=B)

    assert np.all(B[:50] == 1)
    assert np.all(B[100:] == 1)
    assert np.all(B[50:100] == 0)


def test_fill_pure_3d_cpu():
    """``pure`` zeros the 3D CPU sub-block, leaving the rest unchanged."""
    sdfg = _get_multi_dim_sdfg("pure", gpu=False)
    sdfg.name += "_pure_cpu_multi_dim"
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = np.ones((50, 2, 2), dtype=np.float64)
    exe(B=B)

    assert np.all(B[0:40, :, :] == 1)
    assert np.all(B[40:50, :, :] == 0)


@pytest.mark.gpu
def test_fill_pure_1d_gpu():
    """``pure`` zeros the 1D GPU slice, leaving the rest unchanged."""
    import cupy as cp

    sdfg = _get_sdfg("pure", gpu=True)
    sdfg.name += "_pure_gpu"
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = cp.ones((200, ), dtype=cp.float64)
    exe(gpuB=B)

    assert cp.all(B[:50] == 1)
    assert cp.all(B[100:] == 1)
    assert cp.all(B[50:100] == 0)


@pytest.mark.gpu
def test_fill_pure_3d_gpu():
    """``pure`` zeros the 3D GPU sub-block, leaving the rest unchanged."""
    import cupy as cp

    sdfg = _get_multi_dim_sdfg("pure", gpu=True)
    sdfg.name += "_pure_gpu_multi_dim"
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = cp.ones((50, 2, 2), dtype=np.float64)
    exe(gpuB=B)

    assert cp.all(B[0:40, :, :] == 1)
    assert cp.all(B[40:50, :, :] == 0)


@pytest.mark.gpu
def test_fill_cuda_1d_gpu():
    """``CUDA`` zeros the 1D GPU slice, leaving the rest unchanged."""
    import cupy as cp

    sdfg = _get_sdfg("CUDA", gpu=True)
    sdfg.name += "_cuda_gpu"
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = cp.ones((200, ), dtype=cp.float64)
    exe(gpuB=B)

    assert cp.all(B[:50] == 1)
    assert cp.all(B[100:] == 1)
    assert cp.all(B[50:100] == 0)


@pytest.mark.gpu
def test_fill_cuda_3d_gpu():
    """``CUDA`` zeros the 3D GPU sub-block, leaving the rest unchanged."""
    import cupy as cp

    sdfg = _get_multi_dim_sdfg("CUDA", gpu=True)
    sdfg.name += "_cuda_gpu_multi_dim"
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = cp.ones((50, 2, 2), dtype=np.float64)
    exe(gpuB=B)

    assert cp.all(B[0:40, :, :] == 1)
    assert cp.all(B[40:50, :, :] == 0)


@pytest.mark.gpu
def test_fill_cuda_rejects_cpu_storage():
    """``CUDA`` targeting a CPU array is rejected."""
    sdfg = _get_sdfg("CUDA", gpu=False)
    sdfg.name += "_cuda_cpu"
    sdfg.validate()
    sdfg.expand_library_nodes()
    with pytest.raises(Exception):
        sdfg.validate()
        sdfg.compile()


def test_fill_auto_routes_non_contiguous_to_pure_cpu():
    """Auto routes a non-contiguous CPU subset to ``pure`` (one call would write outside the region)."""
    sdfg = make_fill_sdfg(None, (10, 20), "2:8, 5:15", gpu=False, name="fill_noncontig_cpu_auto")
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = np.ones((10, 20), dtype=np.float64)
    exe(B=B)
    # The 6x10 sub-block is zeroed; everything else stays 1.
    expected = np.ones((10, 20), dtype=np.float64)
    for i in range(2, 8):
        for j in range(5, 15):
            expected[i, j] = 0
    np.testing.assert_array_equal(B, expected)


def test_fill_cpu_rejects_non_contiguous_subset():
    """Explicit ``CPU`` expansion rejects a non-contiguous subset (one call would overrun the region)."""
    sdfg = make_fill_sdfg("CPU", (10, 20), "2:8, 5:15", gpu=False, name="fill_noncontig_cpu_explicit")
    sdfg.validate()
    with pytest.raises(ValueError, match="contiguous"):
        sdfg.expand_library_nodes()


@pytest.mark.gpu
def test_fill_cuda_rejects_non_contiguous_subset():
    """Explicit ``CUDA`` expansion rejects a non-contiguous subset (one ``cudaMemsetAsync`` would overrun)."""
    sdfg = make_fill_sdfg("CUDA", (10, 20), "2:8, 5:15", gpu=True, name="fill_noncontig_cuda_explicit")
    sdfg.validate()
    with pytest.raises(ValueError, match="contiguous"):
        sdfg.expand_library_nodes()


def test_fill_register_outside_kernel_routes_to_cpu_tasklet():
    """A Fill on a Register outside a GPU kernel scope lowers to a direct host-side Tasklet."""
    sdfg = dace.SDFG('fill_reg_outside_kernel')
    sdfg.add_array('R', [1], dace.float64, dace.StorageType.Register, transient=True)
    state = sdfg.add_state('s')

    r = state.add_access('R')
    fill_node = FillLibraryNode(name='fill_r')
    state.add_node(fill_node)
    state.add_edge(fill_node, FillLibraryNode.OUTPUT_CONNECTOR_NAME, r, None, dace.Memlet('R[0]'))

    sdfg.expand_library_nodes()

    # Verify no complex structures or CUDA launch strings are generated on the host for raw registers
    nsdfg_count = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    assert nsdfg_count == 0, "Host register fill should expand to a direct Tasklet, not a NestedSDFG."

    assignments = [
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet) and '= 0' in n.code.as_string
    ]
    assert assignments, "Expected a basic literal assignment tasklet on the host."


def test_fill_register_inside_kernel_routes_to_sequential():
    """A multi-element Fill targeting a Register array inside a GPU kernel maps to sequential in-kernel logic."""
    sdfg = dace.SDFG('fill_reg_inside_kernel')
    sdfg.add_array('R', [4], dace.float64, dace.StorageType.Register, transient=True)
    state = sdfg.add_state('s')

    me, mx = state.add_map('kernel', dict(i='0:1'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    r = state.add_access('R')
    fill_node = FillLibraryNode(name='fill_r')
    state.add_node(fill_node)

    state.add_memlet_path(me, fill_node, memlet=dace.Memlet())
    state.add_edge(fill_node, FillLibraryNode.OUTPUT_CONNECTOR_NAME, r, None, dace.Memlet('R[0:4]'))
    state.add_memlet_path(r, mx, memlet=dace.Memlet())

    sdfg.expand_library_nodes()

    # Ensure it did not lower to a host-side or invalid device-side memset call. The expansion
    # names the API after the configured backend, so match both spellings, not just cuda's.
    memsets = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.Tasklet) and ('cudaMemset' in n.code.as_string or 'hipMemset' in n.code.as_string)
    ]
    assert len(memsets) == 0, "Cannot issue a device memset on local GPU registers."

    # It should fall back to an internal loop/unrolled tasklet chain inside the device state
    assert any(isinstance(n, dace.nodes.Tasklet) for n, _ in sdfg.all_nodes_recursive())


def test_fill_single_gpu_shared_inside_kernel_expands_clean():
    """A single-element fill targeting GPU-resident storage *inside* a GPU kernel is valid device
    code (a device-side ``_out = 0``) and must expand cleanly. Regression: the ``tasklet`` guard fired
    on exactly this valid case, and its error path dereferenced the output *name* (a ``str``) as
    ``inp.storage`` -> ``AttributeError``."""
    sdfg = dace.SDFG('fill_shared_inside_kernel')
    sdfg.add_array('s', [1], dace.float64, dace.StorageType.GPU_Shared, transient=True)
    state = sdfg.add_state('s')

    me, mx = state.add_map('kernel', dict(i='0:1'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    s_acc = state.add_access('s')
    fill_node = FillLibraryNode(name='fill_s')
    state.add_node(fill_node)
    state.add_memlet_path(me, fill_node, memlet=dace.Memlet())
    state.add_edge(fill_node, FillLibraryNode.OUTPUT_CONNECTOR_NAME, s_acc, None, dace.Memlet('s[0]'))
    state.add_memlet_path(s_acc, mx, memlet=dace.Memlet())

    sdfg.expand_library_nodes()  # must not raise

    assert any(isinstance(n, dace.nodes.Tasklet) and '= 0' in n.code.as_string
               for n, _ in sdfg.all_nodes_recursive()), "Expected a scalar zero-assignment tasklet."


def test_fill_tasklet_rejects_gpu_storage_from_host_scope():
    """The single-element ``tasklet`` expansion emits ``_out = 0`` in its own scope; from host scope it
    cannot target GPU-resident storage (a scalar assignment cannot write device memory), so it must
    raise a clean ``ValueError``. Regression: the guard tested the wrong side, letting this host->GPU
    case through instead of rejecting it."""
    sdfg = make_fill_sdfg("tasklet", (1, ), "0:1", gpu=True, name="fill_tasklet_host_gpu")
    sdfg.validate()
    with pytest.raises(ValueError):
        sdfg.expand_library_nodes()


def test_fill_pure_strided_map_matches_array():
    """A ``pure`` fill over a strided subset must give the mapped tasklet the same collapsed
    extent as the wrapper array descriptor. Regression: ``map_lengths`` was recomputed from
    ``out_subset.size()`` instead of the collapsed shape used for the array, so the map bounds
    could diverge from the array rank/extent."""
    sdfg = make_fill_sdfg(None, (9, ), "0:9:3", gpu=False, name="fill_strided_cpu")
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()  # a diverged map/array would fail validation here
    exe = sdfg.compile()

    B = np.ones((9, ), dtype=np.float64)
    exe(B=B)

    expected = np.ones((9, ), dtype=np.float64)
    expected[0:9:3] = 0  # indices 0, 3, 6
    np.testing.assert_array_equal(B, expected)


@contextlib.contextmanager
def _pinned_transfer_threshold(value):
    """Pin ``compiler.cpu.parallel_transfer_min_elements`` so Auto selection is deterministic."""
    orig = dace.config.Config.get("compiler", "cpu", "parallel_transfer_min_elements")
    dace.config.Config.set("compiler", "cpu", "parallel_transfer_min_elements", value=value)
    try:
        yield
    finally:
        dace.config.Config.set("compiler", "cpu", "parallel_transfer_min_elements", value=orig)


def cpu_fill_sdfg(extent, name):
    """Single-state CPU_Heap ``FillLibraryNode`` zeroing ``0:extent``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("dst", [extent], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("s")
    libnode = FillLibraryNode(name="ms")
    state.add_edge(libnode, FillLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_access("dst"), None,
                   dace.Memlet(f"dst[0:{extent}]"))
    sdfg.validate()
    return sdfg, libnode


def _generated_code(sdfg):
    return "\n".join(obj.code for obj in sdfg.generate_code())


def test_fill_below_threshold_emits_a_single_call():
    """A constant-size CPU fill below the threshold lowers to one ``std::fill_n``, not an OpenMP loop.
    gcc and clang reduce that call to a memset at the Release level dace builds with when the
    value's object representation allows it."""
    with _pinned_transfer_threshold(1024):
        sdfg, libnode = cpu_fill_sdfg(100, "fill_below_threshold")
        sdfg.expand_library_nodes(recursive=True)
        assert libnode.implementation == 'CPU'
        code = _generated_code(sdfg)
        assert 'std::fill_n' in code
        assert '#pragma omp parallel for' not in code


def test_fill_at_threshold_emits_omp_parallel_for():
    """A constant-size CPU fill at/above the threshold lowers to an OpenMP element map, not one call."""
    with _pinned_transfer_threshold(1024):
        sdfg, libnode = cpu_fill_sdfg(4096, "fill_at_threshold")
        sdfg.expand_library_nodes(recursive=True)
        assert libnode.implementation == 'pure'
        code = _generated_code(sdfg)
        assert '#pragma omp parallel for' in code
        assert 'std::fill_n' not in code


def test_fill_symbolic_size_emits_omp_parallel_for():
    """A symbolic (compile-time-unknown) CPU fill size is assumed large, so it takes the same
    OpenMP-parallel path as a large constant, never the single ``std::fill_n``."""
    with _pinned_transfer_threshold(1024):
        n = dace.symbol('N_fill_symbolic')
        sdfg, libnode = cpu_fill_sdfg(n, "fill_symbolic_size")
        sdfg.expand_library_nodes(recursive=True)
        assert libnode.implementation == 'pure'
        code = _generated_code(sdfg)
        assert '#pragma omp parallel for' in code
        assert 'std::fill_n' not in code


if __name__ == "__main__":
    pytest.main([__file__])

NARROW_FLOATS = [
    pytest.param(dace.float16, 'float16'),
    pytest.param(dace.bfloat16, 'bfloat16'),
    pytest.param(dace.float8_e4m3fn, 'float8_e4m3fn'),
    pytest.param(dace.float8_e5m2, 'float8_e5m2'),
]


@pytest.mark.parametrize('dtype,label', NARROW_FLOATS)
@pytest.mark.parametrize('value', [0.0, 1.0, -1.0, 0.5, 2.0])
def test_fill_narrow_float_writes_the_value(dtype, label, value):
    """A reduced-precision fill must write the value the destination type can hold, whichever
    lowering the object representation selects. One byte (fp8) makes every value byte-splat, so
    those always memset; two bytes (fp16, bfloat16) only do for zero."""
    n = 64
    sdfg = make_fill_sdfg("CPU", (n, ),
                          f"0:{n}",
                          gpu=False,
                          name=f"fill_{label}_{str(value).replace('.', '_').replace('-', 'neg')}",
                          dtype=dtype,
                          value=value)
    npdt = dtype.as_numpy_dtype()
    buf = np.full(n, 7, dtype=npdt)
    sdfg(B=buf)
    assert np.array_equal(buf, np.full(n, value, dtype=npdt)), f"{label} fill of {value}"


@pytest.mark.parametrize('dtype,label', NARROW_FLOATS)
@pytest.mark.parametrize('value', [0.0, 1.0])
def test_fill_narrow_float_host_lowering_is_one_call(dtype, label, value):
    """The host fill is a single ``std::fill_n`` whatever the value's object representation is:
    gcc and clang both reduce it to a memset when the representation allows, and dace always
    builds Release. What must hold here is that the fill stays one call and never degrades to an
    element map."""
    n = 64
    tag = str(value).replace('.', '_')
    sdfg = make_fill_sdfg("CPU", (n, ), f"0:{n}", gpu=False, dtype=dtype, value=value, name=f"lowering_{label}_{tag}")
    code = sdfg.generate_code()[0].clean_code
    assert 'std::fill_n' in code
    assert '#pragma omp parallel for' not in code


@pytest.mark.parametrize('dtype,label', NARROW_FLOATS)
def test_fill_narrow_float_gpu_routing_follows_the_byte_pattern(dtype, label):
    """``cudaMemsetAsync`` writes one byte, and no optimizer can widen it, so the GPU choice must
    still be made from the object representation. A one-byte type (fp8) can memset any value; a
    two-byte type (fp16, bfloat16) only zero, and everything else has to reach the kernel."""
    for value in (0.0, 1.0):
        pattern = byte_pattern(value, dtype)
        expected_memset = (dtype.bytes == 1) or value == 0.0
        assert (pattern is not None) == expected_memset, f"{label} {value}"

    sdfg = make_fill_sdfg(None, (64, ), "0:64", gpu=True, dtype=dtype, value=1.0, name=f"gpu_route_{label}")
    node = next(n for n in sdfg.start_state.nodes() if isinstance(n, FillLibraryNode))
    chosen = select_fill_implementation(node, sdfg.start_state)
    assert chosen == ('CUDA' if dtype.bytes == 1 else 'pure'), f"{label} routed to {chosen}"


def make_dynamic_fill_sdfg(shape: Sequence[int],
                           subset: str,
                           gpu: bool = True,
                           dtype: dace.dtypes.typeclass = dace.dtypes.float64,
                           value_dtype: dace.dtypes.typeclass = dace.dtypes.float64,
                           name: str = "dynamic_fill") -> dace.SDFG:
    """Build an SDFG that fills a sub-region with a value supplied through ``_fill_val``."""
    sdfg = dace.SDFG(name)
    arr_name = "gpuB" if gpu else "B"
    storage = dace.dtypes.StorageType.GPU_Global if gpu else dace.dtypes.StorageType.CPU_Heap
    value_storage = dace.dtypes.StorageType.CPU_Heap
    sdfg.add_array(name=arr_name, shape=list(shape), dtype=dtype, storage=storage, transient=False)
    sdfg.add_array(name="V", shape=[1], dtype=value_dtype, storage=value_storage, transient=False)

    state = sdfg.add_state("main")
    out = state.add_access(arr_name)
    val = state.add_access("V")
    libnode = FillLibraryNode(name="fill_libnode", value=0)
    state.add_edge(val, None, libnode, FillLibraryNode.VALUE_CONNECTOR_NAME, dace.memlet.Memlet("V[0]"))
    state.add_edge(libnode, FillLibraryNode.OUTPUT_CONNECTOR_NAME, out, None,
                   dace.memlet.Memlet(f"{arr_name}[{subset}]"))
    return sdfg


def assert_reads_the_dynamic_value(code: str, call: str) -> None:
    """The emitted ``call`` must take its fill value FROM ``V``, not from a baked-in literal.

    Asserted on the call's arguments rather than on ``_fill_val`` appearing somewhere in the file:
    the readable CPU generator inlines tasklet connectors, so the same correct lowering spells the
    operand ``V[V_idx(0)]`` while the legacy one spells it ``_fill_val``. Pinning the connector name
    would pass only under the legacy generator and would not check the operand either way.
    """
    sites = [ln for ln in code.split('\n') if call in ln]
    assert sites, f'expected a {call} call in the generated code'
    assert any(FillLibraryNode.VALUE_CONNECTOR_NAME in ln or re.search(r'\bV\b', ln) for ln in sites), \
        f'{call} must read the dynamic value; got {sites}'


def test_fill_dynamic_value_cpu_routes_to_cpu_for_contiguous_32bit():
    """A dynamic <=32-bit value on a contiguous CPU subset lowers to ``std::fill_n``."""
    sdfg = make_dynamic_fill_sdfg((100, ),
                                  "0:100",
                                  gpu=False,
                                  dtype=dace.float32,
                                  value_dtype=dace.float32,
                                  name="fill_dyn_cpu_f32")
    sdfg.validate()
    node = next(n for n in sdfg.start_state.nodes() if isinstance(n, FillLibraryNode))
    assert select_fill_implementation(node, sdfg.start_state) == 'CPU'
    sdfg.expand_library_nodes()
    code = _generated_code(sdfg)
    assert_reads_the_dynamic_value(code, 'std::fill_n')


def test_fill_dynamic_value_cpu_64bit_routes_to_pure():
    """A dynamic 64-bit value has no single-call runtime memset, so it routes to ``pure``."""
    sdfg = make_dynamic_fill_sdfg((100, ),
                                  "0:100",
                                  gpu=False,
                                  dtype=dace.float64,
                                  value_dtype=dace.float64,
                                  name="fill_dyn_cpu_f64")
    sdfg.validate()
    node = next(n for n in sdfg.start_state.nodes() if isinstance(n, FillLibraryNode))
    assert select_fill_implementation(node, sdfg.start_state) == 'pure'


def test_fill_dynamic_value_cpu_runs_and_writes_value():
    """A dynamic CPU fill actually writes the supplied value into the destination array."""
    sdfg = make_dynamic_fill_sdfg((64, ),
                                  "0:64",
                                  gpu=False,
                                  dtype=dace.float64,
                                  value_dtype=dace.float64,
                                  name="fill_dyn_cpu_run")
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = np.ones((64, ), dtype=np.float64)
    V = np.array([3.5], dtype=np.float64)
    exe(B=B, V=V)
    np.testing.assert_array_equal(B, np.full(64, 3.5, dtype=np.float64))


def test_fill_dynamic_value_rejects_multi_element_subset():
    """The value connector must address exactly one element."""
    sdfg = make_dynamic_fill_sdfg((64, ),
                                  "0:64",
                                  gpu=False,
                                  dtype=dace.float64,
                                  value_dtype=dace.float64,
                                  name="fill_dyn_multi")
    state = sdfg.start_state
    fill = next(n for n in state.nodes() if isinstance(n, FillLibraryNode))
    # Replace the single-element input memlet with a multi-element one.
    for e in list(state.in_edges(fill)):
        if e.dst_conn == FillLibraryNode.VALUE_CONNECTOR_NAME:
            state.remove_edge(e)
            break
    val = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode) and n.data == "V")
    state.add_edge(val, None, fill, FillLibraryNode.VALUE_CONNECTOR_NAME, dace.memlet.Memlet("V[0:2]"))
    with pytest.raises(dace.sdfg.validation.InvalidSDFGError, match="single element"):
        sdfg.validate()


def test_fill_dynamic_value_rejects_dtype_mismatch():
    """The value descriptor dtype must match the output array dtype."""
    sdfg = make_dynamic_fill_sdfg((64, ),
                                  "0:64",
                                  gpu=False,
                                  dtype=dace.float64,
                                  value_dtype=dace.float32,
                                  name="fill_dyn_mismatch")
    with pytest.raises(dace.sdfg.validation.InvalidSDFGError, match="dtype"):
        sdfg.validate()


@pytest.mark.gpu
def test_fill_dynamic_value_gpu_routes_to_cuda_for_32bit():
    """A dynamic <=32-bit value on a contiguous GPU subset lowers to ``<backend>MemsetAsync``."""
    sdfg = make_dynamic_fill_sdfg((100, ),
                                  "0:100",
                                  gpu=True,
                                  dtype=dace.float32,
                                  value_dtype=dace.float32,
                                  name="fill_dyn_gpu_f32")
    sdfg.validate()
    node = next(n for n in sdfg.start_state.nodes() if isinstance(n, FillLibraryNode))
    assert select_fill_implementation(node, sdfg.start_state) == 'CUDA'
    sdfg.expand_library_nodes()
    code = _generated_code(sdfg)
    assert_reads_the_dynamic_value(code, 'MemsetAsync')
