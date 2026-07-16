# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``CopyLibraryNode`` and its pure, CPU, CUDA, cross-storage, register, and shared-memory expansions."""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import dace
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode, cuda2d_pitch_params, select_copy_implementation

import pytest
import numpy as np


@dataclass
class _ArraySpec:
    """Per-side array spec for :func:`_make_copy_sdfg`.

    :param shape: array shape.
    :param storage: storage type.
    :param strides: explicit strides; ``None`` keeps DaCe's packed-C default.
    :param total_size: explicit buffer total size; only consulted when ``strides`` is set
        (defaults to ``prod(shape)``).
    :param transient: transient-array flag.
    :param subset: memlet subset string; defaults to the full per-dim range.
    :param name: SDFG-visible array name; defaults to ``src`` / ``dst`` from position.
    :param dtype: element type; ``None`` defers to the helper's ``dtype`` argument.
    """
    shape: Sequence[int]
    storage: dace.dtypes.StorageType
    strides: Optional[Sequence[int]] = None
    total_size: Optional[int] = None
    transient: bool = False
    subset: Optional[str] = None
    name: Optional[str] = None
    dtype: Optional[dace.dtypes.typeclass] = None


def _make_copy_sdfg(src: _ArraySpec,
                    dst: _ArraySpec,
                    *,
                    implementation: Optional[str] = None,
                    name: str = "copy_sdfg",
                    libnode_name: str = "cp",
                    dtype: dace.dtypes.typeclass = dace.float64) -> Tuple[dace.SDFG, CopyLibraryNode]:
    """One-state SDFG copying ``src`` -> ``dst`` via a single ``CopyLibraryNode``.

    :param src: source-side array spec.
    :param dst: destination-side array spec.
    :param implementation: pinned ``CopyLibraryNode.implementation`` (``None`` keeps ``'Auto'``).
    :param name: SDFG name.
    :param libnode_name: libnode label.
    :param dtype: fallback dtype when a spec leaves ``dtype=None``.
    :returns: ``(sdfg, libnode)``.
    """
    sdfg, src_name, dst_name, src_acc, dst_acc, src_subset, dst_subset = _make_copy_skeleton(src, dst, name, dtype)
    libnode = CopyLibraryNode(name=libnode_name)
    if implementation is not None:
        libnode.implementation = implementation
    state = sdfg.start_state
    state.add_edge(src_acc, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME,
                   dace.memlet.Memlet(f"{src_name}[{src_subset}]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst_acc, None,
                   dace.memlet.Memlet(f"{dst_name}[{dst_subset}]"))
    return sdfg, libnode


def _make_copy_skeleton(src: _ArraySpec, dst: _ArraySpec, name: str, dtype: dace.dtypes.typeclass):
    """Shared scaffolding for :func:`_make_copy_sdfg` and :func:`_make_legacy_copy_sdfg`: builds the arrays + AccessNodes and returns the subsets."""
    sdfg = dace.SDFG(name)
    src_name = src.name or "src"
    dst_name = dst.name or "dst"
    for arr_name, spec in ((src_name, src), (dst_name, dst)):
        kwargs = {"transient": spec.transient}
        if spec.strides is not None:
            # Sympify each stride so string entries (``"src_stride"``) become
            # SDFG symbols. ``Array.validate`` rejects raw strings, and
            # ``add_array`` only sympifies the shape, not the strides.
            kwargs["strides"] = [dace.symbolic.pystr_to_symbolic(s) for s in spec.strides]
            kwargs["total_size"] = spec.total_size if spec.total_size is not None else int(np.prod(spec.shape))
        sdfg.add_array(arr_name, spec.shape, spec.dtype or dtype, storage=spec.storage, **kwargs)
    state = sdfg.add_state("main")
    src_acc = state.add_access(src_name)
    dst_acc = state.add_access(dst_name)
    src_subset = src.subset if src.subset is not None else ", ".join(f"0:{s}" for s in src.shape)
    dst_subset = dst.subset if dst.subset is not None else ", ".join(f"0:{s}" for s in dst.shape)
    return sdfg, src_name, dst_name, src_acc, dst_acc, src_subset, dst_subset


def _make_legacy_copy_sdfg(src: _ArraySpec,
                           dst: _ArraySpec,
                           *,
                           name: str = "copy_legacy",
                           dtype: dace.dtypes.typeclass = dace.float64) -> dace.SDFG:
    """One-state SDFG copying ``src`` -> ``dst`` via a canonical direct AN -> AN edge.

    Legacy DaCe memlet convention (``data=dst``, ``subset``=dst write region,
    ``other_subset``=src read region) -- the standard copy lowering's output and
    the baseline for comparing against the :class:`CopyLibraryNode` path.
    """
    sdfg, src_name, dst_name, src_acc, dst_acc, src_subset, dst_subset = _make_copy_skeleton(src, dst, name, dtype)
    sdfg.start_state.add_edge(src_acc, None, dst_acc, None,
                              dace.memlet.Memlet(data=dst_name, subset=dst_subset, other_subset=src_subset))
    return sdfg


def _fortran_strides(shape):
    """Column-major Fortran-packed strides, via the same helper ``Array.is_packed_fortran_strides`` checks against."""
    return dace.data.Array(dace.float64, shape=shape)._get_packed_fortran_strides()


def _compile_no_copynd(sdfg: dace.SDFG):
    """Assert the generated C++ contains no ``dace::CopyND`` template, then compile.

    The libnodes displace the runtime CopyND fallback entirely. The only intentional
    ``CopyND`` user is ``ExpandSharedMemoryCollective``; tests exercising that expansion
    inspect tasklet bodies directly and don't run codegen, so this assertion is safe here.
    """
    for obj in sdfg.generate_code():
        assert 'CopyND<' not in obj.code, f"unexpected dace::CopyND in generated code object {obj.name}"
    return sdfg.compile()


def test_copy_pure_cpu():
    """Pure (mapped tasklet) expansion on CPU_Heap -> CPU_Heap."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[200], storage=dace.dtypes.StorageType.CPU_Heap, subset="150:200", name="A"),
        _ArraySpec(shape=[200], storage=dace.dtypes.StorageType.CPU_Heap, subset="50:100", name="B"),
        implementation="MappedTasklet",
        name="copy_pure_cpu",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)

    A = np.ones(200, dtype=np.float64)
    B = np.zeros(200, dtype=np.float64)
    exe(A=A, B=B)

    np.testing.assert_array_equal(B[50:100], A[150:200])
    assert np.all(B[:50] == 0)
    assert np.all(B[100:] == 0)


def test_copy_cpu_memcpy():
    """CPU expansion (std::memcpy) on CPU_Heap -> CPU_Heap."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[200], storage=dace.dtypes.StorageType.CPU_Heap, subset="150:200", name="A"),
        _ArraySpec(shape=[200], storage=dace.dtypes.StorageType.CPU_Heap, subset="50:100", name="B"),
        implementation="MemcpyCPU",
        name="copy_cpu_memcpy",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)

    A = np.arange(200, dtype=np.float64)
    B = np.zeros(200, dtype=np.float64)
    exe(A=A, B=B)

    np.testing.assert_array_equal(B[50:100], A[150:200])


def test_copy_fortran_packed_same_rank():
    """Same-rank Fortran-packed (column-major) full copy is contiguous and same-layout, so the
    Auto path routes it to the serial ``std::memcpy`` (``MemcpyCPU``); a flat byte copy is exact
    for two Fortran-packed operands."""
    sdfg, libnode = _make_copy_sdfg(
        _ArraySpec(shape=(4, 5, 6), storage=dace.dtypes.StorageType.CPU_Heap, strides=(1, 4, 20)),
        _ArraySpec(shape=(4, 5, 6), storage=dace.dtypes.StorageType.CPU_Heap, strides=(1, 4, 20)),
        name="copy_fortran_packed_same_rank",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    assert libnode.implementation == 'MemcpyCPU'

    src_data = np.arange(120, dtype=np.float64).reshape(4, 5, 6, order='F').copy(order='F')
    dst_data = np.zeros((4, 5, 6), dtype=np.float64, order='F')
    sdfg(src=src_data, dst=dst_data)
    assert np.array_equal(dst_data, src_data)


def test_copy_fortran_packed_strided_slice():
    """Same-rank Fortran-packed strided-slice copy via the Auto-routed MappedTasklet."""
    sdfg, libnode = _make_copy_sdfg(
        _ArraySpec(shape=(8, 10, 12),
                   storage=dace.dtypes.StorageType.CPU_Heap,
                   strides=(1, 8, 80),
                   subset="2:6, 3:7, 4:8"),
        _ArraySpec(shape=(8, 10, 12),
                   storage=dace.dtypes.StorageType.CPU_Heap,
                   strides=(1, 8, 80),
                   subset="2:6, 3:7, 4:8"),
        name="copy_fortran_packed_strided_slice",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    assert libnode.implementation == 'MappedTasklet'

    src_data = np.arange(960, dtype=np.float64).reshape(8, 10, 12, order='F').copy(order='F')
    dst_data = np.zeros((8, 10, 12), dtype=np.float64, order='F')
    sdfg(src=src_data, dst=dst_data)
    assert np.array_equal(dst_data[2:6, 3:7, 4:8], src_data[2:6, 3:7, 4:8])
    untouched = dst_data.copy()
    untouched[2:6, 3:7, 4:8] = 0
    assert np.all(untouched == 0)


def test_copy_mixed_c_fortran_via_mapped_tasklet():
    """Mixed C-packed -> Fortran-packed same-rank copy lowers via MappedTasklet."""
    sdfg, libnode = _make_copy_sdfg(
        _ArraySpec(shape=(6, 7), storage=dace.dtypes.StorageType.CPU_Heap, strides=(7, 1)),
        _ArraySpec(shape=(6, 7), storage=dace.dtypes.StorageType.CPU_Heap, strides=(1, 6)),
        name="copy_mixed_c_fortran",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    assert libnode.implementation == 'MappedTasklet'

    src_data = np.arange(42, dtype=np.float64).reshape(6, 7).copy(order='C')
    dst_data = np.zeros((6, 7), dtype=np.float64, order='F')
    sdfg(src=src_data, dst=dst_data)
    assert np.array_equal(dst_data, src_data)


def test_copy_rank_mismatch_mixed_layouts_raises():
    """Rank-mismatch with mixed C/F packed layouts is rejected (1-D walker has no shared layout)."""
    # src is C-packed (3, 8) -- strides (8, 1); dst is Fortran-packed (2, 3, 4)
    # -- strides (1, 2, 6). Same volume = 24.
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=(3, 8), storage=dace.dtypes.StorageType.CPU_Heap),
        _ArraySpec(shape=(2, 3, 4), storage=dace.dtypes.StorageType.CPU_Heap, strides=(1, 2, 6)),
        name="copy_rank_mismatch_mixed_raises",
    )
    sdfg.validate()
    with pytest.raises(ValueError, match="same major order"):
        sdfg.expand_library_nodes()


def test_copy_rank_mismatch_padded_src_raises():
    """Rank-mismatch with padded (neither C- nor F-packed) strides is rejected."""
    # src padded (row stride 8 instead of 6), dst flat (120,).
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=(4, 5, 6),
                   storage=dace.dtypes.StorageType.CPU_Heap,
                   strides=(5 * 8, 8, 1),
                   total_size=4 * 5 * 8),
        _ArraySpec(shape=(120, ), storage=dace.dtypes.StorageType.CPU_Heap),
        name="copy_rank_mismatch_padded_raises",
    )
    sdfg.validate()
    with pytest.raises(ValueError, match="same major order"):
        sdfg.expand_library_nodes()


def test_copy_rank_mismatch_strided_subset_raises():
    """Rank-mismatch with a non-contiguous src subset is rejected (1-D walker requires contiguous data)."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=(8, 10), storage=dace.dtypes.StorageType.CPU_Heap, subset="0:8, 2:6"),
        _ArraySpec(shape=(32, ), storage=dace.dtypes.StorageType.CPU_Heap),
        name="copy_rank_mismatch_strided_subset",
    )
    sdfg.validate()
    with pytest.raises(ValueError, match="contiguous subsets"):
        sdfg.expand_library_nodes()


def test_copy_rank_mismatch_strided_dst_subset_raises():
    """Symmetric to the src-side variant: non-contiguous subset on the dst side is rejected."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=(32, ), storage=dace.dtypes.StorageType.CPU_Heap),
        _ArraySpec(shape=(8, 10), storage=dace.dtypes.StorageType.CPU_Heap, subset="0:8, 2:6"),
        name="copy_rank_mismatch_strided_dst_subset",
    )
    sdfg.validate()
    with pytest.raises(ValueError, match="contiguous subsets"):
        sdfg.expand_library_nodes()


def test_copy_same_subset_different_array_shapes():
    """A ``0:N`` slice copies between arrays of different total shape as long as the per-dim subset sizes match."""
    N = 10
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=(200, ), storage=dace.dtypes.StorageType.CPU_Heap, subset=f"0:{N}", name="A"),
        _ArraySpec(shape=(500, ), storage=dace.dtypes.StorageType.CPU_Heap, subset=f"0:{N}", name="B"),
        name="copy_same_subset_diff_shape",
    )
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)
    A = np.arange(200, dtype=np.float64)
    B = np.zeros(500, dtype=np.float64)
    exe(A=A, B=B)
    np.testing.assert_array_equal(B[:N], A[:N])


def test_copy_1d_slice_from_2d_source():
    """A row-slice ``[i, 0:N]`` of a 2D array copies into a 1D array (singleton dims collapse to same rank)."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=(5, 10), storage=dace.dtypes.StorageType.CPU_Heap, subset="2, 0:10", name="A"),
        _ArraySpec(shape=(10, ), storage=dace.dtypes.StorageType.CPU_Heap, subset="0:10", name="B"),
        name="copy_1d_slice_from_2d",
    )
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)
    A = np.arange(50, dtype=np.float64).reshape(5, 10).copy()
    B = np.zeros(10, dtype=np.float64)
    exe(A=A, B=B)
    np.testing.assert_array_equal(B, A[2])


def test_copy_transpose_pattern_rejected():
    """Same-rank copy with per-dim shapes swapped (transpose) is rejected upfront."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=(3, 4), storage=dace.dtypes.StorageType.CPU_Heap),
        _ArraySpec(shape=(4, 3), storage=dace.dtypes.StorageType.CPU_Heap),
        name="copy_transpose_pattern",
    )
    sdfg.validate()
    with pytest.raises(ValueError, match="matching per-dim shapes"):
        sdfg.expand_library_nodes()


def test_copy_4d_to_1d_flatten_c_packed():
    """4D -> 1D flatten via MappedTasklet rank-mismatch (extends beyond the 3D->1D coverage)."""
    sdfg, libnode = _make_copy_sdfg(
        _ArraySpec(shape=(2, 3, 4, 5), storage=dace.dtypes.StorageType.CPU_Heap),
        _ArraySpec(shape=(120, ), storage=dace.dtypes.StorageType.CPU_Heap),
        name="copy_4d_to_1d_c",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    assert libnode.implementation == 'MappedTasklet'

    src = np.arange(120, dtype=np.float64).reshape(2, 3, 4, 5).copy(order='C')
    dst = np.zeros(120, dtype=np.float64)
    sdfg(src=src, dst=dst)
    assert np.array_equal(dst, src.ravel(order='C'))


def test_copy_1d_to_4d_inflate_c_packed():
    """1D -> 4D inflate (higher-rank destination); inverse direction of the flatten path."""
    sdfg, libnode = _make_copy_sdfg(
        _ArraySpec(shape=(24, ), storage=dace.dtypes.StorageType.CPU_Heap),
        _ArraySpec(shape=(2, 3, 4), storage=dace.dtypes.StorageType.CPU_Heap),
        name="copy_1d_to_3d_c",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    assert libnode.implementation == 'MappedTasklet'

    src = np.arange(24, dtype=np.float64)
    dst = np.zeros((2, 3, 4), dtype=np.float64)
    sdfg(src=src, dst=dst)
    assert np.array_equal(dst, src.reshape(2, 3, 4))


def test_copy_3d_to_2d_collapse_first_two_dims():
    """3D -> 2D collapse of the first two dims (C-order) via MappedTasklet rank-mismatch."""
    sdfg, libnode = _make_copy_sdfg(
        _ArraySpec(shape=(2, 3, 4), storage=dace.dtypes.StorageType.CPU_Heap),
        _ArraySpec(shape=(6, 4), storage=dace.dtypes.StorageType.CPU_Heap),
        name="copy_3d_to_2d_collapse",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    assert libnode.implementation == 'MappedTasklet'

    src = np.arange(24, dtype=np.float64).reshape(2, 3, 4).copy(order='C')
    dst = np.zeros((6, 4), dtype=np.float64)
    sdfg(src=src, dst=dst)
    assert np.array_equal(dst, src.reshape(6, 4))


def test_copy_4d_to_2d_collapse_pair_dims_fortran():
    """4D -> 2D Fortran-packed reshape: walk both sides in column-major order."""
    sdfg, libnode = _make_copy_sdfg(
        _ArraySpec(shape=(2, 3, 4, 5), storage=dace.dtypes.StorageType.CPU_Heap, strides=(1, 2, 6, 24)),
        _ArraySpec(shape=(6, 20), storage=dace.dtypes.StorageType.CPU_Heap, strides=(1, 6)),
        name="copy_4d_to_2d_f",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    assert libnode.implementation == 'MappedTasklet'

    src = np.arange(120, dtype=np.float64).reshape(2, 3, 4, 5, order='F').copy(order='F')
    dst = np.zeros((6, 20), dtype=np.float64, order='F')
    sdfg(src=src, dst=dst)
    assert np.array_equal(dst, src.reshape(6, 20, order='F'))


def test_copy_strided_step_2_cpu_same_rank():
    """Same-rank 1D copy with subset step=2 (every other element)."""
    sdfg, libnode = _make_copy_sdfg(
        _ArraySpec(shape=(10, ), storage=dace.dtypes.StorageType.CPU_Heap, subset="0:10:2"),
        _ArraySpec(shape=(5, ), storage=dace.dtypes.StorageType.CPU_Heap, subset="0:5"),
        name="copy_step2_cpu",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    assert libnode.implementation == 'MappedTasklet'

    src = np.arange(10, dtype=np.float64)
    dst = np.zeros(5, dtype=np.float64)
    sdfg(src=src, dst=dst)
    assert np.array_equal(dst, src[0:10:2])


@pytest.mark.gpu
def test_copy_pure_gpu():
    """Pure (mapped tasklet) expansion on GPU_Global -> GPU_Global."""
    import cupy as cp

    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[200], storage=dace.dtypes.StorageType.GPU_Global, subset="150:200", name="gpu_A"),
        _ArraySpec(shape=[200], storage=dace.dtypes.StorageType.GPU_Global, subset="50:100", name="gpu_B"),
        implementation="MappedTasklet",
        name="copy_pure_gpu",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)

    A = cp.ones(200, dtype=cp.float64)
    B = cp.zeros(200, dtype=cp.float64)
    exe(gpu_A=A, gpu_B=B)

    cp.testing.assert_array_equal(B[50:100], A[150:200])
    assert cp.all(B[:50] == 0)
    assert cp.all(B[100:] == 0)


@pytest.mark.gpu
def test_copy_cuda_d2d():
    """CUDA expansion (cudaMemcpyDeviceToDevice) on GPU_Global -> GPU_Global."""
    import cupy as cp

    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[200], storage=dace.dtypes.StorageType.GPU_Global, subset="150:200", name="gpu_A"),
        _ArraySpec(shape=[200], storage=dace.dtypes.StorageType.GPU_Global, subset="50:100", name="gpu_B"),
        implementation="MemcpyCUDA1D",
        name="copy_cuda_d2d",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)

    A = cp.arange(200, dtype=cp.float64)
    B = cp.zeros(200, dtype=cp.float64)
    exe(gpu_A=A, gpu_B=B)

    cp.testing.assert_array_equal(B[50:100], A[150:200])


@pytest.mark.gpu
def test_copy_cuda_1d_single_element():
    """CUDA expansion (cudaMemcpyDeviceToDevice) on GPU_Global -> GPU_Global for a single element."""
    import cupy as cp

    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[200],
                   strides=["src_stride"],
                   storage=dace.dtypes.StorageType.GPU_Global,
                   subset="130",
                   name="gpu_A"),
        _ArraySpec(shape=[200],
                   strides=["dst_stride"],
                   storage=dace.dtypes.StorageType.GPU_Global,
                   subset="15",
                   name="gpu_B"),
        implementation="MemcpyCUDA1D",
        name="copy_cuda_1d_single_element",
    )
    # ``add_array`` registered the stride symbols at default int.
    sdfg.validate()

    sdfg.expand_library_nodes()
    sdfg.validate()

    exe = _compile_no_copynd(sdfg)

    A = cp.arange(200, dtype=cp.float64)
    B = cp.zeros(200, dtype=cp.float64)

    ref = B.copy()
    ref[15] = A[130]

    exe(gpu_A=A, gpu_B=B, src_stride=1, dst_stride=1)

    cp.testing.assert_array_equal(ref, B)


def test_copy_pure_host_to_device_rejected():
    """Pure expansion must reject CPU_Heap -> GPU_Global (needs cudaMemcpy)."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[128], storage=dace.dtypes.StorageType.CPU_Heap),
        _ArraySpec(shape=[128], storage=dace.dtypes.StorageType.GPU_Global),
        implementation="MappedTasklet",
        name="copy_pure_h2d_reject",
    )
    sdfg.validate()
    with pytest.raises(Exception, match="CPU/GPU boundary"):
        sdfg.expand_library_nodes()


def test_copy_pure_device_to_host_rejected():
    """Pure expansion must reject GPU_Global -> CPU_Heap (needs cudaMemcpy)."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[128], storage=dace.dtypes.StorageType.GPU_Global),
        _ArraySpec(shape=[128], storage=dace.dtypes.StorageType.CPU_Heap),
        implementation="MappedTasklet",
        name="copy_pure_d2h_reject",
    )
    sdfg.validate()
    with pytest.raises(Exception, match="CPU/GPU boundary"):
        sdfg.expand_library_nodes()


@pytest.mark.gpu
def test_copy_cuda_host_to_device():
    """CUDAHostToDevice expansion for CPU_Heap -> GPU_Global."""
    import cupy as cp

    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[128], storage=dace.dtypes.StorageType.CPU_Heap),
        _ArraySpec(shape=[128], storage=dace.dtypes.StorageType.GPU_Global),
        implementation="MemcpyCUDA1D",
        name="copy_cuda_h2d",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)

    src = np.arange(128, dtype=np.float64)
    dst = cp.zeros(128, dtype=cp.float64)
    exe(src=src, dst=dst)

    cp.testing.assert_array_equal(dst, cp.asarray(src))


@pytest.mark.gpu
def test_copy_cuda_device_to_host():
    """CUDADeviceToHost expansion for GPU_Global -> CPU_Heap."""
    import cupy as cp

    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[128], storage=dace.dtypes.StorageType.GPU_Global),
        _ArraySpec(shape=[128], storage=dace.dtypes.StorageType.CPU_Heap),
        implementation="MemcpyCUDA1D",
        name="copy_cuda_d2h",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)

    src = cp.arange(128, dtype=cp.float64)
    dst = np.zeros(128, dtype=np.float64)
    exe(src=src, dst=dst)

    np.testing.assert_array_equal(dst, cp.asnumpy(src))


@pytest.mark.gpu
def test_copy_cuda_4d_strided_host_to_device():
    """A 4D strided CPU_Heap -> GPU_Global slice copy via ``MemcpyCUDANDStrided`` produces correct output."""
    import cupy as cp

    # Slice into a larger array so the outer dims are strided, exercising the
    # per-row strided CUDA path rather than a single contiguous memcpy.
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=(7, 8, 9, 10),
                   storage=dace.dtypes.StorageType.CPU_Heap,
                   subset="1:6, 1:7, 1:8, 1:9",
                   name="A_full"),
        _ArraySpec(shape=(5, 6, 7, 8), storage=dace.dtypes.StorageType.GPU_Global, name="B_dst"),
        implementation="MemcpyCUDANDStrided",
        name="copy_cuda_4d_strided_h2d",
        libnode_name="cp_4d_strided",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)

    # ``reshape`` returns a numpy view; DaCe rejects views by default
    # (``compiler.allow_view_arguments``). Build directly as a fresh array.
    A = np.empty((7, 8, 9, 10), dtype=np.float64)
    A[:] = np.arange(7 * 8 * 9 * 10).reshape(7, 8, 9, 10)
    B = cp.zeros((5, 6, 7, 8), dtype=cp.float64)
    exe(A_full=A, B_dst=B)

    expected = A[1:6, 1:7, 1:8, 1:9]
    cp.testing.assert_array_equal(B, cp.asarray(expected))


def test_copy_fortran_packed_cpu_default_pure():
    """A same-side CPU copy of a Fortran-packed array expands and produces correct output."""
    shape = (4, 5, 6)
    f_strides = _fortran_strides(shape)
    total = int(np.prod(shape))

    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=shape, storage=dace.dtypes.StorageType.CPU_Heap, strides=f_strides, total_size=total),
        _ArraySpec(shape=shape, storage=dace.dtypes.StorageType.CPU_Heap, strides=f_strides, total_size=total),
        name="copy_fortran_cpu",
        libnode_name="cp_fortran_cpu",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)

    A = np.arange(total, dtype=np.float64).reshape(shape, order='F').copy(order='F')
    B = np.zeros(shape, dtype=np.float64, order='F')
    exe(src=A, dst=B)
    np.testing.assert_array_equal(B, A)


@pytest.mark.gpu
def test_copy_fortran_packed_gpu_falls_back_to_pure():
    """A same-side GPU copy of a Fortran-packed array expands and produces correct output."""
    import cupy as cp

    shape = (4, 5, 6)
    f_strides = _fortran_strides(shape)
    total = int(np.prod(shape))

    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=shape, storage=dace.dtypes.StorageType.GPU_Global, strides=f_strides, total_size=total),
        _ArraySpec(shape=shape, storage=dace.dtypes.StorageType.GPU_Global, strides=f_strides, total_size=total),
        implementation="MemcpyCUDA1D",
        name="copy_fortran_gpu",
        libnode_name="cp_fortran_gpu",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)

    host = np.arange(total, dtype=np.float64).reshape(shape, order='F').copy(order='F')
    A = cp.asfortranarray(cp.asarray(host))
    B = cp.asfortranarray(cp.zeros(shape, dtype=cp.float64))
    exe(src=A, dst=B)
    cp.testing.assert_array_equal(B, A)


@pytest.mark.gpu
def test_copy_fortran_packed_cpu_to_gpu_uses_outermost_chunk():
    """A cross-CPU/GPU copy of a Fortran-packed array expands and produces correct output."""
    import cupy as cp

    shape = (4, 5, 6)
    f_strides = _fortran_strides(shape)
    total = int(np.prod(shape))

    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=shape, storage=dace.dtypes.StorageType.CPU_Heap, strides=f_strides, total_size=total),
        _ArraySpec(shape=shape, storage=dace.dtypes.StorageType.GPU_Global, strides=f_strides, total_size=total),
        implementation="MemcpyCUDA1D",
        name="copy_fortran_h2d",
        libnode_name="cp_fortran_h2d",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)

    host = np.arange(total, dtype=np.float64).reshape(shape, order='F').copy(order='F')
    dev = cp.asfortranarray(cp.zeros(shape, dtype=cp.float64))
    exe(src=host, dst=dev)
    cp.testing.assert_array_equal(dev, cp.asarray(host))


def test_copy_no_common_stride1_axis_raises():
    """Cross-CPU/GPU copy with no shared stride-1 axis is rejected."""
    # src C-packed (stride-1 innermost), dst Fortran-packed (stride-1
    # outermost): after the partial slice the two have no shared stride-1 axis.
    shape = (4, 5, 6)
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=shape,
                   storage=dace.dtypes.StorageType.CPU_Heap,
                   strides=(30, 6, 1),
                   total_size=120,
                   subset="0:4, 0:4, 0:5"),
        _ArraySpec(shape=shape,
                   storage=dace.dtypes.StorageType.GPU_Global,
                   strides=(1, 4, 20),
                   total_size=120,
                   subset="0:4, 0:4, 0:5"),
        implementation="Auto",  # exercise the refine-time strided-pattern check
        name="copy_no_common_stride1",
        libnode_name="cp_no_common",
    )
    sdfg.validate()
    with pytest.raises(ValueError, match="cross-CPU/GPU"):
        sdfg.expand_library_nodes()


def test_copy_node_storage_from_edges():
    """``src_storage`` / ``dst_storage`` resolve live from the node's ``_in`` / ``_out`` edges."""
    sdfg, node = _make_copy_sdfg(
        _ArraySpec(shape=[10], storage=dace.dtypes.StorageType.CPU_Heap, name="A"),
        _ArraySpec(shape=[10], storage=dace.dtypes.StorageType.GPU_Global, name="B"),
        name="storage_from_edges",
        libnode_name="edges_to_storage",
    )
    state = sdfg.start_state
    assert node.src_storage(state) == dace.dtypes.StorageType.CPU_Heap
    assert node.dst_storage(state) == dace.dtypes.StorageType.GPU_Global


def test_copy_node_storage_defaults_when_unattached():
    """Without edges, the storage methods fall back to ``StorageType.Default``."""
    sdfg = dace.SDFG("storage_unattached")
    state = sdfg.add_state("main")
    node = CopyLibraryNode(name="unattached")
    state.add_node(node)

    assert node.src_storage(state) == dace.dtypes.StorageType.Default
    assert node.dst_storage(state) == dace.dtypes.StorageType.Default


def test_copy_cross_storage_validation_rejects_without_flag():
    """The ``MemcpyCPU`` expansion rejects a CPU<->GPU storage mismatch at expansion time."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[64], storage=dace.dtypes.StorageType.CPU_Heap),
        _ArraySpec(shape=[64], storage=dace.dtypes.StorageType.GPU_Global),
        implementation="MemcpyCPU",
        name="copy_cross_reject",
    )
    sdfg.validate()  # the SDFG is valid; only the expansion rejects the mismatch
    with pytest.raises(Exception):
        sdfg.expand_library_nodes()


def test_copy_dtype_mismatch_rejected():
    """CopyLibraryNode must reject mismatched dtypes."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[10], storage=dace.dtypes.StorageType.CPU_Heap, dtype=dace.float32, name="A"),
        _ArraySpec(shape=[10], storage=dace.dtypes.StorageType.CPU_Heap, dtype=dace.float64, name="B"),
        name="dtype_mismatch",
        libnode_name="cp_bad",
    )
    with pytest.raises(ValueError, match="data types must match"):
        sdfg.expand_library_nodes()


def test_cpu_memcpy_rejects_non_contiguous_subset():
    """CPU (memcpy) expansion must reject a non-contiguous 2D slice."""
    # Partial dim 0 over a smaller dim 1 makes the source slice non-contiguous.
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[10, 20], storage=dace.dtypes.StorageType.CPU_Heap, subset="2:6, 0:10", name="A"),
        _ArraySpec(shape=[4, 20], storage=dace.dtypes.StorageType.CPU_Heap, subset="0:4, 0:10", name="B"),
        implementation="MemcpyCPU",
        name="cpu_noncontig",
        libnode_name="cp_nc",
    )
    with pytest.raises(Exception, match="contiguous"):
        sdfg.expand_library_nodes()


def test_strided_expansions_accept_non_contiguous():
    """The ``MappedTasklet`` expansion accepts a non-contiguous subset."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[10, 20], storage=dace.dtypes.StorageType.CPU_Heap, subset="2:6, 0:10", name="A"),
        _ArraySpec(shape=[4, 20], storage=dace.dtypes.StorageType.CPU_Heap, subset="0:4, 0:10", name="B"),
        implementation="MappedTasklet",
        name="noncontig_MappedTasklet",
    )
    sdfg.expand_library_nodes()


# A (1, N) array whose unit leading dim carries a padded stride (here 64) is a
# non-packed descriptor, yet the accessed row ``[0, 0:N]`` is one physical run of
# N contiguous elements: the pad sits on an extent-1 axis that is never stepped.
# ``is_contiguous_subset`` therefore reports True (the 1D-slice special case), and
# the copy safely lowers to a single flat block. A fresh contiguous (1, N) array
# backs it with no view (``total_size`` only needs to cover the accessed run).
_PADDED_N = 60
_PADDED_STRIDE = 64
_PADDED_ROWS = 3


def _padded_unit_spec(storage, name):
    """``_ArraySpec`` for a (1, ``_PADDED_N``) array with a padded (non-packed) leading stride."""
    return _ArraySpec(shape=(1, _PADDED_N),
                      storage=storage,
                      strides=(_PADDED_STRIDE, 1),
                      total_size=_PADDED_N,
                      name=name)


def _padded_multirow_spec(storage, name):
    """``_ArraySpec`` for a (``_PADDED_ROWS``, ``_PADDED_N``) array whose rows carry the padded stride.

    Unlike the unit-row spec, the leading dim has extent > 1, so the inter-row pitch gap
    (``_PADDED_STRIDE - _PADDED_N`` unused elements per row) is actually stepped over: the full
    ``[0:ROWS, 0:N]`` copy is genuinely non-contiguous.
    """
    return _ArraySpec(shape=(_PADDED_ROWS, _PADDED_N),
                      storage=storage,
                      strides=(_PADDED_STRIDE, 1),
                      total_size=_PADDED_STRIDE * _PADDED_ROWS,
                      name=name)


def test_copy_padded_unit_dim_same_storage_cpu():
    """Same-storage CPU copy of a padded (1, N) array: contiguous run, CPU<->CPU map fallback, exact result."""
    sdfg, node = _make_copy_sdfg(
        _padded_unit_spec(dace.dtypes.StorageType.CPU_Heap, "A"),
        _padded_unit_spec(dace.dtypes.StorageType.CPU_Heap, "B"),
        name="copy_padded_unit_cpu",
        libnode_name="cp_padded_cpu",
    )
    state = sdfg.start_state
    _, inp, in_sub, _, out, out_sub = node.validate(state.sdfg, state, allow_cross_storage=True)
    # The accessed row is a single contiguous run (1D-slice special case).
    assert in_sub.is_contiguous_subset(inp)
    assert out_sub.is_contiguous_subset(out)
    # CPU<->CPU multi-element copies never route to a memcpy libnode; they fall back to a map.
    assert select_copy_implementation(node, state) == "MappedTasklet"

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)

    A = np.zeros((1, _PADDED_N), dtype=np.float64)  # fresh + contiguous: A.base is None, so no view rejection
    B = np.zeros((1, _PADDED_N), dtype=np.float64)
    A[0, :] = np.arange(1, _PADDED_N + 1, dtype=np.float64)
    exe(A=A, B=B)
    np.testing.assert_array_equal(B, A)


def test_copy_padded_unit_dim_cross_storage_selection():
    """Cross CPU/GPU copy of a padded (1, N) array is a single contiguous row: flat ``cudaMemcpy``, not pitched.

    With only one row the pitch gap is never crossed, so the row is one contiguous run on both sides and
    ``MemcpyCUDA1D`` is exact (a pitched ``cudaMemcpy2D`` would be equivalent but needlessly 2D)."""
    for src_storage, dst_storage in (
        (dace.dtypes.StorageType.CPU_Heap, dace.dtypes.StorageType.GPU_Global),
        (dace.dtypes.StorageType.GPU_Global, dace.dtypes.StorageType.CPU_Heap),
    ):
        sdfg, node = _make_copy_sdfg(
            _padded_unit_spec(src_storage, "A"),
            _padded_unit_spec(dst_storage, "B"),
            name="copy_padded_unit_cross",
            libnode_name="cp_padded_cross",
        )
        state = sdfg.start_state
        _, inp, in_sub, _, out, out_sub = node.validate(state.sdfg, state, allow_cross_storage=True)
        assert in_sub.is_contiguous_subset(inp)
        assert out_sub.is_contiguous_subset(out)
        assert select_copy_implementation(node, state) == "MemcpyCUDA1D"


def test_copy_padded_multirow_cross_storage_uses_pitched():
    """Cross CPU/GPU copy of a padded multi-row (ROWS, N) array must route to the pitched ``cudaMemcpy2D``.

    With more than one row the inter-row pitch gap is stepped over, so the region is genuinely
    non-contiguous and a flat ``MemcpyCUDA1D`` would drag the padding bytes between rows into the copy.
    This pins the dangerous direction: were ``is_contiguous_subset`` to ever wrongly report this subset
    contiguous, ``_refine_cuda_impl_for_subsets`` would keep the flat copy and silently corrupt the data --
    this test would catch it before the numerical damage.
    """
    for src_storage, dst_storage in (
        (dace.dtypes.StorageType.CPU_Heap, dace.dtypes.StorageType.GPU_Global),
        (dace.dtypes.StorageType.GPU_Global, dace.dtypes.StorageType.CPU_Heap),
    ):
        sdfg, node = _make_copy_sdfg(
            _padded_multirow_spec(src_storage, "A"),
            _padded_multirow_spec(dst_storage, "B"),
            name="copy_padded_multirow_cross",
            libnode_name="cp_padded_multirow",
        )
        state = sdfg.start_state
        _, inp, in_sub, _, out, out_sub = node.validate(state.sdfg, state, allow_cross_storage=True)
        assert not in_sub.is_contiguous_subset(inp)
        assert not out_sub.is_contiguous_subset(out)
        assert select_copy_implementation(node, state) == "MemcpyCUDA2D"


@pytest.mark.gpu
def test_copy_padded_unit_dim_cross_storage_gpu():
    """Cross CPU->GPU copy of a padded (1, N) array expands to a pitched copy and is numerically exact."""
    import cupy as cp

    sdfg, _ = _make_copy_sdfg(
        _padded_unit_spec(dace.dtypes.StorageType.CPU_Heap, "A"),
        _padded_unit_spec(dace.dtypes.StorageType.GPU_Global, "B"),
        name="copy_padded_unit_h2d",
        libnode_name="cp_padded_h2d",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)

    A = np.zeros((1, _PADDED_N), dtype=np.float64)
    A[0, :] = np.arange(1, _PADDED_N + 1, dtype=np.float64)
    B = cp.zeros((1, _PADDED_N), dtype=cp.float64)
    exe(A=A, B=B)
    cp.testing.assert_array_equal(B, cp.asarray(A))


def test_register_copy_expands_with_register_storage():
    """A Register -> Register ``MappedTasklet`` copy expands to a Sequential (thread-level) map."""
    reg = dace.dtypes.StorageType.Register
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[8], storage=reg, transient=True, name="R_in"),
        _ArraySpec(shape=[8], storage=reg, transient=True, name="R_out"),
        implementation="MappedTasklet",
        name="reg_copy_ok",
        libnode_name="regcpy",
    )
    sdfg.expand_library_nodes()

    found_sequential = False
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.MapEntry):
            if n.schedule == dace.dtypes.ScheduleType.Sequential:
                found_sequential = True
                break
    assert found_sequential, "RegisterCopy expansion should contain a Sequential map."


def test_direct_assignment_cpu_same_storage():
    """``Tasklet`` impl on CPU_Heap -> CPU_Heap (single element) compiles and runs."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[4], storage=dace.dtypes.StorageType.CPU_Heap, subset="2:3", name="A"),
        _ArraySpec(shape=[4], storage=dace.dtypes.StorageType.CPU_Heap, subset="1:2", name="B"),
        implementation="Tasklet",
        name="direct_assign_cpu",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)

    A = np.arange(4, dtype=np.float64)
    B = np.zeros(4, dtype=np.float64)
    exe(A=A, B=B)
    assert B[1] == A[2]


def test_direct_assignment_register_to_register():
    """A size-1 Register -> Register ``Tasklet`` copy expands to a Python tasklet with no map."""
    reg = dace.dtypes.StorageType.Register
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[1], storage=reg, transient=True, subset="0", name="R_in"),
        _ArraySpec(shape=[1], storage=reg, transient=True, subset="0", name="R_out"),
        implementation="Tasklet",
        name="direct_assign_reg",
        libnode_name="da",
    )
    sdfg.expand_library_nodes()

    found_tasklet = False
    found_map = False
    for n, _ in sdfg.all_nodes_recursive():
        if (isinstance(n, dace.sdfg.nodes.Tasklet) and n.language == dace.Language.Python
                and "_cpy_out = _cpy_in" in n.code.as_string):
            found_tasklet = True
        if isinstance(n, dace.sdfg.nodes.MapEntry):
            found_map = True
    assert found_tasklet, "Tasklet impl should produce a Python tasklet with ``_cpy_out = _cpy_in``."
    assert not found_map, "Tasklet impl should NOT produce a map."


def test_direct_assignment_rejects_multi_element():
    """``Tasklet`` is size-1 only; rejects multi-element copies."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[32], storage=dace.dtypes.StorageType.GPU_Global, transient=True, name="G_in"),
        _ArraySpec(shape=[32], storage=dace.dtypes.StorageType.GPU_Shared, transient=True, name="S_out"),
        implementation="Tasklet",
        name="da_multi_bad",
        libnode_name="da_multi_bad",
    )
    with pytest.raises(Exception, match="single-element subsets"):
        sdfg.expand_library_nodes()


def test_direct_assignment_rejects_cross_boundary():
    """``Tasklet`` rejects CPU<->GPU pairings via the same-storage check."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[1], storage=dace.dtypes.StorageType.CPU_Heap, subset="0", name="C_in"),
        _ArraySpec(shape=[1], storage=dace.dtypes.StorageType.GPU_Global, transient=True, subset="0", name="G_out"),
        implementation="Tasklet",
        name="da_cross_bad",
        libnode_name="da_cross",
    )
    sdfg.validate()
    with pytest.raises(Exception, match="storage types must match"):
        sdfg.expand_library_nodes()


def test_shared_memory_copy_global_to_shared_is_collective():
    """Global -> Shared collective copy emits a CPP tasklet with __syncthreads() and no GPU_ThreadBlock map."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[64], storage=dace.dtypes.StorageType.GPU_Global, transient=True, name="G_in"),
        _ArraySpec(shape=[64], storage=dace.dtypes.StorageType.GPU_Shared, transient=True, name="S_out"),
        implementation="SharedMemoryCollective",
        name="shmcpy_collective",
        libnode_name="shmcpy",
    )
    sdfg.expand_library_nodes()

    found_syncthreads = False
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.Tasklet):
            if n.language == dace.Language.CPP and "__syncthreads" in n.code.as_string:
                found_syncthreads = True
                break
    assert found_syncthreads, ("SharedMemoryCopy (Global->Shared) should generate a CPP tasklet "
                               "containing __syncthreads().")

    # No GPU_ThreadBlock map: the collective tasklet is itself the block-level op.
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.MapEntry):
            assert n.schedule != dace.dtypes.ScheduleType.GPU_ThreadBlock, (
                "SharedMemoryCopy (Global->Shared) should not generate a "
                "GPU_ThreadBlock map.")


def _libnode_in_tblock_scope(src_storage, dst_storage, src_subset, dst_subset, src_shape=None, dst_shape=None):
    """Build an SDFG with a ``CopyLibraryNode`` nested inside a ``GPU_ThreadBlock``
    map; returns ``(sdfg, libnode, state)`` for scope-aware dispatcher tests."""
    src_shape = src_shape or [16]
    dst_shape = dst_shape or [16]
    sdfg = dace.SDFG(f"in_tblock_{src_storage.name}_{dst_storage.name}")
    sdfg.add_array("src",
                   src_shape,
                   dace.float64,
                   storage=src_storage,
                   transient=(src_storage != dace.dtypes.StorageType.CPU_Heap))
    sdfg.add_array("dst",
                   dst_shape,
                   dace.float64,
                   storage=dst_storage,
                   transient=(dst_storage != dace.dtypes.StorageType.CPU_Heap))
    state = sdfg.add_state("main")
    src_acc = state.add_access("src")
    dst_acc = state.add_access("dst")
    ome, omx = state.add_map("device_map", {"bi": "0:1"}, schedule=dace.dtypes.ScheduleType.GPU_Device)
    ime, imx = state.add_map("tblock_map", {"ti": "0:16"}, schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock)
    libnode = CopyLibraryNode(name="cp")
    state.add_memlet_path(src_acc,
                          ome,
                          ime,
                          libnode,
                          dst_conn=CopyLibraryNode.INPUT_CONNECTOR_NAME,
                          memlet=dace.memlet.Memlet(f"src[{src_subset}]"))
    state.add_memlet_path(libnode,
                          imx,
                          omx,
                          dst_acc,
                          src_conn=CopyLibraryNode.OUTPUT_CONNECTOR_NAME,
                          memlet=dace.memlet.Memlet(f"dst[{dst_subset}]"))
    return sdfg, libnode, state


# Auto-dispatch unit tests for Shared-involved copies. One exact-impl
# assertion per unique routing rule (symmetric directions share the rule);
# end-to-end correctness lives in the ``test_copy_*_roundtrip`` tests.
# The "no single-element -> MappedTasklet" invariant is exhaustively
# covered by ``test_auto_dispatch_single_element_never_mapped_tasklet``.


def test_auto_dispatch_multi_element_shared_register_routes_to_mapped_tasklet():
    """Rule 2 (multi): Shared <-> Register multi-element -> ``MappedTasklet``."""
    sdfg, node = _make_copy_sdfg(
        _ArraySpec(shape=[8], storage=dace.dtypes.StorageType.GPU_Shared, transient=True, name="S_in"),
        _ArraySpec(shape=[8], storage=dace.dtypes.StorageType.Register, transient=True, name="R_out"),
        name="auto_shm_to_reg",
        libnode_name="cp_shm_reg",
    )
    assert select_copy_implementation(node, sdfg.start_state) == "MappedTasklet"


def test_auto_dispatch_single_element_shared_register_routes_to_tasklet():
    """Rule 2 (single): Shared <-> Register single-element -> ``Tasklet``."""
    sdfg, node = _make_copy_sdfg(
        _ArraySpec(shape=[8], storage=dace.dtypes.StorageType.GPU_Shared, transient=True, subset="3", name="S_in"),
        _ArraySpec(shape=[1], storage=dace.dtypes.StorageType.Register, transient=True, subset="0", name="R_out"),
        name="auto_shm_reg_single",
        libnode_name="cp_shm_reg_single",
    )
    assert select_copy_implementation(node, sdfg.start_state) == "Tasklet"


def test_auto_dispatch_global_shared_outside_tblock_routes_to_collective():
    """Rule 3 (multi): Global <-> Shared outside a ThreadBlock map -> ``SharedMemoryCollective``."""
    sdfg, node = _make_copy_sdfg(
        _ArraySpec(shape=[64], storage=dace.dtypes.StorageType.GPU_Global, transient=True, name="G_in"),
        _ArraySpec(shape=[64], storage=dace.dtypes.StorageType.GPU_Shared, transient=True, name="S_out"),
        name="auto_global_to_shm",
        libnode_name="cp_global_shm",
    )
    assert select_copy_implementation(node, sdfg.start_state) == "SharedMemoryCollective"


def test_auto_dispatch_single_element_global_shared_outside_tblock_still_collective():
    """Rule 3 (single): Global <-> Shared single-element outside ThreadBlock -> ``SharedMemoryCollective``."""
    sdfg, node = _make_copy_sdfg(
        _ArraySpec(shape=[64], storage=dace.dtypes.StorageType.GPU_Global, transient=True, subset="5", name="G_in"),
        _ArraySpec(shape=[8], storage=dace.dtypes.StorageType.GPU_Shared, transient=True, subset="3", name="S_out"),
        name="auto_global_shm_single",
        libnode_name="cp_global_shm_single",
    )
    assert select_copy_implementation(node, sdfg.start_state) == "SharedMemoryCollective"


def test_auto_dispatch_shared_shared_outside_tblock_routes_to_collective():
    """Rule 3 (Shared<->Shared): outside ThreadBlock -> ``SharedMemoryCollective``."""
    sdfg, node = _make_copy_sdfg(
        _ArraySpec(shape=[32], storage=dace.dtypes.StorageType.GPU_Shared, transient=True, name="S_a"),
        _ArraySpec(shape=[32], storage=dace.dtypes.StorageType.GPU_Shared, transient=True, name="S_b"),
        name="auto_shm_to_shm",
        libnode_name="cp_shm_shm",
    )
    assert select_copy_implementation(node, sdfg.start_state) == "SharedMemoryCollective"


def test_auto_dispatch_global_shared_inside_tblock_routes_to_mapped_tasklet():
    """Rule 4 (multi): Global -> Shared *inside* a ThreadBlock map is per-thread -> ``MappedTasklet``."""
    sdfg, node, state = _libnode_in_tblock_scope(dace.dtypes.StorageType.GPU_Global,
                                                 dace.dtypes.StorageType.GPU_Shared,
                                                 src_subset="0:4",
                                                 dst_subset="0:4")
    assert select_copy_implementation(node, state) == "MappedTasklet"


def test_auto_dispatch_global_shared_inside_tblock_single_element_routes_to_tasklet():
    """Rule 4 (single): Global -> Shared single-element *inside* a ThreadBlock map -> ``Tasklet``."""
    sdfg, node, state = _libnode_in_tblock_scope(dace.dtypes.StorageType.GPU_Global,
                                                 dace.dtypes.StorageType.GPU_Shared,
                                                 src_subset="ti",
                                                 dst_subset="ti")
    assert select_copy_implementation(node, state) == "Tasklet"


def test_shared_memory_collective_single_element_emits_syncthreads():
    """Single-element collective Global -> Shared must emit ``__syncthreads()`` (the barrier is volume-independent)."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[64], storage=dace.dtypes.StorageType.GPU_Global, transient=True, subset="5", name="G_in"),
        _ArraySpec(shape=[8], storage=dace.dtypes.StorageType.GPU_Shared, transient=True, subset="3", name="S_out"),
        name="auto_global_shm_single_e2e",
        libnode_name="cp_global_shm_single_e2e",
    )
    sdfg.expand_library_nodes()
    assert any(isinstance(n, dace.sdfg.nodes.Tasklet) and n.language == dace.Language.CPP
               and "__syncthreads" in n.code.as_string
               for n, _ in sdfg.all_nodes_recursive()), \
        "Single-element collective Global->Shared must still emit __syncthreads()."


_SINGLE_ELT_STORAGES = [
    dace.dtypes.StorageType.CPU_Heap,
    dace.dtypes.StorageType.GPU_Global,
    dace.dtypes.StorageType.GPU_Shared,
    dace.dtypes.StorageType.Register,
]


@pytest.mark.parametrize("src_storage", _SINGLE_ELT_STORAGES)
@pytest.mark.parametrize("dst_storage", _SINGLE_ELT_STORAGES)
def test_auto_dispatch_single_element_never_mapped_tasklet(src_storage, dst_storage):
    """Invariant: no single-element copy is ever routed to ``MappedTasklet`` (a 0-D map crashes in propagation), over every storage pair."""
    src_kwargs = {"transient": True} if src_storage != dace.dtypes.StorageType.CPU_Heap else {}
    dst_kwargs = {"transient": True} if dst_storage != dace.dtypes.StorageType.CPU_Heap else {}
    sdfg, node = _make_copy_sdfg(
        _ArraySpec(shape=[8], storage=src_storage, subset="3", name="src", **src_kwargs),
        _ArraySpec(shape=[8], storage=dst_storage, subset="5", name="dst", **dst_kwargs),
        name=f"auto_single_{src_storage.name}_{dst_storage.name}",
        libnode_name=f"cp_single_{src_storage.name}_{dst_storage.name}",
    )
    state = sdfg.start_state
    impl = select_copy_implementation(node, state)
    assert impl != "MappedTasklet", (
        f"Single-element {src_storage.name} -> {dst_storage.name} routed to MappedTasklet; "
        "single-element copies must use Tasklet / MemcpyCUDA1D / SharedMemoryCollective.")


def test_shared_memory_copy_rejects_no_shared():
    """SharedMemoryCopy expansion rejects if neither side is GPU_Shared."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[32], storage=dace.dtypes.StorageType.GPU_Global, transient=True, name="G_in"),
        _ArraySpec(shape=[32], storage=dace.dtypes.StorageType.Register, transient=True, name="R_out"),
        implementation="SharedMemoryCollective",
        name="shmcpy_bad",
        libnode_name="shmcpy_bad",
    )
    with pytest.raises(Exception, match="GPU_Shared / GPU_Global storages"):
        sdfg.expand_library_nodes()


def test_shared_memory_copy_rejects_cpu():
    """SharedMemoryCopy expansion rejects CPU_Heap storage."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[32], storage=dace.dtypes.StorageType.CPU_Heap, name="C_in"),
        _ArraySpec(shape=[32], storage=dace.dtypes.StorageType.GPU_Shared, transient=True, name="S_out"),
        implementation="SharedMemoryCollective",
        name="shmcpy_cpu",
        libnode_name="shmcpy_cpu",
    )
    with pytest.raises(Exception, match="GPU_Shared / GPU_Global storages"):
        sdfg.expand_library_nodes()


def test_shared_memory_copy_rejects_inside_tblock_map():
    """A collective ``SharedMemoryCollective`` copy nested in a GPU_ThreadBlock map raises at expansion."""
    sdfg = dace.SDFG("shmcpy_in_tblock")
    sdfg.add_array("A", [256], dace.float64, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", [256], dace.float64, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("shmem", [32], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)

    state = sdfg.add_state("main")
    a = state.add_access("A")
    shm = state.add_access("shmem")

    ome, omx = state.add_map("device_map", {"bi": "0:256:32"}, schedule=dace.dtypes.ScheduleType.GPU_Device)
    # ThreadBlock map is an invalid parent for a collective copy.
    ime, imx = state.add_map("tblock_map", {"ti": "0:32"}, schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock)

    libnode = CopyLibraryNode(name="shmcpy_bad")
    libnode.implementation = "SharedMemoryCollective"

    state.add_memlet_path(a,
                          ome,
                          ime,
                          libnode,
                          dst_conn=CopyLibraryNode.INPUT_CONNECTOR_NAME,
                          memlet=dace.Memlet("A[bi:bi+32]"))
    state.add_memlet_path(libnode,
                          imx,
                          omx,
                          shm,
                          src_conn=CopyLibraryNode.OUTPUT_CONNECTOR_NAME,
                          memlet=dace.Memlet("shmem[0:32]"))

    with pytest.raises(Exception, match="GPU_ThreadBlock"):
        sdfg.expand_library_nodes()


@pytest.mark.gpu
def test_copy_roundtrip_variant_a_cooperative_load():
    """Variant A: collective load OUTSIDE the tblock_map (block-cooperative ``dace::CopyND`` + ``__syncthreads()``), per-thread writeback inside it, round-tripping through Global ``B``."""
    import cupy as cp

    N = 256
    TILE = 32
    sdfg = dace.SDFG("roundtrip_variant_a")
    sdfg.add_array("A", [N], dace.float64, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", [N], dace.float64, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("tile", [TILE], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)

    state = sdfg.add_state("main")
    a = state.add_access("A")
    tile = state.add_access("tile")
    b = state.add_access("B")

    ome, omx = state.add_map("device_map", {"bi": f"0:{N}:{TILE}"}, schedule=dace.dtypes.ScheduleType.GPU_Device)

    # Cooperative load: libnode sits OUTSIDE the tblock map (between ome and ime).
    load = CopyLibraryNode(name="load_a_to_tile")
    state.add_memlet_path(a,
                          ome,
                          load,
                          dst_conn=CopyLibraryNode.INPUT_CONNECTOR_NAME,
                          memlet=dace.Memlet(f"A[bi:bi+{TILE}]"))
    state.add_edge(load, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, tile, None, dace.Memlet(f"tile[0:{TILE}]"))

    ime, imx = state.add_map("tblock_map", {"ti": f"0:{TILE}"}, schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock)
    t = state.add_tasklet("writeback", {"v"}, {"o"}, "o = v")
    state.add_memlet_path(tile, ime, t, dst_conn="v", memlet=dace.Memlet("tile[ti]"))
    state.add_memlet_path(t, imx, omx, b, src_conn="o", memlet=dace.Memlet("B[bi+ti]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    A = cp.arange(N, dtype=cp.float64) * 3.0 + 0.5
    B = cp.zeros(N, dtype=cp.float64)
    sdfg(A=A, B=B)
    cp.testing.assert_array_equal(B, A)


@pytest.mark.gpu
def test_copy_roundtrip_variant_b_per_thread_load():
    """Variant B: per-thread load INSIDE the tblock_map -- each thread copies ``A[bi+ti] -> tile[ti] -> B[bi+ti]`` via its own ``Tasklet`` (no block-collective)."""
    import cupy as cp

    N = 256
    TILE = 32
    sdfg = dace.SDFG("roundtrip_variant_b")
    sdfg.add_array("A", [N], dace.float64, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", [N], dace.float64, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("tile", [TILE], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)

    state = sdfg.add_state("main")
    a = state.add_access("A")
    tile = state.add_access("tile")
    b = state.add_access("B")

    ome, omx = state.add_map("device_map", {"bi": f"0:{N}:{TILE}"}, schedule=dace.dtypes.ScheduleType.GPU_Device)
    ime, imx = state.add_map("tblock_map", {"ti": f"0:{TILE}"}, schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock)

    # Per-thread load: libnode INSIDE the tblock map -- each thread copies one cell.
    load = CopyLibraryNode(name="load_a_to_tile_per_thread")
    state.add_memlet_path(a,
                          ome,
                          ime,
                          load,
                          dst_conn=CopyLibraryNode.INPUT_CONNECTOR_NAME,
                          memlet=dace.Memlet("A[bi+ti]"))
    state.add_edge(load, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, tile, None, dace.Memlet("tile[ti]"))

    # Per-thread store: libnode INSIDE the tblock map -- each thread writes its cell.
    store = CopyLibraryNode(name="store_tile_to_b_per_thread")
    state.add_edge(tile, None, store, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet("tile[ti]"))
    state.add_memlet_path(store,
                          imx,
                          omx,
                          b,
                          src_conn=CopyLibraryNode.OUTPUT_CONNECTOR_NAME,
                          memlet=dace.Memlet("B[bi+ti]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    A = cp.arange(N, dtype=cp.float64) * 5.0 - 2.0
    B = cp.zeros(N, dtype=cp.float64)
    sdfg(A=A, B=B)
    cp.testing.assert_array_equal(B, A)


@pytest.mark.gpu
def test_copy_full_pipeline_roundtrip():
    """Pipeline: Global -> Shared (collective) -> per-thread (Register -> Register -> Shared) -> Global; exercises auto-dispatched Shared<->Register libnodes alongside the block-cooperative load."""
    import cupy as cp

    N = 256
    TILE = 32
    sdfg = dace.SDFG("full_pipeline_roundtrip")
    sdfg.add_array("A", [N], dace.float64, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", [N], dace.float64, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("shm_in", [TILE], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)
    sdfg.add_array("shm_out", [TILE], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)
    sdfg.add_array("reg_a", [1], dace.float64, dace.dtypes.StorageType.Register, transient=True)
    sdfg.add_array("reg_b", [1], dace.float64, dace.dtypes.StorageType.Register, transient=True)

    state = sdfg.add_state("main")
    a = state.add_access("A")
    shm_in = state.add_access("shm_in")
    shm_out = state.add_access("shm_out")
    b = state.add_access("B")

    ome, omx = state.add_map("device_map", {"bi": f"0:{N}:{TILE}"}, schedule=dace.dtypes.ScheduleType.GPU_Device)

    # Global -> Shared (collective load).
    load = CopyLibraryNode(name="load_a_to_shm")
    state.add_memlet_path(a,
                          ome,
                          load,
                          dst_conn=CopyLibraryNode.INPUT_CONNECTOR_NAME,
                          memlet=dace.Memlet(f"A[bi:bi+{TILE}]"))
    state.add_edge(load, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, shm_in, None, dace.Memlet(f"shm_in[0:{TILE}]"))

    # Single GPU_ThreadBlock map carries:
    #   Shared(shm_in) -> Register(reg_a) -> Register(reg_b) -> Shared(shm_out)
    #     -> Global(B) (per-thread tasklet for the last leg)
    ime, imx = state.add_map("tblock_map", {"ti": f"0:{TILE}"}, schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock)
    s2r = CopyLibraryNode(name="shm_to_reg_a")
    r2r = CopyLibraryNode(name="reg_a_to_reg_b")
    r2s = CopyLibraryNode(name="reg_b_to_shm")
    reg_a = state.add_access("reg_a")
    reg_b = state.add_access("reg_b")

    state.add_memlet_path(shm_in,
                          ime,
                          s2r,
                          dst_conn=CopyLibraryNode.INPUT_CONNECTOR_NAME,
                          memlet=dace.Memlet("shm_in[ti]"))
    state.add_edge(s2r, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, reg_a, None, dace.Memlet("reg_a[0]"))
    state.add_edge(reg_a, None, r2r, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet("reg_a[0]"))
    state.add_edge(r2r, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, reg_b, None, dace.Memlet("reg_b[0]"))
    state.add_edge(reg_b, None, r2s, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet("reg_b[0]"))
    state.add_memlet_path(r2s,
                          imx,
                          shm_out,
                          src_conn=CopyLibraryNode.OUTPUT_CONNECTOR_NAME,
                          memlet=dace.Memlet("shm_out[ti]"))

    # Per-thread Shared -> Global writeback via a tasklet -- avoids a
    # second block-collective copy in the same kernel.
    ime2, imx2 = state.add_map("writeback_map", {"tj": f"0:{TILE}"}, schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock)
    tw = state.add_tasklet("writeback", {"v"}, {"o"}, "o = v")
    state.add_memlet_path(shm_out, ime2, tw, dst_conn="v", memlet=dace.Memlet("shm_out[tj]"))
    state.add_memlet_path(tw, imx2, omx, b, src_conn="o", memlet=dace.Memlet("B[bi+tj]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    A = cp.arange(N, dtype=cp.float64) * 2.0 + 1.0
    B = cp.zeros(N, dtype=cp.float64)
    sdfg(A=A, B=B)
    cp.testing.assert_array_equal(B, A)


def test_copy_pure_cpu_2d():
    """Pure expansion on a 2D slice copy, CPU_Heap."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[10, 20], storage=dace.dtypes.StorageType.CPU_Heap, subset="2:8, 5:15", name="A"),
        _ArraySpec(shape=[10, 20], storage=dace.dtypes.StorageType.CPU_Heap, subset="0:6, 0:10", name="B"),
        implementation="MappedTasklet",
        name="copy_2d_cpu",
        libnode_name="cp2d",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = _compile_no_copynd(sdfg)

    A = np.arange(200, dtype=np.float64).reshape(10, 20).copy()
    B = np.zeros((10, 20), dtype=np.float64)
    exe(A=A, B=B)

    np.testing.assert_array_equal(B[0:6, 0:10], A[2:8, 5:15])


@pytest.mark.gpu
def test_copy_single_element_h2d():
    """Single-element host -> GPU copy compiles and round-trips."""
    pytest.importorskip('cupy')
    import cupy as cp
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[1], storage=dace.dtypes.StorageType.CPU_Heap, name="host"),
        _ArraySpec(shape=[1], storage=dace.dtypes.StorageType.GPU_Global, name="dev"),
        name="single_elem_h2d",
        libnode_name="copy_h2d",
    )

    host = np.array([3.14159], dtype=np.float64)
    dev = cp.zeros(1, dtype=cp.float64)

    _compile_no_copynd(sdfg)(host=host, dev=dev)
    np.testing.assert_allclose(cp.asnumpy(dev), host)


@pytest.mark.gpu
def test_copy_two_element_h2d():
    """A 2-element host -> GPU copy compiles and round-trips (pointer-typed connectors, unlike single element)."""
    pytest.importorskip('cupy')
    import cupy as cp
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[2], storage=dace.dtypes.StorageType.CPU_Heap, name="host"),
        _ArraySpec(shape=[2], storage=dace.dtypes.StorageType.GPU_Global, name="dev"),
        name="two_elem_h2d",
        libnode_name="copy_h2d_2",
    )

    host = np.array([1.0, 2.0], dtype=np.float64)
    dev = cp.zeros(2, dtype=cp.float64)
    _compile_no_copynd(sdfg)(host=host, dev=dev)
    np.testing.assert_allclose(cp.asnumpy(dev), host)


@pytest.mark.gpu
def test_copy_single_element_d2h():
    """Single-element GPU -> host copy compiles and round-trips."""
    pytest.importorskip('cupy')
    import cupy as cp
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[1], storage=dace.dtypes.StorageType.GPU_Global, name="dev"),
        _ArraySpec(shape=[1], storage=dace.dtypes.StorageType.CPU_Heap, name="host"),
        name="single_elem_d2h",
        libnode_name="copy_d2h",
    )

    dev = cp.array([2.71828], dtype=cp.float64)
    host = np.zeros(1, dtype=np.float64)

    _compile_no_copynd(sdfg)(host=host, dev=dev)
    np.testing.assert_allclose(host, cp.asnumpy(dev))


# Legacy direct-edge miscompile regression pins: each test builds the SDFG twice
# -- with a CopyLibraryNode and with the canonical direct AN -> AN edge -- and checks
# both against a NumPy for-loop. The libnode's advantage is rank-mismatch reshapes
# with per-side layout strides, which the legacy memcpy path miscompiles or fails to
# compile. The legacy-fails assertions are informational: if legacy ever produces
# correct output, the test fails and should be deleted (the advantage is gone).


def _legacy_fails(sdfg_leg: dace.SDFG, expected: np.ndarray, run) -> bool:
    """``True`` if compiling/running the legacy SDFG raises OR produces output diverging from ``expected``.

    :param sdfg_leg: SDFG with libnodes already replaced by direct edges.
    :param expected: NumPy ground truth.
    :param run: a callable ``run(exe) -> np.ndarray`` that runs the compiled SDFG and returns the dst array.
    """
    try:
        exe = sdfg_leg.compile()
        return not np.array_equal(run(exe), expected)
    except Exception:
        return True


def test_legacy_silently_miscompiles_rank_mismatch_fortran_collapse():
    """Pin: legacy direct-edge miscompiles a 4D->2D Fortran-packed reshape."""
    src = _ArraySpec(shape=(2, 3, 4, 5),
                     storage=dace.dtypes.StorageType.CPU_Heap,
                     strides=(1, 2, 6, 24),
                     total_size=120)
    dst = _ArraySpec(shape=(6, 20), storage=dace.dtypes.StorageType.CPU_Heap, strides=(1, 6), total_size=120)
    sdfg_lib, _ = _make_copy_sdfg(src, dst, name="legacy_fortran_collapse_lib")
    sdfg_leg = _make_legacy_copy_sdfg(src, dst, name="legacy_fortran_collapse_leg")

    A = np.arange(120, dtype=np.float64).reshape(2, 3, 4, 5, order='F').copy(order='F')
    expected = np.zeros((6, 20), dtype=np.float64, order='F')
    # Fortran-order flat walk: src index (i,j,k,l) -> flat n = i + j*2 + k*6 + l*24
    # dst index (p, q) -> flat n = p + q*6
    flat = np.empty(120, dtype=np.float64)
    for l in range(5):
        for k in range(4):
            for j in range(3):
                for i in range(2):
                    flat[i + j * 2 + k * 6 + l * 24] = A[i, j, k, l]
    for q in range(20):
        for p in range(6):
            expected[p, q] = flat[p + q * 6]

    B_lib = np.zeros((6, 20), dtype=np.float64, order='F')
    sdfg_lib.expand_library_nodes()
    _compile_no_copynd(sdfg_lib)(src=A, dst=B_lib)
    np.testing.assert_array_equal(B_lib, expected)

    def run(exe):
        out = np.zeros((6, 20), dtype=np.float64, order='F')
        exe(src=A, dst=out)
        return out

    assert _legacy_fails(sdfg_leg, expected, run), ("Legacy direct-edge no longer fails on 4D->2D Fortran reshape; "
                                                    "remove this test, the libnode advantage is gone.")


def test_single_element_in_kernel_register_to_gpu_global_routes_to_tasklet():
    """Single-element in-kernel Register -> GPU_Global routes to a direct Tasklet, not MappedTasklet."""
    sdfg = dace.SDFG('reg_to_gpuglobal_in_kernel')
    sdfg.add_array('R', [1, 1, 1], dace.float64, dace.StorageType.Register, transient=True)
    sdfg.add_array('G', [4, 4, 4], dace.float64, dace.StorageType.GPU_Global, transient=True)
    state = sdfg.add_state('s')

    # Wrap the copy inside a GPU_Device map so ``is_devicelevel_gpu`` returns True.
    me, mx = state.add_map('kernel', dict(i='0:1'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    r = state.add_access('R')
    g = state.add_access('G')
    libnode = CopyLibraryNode(name='reg_to_g')
    state.add_node(libnode)
    state.add_memlet_path(me, r, memlet=dace.Memlet())
    state.add_edge(r, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet('R[0, 0, 0]'))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, g, None, dace.Memlet('G[0, 0, 0]'))
    state.add_memlet_path(g, mx, memlet=dace.Memlet())

    sdfg.expand_library_nodes()

    nsdfg_count = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    assert nsdfg_count == 0, (f"Single-element in-kernel copy should expand to a direct Tasklet, "
                              f"not a NestedSDFG; got {nsdfg_count} NestedSDFG(s).")
    assignments = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.Tasklet) and '_cpy_out = _cpy_in' in n.code.as_string
    ]
    assert assignments, "Expected at least one ``_cpy_out = _cpy_in`` Tasklet from the expansion."


def test_register_location_detection():
    """Register location detection distinguishes in-kernel from host-side copies."""
    sdfg = dace.SDFG('register_location_detection')
    sdfg.add_array('R', [1], dace.float64, dace.StorageType.Register, transient=True)
    sdfg.add_array('G', [1], dace.float64, dace.StorageType.GPU_Global, transient=True)
    state = sdfg.add_state('s')

    r = state.add_access('R')
    g = state.add_access('G')
    libnode = CopyLibraryNode(name='reg_to_g')
    state.add_node(libnode)
    state.add_edge(r, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet('R[0]'))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, g, None, dace.Memlet('G[0]'))

    sdfg.expand_library_nodes()

    nsdfg_count = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    assert nsdfg_count == 0, (f"Single-element in-kernel copy should expand to a direct Memcpy (cross-boundary), "
                              f"not a NestedSDFG; got {nsdfg_count} NestedSDFG(s).")
    assignments = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.Tasklet) and 'cudaMemcpy' in n.code.as_string
    ]
    assert assignments, "Expected at least one ``cudaMemcpy`` Tasklet from the expansion."


def test_cuda2d_pitch_params_branches():
    """``cuda2d_pitch_params`` returns element-count ``(dpitch, spitch, width, height)`` for each
    supported 2D stride pattern and ``None`` otherwise. It is the single source of truth shared by
    the ``MemcpyCUDA2D`` selector gate and the expander, so selector and expander cannot drift."""
    # Contiguous rows (inner stride 1): pitch = outer stride, width = columns, height = rows.
    assert cuda2d_pitch_params([4, 3], [3, 1], [10, 1]) == (10, 3, 3, 4)
    # Contiguous columns (outer stride 1): the roles of the two axes swap.
    assert cuda2d_pitch_params([4, 3], [1, 4], [1, 8]) == (8, 4, 4, 3)
    # Neither axis unit-strided, but outer/inner ratio equals the inner width -> one strided run.
    assert cuda2d_pitch_params([4, 2], [4, 2], [6, 3]) == (3, 2, 1, 8)
    # No single cudaMemcpy2DAsync expresses this pattern.
    assert cuda2d_pitch_params([4, 3], [5, 2], [5, 2]) is None


if __name__ == "__main__":
    pytest.main([__file__])
