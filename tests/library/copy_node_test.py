# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``CopyLibraryNode`` and its pure, CPU, CUDA, cross-storage, register, and shared-memory expansions."""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import dace
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode

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
    """Build a one-state SDFG that copies ``src`` -> ``dst`` via a single ``CopyLibraryNode``.

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
    """Build a one-state SDFG with ``src`` / ``dst`` arrays + AccessNodes, returning subsets too.

    Shared scaffolding for :func:`_make_copy_sdfg` (libnode form) and
    :func:`_make_legacy_copy_sdfg` (canonical direct-edge form).
    """
    sdfg = dace.SDFG(name)
    src_name = src.name or "src"
    dst_name = dst.name or "dst"
    for arr_name, spec in ((src_name, src), (dst_name, dst)):
        kwargs = {"transient": spec.transient}
        if spec.strides is not None:
            kwargs["strides"] = spec.strides
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
    """Build a one-state SDFG that copies ``src`` -> ``dst`` via a canonical direct AN -> AN edge.

    Uses the legacy DaCe memlet convention: ``data=dst``, ``subset`` is the dst
    write region, ``other_subset`` is the src read region. This is what the
    standard DaCe copy lowering produces and the basis for comparing against
    the :class:`CopyLibraryNode` path.
    """
    sdfg, src_name, dst_name, src_acc, dst_acc, src_subset, dst_subset = _make_copy_skeleton(src, dst, name, dtype)
    sdfg.start_state.add_edge(
        src_acc, None, dst_acc, None,
        dace.memlet.Memlet(data=dst_name, subset=dst_subset, other_subset=src_subset))
    return sdfg


def _fortran_strides(shape):
    """Column-major Fortran-packed strides, via the same helper ``Array.is_packed_fortran_strides`` checks against."""
    return dace.data.Array(dace.float64, shape=shape)._get_packed_fortran_strides()


def _compile_no_copynd(sdfg: dace.SDFG):
    """Assert the SDFG's generated C++ contains no ``dace::CopyND`` template, then compile.

    The libnodes are designed to displace the runtime CopyND fallback entirely. The only
    intentional ``CopyND`` user is ``ExpandSharedMemoryCollective`` (block-collective shared
    memory load); tests exercising that expansion inspect tasklet bodies directly and don't
    run codegen, so a universal post-codegen assertion is safe here.
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
    """Same-rank Fortran-packed (column-major) copy lowers via the Auto-routed MappedTasklet."""
    sdfg, libnode = _make_copy_sdfg(
        _ArraySpec(shape=(4, 5, 6), storage=dace.dtypes.StorageType.CPU_Heap, strides=(1, 4, 20)),
        _ArraySpec(shape=(4, 5, 6), storage=dace.dtypes.StorageType.CPU_Heap, strides=(1, 4, 20)),
        name="copy_fortran_packed_same_rank",
    )
    sdfg.validate()
    sdfg.expand_library_nodes()
    assert libnode.implementation == 'MappedTasklet'

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
    assert node.src_storage(state, sdfg) == dace.dtypes.StorageType.CPU_Heap
    assert node.dst_storage(state, sdfg) == dace.dtypes.StorageType.GPU_Global


def test_copy_node_storage_defaults_when_unattached():
    """Without edges, the storage methods fall back to ``StorageType.Default``."""
    sdfg = dace.SDFG("storage_unattached")
    state = sdfg.add_state("main")
    node = CopyLibraryNode(name="unattached")
    state.add_node(node)

    assert node.src_storage(state, sdfg) == dace.dtypes.StorageType.Default
    assert node.dst_storage(state, sdfg) == dace.dtypes.StorageType.Default


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


def test_direct_assignment_rejects_shared_memory():
    """``Tasklet`` is same-storage only (size-1); rejects mismatched storages."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[1], storage=dace.dtypes.StorageType.GPU_Global, transient=True, subset="0", name="G_in"),
        _ArraySpec(shape=[1], storage=dace.dtypes.StorageType.GPU_Shared, transient=True, subset="0", name="S_out"),
        implementation="Tasklet",
        name="da_shm_bad",
        libnode_name="da_bad",
    )
    with pytest.raises(Exception, match="storage types must match"):
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


def test_shared_memory_copy_shared_to_register_is_thread_level():
    """A Shared -> Register ``MappedTasklet`` copy expands to a Sequential (thread-level) map."""
    sdfg, _ = _make_copy_sdfg(
        _ArraySpec(shape=[8], storage=dace.dtypes.StorageType.GPU_Shared, transient=True, name="S_in"),
        _ArraySpec(shape=[8], storage=dace.dtypes.StorageType.Register, transient=True, name="R_out"),
        implementation="MappedTasklet",
        name="shmcpy_thread",
        libnode_name="shmcpy_thr",
    )
    sdfg.expand_library_nodes()

    found_sequential = False
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.MapEntry):
            if n.schedule == dace.dtypes.ScheduleType.Sequential:
                found_sequential = True
                break
    assert found_sequential, ("SharedMemoryCopy (Shared->Register) should contain a Sequential map.")


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


# --- Legacy direct-edge miscompile regression pins ----------------------------
#
# Each test below builds the same SDFG twice: once with a CopyLibraryNode
# (which we expand), and once with the canonical DaCe direct AN -> AN edge
# (``Memlet(data=dst, subset=dst_subset, other_subset=src_subset)``). Ground
# truth is computed via an explicit Python for-loop on NumPy arrays; both
# DaCe paths must agree with it. The libnode does; the legacy path either
# silently miscompiles or fails to compile.
#
# Most simple AN -> AN copies (same-rank slices, strided subsets with step,
# mixed C/F layouts at the same rank) the legacy codegen actually handles
# correctly when given the canonical memlet form. The libnode's clear
# advantage is rank-mismatch reshapes with explicit per-side layout strides:
# the legacy memcpy strategy doesn't bridge a layout-aware flat walk between
# differently-shaped endpoints.
#
# The legacy-fails assertion is informational: if the legacy codegen ever
# starts producing the correct output for the pattern below, this test will
# fail and should be deleted (the advantage is gone).


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
    dst = _ArraySpec(shape=(6, 20),
                     storage=dace.dtypes.StorageType.CPU_Heap,
                     strides=(1, 6),
                     total_size=120)
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

    assert _legacy_fails(sdfg_leg, expected, run), (
        "Legacy direct-edge no longer fails on 4D->2D Fortran reshape; "
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


if __name__ == "__main__":
    test_copy_pure_cpu()
    test_copy_cpu_memcpy()
    test_copy_pure_cpu_2d()

    # Properties
    test_copy_node_storage_from_edges()
    test_copy_node_storage_defaults_when_unattached()

    # Validation
    test_copy_dtype_mismatch_rejected()
    test_cpu_memcpy_rejects_non_contiguous_subset()
    test_strided_expansions_accept_non_contiguous()
    test_copy_pure_host_to_device_rejected()
    test_copy_pure_device_to_host_rejected()

    # DirectAssignment expansion
    test_direct_assignment_cpu_same_storage()
    test_direct_assignment_register_to_register()
    test_direct_assignment_rejects_shared_memory()
    test_direct_assignment_rejects_cross_boundary()

    # RegisterCopy expansion (RegisterCopy is now an alias for MappedTasklet)
    test_register_copy_expands_with_register_storage()

    # SharedMemoryCopy expansion
    test_shared_memory_copy_global_to_shared_is_collective()
    test_shared_memory_copy_shared_to_register_is_thread_level()
    test_shared_memory_copy_rejects_no_shared()
    test_shared_memory_copy_rejects_cpu()
    test_shared_memory_copy_rejects_inside_tblock_map()

    # Fortran-packed + mixed-layout via MappedTasklet
    test_copy_fortran_packed_same_rank()
    test_copy_fortran_packed_strided_slice()
    test_copy_mixed_c_fortran_via_mapped_tasklet()

    # Same-rank subset-shape contract
    test_copy_same_subset_different_array_shapes()
    test_copy_1d_slice_from_2d_source()
    test_copy_transpose_pattern_rejected()

    # Rank-mismatch reshape feature-regression rejection tests
    test_copy_rank_mismatch_mixed_layouts_raises()
    test_copy_rank_mismatch_padded_src_raises()
    test_copy_rank_mismatch_strided_subset_raises()

    # Edge cases (Auto-routed; numpy correctness check)
    test_copy_4d_to_1d_flatten_c_packed()
    test_copy_1d_to_4d_inflate_c_packed()
    test_copy_3d_to_2d_collapse_first_two_dims()
    test_copy_4d_to_2d_collapse_pair_dims_fortran()
    test_copy_strided_step_2_cpu_same_rank()

    # GPU tests
    test_copy_pure_gpu()
    test_copy_cuda_d2d()
    test_copy_cuda_host_to_device()
    test_copy_cuda_device_to_host()
    test_copy_cross_storage_validation_rejects_without_flag()
