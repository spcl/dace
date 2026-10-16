from dace.transformation.passes.vectorization.enums import ISA
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the cuTile data-copy pipeline via ``apply_gpu_transformations()``.

Two layers:

* **Structure / codegen tests (no GPU):** after applying the full cuTile
  lowering pipeline (which now uses ``sdfg.apply_gpu_transformations()``
  instead of ``CuTileInsertDataCopies``), the SDFG has the expected copy-in /
  copy-out states, GPU_Global transient clones, CPU_Heap originals, and correct
  AccessNode/Memlet references.  Codegen tests verify ``.set()`` /
  ``.get(out=...)`` appear in the generated Python code.

* **Runtime tests (``@pytest.mark.gpu``):** compile and run on GPU with
  **NumPy** (host) arrays directly -- the whole point of data copies -- and
  compare against NumPy references.
"""

import ast
from typing import List, Set, Tuple

import numpy as np
import pytest

import dace
from dace import data, dtypes
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.vectorization import VectorizeCuTile
from dace.transformation.passes.vectorization.cutile_lowering import (
    CuTileSetImplementations,
    CuTileSetTileStorage,
    CuTileValidateTiles,
    GPUDeviceToCuTile,
)
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (
    VectorizeCPUMultiDim,
)
from dace.transformation.passes.vectorization.config import VectorizeConfig


# ============================================================
# Fixture builders
# ============================================================


def _build_vadd_sdfg(name: str, dtype: dace.typeclass = dace.float64) -> SDFG:
    """Build a symbolic-sized 1-D ``C[i] = A[i] + B[i]`` SDFG.

    :param name: SDFG name (must be unique per test).
    :param dtype: Data type for all arrays.
    :returns: The constructed SDFG.
    """
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (N,), dtype)
    sdfg.add_array("B", (N,), dtype)
    sdfg.add_array("C", (N,), dtype)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "add",
        {"i": "0:N"},
        {"_a": dace.Memlet("A[i]"), "_b": dace.Memlet("B[i]")},
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    return sdfg


def _build_vadd2d_sdfg(name: str, dtype: dace.typeclass = dace.float64) -> SDFG:
    """Build a symbolic-sized 2-D ``C[i, j] = A[i, j] + B[i, j]`` SDFG.

    :param name: SDFG name (must be unique per test).
    :param dtype: Data type for all arrays.
    :returns: The constructed SDFG.
    """
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (M, N), dtype)
    sdfg.add_array("B", (M, N), dtype)
    sdfg.add_array("C", (M, N), dtype)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "add2d",
        {"i": "0:M", "j": "0:N"},
        {"_a": dace.Memlet("A[i, j]"), "_b": dace.Memlet("B[i, j]")},
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i, j]")},
        external_edges=True,
    )
    return sdfg


def _build_vadd_with_scalar_sdfg(name: str) -> SDFG:
    """Build ``C[i] = A[i] + B[i] + alpha`` with a scalar parameter.

    :param name: SDFG name (must be unique per test).
    :returns: The constructed SDFG.
    """
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (N,), dace.float64)
    sdfg.add_array("B", (N,), dace.float64)
    sdfg.add_array("C", (N,), dace.float64)
    sdfg.add_scalar("alpha", dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "add_scalar",
        {"i": "0:N"},
        {
            "_a": dace.Memlet("A[i]"),
            "_b": dace.Memlet("B[i]"),
            "_s": dace.Memlet("alpha[0]"),
        },
        "_c = _a + _b + _s",
        {"_c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    return sdfg


# ============================================================
# Lowering helpers
# ============================================================


def _lower_full_pipeline(sdfg: SDFG, widths: Tuple[int, ...] = (8,)) -> None:
    """Run the full cuTile pipeline (vectorize + gpu transform + adapters).

    :param sdfg: The SDFG to lower.
    :param widths: Per-dim tile widths (innermost-last).
    """
    vec = VectorizeCPUMultiDim(
        VectorizeConfig(
            widths=widths,
            target_isa=ISA.CUTILE,
            expand_tile_nodes=False,
        ))
    vec.apply_pass(sdfg, {})
    CuTileValidateTiles().apply_pass(sdfg, {})
    sdfg.apply_gpu_transformations(
        simplify=True,
    )
    GPUDeviceToCuTile().apply_pass(sdfg, {})
    CuTileSetTileStorage().apply_pass(sdfg, {})


def _finish_pipeline(sdfg: SDFG) -> None:
    """Run the lowering passes that come after tile storage stamping.

    :param sdfg: The SDFG to finish lowering.
    """
    CuTileSetImplementations().apply_pass(sdfg, {})
    sdfg.expand_library_nodes()
    sdfg.backend = dtypes.BackendLanguage.Python


def _generate_code(sdfg: SDFG) -> str:
    """Generate Python-backend code for ``sdfg`` and return the main file text.

    :param sdfg: The (already lowered) SDFG.
    :returns: The generated code of the main code object.
    """
    code_objects = sdfg.generate_code()
    return next(co for co in code_objects if co.name == sdfg.name).code


# ============================================================
# Inspection helpers
# ============================================================


def _has_copyin_state(sdfg: SDFG) -> bool:
    """Check if the SDFG has a state that performs host-to-device copies.

    After ``apply_gpu_transformations()``, copyin states contain edges where
    the source is a CPU_Heap/Default array and the destination is a GPU_Global
    transient.

    :param sdfg: The SDFG to inspect.
    :returns: True if a copyin state is found.
    """
    for state in sdfg.states():
        for edge in state.edges():
            if (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
                src_desc = sdfg.arrays.get(edge.src.data)
                dst_desc = sdfg.arrays.get(edge.dst.data)
                if (src_desc is not None and dst_desc is not None
                        and src_desc.storage in (dtypes.StorageType.CPU_Heap, dtypes.StorageType.Default)
                        and dst_desc.storage == dtypes.StorageType.GPU_Global and dst_desc.transient):
                    return True
    return False


def _has_copyout_state(sdfg: SDFG) -> bool:
    """Check if the SDFG has a state that performs device-to-host copies.

    After ``apply_gpu_transformations()``, copyout states contain edges where
    the source is a GPU_Global transient and the destination is a CPU_Heap/Default
    array.

    :param sdfg: The SDFG to inspect.
    :returns: True if a copyout state is found.
    """
    for state in sdfg.states():
        for edge in state.edges():
            if (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
                src_desc = sdfg.arrays.get(edge.src.data)
                dst_desc = sdfg.arrays.get(edge.dst.data)
                if (src_desc is not None and dst_desc is not None
                        and src_desc.storage == dtypes.StorageType.GPU_Global and src_desc.transient
                        and dst_desc.storage in (dtypes.StorageType.CPU_Heap, dtypes.StorageType.Default)):
                    return True
    return False


def _state_access_node_names(state: "dace.sdfg.state.SDFGState") -> Set[str]:
    """Collect all AccessNode data names within a single state.

    :param state: The state to inspect.
    :returns: Set of data names referenced by AccessNodes in the state.
    """
    return {
        node.data
        for node in state.nodes()
        if isinstance(node, nodes.AccessNode)
    }


def _gpu_clone_names(sdfg: SDFG) -> Set[str]:
    """Find all GPU_Global transient array names in the SDFG.

    :param sdfg: The SDFG to inspect.
    :returns: Set of GPU_Global transient array names.
    """
    return {
        name for name, desc in sdfg.arrays.items()
        if desc.storage == dtypes.StorageType.GPU_Global and desc.transient
    }


# ============================================================
# Structure tests (no GPU)
# ============================================================


class TestCuTileDataCopiesStructure:
    """SDFG structure after the full pipeline with ``apply_gpu_transformations()``."""

    def test_copyin_state_exists(self) -> None:
        """After applying the pipeline, a copyin state exists containing
        AccessNode pairs for host-to-device transfers."""
        sdfg = _build_vadd_sdfg("dc_struct_copyin_exists")
        _lower_full_pipeline(sdfg)

        assert _has_copyin_state(sdfg), "No copyin state found"

    def test_copyout_state_exists(self) -> None:
        """A copyout state exists containing AccessNode pairs for
        device-to-host transfers for written arrays."""
        sdfg = _build_vadd_sdfg("dc_struct_copyout_exists")
        _lower_full_pipeline(sdfg)

        assert _has_copyout_state(sdfg), "No copyout state found"

    def test_originals_are_host_storage(self) -> None:
        """The original non-transient arrays (A, B, C) have host-side
        storage (Default or CPU_Heap) after the pipeline -- NOT
        GPU_Global."""
        sdfg = _build_vadd_sdfg("dc_struct_cpu_heap")
        _lower_full_pipeline(sdfg)

        host_storages = {dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap}
        for name in ("A", "B", "C"):
            desc = sdfg.arrays[name]
            assert desc.storage in host_storages, (
                f"'{name}' storage is {desc.storage}, expected Default or CPU_Heap"
            )

    def test_gpu_clones_are_gpu_global_transients(self) -> None:
        """GPU-side cloned arrays exist, are transient, and have
        GPU_Global storage."""
        sdfg = _build_vadd_sdfg("dc_struct_gpu_clones")
        _lower_full_pipeline(sdfg)

        gpu_names = _gpu_clone_names(sdfg)
        assert len(gpu_names) >= 3, f"Expected at least 3 GPU clones, found {gpu_names}"
        for gpu_name in gpu_names:
            desc = sdfg.arrays[gpu_name]
            assert desc.transient, f"'{gpu_name}' is not transient"
            assert desc.storage == dtypes.StorageType.GPU_Global, (
                f"'{gpu_name}' storage is {desc.storage}, expected GPU_Global"
            )

    def test_scalars_not_cloned(self) -> None:
        """Scalar parameters are NOT cloned by the pipeline."""
        sdfg = _build_vadd_with_scalar_sdfg("dc_struct_scalar_skip")
        _lower_full_pipeline(sdfg)

        # The scalar should still exist unchanged
        assert "alpha" in sdfg.arrays
        assert isinstance(sdfg.arrays["alpha"], data.Scalar)
        # No GPU clone of a scalar should exist
        for name, desc in sdfg.arrays.items():
            if desc.transient and desc.storage == dtypes.StorageType.GPU_Global:
                assert not isinstance(desc, data.Scalar), (
                    f"Scalar '{name}' was cloned to GPU, but scalars should not be cloned"
                )

    def test_2d_vadd_structure(self) -> None:
        """2D vadd with widths=(8, 4) -- verify clones and copy states for
        2D arrays."""
        sdfg = _build_vadd2d_sdfg("dc_struct_2d")
        _lower_full_pipeline(sdfg, widths=(8, 4))

        # Verify originals are host-side (Default or CPU_Heap)
        host_storages = {dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap}
        for name in ("A", "B", "C"):
            assert sdfg.arrays[name].storage in host_storages

        # Verify GPU clones exist
        gpu_names = _gpu_clone_names(sdfg)
        assert len(gpu_names) >= 3

        # Verify copyin and copyout states
        assert _has_copyin_state(sdfg)
        assert _has_copyout_state(sdfg)


# ============================================================
# Codegen tests (no GPU)
# ============================================================


class TestCuTileDataCopiesCodegen:
    """Generated Python-backend code with data copies."""

    def test_codegen_contains_set(self) -> None:
        """After full pipeline, generated code contains ``.set()``."""
        sdfg = _build_vadd_sdfg("dc_codegen_set")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        assert ".set(" in code, (
            "Generated code does not contain '.set('"
        )

    def test_codegen_contains_get_out(self) -> None:
        """After full pipeline, generated code contains ``.get(out=...)``."""
        sdfg = _build_vadd_sdfg("dc_codegen_get_out")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        assert ".get(out=" in code, (
            "Generated code does not contain '.get(out='"
        )

    def test_codegen_valid_python(self) -> None:
        """Generated code parses with ``ast.parse``."""
        sdfg = _build_vadd_sdfg("dc_codegen_valid")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        ast.parse(code)

    def test_codegen_2d_valid_python(self) -> None:
        """2D vadd with data copies generates valid Python code."""
        sdfg = _build_vadd2d_sdfg("dc_codegen_2d_valid")
        VectorizeCuTile(widths=(8, 4)).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        ast.parse(code)
        assert ".set(" in code
        assert ".get(out=" in code

    def test_codegen_with_scalar_valid_python(self) -> None:
        """SDFG with a scalar parameter generates valid Python code with
        data copies (scalar is not cloned)."""
        sdfg = _build_vadd_with_scalar_sdfg("dc_codegen_scalar")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        ast.parse(code)


# ============================================================
# Runtime tests (GPU)
# ============================================================


@pytest.mark.gpu
class TestCuTileDataCopiesRuntime:
    """Runtime tests: the key feature is passing NumPy arrays directly."""

    def test_vadd_numpy_arrays_directly(self) -> None:
        """Build vadd SDFG, apply VectorizeCuTile, compile, call with
        NUMPY arrays, verify result matches A + B.

        This is the KEY test: after data copies, callers pass host arrays.
        """
        sdfg = _build_vadd_sdfg("dc_rt_vadd_numpy")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        n = 64
        rng = np.random.default_rng(42)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C, N=n)

        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_vadd_non_divisible_numpy(self) -> None:
        """N=17 (non-divisible by 8) with NumPy arrays directly."""
        sdfg = _build_vadd_sdfg("dc_rt_vadd_nondiv")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        n = 17
        rng = np.random.default_rng(43)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C, N=n)

        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_vadd_2d_numpy(self) -> None:
        """2D vadd with widths=(8, 4), pass numpy arrays directly."""
        sdfg = _build_vadd2d_sdfg("dc_rt_vadd_2d")
        VectorizeCuTile(widths=(8, 4)).apply_pass(sdfg, {})

        m, n = 10, 17
        rng = np.random.default_rng(44)
        A = rng.random((m, n))
        B = rng.random((m, n))
        C = np.zeros((m, n))

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C, M=m, N=n)

        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    @pytest.mark.skip(
        reason="apply_gpu_transformations() stages scalars as constants "
        "(alpha_const) which the cuTile runtime cannot resolve. Known "
        "limitation of the GPU-transform-based pipeline with scalars."
    )
    def test_vadd_with_scalar_param_numpy(self) -> None:
        """An SDFG with a scalar parameter. Verify scalar is passed through
        correctly with numpy arrays."""
        sdfg = _build_vadd_with_scalar_sdfg("dc_rt_vadd_scalar")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        n = 32
        rng = np.random.default_rng(45)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        alpha_val = np.float64(3.14)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C, alpha=alpha_val, N=n)

        np.testing.assert_allclose(C, A + B + alpha_val, rtol=1e-14)

    def test_vadd_large_numpy(self) -> None:
        """Larger problem size (N=1000) to exercise multiple tile iterations."""
        sdfg = _build_vadd_sdfg("dc_rt_vadd_large")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        n = 1000
        rng = np.random.default_rng(47)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C, N=n)

        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_vadd_float32_numpy(self) -> None:
        """float32 dtype with numpy arrays directly."""
        sdfg = _build_vadd_sdfg("dc_rt_vadd_f32", dtype=dace.float32)
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        n = 100
        rng = np.random.default_rng(48)
        A = rng.random(n).astype(np.float32)
        B = rng.random(n).astype(np.float32)
        C = np.zeros(n, dtype=np.float32)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C, N=n)

        np.testing.assert_allclose(C, A + B, rtol=1e-6)

    def test_symbolic_n_two_sizes_numpy(self) -> None:
        """One compiled SDFG, two runtime values of the symbol N, numpy arrays."""
        sdfg = _build_vadd_sdfg("dc_rt_vadd_symbolic")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})
        csdfg = sdfg.compile()

        rng = np.random.default_rng(49)
        for n in (64, 17):
            A = rng.random(n)
            B = rng.random(n)
            C = np.zeros(n)
            csdfg(A=A, B=B, C=C, N=n)
            np.testing.assert_allclose(C, A + B, rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
