from dace.transformation.passes.vectorization.enums import ISA
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end integration tests for the cuTile expansion + Python backend pipeline.

Each test creates an SDFG (via API or @dace.program), applies the
``VectorizeCuTile`` orchestrator (vectorize with target_isa=ISA.CUTILE, lower
schedules/storage/implementations, expand library nodes, select the Python
backend), generates Python code, compiles, runs on GPU, and compares results
against a NumPy reference. The structural tests double as regression tests of
the cuTile lowering passes (``cutile_lowering.py``).
"""
import ast

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile

# All GPU execution tests require GPU
pytestmark = pytest.mark.gpu


# ============================================================
# Pipeline helpers
# ============================================================


def _apply_cutile_pipeline(sdfg: SDFG, widths=(8, )) -> None:
    """Apply the full cuTile pipeline (the ``VectorizeCuTile`` orchestrator).

    :param sdfg: The SDFG to transform (modified in-place).
    :param widths: Tile widths for vectorization (must be powers of two).
    """
    VectorizeCuTile(widths=widths).apply_pass(sdfg, {})


def _run_cutile(sdfg, **kwargs):
    """Compile and run a cuTile SDFG, returning results as numpy arrays.

    With the new pipeline, ``VectorizeCuTile`` always includes data copies
    via ``apply_gpu_transformations()``, so the SDFG expects NumPy (host)
    arrays and handles device transfer internally.

    :param sdfg: The SDFG to compile and run.
    :param kwargs: Named arguments for the SDFG (arrays and symbols).
    :returns: Dictionary mapping array names to numpy results.
    """
    csdfg = sdfg.compile()
    csdfg(**kwargs)
    return dict(kwargs)


def _build_vadd_sdfg(name, dtype=dace.float64):
    """Build a symbolic-sized C[i] = A[i] + B[i] SDFG.

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


def _build_binop_sdfg(name, op, dtype=dace.float64):
    """Build a symbolic-sized C[i] = A[i] <op> B[i] SDFG.

    :param name: SDFG name (must be unique per test).
    :param op: Binary operator string (+, -, *, /).
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
        "binop",
        {"i": "0:N"},
        {"_a": dace.Memlet("A[i]"), "_b": dace.Memlet("B[i]")},
        f"_c = _a {op} _b",
        {"_c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    return sdfg


# ============================================================
# 1D Tests: Basic Operations
# ============================================================


class TestBasicOps:
    """Basic 1D element-wise operations."""

    def test_vadd_aligned(self):
        """C[i] = A[i] + B[i] with N aligned to tile width."""
        sdfg = _build_vadd_sdfg("cutile_vadd_aligned")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        n = 64
        rng = np.random.default_rng(42)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-14)

    def test_vadd_unaligned(self):
        """C[i] = A[i] + B[i] with N NOT aligned to tile width (tests masking)."""
        sdfg = _build_vadd_sdfg("cutile_vadd_unaligned")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        n = 17  # Not aligned to 8
        rng = np.random.default_rng(43)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-14)

    def test_vsub(self):
        """C[i] = A[i] - B[i]."""
        sdfg = _build_binop_sdfg("cutile_vsub", "-")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        n = 100
        rng = np.random.default_rng(44)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A - B, rtol=1e-14)

    def test_vmul(self):
        """C[i] = A[i] * B[i]."""
        sdfg = _build_binop_sdfg("cutile_vmul", "*")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        n = 100
        rng = np.random.default_rng(45)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A * B, rtol=1e-14)

    def test_vdiv(self):
        """C[i] = A[i] / B[i]."""
        sdfg = _build_binop_sdfg("cutile_vdiv", "/")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        n = 100
        rng = np.random.default_rng(46)
        A = rng.random(n)
        B = rng.random(n) + 0.1  # Avoid division by zero
        C = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A / B, rtol=1e-14)


# ============================================================
# 1D Tests: Data Types
# ============================================================


class TestDtypes:
    """Test different data types."""

    def test_float32(self):
        """float32 vadd."""
        sdfg = _build_vadd_sdfg("cutile_f32_add", dtype=dace.float32)
        _apply_cutile_pipeline(sdfg, widths=(8,))

        n = 100
        rng = np.random.default_rng(50)
        A = rng.random(n).astype(np.float32)
        B = rng.random(n).astype(np.float32)
        C = np.zeros(n, dtype=np.float32)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-6)

    def test_int32(self):
        """int32 vadd."""
        sdfg = _build_vadd_sdfg("cutile_i32_add", dtype=dace.int32)
        _apply_cutile_pipeline(sdfg, widths=(8,))

        n = 100
        rng = np.random.default_rng(51)
        A = rng.integers(0, 1000, n, dtype=np.int32)
        B = rng.integers(0, 1000, n, dtype=np.int32)
        C = np.zeros(n, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_array_equal(results["C"], A + B)

    def test_float64(self):
        """float64 vadd (the default; explicit test for coverage)."""
        sdfg = _build_vadd_sdfg("cutile_f64_add_explicit", dtype=dace.float64)
        _apply_cutile_pipeline(sdfg, widths=(8,))

        n = 100
        rng = np.random.default_rng(52)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-14)


# ============================================================
# 1D Tests: Various Array Sizes
# ============================================================


class TestAlignedSizes:
    """Aligned sizes (multiples of tile width)."""

    @pytest.mark.parametrize("n", [8, 16, 32, 64, 128, 256])
    def test_aligned_sizes(self, n):
        """Aligned sizes (multiples of 8)."""
        sdfg = _build_vadd_sdfg(f"cutile_aligned_{n}")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        rng = np.random.default_rng(n)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-14)


class TestUnalignedSizes:
    """Unaligned sizes (not multiples of tile width) -- tests boundary masking."""

    @pytest.mark.parametrize(
        "n", [1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 100, 127]
    )
    def test_unaligned_sizes(self, n):
        """Unaligned sizes -- tests boundary masking."""
        sdfg = _build_vadd_sdfg(f"cutile_unaligned_{n}")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        rng = np.random.default_rng(n + 100)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-14)


# ============================================================
# 1D Tests: Tile Widths
# ============================================================


class TestTileWidths:
    """Test different power-of-2 tile widths."""

    @pytest.mark.parametrize("w", [4, 8, 16, 32])
    def test_tile_widths_aligned(self, w):
        """Different tile widths with aligned size."""
        sdfg = _build_vadd_sdfg(f"cutile_w{w}_aligned")
        _apply_cutile_pipeline(sdfg, widths=(w,))

        n = 128  # Aligned for all tile widths 4..32
        rng = np.random.default_rng(w)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-14)

    @pytest.mark.parametrize("w", [4, 8, 16, 32])
    def test_tile_widths_unaligned(self, w):
        """Different tile widths with unaligned size."""
        sdfg = _build_vadd_sdfg(f"cutile_w{w}_unaligned")
        _apply_cutile_pipeline(sdfg, widths=(w,))

        n = 100  # Not aligned to any of 4, 8, 16, 32
        rng = np.random.default_rng(w + 100)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-14)


# ============================================================
# Combined operator + size stress test
# ============================================================


class TestOpSizeCombinations:
    """Cross-product of operators and sizes for broader coverage."""

    @pytest.mark.parametrize("op", ["+", "-", "*"])
    @pytest.mark.parametrize("n", [1, 17, 64])
    def test_op_size_cross(self, op, n):
        """Various operators with various sizes."""
        op_names = {"+": "add", "-": "sub", "*": "mul"}
        sdfg = _build_binop_sdfg(
            f"cutile_{op_names[op]}_n{n}", op
        )
        _apply_cutile_pipeline(sdfg, widths=(8,))

        rng = np.random.default_rng(hash((op, n)) & 0xFFFF_FFFF)
        A = rng.random(n) + 0.1
        B = rng.random(n) + 0.1
        C = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)

        if op == "+":
            expected = A + B
        elif op == "-":
            expected = A - B
        else:
            expected = A * B
        np.testing.assert_allclose(results["C"], expected, rtol=1e-14)


# ============================================================
# Codegen-Only Tests (no GPU required)
# ============================================================


class TestCodegenOnly:
    """Tests that verify codegen output structure.

    These tests exercise the full pipeline through code generation but do not
    compile or execute the SDFG. They verify the generated Python code is
    syntactically valid and contains the expected cuTile primitives.
    """

    def test_vadd_generates_valid_python(self):
        """Generated code should be valid Python (parseable by ast.parse)."""
        sdfg = _build_vadd_sdfg("cutile_codegen_valid")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        code_objects = sdfg.generate_code()
        main_code = next(co for co in code_objects if co.name == sdfg.name)
        # Should be valid Python
        ast.parse(main_code.code)

    def test_vadd_contains_ct_primitives(self):
        """Generated code should contain ct.load, ct.scatter, ct.bid."""
        sdfg = _build_vadd_sdfg("cutile_codegen_prims")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code

        assert "ct.bid(0)" in code, "Missing ct.bid(0) in generated code"
        assert "ct.load(" in code, "Missing ct.load in generated code"
        assert "ct.scatter(" in code, "Missing ct.scatter in generated code"
        assert "ct.arange(" in code, "Missing ct.arange in generated code"
        assert "@ct.kernel" in code, "Missing @ct.kernel decorator"
        assert "ct.launch(" in code, "Missing ct.launch call"

    def test_vadd_has_import_cuda_tile(self):
        """Generated code should import cuda.tile."""
        sdfg = _build_vadd_sdfg("cutile_codegen_import")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "import cuda.tile as ct" in code

    def test_vadd_has_kernel_function(self):
        """Generated code should contain a @ct.kernel decorated function."""
        sdfg = _build_vadd_sdfg("cutile_codegen_kernel")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code

        # The kernel decorator must appear
        assert "@ct.kernel" in code
        # The kernel function definition must follow the decorator
        kernel_idx = code.index("@ct.kernel")
        # There should be a 'def ' after the decorator
        def_idx = code.index("def ", kernel_idx)
        assert def_idx > kernel_idx

    def test_sub_generates_valid_python(self):
        """Subtraction variant should also produce valid Python."""
        sdfg = _build_binop_sdfg("cutile_codegen_sub", "-")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        code_objects = sdfg.generate_code()
        main_code = next(co for co in code_objects if co.name == sdfg.name)
        ast.parse(main_code.code)

    def test_mul_generates_valid_python(self):
        """Multiplication variant should produce valid Python."""
        sdfg = _build_binop_sdfg("cutile_codegen_mul", "*")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        code_objects = sdfg.generate_code()
        main_code = next(co for co in code_objects if co.name == sdfg.name)
        ast.parse(main_code.code)

    def test_float32_generates_valid_python(self):
        """float32 SDFG should produce valid Python."""
        sdfg = _build_vadd_sdfg("cutile_codegen_f32", dtype=dace.float32)
        _apply_cutile_pipeline(sdfg, widths=(8,))

        code_objects = sdfg.generate_code()
        main_code = next(co for co in code_objects if co.name == sdfg.name)
        ast.parse(main_code.code)

    def test_tile_width_16_generates_valid_python(self):
        """Tile width 16 should produce valid Python."""
        sdfg = _build_vadd_sdfg("cutile_codegen_w16")
        _apply_cutile_pipeline(sdfg, widths=(16,))

        code_objects = sdfg.generate_code()
        main_code = next(co for co in code_objects if co.name == sdfg.name)
        ast.parse(main_code.code)

    def test_mask_gen_present_in_code(self):
        """The iteration mask generation code should be in generated output."""
        sdfg = _build_vadd_sdfg("cutile_codegen_maskgen")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code

        # The mask generation uses arange + comparison against upper bound
        assert "ct.arange(8" in code, "Missing arange(8) for mask gen"
        # The mask comparison should reference the symbol N
        assert "< N" in code or "< N)" in code, "Missing N upper bound in mask"

    def test_cupy_import_present(self):
        """Generated code should import cupy for stream management."""
        sdfg = _build_vadd_sdfg("cutile_codegen_cupy")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "import cupy" in code


# ============================================================
# Pipeline structure tests (no GPU required)
# ============================================================


class TestPipelineStructure:
    """Verify the pipeline transforms produce the expected SDFG structure."""

    def test_map_schedule_is_cutile(self):
        """After pipeline, the tiled map should have CuTile schedule."""
        sdfg = _build_vadd_sdfg("cutile_struct_schedule")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        found_cutile_map = False
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry):
                    if node.map.schedule == dtypes.ScheduleType.CuTile:
                        found_cutile_map = True
        assert found_cutile_map, "No CuTile-scheduled map found after pipeline"

    def test_global_arrays_have_gpu_clones(self):
        """After pipeline, original non-transient arrays have host storage
        (Default or CPU_Heap) with GPU_Global transient clones."""
        sdfg = _build_vadd_sdfg("cutile_struct_storage")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        host_storages = {dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap}
        for name in ("A", "B", "C"):
            assert sdfg.arrays[name].storage in host_storages, (
                f"Array {name} has storage {sdfg.arrays[name].storage}, "
                f"expected Default or CPU_Heap"
            )
        # GPU_Global transient clones exist
        gpu_clones = {
            name for name, desc in sdfg.arrays.items()
            if desc.storage == dtypes.StorageType.GPU_Global and desc.transient
        }
        assert len(gpu_clones) >= 3, f"Expected at least 3 GPU clones, found {gpu_clones}"

    def test_tile_transients_are_cutile_tile(self):
        """After pipeline, tile transients should have CuTile_Tile storage."""
        sdfg = _build_vadd_sdfg("cutile_struct_tile_storage")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        found_cutile_tile = False
        for name, desc in sdfg.arrays.items():
            if desc.transient and desc.storage == dtypes.StorageType.CuTile_Tile:
                found_cutile_tile = True
        assert found_cutile_tile, "No CuTile_Tile transient found after pipeline"

    def test_backend_is_python(self):
        """After pipeline, backend should be Python."""
        sdfg = _build_vadd_sdfg("cutile_struct_backend")
        _apply_cutile_pipeline(sdfg, widths=(8,))
        assert sdfg.backend == dtypes.BackendLanguage.Python

    def test_expanded_tasklets_are_python(self):
        """All tasklets in the expanded SDFG should be Python language."""
        sdfg = _build_vadd_sdfg("cutile_struct_python_tasklets")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.Tasklet):
                    assert node.language == dtypes.Language.Python, (
                        f"Tasklet {node.label} has language {node.language}, "
                        f"expected Python"
                    )

    def test_tiled_map_has_stride(self):
        """The tiled map should have a step > 1 matching the tile width."""
        sdfg = _build_vadd_sdfg("cutile_struct_stride")
        _apply_cutile_pipeline(sdfg, widths=(8,))

        found_strided = False
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry):
                    for _, _, step in node.map.range:
                        if str(step) == "8":
                            found_strided = True
        assert found_strided, "No map with stride 8 found after pipeline"


# ============================================================
# Multi-dim (K=2) data-dependent gather + outer block loop
# ============================================================


class TestMultiDimGather:
    """End-to-end K=2 structured gather with an OUTER non-tiled loop.

    Mirrors the ICON ``velocity_zekinh`` cell-from-edges interpolation: a
    3-edge data-dependent gather ``z[edge_blk[jb,jc,e], jk, edge_idx[jb,jc,e]]``
    weighted by per-edge constants ``e_bln[jb,e,jc]``, with the outer block
    loop ``jb`` left UNTILED. This exercises, together:

    * ``ct.gather`` with per-source-dim ``_idx_<d>`` index tiles whose
      ``(ONE, W)`` descriptor is honoured (index-tile reshape);
    * the grid-dim offset so the body's tile block ids skip the outer ``jb``
      grid axis;
    * non-tile dims indexed by their memlet begin (``jb`` / constant edge
      index) instead of a hard ``0``;
    * stride-0 broadcast of the constant ``e_bln`` weights across ``jk``.
    """

    @staticmethod
    def _build():
        NB = dace.symbol("NB")
        NLEV = dace.symbol("NLEV")
        NPROMA = dace.symbol("NPROMA")

        @dace.program
        def icon_zekinh_gather(e_bln: dace.float64[(NB * 8), 3, (NPROMA * 8)],
                               edge_idx: dace.int32[(NB * 8), (NPROMA * 8), 3],
                               edge_blk: dace.int32[(NB * 8), (NPROMA * 8), 3],
                               z_kin_hor_e: dace.float64[(NB * 8), (NLEV * 8), (NPROMA * 8)],
                               z_ekinh: dace.float64[(NB * 8), (NLEV * 8), (NPROMA * 8)]):
            for jb in range((NB * 8)):
                for jk in range((NLEV * 8)):
                    for jc in range((NPROMA * 8)):
                        z_ekinh[jb, jk, jc] = (
                            e_bln[jb, 0, jc] * z_kin_hor_e[edge_blk[jb, jc, 0], jk, edge_idx[jb, jc, 0]] +
                            e_bln[jb, 1, jc] * z_kin_hor_e[edge_blk[jb, jc, 1], jk, edge_idx[jb, jc, 1]] +
                            e_bln[jb, 2, jc] * z_kin_hor_e[edge_blk[jb, jc, 2], jk, edge_idx[jb, jc, 2]])

        return icon_zekinh_gather.to_sdfg()

    @staticmethod
    def _reference(e_bln, edge_idx, edge_blk, z):
        NB8, NLEV8, NPROMA8 = z.shape
        out = np.zeros((NB8, NLEV8, NPROMA8))
        for jb in range(NB8):
            for jk in range(NLEV8):
                for jc in range(NPROMA8):
                    out[jb, jk, jc] = sum(e_bln[jb, e, jc] * z[edge_blk[jb, jc, e], jk, edge_idx[jb, jc, e]]
                                          for e in range(3))
        return out

    @pytest.mark.parametrize("NB_val", [1, 2])
    def test_zekinh_gather_matches_numpy(self, NB_val):
        sdfg = self._build()
        VectorizeCuTile(widths=(8, 8), branch_mode="merge").apply_pass(sdfg, {})

        NLEV_val, NPROMA_val = 2, 2
        NB8, NLEV8, NPROMA8 = NB_val * 8, NLEV_val * 8, NPROMA_val * 8
        rng = np.random.default_rng(0)
        e_bln = rng.standard_normal((NB8, 3, NPROMA8))
        edge_idx = rng.integers(0, NPROMA8, size=(NB8, NPROMA8, 3)).astype(np.int32)
        edge_blk = rng.integers(0, NB8, size=(NB8, NPROMA8, 3)).astype(np.int32)
        z = rng.standard_normal((NB8, NLEV8, NPROMA8))
        ref = self._reference(e_bln, edge_idx, edge_blk, z)

        results = _run_cutile(sdfg, e_bln=e_bln, edge_idx=edge_idx, edge_blk=edge_blk,
                              z_kin_hor_e=z, z_ekinh=np.zeros((NB8, NLEV8, NPROMA8)),
                              NB=NB_val, NLEV=NLEV_val, NPROMA=NPROMA_val)
        np.testing.assert_allclose(results["z_ekinh"], ref, rtol=1e-12, atol=1e-12)

    def test_gather_code_is_pure_cutile(self):
        """No C++/scalar host index reads survive: the gather lowers to
        ``ct.gather`` + ``ct.load`` index tiles, no ``std::`` / for-loops."""
        sdfg = self._build()
        VectorizeCuTile(widths=(8, 8), branch_mode="merge").apply_pass(sdfg, {})
        code = "".join(c.clean_code for c in sdfg.generate_code())
        assert "ct.gather" in code
        assert "std::" not in code and "for (" not in code

    @pytest.mark.parametrize("NB_val", [1, 2])
    def test_gather_indexed_by_non_innermost_dim(self, NB_val):
        """Gather whose index tile walks the MIDDLE tiled loop (``jk``), not the
        innermost (``jc``). The index ``TileGather`` is K=1 on a non-trailing tile
        dim, so the per-lane block id must resolve ``jk``'s grid axis -- the
        positional trailing-K offset would read the wrong ``ct.bid`` axis."""
        NB = dace.symbol("NB")
        NLEV = dace.symbol("NLEV")
        NPROMA = dace.symbol("NPROMA")

        @dace.program
        def gather_by_row(row_idx: dace.int32[(NB * 8), (NLEV * 8)],
                          src: dace.float64[(NB * 8), (NLEV * 8), (NPROMA * 8)],
                          out: dace.float64[(NB * 8), (NLEV * 8), (NPROMA * 8)]):
            for jb in range((NB * 8)):
                for jk in range((NLEV * 8)):
                    for jc in range((NPROMA * 8)):
                        out[jb, jk, jc] = src[jb, row_idx[jb, jk], jc]

        sdfg = gather_by_row.to_sdfg()
        VectorizeCuTile(widths=(8, 8), branch_mode="merge").apply_pass(sdfg, {})

        NLEV_val, NPROMA_val = 2, 2
        NB8, NLEV8, NPROMA8 = NB_val * 8, NLEV_val * 8, NPROMA_val * 8
        rng = np.random.default_rng(7)
        row_idx = rng.integers(0, NLEV8, size=(NB8, NLEV8)).astype(np.int32)
        src = rng.standard_normal((NB8, NLEV8, NPROMA8))
        ref = np.zeros((NB8, NLEV8, NPROMA8))
        for jb in range(NB8):
            for jk in range(NLEV8):
                for jc in range(NPROMA8):
                    ref[jb, jk, jc] = src[jb, row_idx[jb, jk], jc]

        results = _run_cutile(sdfg, row_idx=row_idx, src=src, out=np.zeros((NB8, NLEV8, NPROMA8)),
                              NB=NB_val, NLEV=NLEV_val, NPROMA=NPROMA_val)
        np.testing.assert_allclose(results["out"], ref, rtol=1e-12, atol=1e-12)


# ============================================================
# Scalar-parameter kernels (regression: bug 03)
# ============================================================


class TestScalarParams:
    """Runtime scalar parameters threaded into the cuTile kernel.

    Regression tests for bug 03: a loop-invariant scalar (``alpha``) used
    inside a tiled elementwise op is staged as a ``Register`` scalar bridge
    (``alpha_const``) by the vectorizer. The cuTile codegen must emit the
    rename ``alpha_const = alpha`` so the kernel body can read it; otherwise
    the ``cuda.tile`` compiler raised ``Undefined variable alpha_const``.
    """

    @pytest.mark.parametrize("n", [64, 17])
    @pytest.mark.parametrize("alpha_val", [1.5, -2.0, 0.0])
    def test_scale_by_scalar(self, n, alpha_val):
        """Y[i] = alpha * X[i] with a runtime scalar, aligned and unaligned N."""
        N = dace.symbol("N")

        @dace.program
        def cutile_scale(alpha: dace.float64, X: dace.float64[N], Y: dace.float64[N]):
            Y[:] = alpha * X[:]

        sdfg = cutile_scale.to_sdfg()
        _apply_cutile_pipeline(sdfg, widths=(8,))

        rng = np.random.default_rng(101)
        X = rng.random(n)
        Y = np.zeros(n)
        results = _run_cutile(sdfg, alpha=alpha_val, X=X, Y=Y, N=n)
        np.testing.assert_allclose(results["Y"], alpha_val * X, rtol=1e-14)

    def test_two_scalars_axpy(self):
        """Z[i] = alpha * X[i] + beta * Y[i] -- two runtime scalar bridges."""
        N = dace.symbol("N")

        @dace.program
        def cutile_axpby(alpha: dace.float64, beta: dace.float64, X: dace.float64[N], Y: dace.float64[N],
                         Z: dace.float64[N]):
            Z[:] = alpha * X[:] + beta * Y[:]

        sdfg = cutile_axpby.to_sdfg()
        _apply_cutile_pipeline(sdfg, widths=(8,))

        n = 100  # unaligned to 8
        alpha_val, beta_val = 2.5, -0.75
        rng = np.random.default_rng(102)
        X = rng.random(n)
        Y = rng.random(n)
        Z = np.zeros(n)
        results = _run_cutile(sdfg, alpha=alpha_val, beta=beta_val, X=X, Y=Y, Z=Z, N=n)
        np.testing.assert_allclose(results["Z"], alpha_val * X + beta_val * Y, rtol=1e-14)


# ============================================================
# Entry point
# ============================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=300"])
