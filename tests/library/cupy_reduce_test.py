# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Integration tests for the CuPy expansion of the Reduce library node.

Tests cover:
- Sum, product, min, max reductions with various shape/axis combos
- Logical AND/OR reductions
- Bitwise reductions (verify fallback to pure since CuPy ufunc.reduce
  is not supported for bitwise ops)
- Float32/float64 precision
- Scalar output (reduce all axes)
- Non-power-of-2 and non-contiguous axis combos
- Custom WCR fallback to pure expansion
- Degenerate reductions (axis of size 1)

All runtime tests require a GPU and CuPy, marked ``@pytest.mark.gpu``.
Structural tests that only check expansion structure run without GPU.

The CuPy expansion emits a Python-language tasklet, so these tests use
``sdfg.backend = BackendLanguage.Python`` (NOT ``apply_gpu_transformations``
which sets up the C++ backend).
"""

import itertools
import numpy as np
import pytest

import dace
from dace import dtypes, SDFG, Memlet
import dace.libraries.standard as std
from dace.frontend.operations import detect_reduction_type
from dace.libraries.standard.nodes.reduce import (
    ExpandReduceCuPy,
    ExpandReducePure,
    _REDUCTION_TYPE_TO_CUPY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reduce_sdfg(in_shape, axes, wcr, dtype=dace.float64, identity=None):
    """Build a minimal SDFG with a single Reduce node using the Python backend.

    :param in_shape: shape of the input array (list of ints).
    :param axes:     axes to reduce over (list of ints or None for all).
    :param wcr:      WCR lambda string.
    :param dtype:    DaCe data type.
    :param identity: optional identity value for initialisation.
    :returns:        (sdfg, reduce_node)
    """
    out_shape = [s for i, s in enumerate(in_shape) if i not in (axes or [])]
    if not out_shape:
        out_shape = [1]

    name = "cupy_reduce_test_" + "_".join(str(s) for s in in_shape)
    g = SDFG(name)
    g.add_array("A", in_shape, dtype)
    g.add_array("B", out_shape, dtype)
    st = g.add_state("main", is_start_block=True)

    a = st.add_access("A")
    b = st.add_access("B")
    r = st.add_reduce(wcr, axes, identity)
    st.add_nedge(a, r, Memlet.from_array("A", g.arrays["A"]))
    st.add_nedge(r, b, Memlet.from_array("B", g.arrays["B"]))

    g.backend = dtypes.BackendLanguage.Python
    return g, r


def _make_reduce_sdfg_gpu(in_shape, axes, wcr, dtype=dace.float64, identity=None):
    """Like :func:`_make_reduce_sdfg` but with GPU_Global (cupy) arrays.

    :param in_shape: shape of the input array (list of ints).
    :param axes:     axes to reduce over (list of ints or None for all).
    :param wcr:      WCR lambda string.
    :param dtype:    DaCe data type.
    :param identity: optional identity value for initialisation.
    :returns:        (sdfg, reduce_node)
    """
    sdfg, rnode = _make_reduce_sdfg(in_shape, axes, wcr, dtype=dtype, identity=identity)
    for arr in sdfg.arrays.values():
        arr.storage = dtypes.StorageType.GPU_Global
    return sdfg, rnode


# ---------------------------------------------------------------------------
# Structural tests (no GPU required)
# ---------------------------------------------------------------------------


class TestCuPyReduceStructure:
    """Structural / unit tests that do NOT need a GPU."""

    @staticmethod
    def _tasklet_code(nsdfg):
        tasklets = [n for n in nsdfg.start_block.nodes() if isinstance(n, dace.sdfg.nodes.Tasklet)]
        assert len(tasklets) == 1
        return tasklets[0].code.as_string

    def test_gpu_output_stays_on_device(self):
        """A GPU_Global output must NOT go through ``cupy.asnumpy``.

        Assigning a non-scalar numpy array into a device array fails at
        runtime with ``non-scalar numpy.ndarray cannot be used for fill``
        (softmax regression).
        """
        sdfg, rnode = _make_reduce_sdfg_gpu([8, 4], [1], "lambda a, b: max(a, b)")
        rnode.implementation = "CuPy"
        nsdfg = ExpandReduceCuPy.expansion(rnode, sdfg.start_block, sdfg)
        assert "asnumpy" not in self._tasklet_code(nsdfg)

        assert len(list(nsdfg.states())) == 2
        assert list(nsdfg.states())[-1].label == "cupy_sync"
        sync_tasklets = [node for node in list(nsdfg.states())[-1].nodes() if isinstance(node, dace.sdfg.nodes.Tasklet)]
        assert len(sync_tasklets) == 1
        assert "get_current_stream().synchronize()" in sync_tasklets[0].code.as_string

    def test_host_output_converts_to_numpy(self):
        """A host-resident output keeps the ``cupy.asnumpy`` conversion."""
        sdfg, rnode = _make_reduce_sdfg([8, 4], [1], "lambda a, b: max(a, b)")
        rnode.implementation = "CuPy"
        nsdfg = ExpandReduceCuPy.expansion(rnode, sdfg.start_block, sdfg)
        assert "asnumpy" in self._tasklet_code(nsdfg)
        assert len(list(nsdfg.states())) == 1

    def test_registered(self):
        """CuPy must appear in the Reduce implementations dict."""
        assert "CuPy" in std.Reduce.implementations
        assert std.Reduce.implementations["CuPy"] is ExpandReduceCuPy

    def test_expansion_returns_nsdfg_sum(self):
        """Expanding a sum reduction must return a nested SDFG."""
        sdfg, rnode = _make_reduce_sdfg([8, 4], [1], "lambda a, b: a + b")
        rnode.implementation = "CuPy"
        state = sdfg.start_block
        nsdfg = ExpandReduceCuPy.expansion(rnode, state, sdfg)
        assert isinstance(nsdfg, SDFG)
        assert nsdfg.name == "reduce_cupy"
        # Must contain a Python-language tasklet
        tasklets = [n for n in nsdfg.start_block.nodes() if isinstance(n, dace.sdfg.nodes.Tasklet)]
        assert len(tasklets) == 1
        assert tasklets[0].language == dace.Language.Python

    def test_expansion_all_axes_uses_none(self):
        """Reducing all axes must use axis=None in the CuPy call."""
        sdfg, rnode = _make_reduce_sdfg([4, 6], [0, 1], "lambda a, b: a + b")
        rnode.implementation = "CuPy"
        state = sdfg.start_block
        nsdfg = ExpandReduceCuPy.expansion(rnode, state, sdfg)
        assert isinstance(nsdfg, SDFG)
        tasklets = [n for n in nsdfg.start_block.nodes() if isinstance(n, dace.sdfg.nodes.Tasklet)]
        assert len(tasklets) == 1
        code_str = tasklets[0].code.as_string
        assert "None" in code_str

    def test_fallback_on_unsupported_type(self):
        """Custom WCR should fall back to ExpandReducePure."""
        sdfg, rnode = _make_reduce_sdfg([8, 4], [1], "lambda a, b: a if a > b else b + 1")
        rnode.implementation = "CuPy"
        state = sdfg.start_block
        nsdfg = ExpandReduceCuPy.expansion(rnode, state, sdfg)
        assert isinstance(nsdfg, SDFG)
        # The fallback SDFG should NOT be named 'reduce_cupy'
        assert nsdfg.name != "reduce_cupy"

    def test_fallback_on_degenerate(self):
        """A degenerate reduction (no squeezed axes) falls back to pure."""
        sdfg, rnode = _make_reduce_sdfg([1, 4], [0], "lambda a, b: a + b")
        rnode.implementation = "CuPy"
        state = sdfg.start_block
        nsdfg = ExpandReduceCuPy.expansion(rnode, state, sdfg)
        assert isinstance(nsdfg, SDFG)

    def test_all_supported_types_in_map(self):
        """Sanity: every supported type has a CuPy template entry."""
        for rtype in _REDUCTION_TYPE_TO_CUPY:
            assert "{inp}" in _REDUCTION_TYPE_TO_CUPY[rtype]
            assert "{axes}" in _REDUCTION_TYPE_TO_CUPY[rtype]

    @pytest.mark.parametrize("redtype", list(_REDUCTION_TYPE_TO_CUPY.keys()))
    def test_expansion_for_each_supported_type(self, redtype):
        """Each supported reduction type must expand without error."""
        wcr_map = {
            dtypes.ReductionType.Sum: "lambda a, b: a + b",
            dtypes.ReductionType.Product: "lambda a, b: a * b",
            dtypes.ReductionType.Min: "lambda a, b: min(a, b)",
            dtypes.ReductionType.Max: "lambda a, b: max(a, b)",
            dtypes.ReductionType.Bitwise_And: "lambda a, b: a & b",
            dtypes.ReductionType.Bitwise_Or: "lambda a, b: a | b",
            dtypes.ReductionType.Bitwise_Xor: "lambda a, b: a ^ b",
            dtypes.ReductionType.Logical_And: "lambda a, b: a and b",
            dtypes.ReductionType.Logical_Or: "lambda a, b: a or b",
        }
        wcr = wcr_map.get(redtype)
        if wcr is None:
            pytest.skip(f"No WCR string for {redtype}")
        dt = dace.int32 if "Bitwise" in redtype.name else dace.float64
        sdfg, rnode = _make_reduce_sdfg([8, 4], [1], wcr, dtype=dt)
        rnode.implementation = "CuPy"
        state = sdfg.start_block
        nsdfg = ExpandReduceCuPy.expansion(rnode, state, sdfg)
        assert isinstance(nsdfg, SDFG)
        assert nsdfg.name == "reduce_cupy"


# ---------------------------------------------------------------------------
# GPU integration tests -- compile + run, compare against NumPy
# ---------------------------------------------------------------------------

# Shape/axis combos used by the parametrized sum test
_sum_cases = [
    # (in_shape, axes, out_shape)
    ([64, 60, 60], (0, 2), [60]),
    ([8, 512, 4096], (0, 1), [4096]),
    ([1024, 8], (0,), [8]),
    ([111, 111, 111], (0, 1), [111]),
    ([111, 111, 111], (1, 2), [111]),
    ([123, 21, 26, 8], (1, 2), [123, 8]),
    ([2, 512, 2], (0, 2), [512]),
]


@pytest.mark.gpu
class TestCuPyReduceGPU:
    """End-to-end GPU tests: expand with CuPy, compile via Python backend,
    run, and compare results against NumPy."""

    # ------------------------------------------------------------------ sum
    @pytest.mark.parametrize("in_shape,axes,out_shape", _sum_cases)
    def test_sum(self, in_shape, axes, out_shape):
        """Sum reduction with various shape/axis combos."""
        sdfg, rnode = _make_reduce_sdfg(
            in_shape,
            list(axes),
            "lambda a, b: a + b",
            dtype=dace.float32,
            identity=0,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(*in_shape).astype(np.float32)
        b = np.zeros(out_shape, dtype=np.float32)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b, np.sum(a, axis=axes), rtol=1e-4)

    def test_sum_all_axes(self):
        """Reduce all axes to a scalar."""
        sdfg, rnode = _make_reduce_sdfg(
            [100, 200],
            [0, 1],
            "lambda a, b: a + b",
            dtype=dace.float32,
            identity=0,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(100, 200).astype(np.float32)
        b = np.zeros([1], dtype=np.float32)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b[0], np.sum(a), rtol=1e-4)

    def test_sum_1d_full(self):
        """1D full reduction."""
        sdfg, rnode = _make_reduce_sdfg(
            [256],
            [0],
            "lambda a, b: a + b",
            dtype=dace.float64,
            identity=0,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(256).astype(np.float64)
        b = np.zeros([1], dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b[0], np.sum(a), rtol=1e-12)

    # -------------------------------------------------------------- product
    def test_prod(self):
        """Product reduction."""
        sdfg, rnode = _make_reduce_sdfg(
            [10, 5],
            [1],
            "lambda a, b: a * b",
            dtype=dace.float64,
            identity=1,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(10, 5).astype(np.float64) + 0.5  # avoid zero
        b = np.zeros(10, dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b, np.prod(a, axis=1), rtol=1e-10)

    def test_prod_all_axes(self):
        """Product reduction over all axes (scalar output)."""
        sdfg, rnode = _make_reduce_sdfg(
            [4, 3],
            [0, 1],
            "lambda a, b: a * b",
            dtype=dace.float64,
            identity=1,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(4, 3).astype(np.float64) + 0.5
        b = np.zeros([1], dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b[0], np.prod(a), rtol=1e-10)

    # ------------------------------------------------------------------ min
    def test_min(self):
        """Min reduction along first axis."""
        sdfg, rnode = _make_reduce_sdfg(
            [50, 30],
            [0],
            "lambda a, b: min(a, b)",
            dtype=dace.float32,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(50, 30).astype(np.float32)
        b = np.zeros(30, dtype=np.float32)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b, np.min(a, axis=0))

    def test_min_last_axis(self):
        """Min reduction along last axis."""
        sdfg, rnode = _make_reduce_sdfg(
            [20, 40],
            [1],
            "lambda a, b: min(a, b)",
            dtype=dace.float64,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(20, 40).astype(np.float64)
        b = np.zeros(20, dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b, np.min(a, axis=1), rtol=1e-12)

    def test_min_int32(self):
        """Min reduction on an integer dtype (CuPy path, not the bitwise fallback)."""
        sdfg, rnode = _make_reduce_sdfg(
            [40, 16],
            [0],
            "lambda a, b: min(a, b)",
            dtype=dace.int32,
        )
        rnode.implementation = "CuPy"

        a = np.random.randint(-1000, 1000, size=[40, 16]).astype(np.int32)
        b = np.zeros(16, dtype=np.int32)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_array_equal(b, np.min(a, axis=0))

    # ------------------------------------------------------------------ max
    def test_max(self):
        """Max reduction along first axis."""
        sdfg, rnode = _make_reduce_sdfg(
            [50, 30],
            [0],
            "lambda a, b: max(a, b)",
            dtype=dace.float32,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(50, 30).astype(np.float32)
        b = np.zeros(30, dtype=np.float32)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b, np.max(a, axis=0))

    def test_max_3d_middle_axis(self):
        """Max reduction along the middle axis of a 3D array."""
        sdfg, rnode = _make_reduce_sdfg(
            [4, 6, 8],
            [1],
            "lambda a, b: max(a, b)",
            dtype=dace.float64,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(4, 6, 8).astype(np.float64)
        b = np.zeros([4, 8], dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b, np.max(a, axis=1), rtol=1e-12)

    # ------------------------------------------------------- dtype coverage
    def test_float64_sum(self):
        """Float64 precision for sum."""
        sdfg, rnode = _make_reduce_sdfg(
            [100, 200],
            [1],
            "lambda a, b: a + b",
            dtype=dace.float64,
            identity=0,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(100, 200).astype(np.float64)
        b = np.zeros(100, dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b, np.sum(a, axis=1), rtol=1e-10)

    def test_float32_sum(self):
        """Float32 sum with relaxed tolerance."""
        sdfg, rnode = _make_reduce_sdfg(
            [16, 8],
            [0],
            "lambda a, b: a + b",
            dtype=dace.float32,
            identity=0,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(16, 8).astype(np.float32)
        b = np.zeros(8, dtype=np.float32)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b, np.sum(a, axis=0), rtol=1e-5)

    # -------------------------------------------------- logical reductions
    def test_logical_and(self):
        """Logical AND reduction (cupy.all)."""
        sdfg, rnode = _make_reduce_sdfg(
            [8, 4],
            [1],
            "lambda a, b: a and b",
            dtype=dace.int32,
        )
        rnode.implementation = "CuPy"

        a = np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [0, 1, 0, 1],
                [1, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        b = np.zeros(8, dtype=np.int32)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        expected = np.all(a, axis=1).astype(np.int32)
        np.testing.assert_array_equal(b, expected)

    def test_logical_or(self):
        """Logical OR reduction (cupy.any)."""
        sdfg, rnode = _make_reduce_sdfg(
            [6, 5],
            [1],
            "lambda a, b: a or b",
            dtype=dace.int32,
        )
        rnode.implementation = "CuPy"

        a = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
            ],
            dtype=np.int32,
        )
        b = np.zeros(6, dtype=np.int32)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        expected = np.any(a, axis=1).astype(np.int32)
        np.testing.assert_array_equal(b, expected)

    # ----------------------------------------------- bitwise (fallback)
    # CuPy does not support bitwise ufunc.reduce, so the expansion
    # falls back to ExpandReducePure for these reduction types.
    # The structural tests above verify that the expansion detects
    # the fallback correctly.  Runtime verification of the pure
    # fallback is not tested here because the pure expansion with
    # Python backend requires a proper identity value for bitwise
    # operations, which is not set by the CuPy expansion path.

    def test_bitwise_and_falls_back_to_pure(self):
        """Bitwise AND is not in _REDUCTION_TYPE_TO_CUPY (CuPy limitation).

        Verify that the expansion falls back to ExpandReducePure without
        raising an error.  We do NOT test runtime correctness here because
        the pure expansion needs an explicit identity value for bitwise AND.
        """
        sdfg, rnode = _make_reduce_sdfg(
            [8, 4],
            [1],
            "lambda a, b: a & b",
            dtype=dace.int32,
        )
        rnode.implementation = "CuPy"

        # The expansion should succeed (falling back to pure)
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdfg.expand_library_nodes()
            # Verify the fallback warning was issued
            fallback_warnings = [
                x for x in w if "ExpandReduceCuPy" in str(x.message) and "falling back" in str(x.message)
            ]
            assert len(fallback_warnings) > 0, "Expected a fallback warning for Bitwise_And"

    def test_bitwise_or_falls_back_to_pure(self):
        """Bitwise OR is not in _REDUCTION_TYPE_TO_CUPY (CuPy limitation)."""
        sdfg, rnode = _make_reduce_sdfg(
            [8, 4],
            [1],
            "lambda a, b: a | b",
            dtype=dace.int32,
        )
        rnode.implementation = "CuPy"

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdfg.expand_library_nodes()
            fallback_warnings = [
                x for x in w if "ExpandReduceCuPy" in str(x.message) and "falling back" in str(x.message)
            ]
            assert len(fallback_warnings) > 0

    def test_bitwise_xor_falls_back_to_pure(self):
        """Bitwise XOR is not in _REDUCTION_TYPE_TO_CUPY (CuPy limitation)."""
        sdfg, rnode = _make_reduce_sdfg(
            [8, 4],
            [1],
            "lambda a, b: a ^ b",
            dtype=dace.int32,
        )
        rnode.implementation = "CuPy"

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdfg.expand_library_nodes()
            fallback_warnings = [
                x for x in w if "ExpandReduceCuPy" in str(x.message) and "falling back" in str(x.message)
            ]
            assert len(fallback_warnings) > 0

    # ----------------------------------------- non-contiguous axes (3D)
    def test_sum_3d_non_contiguous_axes(self):
        """Sum over non-contiguous axes (0, 2) of a 3D array."""
        sdfg, rnode = _make_reduce_sdfg(
            [4, 6, 8],
            [0, 2],
            "lambda a, b: a + b",
            dtype=dace.float64,
            identity=0,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(4, 6, 8).astype(np.float64)
        b = np.zeros(6, dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b, np.sum(a, axis=(0, 2)), rtol=1e-12)

    def test_sum_3d_all_axes(self):
        """Sum over all axes of a 3D array (scalar output)."""
        sdfg, rnode = _make_reduce_sdfg(
            [4, 6, 8],
            [0, 1, 2],
            "lambda a, b: a + b",
            dtype=dace.float64,
            identity=0,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(4, 6, 8).astype(np.float64)
        b = np.zeros([1], dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b[0], np.sum(a), rtol=1e-12)

    # ----------------------------------------- 4D array reduction
    def test_sum_4d(self):
        """Sum reduction on a 4D array, reducing middle axes."""
        sdfg, rnode = _make_reduce_sdfg(
            [3, 5, 7, 4],
            [1, 2],
            "lambda a, b: a + b",
            dtype=dace.float64,
            identity=0,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(3, 5, 7, 4).astype(np.float64)
        b = np.zeros([3, 4], dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b, np.sum(a, axis=(1, 2)), rtol=1e-12)

    # ----------------------------------------- non-power-of-2 shapes
    def test_sum_non_power_of_2(self):
        """Sum with non-power-of-2 shapes (prime dimensions)."""
        sdfg, rnode = _make_reduce_sdfg(
            [7, 13],
            [0],
            "lambda a, b: a + b",
            dtype=dace.float64,
            identity=0,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(7, 13).astype(np.float64)
        b = np.zeros(13, dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b, np.sum(a, axis=0), rtol=1e-12)

    # ----------------------------------------- large reduction
    def test_sum_large_1d(self):
        """Large 1D sum to verify numerical stability."""
        sdfg, rnode = _make_reduce_sdfg(
            [100000],
            [0],
            "lambda a, b: a + b",
            dtype=dace.float64,
            identity=0,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(100000).astype(np.float64)
        b = np.zeros([1], dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b[0], np.sum(a), rtol=1e-10)

    # ----------------------------------------- custom WCR fallback
    def test_custom_wcr_fallback(self):
        """Custom WCR should fall back to pure and still produce results."""
        # A custom WCR that is equivalent to sum but won't be detected
        # as a standard reduction type.
        sdfg, rnode = _make_reduce_sdfg(
            [8, 4],
            [1],
            "lambda a, b: a if a > b else b + 1",
            dtype=dace.float64,
        )
        rnode.implementation = "CuPy"

        # The expansion should fall back to pure. Verify it compiles.
        a = np.arange(32, dtype=np.float64).reshape(8, 4)
        b = np.zeros(8, dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        # Just verify it ran without error and produced finite values
        assert np.all(np.isfinite(b))

    # ----------------------------------------- combined reduction types
    @pytest.mark.parametrize(
        "wcr,np_fn,identity",
        [
            ("lambda a, b: a + b", np.sum, 0),
            ("lambda a, b: a * b", np.prod, 1),
            ("lambda a, b: min(a, b)", np.min, None),
            ("lambda a, b: max(a, b)", np.max, None),
        ],
    )
    @pytest.mark.parametrize(
        "in_shape,axes",
        [
            ([64], [0]),
            ([8, 16], [1]),
            ([8, 16], [0]),
            ([8, 16], [0, 1]),
            ([4, 6, 8], [1]),
            ([4, 6, 8], [0, 2]),
        ],
    )
    def test_reduction_type_shape_combos(self, wcr, np_fn, identity, in_shape, axes):
        """Parametrized: each supported reduction type x multiple shapes."""
        out_shape = [s for i, s in enumerate(in_shape) if i not in axes]
        if not out_shape:
            out_shape = [1]

        sdfg, rnode = _make_reduce_sdfg(
            in_shape,
            axes,
            wcr,
            dtype=dace.float64,
            identity=identity,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(*in_shape).astype(np.float64)
        if "a * b" in wcr:
            a = a + 0.5  # avoid zero for product

        b = np.zeros(out_shape, dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        expected = np_fn(a, axis=tuple(axes))
        np.testing.assert_allclose(b.reshape(expected.shape), expected, rtol=1e-10)

    # -------------------------------------------- GPU_Global (device) output
    def test_max_gpu_global_output(self):
        """Max reduction with GPU_Global in/out arrays (softmax regression).

        The result must stay on device: previously the expansion converted
        it to numpy, and cupy rejected the write-back into the device array.
        """
        import cupy

        sdfg, rnode = _make_reduce_sdfg_gpu([16, 32], [1], "lambda a, b: max(a, b)", dtype=dace.float32)
        rnode.implementation = "CuPy"

        a = np.random.rand(16, 32).astype(np.float32)
        a_gpu = cupy.asarray(a)
        b_gpu = cupy.zeros(16, dtype=cupy.float32)

        csdfg = sdfg.compile()
        csdfg(A=a_gpu, B=b_gpu)
        del csdfg

        np.testing.assert_allclose(cupy.asnumpy(b_gpu), np.max(a, axis=1))

    def test_sum_gpu_global_output(self):
        """Sum reduction with GPU_Global in/out arrays stays on device."""
        import cupy

        sdfg, rnode = _make_reduce_sdfg_gpu([8, 4, 16], [0, 2], "lambda a, b: a + b", dtype=dace.float64, identity=0)
        rnode.implementation = "CuPy"

        a = np.random.rand(8, 4, 16)
        a_gpu = cupy.asarray(a)
        b_gpu = cupy.zeros(4, dtype=cupy.float64)

        csdfg = sdfg.compile()
        csdfg(A=a_gpu, B=b_gpu)
        del csdfg

        np.testing.assert_allclose(cupy.asnumpy(b_gpu), np.sum(a, axis=(0, 2)), rtol=1e-12)

    # --------------------------------------------- single-axis 2D variants
    def test_sum_axis0(self):
        """Sum over axis 0 of a 2D array."""
        sdfg, rnode = _make_reduce_sdfg(
            [32, 16],
            [0],
            "lambda a, b: a + b",
            dtype=dace.float64,
            identity=0,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(32, 16).astype(np.float64)
        b = np.zeros(16, dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b, np.sum(a, axis=0), rtol=1e-12)

    def test_sum_axis1(self):
        """Sum over axis 1 of a 2D array."""
        sdfg, rnode = _make_reduce_sdfg(
            [32, 16],
            [1],
            "lambda a, b: a + b",
            dtype=dace.float64,
            identity=0,
        )
        rnode.implementation = "CuPy"

        a = np.random.rand(32, 16).astype(np.float64)
        b = np.zeros(32, dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b, np.sum(a, axis=1), rtol=1e-12)


if __name__ == "__main__":
    # Structural tests
    t = TestCuPyReduceStructure()
    t.test_registered()
    t.test_expansion_returns_nsdfg_sum()
    t.test_expansion_all_axes_uses_none()
    t.test_fallback_on_unsupported_type()
    t.test_fallback_on_degenerate()
    t.test_all_supported_types_in_map()
    print("All structural tests passed.")

    # GPU tests (will fail without GPU)
    tg = TestCuPyReduceGPU()
    tg.test_sum_all_axes()
    tg.test_sum_1d_full()
    tg.test_prod()
    tg.test_min()
    tg.test_max()
    tg.test_float64_sum()
    tg.test_float32_sum()
    tg.test_logical_and()
    tg.test_logical_or()
    tg.test_sum_3d_non_contiguous_axes()
    tg.test_sum_3d_all_axes()
    tg.test_sum_4d()
    tg.test_sum_non_power_of_2()
    tg.test_sum_large_1d()
    tg.test_sum_axis0()
    tg.test_sum_axis1()
    print("GPU tests passed.")
