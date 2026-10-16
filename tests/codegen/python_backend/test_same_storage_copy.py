# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Same-storage AccessNode-to-AccessNode copies in the Python/cuTile backends.

The Python backend has *two* copy-dispatching targets active at once for a
``BackendLanguage.Python`` SDFG: ``PythonCodeGen`` (the default) and
``CuTilePythonCodeGen``.  ``CuTilePythonCodeGen`` is instantiated *after*
``PythonCodeGen`` (see ``codegen.py``) and overrides only the keys it
registers -- the cross-storage ``CPU_Heap <-> GPU_Global`` pairs and every
combination involving ``CuTile_Tile``.  Crucially, it does **not** register
the *same-storage* keys ``CPU_Heap -> CPU_Heap``, ``GPU_Global ->
GPU_Global`` or ``Register -> Register``; those stay with ``PythonCodeGen``.

This module pins down, end to end, what happens for same-storage copies of
**every** storage type, each with a compile + run + compare-against-NumPy
test:

* **Routing** (no GPU): which target the dispatcher selects per storage pair.
* ``CPU_Heap -> CPU_Heap`` (``PythonCodeGen`` -> ``numpy.copyto``): full /
  subset / 2D / symbolic copies.
* ``Register -> Register`` (``PythonCodeGen`` -> ``numpy.copyto`` on numpy
  scratch arrays): copy through a Register-storage transient chain.
* ``GPU_Global -> GPU_Global`` (``PythonCodeGen`` -> ``numpy.copyto`` on
  CuPy arrays via NumPy NEP-18 dispatch), ``@pytest.mark.gpu``.
* ``CuTile_Tile -> CuTile_Tile`` (``CuTilePythonCodeGen`` -> ``dst = src``
  tile rename, emitted by the AccessNode-centric path), exercised inside a
  real lowered cuTile kernel, ``@pytest.mark.gpu``.
"""
from typing import Optional, Tuple

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen import dispatcher as dispatcher_mod
from dace.codegen.py.cutile_target import CuTilePythonCodeGen
from dace.codegen.py.framecode import DaCePythonCodeGenerator
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.python_target import PythonCodeGen
from dace.memlet import Memlet
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile

ST = dtypes.StorageType


# ============================================================
# Helpers
# ============================================================


def _make_python_sdfg(name: str) -> SDFG:
    """Create an SDFG whose backend is the Python backend."""
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


def _build_copy_sdfg(name: str, storage: dtypes.StorageType, shape, subset: Optional[str] = None) -> SDFG:
    """Single-state ``A -> B`` AccessNode copy with both arrays in *storage*.

    :param name: Unique SDFG name.
    :param storage: Storage type for both ``A`` and ``B`` (same-storage copy).
    :param shape: Array shape.
    :param subset: Optional subset string applied to both ends of the memlet;
        ``None`` produces a full-array copy.
    :returns: The constructed SDFG.
    """
    sdfg = _make_python_sdfg(name)
    sdfg.add_array("A", list(shape), dace.float64, storage=storage)
    sdfg.add_array("B", list(shape), dace.float64, storage=storage)
    state = sdfg.add_state("s")
    a = state.add_read("A")
    b = state.add_write("B")
    if subset is not None:
        state.add_edge(a, None, b, None, Memlet(data="A", subset=subset, other_subset=subset))
    else:
        state.add_edge(a, None, b, None, Memlet(data="A"))
    return sdfg


def _full_code(sdfg: SDFG) -> str:
    """Concatenate all generated Python code objects into one string."""
    return "\n".join(obj.code for obj in sdfg.generate_code())


def _compile(sdfg: SDFG):
    """Compile *sdfg* and assert a runnable handle was returned.

    :param sdfg: The SDFG to compile.
    :returns: The non-None compiled callable.
    """
    compiled = sdfg.compile()
    assert compiled is not None
    return compiled


class _StubFrame:
    """Minimal frame-codegen stub exposing a fresh dispatcher."""

    def __init__(self):
        self.dispatcher = dispatcher_mod.TargetDispatcher(self)
        self._initcode = PythonCodeIOStream()
        self._exitcode = PythonCodeIOStream()


def _build_both_targets(sdfg: SDFG) -> Tuple[PythonCodeGen, CuTilePythonCodeGen]:
    """Instantiate both Python-backend targets in production order.

    ``codegen.py`` registers ``PythonCodeGen`` first and ``CuTilePythonCodeGen``
    second on a shared dispatcher; the later registration wins on overlapping
    keys.  This helper reproduces that order on one frame/dispatcher.

    :param sdfg: The Python-backend SDFG.
    :returns: ``(python_target, cutile_target)`` sharing one dispatcher.
    """
    frame = DaCePythonCodeGenerator(sdfg)
    py = PythonCodeGen(frame, sdfg)
    cutile = CuTilePythonCodeGen(frame, sdfg)
    return py, cutile


# ============================================================
# Routing: which target owns each same-storage pair
# ============================================================


class TestSameStorageRouting:
    """The dispatcher routes same-storage copies to the expected target."""

    @pytest.mark.parametrize("storage", [ST.CPU_Heap, ST.GPU_Global, ST.Register])
    def test_non_tile_same_storage_routes_to_python(self, storage):
        """CPU/GPU/Register same-storage copies stay with ``PythonCodeGen``."""
        sdfg = _make_python_sdfg(f"route_{storage.name.lower()}")
        py, _ = _build_both_targets(sdfg)
        handler = py._dispatcher._generic_copy_dispatchers.get((storage, storage, None))
        assert handler is py

    def test_tile_same_storage_routes_to_cutile(self):
        """``CuTile_Tile -> CuTile_Tile`` is owned by ``CuTilePythonCodeGen``."""
        sdfg = _make_python_sdfg("route_tile")
        _, cutile = _build_both_targets(sdfg)
        handler = cutile._dispatcher._generic_copy_dispatchers.get((ST.CuTile_Tile, ST.CuTile_Tile, None))
        assert handler is cutile

    def test_cutile_does_not_own_same_storage_gpu(self):
        """cuTile must NOT intercept ``GPU_Global -> GPU_Global``.

        If it did, its same-storage fallback would emit a bare ``B = A``
        rebind (alias, not a copy) for full-array memlets.  Ownership must
        stay with ``PythonCodeGen``, which emits ``numpy.copyto``.
        """
        sdfg = _make_python_sdfg("route_gpu_not_cutile")
        py, cutile = _build_both_targets(sdfg)
        handler = py._dispatcher._generic_copy_dispatchers.get((ST.GPU_Global, ST.GPU_Global, None))
        assert handler is py
        assert handler is not cutile


# ============================================================
# CPU_Heap -> CPU_Heap, end to end (no GPU)
# ============================================================


class TestCPUSameStorageRuntime:
    """``CPU_Heap -> CPU_Heap`` copies compile, run, and match NumPy."""

    def test_full_array_copy(self):
        """Full-array copy emits ``numpy.copyto`` and copies all elements."""
        sdfg = _build_copy_sdfg("cpu_full", ST.CPU_Heap, (8, ))
        assert "numpy.copyto(B" in _full_code(sdfg)
        compiled = _compile(sdfg)
        a = np.arange(8, dtype=np.float64)
        b = np.zeros(8, dtype=np.float64)
        compiled(A=a, B=b)
        np.testing.assert_array_equal(b, a)

    def test_full_array_copy_is_value_copy_not_alias(self):
        """The destination is an independent copy, not an alias of source."""
        sdfg = _build_copy_sdfg("cpu_full_alias", ST.CPU_Heap, (8, ))
        compiled = _compile(sdfg)
        a = np.arange(8, dtype=np.float64)
        b = np.zeros(8, dtype=np.float64)
        compiled(A=a, B=b)
        # Mutating the source afterwards must not affect the destination.
        a[:] = -1.0
        np.testing.assert_array_equal(b, np.arange(8, dtype=np.float64))

    def test_subset_copy(self):
        """Subset copy writes only the selected range, leaving the rest."""
        sdfg = _build_copy_sdfg("cpu_subset", ST.CPU_Heap, (8, ), subset="2:6")
        compiled = _compile(sdfg)
        a = np.arange(8, dtype=np.float64)
        b = np.zeros(8, dtype=np.float64)
        compiled(A=a, B=b)
        expected = np.zeros(8, dtype=np.float64)
        expected[2:6] = a[2:6]
        np.testing.assert_array_equal(b, expected)

    def test_2d_full_copy(self):
        """2D full-array same-storage copy matches NumPy."""
        sdfg = _build_copy_sdfg("cpu_2d", ST.CPU_Heap, (4, 5))
        compiled = _compile(sdfg)
        a = np.arange(20, dtype=np.float64).reshape(4, 5)
        b = np.zeros((4, 5), dtype=np.float64)
        compiled(A=a, B=b)
        np.testing.assert_array_equal(b, a)

    def test_symbolic_size_copy(self):
        """Symbolically-sized same-storage copy compiles and runs."""
        N = dace.symbol("N")
        sdfg = _make_python_sdfg("cpu_symbolic")
        sdfg.add_array("A", (N, ), dace.float64, storage=ST.CPU_Heap)
        sdfg.add_array("B", (N, ), dace.float64, storage=ST.CPU_Heap)
        state = sdfg.add_state("s")
        a = state.add_read("A")
        b = state.add_write("B")
        state.add_edge(a, None, b, None, Memlet(data="A"))
        compiled = _compile(sdfg)
        n = 13
        a = np.arange(n, dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)
        compiled(A=a, B=b, N=n)
        np.testing.assert_array_equal(b, a)


# ============================================================
# Register -> Register, end to end (no GPU)
# ============================================================


def _build_register_copy_sdfg(name: str, shape, subset: Optional[str] = None) -> SDFG:
    """``A(CPU) -> R1(Register) -> R2(Register) -> B(CPU)`` copy chain.

    The middle ``R1 -> R2`` edge is the ``Register -> Register`` same-storage
    copy under test; Register transients are realized as numpy scratch arrays
    by the Python backend.

    :param name: Unique SDFG name.
    :param shape: Array shape.
    :param subset: Optional subset applied to the ``R1 -> R2`` copy.
    :returns: The constructed SDFG.
    """
    sdfg = _make_python_sdfg(name)
    sdfg.add_array("A", list(shape), dace.float64, storage=ST.CPU_Heap)
    sdfg.add_array("B", list(shape), dace.float64, storage=ST.CPU_Heap)
    sdfg.add_transient("R1", list(shape), dace.float64, storage=ST.Register)
    sdfg.add_transient("R2", list(shape), dace.float64, storage=ST.Register)
    state = sdfg.add_state("s")
    a = state.add_read("A")
    r1 = state.add_access("R1")
    r2 = state.add_access("R2")
    b = state.add_write("B")
    state.add_edge(a, None, r1, None, Memlet(data="A"))
    if subset is not None:
        state.add_edge(r1, None, r2, None, Memlet(data="R1", subset=subset, other_subset=subset))
    else:
        state.add_edge(r1, None, r2, None, Memlet(data="R1"))
    state.add_edge(r2, None, b, None, Memlet(data="R2"))
    return sdfg


class TestRegisterSameStorageRuntime:
    """``Register -> Register`` copies compile, run, and match NumPy."""

    def test_full_array_copy(self):
        """Full-array Register->Register copy routes through ``numpy.copyto``."""
        sdfg = _build_register_copy_sdfg("reg_full", (8, ))
        assert "numpy.copyto(R2" in _full_code(sdfg)
        compiled = _compile(sdfg)
        a = np.arange(8, dtype=np.float64)
        b = np.zeros(8, dtype=np.float64)
        compiled(A=a, B=b)
        np.testing.assert_array_equal(b, a)

    def test_subset_copy(self):
        """Subset Register->Register copy moves only the selected range.

        ``R2`` is uninitialized scratch outside the copied range, so only the
        copied slice is asserted (the rest reaches ``B`` unconstrained).
        """
        sdfg = _build_register_copy_sdfg("reg_subset", (8, ), subset="2:6")
        compiled = _compile(sdfg)
        a = np.arange(8, dtype=np.float64)
        b = np.zeros(8, dtype=np.float64)
        compiled(A=a, B=b)
        np.testing.assert_array_equal(b[2:6], a[2:6])

    def test_2d_full_copy(self):
        """2D Register->Register copy matches NumPy."""
        sdfg = _build_register_copy_sdfg("reg_2d", (4, 5))
        compiled = _compile(sdfg)
        a = np.arange(20, dtype=np.float64).reshape(4, 5)
        b = np.zeros((4, 5), dtype=np.float64)
        compiled(A=a, B=b)
        np.testing.assert_array_equal(b, a)


# ============================================================
# cuTile target -- CuTile_Tile same-storage codegen (no GPU)
# ============================================================


def _cutile_copy_code(name: str, subset: Optional[str] = None) -> str:
    """Drive ``CuTilePythonCodeGen.copy_memory`` for a tile-to-tile copy.

    This exercises the ``copy_memory`` same-storage *fallback* directly (the
    code path normally taken instead is the AccessNode-centric one, tested at
    runtime below).

    :param name: Unique SDFG name.
    :param subset: Optional subset string for both memlet ends.
    :returns: The emitted call-site code.
    """
    sdfg = _make_python_sdfg(name)
    sdfg.add_array("A", [8], dace.float64, storage=ST.CuTile_Tile, transient=True)
    sdfg.add_array("B", [8], dace.float64, storage=ST.CuTile_Tile, transient=True)
    state = sdfg.add_state("s")
    a = state.add_read("A")
    b = state.add_write("B")
    if subset is not None:
        memlet = Memlet(data="A", subset=subset, other_subset=subset)
        state.add_edge(a, None, b, None, memlet)
    else:
        memlet = Memlet(data="A")  # no edge -> subsets stay None (bare rebind)
    codegen = CuTilePythonCodeGen(_StubFrame(), sdfg)
    edge = type("E", (), {"data": memlet})()
    cs = PythonCodeIOStream()
    codegen.copy_memory(sdfg, sdfg, state, 0, a, b, edge, PythonCodeIOStream(), cs)
    return cs.getvalue().strip()


class TestCuTileTileSameStorageCodegen:
    """``CuTile_Tile -> CuTile_Tile`` emits a plain value assignment."""

    def test_subset_emits_slice_assignment(self):
        """A subsetted tile copy slices both sides."""
        code = _cutile_copy_code("tile_subset", subset="0:8")
        assert code == "B[:8] = A[:8]"

    def test_no_subset_emits_bare_rebind(self):
        """No-subset tile copy is a bare rebind (tiles are immutable values)."""
        code = _cutile_copy_code("tile_bare")
        assert code == "B = A"
        assert "numpy.copyto" not in code
        assert ".set(" not in code and ".get(" not in code


# ============================================================
# GPU_Global -> GPU_Global, runtime (needs a GPU)
# ============================================================


@pytest.mark.gpu
class TestGPUSameStorageRuntime:
    """``GPU_Global -> GPU_Global`` same-storage copies run on CuPy arrays.

    These go through ``PythonCodeGen`` (not cuTile) and emit
    ``numpy.copyto``, which dispatches to CuPy via NumPy's NEP-18
    ``__array_function__`` protocol.
    """

    def test_full_array_copy(self):
        """Full-array GPU same-storage copy matches the reference."""
        import cupy

        sdfg = _build_copy_sdfg("gpu_full", ST.GPU_Global, (16, ))
        assert "numpy.copyto(B" in _full_code(sdfg)
        compiled = _compile(sdfg)
        a = cupy.arange(16, dtype=cupy.float64)
        b = cupy.zeros(16, dtype=cupy.float64)
        compiled(A=a, B=b)
        assert bool(cupy.allclose(b, a))

    def test_full_array_copy_is_value_copy_not_alias(self):
        """GPU destination is an independent copy, not an alias."""
        import cupy

        sdfg = _build_copy_sdfg("gpu_full_alias", ST.GPU_Global, (16, ))
        compiled = _compile(sdfg)
        a = cupy.arange(16, dtype=cupy.float64)
        b = cupy.zeros(16, dtype=cupy.float64)
        compiled(A=a, B=b)
        a[:] = -1.0
        assert bool(cupy.allclose(b, cupy.arange(16, dtype=cupy.float64)))

    def test_subset_copy(self):
        """Subsetted GPU same-storage copy writes only the selected range."""
        import cupy

        sdfg = _build_copy_sdfg("gpu_subset", ST.GPU_Global, (16, ), subset="4:12")
        compiled = _compile(sdfg)
        a = cupy.arange(16, dtype=cupy.float64)
        b = cupy.zeros(16, dtype=cupy.float64)
        compiled(A=a, B=b)
        expected = cupy.zeros(16, dtype=cupy.float64)
        expected[4:12] = a[4:12]
        assert bool(cupy.allclose(b, expected))

    def test_2d_full_copy(self):
        """2D GPU same-storage copy matches the reference."""
        import cupy

        sdfg = _build_copy_sdfg("gpu_2d", ST.GPU_Global, (4, 8))
        compiled = _compile(sdfg)
        a = cupy.arange(32, dtype=cupy.float64).reshape(4, 8)
        b = cupy.zeros((4, 8), dtype=cupy.float64)
        compiled(A=a, B=b)
        assert bool(cupy.allclose(b, a))


# ============================================================
# CuTile_Tile -> CuTile_Tile, runtime inside a real kernel (needs a GPU)
# ============================================================


def _build_vadd_with_tile_copy(name: str, width: int = 8) -> Tuple[SDFG, str]:
    """Lower a vadd kernel via ``VectorizeCuTile``, then splice a tile copy.

    After lowering, the result tile (produced by the binop) flows straight
    into the ``TileScatter``.  We insert a new ``CuTile_Tile`` transient
    ``C_tile_copy`` between them, so the store reads from the copy.  The cuTile
    AccessNode-centric codegen emits ``C_tile_copy = <out tile>`` -- the
    ``CuTile_Tile -> CuTile_Tile`` tile-rename path -- and the result must be
    unchanged (``C == A + B``).

    The result tile is located through its store node.

    :param name: Unique SDFG name.
    :param width: Tile width (power of two).
    :returns: The lowered + spliced SDFG and the resolved out-tile name.
    """
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (N, ), dace.float64)
    sdfg.add_array("B", (N, ), dace.float64)
    sdfg.add_array("C", (N, ), dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "add",
        {"i": "0:N"},
        {"_a": dace.Memlet("A[i]"), "_b": dace.Memlet("B[i]")},
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    VectorizeCuTile(widths=(width, )).apply_pass(sdfg, {})

    # Find the output through its store, independent of GPU clone naming.
    from dace.libraries.tileops.nodes.masked_copy import MaskedCopyLibraryNode
    from dace.libraries.tileops.nodes.tile_scatter import TileScatter

    kstate, store_edge = next(
        (state, edge)
        for state in sdfg.states()
        for edge in state.edges()
        if isinstance(edge.src, nodes.AccessNode)
        and sdfg.arrays[edge.src.data].storage == ST.CuTile_Tile
        and ((isinstance(edge.dst, MaskedCopyLibraryNode) and edge.dst.stores(state))
             or isinstance(edge.dst, TileScatter))
        and edge.dst_conn != "_mask"
    )
    c_out = store_edge.src
    out_name = c_out.data
    store = store_edge.dst

    # Splice a CuTile_Tile copy: <out tile> -> C_tile_copy -> TileScatter.
    sdfg.add_array("C_tile_copy", sdfg.arrays[out_name].shape, dace.float64,
                   storage=ST.CuTile_Tile, transient=True)
    c_copy = kstate.add_access("C_tile_copy")
    kstate.remove_edge(store_edge)
    kstate.add_edge(c_out, None, c_copy, None, dace.Memlet(data=out_name))
    kstate.add_edge(c_copy, None, store, store_edge.dst_conn, dace.Memlet(data="C_tile_copy"))
    return sdfg, out_name


@pytest.mark.gpu
class TestCuTileTileSameStorageRuntime:
    """``CuTile_Tile -> CuTile_Tile`` tile rename runs inside a real kernel."""

    def test_tile_to_tile_copy_matches_numpy(self):
        """A spliced tile-to-tile copy keeps ``C == A + B`` (aligned size)."""
        sdfg, out_name = _build_vadd_with_tile_copy("cutile_tile2tile_aligned")
        assert f"C_tile_copy = {out_name}" in _full_code(sdfg)

        n = 64
        rng = np.random.default_rng(7)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        _compile(sdfg)(A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_tile_to_tile_copy_unaligned(self):
        """Tile-to-tile copy with a non-divisible size (masking + rename)."""
        sdfg, _ = _build_vadd_with_tile_copy("cutile_tile2tile_unaligned")

        n = 17  # not a multiple of the tile width
        rng = np.random.default_rng(8)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        _compile(sdfg)(A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
