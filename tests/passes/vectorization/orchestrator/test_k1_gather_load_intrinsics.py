# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""K=1 ``TileLoad`` gather + strided-load intrinsic lowering.

A unit-stride load lowers to a dense SIMD load (``_mm512_loadu_pd``), a constant
non-unit stride to the gather intrinsic over a strided index vector
(``_mm512_i64gather_pd``), and a data-dependent ``a[idx[i]]`` gather to the same
gather intrinsic over the index tile. Each is checked bit-exact against numpy on
SCALAR (reference loop) and AVX512 (intrinsic), masked + unmasked. The masked
forms must ZERO-FILL inactive lanes WITHOUT dereferencing their index (the
hardware masked gather only touches active lanes) -- exercised by leaving the
inactive lanes' index pointing at a non-zero element and asserting the output
lane is 0, not that element.
"""
import os

import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileLoad

W = 8  # fp64 AVX-512 lane count


def _host_flags():
    """x86 CPU feature flags from ``/proc/cpuinfo`` (empty off-Linux / off-x86)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


_FLAGS = _host_flags()
# (implementation, required cpuinfo flag (None = always run), backend header).
# The intrinsic itself (``_mm512_i64gather_pd`` / ``_mm512_loadu_pd``) lives in
# the header, not the generated .cpp; the .cpp carries the ``tile_<op><...>``
# call + the header include. Asserting the include proves the intrinsic backend
# was selected; the numpy comparison proves it computes correctly.
_CASES = [
    ("scalar", None, "dace/tile_ops/scalar.h"),
    ("avx512", "avx512f", "dace/tile_ops/avx512.h"),
]


def _wire_mask(sdfg, state, node):
    """Feed ``node``'s ``_mask`` from a Register transient (the design 10.2 lock:
    ``bool[W]`` Register transient) copied in-state from a global ``msk`` input,
    so the test can drive the mask values from Python."""
    sdfg.add_array("msk", [W], dace.bool_)
    sdfg.add_array("_tile_iter_mask", [W], dace.bool_, storage=dace.dtypes.StorageType.Register, transient=True)
    mask_an = state.add_access("_tile_iter_mask")
    state.add_edge(state.add_access("msk"), None, mask_an, None, dace.Memlet(f"_tile_iter_mask[0:{W}]"))
    state.add_edge(mask_an, None, node, "_mask", dace.Memlet(f"_tile_iter_mask[0:{W}]"))


def _gather_sdfg(name, impl, masked):
    """``dst[l] = src[idx[l]]`` (1-D unit-stride gather), optionally masked."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("src", [256], dace.float64)
    sdfg.add_array("idx", [W], dace.int64)
    sdfg.add_array("dst", [W], dace.float64)
    state = sdfg.add_state()
    node = TileLoad(name="ld",
                    widths=(W, ),
                    dim_strides=(1, ),
                    replicate_factor_per_dim=(1, ),
                    src_dims=(0, ),
                    gather_dims=(0, ),
                    has_mask=masked)
    node.implementation = impl
    state.add_node(node)
    state.add_edge(state.add_access("src"), None, node, "_src", dace.Memlet("src[0:256]"))
    state.add_edge(state.add_access("idx"), None, node, "_idx_0", dace.Memlet(f"idx[0:{W}]"))
    if masked:
        _wire_mask(sdfg, state, node)
    state.add_edge(node, "_dst", state.add_access("dst"), None, dace.Memlet(f"dst[0:{W}]"))
    return sdfg


def _strided_sdfg(name, impl, stride, masked):
    """``dst[l] = src[l * stride]`` (constant non-unit stride), optionally masked."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("src", [W * stride + 8], dace.float64)
    sdfg.add_array("dst", [W], dace.float64)
    state = sdfg.add_state()
    node = TileLoad(name="ld",
                    widths=(W, ),
                    dim_strides=(stride, ),
                    replicate_factor_per_dim=(1, ),
                    src_dims=(0, ),
                    gather_dims=(),
                    has_mask=masked)
    node.implementation = impl
    state.add_node(node)
    state.add_edge(state.add_access("src"), None, node, "_src", dace.Memlet(f"src[0:{W * stride + 8}]"))
    if masked:
        _wire_mask(sdfg, state, node)
    state.add_edge(node, "_dst", state.add_access("dst"), None, dace.Memlet(f"dst[0:{W}]"))
    return sdfg


def _compiled_code(sdfg):
    cpp = os.path.join(sdfg.build_folder, "src", "cpu", sdfg.name + ".cpp")
    with open(cpp) as f:
        return f.read()


@pytest.mark.parametrize("impl,flag,header", _CASES)
@pytest.mark.parametrize("masked", [False, True])
def test_k1_gather_intrinsic(impl, flag, header, masked):
    """``a[idx[i]]`` lowers to ``tile_gather`` and matches numpy; masked lanes
    zero-fill without reading their index."""
    if flag is not None and flag not in _FLAGS:
        pytest.skip(f"host lacks {flag}")
    sdfg = _gather_sdfg(f"tg_{impl}_{int(masked)}", impl, masked)
    csdfg = sdfg.compile()
    code = _compiled_code(sdfg)
    assert "dace::tileops::tile_gather<" in code, f"{impl}: tile_gather not emitted"
    assert header in code, f"{impl}: backend header {header} not included"

    rng = np.random.default_rng(seed=0xA17 + int(masked))
    src = rng.random(256)
    idx = rng.integers(0, 256, size=W).astype(np.int64)
    dst = np.zeros(W)
    args = dict(src=src, idx=idx, dst=dst)
    if masked:
        # Active prefix of 5 lanes; inactive lanes point at a NON-zero element to
        # prove the masked gather never reads them (output must be 0, not src[idx]).
        msk = np.array([True] * 5 + [False] * (W - 5))
        idx[~msk] = int(np.argmax(np.abs(src)))  # a definitely-non-zero element
        args["msk"] = msk
    csdfg(**args)
    expected = src[idx].copy()
    if masked:
        expected[~msk] = 0.0
    np.testing.assert_allclose(dst, expected, rtol=0, atol=0)


@pytest.mark.parametrize("impl,flag,header", _CASES)
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_k1_strided_load_intrinsic(impl, flag, header, masked, stride):
    """``src[l * stride]`` lowers to ``tile_load`` (loadu when stride==1, gather
    when stride!=1) and matches numpy; masked lanes zero-fill."""
    if flag is not None and flag not in _FLAGS:
        pytest.skip(f"host lacks {flag}")
    sdfg = _strided_sdfg(f"tl_{impl}_{stride}_{int(masked)}", impl, stride, masked)
    csdfg = sdfg.compile()
    code = _compiled_code(sdfg)
    assert "dace::tileops::tile_load<" in code, f"{impl}: tile_load not emitted"
    assert header in code, f"{impl}: backend header {header} not included"

    rng = np.random.default_rng(seed=0xB22 + stride + int(masked))
    src = rng.random(W * stride + 8)
    dst = np.zeros(W)
    args = dict(src=src, dst=dst)
    msk = None
    if masked:
        msk = np.array([True] * 6 + [False] * (W - 6))
        args["msk"] = msk
    csdfg(**args)
    expected = np.array([src[l * stride] for l in range(W)])
    if masked:
        expected[~msk] = 0.0
    np.testing.assert_allclose(dst, expected, rtol=0, atol=0)
