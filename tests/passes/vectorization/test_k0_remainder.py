# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""K=0 scalar-postamble tile-op remainder coverage.

When ``VectorizeCPUMultiDim`` is invoked with
``remainder_strategy="scalar_postamble"`` AND
``scalar_remainder_emit="tile_k1"``, the postamble runs through the
tile-op pipeline at ``widths=(1,)`` — single-lane tile ops per element
— rather than the legacy plain scalar loop. This file pins numerical
equivalence on the canonical hard kernels (cloudsc snippets + TSVC
hardening canonicals + a few extra patterns) under that K=0 mode.

The test never depends on the ``--run-full-matrix`` knob expansion: it
runs in the default sweep with a non-divisible LEN so the postamble is
exercised on every kernel.

Known follow-up: the tile runtime (``tile_load`` / ``tile_binop`` in
``dace::tileops``) takes ``T*`` for its tile-shape operands; DaCe's
codegen collapses a ``Register``-storage ``Array(shape=(1,))`` to a
plain ``T`` variable, so the K=1 ``widths=(1,)`` post-amble tile path
hits a ``cannot convert 'T' to 'T*'`` compile error on kernels whose
tile transients reach the runtime call sites unmodified. Fixing it
needs either a runtime overload for ``W==1`` taking ``T&`` or a codegen
adjustment to keep the ``T[1]`` array shape. Kernels that already
synthesize an explicit ``T*`` boundary upstream are not affected and
land cleanly through the K=0 path.
"""
import numpy as np
import pytest

from tests.corpus import tsvc
from tests.passes.vectorization.helpers.harness import run_vectorization_test

_G1D = tsvc.collect(regime="1d")
_G2D = tsvc.collect(regime="2d")

#: TSVC kernels exercising the hardest knob × access patterns. The
#: first 9 mirror :data:`_K6_HARD_CANONICALS` (2D stencil, 1D stencil
#: chains, branch, multi-stmt, gather, broadcast); the remaining 7 add
#: complementary coverage (irregular gather, indirect, broadcast-on-
#: gather, mixed-stride, multi-stmt accumulators).
_K0_TSVC_HARD = (
    "s1119_d_single",  # 2D stencil
    "s291_d_single",  # 1D stencil a[i] = (b[i] + b[i-1]) * 0.5
    "s4114_d_single",  # 1D stencil with arithmetic
    "s273_d_single",  # branch + multi-stmt
    "s1281_d_single",  # multi-stmt chain
    "vbor_d_single",  # multi-stmt accumulator
    "vag_d_single",  # pure gather a[i] = b[ip[i]]
    "s251_d_single",  # broadcast-via-intermediate-scalar
    "s4112_d_single",  # gather + broadcast combined
    "s4117_d_single",  # halve-index multiplex
    "s4115_d_single",  # gather with arithmetic
    "s4116_d_single",  # gather + arithmetic combo
    "s4121_d_single",  # FMA on scalars
    "s1115_d_single",  # 2D combined pattern
    "s3251_d_single",  # branch + accumulator
    "vpvts_d_single",  # vector scalar multiply add
)


def _resolve_canonical(kernel_name: str):
    """Look up a TSVC kernel by short name."""
    for k in list(_G1D) + list(_G2D):
        if k.program.name.endswith(kernel_name):
            return k
    raise KeyError(f"TSVC kernel {kernel_name!r} not in corpus")


@pytest.mark.parametrize("kernel_name", _K0_TSVC_HARD)
def test_k0_remainder_tsvc(kernel_name: str):
    """K=0 scalar-postamble path on hard TSVC kernels: the tile-op
    remainder at ``widths=(1,)`` must stay numerically equivalent to the
    un-transformed scalar reference at a non-divisible LEN.
    """
    kernel = _resolve_canonical(kernel_name)
    # Non-divisible LEN so the postamble actually fires (LEN % 8 != 0).
    if kernel.regime == "1d":
        l1, l2 = 65, tsvc.LEN_2D_FIXED
    else:
        l1, l2 = tsvc.LEN_2D_FIXED, 17

    rng = np.random.default_rng(seed=hash(kernel_name) & 0xFFFF)
    arrays = tsvc.allocate(kernel, l1, l2, rng)
    sym = tsvc.symbols(kernel, l1, l2)
    sparams = tsvc.scalar_params(kernel, l1)
    run_vectorization_test(
        dace_func=kernel.program,
        arrays=arrays,
        params={
            **sym,
            **sparams
        },
        vector_width=8,
        sdfg_name=f"k0_remainder_{kernel_name}",
        branch_mode="merge",
        remainder_strategy="scalar",
        emission_style="default",
        nest_map_bodies=False,
        insert_copies=False,
        vectorize_config="tile_nodes",
        scalar_remainder_emit="tile_k1",
    )
