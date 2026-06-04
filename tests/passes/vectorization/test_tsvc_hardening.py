# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""K6 — hard TSVC canonicals exercised under the full knob matrix.

Lives in ``tests/passes/vectorization/`` (not ``tests/passes/``) so the
opt-in fixtures defined in ``tests/passes/vectorization/conftest.py``
(``tile_emit_mode``, ``branch_mode``, ``remainder_strategy``,
``emission_style``, ``vectorize_config``) are visible to the test.
"""
import pytest
import numpy as np

from tests.corpus import tsvc

_G1D = tsvc.collect(regime="1d")
_G2D = tsvc.collect(regime="2d")

#: K6 hard-canonicals — 9 TSVC kernels exercised under the full knob
#: matrix on top of the existing ``test_tsvc_vectorization`` parametrisation.
#: Every canonical was verified to produce a vector-width-strided map on
#: BOTH orchestrator arms (``VectorizeCPU`` legacy AND
#: ``VectorizeCPUMultiDim`` tile). Pattern coverage:
#:
#: - 2D stencil: ``s1119``
#: - 1D stencil ``a[i] = (b[i] + b[i-1]) * 0.5``: ``s291``
#: - 1D stencil with arithmetic ``a[i] = b[i] + b[i+1] * c[i]``: ``s4114``
#: - branch + multi-stmt: ``s273``
#: - multi-stmt chain: ``s1281``, ``vbor``
#: - pure gather ``a[i] = b[ip[i]]``: ``vag``
#: - broadcast-via-intermediate-scalar ``s = b+c*d; a = s*s``: ``s251``
#: - gather + broadcast combined ``a[i] = a[i] + b[ip[i]] * 2.0``: ``s4112``
_K6_HARD_CANONICALS = (
    "s1119_d_single",
    "s291_d_single",
    "s4114_d_single",
    "s273_d_single",
    "s1281_d_single",
    "vbor_d_single",
    "vag_d_single",
    "s251_d_single",
    "s4112_d_single",
)


def _resolve_canonical(kernel_name):
    """Look up a TSVC kernel by short name (e.g. ``s1119_d_single``)."""
    for k in list(_G1D) + list(_G2D):
        # k.program.name is like ``tests_corpus_tsvc_s1119_d_single``
        if k.program.name.endswith(kernel_name):
            return k
    raise KeyError(f"TSVC kernel {kernel_name!r} not in corpus")


@pytest.mark.parametrize("kernel_name", _K6_HARD_CANONICALS)
def test_tsvc_hardening_canonicals(kernel_name, tile_emit_mode, branch_mode, remainder_strategy, emission_style,
                                   vectorize_config):
    """K6: hard TSVC canonicals under the full knob matrix.

    Nine kernels covering stencil / branch / multi-stmt / gather /
    broadcast patterns (see :data:`_K6_HARD_CANONICALS`). Each runs
    through the harness ``run_vectorization_test`` with the full opt-in
    fixture set so knob × pattern interactions surface. The base
    ``test_tsvc_vectorization`` keeps the kernel + ``(remainder, branch,
    LEN)`` coverage for the rest of the corpus.

    The harness routes ``vectorize_config="legacy_cpu"`` to ``VectorizeCPU``
    and ``"tile_nodes"`` to ``VectorizeCPUMultiDim``; per-arm skip
    predicates filter combos the orchestrator does not support.
    """
    from tests.passes.vectorization.helpers.harness import run_vectorization_test
    nest_map_bodies, insert_copies = tile_emit_mode
    kernel = _resolve_canonical(kernel_name)
    # Use the divisible LEN per regime (the non-divisible variant is
    # already exercised in ``test_tsvc_vectorization``; here we focus on
    # the knob cross, not the remainder shape, which the
    # ``remainder_strategy`` fixture handles independently).
    if kernel.regime == "1d":
        l1, l2 = 64, tsvc.LEN_2D_FIXED
    else:
        l1, l2 = tsvc.LEN_2D_FIXED, 16

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
        sdfg_name=f"tsvc_hardening_{kernel_name}",
        branch_mode=branch_mode,
        remainder_strategy=remainder_strategy,
        emission_style=emission_style,
        nest_map_bodies=nest_map_bodies,
        insert_copies=insert_copies,
        vectorize_config=vectorize_config,
    )
