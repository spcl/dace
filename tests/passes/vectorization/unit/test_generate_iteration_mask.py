# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unit tests for ``GenerateIterationMask`` (P3 vectorization-prep).

The pass attaches a per-iteration boolean lane mask ``_iter_mask`` to the
body of every target innermost map. Mode controls which maps are
targeted:

- ``step_w_only`` (default): only step-W innermost maps (the masked
  remainder shape P2 produces, or any prior pass that strides a map).
- ``all_innermost``: every innermost map (ALWAYS_ITER_MASK regime).

The pass requires P1 (NestInnermostMapBodyIntoNSDFG) to have wrapped
every innermost map's body in a single nested SDFG, the mask is added
to that nested SDFG so it is visible to every state in the body.
"""
import pytest

import dace
from dace.transformation.passes.vectorization.generate_iteration_mask import (
    GenerateIterationMask, )
from dace.transformation.passes.vectorization.nest_innermost_map_body import (
    NestInnermostMapBodyIntoNSDFG, )
from dace.transformation.passes.vectorization.split_map_for_vector_remainder import (
    SplitMapForVectorRemainder, )
from dace.transformation.passes.vectorization.utils.map_predicates import (
    get_single_nsdfg_inside_map,
    is_innermost_map,
)

N = dace.symbol("N")


@dace.program
def add_one(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] + 1.0


def _innermost_nsdfgs(sdfg):
    out = []
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState) and is_innermost_map(g, n):
            ns = get_single_nsdfg_inside_map(g, n)
            if ns is not None:
                out.append((n, ns))
    return out


def test_all_innermost_attaches_mask_to_every_body():
    sdfg = add_one.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    applied = GenerateIterationMask(vector_width=8, mode="all_innermost").apply_pass(sdfg, {})
    assert applied is not None and applied >= 1
    for _, nsdfg in _innermost_nsdfgs(sdfg):
        mask_names = [name for name in nsdfg.sdfg.arrays if name.startswith("_iter_mask")]
        assert mask_names, f"no _iter_mask in {nsdfg.label}"


def test_step_w_only_skips_step_1_maps():
    """Default ``step_w_only`` should not touch the original step-1 map."""
    sdfg = add_one.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    applied = GenerateIterationMask(vector_width=8, mode="step_w_only").apply_pass(sdfg, {})
    assert applied is None
    for _, nsdfg in _innermost_nsdfgs(sdfg):
        assert not any(name.startswith("_iter_mask") for name in nsdfg.sdfg.arrays)


def test_masked_attaches_mask_to_masked_remainder():
    """After P2 in masked mode, the remainder map is step-1 with the
    ``__masked_rem`` label suffix; P3 with ``mode='masked'`` matches the
    marker and attaches the mask to that body. (Renamed from
    ``test_step_w_only_*`` after Option A v2 changed P2 to emit step-1
    remainders + a label marker rather than step-W maps.)"""
    sdfg = add_one.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    sdfg.replace_dict({"N": 17})
    SplitMapForVectorRemainder(vector_width=8, mode="masked").apply_pass(sdfg, {})
    applied = GenerateIterationMask(vector_width=8, mode="masked").apply_pass(sdfg, {})
    assert applied is not None and applied >= 1


def test_pass_is_idempotent():
    sdfg = add_one.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    first = GenerateIterationMask(vector_width=8, mode="all_innermost").apply_pass(sdfg, {})
    second = GenerateIterationMask(vector_width=8, mode="all_innermost").apply_pass(sdfg, {})
    assert first is not None
    assert second is None


def test_invalid_mode_raises():
    sdfg = add_one.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    with pytest.raises(ValueError, match="mode must be"):
        GenerateIterationMask(vector_width=8, mode="bogus").apply_pass(sdfg, {})


def test_bare_tasklet_body_raises_not_implemented():
    """Without P1 the body is bare tasklets, P3 should refuse rather than
    silently miss the mask."""
    sdfg = add_one.to_sdfg(simplify=True)
    with pytest.raises(NotImplementedError, match="NestInnermostMapBodyIntoNSDFG"):
        GenerateIterationMask(vector_width=8, mode="all_innermost").apply_pass(sdfg, {})
