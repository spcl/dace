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


from tests.passes.vectorization.passes.test_split_map_for_vector_remainder import add_one  # noqa: E402 (dedup: canonical add_one)



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


@dace.program
def transposed_read(a: dace.float64[N], t: dace.float64[N, 4]):
    # ``t[i, 1]``: the lane var ``i`` is in dim-0 (stride 4, non-contiguous)
    # while the stride-1 dim (dim-1) holds the constant ``1`` -> a
    # transposed access that fans out per lane.
    for i in dace.map[0:N]:
        a[i] = t[i, 1]


def _remainder_maps(sdfg):
    """Maps P2 split off as the remainder (label contains ``rem`` or
    Sequential schedule), excluding the main vectorised map."""
    out = []
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState):
            if "rem" in n.map.label or n.map.schedule == dace.dtypes.ScheduleType.Sequential:
                out.append(n)
    return out


def test_masked_remainder_with_per_lane_fan_degrades_to_scalar():
    """A masked remainder whose body fans out per lane (transposed
    access) without ``lower_to_intrinsics`` must auto-degrade to a
    SCALAR remainder: Sequential schedule, no ``__masked_rem`` marker,
    no ``_iter_mask`` attached anywhere."""
    sdfg = transposed_read.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    SplitMapForVectorRemainder(vector_width=8, mode="masked").apply_pass(sdfg, {})
    applied = GenerateIterationMask(vector_width=8, mode="masked", lower_to_intrinsics=False).apply_pass(sdfg, {})

    assert applied is None
    assert not any(name.startswith("_iter_mask") for s in sdfg.all_sdfgs_recursive() for name in s.arrays)
    assert not any(
        isinstance(n, dace.nodes.MapEntry) and n.map.label.endswith("__masked_rem")
        for n, _ in sdfg.all_nodes_recursive())
    rems = _remainder_maps(sdfg)
    assert rems, "expected a remainder map after the split"
    assert all(r.map.schedule == dace.dtypes.ScheduleType.Sequential for r in rems), \
        "degraded remainder must be Sequential (scalar)"


def test_masked_remainder_with_per_lane_fan_kept_when_intrinsics_on():
    """With ``lower_to_intrinsics=True`` the per-lane fan is collapsed to
    a masked intrinsic downstream, so the masked remainder is NOT
    degraded — the ``_iter_mask`` is attached as usual."""
    sdfg = transposed_read.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    SplitMapForVectorRemainder(vector_width=8, mode="masked").apply_pass(sdfg, {})
    applied = GenerateIterationMask(vector_width=8, mode="masked", lower_to_intrinsics=True).apply_pass(sdfg, {})

    assert applied is not None and applied >= 1
    assert any(name.startswith("_iter_mask") for s in sdfg.all_sdfgs_recursive() for name in s.arrays)
