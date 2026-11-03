# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Maximal map fusion in the canonicalization recipe (``maximal_fusion``, the recompose phase's fusion fixpoint)
on CloudSC-shaped loops.

The fixture -- four element-wise ``(jl, jk)`` loop nests over shared arrays -- is driven through the recipe up to,
but not including, maximal fusion: the loops have just become maps and nothing has fused them yet.
"""

import copy

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.pipeline import PhaseFixpoint, build_stages

KLON = dace.symbol("KLON")
KLEV = dace.symbol("KLEV")

#: Fixture shape -- small enough to compile in seconds, big enough that a wrong fusion shows up.
FIXTURE_KLON, FIXTURE_KLEV = 12, 7


@dace.program
def elementwise_chain(
    pt: dace.float64[KLON, KLEV],
    pq: dace.float64[KLON, KLEV],
    pa: dace.float64[KLON, KLEV],
    tend_t: dace.float64[KLON, KLEV],
    tend_q: dace.float64[KLON, KLEV],
):
    """Four CloudSC-shaped element-wise sweeps over the same buffers. The 1st and 3rd are a
    read-after-write chain on ``tend_t``, and the 3rd reads ``tend_q`` that the 4th overwrites -- so a
    fusion that ignores ordering would silently change the answer."""
    for jk in range(KLEV):
        for jl in range(KLON):
            tend_t[jl, jk] = pt[jl, jk] * 2.0 + pq[jl, jk]
    for jk in range(KLEV):
        for jl in range(KLON):
            tend_q[jl, jk] = pq[jl, jk] * 0.5 - pa[jl, jk]
    for jk in range(KLEV):
        for jl in range(KLON):
            tend_t[jl, jk] = tend_t[jl, jk] + tend_q[jl, jk] * pa[jl, jk]
    for jk in range(KLEV):
        for jl in range(KLON):
            tend_q[jl, jk] = tend_q[jl, jk] * pt[jl, jk] + 1.0


def nmaps(sdfg: dace.SDFG) -> int:
    return sum(1 for node, scope in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry))


def fusion_index() -> int:
    """Position of the recipe's first maximal-fusion fixpoint (recompose's; a second one follows the last lifts)."""
    at = [
        i
        for i, (label, unit) in enumerate(build_stages(target="cpu"))
        if isinstance(unit, PhaseFixpoint) and unit.name == "maximal_fusion"
    ]
    assert at, "the recipe has no maximal-fusion fixpoint"
    return at[0]


def premapped() -> dace.SDFG:
    """The fixture driven through the recipe up to -- and stopping before -- maximal fusion."""
    sdfg = elementwise_chain.to_sdfg(simplify=False)
    for stage_label, unit in build_stages(target="cpu")[: fusion_index()]:
        unit.apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def apply_fusion(sdfg: dace.SDFG):
    """Apply the recipe's maximal fusion to ``sdfg``; return its ``apply_pass`` result (rounds that changed it)."""
    return build_stages(target="cpu")[fusion_index()][1].apply_pass(sdfg, {})


def run_fixture(sdfg: dace.SDFG, tag: str):
    """Run a private copy of ``sdfg`` on fixed seeded inputs; return the mutated output buffers."""
    candidate = copy.deepcopy(sdfg)
    candidate.name = f"coalesce_fixture_{tag}"
    rng = np.random.default_rng(0)
    shape = (FIXTURE_KLON, FIXTURE_KLEV)
    args = {
        "pt": rng.random(shape),
        "pq": rng.random(shape),
        "pa": rng.random(shape),
        "tend_t": np.zeros(shape),
        "tend_q": np.zeros(shape),
    }
    candidate(KLON=FIXTURE_KLON, KLEV=FIXTURE_KLEV, **args)
    return args["tend_t"], args["tend_q"]


def test_maximal_fusion_fuses_the_maps():
    """The map count strictly drops and the graph still validates."""
    sdfg = premapped()
    before = nmaps(sdfg)
    assert before > 1, "fixture has no adjacent maps -- fusion would have nothing to do"
    assert apply_fusion(sdfg), "maximal fusion reported no change"
    sdfg.validate()
    assert nmaps(sdfg) < before, f"no maps fused: {before} -> {nmaps(sdfg)}"


def test_maximal_fusion_does_not_change_the_numbers():
    """Inlining and fusion are value-preserving: bit-identical outputs, not merely close."""
    sdfg = premapped()
    reference = run_fixture(sdfg, "before")
    apply_fusion(sdfg)
    sdfg.validate()
    fused = run_fixture(sdfg, "after")
    for name, ref, out in zip(("tend_t", "tend_q"), reference, fused):
        assert np.array_equal(ref, out), f"{name} changed: max |diff| = {np.max(np.abs(ref - out))}"


def test_the_accumulation_reaching_fusion_needs_no_atomic():
    """The third sweep updates one element per iteration, so it reaches fusion as a plain read-modify-write:
    neither a WCR nor an atomic the write does not need."""
    code = premapped().generate_code()[0].code
    assert ">::reduce" not in code, "a per-element update reached fusion as a WCR"
    assert "reduction(" not in code, "a per-element update was emitted as an OpenMP reduction"


def test_reapplying_maximal_fusion_is_a_no_op():
    """Idempotence: one application fuses fully; the second leaves the graph bit-identical and reports no change."""
    sdfg = premapped()
    apply_fusion(sdfg)
    settled = sdfg.hash_sdfg()
    again = apply_fusion(sdfg)
    assert sdfg.hash_sdfg() == settled, "maximal fusion kept mutating a graph it had already fused"
    assert not again, "maximal fusion changed the graph a second time"


if __name__ == "__main__":
    test_maximal_fusion_fuses_the_maps()
    test_maximal_fusion_does_not_change_the_numbers()
    test_the_accumulation_reaching_fusion_needs_no_atomic()
    test_reapplying_maximal_fusion_is_a_no_op()
