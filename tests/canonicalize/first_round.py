# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The canonicalize recipe with its fixpoints opened: a test that stops after one pass reads its output there."""

from dace.transformation.passes.canonicalize.pipeline import PhaseFixpoint, Stages, prepared


def first_round_stages(stages: Stages) -> Stages:
    """``stages`` with each top-level ``PhaseFixpoint`` replaced by one round of its units, under the same label.

    A test that stops right after one pass (``SplitStatements``) or slices the recipe at one reads its output
    there, and that pass may run inside a fixpoint.
    """
    flat: Stages = []
    for label, unit in stages:
        if isinstance(unit, PhaseFixpoint):
            flat += [(label, inner) for inner in prepared(unit.factory())]
        else:
            flat.append((label, unit))
    return flat
