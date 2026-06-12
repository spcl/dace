# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pre- and post-condition invariant checks for K-dim vectorization passes.

Per user direction 2026-06-12: ``To improve testing we can have post-condition
and pre-condition checks to each subpass implemented``.

Each K-dim pass overrides ``_pre_conditions`` and ``_post_conditions`` to
return a list of ``(description, predicate)`` tuples. The predicate
receives the SDFG and returns ``True`` when the invariant holds; on
violation, the pass raises :class:`AssertionError` with the description
and the offending node / edge / state printed.

Invariants are gated by the env var ``DACE_VEC_KDIM_VALIDATE=1`` so
production runs (without the var) skip the checks. Test runners + dev
should opt in to surface regressions early.

Reusable invariant checkers live below as module-level helpers; passes
import them by name to keep the per-pass invariant table compact.
"""
import os
from typing import Callable, List, Optional, Tuple

import dace
from dace.sdfg import SDFG
from dace.sdfg.nodes import AccessNode, NestedSDFG, Tasklet


_ENV_FLAG = "DACE_VEC_KDIM_VALIDATE"


def invariants_enabled() -> bool:
    """True iff invariant checks should run (``DACE_VEC_KDIM_VALIDATE=1``)."""
    return os.environ.get(_ENV_FLAG, "0") == "1"


# ---------------------------------------------------------------------------
# Generic structural invariants (composable across passes).
# ---------------------------------------------------------------------------


def _all_states_recursive(sdfg: SDFG):
    """Yield ``(sub_sdfg, state)`` for every state in ``sdfg`` and every
    nested SDFG reachable from it."""
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            yield sd, state


def no_memlet_dim_mismatch(sdfg: SDFG) -> Optional[str]:
    """Every memlet's ``subset`` and ``other_subset`` (when both present)
    must have matching dimensionality. Returns ``None`` on success or a
    description string identifying the first offender.
    """
    for sd, state in _all_states_recursive(sdfg):
        for edge in state.edges():
            mem = edge.data
            if mem is None:
                continue
            if mem.subset is None or mem.other_subset is None:
                continue
            if len(mem.subset.size()) != len(mem.other_subset.size()):
                return (f"{sd.name}.{state.label}: memlet ``{mem.data}`` subset dim={len(mem.subset.size())} "
                        f"!= other_subset dim={len(mem.other_subset.size())} on edge "
                        f"{type(edge.src).__name__}->{type(edge.dst).__name__}")
    return None


def no_isolated_access_nodes(sdfg: SDFG) -> Optional[str]:
    """No AccessNode may have zero in-edges AND zero out-edges in its
    state. Returns ``None`` on success or a description string for the
    first offender.
    """
    for sd, state in _all_states_recursive(sdfg):
        for node in state.nodes():
            if not isinstance(node, AccessNode):
                continue
            if state.in_degree(node) == 0 and state.out_degree(node) == 0:
                return f"{sd.name}.{state.label}: isolated AccessNode ``{node.data}``"
    return None


def no_duplicate_connector_edges(sdfg: SDFG) -> Optional[str]:
    """Every NSDFG / Tasklet / lib-node connector must have at most ONE
    incoming edge and at most ONE outgoing edge. Returns ``None`` on
    success or a description identifying the first offender.
    """
    for sd, state in _all_states_recursive(sdfg):
        for node in state.nodes():
            in_conns = {}
            for e in state.in_edges(node):
                if e.dst_conn is None:
                    continue
                in_conns.setdefault(e.dst_conn, 0)
                in_conns[e.dst_conn] += 1
            for conn, count in in_conns.items():
                if count > 1:
                    return (f"{sd.name}.{state.label}: {type(node).__name__} ``{getattr(node, 'label', node)}`` "
                            f"in-connector ``{conn}`` has {count} edges (max 1)")
            out_conns = {}
            for e in state.out_edges(node):
                if e.src_conn is None:
                    continue
                out_conns.setdefault(e.src_conn, 0)
                out_conns[e.src_conn] += 1
            for conn, count in out_conns.items():
                if count > 1:
                    return (f"{sd.name}.{state.label}: {type(node).__name__} ``{getattr(node, 'label', node)}`` "
                            f"out-connector ``{conn}`` has {count} edges (max 1)")
    return None


def sdfg_validates(sdfg: SDFG) -> Optional[str]:
    """The SDFG passes ``sdfg.validate()``. Returns ``None`` on success
    or the validator's error message.
    """
    try:
        sdfg.validate()
        return None
    except Exception as e:  # noqa: BLE001
        return f"SDFG validation failed: {type(e).__name__}: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Pass mixin.
# ---------------------------------------------------------------------------


class PrePostConditionMixin:
    """Mixin that wraps ``apply_pass`` with pre/post-condition checks.

    Subclasses override ``_pre_conditions`` and ``_post_conditions`` to
    return ``[(description, predicate), ...]``. Each predicate takes the
    SDFG and returns ``None`` on success or a string describing the
    violation; the mixin raises :class:`AssertionError` with the pass
    name + description on the first violation.

    Subclasses must implement ``_apply_pass`` with the actual work;
    ``apply_pass`` becomes a thin wrapper that runs pre-check ->
    ``_apply_pass`` -> post-check.

    Checks only run when :func:`invariants_enabled` returns ``True``
    (env var ``DACE_VEC_KDIM_VALIDATE=1``); otherwise ``apply_pass``
    passes through to ``_apply_pass``.
    """

    def _pre_conditions(self, sdfg: SDFG) -> List[Tuple[str, Callable[[SDFG], Optional[str]]]]:
        """Override to return ``[(description, predicate), ...]``. Each
        predicate takes the SDFG and returns ``None`` on success or a
        violation message.
        """
        return []

    def _post_conditions(self, sdfg: SDFG) -> List[Tuple[str, Callable[[SDFG], Optional[str]]]]:
        """Symmetric to ``_pre_conditions`` -- post-conditions checked
        AFTER ``_apply_pass`` runs."""
        return []

    def _check_conditions(self, sdfg: SDFG, kind: str) -> None:
        conds = self._pre_conditions(sdfg) if kind == "pre" else self._post_conditions(sdfg)
        for desc, predicate in conds:
            err = predicate(sdfg)
            if err is not None:
                raise AssertionError(f"{type(self).__name__} {kind}-condition violated -- {desc}: {err}")

    def apply_pass(self, sdfg, pipeline_results):
        """Thin wrapper around ``_apply_pass`` that runs pre/post-checks
        when the env var is set."""
        if invariants_enabled():
            self._check_conditions(sdfg, "pre")
        result = self._apply_pass(sdfg, pipeline_results)
        if invariants_enabled():
            self._check_conditions(sdfg, "post")
        return result
