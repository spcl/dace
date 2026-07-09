# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Specialize a loop into ``if (condition) { parallel } else { sequential }``.

A loop data-parallel only under a runtime predicate (symbolic stride nonzero, wrap
offset below the modulus) is replaced by a two-way conditional: true branch = parallel
form, else = original sequential loop. Values satisfying the predicate take the fast
path; violating ones still compute correctly on the fallback. Contrast the abort-only
trap guard for whole-pipeline preconditions (symbol nonnegativity) with no sequential
alternative.

The parallel form comes from a caller-supplied callback mutating the true-branch clone
in place (lift to Map, split into affine segments, ...): specialization owns the
control-flow surgery, the callback owns what "parallel" means.
"""
import copy
from typing import Callable, Optional

from dace.properties import CodeBlock
from dace.sdfg import SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion


def _owner_sdfg(loop: LoopRegion) -> SDFG:
    """The top-level SDFG owning ``loop`` (walk up nested-SDFG parents)."""
    sd = loop.sdfg
    while sd.parent_sdfg is not None:
        sd = sd.parent_sdfg
    return sd


def specialize_loop_under_condition(
    loop: LoopRegion,
    condition,
    parallelize: Callable[[LoopRegion, ControlFlowRegion, SDFG], None],
    owner_sdfg: Optional[SDFG] = None,
    assume: bool = False,
) -> Optional[ConditionalBlock]:
    """Replace ``loop`` with ``if (condition) { parallel } else { original loop }``.

    :param loop: Loop to specialize; removed from its parent graph, deep-copied into each branch.
    :param condition: Predicate (``CodeBlock`` or C/Python string) under which the parallel
        form is valid -- the true-branch guard.
    :param parallelize: Callback ``(par_loop, par_region, owner)`` mutating the true-branch clone
        into its parallel form (run ``LoopToMap``, split into affine segments, ...). Runs after
        the conditional is spliced in, so ``par_loop.parent_graph is par_region`` with SDFG links set.
    :param owner_sdfg: Top-level SDFG (derived from ``loop`` if omitted).
    :param assume: When ``True``, the caller *asserts* ``condition`` always holds at runtime.
        Skip the two-way conditional entirely and parallelize ``loop`` in place -- emit only the
        parallel form, with no sequential fallback. Unsound if the condition is ever violated
        (the caller owns that contract); returns ``None`` since no ``ConditionalBlock`` is built.
    :returns: The inserted :class:`ConditionalBlock`, or ``None`` in ``assume`` mode.
    """
    if owner_sdfg is None:
        owner_sdfg = _owner_sdfg(loop)
    # assume mode: the condition is taken as always-true, so there is no fallback to build --
    # parallelize the original loop in its own parent graph and return (no ConditionalBlock).
    if assume:
        parallelize(loop, loop.parent_graph, owner_sdfg)
        return None
    if not isinstance(condition, CodeBlock):
        condition = CodeBlock(condition)

    parent = loop.parent_graph
    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))
    is_start = parent.start_block is loop

    conditional = ConditionalBlock(label=f'{loop.label}_specialize')

    # True branch: a clone the caller parallelizes. Else branch (condition ``None``):
    # a clone kept as sequential fallback. Deep-copy both (not re-host the original) to
    # keep re-parenting simple and leave no stale refs to the removed loop.
    par_loop = copy.deepcopy(loop)
    par_region = ControlFlowRegion(label=f'{loop.label}_par')
    par_region.add_node(par_loop, is_start_block=True, ensure_unique_name=True)
    conditional.add_branch(condition, par_region)

    seq_loop = copy.deepcopy(loop)
    # Pin the fallback sequential so no later parallelizer (LoopToMap / LoopToReduce)
    # lifts it back to a Map -- it exists for values violating ``condition`` and must
    # stay sequential.
    seq_loop.pinned_sequential = True
    seq_region = ControlFlowRegion(label=f'{loop.label}_seq')
    seq_region.add_node(seq_loop, is_start_block=True, ensure_unique_name=True)
    conditional.add_branch(None, seq_region)

    parent.add_node(conditional, ensure_unique_name=True)
    for e in in_edges:
        parent.add_edge(e.src, conditional, copy.deepcopy(e.data))
    for e in out_edges:
        parent.add_edge(conditional, e.dst, copy.deepcopy(e.data))
    for e in in_edges + out_edges:
        parent.remove_edge(e)
    parent.remove_node(loop)
    if is_start:
        parent.start_block = parent.node_id(conditional)
    parent.reset_cfg_list()

    parallelize(par_loop, par_region, owner_sdfg)
    return conditional


__all__ = ['specialize_loop_under_condition']
