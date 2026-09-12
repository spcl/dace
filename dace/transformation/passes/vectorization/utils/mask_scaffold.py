# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared scaffolding for the iteration-mask passes.

:class:`~dace.transformation.passes.vectorization.generate_iteration_mask.GenerateIterationMask`
(legacy, CPP-tasklet mask) and
:class:`~dace.transformation.passes.vectorization.generate_tile_iteration_mask.GenerateTileIterationMask`
(tile-lib-node mask) both need to (1) emit the mask producer in a state that
DOMINATES every (possibly branch-local) consumer, and (2) thread the bound
symbols the producer references into the body NestedSDFG. These two helpers
capture that shared structure so the two passes do not drift apart.
"""
from collections.abc import Callable, Iterable

import dace
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.nodes import NestedSDFG
from dace.transformation.passes.analysis import scopes


def prepend_dominating_init_state(sdfg: SDFG, label: str, build_producer: Callable[[SDFGState], None]) -> SDFGState:
    """Prepend a fresh start state to ``sdfg`` and populate it via ``build_producer``.

    The new state becomes the SDFG's sole start block -- so it dominates every
    other state -- and is wired to the FORMER entry with an interstate edge.
    ``build_producer(init_state)`` adds the producing node(s) + edges into the new
    state (a mask-fill tasklet or a ``TileMaskGen`` lib node). The iteration-mask
    passes use this to emit the mask where it dominates all of its consumers,
    including the per-branch reads of a data-dependent ``if`` (TileITE) body.

    Robust to an ambiguous / empty former entry: ``start_block`` raises when the
    SDFG has several disjoint source blocks, so we fall back to ``source_nodes()``
    and wire every former source (or none, for an empty SDFG).

    :param sdfg: SDFG to prepend the start state to.
    :param label: Label for the new start state.
    :param build_producer: Callback ``(init_state) -> None`` populating the state.
    :returns: The new start :class:`~dace.sdfg.state.SDFGState`.
    """
    # Capture the former entry BEFORE adding the new state -- the new state would
    # itself become a source node and make start_block / source_nodes ambiguous.
    # start_block raises when the entry is already ambiguous (several disjoint
    # sources); fall back to source_nodes() so we wire every former source (or
    # none, for an empty SDFG) instead of crashing.
    try:
        former_entries = [sdfg.start_block]
    except ValueError:
        former_entries = list(sdfg.source_nodes())
    init_state = sdfg.add_state(label, is_start_block=True)
    build_producer(init_state)
    for entry in former_entries:
        sdfg.add_edge(init_state, entry, dace.InterstateEdge())
    return init_state


def thread_symbols_into_nsdfg(inner_sdfg: SDFG, nsdfg_node: NestedSDFG, symbol_names: Iterable[str], parent_sdfg: SDFG,
                              parent_state: SDFGState, resolver: scopes.ScopedSymbolResolver) -> None:
    """Make each name in ``symbol_names`` visible inside ``inner_sdfg``.

    A bound symbol the mask producer references (a loop bound such as ``kfdia``)
    that the body NestedSDFG does not otherwise use is absent from both the inner
    SDFG's symbol table AND the NestedSDFG's ``symbol_mapping``, so the generated
    body function never receives it and the producer fails to compile
    (``'kfdia' was not declared``). Declare each name on ``inner_sdfg`` and
    identity-map it on ``nsdfg_node.symbol_mapping``.

    The dtype is RESOLVED at ``nsdfg_node``'s scope, not looked up in
    ``parent_sdfg.symbols``. Most of these names are the enclosing map's iteration
    variables, which no symbol table declares at all -- ``symbols_defined_at`` is
    the only source that sees a map parameter. Re-declaring one at a guessed
    ``int64`` makes the inner copy a SECOND symbol of the same name, so the mask's
    ``iv < ub`` comparison never folds against the parameter it was derived from.

    :param inner_sdfg: The body NestedSDFG's inner SDFG.
    :param nsdfg_node: The NestedSDFG node wrapping ``inner_sdfg``.
    :param symbol_names: Symbol-name strings to thread through.
    :param parent_sdfg: SDFG owning ``nsdfg_node``.
    :param parent_state: State holding ``nsdfg_node``; carries the enclosing map scope.
    :param resolver: The pass run's shared symbol resolver.
    :raises UndeterminedSymbolDType: when no rung of the ladder declares a threaded name.
    """
    for sname in symbol_names:
        if sname not in inner_sdfg.symbols:
            inner_sdfg.add_symbol(sname, resolver.resolve_dtype(sname, parent_sdfg, state=parent_state,
                                                                node=nsdfg_node))
        if sname not in nsdfg_node.symbol_mapping:
            nsdfg_node.symbol_mapping[sname] = dace.symbolic.pystr_to_symbolic(sname)
