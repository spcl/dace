# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Precomputed symbol scopes for code generation.

:func:`~dace.sdfg.state.SDFGState.symbols_defined_at` answers "what symbols can this node see" by
replaying the whole chain per node: the SDFG's symbols, every array's free symbols, the interstate
edges, the enclosing loop variables, then the scope entries on the path down to the node. Only the
last step depends on the node, so a per-node call redoes an O(#arrays) sympy walk that is invariant
across the entire SDFG.

Symbols are *defined going down* -- a map entry binds its parameters and dynamic-range connectors, a
consume entry its PE index, a loop region its iterator -- so the natural shape is one top-down pass
that inherits the enclosing scope's table and adds what the scope itself binds. That is also what
correctness wants: ``new_symbols`` receives the accumulated table because it types a map range
against the symbols already in scope.

Valid only while the SDFG is frozen (see ``DaCeCodeGenerator.preprocess``); transformations that
mutate the graph must keep calling ``symbols_defined_at``.
"""
import collections
from typing import Dict, Optional, Union

from dace import dtypes
from dace.sdfg import nodes
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import LoopRegion, SDFGState

#: Per state, the symbols visible at each scope entry; ``None`` keys the state's own top level.
SymbolScopes = Dict[SDFGState, Dict[Optional[nodes.EntryNode], Dict[str, dtypes.typeclass]]]


def sdfg_symbols(sdfg: SDFG) -> Dict[str, dtypes.typeclass]:
    """Symbols visible anywhere in ``sdfg``: its own symbols, array extents, and interstate edges."""
    symbols = collections.OrderedDict(sdfg.symbols)
    for desc in sdfg.arrays.values():
        symbols.update([(str(s), s.dtype) for s in desc.free_symbols])

    # NOTE: this mirrors symbols_defined_at exactly, INCLUDING passing the start state to
    #  predecessor_state_transitions -- which walks backwards from it and so yields nothing on an
    #  acyclic CFG. Fixing that here would define symbols codegen does not currently see; it needs
    #  its own change and its own test.
    try:
        start_state = sdfg.start_state
        for e in sdfg.predecessor_state_transitions(start_state):
            symbols.update(e.data.new_symbols(sdfg, symbols))
    except ValueError:
        # Starting state is ambiguous (some interstate edges may not exist yet)
        for e in sdfg.edges():
            symbols.update(e.data.new_symbols(sdfg, symbols))
    return symbols


def state_symbols(state: SDFGState, base: Dict[str, dtypes.typeclass]) -> Dict[str, dtypes.typeclass]:
    """``base`` plus the iterators of the loop regions enclosing ``state``, outermost first."""
    symbols = collections.OrderedDict(base)
    enclosing_loops = []
    cfg = state.parent_graph
    while cfg is not None:
        if isinstance(cfg, LoopRegion) and cfg.loop_variable:
            enclosing_loops.append(cfg)
        cfg = cfg.parent_graph
    for loop in reversed(enclosing_loops):
        symbols.update(loop.new_symbols(symbols))
    return symbols


def symbol_scopes(top_sdfg: SDFG) -> SymbolScopes:
    """Symbols visible at every scope of every state, computed once per SDFG rather than per node."""
    scopes: SymbolScopes = {}
    for sdfg in top_sdfg.all_sdfgs_recursive():
        base = sdfg_symbols(sdfg)
        for state in sdfg.states():
            per_scope = {None: state_symbols(state, base)}
            children = state.scope_children()
            stack = [None]
            while stack:  # outer-to-inner, so each entry inherits a table its parent already finished
                parent = stack.pop()
                for node in children[parent]:
                    if not isinstance(node, nodes.EntryNode):
                        continue
                    symbols = collections.OrderedDict(per_scope[parent])
                    symbols.update(node.new_symbols(sdfg, state, symbols))
                    per_scope[node] = symbols
                    stack.append(node)
            scopes[state] = per_scope
    return scopes


def defined_at(scopes: SymbolScopes, state: SDFGState, node: Union[nodes.Node, None]) -> Dict[str, dtypes.typeclass]:
    """Table for ``node``, i.e. the one for its innermost enclosing scope entry."""
    if node is None:
        return collections.OrderedDict()
    return scopes[state][state.entry_node(node)]
