# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Precomputed lookup tables for :meth:`DaCeCodeGenerator.determine_allocation_lifetime`.

That routine decides, per transient, where to declare/allocate/deallocate it. Its scans are written
per descriptor, so each one re-walks the whole SDFG: which states hold the container, whether it
appears in an interstate edge or a block condition, and every state's scope dictionary. With ``D``
descriptors over ``S`` states of ``N`` nodes that is O(D*S*N), and it is 31% of code generation.

The worst offender is loop-invariant in a stronger sense: ``cfg.used_symbols(...)`` parses conditions
through sympy/ast and does not depend on the descriptor at all -- only the ``name in`` test does. It
runs D*|CFGs| times to answer |CFGs| questions.

This module only builds dictionaries; every allocation decision stays in ``framecode``. Tables are
keyed and ORDERED exactly like the scans they replace (``sdfg.states()`` order, not topological), so
substituting a lookup cannot change which state an allocation lands in.

Valid only while the SDFG is frozen (see ``DaCeCodeGenerator.preprocess``).
"""
import collections
from dataclasses import dataclass, field
from typing import Dict, List, Set

from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState


@dataclass
class AllocationScopes:
    """Lookup tables for the allocation-lifetime decision. Pure analysis: no decisions, no mutation."""

    #: cfg_id -> data name -> states holding an AccessNode with exactly that name, in ``sdfg.states()`` order.
    data_states: Dict[int, Dict[str, List[SDFGState]]] = field(default_factory=dict)

    #: cfg_id -> root data name -> states holding an AccessNode whose ``root_data`` is that name.
    root_data_states: Dict[int, Dict[str, Set[SDFGState]]] = field(default_factory=dict)

    #: cfg_id -> every symbol named by an interstate edge or a block condition. Descriptor-independent.
    meta_symbols: Dict[int, Set[str]] = field(default_factory=dict)

    #: state -> its scope dictionary, built once instead of once per (descriptor, state).
    scope_dicts: Dict[SDFGState, Dict] = field(default_factory=dict)

    def uses_data(self, cfg_id: int, name: str) -> List[SDFGState]:
        """States containing an AccessNode named exactly ``name``."""
        return self.data_states[cfg_id].get(name, [])

    def uses_root_data(self, cfg_id: int, name: str) -> Set[SDFGState]:
        """States containing an AccessNode whose ``root_data`` is ``name``."""
        return self.root_data_states[cfg_id].get(name, frozenset())

    def in_meta_code(self, cfg_id: int, name: str) -> bool:
        """True if ``name`` is referenced by an interstate edge or a control-flow condition."""
        return name in self.meta_symbols[cfg_id]


def allocation_scopes(top_sdfg: SDFG, free_symbols) -> AllocationScopes:
    """Build the tables. ``free_symbols`` is the frame codegen's memoized free-symbol helper."""
    scopes = AllocationScopes()
    for sdfg in top_sdfg.all_sdfgs_recursive():
        cfg_id = sdfg.cfg_id
        by_data: Dict[str, List[SDFGState]] = collections.defaultdict(list)
        by_root: Dict[str, Set[SDFGState]] = collections.defaultdict(set)

        # sdfg.states() order, matching the scans this replaces
        for state in sdfg.states():
            scopes.scope_dicts[state] = state.scope_dict()
            seen: Set[str] = set()
            for node in state.data_nodes():
                if node.data not in seen:
                    seen.add(node.data)
                    by_data[node.data].append(state)
                by_root[node.root_data].add(state)

        # Descriptor-independent: computed once per SDFG rather than once per (descriptor, CFG)
        meta: Set[str] = set()
        for isedge in sdfg.all_interstate_edges():
            meta |= free_symbols(isedge.data)
        for cfg in sdfg.all_control_flow_regions():
            meta |= cfg.used_symbols(all_symbols=True, with_contents=False)

        scopes.data_states[cfg_id] = by_data
        scopes.root_data_states[cfg_id] = by_root
        scopes.meta_symbols[cfg_id] = meta
    return scopes
