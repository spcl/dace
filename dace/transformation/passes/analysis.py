# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from dace.transformation import pass_pipeline as ppl
from dace import SDFG, SDFGState
from typing import Any, Dict, Set, Tuple, Optional
import networkx as nx


class StateReachability(ppl.Pass):
    """
    Evaluates state reachability (which other states can be executed after each state).
    """
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.States

    def apply_pass(self, sdfg: SDFG, _) -> Dict[SDFGState, Set[SDFGState]]:
        """
        :return: A dictionary mapping each state to its other reachable states.
        """
        reachable: Dict[SDFGState, Set[SDFGState]] = {}
        tc: nx.DiGraph = nx.transitive_closure(sdfg.nx)
        for state in sdfg.nodes():
            reachable[state] = set(tc.successors(state))
        return reachable


class AccessSets(ppl.Pass):
    """
    Evaluates memory access sets (which arrays/data descriptors are read/written in each state).
    """
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified & ppl.Modifies.AccessNodes

    def apply_pass(self, sdfg: SDFG, _) -> Dict[SDFGState, Tuple[Set[str], Set[str]]]:
        """
        :return: A dictionary mapping each state to its other reachable states.
        """
        result: Dict[SDFGState, Tuple[Set[str], Set[str]]] = {}
        for state in sdfg.nodes():
            readset, writeset = set(), set()
            for anode in state.data_nodes():
                if state.in_degree(anode) > 0:
                    writeset.add(anode.data)
                if state.out_degree(anode) > 0:
                    readset.add(anode.data)

            result[state] = (readset, writeset)

        # Edges that read from arrays add to both ends' access sets
        anames = sdfg.arrays.keys()
        for e in sdfg.edges():
            fsyms = e.data.free_symbols & anames
            if fsyms:
                result[e.src][0].update(fsyms)
                result[e.dst][0].update(fsyms)
        return result
