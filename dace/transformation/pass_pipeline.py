# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
API for SDFG analysis and manipulation passes, as well as pipelines that contain multiple dependent passes.
"""
from dace.sdfg import SDFG, SDFGState, nodes

from enum import Flag, auto
from typing import Any, Dict, List, Optional, Set, Type, Union
from dataclasses import dataclass


class Modifies(Flag):
    Nothing = 0
    Descriptors = auto()
    Symbols = auto()
    States = auto()
    InterstateEdges = auto()
    AccessNodes = auto()
    Scopes = auto()
    Tasklets = auto()
    NestedSDFGs = auto()
    Memlets = auto()
    Nodes = AccessNodes | Scopes | Tasklets | NestedSDFGs
    Edges = InterstateEdges | Memlets
    Everything = Descriptors | Symbols | States | InterstateEdges | Nodes | Memlets


class Pass:
    def depends_on(self) -> Set[Type['Pass']]:
        return set()

    def modifies(self) -> Modifies:
        """
        :return: Flags of what this pass modifies.
        """
        raise NotImplementedError

    def should_reapply(self, modified: Modifies) -> bool:
        raise NotImplementedError

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        """
        :return: Some object if pass was applied, or None if nothing changed
        """
        raise NotImplementedError


class VisitorPass(Pass):
    def generic_visit(self, element: Any, parent: Any, pipeline_results: Dict[str, Any]) -> Any:
        return None

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        results = {}
        for node, parent in sdfg.all_nodes_recursive():
            f = getattr(self, f'visit_{type(node).__name__}', self.generic_visit)
            res = f(node, parent, pipeline_results)
            if res is not None:
                results[node] = res
        for edge, parent in sdfg.all_edges_recursive():
            f = getattr(self, f'visit_{type(edge).__name__}', self.generic_visit)
            res = f(edge, parent, pipeline_results)
            if res is not None:
                results[edge] = res

        if not results:
            return None
        return results


class StatePass(Pass):
    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[SDFGState, Optional[Any]]]:
        """
        :return: Some object if pass was applied, or None if nothing changed
        """
        result = {}
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.nodes():
                retval = self.apply(state, pipeline_results)
                if retval is not None:
                    result[state] = retval

        if not result:
            return None
        return result

    def apply(self, state: SDFGState, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        raise NotImplementedError


class ScopePass(Pass):
    def apply_pass(
        self,
        sdfg: SDFG,
        pipeline_results: Dict[str, Any],
    ) -> Optional[Dict[nodes.EntryNode, Optional[Any]]]:
        """
        :return: Some object if pass was applied, or None if nothing changed
        """
        result = {}
        for node, state in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.EntryNode):
                continue
            retval = self.apply(node, state, pipeline_results)
            if retval is not None:
                result[node] = retval

        if not result:
            return None
        return result

    def apply(self, scope: nodes.EntryNode, state: SDFGState, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        raise NotImplementedError


@dataclass
class Pipeline(Pass):
    passes: List[Pass]

    def __init__(self, passes: List[Pass]):
        # TODO: Add missing dependencies and compute graph
        self.passes = []
        self.passes.extend(passes)
        self._modified = Modifies.Nothing

    def modifies(self) -> Modifies:
        """
        :return: Flags of what this pass modifies.
        """
        result = Modifies.Nothing
        for p in self.passes:
            result |= p.modifies()

        return result

    def should_reapply(self, modified: Modifies) -> bool:
        return any(p.should_reapply(modified) for p in self.passes)

    def depends_on(self) -> Set[Type[Pass]]:
        result = set()
        for p in self.passes:
            result.update(p.depends_on())
        return result

    def iterate_over_passes(self):

        # TODO: make dependency graph and BFS over it
        # TODO: For every node, check if modified elements require reapplication of dependencies
        for p in self.passes:
            # TODO: if dependency node was visited and needs reapply (using self._modified)
            yield p

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        state = pipeline_results
        retval = {}
        self._modified = Modifies.Nothing
        for p in self.iterate_over_passes():
            r = p.apply_pass(sdfg, state)
            if r is not None:
                state[type(p).__name__] = r
                retval[type(p).__name__] = r
                self._modified |= p.modifies()

        if retval:
            return retval
        return None


class FixedPointPipeline(Pipeline):
    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        state = pipeline_results
        retval = {}
        while True:
            newret = super().apply_pass(sdfg, state)
            if newret is None:
                if retval:
                    return retval
                return None
            state.update(newret)
            retval.update(newret)
