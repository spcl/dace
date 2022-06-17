# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from dataclasses import dataclass
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes import analysis as ap
from dace import SDFG, SDFGState
from typing import Any, Dict, Set, Optional, Tuple, Type


@dataclass
class DeadDataflowElimination(ppl.Pass):
    """
    Removes unused computations from SDFG states.
    Traverses the graph backwards, removing any computations that result in transient descriptors
    that are not used again. Removal propagates through scopes (maps), tasklets, and optionally library nodes.
    """
    remove_library_nodes: bool = False  #: If True, removes library nodes if their results are unused (disabled by default as it could lead to removing side effects)
    remove_persistent_memory: bool = True  #: If True, marks code with Persistent allocation lifetime as dead

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If dataflow or states changed, new dead code may be exposed
        return modified & (ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.States)

    def depends_on(self) -> Set[Type[ppl.Pass]]:
        return {ap.StateReachability, ap.AccessSets}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[SDFGState, Set[str]]]:
        """
        Removes unreachable dataflow throughout SDFG states.
        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary mapping states to their removed descriptors, or None if nothing was changed.
        """
        reachable: Dict[SDFGState, Set[SDFGState]] = pipeline_results['StateReachability']
        access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]] = pipeline_results['AccessSets']
        result: Dict[SDFGState, Set[str]] = defaultdict(set)
        # Potentially depends on the following analysis passes:
        #  * State reachability
        #  * Read/write access sets per state

        # TODO: If a tasklet has any callbacks, mark as "live" due to possible side effects
        # TODO: If access node is persistent, mark as dead only if self.remove_persistent_memory is set
        return result or None
