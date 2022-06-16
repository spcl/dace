# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from dataclasses import dataclass
from dace.transformation import pass_pipeline as ppl
from dace import SDFG, SDFGState
from typing import Set, Optional


@dataclass
class DeadStateElimination(ppl.Pass):
    """
    Removes all unreachable states (e.g., due to a branch that will never be taken) from an SDFG.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If connectivity or any edges were changed, some more states might be dead
        return modified & (ppl.Modifies.InterstateEdges | ppl.Modifies.States)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[SDFGState]]:
        """
        Removes unreachable states throughout an SDFG.
        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :param initial_symbols: If not None, sets values of initial symbols.
        :return: A set of the removed states, or None if nothing was changed.
        """
        result: Set[SDFGState] = set()

        return result or None
