# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from dace.transformation import pass_pipeline as ppl
from dace.sdfg import utils as sdutil
from dace import SDFG
from typing import Optional

class ConsolidateEdges(ppl.Pass):
    """
    Removes extraneous edges with memlets that refer to the same data containers within the same scope.

    Memlet subsets are unioned in each scope. This effectively reduces the number of connectors and allows more
    transformations to be performed, at the cost of losing the individual
    per-tasklet memlets.
    """
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return (modified & ppl.Modifies.AccessNodes) or (modified & ppl.Modifies.Memlets)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """
        Consolidates edges on the given SDFG.
        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: Number of edges removed, or None if nothing was performed.
        """
        edges_removed = sdutil.consolidate_edges(sdfg)
        if edges_removed == 0:
            return None
        return edges_removed
