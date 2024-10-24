# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""

import copy
from typing import Any, Dict
from dace.sdfg import SDFG, SDFGState
from dace.properties import DictProperty, make_properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.data import Structure, View
import re
from dace.transformation import pass_pipeline as ppl

@make_properties
class CleanMappedViews(ppl.Pass):
    access_names_map = DictProperty(key_type=str, value_type=str)

    def __init__(self, access_names_map):
        self.access_names_map = access_names_map
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> int:
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode):
                    if node.data in self.access_names_map:
                        node.data = self.access_names_map[node.data]
            for edge in state.edges():
                if edge.data.data in self.access_names_map:
                    edge.data.data = self.access_names_map[edge.data.data]
        return 0

    def annotates_memlets():
        return False
