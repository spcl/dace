# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import ast

from typing import Any, Generator, Optional

from dace import SDFG, nodes
from dace.sdfg.analysis import cutout as cutter
from dace.optimization.cutout_tuning.cutout_space import CutoutSpace
from dace.transformation import helpers as xfh
from dace.transformation.dataflow import Vectorization


class VectorizationSpace(CutoutSpace):
    def name(self) -> str:
        return 'VectorizationSpace'

    def apply_config(self, cutout: SDFG, config: Any, make_copy: bool = True) -> Optional[SDFG]:
        if make_copy:
            cutout_ = copy.deepcopy(cutout)
        else:
            cutout_ = cutout

        param, target = config
        state_id, node_id = target
        state = cutout_.node(state_id)
        map_entry = state.node(node_id)

        vec = Vectorization()
        vec.map_entry = map_entry
        vec.vector_len = param
        if vec.can_be_applied(state, expr_index=0, sdfg=cutout_):
            vec.apply(state, cutout_)
            print("applied")
            return cutout_

        return None

    def translate_config(self, cutout: SDFG, sdfg: SDFG, config: Any) -> Any:
        param, target = config
        state_id, node_id = target
        map_entry = cutout.node(state_id).node(node_id)

        sstate_id = int(cutout.name.split("_")[-1])

        # TODO: How to find same node but on copy?
        snode_id = None
        for node in sdfg.node(sstate_id):
            if node.label == map_entry.label:
                snode_id = sdfg.node(sstate_id).node_id(node)
                break

        return param, (sstate_id, snode_id)

    def encode_config(self, config: Any) -> str:
        return str(config)

    def decode_config(self, config: str) -> Any:
        param, target = ast.literal_eval(config)
        return param, target

    def cutouts(self, sdfg: SDFG) -> Generator[SDFG, None, None]:
        for state in sdfg.nodes():
            for node in state.nodes():
                if not isinstance(node, nodes.MapEntry) or xfh.get_parent_map(state, node) is not None:
                    continue

                subgraph_nodes = state.scope_subgraph(node).nodes()
                cutout = cutter.cutout_state(state, *subgraph_nodes, make_copy=False)
                yield cutout

    def configurations(self, cutout: SDFG) -> Generator[Any, None, None]:
        map_entry = VectorizationSpace.top_map(cutout)

        state_id = 0
        node_id = cutout.start_state.node_id(map_entry)
        vector_lengths = [2, 4, 8]
        for vl in vector_lengths:
            yield vl, (state_id, node_id)

    @staticmethod
    def top_map(cutout: SDFG):
        map_entry = None
        for node in cutout.start_state.nodes():
            if not isinstance(node, nodes.MapEntry) or xfh.get_parent_map(cutout.start_state, node) is not None:
                continue

            map_entry = node
            break

        return map_entry
