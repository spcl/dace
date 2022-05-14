# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import copy
import itertools
import numpy as np

from typing import Any, Generator, Optional

from dace import SDFG, nodes
from dace.transformation import helpers as xfh
from dace.sdfg.analysis import cutout as cutter
from dace.optimization.cutout_tuning.cutout_space import CutoutSpace


class MapPermutationSpace(CutoutSpace):
    def name(self) -> str:
        return 'MapPermutationSpace'

    def apply_config(self, cutout: SDFG, config: Any, make_copy: bool = True) -> Optional[SDFG]:
        if make_copy:
            cutout_ = copy.deepcopy(cutout)
        else:
            cutout_ = cutout

        param, target = config
        state_id, node_id = target
        map_entry = cutout_.node(state_id).node(node_id)

        new_order = [map_entry.map.params[i] for i in param]
        map_entry.range.ranges = [
            r for list_param in new_order for map_param, r in zip(map_entry.map.params, map_entry.range.ranges)
            if list_param == map_param
        ]
        map_entry.map.params = new_order

        return cutout_

    def translate_config(self, cutout: SDFG, sdfg: SDFG, config: Any) -> Any:
        cutout_hash = cutout.hash_sdfg()
        sdfg_hash = sdfg.hash_sdfg()
        if cutout_hash == sdfg_hash:
            return config

        param, target = config
        state_id, node_id = target
        map_entry = cutout.node(state_id).node(node_id)

        # TODO: Find same node on copy
        sstate_id = int(cutout.name.split("_")[-1])
        snode_id = None
        for node in sdfg.node(sstate_id):
            if node.label == map_entry.label:
                snode_id = sdfg.node(sstate_id).node_id(node)
                break

        translated_target = (sstate_id, snode_id)
        return param, translated_target

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
        map_entry = MapPermutationSpace.top_map(cutout)
        state_id = 0
        node_id = cutout.start_state.node_id(map_entry)

        positions = tuple(range(len(map_entry.map.params)))
        for param in itertools.permutations(positions):
            if param == positions:
                continue

            yield param, (state_id, node_id)

    @staticmethod
    def top_map(cutout: SDFG):
        map_entry = None
        for node in cutout.start_state.nodes():
            if not isinstance(node, nodes.MapEntry) or xfh.get_parent_map(cutout.start_state, node) is not None:
                continue

            map_entry = node
            break

        return map_entry
