# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import ast

from typing import Any, Generator, Optional

from dace import SDFG, nodes, data
from dace.dtypes import ScheduleType
from dace.sdfg.analysis import cutout as cutter
from dace.optimization.cutout_tuning.cutout_space import CutoutSpace
from dace.transformation import helpers as xfh


class ScheduleSpace(CutoutSpace):
    def name(self) -> str:
        return 'ScheduleSpace'

    def apply_config(self, cutout: SDFG, config: Any, make_copy: bool = True) -> Optional[SDFG]:
        if make_copy:
            cutout_ = copy.deepcopy(cutout)
        else:
            cutout_ = cutout

        param, target = config
        state_id, node_id = target
        map_entry = cutout_.node(state_id).node(node_id)

        map_entry.map.schedule = param

        return cutout_

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
        state_id = 0

        map_ids = []
        for node in cutout.start_state.nodes():
            if isinstance(node, nodes.MapEntry):
                map_ids.append(cutout.start_state.node_id(node))

        schedules = [ScheduleType.Sequential] * len(map_ids)
        for i in range(len(map_ids)):
            schedule = schedules[i].copy()
            schedule[i] = ScheduleType.CPU_Multicore

        for schedule in schedules:
            yield tuple(schedule), (state_id, map_ids)

    @staticmethod
    def top_map(cutout: SDFG):
        map_entry = None
        for node in cutout.start_state.nodes():
            if not isinstance(node, nodes.MapEntry) or xfh.get_parent_map(cutout.start_state, node) is not None:
                continue

            map_entry = node
            break

        return map_entry
