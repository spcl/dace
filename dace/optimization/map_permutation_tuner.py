# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace

import itertools
import numpy as np

from typing import Generator, Tuple, Dict, List

from dace import SDFG, dtypes
from dace.optimization import cutout_tuner
from dace.transformation import helpers as xfh
from dace.sdfg.analysis import cutout as cutter
from dace.codegen.instrumentation.data import data_report

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x


class MapPermutationTuner(cutout_tuner.CutoutTuner):

    def __init__(self, sdfg: SDFG, measurement: dtypes.InstrumentationType = dtypes.InstrumentationType.Timer) -> None:
        super().__init__(task="MapPermutation", sdfg=sdfg)
        self.instrument = measurement

    def cutouts(self) -> Generator[Tuple[dace.SDFGState, str], None, None]:
        for node, state in self._sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry):
                if xfh.get_parent_map(state, node) is not None:
                    continue

                node_id = state.node_id(node)
                state_id = self._sdfg.node_id(state)
                subgraph_nodes = state.scope_subgraph(node).nodes()
                cutout = cutter.cutout_state(state, *subgraph_nodes, make_copy=False)
                yield cutout, f"{state_id}.{node_id}.{node.label}"

    def space(self, map_entry: dace.nodes.MapEntry, **kwargs) -> Generator[Tuple[str], None, None]:
        return itertools.permutations(map_entry.map.params)

    def config_from_key(self, key: str, **kwargs) -> List[str]:
        return key.split(".")

    def apply(self, config: List[str], label: str, **kwargs) -> None:
        state_id, node_id, node_label = label.split(".")
        map_entry = self._sdfg.node(int(state_id)).node(int(node_id))
        
        map_entry.range.ranges = [
            r for list_param in config for map_param, r in zip(map_entry.map.params, map_entry.range.ranges)
            if list_param == map_param
        ]
        map_entry.map.params = config

    def pre_evaluate(self, cutout: dace.SDFG, dreport: data_report.InstrumentedDataReport, measurements: int, **kwargs) -> Dict:
        cutout.instrument = self.instrument
        arguments = {}
        for dnode in cutout.start_state.data_nodes():
            if cutout.arrays[dnode.data].transient:
                continue

            arguments[dnode.data] = dreport.get_first_version(dnode.data)

        map_entry = None
        for node in cutout.start_state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and xfh.get_parent_map(cutout.start_state, node) is None:
                map_entry = node
                break
        assert map_entry is not None
                
        new_kwargs = {"space_kwargs": {"map_entry": map_entry}, "cutout": cutout.to_json(), "map_entry_id": cutout.start_state.node_id(map_entry), "arguments": arguments, "measurements": measurements, "key": lambda point: ".".join(point)}
        return new_kwargs

    def evaluate(self, config, cutout, map_entry_id: int, arguments: Dict, measurements: int, **kwargs) -> float:
        cutout_ = dace.SDFG.from_json(cutout)
        map_ = cutout_.start_state.node(map_entry_id)

        map_.range.ranges = [
            r for list_param in config for map_param, r in zip(map_.map.params, map_.range.ranges)
            if list_param == map_param
        ]
        map_.map.params = config

        return self.measure(cutout_, arguments, measurements)
