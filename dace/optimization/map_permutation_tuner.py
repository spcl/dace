# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import math

import itertools
import numpy as np

from typing import Generator, Tuple, Dict

from dace import SDFG, dtypes
from dace.optimization import cutout_tuner
from dace.transformation import helpers as xfh
from dace.sdfg.analysis import cutout as cutter


class MapPermutationTuner(cutout_tuner.CutoutTuner):

    def __init__(self, sdfg: SDFG, measurement: dtypes.InstrumentationType = dtypes.InstrumentationType.Timer) -> None:
        super().__init__(sdfg=sdfg)
        self.instrument = measurement

    def cutouts(self) -> Generator[Tuple[dace.SDFGState, dace.nodes.Node], None, None]:
        for node, state in self._sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry):
                if xfh.get_parent_map(state, node) is not None:
                    continue
                yield state, node

    def space(self, parent_map: dace.nodes.MapEntry) -> Generator[Tuple[str], None, None]:
        return itertools.permutations(parent_map.map.params)

    def optimize(self, apply: bool = True, measurements: int = 30) -> Dict:
        dreport = self._sdfg.get_instrumented_data()

        tuning_report = {}
        for state, parent_map in self.cutouts():
            subgraph_nodes = state.scope_subgraph(parent_map).nodes()
            cutout = cutter.cutout_state(state, *subgraph_nodes)
            cutout.instrument = self.instrument

            arguments = {}
            for cstate in cutout.nodes():
                for dnode in cstate.data_nodes():
                    if cutout.arrays[dnode.data].transient:
                        continue

                    arguments[dnode.data] = np.copy(dreport.get_first_version(dnode.data))

            results = {}
            best_choice = None
            best_runtime = math.inf
            for point in self.space(parent_map):
                parent_map.range.ranges = [
                    r for list_param in point for map_param, r in zip(parent_map.map.params, parent_map.range.ranges)
                    if list_param == map_param
                ]
                parent_map.map.params = point

                runtime = self.measure(cutout, arguments, measurements)
                results[point] = runtime

                if runtime < best_runtime:
                    best_choice = point
                    best_runtime = runtime

            if apply and best_choice is not None:
                parent_map.range.ranges = [
                    r for list_param in best_choice
                    for map_param, r in zip(parent_map.map.params, parent_map.range.ranges) if list_param == map_param
                ]
                parent_map.map.params = best_choice

            tuning_report[parent_map.label] = results

        return tuning_report
