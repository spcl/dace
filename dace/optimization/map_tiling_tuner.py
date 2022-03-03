# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from typing import Generator, Tuple, Dict, List

from dace.optimization import cutout_tuner
from dace.transformation import dataflow as df
from dace.transformation import helpers as xfh
from dace.sdfg.analysis import cutout as cutter


class MapTilingTuner(cutout_tuner.CutoutTuner):

    def __init__(self, sdfg: dace.SDFG) -> None:
        super().__init__(sdfg=sdfg)

    def cutouts(self) -> Generator[Tuple[dace.SDFGState, dace.nodes.Node], None, None]:
        for node, state in self._sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry):
                if xfh.get_parent_map(state, node) is not None:
                    continue
                yield state, node

    def space(self, parent_map: dace.nodes.MapEntry) -> Generator[Tuple[int], None, None]:
        # TODO: choices
        choices = [
            (8, 8, 8),
            (16, 16, 16),
            (32, 32, 32),
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
        ]

        return choices

    def optimize(self, apply: bool = True, measurements: int = 30) -> Dict:
        dreport = self._sdfg.get_instrumented_data()

        tuning_report = {}
        for state, parent_map in self.cutouts():
            subgraph_nodes = state.scope_subgraph(parent_map).nodes()
            cutout = cutter.cutout_state(state, *subgraph_nodes)
            cutout.instrument = dace.InstrumentationType.Timer

            arguments = {}
            for cstate in cutout.nodes():
                for dnode in cstate.data_nodes():
                    if cutout.arrays[dnode.data].transient:
                        continue

                    arguments[dnode.data] = np.copy(dreport.get_first_version(dnode.data))

            results = {}
            baseline = self.measure(cutout, arguments, measurements)
            results[None] = baseline

            cutout_json = cutout.to_json()
            cutout_map_id = None
            for node in cutout.start_state.nodes():
                if isinstance(node, dace.nodes.MapEntry) and xfh.get_parent_map(cutout.start_state, node) is None:
                    cutout_map_id = cutout.start_state.node_id(node)
                    break

            best_choice = None
            best_runtime = baseline
            for point in self.space(parent_map):
                tiled_sdfg = dace.SDFG.from_json(cutout_json)
                tiled_map = tiled_sdfg.start_state.node(cutout_map_id)
                df.MapTiling.apply_to(tiled_sdfg, map_entry=tiled_map, options={"tile_sizes": point})

                runtime = self.measure(tiled_sdfg, arguments, measurements)
                results[point] = runtime

                if runtime < best_runtime:
                    best_runtime = runtime
                    best_choice = point

            if apply and best_choice is not None:
                df.MapTiling.apply_to(self._sdfg, map_entry=parent_map, options={"tile_sizes": best_choice})

            tuning_report[parent_map.label] = results

        # TODO: Tuning report to file format

        return tuning_report
