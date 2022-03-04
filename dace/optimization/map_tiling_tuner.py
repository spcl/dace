# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from typing import Generator, Tuple, Dict

from dace import SDFG, dtypes
from dace.optimization import cutout_tuner
from dace.transformation import dataflow as df
from dace.transformation import helpers as xfh
from dace.sdfg.analysis import cutout as cutter
from dace.codegen.instrumentation.data import data_report

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x

class MapTilingTuner(cutout_tuner.CutoutTuner):
    def __init__(self, sdfg: SDFG, measurement: dtypes.InstrumentationType = dtypes.InstrumentationType.Timer) -> None:
        super().__init__(task="MapTiling", sdfg=sdfg)
        self.instrument = measurement

    def cutouts(self) -> Generator[Tuple[dace.SDFGState, dace.nodes.Node], None, None]:
        for node, state in self._sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry):
                if xfh.get_parent_map(state, node) is not None:
                    continue
                
                node_id = state.node_id(node)
                state_id = self._sdfg.node_id(state)
                yield (state_id, node_id), (state, node)

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

    def evaluate(self, state: dace.SDFGState, node: dace.nodes.MapEntry, dreport: data_report.InstrumentedDataReport, measurements: int) -> Dict:
        subgraph_nodes = state.scope_subgraph(node).nodes()
        cutout = cutter.cutout_state(state, *subgraph_nodes)
        cutout.instrument = self.instrument

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
        # TODO: No copy for cutout will fix
        for node in cutout.start_state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and xfh.get_parent_map(cutout.start_state, node) is None:
                cutout_map_id = cutout.start_state.node_id(node)
                break

        for point in tqdm(list(self.space(node))):
            tiled_sdfg = dace.SDFG.from_json(cutout_json)
            tiled_map = tiled_sdfg.start_state.node(cutout_map_id)
            df.MapTiling.apply_to(tiled_sdfg, map_entry=tiled_map, options={"tile_sizes": point})

            runtime = self.measure(tiled_sdfg, arguments, measurements)
            
            key = ".".join(map(lambda p: str(p), point))
            results[key] = runtime

        return results
