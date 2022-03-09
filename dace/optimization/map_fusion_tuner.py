# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import math
import json
import numpy as np

from typing import Generator, Dict, List

from dace import SDFG, dtypes
from dace.optimization import cutout_tuner
from dace.sdfg.analysis import cutout as cutter
from dace.codegen.instrumentation.data import data_report

from dace.transformation.subgraph import composite as comp
from dace.transformation.estimator import enumeration as en
from dace.transformation.subgraph import helpers

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x


class MapFusionTuner(cutout_tuner.CutoutTuner):

    def __init__(self, sdfg: SDFG, measurement: dtypes.InstrumentationType = dtypes.InstrumentationType.Timer) -> None:
        super().__init__(task="MapFusion", sdfg=sdfg)
        self.instrument = measurement

    def cutouts(self):
        for state in self._sdfg.nodes():
            state_id = self._sdfg.node_id(state)
            cutout = cutter.cutout_state(state, *(state.nodes()), make_copy=False)
            yield cutout, f"{state_id}.{state.label}"

    def space(self, cutout: dace.SDFG) -> Generator[List[bool], None, None]:
        enumerator = en.ConnectedEnumerator(cutout, cutout.start_state)
        return enumerator

    def search(self, cutout: dace.SDFG, dreport: data_report.InstrumentedDataReport, measurements: int,
                 **kwargs) -> Dict[str, float]:
        cutout.instrument = self.instrument

        arguments = {}
        for cstate in cutout.nodes():
            for dnode in cstate.data_nodes():
                if cutout.arrays[dnode.data].transient:
                    continue

                arguments[dnode.data] = np.copy(dreport.get_first_version(dnode.data))

        report = {}
        report[None] = self.measure(cutout, arguments, measurements)

        sdfg_json = cutout.to_json()
        for i, maps in tqdm(enumerate(list(self.space(cutout)))):
            if len(maps) < 2:
                report[str(i)] = math.inf
                continue

            # Check
            subgraph = helpers.subgraph_from_maps(sdfg=cutout, graph=cutout.start_state, map_entries=maps)
            fusion = comp.CompositeFusion(subgraph, cutout.sdfg_id, cutout.node_id(cutout.start_state))
            fusion.allow_tiling = True
            if not fusion.can_be_applied(cutout, subgraph):
                report[str(i)] = math.inf
                continue

            # Apply on copy
            candidate = SDFG.from_json(sdfg_json)
            maps_ = list(map(lambda m: candidate.start_state.node(cutout.start_state.node_id(m)), maps))
            subgraph = helpers.subgraph_from_maps(sdfg=candidate, graph=candidate.start_state, map_entries=maps_)

            fusion = comp.CompositeFusion(subgraph, candidate.sdfg_id, candidate.node_id(candidate.start_state))
            fusion.allow_tiling = True
            fusion.apply(candidate)

            report[str(i)] = self.measure(cutout, arguments, measurements)

        return report
