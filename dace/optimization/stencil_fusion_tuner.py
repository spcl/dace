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

class StencilFusionTuner(cutout_tuner.CutoutTuner):

    def __init__(self, sdfg: SDFG, measurement: dtypes.InstrumentationType = dtypes.InstrumentationType.Timer) -> None:
        super().__init__(task="StencilFusion", sdfg=sdfg)
        self.instrument = measurement

    def cutouts(self):
        for state in self._sdfg.nodes():
            state_id = self._sdfg.node_id(state)
            yield (state_id, 0), (state, [])

            enumerator = en.ConnectedEnumerator(self._sdfg, state)
            for fusion_i, (maps, _) in enumerate(enumerator):
                if len(maps) > 1:
                    yield (state_id, fusion_i + 1), (state, maps)


    def space(self, subgraph: dace.SDFG) -> Generator[List[bool], None, None]:
        return []

    def evaluate(self, state: dace.SDFGState, maps: List, dreport: data_report.InstrumentedDataReport,
                 measurements: int, **kwargs) -> Dict:
        candidate = cutter.cutout_state(state, *(state.nodes()), make_copy=False)

        if len(maps) > 0:
            if len(maps) < 2:
                return math.inf

            # Check
            subgraph = helpers.subgraph_from_maps(candidate, candidate.start_state, maps)
            fusion = comp.CompositeFusion(subgraph, candidate.sdfg_id, candidate.node_id(candidate.start_state))
            fusion.allow_tiling = True
            if not fusion.can_be_applied(candidate, subgraph):
                return math.inf
            
            # Apply on copy
            candidate = cutter.cutout_state(state, *(state.nodes()), make_copy=True)
            maps_ = list(map(lambda m: candidate.start_state.node(state.node_id(m)), maps))
            subgraph = helpers.subgraph_from_maps(candidate, candidate.start_state, maps_)
            
            fusion = comp.CompositeFusion(subgraph, candidate.sdfg_id, candidate.node_id(candidate.start_state))
            fusion.allow_tiling = True
            fusion.apply(candidate)

        candidate.instrument = self.instrument
        arguments = {}
        for cstate in candidate.nodes():
            for dnode in cstate.data_nodes():
                if candidate.arrays[dnode.data].transient:
                    continue

                arguments[dnode.data] = np.copy(dreport.get_first_version(dnode.data))

        arguments = {**kwargs, **arguments}
        runtime = self.measure(candidate, arguments, measurements)
        return runtime

    def optimize(self, measurements: int = 30, **kwargs) -> Dict:
        dreport: data_report.InstrumentedDataReport = self._sdfg.get_instrumented_data()

        tuning_report = {}
        for candidate in tqdm(list(self.cutouts())):
            (state_id, candidate_id), (state, maps) = candidate
            fn = self.file_name(state_id, candidate_id, state.label)
            results = self.try_load(fn)

            if results is None:
                results = self.evaluate(state, maps, dreport, measurements, **kwargs)
                if results == math.inf:
                    continue

                with open(fn, 'w') as fp:
                    json.dump(results, fp)

            key = ".".join((str(state_id), str(candidate_id)))
            tuning_report[key] = results

        return tuning_report
