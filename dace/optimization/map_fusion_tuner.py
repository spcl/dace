# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import math
import json
import numpy as np

from typing import Generator, Dict, List, Tuple

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
            nodes = state.nodes()
            cutout = cutter.cutout_state(state, *(nodes), make_copy=False)
            yield cutout, f"{state_id}.{state.label}"

    def config_from_key(self, key: str, cutout: dace.SDFG, **kwargs) -> Tuple[int, List[int]]:
        fusion_id = int(key)
        if fusion_id == 0:
            return (0, [])
        
        sp = list(self.space(cutout=cutout))
        return sp[fusion_id]

    def apply(self, config: Tuple[int, List[int]], label: str, **kwargs) -> None:
        if config[0] == 0:
            return

        state_id = label.split(".")[0]
        state_id = int(state_id)
        state = self._sdfg.node(state_id)
        nodes = state.nodes()
        cutout = cutter.cutout_state(state, *(nodes), make_copy=False)

        map_ids = config[1]
        maps_ = map(lambda m: cutout.start_state.node(m), map_ids)
        subgraph = helpers.subgraph_from_maps(sdfg=self._sdfg, graph=state, map_entries=maps_)
        fusion = comp.CompositeFusion(subgraph, self._sdfg.sdfg_id, state_id)
        fusion.allow_tiling = True

        if not fusion.can_be_applied(self._sdfg, subgraph):
            raise ValueError("Invalid config")

        print(f"Fusing {len(map_ids)} maps in state {state.label}")
        fusion.apply(self._sdfg)

    def space(self, cutout: dace.SDFG) -> Generator[List[bool], None, None]:
        subgraphs = en.ConnectedEnumerator(cutout, cutout.start_state)
        yield 0, []
        
        for i, (subgraph, score) in enumerate(subgraphs):
            yield i + 1, list(map(lambda m: cutout.start_state.node_id(m), subgraph))

    def pre_evaluate(self, cutout: dace.SDFG, dreport: data_report.InstrumentedDataReport, measurements: int, **kwargs) -> Dict:
        cutout.instrument = self.instrument
        arguments = {}
        for cstate in cutout.nodes():
            for dnode in cstate.data_nodes():
                if cutout.arrays[dnode.data].transient:
                    continue

                arguments[dnode.data] = dreport.get_first_version(dnode.data)

        new_kwargs = {"space_kwargs": {"cutout": cutout}, "cutout": cutout.to_json(), "arguments": arguments, "measurements": measurements, "key": lambda point: str(point[0])}
        return new_kwargs

    def evaluate(self, config, cutout, arguments: Dict, measurements: int, **kwargs) -> float:
        cutout_ = dace.SDFG.from_json(cutout)
        map_ids = config[1]
        if config[0] == 0 and len(map_ids) == 0:
            # Baseline
            return self.measure(cutout_, arguments, measurements)

        if len(map_ids) < 2:
            return math.inf

        # Check
        maps_ = list(map(lambda m: cutout_.start_state.node(m), map_ids))
        subgraph = helpers.subgraph_from_maps(sdfg=cutout_, graph=cutout_.start_state, map_entries=maps_)
        fusion = comp.CompositeFusion(subgraph, cutout_.sdfg_id, cutout_.node_id(cutout_.start_state))
        fusion.allow_tiling = True
        if not fusion.can_be_applied(cutout_, subgraph):
            return math.inf
        
        # Apply on copy
        candidate = SDFG.from_json(cutout)
        maps_ = list(map(lambda m: candidate.start_state.node(m), map_ids))
        subgraph = helpers.subgraph_from_maps(sdfg=candidate, graph=candidate.start_state, map_entries=maps_)

        fusion = comp.CompositeFusion(subgraph, candidate.sdfg_id, candidate.node_id(candidate.start_state))
        fusion.allow_tiling = True
        fusion.apply(candidate)

        return self.measure(candidate, arguments, measurements)
