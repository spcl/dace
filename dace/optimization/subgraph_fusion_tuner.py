# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import math

from typing import Generator, Dict, List, Tuple

from dace import SDFG, dtypes
from dace.optimization import cutout_tuner
from dace.sdfg.analysis import cutout as cutter
from dace.codegen.instrumentation.data import data_report

from dace.transformation import subgraph as sg
from dace.transformation.estimator import enumeration as en
from dace.transformation.subgraph import helpers

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x


class SubgraphFusionTuner(cutout_tuner.CutoutTuner):

    def __init__(self, sdfg: SDFG, measurement: dtypes.InstrumentationType = dtypes.InstrumentationType.Timer) -> None:
        super().__init__(task="SubgraphFusion", sdfg=sdfg)
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
        maps_ = list(map(cutout.start_state.node, map_ids))
        subgraph = helpers.subgraph_from_maps(sdfg=self._sdfg, graph=state, map_entries=maps_)

        subgraph_fusion = sg.CompositeFusion(subgraph, self._sdfg.sdfg_id, state_id)
        subgraph_fusion.allow_tiling = True
        subgraph_fusion.schedule_innermaps = dace.ScheduleType.GPU_Device
        if subgraph_fusion.can_be_applied(self._sdfg, subgraph):
            subgraph_fusion.apply(self._sdfg)

    def space(self, cutout: dace.SDFG) -> Generator[List[bool], None, None]:
        subgraphs = en.ConnectedEnumerator(cutout, cutout.start_state)
        yield 0, []

        for i, (subgraph, score) in enumerate(subgraphs):
            yield i + 1, list(map(lambda m: cutout.start_state.node_id(m), subgraph))

    def pre_evaluate(self, cutout: dace.SDFG, measurements: int, **kwargs) -> Dict:
        cutout.start_state.instrument = self.instrument

        new_kwargs = {
            "space_kwargs": {
                "cutout": cutout
            },
            "cutout": cutout.to_json(),
            "measurements": measurements,
            "key": lambda point: str(point[0])
        }
        return new_kwargs

    def evaluate(self, config, cutout, measurements: int, **kwargs) -> float:
        candidate = dace.SDFG.from_json(cutout)
        for node in candidate.start_state:
            if isinstance(node, dace.nodes.MapEntry):
                break
        else:
            # Skip no-map-states
            return math.inf

        if config[0] == 0:
            # Baseline
            return self.measure(candidate, measurements)

        map_ids = config[1]
        if len(map_ids) < 2:
            return math.inf

        maps_ = list(map(candidate.start_state.node, map_ids))
        subgraph = helpers.subgraph_from_maps(sdfg=candidate, graph=candidate.start_state, map_entries=maps_)

        subgraph_fusion = sg.CompositeFusion(subgraph, candidate.sdfg_id, candidate.node_id(candidate.start_state))
        subgraph_fusion.allow_tiling = True
        subgraph_fusion.schedule_innermaps = dace.ScheduleType.GPU_Device
        if subgraph_fusion.can_be_applied(candidate, subgraph):
            subgraph_fusion.apply(candidate)
        else:
            return math.inf

        candidate.save(f"{candidate.start_state.label}_{config[0]}.sdfg")
        return self.measure(candidate, measurements)
