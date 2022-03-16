# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import math

from typing import Generator, Dict, List, Tuple
from collections import Counter

from dace import SDFG, dtypes
from dace.optimization import cutout_tuner
from dace.sdfg.analysis import cutout as cutter

from dace.transformation import subgraph as sg
from dace.transformation.estimator import enumeration as en
from dace.transformation.subgraph import helpers
from dace.optimization import utils as optim_utils

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x


class SubgraphFusionTuner(cutout_tuner.CutoutTuner):

    def __init__(self, sdfg: SDFG, measurement: dtypes.InstrumentationType = dtypes.InstrumentationType.Timer) -> None:
        super().__init__(task="SubgraphFusion", sdfg=sdfg)
        self.instrument = measurement

    def cutouts(self, sdfg=None):
        if sdfg is None:
            sdfg = self._sdfg
            
        for state in sdfg.nodes():
            state_id = sdfg.node_id(state)
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

    def _transfer_apply(self, sdfg: dace.SDFG, patterns: List[Tuple[str, List[int]]]):
        # Describe successful fusions as set of map descriptors
        subgraph_patterns = []
        for label, config in patterns:
            state_id = label.split(".")[0]
            state_id = int(state_id)
            state = self._sdfg.node(state_id)
            nodes = state.nodes()
            cutout = cutter.cutout_state(state, *(nodes), make_copy=False)

            pattern_desc = Counter()
            fusion_id, map_ids = self.config_from_key(config, cutout)
            if fusion_id == 0:
                continue

            for map_id in map_ids:
                map_entry = cutout.start_state.node(map_id)
                map_desc = SubgraphFusionTuner.map_descriptor(cutout.start_state, map_entry)
                pattern_desc.update({map_desc: 1})

            subgraph_patterns.append(pattern_desc)

        cutouts = list(self.cutouts(sdfg))

        # Split work
        rank = optim_utils.get_world_rank()
        num_ranks = optim_utils.get_world_size()
        chunk_size = len(cutouts) // max(num_ranks, 1)
        chunks = list(optim_utils.partition(cutouts, chunk_size))

        if rank >= len(chunks):
            return

        # Find set of map descriptors in other sdfg
        chunk = chunks[rank]
        for cutout, label in tqdm(chunk):
            # Try to apply every subgraph_pattern greedily, i.e., highest expected speedup first
            for pattern in subgraph_patterns:
                maps = helpers.get_outermost_scope_maps(cutout, cutout.start_state)
                cutout_maps = {}
                cutout_desc = Counter()
                for map_entry in maps:
                    map_desc = SubgraphFusionTuner.map_descriptor(cutout.start_state, map_entry)
                    cutout_desc.update({map_desc: 1})
                    
                    if not map_desc in cutout_maps:
                        cutout_maps[map_desc] = []

                    cutout_maps[map_desc].append(map_entry)

                if cutout_desc != pattern:
                    continue

                # Construct subgraph greedily
                subgraph_maps = []
                for desc in pattern:
                    num = pattern[desc]
                    subgraph_maps.extend(cutout_maps[desc][:num])

                # 1. Check speedup on cutout
                # subgraph = helpers.subgraph_from_maps(sdfg=cutout, graph=cutout.start_state, map_entries=subgraph_maps)
                # map_fusion = sg.MapFusion(subgraph, cutout.sdfg_id, cutout.node_id(cutout.start_state))
                # if map_fusion.can_be_applied(cutout.start_state, cutout):                    
                #     # baseline_cutout = copy.deepcopy(cutout)
                #     # baseline_runtime = self.measure(baseline_cutout)

                #     fuse_counter = map_fusion.apply(cutout.start_state, cutout)
                #     if fuse_counter == 0:
                #         continue

                #     print(fuse_counter)

                #     # runtime = self.measure(cutout)
                #     # if runtime > baseline_runtime:
                #     #     # Reset to original cutout
                #     #     cutout = baseline_cutout
                #     #     continue

                # 2. Apply to actual SDFG
                state_id = int(label.split(".")[0])
                state = sdfg.node(state_id)
                subgraph = helpers.subgraph_from_maps(sdfg=sdfg, graph=state, map_entries=subgraph_maps)
                subgraph_fusion = sg.CompositeFusion(subgraph, sdfg.sdfg_id, state_id)
                subgraph_fusion.allow_tiling = True
                subgraph_fusion.schedule_innermaps = dace.ScheduleType.GPU_Device
                if subgraph_fusion.can_be_applied(sdfg, subgraph):
                    subgraph_fusion.apply(sdfg)

    @staticmethod
    def map_descriptor(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> str:
        tasklets = filter(lambda node: isinstance(node, dace.nodes.Tasklet), map(lambda edge: edge.dst, state.out_edges(map_entry)))
        tasklets = set(tasklets)

        desc = []
        for tasklet in tasklets:
            label = tasklet.label.split("_")[:-2]
            label = "_".join(label)
            desc.append(label)

        return ":".join(desc)
