# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import pickle
import math
import copy
import numpy as np

from typing import Generator, Dict, List, Tuple
from collections import Counter

from dace import SDFG, dtypes
from dace.optimization import cutout_tuner
from dace.sdfg.analysis import cutout as cutter

from dace.transformation import subgraph as sg
from dace.transformation.estimator import enumeration as en
from dace.transformation.subgraph import helpers
from dace.transformation import helpers as xfh
from dace.optimization import utils as optim_utils

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x


class OnTheFlyMapFusionTuner(cutout_tuner.CutoutTuner):

    def __init__(self, sdfg: SDFG, measurement: dtypes.InstrumentationType = dtypes.InstrumentationType.Timer) -> None:
        super().__init__(task="OnTheFlyMapFusion", sdfg=sdfg)
        self.instrument = measurement

    def cutouts(self):
        for nsdfg_id, nsdfg in enumerate(self._sdfg.all_sdfgs_recursive()):
            for state in nsdfg.nodes(): 
                state_id = nsdfg.node_id(state)
                nodes = state.nodes()

                try:
                    cutout = cutter.cutout_state(state, *(nodes), make_copy=False)
                    yield cutout, f"{nsdfg_id}.{state_id}.{state.label}"
                except AttributeError:
                    continue

    def config_from_key(self, key: str, cutout: dace.SDFG, **kwargs) -> Tuple[int, List[int]]:
        fusion_id = int(key)
        if fusion_id == 0:
            return (0, [])

        sp = list(self.space(cutout=cutout))
        return sp[fusion_id]

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
        dreport = self._sdfg.get_instrumented_data()

        candidate = dace.SDFG.from_json(cutout)
        for node in candidate.start_state:
            if isinstance(node, dace.nodes.MapEntry):
                break
        else:
            # Skip no-map-states
            return math.inf

        
        if config[0] == 0:
            # Baseline
            return self.measure(candidate, dreport, measurements)

        map_ids = config[1]
        if len(map_ids) < 2:
            return math.inf

        maps_ = list(map(candidate.start_state.node, map_ids))
        subgraph = helpers.subgraph_from_maps(sdfg=candidate, graph=candidate.start_state, map_entries=maps_)

        map_fusion = sg.MapFusion(subgraph, candidate.sdfg_id, candidate.node_id(candidate.start_state))
        if map_fusion.can_be_applied(candidate.start_state, candidate):
            fuse_counter = map_fusion.apply(candidate.start_state, candidate)

            if fuse_counter == 0:
                return math.inf

        return self.measure(candidate, dreport,  measurements)

    def apply(self, config: Tuple[int, List[int]], label: str, **kwargs) -> None:
        if config[0] == 0:
            return

        nsdfg_id, state_id, state_label = label.split(".")
        nsdfg_id = int(nsdfg_id)
        state_id = int(state_id)
        sdfg = list(self._sdfg.all_sdfgs_recursive())[nsdfg_id]
        state = sdfg.node(state_id)
        nodes = state.nodes()
        cutout = cutter.cutout_state(state, *(nodes), make_copy=False)

        map_ids = config[1]
        maps_ = list(map(cutout.start_state.node, map_ids))
        subgraph = helpers.subgraph_from_maps(sdfg=sdfg, graph=state, map_entries=maps_)

        map_fusion = sg.MapFusion(subgraph, sdfg.sdfg_id, state_id)
        if map_fusion.can_be_applied(state, sdfg):
            fuse_counter = map_fusion.apply(state, sdfg)
            print(f"Fusing {fuse_counter} maps")

    def _extract_patterns(self, configs: List[Tuple[str, List[int]]]):
        # Describe successful fusions as set of map descriptors
        subgraph_patterns = []
        for label, config in configs:
            nsdfg_id, state_id, _ = label.split(".")
            nsdfg_id = int(nsdfg_id)
            state_id = int(state_id)
            state = list(self._sdfg.all_sdfgs_recursive())[nsdfg_id].node(state_id)
            nodes = state.nodes()
            cutout = cutter.cutout_state(state, *(nodes), make_copy=False)

            pattern_desc = Counter()
            fusion_id, map_ids = self.config_from_key(config, cutout)
            if fusion_id == 0:
                continue

            for map_id in map_ids:
                map_entry = cutout.start_state.node(map_id)
                map_desc = OnTheFlyMapFusionTuner.map_descriptor(cutout.start_state, map_entry)
                pattern_desc.update({map_desc: 1})

            subgraph_patterns.append(pattern_desc)

        return subgraph_patterns

    @staticmethod
    def transfer(sdfg: dace.SDFG, tuner, k: int = 5):
        assert isinstance(tuner, OnTheFlyMapFusionTuner)

        dreport = sdfg.get_instrumented_data()
        assert dreport is not None

        dreport_bytes = pickle.dumps(dreport)

        tuning_report = tuner.optimize(apply=False)
        best_configs = cutout_tuner.CutoutTuner.top_k_configs(tuning_report, k=k)
        subgraph_patterns = tuner._extract_patterns(best_configs)
        i = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                print(i, state.label)
                i = i + 1            

                top_maps = []
                for node in state.nodes():
                    if isinstance(node, dace.nodes.MapEntry) and xfh.get_parent_map(state, node) is None:
                        top_maps.append(node)
                
                if len(top_maps) < 2:
                    continue
                
                try:
                    cutout = cutter.cutout_state(state, *(state.nodes()), make_copy=False)
                except AttributeError as e:
                    print(e)
                    continue

                # Try to apply every subgraph_pattern greedily, i.e., highest expected speedup first
                base_runtime = None
                for j, pattern in enumerate(subgraph_patterns):
                    maps = []
                    for node in state.nodes():
                        if isinstance(node, dace.nodes.MapEntry) and xfh.get_parent_map(state, node) is None:
                            maps.append(node)

                    if len(maps) < 2:
                        continue

                    maps_desc = {}
                    state_desc = Counter()
                    for map_entry in maps:
                        map_desc = OnTheFlyMapFusionTuner.map_descriptor(state, map_entry)
                        state_desc.update({map_desc: 1})
                    
                        if not map_desc in maps_desc:
                            maps_desc[map_desc] = []

                        maps_desc[map_desc].append(map_entry)
                
                    included = True
                    for key in pattern:
                        if not key in state_desc or pattern[key] > state_desc[key]:
                            included = False                        
                            break

                    if not included:
                        continue

                    # Construct subgraph greedily
                    subgraph_maps = []
                    for desc in pattern:
                        num = pattern[desc]
                        subgraph_maps.extend(maps_desc[desc][:num])

                    # Apply
                    experiment_sdfg_ = cutter.cutout_state(state, *(state.nodes()), make_copy=False)
                    experiment_state_ = experiment_sdfg_.start_state
                    experiment_maps_ids = list(map(lambda me: experiment_state_.node_id(me), subgraph_maps))
                    
                    experiment_sdfg = copy.deepcopy(experiment_sdfg_)
                    experiment_state = experiment_sdfg.start_state
                    experiment_state.instrument = dace.InstrumentationType.GPU_Events
                    
                    experiment_maps = list(map(lambda m_id: experiment_state.node(m_id), experiment_maps_ids))
                    experiment_subgraph = helpers.subgraph_from_maps(sdfg=experiment_sdfg, graph=experiment_state, map_entries=experiment_maps)
                
                    map_fusion = sg.MapFusion(experiment_subgraph, experiment_sdfg.sdfg_id, experiment_sdfg.node_id(experiment_state))
                    if map_fusion.can_be_applied(experiment_state, experiment_sdfg):
                        print("Comparing")
                        if base_runtime is None:
                            baseline = cutter.cutout_state(experiment_state, *(experiment_state.nodes()), make_copy=True)                    
                            baseline.start_state.instrument = dace.InstrumentationType.GPU_Events
                            base_runtime = optim_utils.subprocess_measure(baseline, dreport_bytes)
                            if base_runtime == math.inf:
                                break
                        try:
                            experiment_fuse_counter = map_fusion.apply(experiment_state, experiment_sdfg)
                        except:
                            continue
                        
                        if experiment_fuse_counter == 0:
                            continue

                        fused_runtime = optim_utils.subprocess_measure(experiment_sdfg, dreport_bytes)
                        print(base_runtime, fused_runtime, experiment_fuse_counter)
                        if fused_runtime > base_runtime:
                            continue

                        print(j, pattern)
                        print(f"Fusing {experiment_fuse_counter} maps. Performance improvement: {base_runtime - fused_runtime}")

                        subgraph = helpers.subgraph_from_maps(sdfg=nsdfg, graph=state, map_entries=subgraph_maps)
                        map_fusion = sg.MapFusion(subgraph, nsdfg.sdfg_id, nsdfg.node_id(state))
                        actual_fuse_counter = map_fusion.apply(state, nsdfg)
                        assert actual_fuse_counter == experiment_fuse_counter

                        base_runtime = fused_runtime

                print()
                print()

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
