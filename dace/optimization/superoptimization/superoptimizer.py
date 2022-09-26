import ast
import re
import copy
import math
import time
import json
import itertools

import dace.transformation.helpers as xfh
import dace.optimization.superoptimization.utils as utils
import dace.sdfg.analysis.cutout as cutter

from dace.optimization import auto_tuner

from pathlib import Path
from typing import Union, Dict, Tuple, List
from tqdm import tqdm

from dace.sdfg.sdfg import SDFG, SDFGState
from dace import DeviceType, ScheduleType, nodes

from dace.transformation.auto.auto_optimize import (
    set_fast_implementations,
    make_transients_persistent,
    move_small_arrays_to_stack,
)

from dace.transformation import PatternTransformation
from dace.transformation.dataflow import (InlineNestedReduce, RedundantSecondArray, TrivialMapElimination, MapCollapse,
                                          OTFMapFusion)

from dace.codegen.instrumentation.data.data_report import InstrumentedDataReport

from dace.optimization.measure import measure, random_arguments, create_data_report, arguments_from_data_report
from dace.optimization.superoptimization.enumerator import map_fusion_enumerator, map_schedule_enumerator

MINIMUM_SPEEDUP = 1.05
LOWER_BOUND_SCALING = 1.2


class Superoptimizer(auto_tuner.AutoTuner):

    def __init__(self,
                 sdfg: SDFG,
                 device_type: DeviceType = DeviceType.CPU,
                 measurements: int = 20,
                 warmup: int = 10) -> None:
        super().__init__(sdfg)
        assert device_type == DeviceType.CPU
        self._device_type = device_type
        self._profile = {
            'SubgraphOpt': {},
            'MapOpt': {},
        }

        self._measurements = measurements
        self._warmup = warmup

        # Create cache directories
        self._build_folder = sdfg.build_folder
        self._cache_folder = Path(sdfg.build_folder) / "tuning"
        self._cache_folder.mkdir(parents=True, exist_ok=True)

        self._map_schedule_cache_folder = self._cache_folder / "map_schedule"
        self._map_schedule_cache_folder.mkdir(parents=False, exist_ok=True)

        self._map_fusion_cache_folder = self._cache_folder / "map_fusion"
        self._map_fusion_cache_folder.mkdir(parents=False, exist_ok=True)

    def _register_profile_event(self, name: str):
        self._profile[name] = time.perf_counter() - self._event_start_ts
        self._event_start_ts = time.perf_counter()

    def tune(self, apply: bool = True, compile_folder: Union[str, Path] = None) -> SDFG:
        """
        
        """
        tuning_start = time.perf_counter()
        self._event_start_ts = time.perf_counter()

        dreport = self._sdfg.get_instrumented_data()
        if dreport is None:
            print("No data report available. Aborting")
            return sdfg

        self._register_profile_event('get_instr')

        if not apply:
            sdfg = copy.deepcopy(self._sdfg)
        else:
            sdfg = self._sdfg

        if not compile_folder is None:
            sdfg.build_folder = compile_folder

        self._register_profile_event('initial_copy')

        print("Measuring initial runtime")
        arguments = arguments_from_data_report(sdfg, data_report=dreport)
        self._register_profile_event('args_from_drep')
        initial_time, _ = measure(sdfg, arguments=arguments, measurements=self._measurements, warmup=self._warmup)
        self._register_profile_event('initial_measure')
        print(f"Initial time {initial_time}")

        Superoptimizer._canonicalize_sdfg(sdfg, device_type=self._device_type)
        self._register_profile_event('canonicalization')
        canon_time, _ = measure(sdfg, arguments=arguments, measurements=self._measurements, warmup=self._warmup)
        self._register_profile_event('canon_measure')
        print(f"Canonicalized time {canon_time}")

        print("Optimizing subgraphs")
        self._optimize_subgraphs(sdfg=sdfg, data_report=dreport)
        self._register_profile_event('sg_optimization')

        with open(self._cache_folder / "fused.sdfg", "w") as handle:
            json.dump(sdfg.to_json(), handle)
        self._register_profile_event('dumping_fused')

        print("Optimizing maps")
        self._optimize_maps(sdfg=sdfg, data_report=dreport)
        self._register_profile_event('optimize_maps')

        with open(self._cache_folder / "optimized.sdfg", "w") as handle:
            json.dump(sdfg.to_json(), handle)
        self._register_profile_event('dumping_optimized')

        args = arguments_from_data_report(sdfg, dreport)
        self._register_profile_event('args_from_drep_post_opt')
        tuned_runtime, _ = measure(sdfg, arguments=args, measurements=self._measurements, warmup=self._warmup)
        self._register_profile_event('opt_measure')
        print(f"Tuned runtime {tuned_runtime}, speedup {initial_time / tuned_runtime}")

        sdfg.build_folder = self._build_folder

        tuning_end = time.perf_counter()
        self._profile['tuning'] = tuning_end - tuning_start

        print(f"Tuning took {(tuning_end - tuning_start) / 60.0:.3f} minutes")

        print('Profile:')
        print(self._profile)

        return sdfg

    def _optimize_subgraphs(self, sdfg: SDFG, data_report: InstrumentedDataReport) -> None:
        sg_timer = time.perf_counter()
        initial_map_runtimes = self._measure_initial_map_runtimes(sdfg, data_report=data_report)
        self._profile['SubgraphOpt']['initial_measure'] = time.perf_counter() - sg_timer
        sg_timer = time.perf_counter()

        self._profile['SubgraphOpt']['enumeration'] = []
        self._profile['SubgraphOpt']['cutout'] = []
        self._profile['SubgraphOpt']['cache_init'] = []
        self._profile['SubgraphOpt']['collect'] = []
        self._profile['SubgraphOpt']['fusions'] = []
        self._profile['SubgraphOpt']['cdump'] = []
        self._profile['SubgraphOpt']['apply'] = []
        for subgraph, matches in map_fusion_enumerator(sdfg):
            self._profile['SubgraphOpt']['enumeration'].append(time.perf_counter() - sg_timer)
            sg_timer = time.perf_counter()
            print("Finding optimal map fusions in subgraph")

            # Create independent cutout of subgraph
            cutout = cutter.cutout_state(subgraph.graph, *subgraph.nodes(), make_copy=False)
            cutout.build_folder = sdfg.build_folder
            cutout_hash = cutout.hash_sdfg()
            cutout_tmp_copy = cutout.to_json()
            self._profile['SubgraphOpt']['cutout'].append(time.perf_counter() - sg_timer)
            sg_timer = time.perf_counter()

            # Handle cache initialization
            subgraph_cache = {}
            subgraph_cache_path = self._map_fusion_cache_folder / f"{cutout_hash}.json"
            if subgraph_cache_path.is_file():
                with open(subgraph_cache_path, "r") as handle:
                    subgraph_cache = json.load(handle)
            else:
                args = arguments_from_data_report(cutout, data_report=data_report)
                subgraph_time, subgraph_process_time = measure(cutout,
                                                               args,
                                                               measurements=self._measurements,
                                                               warmup=self._warmup)

                subgraph_cache[cutout_hash] = {
                    "runtime": subgraph_time,
                    "process time": subgraph_process_time,
                    "fusions": {}
                }

                with open(subgraph_cache_path, "w") as handle:
                    json.dump(subgraph_cache, handle)
                with open(self._map_fusion_cache_folder / f"{cutout_hash}.sdfg", "w") as handle:
                    json.dump(cutout_tmp_copy, handle)

            self._profile['SubgraphOpt']['cache_init'].append(time.perf_counter() - sg_timer)
            sg_timer = time.perf_counter()

            # Collect map entries for lower bounds
            subgraph_nodes = set(subgraph.nodes())
            subgraph_map_runtimes = []
            for map_entry in initial_map_runtimes:
                if map_entry in subgraph_nodes:
                    subgraph_map_runtimes.append(initial_map_runtimes[map_entry])
            subgraph_map_runtimes = sorted(subgraph_map_runtimes)

            if "best fusion" in subgraph_cache[cutout_hash]:
                best_fusion_desc = subgraph_cache[cutout_hash]["best fusion"]["fusion"]
                best_runtime = subgraph_cache[cutout_hash]["best fusion"]["runtime"]
                best_process_time = subgraph_cache[cutout_hash]["best fusion"]["process time"]

                if not best_fusion_desc is None:
                    best_fusion = []
                    for match in ast.literal_eval(best_fusion_desc):
                        first_map_entry_id, access_node_id, second_map_entry_id = match
                        first_map_entry = cutout.start_state.node(first_map_entry_id)
                        access_node = cutout.start_state.node(access_node_id)
                        second_map_entry = cutout.start_state.node(second_map_entry_id)
                        first_map_exit = subgraph.graph.exit_node(first_map_entry)

                        best_fusion.append((first_map_exit, access_node, second_map_entry))

                    for (f, a, s) in best_fusion:
                        OTFMapFusion.apply_to(sdfg=sdfg,
                                              first_map_exit=f,
                                              array=a,
                                              second_map_entry=s,
                                              verify=True,
                                              save=True)
                    continue

            self._profile['SubgraphOpt']['collect'].append(time.perf_counter() - sg_timer)
            sg_timer = time.perf_counter()

            # Optimize over fusions
            best_process_time = subgraph_cache[cutout_hash]["process time"]
            best_runtime = subgraph_cache[cutout_hash]["runtime"]
            best_fusion_desc = None
            for k in range(len(matches), -1, -1):
                t = len(matches) - k + 1
                lb = sum(subgraph_map_runtimes[:t]) * LOWER_BOUND_SCALING
                print(f"{k}-lower bound: {lb}")

                cache_timer = time.time()
                for k_fusions in tqdm(list(itertools.combinations(matches, r=k))):
                    start = time.time()

                    fused = SDFG.from_json(cutout_tmp_copy)
                    fused.build_folder = cutout.build_folder

                    # Converting subgraph nodes to cutout nodes
                    k_fusions_ = []
                    k_fusion_desc = []
                    for match in k_fusions:
                        first_map_entry, access_node, second_map_entry = match
                        f_node_id = cutout.start_state.node_id(first_map_entry)
                        first_map_entry = fused.start_state.node(f_node_id)
                        array_node_id = cutout.start_state.node_id(access_node)
                        access_node = fused.start_state.node(array_node_id)
                        s_node_id = cutout.start_state.node_id(second_map_entry)
                        second_map_entry = fused.start_state.node(s_node_id)

                        first_map_exit = fused.start_state.exit_node(first_map_entry)
                        k_fusions_.append((first_map_exit, access_node, second_map_entry))

                        k_fusion_desc.append((f_node_id, array_node_id, s_node_id))

                    k_fusion_desc = str(k_fusion_desc)

                    if not k_fusion_desc in subgraph_cache[cutout_hash]["fusions"]:
                        # Applying fusion in topological order
                        for fusion in k_fusions_:
                            first_map_exit, array, second_map_entry = fusion
                            OTFMapFusion.apply_to(sdfg=fused,
                                                  first_map_exit=first_map_exit,
                                                  array=array,
                                                  second_map_entry=second_map_entry,
                                                  verify=False,
                                                  save=False)

                        try:
                            fused.validate()
                            args = arguments_from_data_report(cutout, data_report=data_report)
                            fused_time, fused_process_time = measure(fused,
                                                                     args,
                                                                     measurements=self._measurements,
                                                                     warmup=self._warmup,
                                                                     timeout=best_process_time * 1.5)
                        except:
                            fused_time = math.inf

                        subgraph_cache[cutout_hash]["fusions"][k_fusion_desc] = {
                            "runtime": fused_time,
                            "process time": fused_process_time
                        }

                    fused_time = subgraph_cache[cutout_hash]["fusions"][k_fusion_desc]["runtime"]
                    fused_process_time = subgraph_cache[cutout_hash]["fusions"][k_fusion_desc]["process time"]

                    if best_runtime / fused_time >= MINIMUM_SPEEDUP:
                        best_process_time = fused_process_time
                        best_runtime = fused_time
                        best_fusion_desc = k_fusion_desc

                    end = time.time()
                    print(
                        f"Hypothesis took {end - start}, compile+measure was {fused_process_time}, runtime {fused_time}"
                    )

                    if best_runtime <= lb:
                        break

                    if ((time.time() - cache_timer) / 60.0) > 2.0:
                        with open(subgraph_cache_path, "w") as handle:
                            json.dump(subgraph_cache, handle)
                        cache_timer = time.time()

                        print("Cache dumped")

                with open(subgraph_cache_path, "w") as handle:
                    json.dump(subgraph_cache, handle)

                if best_runtime <= lb:
                    break

            self._profile['SubgraphOpt']['fusions'].append(time.perf_counter() - sg_timer)
            sg_timer = time.perf_counter()

            subgraph_cache[cutout_hash]["best fusion"] = {
                "runtime": best_runtime,
                "process time": best_process_time,
                "fusion": best_fusion_desc
            }
            with open(subgraph_cache_path, "w") as handle:
                json.dump(subgraph_cache, handle)

            self._profile['SubgraphOpt']['cdump'].append(time.perf_counter() - sg_timer)
            sg_timer = time.perf_counter()

            if not best_fusion_desc is None:
                best_fusion = []
                for match in ast.literal_eval(best_fusion_desc):
                    first_map_entry_id, access_node_id, second_map_entry_id = match
                    first_map_entry = cutout.start_state.node(first_map_entry_id)
                    access_node = cutout.start_state.node(access_node_id)
                    second_map_entry = cutout.start_state.node(second_map_entry_id)
                    first_map_exit = subgraph.graph.exit_node(first_map_entry)

                    best_fusion.append((first_map_exit, access_node, second_map_entry))

                for (f, a, s) in best_fusion:
                    OTFMapFusion.apply_to(sdfg=sdfg,
                                          first_map_exit=f,
                                          array=a,
                                          second_map_entry=s,
                                          verify=True,
                                          save=True)

            self._profile['SubgraphOpt']['apply'].append(time.perf_counter() - sg_timer)
            sg_timer = time.perf_counter()

    def _optimize_maps(self, sdfg: SDFG, data_report: InstrumentedDataReport) -> None:
        maps = {}
        maps_schedules = {}
        maps_runtime = {}
        for nsdfg in tqdm(sdfg.all_sdfgs_recursive()):
            for state in tqdm(nsdfg.nodes()):
                state_start_time = time.time()

                # Collecting outermost map entries + maps cutouts
                outermost_maps = {}
                for node in state.nodes():
                    if not isinstance(node, nodes.MapEntry):
                        continue

                    if not xfh.get_parent_map(state, node) is None:
                        continue

                    map_cutout = utils.cutout_map(state, node, make_copy=False)
                    map_cutout.transformation_hist.clear()
                    map_cutout.build_folder = sdfg.build_folder

                    maps_hash = map_cutout.hash_sdfg()
                    if not maps_hash in maps:
                        maps[maps_hash] = map_cutout

                    outermost_maps[node] = maps_hash

                for map_entry in tqdm(outermost_maps):
                    maps_hash = outermost_maps[map_entry]
                    map_cutout = maps[maps_hash]

                    if not maps_hash in maps_schedules:
                        schedule, schedule_runtime = self._find_best_map_schedule(map_cutout,
                                                                                  maps_hash,
                                                                                  data_report=data_report)
                        maps_schedules[maps_hash] = schedule
                        maps_runtime[maps_hash] = schedule_runtime

                    schedule = maps_schedules[maps_hash]
                    Superoptimizer._apply_map_schedule(nsdfg, state, map_cutout, schedule)

                state_end_time = time.time()
                print(f"Optimization of {state} took {state_end_time - state_start_time}")

    def _find_best_map_schedule(self, mp, map_hash, data_report) -> Tuple[List, float]:
        print("Optimizing map")

        # Create arguments once
        arguments = arguments_from_data_report(mp, data_report=data_report)

        map_cache_path = self._map_schedule_cache_folder / f"{map_hash}.json"
        map_cache = {}
        last_tile_size = None
        if map_cache_path.is_file():
            with open(map_cache_path, "r") as handle:
                map_cache = json.load(handle)
                last_tile_sizes = map_cache[next(iter(map_cache))]["schedules"]
                last_tile_sizes = list(last_tile_sizes.keys())
                last_tile_sizes_lst = list(
                    map(lambda s: int(re.findall(r'\b\d+\b', ((s.split(':')[3]).strip()))[0]), last_tile_sizes)
                )
                if last_tile_sizes_lst:
                    last_tile_size = max(last_tile_sizes_lst)
                print("Tile size: ", last_tile_size)
        else:
            map_runtime, map_process_time = measure(mp,
                                                    arguments,
                                                    measurements=self._measurements,
                                                    warmup=self._warmup)

            map_cache[map_hash] = {"runtime": map_runtime, "process time": map_process_time, "schedules": {}}

            with open(map_cache_path, "w") as handle:
                json.dump(map_cache, handle)
            with open(self._map_schedule_cache_folder / f"{map_hash}.sdfg", "w") as handle:
                json.dump(mp.to_json(), handle)

        if "best schedule" in map_cache[map_hash]:
            best_schedule = map_cache[map_hash]["best schedule"]["schedule"]
            best_runtime = map_cache[map_hash]["best schedule"]["runtime"]
            return best_schedule, best_runtime

        initial_time = map_cache[map_hash]["runtime"]
        best_process_time = map_cache[map_hash]["process time"]
        best_runtime = initial_time
        best_schedule = []
        best_schedule_desc = None

        cache_timer = time.time()
        start = time.time()
        print(f"Initial time {initial_time}")
        for scheduled_map, schedule_desc in map_schedule_enumerator(mp, last_tile_size):
            scheduled_map.build_folder = mp.build_folder
            if not schedule_desc in map_cache[map_hash]["schedules"]:
                runtime, process_time = measure(scheduled_map,
                                                arguments,
                                                measurements=self._measurements,
                                                warmup=self._warmup,
                                                timeout=best_process_time * 1.5)

                schedule = []
                for trans in scheduled_map.transformation_hist:
                    schedule.append(trans.to_json())

                map_cache[map_hash]["schedules"][schedule_desc] = {
                    "runtime": runtime,
                    "process time": process_time,
                    "schedule": schedule
                }

            runtime = map_cache[map_hash]["schedules"][schedule_desc]["runtime"]
            process_time = map_cache[map_hash]["schedules"][schedule_desc]["process time"]
            schedule = map_cache[map_hash]["schedules"][schedule_desc]["schedule"]

            end = time.time()
            print(f"Hypothesis took {end -start}, {process_time}")
            print(runtime, schedule_desc)

            start = time.time()

            if ((time.time() - cache_timer) / 60.0) > 2.0:
                with open(map_cache_path, "w") as handle:
                    json.dump(map_cache, handle)

            if runtime == math.inf:
                continue
            else:
                scheduled_map.clear_instrumentation_reports()

            if best_runtime / runtime >= MINIMUM_SPEEDUP:
                best_runtime = runtime
                best_schedule = schedule
                best_process_time = process_time
                best_schedule_desc = schedule_desc

        print(f"Best schedule {best_runtime}, {best_schedule}")

        map_cache[map_hash]["best schedule"] = {
            "runtime": best_runtime,
            "schedule": best_schedule,
            "process time": best_process_time,
            "schedule desc": best_schedule_desc
        }

        with open(map_cache_path, "w") as handle:
            json.dump(map_cache, handle)

        return best_schedule, best_runtime

    def _measure_initial_map_runtimes(self, sdfg: SDFG,
                                      data_report: InstrumentedDataReport) -> Dict[nodes.MapEntry, float]:
        print("Measuring initial map runtimes")

        cache_path = self._map_fusion_cache_folder / "initial_map_runtimes.json"
        cache = {}
        if cache_path.is_file():
            with open(cache_path, "r") as handle:
                cache = json.load(handle)

        initial_map_runtimes = {}
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.nodes():
                for node in state.nodes():
                    if not isinstance(node, nodes.MapEntry) or not xfh.get_parent_map(state, node) is None:
                        continue

                    cutout = utils.cutout_map(state, node, make_copy=False)
                    cutout.build_folder = sdfg.build_folder

                    cutout_hash = f"{nsdfg.sdfg_id}:{nsdfg.node_id(state)}:{state.node_id(node)}"
                    if not cutout_hash in cache:
                        args = arguments_from_data_report(cutout, data_report=data_report)
                        runtime, _ = measure(cutout,
                                             arguments=args,
                                             measurements=self._measurements,
                                             warmup=self._warmup)

                        cache[cutout_hash] = runtime

                    initial_map_runtimes[node] = cache[cutout_hash]

            with open(cache_path, "w") as handle:
                json.dump(cache, handle)

        return initial_map_runtimes

    @staticmethod
    def _apply_map_schedule(sdfg: SDFG, state: SDFGState, cutout: SDFG, transformations: List):
        # Restrict area of interest in state to subgraph of cutout
        subgraph = cutout

        cutout_ = copy.deepcopy(cutout)
        for trans in transformations:
            pattern = copy.deepcopy(trans)
            for pattern_node in trans["_subgraph"]:
                node = cutout_.start_state.node(trans["_subgraph"][pattern_node])
                node_type = type(node)

                scope_level = 0
                scope_entry = cutout_.start_state.entry_node(node)
                while not scope_entry is None:
                    scope_level += 1
                    scope_entry = cutout_.start_state.entry_node(scope_entry)

                pattern["_subgraph"][pattern_node] = {"type": node_type, "scope_level": scope_level}

            sdfg_trans = copy.deepcopy(trans)
            nodes_picked = set()
            for pattern_node in pattern["_subgraph"]:
                picked = False
                for node in subgraph.start_state.nodes():
                    if node in nodes_picked:
                        continue

                    node_type = type(node)

                    scope_level = 0
                    scope_entry = state.entry_node(node)
                    while not scope_entry is None:
                        scope_level += 1
                        scope_entry = state.entry_node(scope_entry)

                    if pattern["_subgraph"][pattern_node]["type"] == node_type and pattern["_subgraph"][pattern_node][
                            "scope_level"] == scope_level:
                        sdfg_trans["_subgraph"][pattern_node] = state.node_id(node)
                        nodes_picked.add(node)
                        picked = True
                        break

                if not picked:
                    raise ValueError("Pattern not mached")

            xform = PatternTransformation.from_json(trans)
            xform._sdfg = cutout_
            xform.state_id = cutout_.node_id(cutout_.start_state)
            xform.apply(cutout_.start_state, cutout_)

            xform = PatternTransformation.from_json(sdfg_trans)
            xform._sdfg = sdfg
            xform.state_id = sdfg.node_id(state)
            xform.apply(state, sdfg)

            top_level_map = None
            for node in subgraph.start_state.nodes():
                if isinstance(node, nodes.MapEntry):
                    top_level_map = node
                    break

            while not state.entry_node(top_level_map) is None:
                top_level_map = state.entry_node(top_level_map)

            subgraph = utils.cutout_map(state, top_level_map, make_copy=False)

        sdfg.validate()

    @staticmethod
    def _canonicalize_sdfg(sdfg: SDFG, device_type: DeviceType) -> None:
        """
        Canonicalized the SDFG before superoptimization.

        :param sdfg: the SDFG.
        :param device_type: the target device type of the optimization.
        """
        sdfg.simplify()

        # Specialize library nodes
        set_fast_implementations(sdfg, device_type)

        # Enforce fusion opportunities over reductions
        sdfg.expand_library_nodes()
        # sdfg.apply_transformationed_repeated(InlineSDFG)
        sdfg.apply_transformations_repeated(InlineNestedReduce)
        sdfg.apply_transformations_repeated(RedundantSecondArray)

        # Canonicalize pure maps
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.nodes():
                for node in state.nodes():
                    if not isinstance(node, nodes.MapEntry):
                        continue

                    node.schedule = ScheduleType.Sequential
                    node.collapse = 1

        # Bring all maps into collapsed normal-form
        sdfg.apply_transformations_repeated(TrivialMapElimination)
        sdfg.apply_transformations_repeated(MapCollapse)

        sdfg.simplify()

    @staticmethod
    def dry_run(sdfg: SDFG) -> InstrumentedDataReport:
        """
        Generates a data report suitable for superoptimization (including all transients).
        This instrumentation can take considerable amount of disk space and time. This needs to be considered when
        choosing the build folder.

        :param sdfg: sdfg to instrument.
        :return: data report.
        """
        dreport = sdfg.get_instrumented_data()
        if dreport is None:
            print("Creating data report")
            args = random_arguments(sdfg)
            dreport = create_data_report(sdfg, args, transients=True)

        return dreport
