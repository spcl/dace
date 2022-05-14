# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import math
import json
import copy

from typing import Dict, Any, List, Union
from pathlib import Path

from pydoc import locate

import dace.optimization.cutout_tuning.utils as utils

from dace import SDFG, InstrumentationType, DataInstrumentationType, nodes
from dace.optimization.cutout_tuning.cutout_space import CutoutSpace
from dace.optimization.auto_tuner import AutoTuner

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x


class CutoutTuner(AutoTuner):
    """
    AutoTuner which tunes SDFG cutouts independently.
    Usage:
        1. Run dry_run to obtain data report.
        2. Run tune to search configurations and tune different cutouts.
    """
    def __init__(self,
                 sdfg: SDFG,
                 dreport,
                 config_path: Union[str, Path],
                 instrumentation_type: InstrumentationType = InstrumentationType.Timer) -> None:
        """
        
        :param sdfg: the SDFG.
        :param dreport: the data report (obtained from dry_run function).
        :param config_path: the tuner config
        :param instrumentation_type: the instrumentation type defines the metric to compare configurations.
        """
        super().__init__(sdfg=sdfg)
        self._dreport = dreport
        self._instrumentation_type = instrumentation_type

        with open(config_path, "r") as handle:
            self._config = json.load(handle)

            self._name = self._config["name"]
            self._stages = []
            for stage_desc in self._config["stages"]:
                stage = []
                for space in stage_desc:
                    space_class = locate(f'dace.optimization.cutout_tuning.spaces.{space}')
                    stage.append(space_class())

                self._stages.append(stage)

        self._cache_folder = Path(self._sdfg.build_folder) / "tuning" / self._name

        self.measure = utils.measure

    def _write_cache(self, stage_key: str, search_cache: Dict) -> None:
        stage_cache_folder = self._cache_folder / stage_key
        if not stage_cache_folder.is_dir():
            stage_cache_folder.mkdir(parents=True, exist_ok=True)

        # Cutout
        for index in search_cache:
            cutout_cache_folder = stage_cache_folder / str(index)
            if not cutout_cache_folder.is_dir():
                cutout_cache_folder.mkdir(parents=True, exist_ok=True)

            # 1. Header
            header = {
                "index": index,
                "base_runtime": search_cache[index]["base_runtime"],
                "best_config": search_cache[index]["best_config"],
                "best_runtime": search_cache[index]["best_runtime"]
            }
            with open(cutout_cache_folder / "header.json", "w") as handle:
                json.dump(header, handle)

            # 2. Configs
            for j, key in enumerate(search_cache[index]["configs"]):
                with open(cutout_cache_folder / f"{j}.json", "w") as handle:
                    data = {"key": key, "config": search_cache[index]["configs"][key]}
                    json.dump(data, handle)

    def _load_cache(self) -> Dict:
        if not self._cache_folder.is_dir():
            self._cache_folder.mkdir(parents=True, exist_ok=True)
            return {}

        search_cache = {}
        for stage in self._stages:
            stage_key = CutoutTuner._stage_desc(stage)
            stage_cache_folder = self._cache_folder / stage_key
            if not stage_cache_folder.is_dir():
                # stages depend on each other, no later stages loaded
                break

            stage_cache = {}
            for cutout_cache_folder in stage_cache_folder.iterdir():
                if not cutout_cache_folder.is_dir():
                    continue

                # 1. Header
                header_path = cutout_cache_folder / "header.json"
                if not header_path.is_file():
                    continue

                index = cutout_cache_folder.name
                stage_cache[index] = {}
                with open(header_path, "r") as handle:
                    header = json.load(handle)
                    stage_cache[index]["base_runtime"] = header["base_runtime"]
                    stage_cache[index]["best_runtime"] = header["best_runtime"]
                    stage_cache[index]["best_config"] = header["best_config"]

                # 2. Configs
                stage_cache[index]["configs"] = {}
                for config_path in cutout_cache_folder.glob("*.json"):
                    if config_path.stem == "header":
                        continue

                    with open(config_path, "r") as handle:
                        res = json.load(handle)
                        key = res["key"]
                        stage_cache[index]["configs"][key] = res["config"]

            search_cache[stage_key] = stage_cache

        return search_cache

    def _search(self, stage: List[CutoutSpace], sdfg: SDFG, search_cache: Dict, top_level: bool = True) -> Dict:
        """
        Searches the space of the stage brute-force.
        The best config of each cutout is directly applied to the SDFG.

        :param stage: the stage.
        :param sdfg: the sdfg.
        :param search_cache: a (partial) search cache from previous runs.
        :param top_level: whether to write the search cache to the dacecache folder of the SDFG.
        :return: returns the updated search cache.
        """
        stage_key = CutoutTuner._stage_desc(stage)
        if stage_key not in search_cache:
            search_cache[stage_key] = {}

        if top_level:
            bar = tqdm
        else:
            bar = lambda x, **kwargs: x

        current_space = stage[0]
        for i, cutout in bar(list(enumerate(current_space.cutouts(sdfg)))):
            index = str(i)
            if index not in search_cache[stage_key]:
                # Cutout not in cache, measure baseline
                search_cache[stage_key][index] = {}
                search_cache[stage_key][index]["configs"] = {}
                search_cache[stage_key][index]["base_runtime"] = self.measure(
                    cutout, dreport=self._dreport, instrumentation_type=self._instrumentation_type)

            base_runtime = search_cache[stage_key][index]["base_runtime"]
            if base_runtime < 0 or base_runtime == math.inf:
                continue

            # Iterate through config space and measure configs if necessary
            best_runtime = base_runtime
            best_config = None
            for config in bar(list(current_space.configurations(cutout))):
                key = current_space.encode_config(config)
                if key in search_cache[stage_key][index]["configs"]:
                    # a) Results from cache
                    runtime = search_cache[stage_key][index]["configs"][key]["runtime"]
                    if runtime < 0 or runtime == math.inf:
                        continue

                    if runtime < best_runtime:
                        best_runtime = runtime
                        best_config = key

                    continue

                # b) Measure config
                search_cache[stage_key][index]["configs"][key] = {}
                search_cache[stage_key][index]["configs"][key]["runtime"] = math.inf
                search_cache[stage_key][index]["configs"][key]["subspace"] = {}

                runtime = math.inf
                cutout_ = current_space.apply_config(cutout, config, make_copy=True)
                if cutout_ is not None:
                    if len(stage) > 1:
                        subspace_cache = search_cache[stage_key][index]["configs"][key]["subspace"]

                        self._search(stage[1:], cutout_, search_cache=subspace_cache, top_level=False)

                        search_cache[stage_key][index]["configs"][key]["subspace"] = subspace_cache

                    runtime = self.measure(cutout_,
                                           dreport=self._dreport,
                                           instrumentation_type=self._instrumentation_type)

                search_cache[stage_key][index]["configs"][key]["runtime"] = runtime

                if runtime < 0 or runtime == math.inf:
                    continue

                if runtime < best_runtime:
                    best_runtime = runtime
                    best_config = key

            # Store best config
            search_cache[stage_key][index]["best_config"] = best_config
            search_cache[stage_key][index]["best_runtime"] = best_runtime

        if top_level:
            self._write_cache(stage_key, search_cache[stage_key])

        # Update SDFG with best configs
        self._apply_best_configs(stage, sdfg, search_cache=search_cache[stage_key])

        return search_cache

    def _apply_best_configs(self, stage: List[CutoutSpace], sdfg: SDFG, search_cache: Dict, context: List[SDFG] = None):
        current_space = stage[0]
        if context is None:
            current_sdfg = sdfg
            context = []
        else:
            current_sdfg = context[-1]

        # Apply all cutout best configs to original SDFG
        for i, cutout in enumerate(current_space.cutouts(current_sdfg)):
            index = str(i)
            key = search_cache[index]["best_config"]
            if key is None:
                continue

            # Apply configs recursively
            # a. translate to parent context on non-modified SDFG
            # b. apply to copy to avoid double application
            config = current_space.decode_config(key)
            m_cutout = current_space.apply_config(cutout, config, make_copy=True)

            child_sdfg = cutout
            context_configs = [config]
            modified_context = [m_cutout]
            for i in range(len(context)):
                parent_sdfg = context[-(i + 1)]

                config = current_space.translate_config(child_sdfg, parent_sdfg, config)
                context_configs.insert(0, config)

                m_parent_sdfg = current_space.apply_config(parent_sdfg, config, make_copy=True)
                modified_context.insert(0, m_parent_sdfg)

                child_sdfg = parent_sdfg

            # Apply to SDFG without copy
            sdfg_config = current_space.translate_config(child_sdfg, sdfg, context_configs[0])
            sdfg = current_space.apply_config(sdfg, sdfg_config, make_copy=False)

            # If nested configs, translate configs over cutout back to SDFG recursively and apply
            # cutout needs to have the configs applied as well to remain equivalent to SDFG
            if len(stage) > 1:
                nested_stages = stage.copy()
                nested_stages.pop(0)

                nested_stage_desc = CutoutTuner._stage_desc(nested_stages)
                nested_search_cache = search_cache[index]["configs"][key]["subspace"][nested_stage_desc]

                self._apply_best_configs(nested_stages,
                                         sdfg,
                                         search_cache=nested_search_cache,
                                         context=modified_context)

    def tune(self, in_place: bool = True) -> SDFG:
        """
        Auto-tunes the SDFG with stages defined in the tuning config.
        Stages run sequentially and each stage tunes multiple cutouts of the SDFG.

        :param in_place: whether to apply on original SDFG or on copy.
        :return: Returns the tuned SDFG (eiher original or copy).
        """
        print()
        print("***** Tuning *****")
        sdfg = self._sdfg
        if not in_place:
            sdfg = copy.deepcopy(self._sdfg)

        print("Measuring Base Runtime")
        init_runtime = self.measure(sdfg, dreport=self._dreport, instrumentation_type=self._instrumentation_type)
        print(f"Base runtime: {init_runtime:.3f} ms")
        print()

        search_cache = self._load_cache()
        for stage in self._stages:
            print("Stage: ", CutoutTuner._stage_desc(stage))
            self._search(stage, sdfg, search_cache)
            print()

        final_runtime = self.measure(sdfg, dreport=self._dreport, instrumentation_type=self._instrumentation_type)

        print("***** Tuning Finished *****")
        print(f"Result: {init_runtime:.3f} ms (base), {final_runtime:.3f} ms (final)")

        return sdfg

    @staticmethod
    def _stage_desc(stage: List[CutoutSpace]) -> str:
        return ".".join(list(map(lambda s: s.name(), stage)))

    @staticmethod
    def dry_run(sdfg: SDFG, *args, **kwargs) -> Any:
        """
        A dry run executes the SDFG with data instrumentation and stores the data of
        all non-transient arrays in the dacecache. The method checks whether a valid data
        report is already available.
        
        The data report is also returned at the end of the function.

        :param sdfg: the SDFG.
        :param args: args to the SDFG.
        :param kwargs: kwargs to the SDFG.
        :return: the data report
        """
        print("***** Dry run *****")

        # Check existing instrumented data for shape mismatch
        kwargs.update({aname: a for aname, a in zip(sdfg.arg_names, args)})

        dreport = sdfg.get_instrumented_data()
        if dreport is not None:
            for data in dreport.keys():
                rep_arr = dreport.get_first_version(data)
                sdfg_arr = sdfg.arrays[data]
                # Potential shape mismatch
                if rep_arr.shape != sdfg_arr.shape:
                    # Check given data first
                    if hasattr(kwargs[data], 'shape') and rep_arr.shape != kwargs[data].shape:
                        sdfg.clear_data_reports()
                        dreport = None
                        break

        # If there is no valid instrumented data available yet, run in data instrumentation mode
        if dreport is None:
            for state in sdfg.nodes():
                for node in state.nodes():
                    if isinstance(node, nodes.AccessNode) and not node.desc(sdfg).transient:
                        node.instrument = DataInstrumentationType.Save

            result = sdfg(**kwargs)

            # Disable data instrumentation from now on
            for state in sdfg.nodes():
                for node in state.nodes():
                    if isinstance(node, nodes.AccessNode):
                        node.instrument = DataInstrumentationType.No_Instrumentation

        dreport = sdfg.get_instrumented_data()
        print("Data report created")
        return dreport
