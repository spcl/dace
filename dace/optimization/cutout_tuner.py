# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import tempfile
import math
import dace
import json

from typing import Dict, Generator, Any, List, Tuple
from dace.optimization import auto_tuner
from dace.optimization import utils as optim_utils
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x


class CutoutTuner(auto_tuner.AutoTuner):
    """
    An auto-tuner that cuts out subgraphs of the original SDFG to tune separately.
    In order to tune an SDFG, a "dry run" must first be called to collect data from intermediate
    access nodes (in order to ensure correctness of the tuned subgraph). Subsequently, sub-classes of
    this cutout tuning interface will select subgraphs to test transformations on.

    For example::

        tuner = DataLayoutTuner(sdfg)
        
        # Create instrumented data report
        tuner.dry_run(sdfg, arg1, arg2, arg3=4)

        results = tuner.optimize()
        # results will now contain the fastest data layout configurations for each array
    """

    def __init__(self, task: str, sdfg: SDFG) -> None:
        """
        Creates a cutout tuner.
        
        :param task: Name of tuning task (for filename labeling).
        :param sdfg: The SDFG to tune.
        """
        super().__init__(sdfg=sdfg)
        self._task = task

    @property
    def task(self) -> str:
        return self._task

    def file_name(self, label: str) -> str:
        return f"{self._task}.{label}.tuning"

    def try_load(self, file_name) -> Dict:
        results = None
        if os.path.exists(file_name):
            print(f'Using cached {file_name}')

            with open(file_name, 'r') as fp:
                results = json.load(fp)

        return results

    def cutouts(self) -> Generator[Tuple[SDFGState, str], None, None]:
        raise NotImplementedError

    def space(self, **kwargs) -> Generator[Any, None, None]:
        raise NotImplementedError

    def search(self, cutout: SDFG, measurements: int, **kwargs) -> Dict:
        raise NotImplementedError

    def pre_evaluate(self, **kwargs) -> Dict:
        raise NotImplementedError

    def evaluate(self, **kwargs) -> float:
        raise NotImplementedError

    def config_from_key(self, key: str, cutout: dace.SDFG, **kwargs) -> Any:
        raise NotImplementedError

    def apply(self, config, cutout, **kwargs) -> None:
        raise NotImplementedError

    def measure(self, cutout, dreport, repetitions: int = 30, timeout: float = 300.0) -> float:
        dreport_ = {}
        for cstate in cutout.nodes():
            for dnode in cstate.data_nodes():
                array = cutout.arrays[dnode.data]
                if array.transient:
                    continue
            try:
                data = dreport.get_first_version(dnode.data)
                dreport_[dnode.data] = data
            except:
                continue
        
        runtime = optim_utils.subprocess_measure(cutout=cutout, dreport=dreport_, repetitions=repetitions, timeout=timeout)
        return runtime

    def optimize(self, measurements: int = 30, apply: bool = False, **kwargs) -> Dict[Any, Any]:
        tuning_report = {}
        for cutout, label in tqdm(list(self.cutouts())):
            fn = self.file_name(label)
            results = self.try_load(fn)

            if results is None:
                results = self.search(cutout, measurements, **kwargs)
                if results is None:
                    tuning_report[label] = None
                    continue

                with open(fn, 'w') as fp:
                    json.dump(results, fp)

            best_config = min(results, key=results.get)
            if apply:
                config = self.config_from_key(best_config, cutout=cutout)
                self.apply(config, label=label)

            tuning_report[label] = results

        return tuning_report

    def search(self, cutout: SDFG, measurements: int,
               **kwargs) -> Dict[str, float]:
        kwargs = self.pre_evaluate(cutout=cutout, measurements=measurements, **kwargs)

        results = {}
        key = kwargs["key"]
        for config in tqdm(list(self.space(**(kwargs["space_kwargs"])))):
            kwargs["config"] = config
            runtime = self.evaluate(**kwargs)
            results[key(config)] = runtime

        return results

    @staticmethod
    def top_k_configs(tuning_report, k: int) -> List[Tuple[str, float]]:
        all_configs = []
        for cutout_label in tuning_report:
            configs = tuning_report[cutout_label]
            best_k_configs = [(key, value) for key, value in sorted(configs.items(), key=lambda item: item[1])][:min(len(configs), k)]
            best_k_configs = filter(lambda c: c[1] != math.inf, best_k_configs)
            best_k_configs = list(map(lambda c: (cutout_label, c[0]), best_k_configs))

            all_configs.extend(best_k_configs)

        return all_configs

    @staticmethod
    def dry_run(sdfg: SDFG, *args, **kwargs) -> Any:
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
                    if isinstance(node, dace.nodes.AccessNode) and not node.desc(sdfg).transient:
                        node.instrument = dace.DataInstrumentationType.Save

            result = sdfg(**kwargs)

            # Disable data instrumentation from now on
            for state in sdfg.nodes():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.AccessNode):
                        node.instrument = dace.DataInstrumentationType.No_Instrumentation
        else:
            return None

        return result
