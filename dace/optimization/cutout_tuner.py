# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import math
import dace
import json
import traceback
import numpy as np

from typing import Dict, Generator, Any, Tuple, List
import multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method("spawn")

from dace.optimization import auto_tuner
from dace.codegen.instrumentation.data import data_report

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x


class CutoutTuner(auto_tuner.AutoTuner):

    def __init__(self, task: str, sdfg: dace.SDFG) -> None:
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

    def cutouts(self) -> Generator[Tuple[dace.SDFGState, str], None, None]:
        raise NotImplementedError

    def space(self, **kwargs) -> Generator[Any, None, None]:
        raise NotImplementedError

    def search(self, cutout: dace.SDFG, measurements: int, **kwargs) -> Dict:
        raise NotImplementedError

    def pre_evaluate(self, **kwargs) -> Dict:
        raise NotImplementedError

    def evaluate(self, **kwargs) -> float:
        raise NotImplementedError

    def config_from_key(self, key: str, cutout: dace.SDFG, **kwargs) -> Any:
        raise NotImplementedError

    def apply(self, config, cutout, **kwargs) -> None:
        raise NotImplementedError

    def measure(self, sdfg: dace.SDFG, repetitions: int = 30, timeout: float = 120.0) -> float:
        parent_conn, child_conn = mp.Pipe()        
        proc = MeasureProcess(target=_measure, args=(self._sdfg.to_json(), sdfg.to_json(), repetitions, child_conn))
                
        proc.start()
        
        if parent_conn.poll(timeout):
            runtime = parent_conn.recv()
        else:
            print("Error occured during measuring: timeout")
            runtime = math.inf

        proc.join()

        if proc.exception:
            error, traceback = proc.exception
            print("Error occured during measuring: ", error)
            print(traceback)
            runtime = math.inf

        return runtime

    def optimize(self, measurements: int = 30, apply: bool = False, **kwargs) -> Dict:
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

    def search(self, cutout: dace.SDFG, measurements: int,
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
    def top_k_configs(tuning_report, k: int):
        all_configs = []
        for cutout_label in tuning_report:
            configs = tuning_report[cutout_label]
            best_k_configs = [(key, value) for key, value in sorted(configs.items(), key=lambda item: item[1])][:min(len(configs), k)]
            best_k_configs = filter(lambda c: c[1] != math.inf, best_k_configs)
            best_k_configs = list(map(lambda c: (cutout_label, c[0]), best_k_configs))

            all_configs.extend(best_k_configs)

        return all_configs

    @staticmethod
    def dry_run(sdfg, *args, **kwargs) -> Any:
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

def _measure(sdfg_json: Dict, cutout_json: Dict, repetitions: int, pipe: mp.Pipe) -> float:
    sdfg = dace.SDFG.from_json(sdfg_json)
    cutout = dace.SDFG.from_json(cutout_json)
    dreport: data_report.InstrumentedDataReport = sdfg.get_instrumented_data()

    arguments = {}
    for cstate in cutout.nodes():
        for dnode in cstate.data_nodes():
            array = cutout.arrays[dnode.data]
            if array.transient:
                continue

            try:
                data = dreport.get_first_version(dnode.data)
                arguments[dnode.data] = dace.data.make_array_from_descriptor(array, data)
            except KeyError:
                print("Missing data in dreport, random array")
                arguments[dnode.data] = dace.data.make_array_from_descriptor(array)

    print("Nice new arrays")

    with dace.config.set_temporary('debugprint', value=False):
        with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=False):
            with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
                csdfg = cutout.compile()

                for _ in range(repetitions):
                    csdfg(**arguments)

                csdfg.finalize()

    report = cutout.get_latest_report()
    durations = next(iter(next(iter(report.durations.values())).values()))

    pipe.send(np.median(np.array(durations)))
    pipe.close()

class MeasureProcess(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception