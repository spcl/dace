# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace
import json
import numpy as np

from typing import Dict, Generator, Any, Tuple, List

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

    def space(self, cutout: dace.SDFG) -> Generator[Any, None, None]:
        raise NotImplementedError

    def search(self, cutout: dace.SDFG, dreport: data_report.InstrumentedDataReport, measurements: int,
                 **kwargs) -> Dict:
        raise NotImplementedError

    def evaluate(self, config: Any, cutout: dace.SDFG, dreport: data_report.InstrumentedDataReport,
                        measurements: int, **kwargs) -> Dict:
        raise NotImplementedError

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

    def measure(self, sdfg: dace.SDFG, arguments: Dict[str, dace.data.ArrayLike], repetitions: int = 30) -> float:
        with dace.config.set_temporary('debugprint', value=False):
            with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=False):
                with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
                    csdfg = sdfg.compile()
                    for _ in range(repetitions):
                        csdfg(**arguments)

                    csdfg.finalize()

        report = sdfg.get_latest_report()
        durations = next(iter(next(iter(report.durations.values())).values()))
        return np.median(np.array(durations))

    def optimize(self, measurements: int = 30, **kwargs) -> Dict:
        dreport: data_report.InstrumentedDataReport = self._sdfg.get_instrumented_data()

        tuning_report = {}
        for cutout, label in tqdm(list(self.cutouts())):
            fn = self.file_name(label)
            results = self.try_load(fn)

            if results is None:
                results = self.search(cutout, dreport, measurements, **kwargs)

                with open(fn, 'w') as fp:
                    json.dump(results, fp)

            tuning_report[label] = results

        return tuning_report
