# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import abc
import dace
import numpy as np

from typing import Dict, Generator, Any, Tuple, List

from dace.optim import auto_tuner
from dace.codegen.instrumentation.data import data_report


class CutoutTuner(auto_tuner.AutoTuner, abc.ABC):

    def __init__(self, sdfg: dace.SDFG) -> None:
        super().__init__(sdfg=sdfg)

    @abc.abstractmethod
    def cutouts(self) -> Generator[Tuple[dace.SDFGState, List[dace.nodes.Node]], None, None]:
        pass

    @abc.abstractmethod
    def space(self, cutout: dace.SDFG) -> Generator[Any, None, None]:
        pass

    def dry_run(self, *args, **kwargs) -> None:
        # Check existing instrumented data for shape mismatch
        kwargs.update({aname: a for aname, a in zip(self._sdfg.arg_names, args)})

        dreport = self._sdfg.get_instrumented_data()
        if dreport is not None:
            for data in dreport.keys():
                rep_arr = dreport.get_first_version(data)
                sdfg_arr = self._sdfg.arrays[data]
                # Potential shape mismatch
                if rep_arr.shape != sdfg_arr.shape:
                    # Check given data first
                    if hasattr(kwargs[data], 'shape') and rep_arr.shape != kwargs[data].shape:
                        self._sdfg.clear_data_reports()
                        dreport = None
                        break

        # If there is no valid instrumented data available yet, run in data instrumentation mode
        if dreport is None:
            for node, _ in self._sdfg.all_nodes_recursive():
                if isinstance(node, dace.nodes.AccessNode) and not node.desc(self._sdfg).transient:
                    node.instrument = dace.DataInstrumentationType.Save

            self._sdfg(**kwargs)

            # Disable data instrumentation from now on
            for node, _ in self._sdfg.all_nodes_recursive():
                if isinstance(node, dace.nodes.AccessNode):
                    node.instrument = dace.DataInstrumentationType.No_Instrumentation

    def measure(self, sdfg: dace.sdfg, arguments: Dict[str, dace.data.ArrayLike], repetitions: int = 30) -> np.float:
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
