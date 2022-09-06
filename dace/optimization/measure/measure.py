# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import math
import time
import numpy as np

import traceback
import multiprocessing as mp

from typing import Dict, Tuple

from dace.codegen.compiled_sdfg import CompiledSDFG, ReloadableDLL
from dace import SDFG, config, InstrumentationType


def measure(sdfg: SDFG,
            arguments: Dict,
            measurements: int,
            warmup: int = 0,
            timeout: float = None) -> Tuple[float, float, Dict]:
    """
    A helper function to measure the median runtime of a SDFG over several measurements. The measurement is executed in a subprocess that can be killed after a specific timeout. This function will add default Timer instrumentation to the SDFG and return the full SDFG's runtime. The instrumentation report with the individual runtimes and the additional instrumentation is available afterwards as well.

    :param SDFG: the SDFG to be measured.
    :param arguments: the arguments provided to the SDFG.
    :param measurements: the number of measurements.
    :param warmup: optional warmup iterations (default = 0) which are excluded from the median.
    :param timeout: optional timeout to kill the measurement.
    :return: a tuple of median runtime, time of the whole measurement and the modified arguments (results). The second time is useful to determine a tight timeout for a transformed SDFG.
    """
    with config.set_temporary('instrumentation', 'report_each_invocation', value=False):
        with config.set_temporary('compiler', 'allow_view_arguments', value=True):
            sdfg.instrument = InstrumentationType.Timer
            try:
                csdfg = sdfg.compile(validate=False, in_place=True)
            except:
                return math.inf, math.inf

            proc = MeasureProcess(target=_measure,
                                  args=(sdfg.to_json(), sdfg.build_folder, csdfg._lib._library_filename, arguments,
                                        warmup, measurements))

            start = time.time()
            proc.start()
            proc.join(timeout)
            process_time = time.time() - start

            # Handle failure
            if proc.exitcode != 0:
                if proc.is_alive():
                    proc.kill()
                return math.inf, process_time

            if proc.exception:
                if proc.is_alive():
                    proc.kill()
                error, traceback = proc.exception
                print(error)
                print(traceback)
                return math.inf, process_time

            # Handle success
            if proc.is_alive():
                proc.kill()

            report = sdfg.get_latest_report()
            durations = list(report.durations.values())[0]
            durations = list(durations.values())[0]
            durations = list(durations.values())[0]
            runtime = np.median(np.array(durations[warmup:]))
            return runtime, process_time


def _measure(sdfg_json: Dict, build_folder: str, filename: str, arguments: Dict, warmup: int, measurements: int):
    sdfg = SDFG.from_json(sdfg_json)
    sdfg.build_folder = build_folder
    lib = ReloadableDLL(filename, sdfg.name)
    csdfg = CompiledSDFG(sdfg, lib, arguments.keys())

    with config.set_temporary('instrumentation', 'report_each_invocation', value=False):
        with config.set_temporary('compiler', 'allow_view_arguments', value=True):
            for _ in range(warmup):
                csdfg(**arguments)

            for _ in range(measurements):
                csdfg(**arguments)

            csdfg.finalize()


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

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception
