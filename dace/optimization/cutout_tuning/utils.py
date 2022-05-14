# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import math
import numpy as np

import traceback
import multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method("spawn")

from dace import SDFG, config, InstrumentationType
from dace import data as dt


def measure(sdfg: SDFG,
            dreport,
            instrumentation_type: InstrumentationType = InstrumentationType.Timer,
            repeat: int = 30,
            timeout: float = 120.0) -> float:
    sdfg.instrument = instrumentation_type

    queue = mp.Queue()
    proc = MeasureProcess(target=_measure, args=(sdfg.to_json(), dreport, instrumentation_type, repeat, queue))
    proc.start()
    proc.join(timeout)

    if proc.exitcode != 0:
        print("Error occured during measuring")
        return math.inf

    if proc.exception:
        error, traceback = proc.exception
        print(traceback)
        print("Error occured during measuring: ", error)
        runtime = math.inf

    try:
        runtime = queue.get(block=True, timeout=10)
    except:
        return math.inf

    return runtime


def _measure(sdfg_json, dreport, instrumentation_type: InstrumentationType, repetitions: int, queue: mp.Queue) -> float:
    sdfg = SDFG.from_json(sdfg_json)
    sdfg.instrument = instrumentation_type

    arguments = {}
    if len(sdfg.free_symbols) > 0:
        raise ValueError("Free symbols found")

    for state in sdfg.nodes():
        for dnode in state.data_nodes():
            array = sdfg.arrays[dnode.data]
            if array.transient:
                continue

            try:
                data = dreport[dnode.data]
                arguments[dnode.data] = dt.make_array_from_descriptor(array, data, constants=sdfg.constants)
            except KeyError:
                arguments[dnode.data] = dt.make_array_from_descriptor(array, constants=sdfg.constants)

    for name, array in list(sdfg.arrays.items()):
        if array.transient:
            continue

        if not name in arguments:
            del sdfg.arrays[name]

    try:
        sdfg.clear_instrumentation_reports()
    except FileNotFoundError:
        pass

    with config.set_temporary('debugprint', value=False):
        with config.set_temporary('compiler', 'allow_view_arguments', value=True):
            with config.set_temporary('instrumentation', 'report_each_invocation', value=False):
                csdfg = sdfg.compile()

                for _ in range(repetitions):
                    csdfg(**arguments)

                csdfg.finalize()

    report = sdfg.get_latest_report()
    durations = next(iter(next(iter(report.durations.values())).values()))

    queue.put(np.median(np.array(durations)))


def measure_main(sdfg: SDFG,
                 dreport,
                 instrumentation_type: InstrumentationType = InstrumentationType.Timer,
                 repeat: int = 30) -> float:
    sdfg.instrument = instrumentation_type

    arguments = {}
    if len(sdfg.free_symbols) > 0:
        raise ValueError("Free symbols found")

    for state in sdfg.nodes():
        for dnode in state.data_nodes():
            array = sdfg.arrays[dnode.data]
            if array.transient:
                continue

            try:
                data = dreport[dnode.data]
                arguments[dnode.data] = dt.make_array_from_descriptor(array, data, constants=sdfg.constants)
            except KeyError:
                arguments[dnode.data] = dt.make_array_from_descriptor(array, constants=sdfg.constants)

    for name, array in list(sdfg.arrays.items()):
        if array.transient:
            continue

        if not name in arguments:
            del sdfg.arrays[name]

    try:
        sdfg.clear_instrumentation_reports()
    except FileNotFoundError:
        pass

    with config.set_temporary('debugprint', value=False):
        with config.set_temporary('compiler', 'allow_view_arguments', value=True):
            with config.set_temporary('instrumentation', 'report_each_invocation', value=False):
                csdfg = sdfg.compile()

                for _ in range(repeat):
                    csdfg(**arguments)

                csdfg.finalize()

    report = sdfg.get_latest_report()
    durations = next(iter(next(iter(report.durations.values())).values()))
    return np.median(np.array(durations))


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
