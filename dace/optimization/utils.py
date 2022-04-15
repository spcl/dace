import os
import json
import pickle
import tempfile
import math
import dace
import itertools
import numpy as np

from typing import Dict

from dace.codegen.instrumentation.data import data_report

def measure(sdfg, dreport=None, repetitions = 30, print_report : bool = False):
    arguments = {}
 
    for cstate in sdfg.nodes():
        for dnode in cstate.data_nodes():
            array = sdfg.arrays[dnode.data]
            if array.transient:
                continue

            if dreport is not None:
                try:
                    data = dreport.get_first_version(dnode.data)
                    if data.shape != array.shape:
                        data = np.random.rand(*array.shape)
                    arguments[dnode.data] = dace.data.make_array_from_descriptor(array, data)
                except KeyError:
                    arguments[dnode.data] = dace.data.make_array_from_descriptor(array)
            else:
                arguments[dnode.data] = dace.data.make_array_from_descriptor(array, np.random.rand(*array.shape))

    

    try:
        with dace.config.set_temporary('debugprint', value=True):
            with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=False):
                with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
                    csdfg = sdfg.compile()

                    for _ in range(repetitions):
                        csdfg(**arguments)

                    csdfg.finalize()
    except Exception as e:
        return math.inf

    report = sdfg.get_latest_report()
    if print_report:
        print(report)

    durations = next(iter(next(iter(report.durations.values())).values()))
    return np.median(np.array(durations))

def partition(it, size):
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it, size)), ())


def get_world_rank():
    if 'MV2_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_RANK'])
    elif 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    else:
        print('Cannot get world rank, running in sequential mode')
        return 0


def get_world_size():
    if 'MV2_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_SIZE'])
    elif 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    else:
        print('Cannot get world size, running in sequential mode')
        return 1

import traceback

import multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method("spawn")


def subprocess_measure(cutout: dace.SDFG, dreport, repetitions: int = 30, timeout: float = 600.0) -> float:
    q = mp.Queue()
    proc = MeasureProcess(target=_subprocess_measure, args=(cutout.to_json(), dreport, repetitions, q))
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
        runtime = q.get(block=True, timeout=30)
    except:
        return math.inf

    return runtime

def _subprocess_measure(cutout_json: Dict, dreport, repetitions: int, q: mp.Queue) -> float:
    cutout = dace.SDFG.from_json(cutout_json)
    
    arguments = {}
    # TODO: Store symbolic arguments in file
    for symbol in cutout.free_symbols:
        arguments[str(symbol)] = 32

    for state in cutout.nodes():
        for dnode in state.data_nodes():
            array = cutout.arrays[dnode.data]
            if array.transient:
               continue

            try:
                data = dreport[dnode.data]
                arguments[dnode.data] = dace.data.make_array_from_descriptor(array, data)
            except KeyError:
                arguments[dnode.data] = dace.data.make_array_from_descriptor(array)


    for name, array in list(cutout.arrays.items()):
        if array.transient:
            continue

        if not name in arguments:
            del cutout.arrays[name]


    with dace.config.set_temporary('debugprint', value=False):
        with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=False):
            with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
                cutout.build_folder = "/dev/shm"
                csdfg = cutout.compile()
                for _ in range(repetitions):
                    csdfg(**arguments)

                csdfg.finalize()

    report = cutout.get_latest_report()
    durations = next(iter(next(iter(report.durations.values())).values()))
    q.put(np.median(np.array(durations)))

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

