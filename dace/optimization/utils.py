import os
import pickle
import math
import dace
import itertools
import numpy as np
import traceback

from typing import Dict
import multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method("spawn")

from dace.codegen.instrumentation.data import data_report
from dace.codegen import exceptions as cgx

def measure(sdfg, dreport=None):
    arguments = {}
    for cstate in sdfg.nodes():
        for dnode in cstate.data_nodes():
            array = sdfg.arrays[dnode.data]

            if dreport is not None:
                try:
                    data = dreport.get_first_version(dnode.data)
                    arguments[dnode.data] = dace.data.make_array_from_descriptor(array, data)
                except KeyError:
                    print("Missing data in dreport, random array")
                    arguments[dnode.data] = dace.data.make_array_from_descriptor(array, np.random.rand(*array.shape))
            else:
                print("No dreport available")
                arguments[dnode.data] = dace.data.make_array_from_descriptor(array, np.random.rand(*array.shape))

    try:
        with dace.config.set_temporary('debugprint', value=False):
            with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=False):
                with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
                    sdfg.build_folder = "/dev/shm"
                    csdfg = sdfg.compile()

                    for _ in range(50):
                        csdfg(**arguments)

                    csdfg.finalize()
    except (KeyError, cgx.CompilationError) as e:
        print(e)
        return math.inf

    report = sdfg.get_latest_report()
    durations = next(iter(next(iter(report.durations.values())).values()))
    return np.median(np.array(durations))

def subprocess_measure(cutout: dace.SDFG, sdfg, repetitions: int = 5, timeout: float = 300.0) -> float:
    q = mp.Queue()
    proc = MeasureProcess(target=_subprocess_measure, args=(sdfg, cutout.to_json(), repetitions, q))
    proc.start()
    proc.join(timeout)

    if proc.exitcode != 0:
        print("Error occured during measuring")
        return math.inf

    if proc.exception:
        error, traceback = proc.exception
        print("Error occured during measuring: ", error)
        runtime = math.inf

    runtime = q.get()

    return runtime

def _subprocess_measure(sdfg_json: Dict, cutout_json: Dict, repetitions: int, q) -> float:
    #sdfg = dace.SDFG.from_json(sdfg_json)
    #print("SDFG loaded")
    cutout = dace.SDFG.from_json(cutout_json)
    dreport = pickle.loads(sdfg_json)

    arguments = {}
    for symbol in cutout.free_symbols:
        arguments[str(symbol)] = 4

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
                cutout.build_folder = "/dev/shm"
                csdfg = cutout.compile()
                print("compiled")
                for _ in range(repetitions):
                    csdfg(**arguments)

                csdfg.finalize()

    print("ran")
    report = cutout.get_latest_report()
    durations = next(iter(next(iter(report.durations.values())).values()))
    q.put(np.median(np.array(durations)))
    print("Finished")

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
