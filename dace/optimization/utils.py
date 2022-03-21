import os
import math
import dace
import itertools
import numpy as np

from typing import Dict

from dace.codegen.instrumentation.data import data_report

def measure(sdfg, dreport=None, repetitions = 5):
    print("Measuring on main")
    arguments = {}
    for cstate in sdfg.nodes():
        for dnode in cstate.data_nodes():
            array = sdfg.arrays[dnode.data]

            if dreport is not None:
                try:
                    data = dreport.get_first_version(dnode.data)
                    if data.shape != array.shape:
                        data = np.random.rand(*array.shape)
                except KeyError:
                    print("Missing data in dreport, random array")
                    data = np.random.rand(*array.shape)
                
                arguments[dnode.data] = dace.data.make_array_from_descriptor(array, data)
            else:
                print("No dreport available")
                arguments[dnode.data] = dace.data.make_array_from_descriptor(array, np.random.rand(*array.shape))

    try:
        with dace.config.set_temporary('debugprint', value=False):
            with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=False):
                with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
                    sdfg.build_folder = "/dev/shm"
                    csdfg = sdfg.compile()

                    for _ in range(repetitions):
                        csdfg(**arguments)

                    csdfg.finalize()
    except Exception as e:
        print(e)
        return math.inf

    report = sdfg.get_latest_report()
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
