import numpy as np
from numbers import Number
from typing import Dict, List, Optional
import copy
from tabulate import tabulate
from argparse import ArgumentParser
import json
from datetime import datetime

import dace
from dace.transformation.auto.auto_optimize import auto_optimize
from utils import read_source, get_fortran, get_sdfg, get_inputs, get_outputs, get_programs_data


# Copied and adapted from tests/fortran/cloudsc.py
def test_program(program: str, device: dace.DeviceType, normalize_memlets: bool):
    """
    Tests the given program by comparing the output of the SDFG compiled version to the one compiled directly from
    fortran

    :param program: The program name
    :type program: str
    :param device: The deive
    :type device: dace.DeviceType
    :param normalize_memlets: If memlets should be normalized
    :type normalize_memlets: bool
    """

    programs_data = get_programs_data()
    fsource = read_source(program)
    program_name = programs_data['programs'][program]
    routine_name = f'{program_name}_routine'
    ffunc = get_fortran(fsource, program_name, routine_name)
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    if device == dace.DeviceType.GPU:
        auto_optimize(sdfg, device)

    rng = np.random.default_rng(42)
    inputs = get_inputs(program, rng, testing_dataset=True)
    outputs_f = get_outputs(program, rng, testing_dataset=True)
    outputs_d = copy.deepcopy(outputs_f)
    sdfg.validate()
    sdfg.simplify(validate_all=True)

    ffunc(**{k.lower(): v for k, v in inputs.items()}, **{k.lower(): v for k, v in outputs_f.items()})
    sdfg(**inputs, **outputs_d)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {program} ({program_name}) on {device} with"
          f"{' ' if normalize_memlets else 'out '}normalize memlets")
    for k in outputs_f.keys():
        farr = outputs_f[k]
        darr = outputs_f[k]
        assert np.allclose(farr, darr)
        print(f"variable {k:20} ", end="")
        print(f"Sum: {farr.sum():.2e}", end=", ")
        print(f"avg: {np.average(farr):.2e}", end=", ")
        print(f"median: {np.median(farr):.2e}", end=", ")
        print(f"nnz: {np.count_nonzero(farr)}", end=", ")
        print(f"#: {np.prod(farr.shape)}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Success")


def get_stats(array: List):
    return {'max': min(array), 'min': min(array), 'avg': np.average(array), 'median': np.median(array)}


def print_stats(array: List):
    print(f"max: {max(array)}, min: {min(array)}, avg: {np.average(array)}, median: {np.median(array)}")


def compile_for_profile(program: str, device: dace.DeviceType, normalize_memlets: bool) -> dace.SDFG:
    programs = get_programs_data()['programs']
    fsource = read_source(program)
    program_name = programs[program]
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    auto_optimize(sdfg, device)
    sdfg.instrument = dace.InstrumentationType.Timer
    sdfg.compile()
    return sdfg


def profile_program(program: str, device=dace.DeviceType.GPU, normalize_memlets=False, repetitions=10)\
        -> Dict[str, List[Number]]:

    programs = get_programs_data()['programs']
    print(f"Profile {program}({programs[program]}) rep={repetitions}")
    routine_name = f"{programs[program]}_routine"
    results = {}

    sdfg = compile_for_profile(program, device, normalize_memlets)

    rng = np.random.default_rng(42)
    inputs = get_inputs(program, rng)
    outputs = get_outputs(program, rng)

    # for node, parent in sdfg.all_nodes_recursive():
    #     if type(node) == dace.sdfg.nodes.MapEntry:
    #         id = parent.node_id(node)
    #         node.instrument = dace.InstrumentationType.GPU_Events
    #         print(f"Instrumented map with id {id} with {node.instrument}")

    # sdfg.clear_instrumentation_reports()
    # for i in range(repetitions):
    #     sdfg(**inputs, **outputs)
    # reports = sdfg.get_instrumentation_reports()
    # runtimes_gpu = []
    # keys_gpu = set()
    # for report in reports:
    #     key = list(report.durations.keys())[0]
    #     keys_gpu.add(key)
    #     state_key = list(report.durations[key])[0]
    #     if (len(list(report.durations.keys())) > 1 or len(list(report.durations[key])) > 1):
    #         print("*** WARNING: more than one keys found, but only using one")
    #     runtimes_gpu.append(report.durations[key][state_key][13533068620211794961][0])

    # results["GPU time [ms]"] = get_stats(runtimes_gpu)
    # results["GPU time [ms]"]["data"] = [float(i) for i in runtimes_gpu]
    # results["GPU keys"] = list(keys_gpu)

    # sdfg.instrument = dace.InstrumentationType.Timer
    # for node, parent in sdfg.all_nodes_recursive():
    #     if type(node) == dace.sdfg.nodes.MapEntry:
    #         id = parent.node_id(node)
    #         node.instrument = dace.InstrumentationType.LIKWID_GPU
    #         print(f"Instrumented map with id {id} with {node.instrument}")

    sdfg.clear_instrumentation_reports()
    for i in range(repetitions):
        sdfg(**inputs, **outputs)
    reports = sdfg.get_instrumentation_reports()
    total_times = []
    # runtimes = []
    # adds, muls, fmas = [], [], []
    # keys_likwid = set()
    for report in reports:
        # keys = list(report.durations.keys())
        # keys = [k for k in keys if k[1] != -1]
        # key = keys[0]
        # keys_likwid.add(key)
        # if (len(keys) > 1 or len(list(report.durations[key])) > 1):
        #     print("*** WARNING: more than one keys found, but only using one")
        # runtimes.append(report.durations[key]['Timer'][0][0])
        total_times.append(report.durations[(0, -1, -1)][f"SDFG {routine_name}"][13533068620211794961][0])

    # results["LIKWID time [ms]"] = get_stats(runtimes)
    # results["LIKWID time [ms]"]["data"] = [float(r) for r in runtimes]
    # results["LIKWID keys"] = list(keys_likwid)
    results["Total time [ms]"] = get_stats(total_times)
    results["Total time [ms]"]["data"] = [float(r) for r in total_times]

    return results


def run_program(program: str, repetitions: int = 1, device=dace.DeviceType.GPU, normalize_memlets=False):
    programs = get_programs_data()['programs']
    print(f"Run {program} ({programs[program]}) for {repetitions} time on device {device}")
    fsource = read_source(program)
    program_name = programs[program]
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    auto_optimize(sdfg, device)

    rng = np.random.default_rng(42)
    inputs = get_inputs(program, rng)
    outputs = get_outputs(program, rng)
    for _ in range(repetitions):
        sdfg(**inputs, **outputs)


def test_programs(programs: List[str], repetitions: int, device: dace.DeviceType):
    for program in programs:
        try:
            test_program(program, dace.DeviceType.CPU, False)
            test_program(program, dace.DeviceType.CPU, True)
            test_program(program, dace.DeviceType.GPU, False)
            test_program(program, dace.DeviceType.GPU, True)
        except AttributeError:
            print(f"ERROR: could not run {program} due to an AttributeError")


def profile_programs(programs: List[str], repetitions: int, device: dace.DeviceType,
                     output_file: Optional[str] = None):
    results_flat = []
    results_dict = {}
    headers = ['Program', 'measurement', 'max', 'min', 'avg', 'median']
    for program in programs:
        result = profile_program(program, repetitions=repetitions, device=device)
        results_dict[program] = result
        for key in result:
            if key not in ['LIKWID keys', 'GPU keys']:
                results_flat.append([program, key])
                for name in headers[2:]:
                    results_flat[-1].append(result[key][name])

    if output_file is None:
        print(tabulate(results_flat, headers=headers))
    else:
        with open(output_file, mode='w') as file:
            json.dump(results_dict, file)


def run_programs(programs: List[str], repetitions: int, device: dace.DeviceType):
    for program in programs:
        run_program(program, repetitions=repetitions, device=device)


def main():
    parser = ArgumentParser()
    parser.add_argument(
            'action',
            type=str,
            choices=['test', 'profile', 'run'],
            help='The action to perform, test will ignore device and repetitions flags.')
    parser.add_argument(
        '-p', '--programs',
        type=str,
        nargs='+',
        help='Names of the programs to use. Can be several separated by space')
    parser.add_argument(
            '-r', '--repetitions',
            type=int,
            default=1,
            help='Number of repetitions')
    parser.add_argument(
            '-d', '--device',
            type=str,
            default='GPU',
            choices=['GPU', 'CPU'],
            help="The device to run the code on")
    parser.add_argument(
            '-o', '--output',
            type=str,
            default=None,
            help="Writes the output to the given file in json format, only works for profile")
    parser.add_argument(
            '-c', '--class',
            type=int,
            choices=[1, 2, 3],
            default=None,
            dest='kernel_class',
            help="Run all programs of a given class")

    args = parser.parse_args()

    programs = get_programs_data()['programs']
    devices = {'GPU': dace.DeviceType.GPU, 'CPU': dace.DeviceType.CPU}
    selected_programs = programs if args.programs is None else args.programs
    if args.kernel_class is not None:
        selected_programs = [p for p in programs if p.startswith(f"cloudsc_class{args.kernel_class}")]
    action_functions = {'test': test_programs, 'profile': profile_programs, 'run': run_programs}
    function_args = [selected_programs, args.repetitions, devices[args.device]]
    if args.action == 'profile':
        function_args.append(args.output)
    action_functions[args.action](*function_args)


if __name__ == "__main__":
    main()
