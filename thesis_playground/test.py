import numpy as np
from numpy import f2py
from numbers import Number
from typing import Dict, Union, List
import copy
from importlib import import_module
import os
import sys
import tempfile
from tabulate import tabulate
from argparse import ArgumentParser
import json

import dace
from dace.frontend.fortran import fortran_parser
from dace.sdfg import utils
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes import RemoveUnusedSymbols, ScalarToSymbolPromotion

with open('thesis_playground/programs.json') as file:
    programs_data = json.load(file)
    programs = programs_data['programs']
    program_parameters = programs_data['program_parameters']
    program_inputs = programs_data['program_inputs']
    program_outputs = programs_data['program_outputs']
    parameters = programs_data['parameters']

not_working = ['cloudsc_class1_670', 'cloudsc_class2_1001']

# Copied from tests/fortran/cloudsc.py as well as the functions/dicts below
def read_source(filename: str, extension: str = 'f90') -> str:
    source = None
    with open(os.path.join(os.path.dirname(__file__), f'{filename}.{extension}'), 'r') as file:
        source = file.read()
    assert source
    return source


def get_fortran(source: str, program_name: str, subroutine_name: str, fortran_extension: str = '.f90'):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        # Can set verbose to true to get more output when compiling
        f2py.compile(source, modulename=program_name, verbose=False, extension=fortran_extension)
        sys.path.append(tmp_dir)
        module = import_module(program_name)
        function = getattr(module, subroutine_name)
        os.chdir(cwd)
        return function


def get_sdfg(source: str, program_name: str, normalize_offsets: bool = False) -> dace.SDFG:

    intial_sdfg = fortran_parser.create_sdfg_from_string(source, program_name)
    
    # Find first NestedSDFG
    sdfg = None
    for state in intial_sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                sdfg = node.sdfg
                break
    if not sdfg:
        raise ValueError("SDFG not found.")

    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()

    if normalize_offsets:
        my_simplify = Pipeline([RemoveUnusedSymbols(), ScalarToSymbolPromotion()])
    else:
        my_simplify = Pipeline([RemoveUnusedSymbols()])
    my_simplify.apply_pass(sdfg, {})

    if normalize_offsets:
        utils.normalize_offsets(sdfg)

    return sdfg


data = {
    'PTSPHY': (0,),
    'R2ES': (0,),
    'R3IES': (0,),
    'R3LES': (0,),
    'R4IES': (0,),
    'R4LES': (0,),
    'RALSDCP': (0,),
    'RALVDCP': (0,),
    'RAMIN': (0,),
    'RCOVPMIN': (0,),
    'RCLDTOPCF': (0,),
    'RD': (0,),
    'RDEPLIQREFDEPTH': (0,),
    'RDEPLIQREFRATE': (0,),
    'RG': (0,),
    'RICEINIT': (0,),
    'RKOOP1': (0,),
    'RKOOP2': (0,),
    'RKOOPTAU': (0,),
    'RLMIN': (0,),
    'RLSTT': (0,),
    'RLVTT': (0,),
    'RPECONS': (0,),
    'RPRECRHMAX': (0,),
    'RTAUMEL': (0,),
    'RTHOMO': (0,),
    'RTT': (0,),
    'RV': (0,),
    'RVRFACTOR': (0,),
    'ZEPSEC': (0,),
    'ZEPSILON': (0,),
    'ZRG_R': (0,),
    'ZRLDCP': (0,),
    'ZQTMST': (0,),
    'ZVPICE': (0,),
    'ZVPLIQ': (0,),
    'IPHASE': (parameters['NCLV'],),
    'PAPH': (parameters['KLON'], parameters['KLEV']+1),
    'PAP': (parameters['KLON'], parameters['KLEV']),
    'PCOVPTOT': (parameters['KLON'], parameters['KLEV']),
    'PFCQLNG': (parameters['KLON'], parameters['KLEV']+1),
    'PFCQNNG': (parameters['KLON'], parameters['KLEV']+1),
    'PFCQRNG': (parameters['KLON'], parameters['KLEV']+1),
    'PFCQSNG': (parameters['KLON'], parameters['KLEV']+1),
    'PFHPSL': (parameters['KLON'], parameters['KLEV']+1),
    'PFHPSN': (parameters['KLON'], parameters['KLEV']+1),
    'PFPLSL': (parameters['KLON'], parameters['KLEV']+1),
    'PFPLSN': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQIF': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQITUR': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQLF': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQLTUR': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQRF': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQSF': (parameters['KLON'], parameters['KLEV']+1),
    'PLUDE': (parameters['KLON'], parameters['KLEV']),
    'PSUPSAT': (parameters['KLON'], parameters['KLEV']),
    'PVFI': (parameters['KLON'], parameters['KLEV']),
    'PVFL': (parameters['KLON'], parameters['KLEV']),
    'tendency_loc_a': (parameters['KLON'], parameters['KLEV']),
    'tendency_loc_cld': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'tendency_loc_q': (parameters['KLON'], parameters['KLEV']),
    'tendency_loc_T': (parameters['KLON'], parameters['KLEV']),
    'tendency_tmp_t': (parameters['KLON'], parameters['KLEV']),
    'tendency_tmp_q': (parameters['KLON'], parameters['KLEV']),
    'tendency_tmp_a': (parameters['KLON'], parameters['KLEV']),
    'tendency_tmp_cld': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZA': (parameters['KLON'], parameters['KLEV']),
    'ZAORIG': (parameters['KLON'], parameters['KLEV']),
    'ZCLDTOPDIST': (parameters['KLON'],),
    'ZCONVSINK': (parameters['KLON'], parameters['NCLV']),
    'ZCONVSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZCORQSICE': (parameters['KLON']),
    'ZCORQSLIQ': (parameters['KLON']),
    'ZCOVPTOT': (parameters['KLON'],),
    'ZCOVPCLR': (parameters['KLON'],),
    'ZCOVPMAX': (parameters['KLON'],),
    'ZDTGDP': (parameters['KLON'],),
    'ZICENUCLEI': (parameters['KLON'],),
    'ZRAINCLD': (parameters['KLON'],),
    'ZSNOWCLD': (parameters['KLON'],),
    'ZDA': (parameters['KLON'],),
    'ZDP': (parameters['KLON'],),
    'ZRHO': (parameters['KLON'],),
    'ZFALLSINK': (parameters['KLON'], parameters['NCLV']),
    'ZFALLSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZFOKOOP': (parameters['KLON'],),
    'ZFLUXQ': (parameters['KLON'], parameters['NCLV']),
    'ZFOEALFA': (parameters['KLON'], parameters['KLEV']+1),
    'ZICECLD': (parameters['KLON'],),
    'ZICEFRAC': (parameters['KLON'], parameters['KLEV']),
    'ZICETOT': (parameters['KLON'],),
    'ZLI': (parameters['KLON'], parameters['KLEV']),
    'ZLIQFRAC': (parameters['KLON'], parameters['KLEV']),
    'ZLNEG': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZPFPLSX': (parameters['KLON'], parameters['KLEV']+1, parameters['NCLV']),
    'ZPSUPSATSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZMELTMAX': (parameters['KLON'],),
    'ZQPRETOT': (parameters['KLON'],),
    'ZQSLIQ': (parameters['KLON'], parameters['KLEV']),
    'ZQSICE': (parameters['KLON'], parameters['KLEV']),
    'ZQX': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZQX0': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZQXFG': (parameters['KLON'], parameters['NCLV']),
    'ZQXN': (parameters['KLON'], parameters['NCLV']),
    'ZQXN2D': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZSOLAC': (parameters['KLON'],),
    'ZSOLQA': (parameters['KLON'], parameters['NCLV'], parameters['NCLV']),
    'ZSUPSAT': (parameters['KLON'],),
    'ZTP1': (parameters['KLON'], parameters['KLEV']),
    'PT': (parameters['KLON'], parameters['KLEV']),
    'PQ': (parameters['KLON'], parameters['KLEV']),
    'PA': (parameters['KLON'], parameters['KLEV']),
    'PCLV': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
}

# Copied from tests/fortran/cloudsc.py
def get_inputs(program: str, rng: np.random.Generator) -> Dict[str, Union[Number, np.ndarray]]:
    inp_data = dict()
    for p in program_parameters[program]:
        inp_data[p] = parameters[p]
    for inp in program_inputs[program]:
        shape = data[inp]
        if shape == (0,):  # Scalar
            inp_data[inp] = rng.random()
        else:
            inp_data[inp] = np.asfortranarray(rng.random(shape))
    return inp_data


# Copied from tests/fortran/cloudsc.py
# TODO: Init to 0?
def get_outputs(program: str, rng: np.random.Generator) -> Dict[str, Union[Number, np.ndarray]]:
    out_data = dict()
    for out in program_outputs[program]:
        shape = data[out]
        if shape == (0,):  # Scalar
            raise NotImplementedError
        else:
            out_data[out] = np.asfortranarray(rng.random(shape))
    return out_data


# Copied from tests/fortran/cloudsc.py
def test_program(program: str, device: dace.DeviceType, normalize_memlets: bool):

    fsource = read_source(program)
    program_name = programs[program]
    routine_name = f'{program_name}_routine'
    ffunc = get_fortran(fsource, program_name, routine_name)
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    if device == dace.DeviceType.GPU:
        auto_optimize(sdfg, device)

    rng = np.random.default_rng(42)
    inputs = get_inputs(program, rng)
    outputs_f = get_outputs(program, rng)
    outputs_d = copy.deepcopy(outputs_f)
    sdfg.validate()
    sdfg.simplify(validate_all=True)

    ffunc(**{k.lower(): v for k, v in inputs.items()}, **{k.lower(): v for k, v in outputs_f.items()})
    sdfg(**inputs, **outputs_d)


    print(f"{program} ({program_name}) on {device} with{' ' if normalize_memlets else 'out '}normalize memlets")
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
    print("Success")


def get_stats(array: List):
    return [max(array), min(array), np.average(array), np.median(array)]


def print_stats(array: List):
    print(f"max: {max(array)}, min: {min(array)}, avg: {np.average(array)}, median: {np.median(array)}")


def profile_program(program: str, device=dace.DeviceType.GPU, normalize_memlets=False, repetitions=10,
        print_results=False) -> Dict[str, List[Number]]:

    results = {}
    fsource = read_source(program)
    program_name = programs[program]
    routine_name = f'{program_name}_routine'
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    auto_optimize(sdfg, device)

    rng = np.random.default_rng(42)
    inputs = get_inputs(program, rng)
    outputs = get_outputs(program, rng)

    for node, parent in sdfg.all_nodes_recursive():
        if type(node) == dace.sdfg.nodes.MapEntry:
            id = parent.node_id(node)
            node.instrument = dace.InstrumentationType.GPU_Events
            print(f"Instrumented map with id {id} with {node.instrument}")

    sdfg.clear_instrumentation_reports()
    for i in range(repetitions):
        sdfg(**inputs, **outputs)
    reports = sdfg.get_instrumentation_reports()
    runtimes_gpu = []
    for report in reports:
        key = list(report.durations.keys())[0]
        state_key = list(report.durations[key])[0]
        if (len(list(report.durations.keys())) > 1 or len(list(report.durations[key])) > 1):
            print("*** WARNING: more than one keys found, but only using one")
        runtimes_gpu.append(report.durations[key][state_key][13533068620211794961][0])

    results["GPU time [ms]"] = get_stats(runtimes_gpu)

    for node, parent in sdfg.all_nodes_recursive():
        if type(node) == dace.sdfg.nodes.MapEntry:
            id = parent.node_id(node)
            node.instrument = dace.InstrumentationType.LIKWID_GPU
            print(f"Instrumented map with id {id} with {node.instrument}")

    sdfg.clear_instrumentation_reports()
    for i in range(repetitions):
        sdfg(**inputs, **outputs)
    reports = sdfg.get_instrumentation_reports()
    runtimes = []
    adds, muls, fmas = [], [], []
    for report in reports:
        key = list(report.durations.keys())[0]
        scope_key = list(report.counters[key].keys())[0]
        if (len(list(report.durations.keys())) > 1 or len(list(report.durations[key])) > 1):
            print("*** WARNING: more than one keys found, but only using one")
        runtimes.append(report.durations[key]['Timer'][0][0])
        adds.append(report.counters[key][scope_key]['SMSP_SASS_THREAD_INST_EXECUTED_OP_FADD_PRED_ON_SUM'][0])
        muls.append(report.counters[key][scope_key]['SMSP_SASS_THREAD_INST_EXECUTED_OP_FMUL_PRED_ON_SUM'][0])
        fmas.append(report.counters[key][scope_key]['SMSP_SASS_THREAD_INST_EXECUTED_OP_FFMA_PRED_ON_SUM'][0])

    results["LIKWID time [ms]"] = get_stats(runtimes)
    results["LIKWID adds"] = get_stats(adds)
    results["LIKWID muls"] = get_stats(muls)
    results["LIKWID fmas"] = get_stats(fmas)
    if print_results:
        print(f"{program}({programs[program]} rep={repetitions}")
        print("GPU Time [ms]:     ", end="")
        print_stats(runtimes_gpu)
        print("LIKWID Time [ms]: ", end="")
        print_stats(runtimes)
        print("LIKWID adds:      ", end="")
        print_stats(adds)
        print("LIKWID muls:      ", end="")
        print_stats(muls)
        print("LIKWID fmas:      ", end="")
        print_stats(fmas)
    return results


def run_program(program: str, repetitions: int=1, device=dace.DeviceType.GPU, normalize_memlets=False):
    fsource = read_source(program)
    program_name = programs[program]
    routine_name = f'{program_name}_routine'
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    auto_optimize(sdfg, device)

    rng = np.random.default_rng(42)
    inputs = get_inputs(program, rng)
    outputs = get_outputs(program, rng)
    for _ in range(repetitions):
        sdfg(**inputs, **outputs)


def test_programs(programs: List[str], repetitions: int, device: dace.DeviceType):
    for program in programs:
        if program not in not_working:
            try:
                test_program(program, dace.DeviceType.CPU, False)
                test_program(program, dace.DeviceType.CPU, True)
                test_program(program, dace.DeviceType.GPU, False)
                test_program(program, dace.DeviceType.GPU, True)
            except AttributeError:
                print(f"ERROR: could not run {program} due to an AttributeError")


def profile_programs(programs: List[str], repetitions: int, device: dace.DeviceType):
    results = []
    headers = ['Program', 'measurement', 'max', 'min', 'avg', 'median']
    for program in programs:
        if program not in not_working:
            result = profile_program(program, repetitions=repetitions, device=device)
            for key in result:
                results.append([program, key, *result[key]])

    print(tabulate(results, headers=headers))


def run_programs(programs: List[str], repetitions: int, device: dace.DeviceType):
    for program in programs:
        run_program(program, repetitions=repetitions, device=device)


def main():
    parser = ArgumentParser()
    parser.add_argument('action',
            type=str,
            choices=['test', 'profile', 'run'],
            help='The action to perform, test will ignore device and repetitions flags.')
    parser.add_argument(
        '-c', '--classes',
        type=str,
        nargs='+',
        help='Names of the loop classes/programs to use. Can be several separated by space')
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

    args = parser.parse_args()

    print(args)
    devices = {'GPU': dace.DeviceType.GPU, 'CPU': dace.DeviceType.CPU}
    selected_programs = programs if args.classes is None else args.classes
    action_functions = {'test': test_programs, 'profile': profile_programs, 'run': run_programs}
    action_functions[args.action](selected_programs, args.repetitions, devices[args.device])


if __name__=="__main__":
    main()
    # test_program('cloudsc_class2_1516', dace.DeviceType.GPU, False)
    # test_program('cloudsc_class1_658', dace.DeviceType.GPU, False)
    # test_program('cloudsc_class1_2783', dace.DeviceType.GPU, False)
    # profile_program('cloudsc_class1_658', print_results=True, repetitions=20)
    # profile_program('cloudsc_class1_2783', print_results=True, repetitions=20)
    # test_program('cloudsc_class2_1001', dace.DeviceType.GPU, False)
    # test_program('cloudsc_class1_670', dace.DeviceType.GPU, False)
    # test_program('mwe_test', dace.DeviceType.GPU, False)
    # test_programs()
    # profile_programs()
