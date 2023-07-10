import numpy as np
import copy
from numbers import Number
from typing import Tuple, Optional, List, Union, Dict
import os
from subprocess import run
from argparse import Namespace

import dace

from execute.parameters import ParametersProvider
from execute.data import set_input_pattern
from utils.general import get_programs_data, read_source, get_fortran, get_sdfg, get_inputs, get_outputs, \
                          compare_output, compare_output_all, optimize_sdfg
from utils.gpu_general import copy_to_device, print_non_zero_percentage
from measurements.flop_computation import FlopCount, get_number_of_bytes, get_number_of_flops
from measurements.data import ProgramMeasurement
from utils.print import print_with_time

RNG_SEED = 424388


class RunConfig:
    pattern: str
    use_dace_auto_opt: bool
    device: dace.DeviceType
    specialise_symbols: bool

    def __init__(self, pattern: str = None, use_dace_auto_opt: bool = False,
                 device: dace.DeviceType = dace.DeviceType.GPU, specialise_symbols: bool = True):
        self.pattern = pattern
        self.use_dace_auto_opt = use_dace_auto_opt
        self.device = device
        self.specialise_symbols = specialise_symbols

    def set_from_args(self, args: Namespace):
        keys = ['pattern', 'use_dace_auto_opt']
        args_dict = vars(args)
        for key in args_dict:
            if key in keys:
                setattr(self, key, args_dict[key])
        if 'specialise_symbols' in args_dict and args_dict['specialise_symbols']:
            self.specialise_symbols = True
        if 'not_specialise_symbols' in args_dict and args_dict['not_specialise_symbols']:
            self.specialise_symbols = False

    def __len__(self):
        return len(self.pattern)


# Copied and adapted from tests/fortran/cloudsc.py
def test_program(program: str, run_config: RunConfig, sdfg_file: Optional[str] = None) -> bool:
    """
    Tests the given program by comparing the output of the SDFG compiled version to the one compiled directly from
    fortran

    :param program: The program name
    :type program: str
    :param run_config: Configuration how to run it
    :type run_config: RunConfig
    :param sdfg_file: Path to sdfg file. If set will not recreate SDFG but use this one instead, defaults to None
    :type sdfg_file: str, optional
    :return: True if test passes, False otherwise
    :rtype: bool
    """
    assert run_config.device == dace.DeviceType.GPU

    programs_data = get_programs_data()
    params = ParametersProvider(program, testing=True)
    fsource = read_source(program)
    program_name = programs_data['programs'][program]
    routine_name = f'{program_name}_routine'
    ffunc = get_fortran(fsource, program_name, routine_name)
    if sdfg_file is None:
        sdfg = get_sdfg(fsource, program_name)
        add_args = {}
        if run_config.specialise_symbols:
            add_args['symbols'] = params.get_dict()
        optimize_sdfg(sdfg, run_config.device, use_my_auto_opt=not run_config.use_dace_auto_opt, **add_args)
    else:
        print(f"Reading SDFG from {sdfg_file} and compile it")
        sdfg = dace.sdfg.sdfg.SDFG.from_file(sdfg_file)
        sdfg.compile()

    rng = np.random.default_rng(RNG_SEED)
    inputs = get_inputs(program, rng, params)
    outputs_f = get_outputs(program, rng, params)
    outputs_original = copy.deepcopy(outputs_f)
    np.set_printoptions(precision=3)
    if run_config.pattern is not None:
        set_input_pattern(inputs, outputs_f, params, program, run_config.pattern)
    outputs_d_device = copy_to_device(copy.deepcopy(outputs_f))
    sdfg.validate()
    sdfg.simplify(validate_all=True)

    ffunc(**{k.lower(): v for k, v in inputs.items()}, **{k.lower(): v for k, v in outputs_f.items()})
    inputs_device = copy_to_device(inputs)
    sdfg(**inputs_device, **outputs_d_device)

    print_with_time(f"{program} ({program_name}) on {run_config.device}")
    outputs_d = outputs_d_device
    passes_test = compare_output(outputs_f, outputs_d, program, params)
    if compare_output_all(outputs_f, outputs_original, print_if_differ=False):
        print_with_time("!!! Fortran has not changed any output values !!!")
        passes_test = False
    if compare_output_all(outputs_d, outputs_original, print_if_differ=False):
        print_with_time("!!! DaCe has not changed any output values !!!")
        passes_test = False

    # passes_test = compare_output_all(outputs_f, outputs_d)

    if passes_test:
        print_with_time('Success')
    else:
        print_with_time('!!!TEST NOT PASSED!!!')
    return passes_test


def run_program(program: str,  run_config: RunConfig, params: ParametersProvider, repetitions: int = 1,
                sdfg_file: Optional[str] = None):
    """
    Runs Programs

    :param program: Name of the program
    :type program: str
    :param run_config: Configuration how to run it
    :type run_config: RunConfig
    :param parameters: The parameters to use.
    :type parameters: ParametersProvider
    :param repetitions: The number of repetitions to run the program, defaults to 1
    :type repetitions: int, optional
    :param sdfg_file: Path to sdfg file. If set will not recreate SDFG but use this one instead, defaults to None
    :type sdfg_file: str, optional
    """
    programs = get_programs_data()['programs']
    print(f"Run {program} ({programs[program]}) for {repetitions} time on device {run_config.device}")
    fsource = read_source(program)
    program_name = programs[program]
    if sdfg_file is None:
        sdfg = get_sdfg(fsource, program_name)
        additional_args = {}
        if run_config.specialise_symbols:
            additional_args['symbols'] = params.get_dict()

        optimize_sdfg(sdfg, run_config.device, use_my_auto_opt=not run_config.use_dace_auto_opt, **additional_args)
    else:
        print(f"Reading SDFG from {sdfg_file} and compile it")
        sdfg = dace.sdfg.sdfg.SDFG.from_file(sdfg_file)
        sdfg.compile()

    rng = np.random.default_rng(RNG_SEED)
    inputs = get_inputs(program, rng, params)
    print(f"[execute_dace::run_program] KLON: {inputs['KLON']}, KLEV: {inputs['KLEV']}, NCLV: {inputs['NCLV']}, NBLOCKS: {inputs['NBLOCKS']}")
    inputs = copy_to_device(inputs)
    outputs = copy_to_device(get_outputs(program, rng, params))
    if run_config.pattern is not None:
        set_input_pattern(inputs, outputs, params, program, run_config.pattern)

    for _ in range(repetitions):
        sdfg(**inputs, **outputs)


def compile_for_profile(program: str, params: Union[ParametersProvider, Dict[str, Number]],
                        run_config: RunConfig) -> dace.SDFG:
    """
    Compile the given program for profiliation. Meaning a total runtime timer is added

    :param program: Name of the program
    :type program: str
    :param params: The parameters to use.
    :type params: Union[ParametersProvider, Dict[str, Number]]
    :param run_config: Configuration how to run it
    :type run_config: RunConfig
    :return: Generated SDFG
    :rtype: dace.SDFG
    """
    programs = get_programs_data()['programs']
    fsource = read_source(program)
    program_name = programs[program]
    sdfg = get_sdfg(fsource, program_name)
    add_args = {}
    params_dict = params
    if isinstance(params, ParametersProvider):
        params_dict = params.get_dict()
    if run_config.specialise_symbols:
        add_args['symbols'] = params_dict
    optimize_sdfg(sdfg, run_config.device, use_my_auto_opt=not run_config.use_dace_auto_opt, **add_args)

    sdfg.instrument = dace.InstrumentationType.Timer
    sdfg.compile()
    return sdfg


def profile_program(program: str, run_config: RunConfig, params: ParametersProvider,
                    repetitions=10) -> ProgramMeasurement:

    results = ProgramMeasurement(program, params)

    programs = get_programs_data()['programs']
    print_with_time(f"Profile {program}({programs[program]}) rep={repetitions}")
    routine_name = f"{programs[program]}_routine"

    sdfg = compile_for_profile(program, params, run_config)

    rng = np.random.default_rng(RNG_SEED)
    inputs = get_inputs(program, rng, params)
    outputs = get_outputs(program, rng, params)
    if run_config.pattern is not None:
        set_input_pattern(inputs, outputs, params, program, run_config.pattern)

    sdfg.clear_instrumentation_reports()
    print_with_time("Measure total runtime")
    inputs = copy_to_device(inputs)
    outputs = copy_to_device(outputs)
    for i in range(repetitions):
        sdfg(**inputs, **outputs)

    # variables = {'cloudsc_class2_781': 'ZLIQFRAC', 'cloudsc_class2_1762': 'ZSNOWCLD2',
    #              'cloudsc_class2_1516': 'ZCLDTOPDIST2', 'my_test': 'ARRAY_A'}
    # print_non_zero_percentage(outputs, variables[program])
    reports = sdfg.get_instrumentation_reports()

    results.add_measurement("Total time", "ms")
    for report in reports:
        keys = list(report.durations[(0, -1, -1)][f"SDFG {routine_name}"].keys())
        key = keys[0]
        if len(keys) > 1:
            print(f"WARNING: Report has more than one key, taking only the first one. keys: {keys}")
        results.add_value("Total time",
                          float(report.durations[(0, -1, -1)][f"SDFG {routine_name}"][key][0]))

    return results


def get_roofline_data(program: str, params: ParametersProvider,
                      pattern: Optional[str] = None) -> Tuple[FlopCount, Number]:
    rng = np.random.default_rng(RNG_SEED)
    inputs = get_inputs(program, rng, params)
    outputs = get_outputs(program, rng, params)
    if pattern is not None:
        set_input_pattern(inputs, outputs, params, program, pattern)
    flop_count = get_number_of_flops(params, inputs, outputs, program)
    bytes = get_number_of_bytes(params, inputs, outputs, program)
    return (flop_count, bytes)


def get_command_args_single_run(program: str, run_config: RunConfig) -> List[str]:
    """
    Gets the commands required to run the given program with the given config once.

    :param program: The name of the program
    :type program: str
    :param run_config: The desired config to run it with
    :type run_config: RunConfig
    :return: List of commands, as required to subprocess.run
    :rtype: List[str]
    """
    test_program_path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'run_program.py')
    command_program = ['python3', test_program_path, program, '--repetitions', '1']
    if run_config.use_dace_auto_opt:
        command_program.append('--use-dace-auto-opt')
    if run_config.pattern is not None:
        command_program.extend(['--pattern', run_config.pattern])
    return command_program


def gen_ncu_report(program: str, report_filename: str, run_config: RunConfig, ncu_args: List[str] = [],
                   program_args: List[str] = []) -> bool:
    """
    Generates a ncu report into the given report filename using the given additional ncu arguments

    :param program: The name of the program to profile using ncu
    :type program: str
    :param report_filename: The path to where the report should be save
    :type report_filename: str
    :param run_config: The config to run the given program with
    :type run_config: RunConfig
    :param ncu_args: Any additional arguments for ncu, optional
    :type ncu_args: List[str]
    :param program_args: Any additional arguments for the program, optional
    :type program_args: List[str]
    :return: True if ncu report was successfully created, False otherwise
    :rtype: bool
    """
    print_with_time(f"Create ncu report and save it into {report_filename}")
    command_program = get_command_args_single_run(program, run_config)
    command_program.extend(program_args)
    ncu_command = ['ncu', '--force-overwrite', '--export', report_filename, *ncu_args]
    ncu_output = run([*ncu_command, *command_program], capture_output=True)
    if ncu_output.returncode != 0:
        print("Failed to run the program with ncu")
        print(ncu_output.stdout.decode('UTF-8'))
        print(ncu_output.stderr.decode('UTF-8'))
        return False
    return True


def gen_nsys_report(program: str, report_filename: str, run_config: RunConfig) -> bool:
    """
    Generates a nsys report and saves it into the given report filename.

    :param program: The name of the program to profile using ncu
    :type program: str
    :param report_filename: The path to where the report should be save
    :type report_filename: str
    :param run_config: The config to run the given program with
    :type run_config: RunConfig
    :return: True if ncu report was successfully created, False otherwise
    :rtype: bool
    """
    print_with_time(f"Create nsys report and save it into {report_filename}")
    command_nsys = ['nsys', 'profile', '--force-overwrite', 'true', '--output', report_filename]
    command_program = get_command_args_single_run(program, run_config)
    nsys_output = run([*command_nsys, *command_program], capture_output=True)
    if nsys_output.returncode != 0:
        print("Failed to run the program with nsys")
        print(nsys_output.stdout.decode('UTF-8'))
        print(nsys_output.stderr.decode('UTF-8'))
        return False
    return True
