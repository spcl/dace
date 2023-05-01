import numpy as np
import copy
from numbers import Number
from typing import Tuple, Optional

import dace

from data import get_program_parameters_data, set_input_pattern
from utils.general import get_programs_data, read_source, get_fortran, get_sdfg, get_inputs, get_outputs, \
                          compare_output, compare_output_all, copy_to_device, optimize_sdfg, copy_to_host, \
                          print_non_zero_percentage
from flop_computation import FlopCount, get_number_of_bytes, get_number_of_flops
from measurement_data import ProgramMeasurement
from utils.print import print_with_time

RNG_SEED = 42


# Copied and adapted from tests/fortran/cloudsc.py
def test_program(program: str, use_my_auto_opt: bool, device: dace.DeviceType, normalize_memlets: bool,
                 pattern: Optional[str] = None) -> bool:
    """
    Tests the given program by comparing the output of the SDFG compiled version to the one compiled directly from
    fortran

    :param program: The program name
    :type program: str
    :param use_my_auto_opt: Flag to control if my custon auto_opt should be used
    :type use_my_auto_opt: bool
    :param device: The deive
    :type device: dace.DeviceType
    :param normalize_memlets: If memlets should be normalized
    :type normalize_memlets: bool
    :param pattern: Name of pattern to apply to in- and output. If empty none will be applied. defaults to None,
    optional
    :type pattern: Optional[str]
    :return: True if test passes, False otherwise
    :rtype: bool
    """
    assert device == dace.DeviceType.GPU

    programs_data = get_programs_data()
    fsource = read_source(program)
    program_name = programs_data['programs'][program]
    routine_name = f'{program_name}_routine'
    ffunc = get_fortran(fsource, program_name, routine_name)
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    optimize_sdfg(sdfg, device, use_my_auto_opt=use_my_auto_opt)

    rng = np.random.default_rng(RNG_SEED)
    inputs = get_inputs(program, rng, testing_dataset=True)
    outputs_f = get_outputs(program, rng, testing_dataset=True)
    if pattern is not None:
        set_input_pattern(inputs, outputs_f, program, pattern)
    outputs_d_device = copy_to_device(copy.deepcopy(outputs_f))
    sdfg.validate()
    sdfg.simplify(validate_all=True)

    ffunc(**{k.lower(): v for k, v in inputs.items()}, **{k.lower(): v for k, v in outputs_f.items()})
    inputs_device = copy_to_device(inputs)
    sdfg(**inputs_device, **outputs_d_device)

    print_with_time(f"{program} ({program_name}) on {device} with"
                    f"{' ' if normalize_memlets else 'out '}normalize memlets")
    outputs_d = outputs_d_device
    passes_test = compare_output(outputs_f, outputs_d, program)
    # passes_test = compare_output_all(outputs_f, outputs_d)

    if passes_test:
        print_with_time('Success')
    else:
        print_with_time('!!!TEST NOT PASSED!!!')
    return passes_test


def run_program(program: str, use_my_auto_opt: bool, repetitions: int = 1, device=dace.DeviceType.GPU,
                normalize_memlets=False, pattern: Optional[str] = None):
    programs = get_programs_data()['programs']
    print(f"Run {program} ({programs[program]}) for {repetitions} time on device {device}")
    fsource = read_source(program)
    program_name = programs[program]
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    optimize_sdfg(sdfg, device, use_my_auto_opt=use_my_auto_opt)

    rng = np.random.default_rng(RNG_SEED)
    inputs = copy_to_device(get_inputs(program, rng))
    outputs = copy_to_device(get_outputs(program, rng))
    if pattern is not None:
        set_input_pattern(inputs, outputs, program, pattern)

    for _ in range(repetitions):
        sdfg(**inputs, **outputs)


def compile_for_profile(program: str, use_my_auto_opt: bool, device: dace.DeviceType,
                        normalize_memlets: bool) -> dace.SDFG:
    programs = get_programs_data()['programs']
    fsource = read_source(program)
    program_name = programs[program]
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    optimize_sdfg(sdfg, device, use_my_auto_opt=use_my_auto_opt)

    sdfg.instrument = dace.InstrumentationType.Timer
    sdfg.compile()
    return sdfg


def profile_program(program: str, use_my_auto_opt, device=dace.DeviceType.GPU, normalize_memlets=False,
                    repetitions=10, pattern: Optional[str] = None) -> ProgramMeasurement:

    results = ProgramMeasurement(program, get_program_parameters_data(program)['parameters'])

    programs = get_programs_data()['programs']
    print_with_time(f"Profile {program}({programs[program]}) rep={repetitions}")
    routine_name = f"{programs[program]}_routine"

    sdfg = compile_for_profile(program, use_my_auto_opt, device, normalize_memlets)

    rng = np.random.default_rng(RNG_SEED)
    inputs = get_inputs(program, rng)
    outputs = get_outputs(program, rng)
    if pattern is not None:
        set_input_pattern(inputs, outputs, program, pattern)

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


def get_roofline_data(program: str, pattern: Optional[str] = None) -> Tuple[FlopCount, Number]:
    rng = np.random.default_rng(RNG_SEED)
    params_data = get_program_parameters_data(program)
    params = params_data['parameters']
    inputs = get_inputs(program, rng)
    outputs = get_outputs(program, rng)
    if pattern is not None:
        set_input_pattern(inputs, outputs, program, pattern)
    flop_count = get_number_of_flops(params, inputs, outputs, program)
    bytes = get_number_of_bytes(params, inputs, outputs, program)
    return (flop_count, bytes)
