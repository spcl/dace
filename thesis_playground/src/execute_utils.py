import numpy as np
import copy
import cupy as cp

import dace

# from dace.transformation.auto.auto_optimize import auto_optimize
from my_auto_opt import auto_optimize

from data import get_program_parameters_data
from utils import get_programs_data, read_source, get_fortran, get_sdfg, get_inputs, get_outputs, print_with_time, \
                  compare_output, compare_output_all, copy_to_device
from measurement_data import ProgramMeasurement

RNG_SEED = 42


# Copied and adapted from tests/fortran/cloudsc.py
def test_program(program: str, device: dace.DeviceType, normalize_memlets: bool) -> bool:
    """
    Tests the given program by comparing the output of the SDFG compiled version to the one compiled directly from
    fortran

    :param program: The program name
    :type program: str
    :param device: The deive
    :type device: dace.DeviceType
    :param normalize_memlets: If memlets should be normalized
    :type normalize_memlets: bool
    :return: True if test passes, False otherwise
    :rtype: bool
    """

    programs_data = get_programs_data()
    fsource = read_source(program)
    program_name = programs_data['programs'][program]
    routine_name = f'{program_name}_routine'
    ffunc = get_fortran(fsource, program_name, routine_name)
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    if device == dace.DeviceType.GPU:
        auto_optimize(sdfg, device)

    rng = np.random.default_rng(RNG_SEED)
    inputs = get_inputs(program, rng, testing_dataset=True)
    outputs_f = get_outputs(program, rng, testing_dataset=True)
    outputs_d = copy.deepcopy(outputs_f)
    sdfg.validate()
    sdfg.simplify(validate_all=True)

    ffunc(**{k.lower(): v for k, v in inputs.items()}, **{k.lower(): v for k, v in outputs_f.items()})
    sdfg(**inputs, **outputs_d)

    print_with_time(f"{program} ({program_name}) on {device} with"
                    f"{' ' if normalize_memlets else 'out '}normalize memlets")
    passes_test = compare_output(outputs_f, outputs_d, program)
    # passes_test = compare_output_all(outputs_f, outputs_d)

    if passes_test:
        print_with_time('Success')
    else:
        print_with_time('!!!TEST NOT PASSED!!!')
    return passes_test


def run_program(program: str, repetitions: int = 1, device=dace.DeviceType.GPU, normalize_memlets=False):
    programs = get_programs_data()['programs']
    print(f"Run {program} ({programs[program]}) for {repetitions} time on device {device}")
    fsource = read_source(program)
    program_name = programs[program]
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    auto_optimize(sdfg, device)

    rng = np.random.default_rng(RNG_SEED)
    inputs = get_inputs(program, rng)
    outputs = get_outputs(program, rng)

    for _ in range(repetitions):
        sdfg(**inputs, **outputs)


def compile_for_profile(program: str, device: dace.DeviceType, normalize_memlets: bool) -> dace.SDFG:
    programs = get_programs_data()['programs']
    fsource = read_source(program)
    program_name = programs[program]
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    auto_optimize(sdfg, device)

    for k, v in sdfg.arrays.items():
        if not v.transient and type(v) == dace.data.Array:
            v.storage = dace.dtypes.StorageType.GPU_Global

    sdfg.instrument = dace.InstrumentationType.Timer
    sdfg.compile()
    return sdfg


def profile_program(program: str, device=dace.DeviceType.GPU, normalize_memlets=False, repetitions=10) \
        -> ProgramMeasurement:

    results = ProgramMeasurement(program, get_program_parameters_data(program)['parameters'])

    programs = get_programs_data()['programs']
    print_with_time(f"Profile {program}({programs[program]}) rep={repetitions}")
    routine_name = f"{programs[program]}_routine"

    sdfg = compile_for_profile(program, device, normalize_memlets)

    rng = np.random.default_rng(RNG_SEED)
    inputs = get_inputs(program, rng)
    outputs = get_outputs(program, rng)

    sdfg.clear_instrumentation_reports()
    print_with_time("Measure total runtime")
    inputs = copy_to_device(inputs)
    outputs = copy_to_device(outputs)
    for i in range(repetitions):
        sdfg(**inputs, **outputs)

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
