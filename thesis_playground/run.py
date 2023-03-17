from argparse import ArgumentParser
import os
from subprocess import run
import re
import json
import numpy as np
from typing import List, Dict
from numbers import Number

import dace

from utils import get_programs_data, get_results_dir, print_results_v2, get_program_parameters_data, print_with_time,\
                  get_inputs, get_outputs
from test import test_program, compile_for_profile
from parse_ncu import read_csv, Data
from measurement_data import ProgramMeasurement

# --- TOOD Next ---
# - plot etc.


def convert_ncu_data_into_program_measurement(ncu_data: List[Data], program_measurement: ProgramMeasurement):
    for data in ncu_data:
        match = re.match(r"[a-z_0-9]*_([0-9]*_[0-9]*_[0-9]*)\(", data.kernel_name)
        id_triplet = tuple([int(id) for id in match.group(1).split('_')])
        program_measurement.add_measurement('Kernel Time',
                                            data.durations_unit,
                                            data=data.durations,
                                            kernel_name=str(id_triplet))
        program_measurement.add_measurement('Kernel Cycles',
                                            data.cycles_unit,
                                            data=data.cycles,
                                            kernel_name=str(id_triplet))


def profile_program(program: str, device=dace.DeviceType.GPU, normalize_memlets=False, repetitions=10) \
        -> ProgramMeasurement:

    results = ProgramMeasurement(program, get_program_parameters_data(program)['parameters'])

    programs = get_programs_data()['programs']
    print_with_time(f"Profile {program}({programs[program]}) rep={repetitions}")
    routine_name = f"{programs[program]}_routine"

    sdfg = compile_for_profile(program, device, normalize_memlets)

    rng = np.random.default_rng(42)
    inputs = get_inputs(program, rng)
    outputs = get_outputs(program, rng)

    sdfg.clear_instrumentation_reports()
    print_with_time("Measure total runtime")
    for i in range(repetitions):
        sdfg(**inputs, **outputs)
    reports = sdfg.get_instrumentation_reports()

    # TOOD: Check if unit is always ms
    results.add_measurement("Total time", "ms")
    for report in reports:
        results.add_value("Total time",
                          float(report.durations[(0, -1, -1)][f"SDFG {routine_name}"][13533068620211794961][0]))

    return results


def main():
    parser = ArgumentParser(
        description='Test and profiles the given programs. Results are saved into the results folder')
    parser.add_argument('-p',
                        '--programs',
                        type=str,
                        nargs='+',
                        help='Names of the programs to use. Can be several separated by space')
    parser.add_argument('-c',
                        '--class',
                        type=int,
                        choices=[1, 2, 3],
                        default=None,
                        dest='kernel_class',
                        help="Run all programs of a given class")
    parser.add_argument('--cache',
                        default=False,
                        action='store_true',
                        help='Use the cache, does not regenerate and rebuild the code')
    parser.add_argument('-r', '--repetitions', type=int, default=5, help='Number of repetitions')
    parser.add_argument('--nsys', default=False, action='store_true', help='Also run nsys profile to generate a report')
    parser.add_argument('--no-ncu', default=False, action='store_true', help='Do not run ncu')
    parser.add_argument('-o', '--output', type=str, default=None, help='Also run nsys profile to generate a report')

    args = parser.parse_args()
    test_program_path = os.path.join(os.path.dirname(__file__), 'test.py')

    programs = get_programs_data()['programs']
    selected_programs = [] if args.programs is None else args.programs
    if args.kernel_class is not None:
        selected_programs = [p for p in programs if p.startswith(f"cloudsc_class{args.kernel_class}")]

    if len(selected_programs) == 0:
        print("ERRROR: Need to specify programs either with --programs or --class")
        return 1

    if args.cache:
        os.environ['DACE_compiler_use_cache'] = '1'
        os.putenv('DACE_compiler_use_cache', '1')

    program_results = []
    for program in selected_programs:
        print(f"Run program {program}")
        if args.cache:
            print("Build it without regenerating the code")
            parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
            build = run(['make'],
                        cwd=os.path.join(parent_dir, '.dacecache', f"{programs[program]}_routine", 'build'),
                        capture_output=True)
            if build.returncode != 0:
                print("ERROR: Error encountered while building")
                print(build.stdout.decode('UTF-8'))
                return 1

        test_program(program, dace.DeviceType.GPU, False)
        program_results.append(profile_program(program, repetitions=args.repetitions))
        command_ncu = [
            'ncu',
            '--force-overwrite',
            '--export',
            '/tmp/profile',
        ]
        command_program = ['python3', test_program_path, 'run', '--repetitions', '1', '--program', program]

        if not args.no_ncu:
            print_with_time("Measure kernel runtime")
            run([*command_ncu, *command_program], capture_output=True)
            csv_stdout = run(['ncu', '--import', '/tmp/profile.ncu-rep', '--csv'], capture_output=True)
            ncu_data = read_csv(csv_stdout.stdout.decode('UTF-8').split('\n')[:-1])
            convert_ncu_data_into_program_measurement(ncu_data, program_results[-1])

        if args.nsys:
            report_name = f"report_{program}.nsys-rep"
            command_nsys = ['nsys', 'profile', '--force-overwrite', 'true', '--output', report_name]
            print(f"Save nsys report into {report_name}")
            run([*command_nsys, *command_program], capture_output=True)

    if args.output is not None:
        with open(os.path.join(get_results_dir(), args.output), 'w') as file:
            json.dump(program_results, file, default=ProgramMeasurement.to_json)

    print_results_v2(program_results)


if __name__ == '__main__':
    main()
