from argparse import ArgumentParser
import os
from subprocess import run
import re
import json
from typing import List

import dace

from utils import get_programs_data, get_results_dir, print_results_v2, print_with_time, use_cache, \
                  get_program_parameters_data
from execute_utils import test_program, profile_program
from parse_ncu import read_csv, Data
from measurement_data import ProgramMeasurement, MeasurementRun


def convert_ncu_data_into_program_measurement(ncu_data: List[Data], program_measurement: ProgramMeasurement):
    for data in ncu_data:
        match = re.match(r"[a-z_0-9]*_([0-9]*_[0-9]*_[0-9]*)\(", data.kernel_name)
        id_triplet = tuple([int(id) for id in match.group(1).split('_')])
        time_measurement = program_measurement.get_measurement('Kernel Time', kernel=str(id_triplet))
        cycles_measurement = program_measurement.get_measurement('Kernel Cycles', kernel=str(id_triplet))
        if time_measurement is None:
            program_measurement.add_measurement('Kernel Time',
                                                data.durations_unit,
                                                data=data.durations,
                                                kernel_name=str(id_triplet))
        else:
            for value in data.durations:
                time_measurement.add_value(value)
        if cycles_measurement is None:
            program_measurement.add_measurement('Kernel Cycles',
                                                data.cycles_unit,
                                                data=data.cycles,
                                                kernel_name=str(id_triplet))
        else:
            for value in data.cycles:
                cycles_measurement.add_value(value)


def main():
    normalize_memlets = True
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
    parser.add_argument('--cache', default=False,
                        action='store_true',
                        help='Use the cache, does not regenerate and rebuild the code')
    parser.add_argument('-r', '--repetitions', type=int, default=5, help='Number of repetitions')
    parser.add_argument('--nsys', default=False, action='store_true', help='Also run nsys profile to generate a report')
    parser.add_argument('--no-ncu', default=False, action='store_true', help='Do not run ncu')
    parser.add_argument('-o', '--output', type=str, default=None, help='Also run nsys profile to generate a report')
    parser.add_argument('-d',
                        '--description',
                        type=str,
                        default='Unknown',
                        help='Description to be saved into the json file')
    parser.add_argument('--no-total', default=False, action='store_true',
                        help='Do not run measurement of total runtime')
    parser.add_argument('--ncu-repetitions', type=int, default=5,
                        help='Number of times ncu is run to measure kernel runtime')
    parser.add_argument('--skip-test', default=False, action='store_true',
                        help='Dont compare output to fortran output before profiling')
    parser.add_argument('--use-dace-auto-opt', default=False, action='store_true',
                        help='Use DaCes auto_opt instead of mine')

    args = parser.parse_args()
    test_program_path = os.path.join(os.path.dirname(__file__), 'run_program.py')

    programs = get_programs_data()['programs']
    selected_programs = [] if args.programs is None else args.programs
    if args.kernel_class is not None:
        selected_programs = [p for p in programs if p.startswith(f"cloudsc_class{args.kernel_class}")]

    if len(selected_programs) == 0:
        print("ERRROR: Need to specify programs either with --programs or --class")
        return 1

    run_data = MeasurementRun(args.description)

    run_data.properties['auto_opt'] = 'DaCe' if args.use_dace_auto_opt else 'My'

    for program in selected_programs:
        print(f"Run program {program}")
        if args.cache:
            if not use_cache(program):
                return 1

        if not args.skip_test:
            if not test_program(program, not args.use_dace_auto_opt, dace.DeviceType.GPU, normalize_memlets):
                continue
        if not args.no_total:
            program_data = profile_program(program, not args.use_dace_auto_opt, repetitions=args.repetitions,
                                           normalize_memlets=normalize_memlets)
        else:
            program_data = ProgramMeasurement(program, get_program_parameters_data(program)['parameters'])
        command_ncu = [
            'ncu',
            '--force-overwrite',
            '--export',
            '/tmp/profile',
        ]
        command_program = ['python3', test_program_path, program, '--repetitions', '1']
        if normalize_memlets:
            command_program.append('--normalize-memlets')
        if args.use_dace_auto_opt:
            command_program.append('--use-dace-auto-opt')

        if not args.no_ncu:
            print_with_time("Measure kernel runtime")
            for _ in range(args.ncu_repetitions):
                ncu_output = run([*command_ncu, *command_program], capture_output=True)
                if ncu_output.returncode != 0:
                    print("Failed to run the program with ncu")
                    print(ncu_output.stdout.decode('UTF-8'))
                    print(ncu_output.stderr.decode('UTF-8'))
                else:
                    csv_stdout = run(['ncu', '--import', '/tmp/profile.ncu-rep', '--csv'], capture_output=True)
                    if csv_stdout.returncode != 0:
                        print("Failed to read the ncu report")
                        print(csv_stdout.stdout.decode('UTF-8'))
                        print(csv_stdout.stderr.decode('UTF-8'))
                    else:
                        ncu_data = read_csv(csv_stdout.stdout.decode('UTF-8').split('\n')[:-1])
                        convert_ncu_data_into_program_measurement(ncu_data, program_data)

        if args.nsys:
            print_with_time("Create nsys report")
            report_name = f"report_{program}.nsys-rep"
            command_nsys = ['nsys', 'profile', '--force-overwrite', 'true', '--output', report_name]
            print(f"Save nsys report into {report_name}")
            run([*command_nsys, *command_program], capture_output=True)

        run_data.add_program_data(program_data)

    if args.output is not None:
        filename = os.path.join(get_results_dir(), args.output)
        print_with_time(f"Save results into {filename}")
        with open(filename, 'w') as file:
            json.dump(run_data, file, default=MeasurementRun.to_json)

    print_with_time("Print results")
    print_results_v2(run_data)


if __name__ == '__main__':
    main()
