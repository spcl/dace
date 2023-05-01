from argparse import ArgumentParser
import json
import os
from typing import Tuple, List, Union, Optional
from numbers import Number
from tabulate import tabulate

from utils.general import get_results_dir
from measurement_data import MeasurementRun, ProgramMeasurement, Measurement


def find_program_measurement_in_run(run: MeasurementRun, program_name: str) -> Optional[ProgramMeasurement]:
    """
    Finds the ProgramMeasurement given by its program name in the given MeasurementRun

    :param run: The MeasurementRun to search through
    :type run: MeasurementRun
    :param program_name: The name of the program we are looking for
    :type program_name: str
    :return: The ProgramMeasurement or None if it could not be found
    :rtype: Optional[ProgramMeasurement]
    """
    for program in run.data:
        if program.program == program_name:
            return program
    return None


def get_measurement_in_program_measurement(
        program_measurement: ProgramMeasurement,
        measurement_name: str,
        kernel_name: Optional[str] = None) -> Optional[Measurement]:
    """
    Get the measurement given the measurement name and optionally the kernel from a ProgramMeasurement

    :param program_measurement: The ProgramMeasurement to search in
    :type program_measurement: ProgramMeasurement
    :param measurement_name: The name of the measurement
    :type measurement_name: str
    :param kernel_name: The optional kernel name, defaults to None
    :type kernel_name: Optional[str], optional
    :return: The found Measurement or None if not found
    :rtype: Optional[Measurement]
    """

    measurement = None
    if kernel_name is None:
        if measurement_name not in program_measurement.measurements:
            print(f"ERROR: Could not find {measurement_name} in {program_measurement}")
        elif len(program_measurement.measurements[measurement_name]) > 2:
            measurement = None
            print(f"ERROR: There is more than one measurement for {measurement_name} while"
                  f"ignoring kernels for {program_measurement}")
        else:
            measurement = program_measurement.measurements[measurement_name][0]
    else:
        measurement = program_measurement.get_measurement(measurement_name,
                                                          kernel=kernel_name)
    return measurement


def compute_speedup(
        baseline: MeasurementRun,
        data: MeasurementRun,
        ignore_kernel_name: bool = False) -> Tuple[List[str], List[List[Union[str, Number]]]]:
    """
    Computes the speedup of the given MeasurementRun compared to a given baseline using the average

    :param baseline: The baseline MeasurementRun
    :type baseline: MeasurementRun
    :param data: The other MeasurementRun
    :type data: MeasurementRun
    :param ignore_kernel_name: If set to True, ignores kernel names, requires that there is only one measurement per
    measurement name
    :type ignore_kernel_name: bool
    :return: Two lists, one with headers, other with the data to print for the speedup
    :rtype: Tuple[List[str], List[List[Union[str, Number]]]]
    """
    header = ["program", "measurement", "speedup (average)"]
    speedup_data = []
    for program in data.data:
        baseline_program = find_program_measurement_in_run(baseline, program.program)
        if baseline_program is None:
            print(f"WARNING: Could not find a baseline program for {program.program}")
        else:
            for measurement_name, measurement_list in program.measurements.items():
                if measurement_name in baseline_program.measurements:
                    for measurement in measurement_list:
                        baseline_measurement = get_measurement_in_program_measurement(
                                baseline_program, measurement_name,
                                kernel_name=None if ignore_kernel_name else measurement.kernel_name)
                        if baseline_measurement is None:
                            print(f"WARNING: could not find a baseline measurement for {measurement} for program"
                                  f"{program.program}")
                        else:
                            name = measurement_name
                            if measurement.kernel_name is not None and not ignore_kernel_name:
                                name += f" of {measurement.kernel_name}"
                            name += f" [{measurement.unit}]"
                            speedup = baseline_measurement.average() / measurement.average()
                            speedup_data.append([program.program, name, speedup])
    return (header, speedup_data)


def main():
    parser = ArgumentParser()
    parser.add_argument('--ignore-kernel-names', action='store_true', default=False,
                        help="Ignore kernel names, requires that there is only one kernel per measurement")
    parser.add_argument('--baseline', type=str, required=True, help="Json file with the baseline data")
    parser.add_argument('--format', type=str, default="plain", help="The format for tabulate to use")
    parser.add_argument('files', type=str, nargs='+', help="Json file(s) with the (faster) runtime")

    args = parser.parse_args()
    with open(os.path.join(get_results_dir(), args.baseline)) as baseline_file:
        baseline_data = json.load(baseline_file, object_hook=MeasurementRun.from_json)
        for file in args.files:
            with open(os.path.join(get_results_dir(), file)) as f:
                data = json.load(f, object_hook=MeasurementRun.from_json)
                header, speedups = compute_speedup(baseline_data, data, args.ignore_kernel_names)
                print(tabulate(speedups, headers=header, tablefmt=args.format))


if __name__ == '__main__':
    main()
