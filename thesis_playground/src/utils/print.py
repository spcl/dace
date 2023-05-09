from datetime import datetime
from typing import Dict, Tuple, Optional, List
from numbers import Number
from tabulate import tabulate

from measurements.data import MeasurementRun
from measurements.flop_computation import FlopCount, get_number_of_bytes_2


def print_with_time(text: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")


def sort_by_program_number(flat_data: List[List]):
    """
    Sorts the given flat data 2D list (to be printed by tabulate) by the progam line number

    :param flat_data: 2D list of data where the first entry in each row is the program name
    :type flat_data: List[List]
    """
    if len(flat_data) > 1 and flat_data[0][0].startswith('cloudsc_class'):
        flat_data.sort(key=lambda row: int(row[0].split('_')[2]))


def print_results_v2(run_data: MeasurementRun):
    headers = ["program", "measurement", "avg", "median", "min", "max"]
    flat_data = []
    for program_measurement in run_data.data:
        for measurement_list in program_measurement.measurements.values():
            for measurement in measurement_list:
                name = measurement.name
                if measurement.kernel_name is not None:
                    name += f" of {measurement.kernel_name}"
                name += f" [{measurement.unit}] (#={measurement.amount()})"
                if not measurement.is_empty():
                    flat_data.append([program_measurement.program, name,
                                      measurement.average(), measurement.median(), measurement.min(),
                                      measurement.max()])

    print(run_data.description)
    print(run_data.properties)
    print(f"Node: {run_data.node} git commit: {run_data.git_hash} date: {run_data.date.strftime('%Y-%m-%d %H:%M')}")
    sort_by_program_number(flat_data)
    print(tabulate(flat_data, headers=headers))


def print_performance(roofline_data: Dict[str, Tuple[FlopCount, Number]], run_data: MeasurementRun,
                      hardware_dict: Optional[Dict] = None):

    # Avoid circular import
    from utils.general import convert_to_seconds
    if hardware_dict is not None:
        headers = ["program",
                   "P [flop/cycle]", "P [flop/sec.]",
                   "P % cycle", "P % sec.",
                   "beta [bytes/cycle]", "beta [bytes/sec.]",
                   "beta % cycle", "beta % sec.",
                   "I [flop/byte]", "W [Flop]", "Q [byte]", "f [Hz]"]
        floatfmt = (None, ".1f", ".2E", ".1f", ".1f", ".1f", ".2E", ".1f", ".1f", ".2f", None, None, ".2E")
    else:
        headers = ["program",
                   "P [flop/cycle]", "P [flop/second]",
                   "beta [bytes/cycle]", "beta [bytes/second]",
                   "I [flop/byte]", "W [Flop]", "Q [byte]", "f [Hz]"]
        floatfmt = (None, ".1f", ".2E", ".1f", ".2E", ".2f", None, None, ".2E")

    flat_data = []
    for program_measurement in run_data.data:
        program = program_measurement.program
        for measurement_cycle, measurement_time in zip(program_measurement.measurements['Kernel Cycles'],
                                                       program_measurement.measurements['Kernel Time']):
            cycles = measurement_cycle.average()
            seconds = convert_to_seconds(measurement_time.average(), measurement_time.unit)
            flop = roofline_data[program][0].get_total_flops()
            bytes = roofline_data[program][1]
            performance_cycle = flop / cycles
            performance_second = flop / seconds
            memory_cycle = bytes / cycles
            memory_second = bytes / seconds
            intensity = flop / bytes
            frequency = cycles / seconds
            if hardware_dict is not None:
                max_p_s = hardware_dict['flop_per_second']['theoretical']
                max_b_s = hardware_dict['bytes_per_second']['global']['theoretical']
                gpu_clock = hardware_dict['graphics_clock']
                memory_clock = hardware_dict['global_memory_clock']
                flat_data.append([program,
                                  performance_cycle, performance_second,
                                  performance_cycle / max_p_s * gpu_clock * 100, performance_second / max_p_s * 100,
                                  memory_cycle, memory_second,
                                  memory_cycle / max_b_s * memory_clock * 100, memory_second / max_b_s * 100,
                                  intensity, int(flop), int(bytes), frequency])
            else:
                flat_data.append([program,
                                  performance_cycle, performance_second,
                                  memory_cycle, memory_second,
                                  intensity, int(flop), int(bytes), frequency])

    sort_by_program_number(flat_data)
    print(tabulate(flat_data, headers=headers, intfmt=",", floatfmt=floatfmt))


def print_flop_counts(roofline_data: Dict[str, Tuple[FlopCount, Number]]):
    headers = ["program", "Q [byte]", "W [flop]", "I [flop/byte]", "adds", "muls", "divs", "minmax", "abs", "powers",
               "roots"]
    flat_data = []
    for program in roofline_data:
        flop_data = roofline_data[program][0]
        Q = roofline_data[program][1]
        W = flop_data.get_total_flops()
        flat_data.append([program, Q, W, W/Q,
                         flop_data.adds, flop_data.muls, flop_data.divs, flop_data.minmax, flop_data.abs,
                         flop_data.powers, flop_data.roots])

    sort_by_program_number(flat_data)
    print(tabulate(flat_data, headers=headers, intfmt=",", floatfmt=".2E"))


def print_memory_details(run_data: MeasurementRun):
    headers = ["program", "Q [byte]", "read [byte]", "written [byte]"]
    flat_data = []

    for program_measurement in run_data.data:
        program = program_measurement.program
        params = program_measurement.parameters
        total, read, written = get_number_of_bytes_2(params, program)
        flat_data.append([program, read + written, read, written])

    sort_by_program_number(flat_data)
    print(tabulate(flat_data, headers=headers, intfmt=",", floatfmt=".2E"))
