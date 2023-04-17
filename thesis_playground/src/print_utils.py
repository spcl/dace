from datetime import datetime
from typing import Dict, Tuple, Optional
from numbers import Number
from tabulate import tabulate

from measurement_data import MeasurementRun
from flop_computation import FlopCount


def print_with_time(text: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")


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
    print(tabulate(flat_data, headers=headers))


def print_performance(roofline_data: Dict[str, Tuple[FlopCount, Number]], run_data: MeasurementRun,
                      hardware_dict: Optional[Dict] = None):

    # Avoid circular import
    from utils import convert_to_seconds
    if hardware_dict is not None:
        headers = ["program", "Kernel",
                   "P [flop/cycle]", "P [flop/sec.]",
                   "P % cycle", "P % sec.",
                   "beta [bytes/cycle]", "beta [bytes/sec.]",
                   "beta % cycle", "beta % sec.",
                   "I [flop/byte]", "W [Flop]", "Q [byte]", "f [Hz]"]
    else:
        headers = ["program", "Kernel",
                   "P [flop/cycle]", "P [flop/second]",
                   "beta [bytes/cycle]", "beta [bytes/second]",
                   "I [flop/byte]", "W [Flop]", "Q [byte]", "f [Hz]"]

    flat_data = []
    for program_measurement in run_data.data:
        program = program_measurement.program
        for measurement_cycle, measurement_time in zip(program_measurement.measurements['Kernel Cycles'],
                                                       program_measurement.measurements['Kernel Time']):
            cycles = measurement_cycle.average()
            seconds = convert_to_seconds(measurement_time.average(), measurement_time.unit)
            kernel_name = measurement_cycle.kernel_name
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
                flat_data.append([program, kernel_name,
                                  performance_cycle, performance_second,
                                  performance_cycle / max_p_s * gpu_clock * 100, performance_second / max_p_s * 100,
                                  memory_cycle, memory_second,
                                  memory_cycle / max_b_s * memory_clock * 100, memory_second / max_b_s * 100,
                                  intensity, flop, bytes, frequency])
            else:
                flat_data.append([program, kernel_name,
                                  performance_cycle, performance_second,
                                  memory_cycle, memory_second,
                                  intensity, flop, bytes, frequency])

    print(tabulate(flat_data, headers=headers, intfmt=",", floatfmt=".2E"))
