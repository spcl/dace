from datetime import datetime
from typing import Dict, Tuple
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
                print(name)
                print(measurement)
                if not measurement.is_empty():
                    flat_data.append([program_measurement.program, name,
                                      measurement.average(), measurement.median(), measurement.min(),
                                      measurement.max()])

    print(run_data.description)
    print(run_data.properties)
    print(f"Node: {run_data.node} git commit: {run_data.git_hash} date: {run_data.date.strftime('%Y-%m-%d %H:%M')}")
    print(tabulate(flat_data, headers=headers))


def print_performance(roofline_data: Dict[str, Tuple[FlopCount, Number]], run_data: MeasurementRun):
    headers = ["program", "Kernel", "performance [flops/cycle]", "memory [bytes/cycle]", "Total memory [bytes]"]
    flat_data = []
    for program_measurement in run_data.data:
        program = program_measurement.program
        for measurement in program_measurement.measurements['Kernel Cycles']:
            cycles = measurement.average()
            kernel_name = measurement.kernel_name
            performance = roofline_data[program][0].get_total_flops() / cycles
            memory = roofline_data[program][1] / cycles
            flat_data .append([program, kernel_name, performance, memory, roofline_data[program][1]])

    print(tabulate(flat_data, headers=headers, intfmt=","))
