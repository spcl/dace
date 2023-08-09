"""
Utils functions for plotting which read data directly from ncu objects
"""

from typing import Dict, Tuple, Optional
from numbers import Number
import matplotlib

from measurements.data import MeasurementRun
from measurements.flop_computation import FlopCount
from utils.ncu import get_achieved_bytes, get_achieved_work, get_runtime
from utils.ncu_report import IAction
from utils.plot import draw_program_points, update_min_max, get_color_from_program_name, draw_rooflines_seconds


def draw_ncu_points(action: IAction, label: str, ax: matplotlib.axis.Axis, program: Optional[str] = None,
                    marker: str = "x"):
    """
    Draws points onto the given axis. Uses the Q, W and T data from the given ncu_report action object

    :param action: The ncu action object
    :type action: Action
    :param label: The label of the measurement
    :type label: str
    :param ax: The axis
    :type ax: matplotlib.axis.Axis
    :param program: Name of the program of the measurement. If given will color all points of the same program in the
    same color
    :type program: Optional[str]
    :param marker: Marker for the points as passed to matplotlib, defaults to "x"
    :type marker: str
    """
    bytes_count = get_achieved_bytes(action)
    flop = get_achieved_work(action)['dW']
    time = get_runtime(action)
    performance = flop / time
    intensity = flop / bytes_count
    if program is None:
        ax.scatter(intensity, performance, label=label, marker=marker)
    else:
        ax.scatter(intensity, performance, label=label, marker=marker, color=get_color_from_program_name(program))
    return intensity


def plot_roofline_seconds(run_data: MeasurementRun, roofline_data: Dict[str, Tuple[FlopCount, Number]],
                          hardware_data: Dict, ax: matplotlib.axis.Axis, points_only: bool = False,
                          actions: Optional[Dict[str, IAction]] = None):

    ax.set_xlabel("Operational Intensity")
    ax.set_ylabel("Performance [flop/seconds]")

    min_intensity = None
    max_intensity = None
    for program_measurement in run_data.data:
        program = program_measurement.program
        label = f"{program} using {run_data.description}"
        for measurement in program_measurement.measurements['Kernel Time']:
            min_intensity, max_intensity = update_min_max(
                    draw_program_points(measurement, label, ax, *roofline_data[program]), min_intensity,
                    max_intensity)
            if actions is not None:
                min_intensity, max_intensity = update_min_max(draw_ncu_points(actions[program], label, ax),
                                                              min_intensity, max_intensity)

    if not points_only:
        draw_rooflines_seconds()

    ax.legend()
    ax.grid()


