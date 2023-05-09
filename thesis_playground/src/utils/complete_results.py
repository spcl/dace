import os
from typing import Dict, List, Tuple
import re
import json
from numbers import Number

from utils.ncu_report import IAction
from utils.ncu import get_all_actions, action_list_to_dict
from utils.paths import get_complete_results_dir
from measurements.data import ProgramMeasurement, MeasurementRun
from measurements.flop_computation import FlopCount, read_roofline_data


def get_actions(name: str) -> Dict[str, IAction]:
    """
    Get dictionary of all action objects, one for each program, stored in the given subfolder of "complete_results"

    :param name: Name of the subfolder of "complete_results"
    :type name: str
    :return: Dictionary, key is the program name, value the action
    :rtype: Dict[str, IAction]
    """
    ncu_folder = os.path.join(get_complete_results_dir(), name, 'raw_data')
    ncu_files = [f for f in os.listdir(ncu_folder)
                 if re.match(r"report_class_[1-3]_cloudsc_class[1-3]_[0-9]{3,4}.ncu-rep", f)]
    action_dict = {}
    for ncu_file in ncu_files:
        ncu_file = os.path.join(ncu_folder, ncu_file)
        program = re.match("[a-z_0-9]*(cloudsc_class[1-3]_[0-9]{3,4})", os.path.basename(ncu_file)).group(1)
        actions = get_all_actions(ncu_file)
        if len(actions) > 1:
            actions = action_list_to_dict(actions)
            actions_to_consider = []
            for name, action in actions.items():
                if re.match("[a-z_]*_map", name) is not None:
                    actions_to_consider.append(*action)
            if len(actions_to_consider) > 1:
                print(f"Found multiple possible actions, taking first only with name: {actions_to_consider[0].name()}")
            if len(actions_to_consider) == 0:
                print(f"No possible action found, actions are: {actions}")
            action = actions_to_consider[0]
        else:
            action = actions[0]

        action_dict[program] = action

    return action_dict


def get_program_measurements(name: str) -> List[ProgramMeasurement]:
    """
    Get list of all ProgramMeasurement, one for each program, stored in the given subfolder of "complete_results"

    :param name: Name of the subfolder of "complete_results"
    :type name: str
    :return: List of all ProgramMeasurements
    :rtype: List[ProgramMeasurement]
    """
    results_folder = os.path.join(get_complete_results_dir(), name, 'raw_data')
    result_files = [f for f in os.listdir(results_folder)
                    if re.match(r"class_[1-3].json", f)]

    measurements = []
    for result_file in result_files:
        result_file = os.path.join(results_folder, result_file)
        with open(result_file) as file:
            run_data = json.load(file, object_hook=MeasurementRun.from_json)
            for program_measurement in run_data.data:
                measurements.append(program_measurement)

    return measurements


def get_roofline_data(name: str) -> Dict[str, Tuple[FlopCount, Number]]:
    """
    Get dictionary of all roofline data, one for each program, stored in the given subfolder of "complete_results"

    :param name: Name of the subfolder of "complete_results"
    :type name: str
    :return: Dictionary, keys are program names, values are tuples of FlopCount and byte count
    :rtype: Dict[str, Tuple[FlopCount, Number]]
    """
    roofline_data = {}
    results_folder = os.path.join(get_complete_results_dir(), name, 'raw_data')
    result_files = [f for f in os.listdir(results_folder)
                    if re.match(r"class_[1-3]_roofline.json", f)]

    for result_file in result_files:
        result_file = os.path.join(results_folder, result_file)
        roofline_data.update(read_roofline_data(result_file))

    return roofline_data
