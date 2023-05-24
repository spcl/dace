import os
from typing import Dict, List, Tuple
import re
import json
from numbers import Number
import pandas as pd

from utils.ncu_report import IAction
from utils.ncu import get_achieved_performance, get_achieved_bytes, get_peak_performance, get_achieved_work
from utils.general import convert_to_seconds
from utils.ncu import get_all_actions, action_list_to_dict
from utils.paths import get_complete_results_dir
from measurements.data import ProgramMeasurement, MeasurementRun
from measurements.flop_computation import FlopCount, read_roofline_data, get_number_of_bytes_2


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


def get_dataframe(names: List[str]) -> Tuple[pd.DataFrame]:
    """
    Returns the data stored in 'complete_results' in dataframes

    :param names: List of names/subfolders in 'complete_results'
    :type names: List[str]
    :return: The data as dataframes in a tuple. First dataframe is the measured data, second is the mapping program ->
    class number, third the parameters for each run
    :rtype: Tuple[pd.DataFrame]
    """
    result_files = []
    for name in names:
        results_folder = os.path.join(get_complete_results_dir(), name, 'raw_data')
        result_files.extend([os.path.join(results_folder, f) for f in os.listdir(results_folder)
                             if re.match(r"class_[1-3].json", f)])
    data = []
    parameter_data = []
    for result_file in result_files:
        with open(result_file) as file:
            run_data = json.load(file, object_hook=MeasurementRun.from_json)
            for program_measurement in run_data.data:
                for measurement_list in program_measurement.measurements.values():
                    for measurement in measurement_list:
                        for index, value in enumerate(measurement.data):
                            unit = measurement.unit
                            if measurement.name in ['Kernel Time', 'Total time', 'Total Time']:
                                value = convert_to_seconds(value, unit)
                                unit = 'seconds'

                            measurement_dict = {
                                'run description': run_data.description,
                                'experiment name': os.path.split(os.path.split(os.path.dirname(result_file))[0])[1],
                                'program': program_measurement.program,
                                'unit': unit,
                                'measurement name': measurement.name,
                                'kernel name': measurement.kernel_name,
                                'value': value,
                                'run number': index,
                                'node': run_data.node,
                                }
                            measurement_dict.update(run_data.properties)
                            data.append(measurement_dict)

                parameter_dict = program_measurement.parameters.get_dict()
                parameter_dict.update({
                    'program': program_measurement.program,
                    'run description': run_data.description,
                    'experiment name': os.path.split(os.path.split(os.path.dirname(result_file))[0])[1]
                    })
                parameter_data.append(parameter_dict)
    df = pd.DataFrame.from_dict(data)
    programs = df['program'].unique()
    df.set_index(['program', 'run description', 'experiment name', 'measurement name', 'run number', 'kernel name'],
                 inplace=True)

    classes = []
    for program in programs:
        match = re.match(r'cloudsc_class([0-9])_[0-9]{3,4}', program)
        class_number = int(match.group(1))
        classes.append([program, class_number])
    classes = pd.DataFrame(classes, columns=['program', 'class']).set_index('program')

    parameters_df = pd.DataFrame.from_dict(parameter_data) \
        .set_index(['program', 'run description', 'experiment name'])

    return df, classes, parameters_df


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


def get_roofline_dataframe(names: List[str]) -> pd.DataFrame:
    """
    Gets the roofline relevant data (peak/achieved performance/bytes/work etc). for the given names/subfolders of
    "complete_results"

    :param names: List of names/subfolders in 'complete_results'
    :type names: List[str]
    :return: Dataframe with all data
    :rtype: pd.DataFrame
    """
    wide_data = []
    for name in names:
        dict_data = {}
        for program, (flop_count, bytes) in get_roofline_data(name).items():
            dict_data[program] = {}
            dict_data[program] = {
                'flop': flop_count.get_total_flops(),
                'theoretical bytes': bytes
                }
        for program, action in get_actions(name).items():
            dict_data[program].update({
                'measured bytes': get_achieved_bytes(action),
                'achieved bandwidth': get_achieved_performance(action)[1],
                'peak bandwidth': get_peak_performance(action)[1],
                'achieved double flops': get_achieved_performance(action)[0],
                'peak double flops': get_peak_performance(action)[0]
                })
            dict_data[program].update(get_achieved_work(action))

        for program in dict_data:
            wide_data.append({
                'experiment name': name,
                'program': program
            })
            wide_data[-1].update(dict_data[program])

    return pd.DataFrame.from_dict(wide_data)
