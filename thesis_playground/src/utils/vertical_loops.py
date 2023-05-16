import os
import re
import numpy as np
import json
from typing import Dict, Union, List
from numbers import Number
import pandas as pd

from utils.paths import get_vert_loops_dir
from utils.ncu import get_achieved_bytes, get_all_actions, get_achieved_performance, get_peak_performance, \
                      action_list_to_dict, get_runtime
from measurements.flop_computation import get_number_of_bytes_2
from measurements.data import MeasurementRun


def get_data(filename_regex: str = None) -> Dict[str, Dict[str, Union[Number, np.ndarray]]]:
    """
    Read data from the vertical loops folder and put it into a dictionary

    :param filename_regex: Regex for files to include. Regex needs to match anywhere in filename. If None takes all
    files in filder, defaults to None
    :type filename_regex: str, optional
    :return: Dictionary with data, first key is label of the program/version, 2nd is the measurement name
    :rtype: Dict[Dict[str, Union[Number, np.ndarray]]]
    """
    data = {}
    folder = get_vert_loops_dir()
    files = os.listdir(folder)
    if filename_regex is not None:
        files = [f for f in files if re.search(filename_regex, f)]

    for file in files:
        if file.split('.')[-1] == 'ncu-rep':
            label = '_'.join(file.split('.')[0].split('_')[1:-1])
            actions = get_all_actions(os.path.join(folder, file))
            if len(actions) > 1:
                actions = action_list_to_dict(actions)
                actions_to_consider = []
                for name, action in actions.items():
                    if re.match("[a-z_]*_map", name) is not None:
                        actions_to_consider.append(*action)
                if len(actions_to_consider) > 1:
                    print(f"Found multiple possible actions, taking first only with name: "
                          f"{actions_to_consider[0].name()}")
                if len(actions_to_consider) == 0:
                    print(f"No possible action found, actions are: {actions}")
                action = actions_to_consider[0]
            else:
                action = actions[0]

            if label not in data:
                data[label] = {}
            if 'measured bytes' not in data[label]:
                data[label]['measured bytes'] = np.ndarray((0), dtype=np.int32)
                data[label]['achieved bandwidth'] = np.ndarray((0), dtype=np.int32)
                data[label]['peak bandwidth'] = np.ndarray((0), dtype=np.int32)
                data[label]['runtime'] = np.ndarray((0), dtype=np.float64)
                data[label]['run number'] = np.ndarray((0), dtype=np.int32)
            data[label]['measured bytes'] = np.append(data[label]['measured bytes'], get_achieved_bytes(action))
            data[label]['achieved bandwidth'] = np.append(data[label]['achieved bandwidth'],
                                                          get_achieved_performance(action)[1])
            data[label]['peak bandwidth'] = np.append(data[label]['peak bandwidth'], get_peak_performance(action)[1])
            data[label]['runtime'] = np.append(data[label]['runtime'], get_runtime(action))
            data[label]['run number'] = np.append(data[label]['run number'], int(file.split('_')[-1].split('.')[0]))
            if get_achieved_bytes(action) is None:
                print(f"WARNING: measured bytes is None in {file}")
            if get_achieved_performance(action)[1] is None:
                print(f"WARNING: achieved bandwidth is None in {file}")
            if get_peak_performance(action)[1] is None:
                print(f"WARNING: peak bandwidth is None in {file}")
        elif file.split('.')[-1] == 'json':
            label = '_'.join(file.split('.')[0].split('_')[1:])
            with open(os.path.join(folder, file)) as f:
                run_data = json.load(f, object_hook=MeasurementRun.from_json)
                for program_measurement in run_data.data:
                    myQ = get_number_of_bytes_2(program_measurement.parameters, program_measurement.program)[0]
                    myQ_temp = get_number_of_bytes_2(program_measurement.parameters, program_measurement.program,
                                                     temp_arrays=True)[0]
                    if label not in data:
                        data[label] = {}
                    data[label]['theoretical bytes'] = myQ
                    data[label]['theoretical bytes temp'] = myQ_temp
                    data[label]['node'] = run_data.node
    return data


def get_non_list_indices() -> List[str]:
    """
    Returns list of indices/keys for which the data by get_data does not feature a list. Meaning the dataframe from
    get_dataframe can be grouped by.

    :return: List of indices
    :rtype: List[str]
    """
    return ['program', 'size', 'theoretical bytes', 'theoretical bytes temp', 'node']


def get_dataframe(filename_regex: str = None) -> pd.DataFrame:
    """
    Read data from the vertical loops folder and put it into a pandas Dataframe.

    :param filename_regex: Regex for files to include. Regex needs to match anywhere in filename. If None takes all
    files in filder, defaults to None
    :type filename_regex: str, optional
    :return: Data
    :rtype: pd.DataFrame
    """
    data = get_data(filename_regex)
    data_list = []
    for label, entry in data.items():
        program = ' '.join(label.split('_')[:-1])
        size = label.split('_')[-1]
        entry.update({'program': program, 'size': size})
        data_list.append(entry)
    df = pd.DataFrame.from_dict(data_list)
    # explode/unpack the list entries stemming from the different measurement runs
    columns = [c for c in df.keys() if c not in get_non_list_indices()]
    df = df.explode(columns, ignore_index=True)
    df[columns] = df[columns].apply(pd.to_numeric)
    df.set_index(['program', 'size', 'run number'], inplace=True, verify_integrity=True)
    return df
