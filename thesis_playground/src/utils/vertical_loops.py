"""
Functions specific to the vertical loop results
"""
import os
import re
import numpy as np
import json
from typing import Dict, Union, List, Tuple, Optional
from numbers import Number
import pandas as pd

from utils.paths import get_vert_loops_dir
from utils.ncu import get_achieved_bytes, get_all_actions, get_achieved_performance, get_peak_performance, \
                      action_list_to_dict, get_runtime
from measurements.flop_computation import get_number_of_bytes_2
from measurements.data import MeasurementRun

# Dictionary with a list of actions to ignore for each program
ignore_actions = {
    'microbenchmark_v1': ['single_state_body_map_0_0_8'],
    'microbenchmark_v3': ['single_state_body_map_0_0_8'],
    'cloudsc_vert_loop_mwe_wip': ['single_state_body_map_0_0_9'],
}


def parse_ncu_file(path: str) -> Dict[str, Number]:
    filename = os.path.split(path)[-1]
    program = '_'.join(filename.split('_')[0:-1])
    actions = get_all_actions(path)
    data = {}
    if len(actions) > 1:
        actions = action_list_to_dict(actions)
        actions_to_consider = []
        for name, action in actions.items():
            if re.match("[a-z_0-9]*_map", name) is not None and \
                    (program not in ignore_actions or name not in ignore_actions[program]):
                actions_to_consider.append(*action)
        if len(actions_to_consider) > 1:
            print(f"Found multiple possible actions for {program}, taking first only with name: "
                  f"{actions_to_consider[0].name()}")
        if len(actions_to_consider) == 0:
            print(f"No possible action found, actions are: {actions}")
        action = actions_to_consider[0]
    else:
        action = actions[0]

    data['measured bytes'] = get_achieved_bytes(action)
    data['achieved bandwidth'] = get_achieved_performance(action)[1]
    data['peak bandwidth'] = get_peak_performance(action)[1]
    data['runtime'] = get_runtime(action)
    data['run number'] = int(filename.split('_')[-1].split('.')[0])
    if get_achieved_bytes(action) is None:
        print(f"WARNING: measured bytes is None in {path}")
    if get_achieved_performance(action)[1] is None:
        print(f"WARNING: achieved bandwidth is None in {path}")
    if get_peak_performance(action)[1] is None:
        print(f"WARNING: peak bandwidth is None in {path}")

    return data


def parse_json_results_file(path: str) -> Dict[str, Number]:
    data = {}
    with open(path) as f:
        run_data = json.load(f, object_hook=MeasurementRun.from_json)
        for program_measurement in run_data.data:
            myQ = get_number_of_bytes_2(program_measurement.parameters, program_measurement.program)
            myQ_temp = get_number_of_bytes_2(program_measurement.parameters, program_measurement.program,
                                             temp_arrays=True)
            myQ = myQ[0] if myQ is not None else np.nan
            myQ_temp = myQ_temp[0] if myQ_temp is not None else np.nan
            data['theoretical bytes'] = myQ
            data['theoretical bytes temp'] = myQ_temp
        data['node'] = run_data.node
        data['description'] = run_data.description
    return data


def get_dataframe(version_regex: str = None) -> Tuple[pd.DataFrame]:
    """
    Read data from the vertical loops folder and put it into several pandas Dataframe.

    :param version_regex: Regex for version folders to include. Regex needs to match anywhere in foldername. If None
    takes all folders, defaults to None
    :type version_regex: str, optional
    :return: Tuple with different dataframe. First all measured data, then mapping of short_desc to description, then
    mapping (program, size, short_desc) to node
    :rtype: Tuple[pd.DataFrame]
    """
    data = []
    version_folders = os.listdir(get_vert_loops_dir())
    ignore_dirs = ['old_results', 'plots']
    if version_regex is not None:
        version_folders = [f for f in version_folders if f not in ignore_dirs and re.search(version_regex, f)]
    else:
        version_folders = [f for f in version_folders if f not in ignore_dirs]

    # for version_dir in os.listdir()
    for version_folder in version_folders:
        version = str(version_folder)
        version_folder = os.path.join(get_vert_loops_dir(), version_folder)
        for subfolder in os.listdir(version_folder):
            short_desc = str(subfolder)
            subfolder = os.path.join(version_folder, subfolder)
            ncu_data = {}
            json_data = {}
            for file in os.listdir(subfolder):
                regex_str = r'(report|result)_'+version+r'_([0-9]E\+[0-9]{1,2})_?[0-9]*\.(json|ncu-rep)'
                match = re.match(regex_str, file)
                size = match.group(2)
                filetype = match.group(3)
                if size not in ncu_data:
                    ncu_data[size] = []
                if filetype == 'ncu-rep':
                    ncu_data[size].append(parse_ncu_file(os.path.join(subfolder, file)))
                elif filetype == 'json':
                    json_data[size] = parse_json_results_file(os.path.join(subfolder, file))
            for size in ncu_data:
                for ncu_values in ncu_data[size]:
                    ncu_values.update({'size': int(float(size)), 'program': version, 'short_desc': short_desc})
                    ncu_values.update(json_data[size])
                    data.append(ncu_values)

    df = pd.DataFrame.from_dict(data)

    # Extract short_desc -> description mapping
    descriptions = df[['short_desc', 'description']].drop_duplicates()
    descriptions.set_index(['short_desc'])
    df.drop(['description'], axis='columns', inplace=True)

    # Extract (program, size, short_desc) -> node mapping
    nodes = df[['program', 'size', 'short_desc', 'node']].drop_duplicates()
    nodes.set_index(['program', 'size', 'short_desc'])
    df.drop(['node'], axis='columns', inplace=True)

    df.set_index(['program', 'size', 'run number', 'short_desc'], inplace=True, verify_integrity=True)
    return (df, descriptions, nodes)


def get_speedups(data: pd.DataFrame, baseline_program: Optional[str] = None,
                 baseline_short_desc: Optional[str] = None) -> pd.DataFrame:
    """
    Compute the "speedups" for all "list based" metrics. This means to divide the baseline by the value for all values
    in all programs. Removes any "non list" values except for program and size

    :param data: The complete data
    :type data: pd.DataFrame
    :param baseline_program: The name of the baseline program
    :type baseline_program: str
    :param baseline_short_desc: The name of the baseline short_desc. If None will use respective short_desc for each
    measurement. Defaults to None
    :type baseline_short_desc: Optional[str]
    :return: The speedup data
    :rtype: pd.DataFrame
    """
    # Groupd by program and size, remove any other non-list columns
    indices = ['program', 'size', 'short_desc']
    avg_data = data.groupby(indices).mean()
    speedup_data = avg_data.copy()

    def compute_speedup(col: pd.Series) -> pd.Series:
        if baseline_short_desc is None and baseline_program is not None:
            return col.div(col[baseline_program, :, :]).apply(np.reciprocal)
        elif baseline_short_desc is not None and baseline_program is not None:
            return col.div(col[baseline_program, :, baseline_short_desc]).apply(np.reciprocal)
        elif baseline_short_desc is not None and baseline_program is None:
            return col.div(col[:, :, baseline_short_desc]).apply(np.reciprocal)

    return speedup_data.apply(compute_speedup, axis='index')


def key_program_sort(programs):
    """
    Sort by program name/index/version
    """
    programs = programs.str.extract("cloudsc_vert_loop_([0-9])_?([0-9])?")
    programs.fillna(0, inplace=True)
    programs[0] = pd.to_numeric(programs[0])
    programs[1] = pd.to_numeric(programs[1])
    programs = programs[0].apply(lambda x: 10*x) + programs[1]
    return programs


def switch_to_zsloqa_versions(data: pd.DataFrame):
    """
    Removes and renames all ZSLOQA version to the default once. Uses versions where ZSOLQA is passed out

    :param data: The dataframe with the data
    :type data: pd.DataFrame
    """
    rename_map = {
            'cloudsc_vert_loop_4_ZSOLQA': 'cloudsc_vert_loop_4',
            'cloudsc_vert_loop_6_ZSOLQA': 'cloudsc_vert_loop_6',
            'cloudsc_vert_loop_6_1_ZSOLQA': 'cloudsc_vert_loop_6_1',
            'cloudsc_vert_loop_7_3': 'cloudsc_vert_loop_7',
        }
    for program in rename_map.values():
        data.drop(program, level='program', inplace=True)

    indices = list(data.index.names)
    data.reset_index(inplace=True)
    data['program'] = data['program'].map(lambda p: rename_map[p] if p in rename_map else p)
    data.set_index(indices, inplace=True)


def limit_to_size(data: pd.DataFrame, min_size: Optional[int] = None, max_size: Optional[int] = None):
    """
    Limits the given dataframe to exclude measurements above or below a given size/NBLOCKS value. If min or max is not
    set, will not limit in that direction

    :param data: The data
    :type data: pd.DataFrame
    :param min_size: Remove all below this value, defaults to None
    :type min_size: Optional[int], optional
    :param max_size: Remove all above this value, defaults to None
    :type max_size: Optional[int], optional
    """
    sizes = data.reset_index()['size'].unique()
    if max_size is not None:
        sizes = [s for s in sizes if s > max_size]
    if min_size is not None:
        sizes = [s for s in sizes if s < min_size]
    for size in sizes:
        data.drop(size, level='size', inplace=True)
