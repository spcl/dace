import os
import pandas as pd
from typing import List

from utils.paths import get_results_2_folder


def get_data_longformat(experiment_ids: List[int]) -> pd.DataFrame:
    """
    Get all results data from the given experiment ids. DataFrame will be in long format with the following columns:
        - "program" (part of index)
        - "experiment id" (part of index)
        - "run number" (part of index)
        - "measurement" (part of index)
        - "value"

    :param experiment_ids: List of experiment ids to get
    :type experiment_ids: List[int]
    :return: DataFrame with data in long format
    :rtype: pd.DataFrame
    """
    index_cols = set(['program', 'experiment id', 'run number', 'measurement'])
    df = pd.DataFrame()
    for experiment_id in experiment_ids:
        for program_folder in os.listdir(get_results_2_folder()):
            program_folder_path = os.path.join(get_results_2_folder(), program_folder)
            if os.path.isdir(program_folder_path) and str(experiment_id) in list(os.listdir(program_folder_path)):
                data_file = os.path.join(get_results_2_folder(), program_folder, experiment_id, 'results.csv')
                print(f"Read data from {data_file}")
                experiment_df = pd.read_csv(data_file)

                # Add any columns which are not value to the index
                columns = list(experiment_df.columns)
                if len(columns) > 1:
                    columns.remove('value')
                    for col in columns:
                        index_cols.add(col)

                experiment_df['experiment id'] = int(experiment_id)
                experiment_df['program'] = program_folder
                df = pd.concat([experiment_df.set_index(list(index_cols)), df])
    return df


def get_data_wideformat(experiment_ids: List[int]) -> pd.DataFrame:
    long_df = get_data_longformat(experiment_ids)
    wide_df = long_df.unstack(level='measurement')
    # Remove 'value' index-level from columns
    wide_df.columns = wide_df.columns.droplevel()
    return wide_df
