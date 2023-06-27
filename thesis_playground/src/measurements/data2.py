import os
import pandas as pd
from typing import List

from utils.paths import get_results_2_folder
from utils.experiments2 import get_experiment_list_df


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
                data_file = os.path.join(get_results_2_folder(), program_folder, str(experiment_id), 'results.csv')
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


def average_data(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Averages the data over multiple runs. This removes the run number from the index

    :param wide_df: The data in wideformat
    :type wide_df: pd.DataFrame
    :return: The Averaged data
    :rtype: pd.DataFrame
    """
    index_cols = list(wide_df.index.names)
    index_cols.remove('run number')
    return wide_df.groupby(index_cols).mean()


def get_full_averaged_data_wideformat(experiments_ids: List[int]) -> pd.DataFrame:
    """
    Reads the data from the given experiment ids and adds the additional data stored in the experiments file and returns
    it in wide format

    :param experiments_ids: List of experiment ids to get
    :type experiments_ids: List[int]
    :return: The data in wideformat
    :rtype: pd.DataFrame
    """
    df = average_data(get_data_wideformat(experiments_ids).dropna())
    df = df.join(get_experiment_list_df(), on='experiment id')
    return df
