"""
Module with methods concered with metadata for the results_v2 folders
"""
import os
import pandas as pd

from utils.paths import get_experiments_2_file, get_results_2_folder


def get_experiment_list_df() -> pd.DataFrame():
    """
    Returns dataframe with information about the different experiments

    :return: DataFrame with experiments data
    :rtype: pd.DataFrame()
    """
    if not os.path.exists(get_experiments_2_file()):
        os.makedirs(os.path.dirname(get_experiments_2_file()), exist_ok=True)
        return pd.DataFrame()
    else:
        return pd.read_csv(get_experiments_2_file(), index_col=['experiment id'])


def get_program_infos() -> pd.DataFrame():
    program_infos = pd.read_csv(os.path.join(get_results_2_folder(), 'program_infos.csv'), index_col=['program'])
    base_program_infos = pd.read_csv(os.path.join(get_results_2_folder(), 'base_program_infos.csv'),
                                     index_col=['base id'])
    df = program_infos.join(base_program_infos, on='base id')
    df['full description'] = df['base description'] + df['variant description']
    return df
