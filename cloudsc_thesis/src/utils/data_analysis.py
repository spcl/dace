from typing import Tuple, Union
from numbers import Number
import numpy as np
import pandas as pd


def compute_speedups(data: pd.DataFrame, col_keys: Tuple[Union[Number, str]], col_levels: Tuple[str]) -> pd.DataFrame:
    """
    Compute the "speedups" for all "list based" metrics. This means to divide the baseline by the value for all values
    in all programs. Removes any "non list" values except for program and size

    :param data: The complete averaged data
    :type data: pd.DataFrame
    :param col_keys: Tuple of the key-values to fix as a baseline for computing the speedup
    :type col_keys: Tuple[Union[Number, str]]
    :param col_levels: Tuple of the levels used in col_keys
    :type col_levels: Tuple[str]
    :return: The speedup data
    :rtype: pd.DataFrame
    """
    def compute_speedup(col: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(col):
            new_col = col.div(col.xs(col_keys, level=col_levels, drop_level=True)).apply(np.reciprocal)
        else:
            new_col = col
        return new_col.reorder_levels(col.index.names)

    return data.apply(compute_speedup, axis='index')


def compute_speedups_min_max(data: pd.DataFrame, col_keys: Tuple[Union[Number, str]], col_levels: Tuple[str],
                             run_number: str) -> pd.DataFrame:
    def compute_speedup(col: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(col):
            new_col = col.div(col.xs(col_keys, level=col_levels, drop_level=True)).apply(np.reciprocal)
        else:
            new_col = col
        return new_col.reorder_levels(col.index.names)

    index_cols = list(data.index.names)
    index_cols.remove(run_number)
    min_data = data.reset_index().groupby(index_cols).min()
    max_data = data.reset_index().groupby(index_cols).max()
    max_speedups = pd.concat([min_data.drop(col_keys, level=col_levels), max_data.xs(col_keys, level=col_levels,
                                                                                     drop_level=False)]).apply(compute_speedup, axis='index')
    min_speedups = pd.concat([max_data.drop(col_keys, level=col_levels), min_data.xs(col_keys, level=col_levels,
                                                                                     drop_level=False)]).apply(compute_speedup, axis='index')
    return (min_speedups, max_speedups)

