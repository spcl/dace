from datetime import datetime
from typing import Dict, Tuple, Optional, List
from tabulate import tabulate
import numpy as np
import pandas as pd


def print_with_time(text: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")


def sort_by_program_number(flat_data: List[List]):
    """
    Sorts the given flat data 2D list (to be printed by tabulate) by the progam line number

    :param flat_data: 2D list of data where the first entry in each row is the program name
    :type flat_data: List[List]
    """
    if len(flat_data) > 1 and flat_data[0][0].startswith('cloudsc_class'):
        flat_data.sort(key=lambda row: int(row[0].split('_')[2]))


def print_dataframe(columns: Dict[str, Tuple[str, Optional[str]]], df: pd.DataFrame, tablefmt: str = 'plain'):
    """
    Prints the given dataframe

    :param columns: Dictionary with all columns to print. Key is the name of the dataframe column. Tuple contains
    header/name to print followed by the format
    :type columns: Dict[str, Tuple[str, Optional[str]]]
    :param df: The dataframe to print
    :type df: pd.DataFrame
    :param tablefmt: String which specifies how the table should be formatted, defaults to 'plain'
    :type tablefmt: str, optional
    """
    df_columns = []
    headers = []
    floatfmt = []
    intfmt = []
    for c in columns:
        df_columns.append(c)
        headers.append(columns[c][0])
        if df[c].dtype == np.float64:
            floatfmt.append(columns[c][1])
            intfmt.append('')
        elif df[c].dtype == np.int64:
            floatfmt.append('')
            intfmt.append(columns[c][1])
        else:
            floatfmt.append('')
            intfmt.append('')

    print(tabulate(df[df_columns], headers=headers, floatfmt=floatfmt, intfmt=intfmt, showindex=False,
                   tablefmt=tablefmt))
