"""
General helper methods for GPU, requires cupy
"""
from typing import Dict, Union
from numbers import Number
import numpy as np
import cupy as cp


def copy_to_device(host_data: Dict[str, Union[Number, np.ndarray]]) -> Dict[str, cp.ndarray]:
    """
    Take a dictionary with data and converts all numpy arrays to cupy arrays

    :param host_data: The numpy (host) data
    :type host_data: Dict[str, Union[Number, np.ndarray]]
    :return: The converted cupy (device) data
    :rtype: Dict[str, cp.ndarray]
    """
    device_data = {}
    for name, array in host_data.items():
        if isinstance(array, np.ndarray):
            device_data[name] = cp.asarray(array)
        else:
            device_data[name] = array
    return device_data


def copy_to_host(device_data: Dict[str, Union[Number, cp.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Take a dictionary with data and converts all cupy arrays to numpy arrays

    :param host_data: The cupy (device) data
    :type host_data: Dict[str, Union[Number, cp.ndarray]]
    :return: The converted cupy (host) data
    :rtype: Dict[str, np.ndarray]
    """
    host_data = {}
    for name, array in device_data.items():
        if isinstance(array, cp.ndarray):
            host_data[name] = cp.asnumpy(array)
        else:
            host_data[name] = array
    return host_data


def print_non_zero_percentage(data: Dict[str, Union[Number, Union[np.ndarray, cp.ndarray]]], variable: str):
    """
    Prints the number of total and nonzero values and the percentage of a given variable. Data can have numpy or cupy
    arrays

    :param data: The data where the variable is inside
    :type data: Dict[str, Union[Number, Union[np.ndarray, cp.ndarray]]]
    :param variable: The variable name
    :type variable: str
    """
    variable_data = data[variable]
    if isinstance(variable_data, cp.ndarray):
        variable_data = cp.asnumpy(variable_data)
    num_nnz = np.count_nonzero(variable_data)
    num_tot = np.prod(variable_data.shape)
    print(f"{variable}: {num_nnz}/{num_tot} = {num_nnz/num_tot*100}%")
