# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import random
import numpy as np

from typing import Dict, List, Set, Tuple, Union, Sequence
from dace import dtypes as ddtypes
from dace.data import Data, Scalar, make_array_from_descriptor, Array
from dace.sdfg import SDFG

from dace import SDFG, DataInstrumentationType
from dace import config, nodes, symbolic
from dace.codegen.instrumentation.data.data_report import InstrumentedDataReport
from dace.libraries.standard.memory import aligned_ndarray


def random_arguments(sdfg: SDFG) -> Dict:
    """
    Creates random inputs and empty output containers for the SDFG.

    :param SDFG: the SDFG.
    :return: a dict containing the arguments.
    """
    # Symbols
    symbols = {}
    for k, v in sdfg.constants.items():
        symbols[k] = int(v)
    for k in sdfg.free_symbols:
        symbols[k] = random.randint(1, 3)

    arguments = {**symbols}
    for state in sdfg.nodes():
        for dnode in state.data_nodes():
            if dnode.data in arguments:
                continue

            array = sdfg.arrays[dnode.data]
            if state.in_degree(dnode) == 0 or state.out_degree(dnode) == 0 or not array.transient:
                if state.in_degree(dnode) == 0:
                    np_array = _random_container(array, symbols_map=symbols)
                else:
                    np_array = _empty_container(array, symbols_map=symbols)

                arguments[dnode.data] = np_array

    return arguments


def create_data_report(sdfg: SDFG, arguments: Dict, transients: bool = False) -> InstrumentedDataReport:
    """
    Creates a data instrumentation report for the given SDFG and arguments.

    :param SDFG: the SDFG.
    :param arguments: the arguments to use.
    :param transients: whether to instrument transient array.
    :return: the data report.
    """
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                if sdfg.arrays[node.data].transient and not transients:
                    continue
                if state.entry_node(node) is not None:
                    continue

                node.instrument = DataInstrumentationType.Save

    with config.set_temporary('compiler', 'allow_view_arguments', value=True):
        _ = sdfg(**arguments)

    # Disable data instrumentation again
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                if sdfg.arrays[node.data].transient and not transients:
                    continue
                if state.entry_node(node) is not None:
                    continue

                node.instrument = DataInstrumentationType.No_Instrumentation

    dreport = sdfg.get_instrumented_data()
    return dreport


def arguments_from_data_report(sdfg: SDFG, data_report: InstrumentedDataReport) -> Dict:
    """
    Creates the arguments for the SDFG from the data report.

    :param SDFG: the SDFG.
    :param data_report: the data report.
    :return: a dict containing the arguments.
    """
    symbols = {}
    for k, v in sdfg.constants.items():
        symbols[k] = int(v)

    arguments = {**symbols}
    for state in sdfg.nodes():
        for dnode in state.data_nodes():
            if dnode.data in arguments:
                continue

            array = sdfg.arrays[dnode.data]
            if state.in_degree(dnode) == 0 or state.out_degree(dnode) == 0 or not array.transient:
                data = data_report[dnode.data]
                if isinstance(data, Sequence):
                    data = data.__iter__().__next__()

                if isinstance(array, Array):
                    arguments[dnode.data] = make_array_from_descriptor(array, data, symbols=sdfg.constants)
                else:
                    scalar = data.astype(array.dtype.as_numpy_dtype()).item()
                    arguments[dnode.data] = scalar

    return arguments


def _random_container(array: Data, symbols_map: Dict[str, int]) -> np.ndarray:
    shape = symbolic.evaluate(array.shape, symbols=symbols_map)
    newdata = _uniform_sampling(array, shape)
    if isinstance(array, Scalar):
        return newdata
    else:
        return _align_container(array, symbols_map, newdata)


def _empty_container(array: Data, symbols_map: Dict[str, int]) -> Union[int, float, np.ndarray]:
    if isinstance(array, Scalar):
        npdt = array.dtype.as_numpy_dtype()
        if npdt in [np.float16, np.float32, np.float64]:
            return 0.0
        else:
            return 0
    else:
        shape = symbolic.evaluate(array.shape, symbols=symbols_map)
        empty_container = np.zeros(shape).astype(array.dtype.as_numpy_dtype())
        return _align_container(array, symbols_map, empty_container)


def _align_container(array: Data, symbols_map: Dict[str, int], container: np.ndarray) -> np.ndarray:
    view: np.ndarray = make_array_from_descriptor(array, container, symbols_map)
    if isinstance(array, Array) and array.alignment:
        return aligned_ndarray(view, array.alignment)
    else:
        return view


def _uniform_sampling(array: Data, shape: Union[List, Tuple]):
    npdt = array.dtype.as_numpy_dtype()
    if npdt in [np.float16, np.float32, np.float64]:
        low = 0.0
        high = 1.0
        if isinstance(array, Scalar):
            return np.random.uniform(low=low, high=high)
        else:
            return np.random.uniform(low=low, high=high, size=shape).astype(npdt)
    elif npdt in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
        low = 0
        high = 1
        if isinstance(array, Scalar):
            return np.random.randint(low, high)
        else:
            return np.random.randint(low, high, size=shape).astype(npdt)
    elif array.dtype in [ddtypes.bool, ddtypes.bool_]:
        if isinstance(array, Scalar):
            return np.random.randint(low=0, high=2)
        else:
            return np.random.randint(low=0, high=2, size=shape).astype(npdt)
    else:
        raise TypeError()
