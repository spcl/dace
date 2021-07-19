# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import List, Union, Tuple
from dace import data as dt, SDFG, dtypes, subsets, symbolic

_FPGA_STORAGE_TYPES = {
    dtypes.StorageType.FPGA_Global, dtypes.StorageType.FPGA_Local,
    dtypes.StorageType.FPGA_Registers, dtypes.StorageType.FPGA_ShiftRegister
}


def is_hbm_array(array: dt.Data):
    """
    :return: True if this array is placed on HBM
    """
    if (isinstance(array, dt.Array)
            and array.storage == dtypes.StorageType.FPGA_Global):
        res = parse_location_bank(array)
        return res is not None and res[0] == "HBM"
    else:
        return False


def is_fpga_array(array: dt.Data):
    """
    :return: True if this array is placed on FPGA memory
    """
    return isinstance(array, dt.Array) and array.storage in _FPGA_STORAGE_TYPES


def iterate_hbm_multibank_arrays(array_name: str, array: dt.Array, sdfg: SDFG):
    """
    Small helper function that iterates over the bank indices
    if the provided array is spanned across multiple HBM banks.
    Otherwise just returns 0 once.
    """
    res = parse_location_bank(array)
    if res is not None:
        bank_type, bank_place = res
        if (bank_type == "HBM"):
            low, high = get_multibank_ranges_from_subset(bank_place, sdfg)
            for i in range(high - low):
                yield i
        else:
            yield 0
    else:
        yield 0


def modify_distributed_subset(subset: Union[subsets.Subset, list, tuple],
                              change: int):
    """
    Modifies the first index of :param subset: (the one used for distributed subsets).
    :param subset: is deepcopied before any modification to it is done.
    :param change: the first index is set to this value, unless it's (-1) in which case
        the first index is completly removed
    """
    cps = copy.deepcopy(subset)
    if isinstance(subset, subsets.Subset):
        if change == -1:
            cps.pop([0])
        else:
            cps[0] = (change, change, 1)
    elif isinstance(subset, list) or isinstance(subset, tuple):
        if isinstance(subset, tuple):
            cps = list(cps)
        if change == -1:
            cps.pop(0)
        else:
            cps[0] = change
        if isinstance(subset, tuple):
            cps = tuple(cps)
    else:
        raise ValueError("unsupported type passed to modify_distributed_subset")

    return cps


def get_multibank_ranges_from_subset(subset: Union[subsets.Subset, str],
                                     sdfg: SDFG) -> Tuple[int, int]:
    """
    Returns the upper and lower end of the accessed HBM-range, evaluated using the
    constants on the SDFG.
    :returns: (low, high) where low = the lowest accessed bank and high the 
        highest accessed bank + 1.
    """
    if isinstance(subset, str):
        subset = subsets.Range.from_string(subset)
    low, high, stride = subset[0]
    if stride != 1:
        raise NotImplementedError(f"Strided HBM subsets not supported.")
    try:
        low = int(symbolic.resolve_symbol_to_constant(low, sdfg))
        high = int(symbolic.resolve_symbol_to_constant(high, sdfg))
    except:
        raise ValueError(
            f"Only constant evaluatable indices allowed for HBM-memlets on the bank index."
        )
    return (low, high + 1)


def parse_location_bank(array: dt.Array) -> Tuple[str, str]:
    """
    :param array: an array on FPGA global memory
    :return: None if an array is given which does not have a location['memorytype'] value. 
        Otherwise it will return a tuple (bank_type, bank_assignment), where bank_type
        is one of 'DDR', 'HBM' and bank_assignment a string that describes which banks are 
        used.
    """
    if "memorytype" in array.location:
        if "bank" not in array.location:
            raise ValueError(
                "If 'memorytype' is specified for an array 'bank' must also be specified"
            )
        val: str = array.location["bank"]
        memorytype: str = array.location["memorytype"]
        memorytype = memorytype.upper()
        if (memorytype == "DDR" or memorytype == "HBM"):
            return (memorytype, array.location["bank"])
        else:
            raise ValueError(
                f"{memorytype} is an invalid memorytype. Supported are HBM and DDR."
            )
    else:
        return None


def ptr(name: str,
        desc: dt.Data = None,
        sdfg: SDFG = None,
        subset_info_hbm: Union[subsets.Subset, int] = None,
        is_write: bool = None,
        dispatcher=None,
        ancestor: int = 0,
        is_array_interface: bool = False,
        interface_id: Union[int, List[int]] = None):
    """
    Returns a string that points to the data based on its name, and various other conditions
    that may apply for that data field.
    :param name: Data name.
    :param desc: Data descriptor.
    :param subset_info_hbm: Any additional information about the accessed subset. 
    :param ancestor: The ancestor level where the variable should be searched for if
        is_array_interface is True when dispatcher is not None
    :param is_array_interface: Data is pointing to an interface in FPGA-Kernel compilation
    :param interface_id: An optional interface id that will be added to the name (only for array interfaces)
    :return: C-compatible name that can be used to access the data.
    """
    if (desc is not None and is_hbm_array(desc)):
        if (subset_info_hbm == None):
            raise ValueError(
                "Cannot generate name for HBM bank without subset info")
        elif (isinstance(subset_info_hbm, int)):
            name = f"hbm{subset_info_hbm}_{name}"
        elif (isinstance(subset_info_hbm, subsets.Subset)):
            if (sdfg == None):
                raise ValueError(
                    "Cannot generate name for HBM bank using subset if sdfg not provided"
                )
            low, high = get_multibank_ranges_from_subset(subset_info_hbm, sdfg)
            if (low + 1 != high):
                raise ValueError(
                    "ptr cannot generate HBM names for subsets accessing more than one HBM bank"
                )
            name = f"hbm{low}_{name}"
            subset_info_hbm = low  #used for arrayinterface name where it must be int
    if is_array_interface:
        if is_write is None:
            raise ValueError("is_write must be set for ArrayInterface.")
        ptr_in = f"__{name}_in"
        ptr_out = f"__{name}_out"
        if dispatcher is not None:
            # DaCe allows reading from an output connector, even though it
            # is not an input connector. If this occurs, panic and read
            # from the output interface instead
            if is_write or not dispatcher.defined_vars.has(ptr_in, ancestor):
                # Throw a KeyError if this pointer also doesn't exist
                dispatcher.defined_vars.get(ptr_out, ancestor)
                # Otherwise use it
                name = ptr_out
            else:
                name = ptr_in
        else:
            # We might call this before the variable is even defined (e.g., because
            # we are about to define it), so if the dispatcher is not passed, just
            # return the appropriate string
            name = ptr_out if is_write else ptr_in
        # Append the interface id, if provided
        if interface_id is not None:
            if isinstance(interface_id, tuple):
                name = f"{name}_{interface_id[subset_info_hbm]}"
            else:
                name = f"{name}_{interface_id}"
    return name
