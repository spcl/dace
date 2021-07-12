import copy
from typing import Union, Tuple
from dace import data as dt, SDFG, dtypes, subsets as sbs, symbolic

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


def modify_distributed_subset(subset: Union[sbs.Subset, list, tuple],
                              change: int):
    """
    Modifies the first index of :param subset: (the one used for distributed subsets).
    :param subset: is deepcopied before any modification to it is done.
    :param change: the first index is set to this value, unless it's (-1) in which case
        the first index is completly removed
    """
    cps = copy.deepcopy(subset)
    if isinstance(subset, sbs.Subset):
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


def get_multibank_ranges_from_subset(subset: Union[sbs.Subset, str],
                                     sdfg: SDFG) -> Tuple[int, int]:
    """
    Returns the upper and lower end of the accessed HBM-range, evaluated using the
    constants on the SDFG.
    :returns: (low, high) where low = the lowest accessed bank and high the 
        highest accessed bank + 1.
    """
    if isinstance(subset, str):
        subset = sbs.Range.from_string(subset)
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


def parse_location_bank(array_or_bank: Union[dt.Array, str]) -> Tuple[str, str]:
    """
    :param array_or_bank: Either an array on FPGA or a valid memory bank specifier string
    :return: None if an array is given which does not have a location['bank'] value. 
        Otherwise it will return a tuple (bank_type, bank_assignment), where bank_type
        is one of 'DDR', 'HBM' and bank_assignment a string that describes which banks are 
        used.
    """
    if isinstance(array_or_bank, str) or "bank" in array_or_bank.location:
        if isinstance(array_or_bank, str):
            val: str = array_or_bank
        else:
            val: str = array_or_bank.location["bank"]
        split = val.split(".")
        if (len(split) != 2):
            raise ValueError(
                f"Failed to parse memory bank specifier {val}, set in location['bank']. "
                "Expected format is <type>.<id> (e.g. ddr.2 or hbm.0:2)")
        split[0] = split[0].upper()

        if (split[0] == "DDR" or split[0] == "HBM"):
            return (split[0], split[1])
        else:
            raise ValueError(
                f"{split[0]} is an invalid bank type for location['bank']. Supported are HBM and DDR."
            )
    else:
        return None