from dace import data, memlet, subsets, symbolic, dtypes, sdfg as sd
from typing import Iterable, Union, Any

def parseHBMArray(arrayname : str, array : data.Array) -> "dict[str, Any]":
    """
    parses HBM properties of an array (hbmbank and hbmalignment). 
    Returns none if hbmbank is not present as property

    :return: A mapping from (arrayname, sdfg of the array) to a mapping
    from string that contains collected information.
    'ndim': contains the dimension of the array == len(shape)
    'lowbank': The lowest bank index this array is placed on
    'shape': The shape of the whole array
    'numbanks': The number of banks across which this array spans
    """
    if("hbmbank" not in array.location):
        return None
    hbmbankrange : subsets.Range = array.location["hbmbank"]
    if(not isinstance(hbmbankrange, subsets.Range)):
        raise TypeError(f"Locationproperty 'hbmbank' must be of type subsets.Range for {arrayname}")
    low, high, stride = hbmbankrange[0]
    if(stride != 1):
        raise NotImplementedError(f"Locationproperty 'hbmbank' does not support stride != 1 for {arrayname}")
    numbank = high - low + 1
    shape = array.shape
    ndim = len(shape)
    if(low == high):
        return {"ndim" : ndim, "shape" : array.shape,
            "lowbank" : low, "numbank": 1}
    if(shape[0] != numbank):
        raise ValueError("The size of the first dimension for an array divided "
            "accross k HBM-Banks must equal k. This does not hold for "
            f"{arrayname}")
    return {"ndim" : ndim, "shape" : array.shape, "lowbank" : low,
            "numbank": numbank}