# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union
import os
from dace import dtypes, SDFG

import numpy as np
try:
    from numpy.typing import ArrayLike
except (ModuleNotFoundError, ImportError):
    ArrayLike = Any


@dataclass
class InstrumentedDataReport:
    """
    Instrumented data reports are folders saved by the Save data instrumentation provider.
    Their goal is to represent a set of data contents for completely reproducing a run of an SDFG.
    This can be used to create consistent inputs for benchmarking, or for retrieving intermediate data
    automatically for correctness checking / data debugging.

    The folder structure of a data report is as follows:
    /path/to/report/<array name>/<uuid>_<version>.bin
    where <array name> is the array in the SDFG, <uuid> is a unique identifier to the access node from which
    this array was saved, and <version> is a running number for the currently-saved array (e.g., when an access node is
    written to multiple times in a loop).
    
    The files themselves are direct binary representations of the whole data (with padding and strides), for complete
    reproducibility. When accessed from the report, a numpy wrapper shows the user-accessible view of that array.
    Example of reading a file::

        dreport = sdfg.get_instrumented_data()  # returns a report
        print(dreport.keys())  # will print 'A', 'versioned'
        array = dreport['A']  # return value is a single array if there is only one version
        varrays = dreport['versioned']  # otherwise, return value is a sorted list of versions
        # after loading, arrays can be used normally with numpy
        assert np.allclose(array, real_A)
        for arr in varrays:
            print(arr[5, :])



    :seealso: dace.dtypes.DataInstrumentationType.Save
    :seealso: dace.dtypes.DataInstrumentationType.Restore
    """
    sdfg: SDFG
    folder: str
    files: Dict[str, List[str]]
    loaded_arrays: Dict[Tuple[str, int], ArrayLike]

    def __init__(self, sdfg: SDFG, folder: str) -> None:
        """
        Loads a data instrumentation report of an SDFG from the specified folder.
        :param sdfg: SDFG from which the report was created.
        :param folder: Root folder of the report.
        """
        self.sdfg = sdfg
        self.folder = folder
        self.files = {}
        self.loaded_arrays = {}

        # Prepare file mapping
        array_names = os.listdir(folder)
        for aname in array_names:
            files = []

            # Sort files numerically
            filenames = os.listdir(os.path.join(folder, aname))
            filenames = sorted([(*(int(s) for s in f.split('.')[0].split('_')), f) for f in filenames])
            for entry in filenames:
                files.append(os.path.join(folder, aname, entry[-1]))

            self.files[aname] = files

    def keys(self) -> Set[str]:
        """ Returns the array names available in this data report. """
        return self.files.keys()

    def __getitem__(self, item: str) -> Union[ArrayLike, List[ArrayLike]]:
        """
        Returns the instrumented (saved) data from the report according to the data descriptor (array) name. 
        :param item: Name of the array to read.
        :return: An array (if a single entry in the report is given) or a list of versions of the array across
                 the report.
        """
        filenames = self.files[item]
        desc = self.sdfg.arrays[item]
        dtype: dtypes.typeclass = desc.dtype

        results = []
        for i, file in enumerate(filenames):
            # Make numpy array from data descriptor
            nparr = np.fromfile(file, dtype=dtype.as_numpy_dtype())
            # No need to use ``start_offset`` because the unaligned version is saved
            view = np.reshape(nparr, desc.shape)
            view.strides = tuple(s * dtype.bytes for s in desc.strides)
            self.loaded_arrays[item, i] = nparr
            results.append(view)

        if len(results) == 1:
            return results[0]
        return results

    def update_report(self):
        """
        Stores the retrieved arrays from the report back to the files. Can be used to modify data that will be loaded
        when restoring a data instrumentation report.
        :see: dace.dtypes.DataInstrumentationType.Restore
        """
        for (k, i), loaded in self.loaded_arrays.items():
            loaded.tofile(self.files[k][i])
