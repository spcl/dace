# DaCe Python-Frontend

The Python-Frontend aims to assist users in creating SDFGs from Python code
relatively quickly. You may read a list of supported Python features
[here](python_supported_features.md). The frontend supports also operations
among DaCe arrays, in a manner similar to NumPy. A short tutorial can be bound
[here](https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/numpy_frontend.ipynb).
Please note that the Python-Frontend is still in an alpha versions. For any issues
and feature requests, you can create an issue in the main DaCe project. You can
also address any questions you have to alziogas@inf.ethz.ch

## Known Issues

### Multiple array accesses in a nested scope lead to invalid SDFGs
When accesing an array in a nested scope (typically the body of a Map or a
nested SDFG/DaCe program call) multiple times, but with different and partially
overlapping indices/ranges, an invalid SDFG may be generated.  

Status: Under investigation  
Temporary workaround: None

### Sequential statements in the body of a loop are parallelized in the generated SDFG
This is an issue with the transformation StateFusion. Typically this occurs when
two statements have a sequential dependency due to the use of a Map or loop index/variable.
There are cases where StateFusion does not detect such dependencies properly.

Status: Pending fix in StateFusion  
Temporary workaround: Disable in the configuration file the automatic strict transformations

### A number of binary operators do not work as expected
Operators like Div and Mod behave in a way that is consistent with C/C++ and not Python.

Status: Under investigation  
Temporary workaround: None
