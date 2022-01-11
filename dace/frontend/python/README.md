# DaCe Python-Frontend

The Python-Frontend aims to assist users in creating SDFGs from Python code
relatively quickly. You may read a list of supported Python features
[here](python_supported_features.md). The frontend supports also operations
among DaCe arrays, in a manner similar to NumPy. A short tutorial can be bound
[here](https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/numpy_frontend.ipynb).
Please note that the Python-Frontend is still in an early version. For any issues
and feature requests, you can create an issue in the main DaCe project. You can
also address any questions you have to alziogas@inf.ethz.ch

## Supported Python Versions

The DaCe framework officially supports Python 3 up to version 3.7.
The Python-Frontend also works with version 3.8. However, the module SymPy
must be updated to version 1.6.2 or newer. Please note that there are some
issues between DaCe and SymPy 1.6.2 (see [#367](https://github.com/spcl/dace/pull/367)).  

**Neither the DaCe framework nor the Python-Frontend have been tested with
Python version 3.9**

## Main Limitations

- Classes are not supported.
- Lists, sets, and dictionaries are not supported as data. There is limited support for other uses, e.g., as arguments to some methods.
- Only `range`, `parrange`, and `dace.map` iterators are supported.
- Recursion is not supported.

## NumPy Compatibility

The Python-Frontend currently supports a limited subset of NumPy:
- Python unary and binary operations among NumPy arrays, constants, and symbols. Binary operations mainly work between arrays that have the same shape. Operations between arrays of size 1 and arrays of any size are also supported.
- Array creation routines `ndarray`, `eye`
- Array manipulation routine `transpose`
- Math routines `eye`, `exp`, `sin`, `cos`, `sqrt`, `log`, `conj`, `real`, `imag` (only the input positional argument supported)
- Reduction routines `sum`, `mean`, `amax`, `amin`, `argmax`, `argmin` (input positional and `axis` keyword arguments supported)
- Type conversion routines, e.g., `int32`, `complex64`, etc.

There is also upcoming support for NumPy ufuncs. You may preview ufunc support with `add`, `subtract`, `multiply`, and `minimum`. The following are supported:
- Ufunc call with optional `out`, `where`, and `dtype` keyword arguments. Standard NumPy broadcasting rules are applied.
- Ufunc `reduce` method with optional `out`, `keepdims`, `axis`, and `initial` keyword arguments.
- Ufunc `accumulate` method with optional `out`, `axis` keyword arguments.
- Ufunc `outer` method with optional `out`, `where`, and `dtype` keyword arguments.

## Known Issues

### Issues when automatic simplification is enabled

When automatic simplification is enabled, SDFGs created using the
Python-Frontend are automatically transformed using:
- InlineSDFG
- EndStateElimination
- StateFusion
- InMergeArrays
- OutMergeArrays
- RedundantArrayCopyingIn
- RedundantArrayCopying
- RedundantArray
- RedundantSecondArray

These transformations clean up the SDFG by removing extraneous arrays, and reducing
the number of states and nested scopes/SDFGs, enabling in the process further
optimizations. However, there exist cases where applying these transformations
automatically may result in invalid SDFGs. Currently known issues include:
- When accessing inside a Map an Array multiple times with different but overlapping
ranges, leading to RW/WR/WW dependencies, InlineSDFG and StateFusion may violate them.
- When there are sequential dependencies between statements due to updating a loop variable,
StateFusion may erroneously lead to concurrent execution of those statements (see [#315](https://github.com/spcl/dace/issues/315)).
  
Temporary workaround: Disable the automatic simplification pass flag in the configuration file `.dace.conf`.
