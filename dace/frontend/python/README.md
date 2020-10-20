# DaCe Python-Frontend

The Python-Frontend aims to assist users in creating SDFGs from Python code
relatively quickly. You may read a list of supported Python features
[here](python_supported_features.md). The frontend supports also operations
among DaCe arrays, in a manner similar to NumPy. A short tutorial can be bound
[here](https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/numpy_frontend.ipynb).
Please note that the Python-Frontend is still in an alpha version. For any issues
and feature requests, you can create an issue in the main DaCe project. You can
also address any questions you have to alziogas@inf.ethz.ch

## Known Issues

### Issues when automatic strict transformations are enabled

When automatic strict transformations are enabled, SDFGs created using the
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
ranges, leading to RW/WR/WW dependencies, InlineSDFG and StateFusion may violate them (see #352).
- When there are sequential dependencies between statements due to updating a loop variable,
StateFusion may erroneously lead to concurrent execution of those statements (see #315).
- The RedundantArray transformations may produce erroneous Memlets for certain ranges (see #371).
  
Temporary workaround: Disable the automatic strict transformations flag in the configuration file `.dace.conf`.

### Issues with symbolic expressions

- Symbolic expressions with complex numbers generate invalid code. This is due to the use of the SymPy library.
- Data that have their value set by a symbolic expression may have wrong data type.

Temporary workaround: Break symbolic expressions with many binary operations to simple ones,
where each one has only one binary operation. Store the intermediate result to
distinct variables.
