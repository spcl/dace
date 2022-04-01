In this folder, the SDFG construction API is demonstrated. This API is used in the various DaCe frontends and when
writing transformations. In particular, the samples are as follows:

* `control_flow.py`: Showcases state machine creation with control flow constructs such as loops, as well as the 
  `simplify` call, which will fuse unnecessary states and remove redundant memory.
* `cublas_tasklet.py`: Shows manual creation of a GPU SDFG, as well as a custom-code C++ tasklet that calls CUBLAS.
* `jagged_arrays.py`: Demonstrates indirect memory access and dynamic ranges in maps by iterating over a jagged array.
* `nested_states.py`: Sample that showcases nested SDFG creation with the API.
* `stencil_boundaries.py`: A 7x7 stencil with programmatically-generated boundary conditions that run in parallel.
  Also demonstrates maps without inputs and the `dace.subsets.Range` syntax.
