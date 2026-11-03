# DaCe Python-Frontend

The Python-Frontend aims to assist users in creating SDFGs from Python code
relatively quickly. You may read a list of supported Python features
[here](python_supported_features.md). The frontend supports also operations
among DaCe arrays, in a manner similar to NumPy. A short tutorial can be bound
[here](https://nbviewer.jupyter.org/github/spcl/dace/blob/main/tutorials/numpy_frontend.ipynb).
Please note that the Python-Frontend is still in an early version. For any issues
and feature requests, you can create an issue in the main DaCe project. You can
also address any questions you have to alziogas@inf.ethz.ch

## Supported Python Versions

The DaCe framework officially supports Python 3 from version 3.7.
The Python-Frontend also works with version 3.8-3.10. However, the module SymPy
must be updated to version 1.6.2 or newer. Please note that there are some
issues between DaCe and SymPy 1.6.2 (see [#367](https://github.com/spcl/dace/pull/367)).

## Main Limitations

- Classes are only supported in JIT mode.
- Lists, sets, and dictionaries are not supported as data. There is limited support for other uses, e.g., as arguments to some methods.
- Only `range`, `parrange`, and `dace.map` iterators are supported.
- Recursion is not supported.

## Automatic parsing

By default, DaCe tries to parse every call as a dace.program. If the object being called has an `__sdfg__` method, it will
be used instead of trying to parse `__call__`. Additionally, a function called `dace.in_program()` returns `True` while in a
DaCe parsing context.

If parsing fails, DaCe will try to automatically generate a callback to the Python interpreter, marshalling types such
as NumPy/CuPy arrays such that they match internal data containers. A warning will also be raised when this happens.

## NumPy Compatibility

The Python-Frontend currently supports a limited subset of NumPy:
- Python unary and binary operations among NumPy arrays, constants, and symbols. Binary operations mainly work between arrays that have the same shape. Operations between arrays of size 1 and arrays of any size are also supported.
- Array creation routines `ndarray`, `eye`
- Array manipulation routine `transpose`
- Math routines `eye`, `exp`, `sin`, `cos`, `sqrt`, `log`, `conj`, `real`, `imag` (only the input positional argument supported)
- Reduction routines `sum`, `mean`, `amax`, `amin`, `argmax`, `argmin` (input positional and `axis` keyword arguments supported)
- Type conversion routines, e.g., `int32`, `complex64`, etc.

### Routines lowered onto library nodes

Some routines are not expanded into Maps by the frontend at all: they are emitted as a library
node, so each backend picks its own implementation (an optimized CPU/GPU library call, or a
portable loop) instead of inheriting whatever shape the frontend happened to write.

| Routine | Library node | Notes |
| --- | --- | --- |
| `matmul`, `dot`, `@` | `MatMul`, `BatchedMatMul`, `Gemv`, `Ger`, `Dot` | picked by the operand ranks |
| `transpose`, `.T` | `Transpose`, `TensorTranspose` | 2-D goes to `Transpose`, N-D to `TensorTranspose` |
| `tensordot` | `TensorDot` | |
| `einsum` | `Einsum` | |
| `linalg.cholesky`, `linalg.inv`, `linalg.solve` | `Cholesky`, `Inv`, `Solve` | LAPACK-backed |
| `fft.fft`, `fft.ifft`, `fft.fftn`, `fft.ifftn` | `FFT`, `IFFT` | |
| `cumsum`, `cumprod` | `Scan` | inclusive scan along the last axis (or a 1-D operand); see below |
| `roll` | `CShift` | one node per `(shift, axis)` pair, chained in numpy's order; see below |

`cumsum` / `cumprod` scan the LAST axis. A rank > 1 operand becomes a Map over the leading axes
with the scan inside, so the batch stays parallel and only the recurrence is sequential. numpy's
integer widening applies (signed to `int64`, unsigned to `uint64`; floats keep their dtype). An
inner axis, and an axis-less scan over a rank > 1 operand, are refused rather than silently
scanned along the last axis.

`roll` builds its nodes with `ShiftDirection.NUMPY`. `CShift` is Fortran `CSHIFT` by default and
rotates the OTHER WAY (`CSHIFT(x, s)(i)` reads `x(mod(i + s, n))`, `roll(x, s)[i]` reads
`x[(i - s) % n]`), so the direction is named on the node rather than negated by the caller.
`axis=None` over a rank > 1 operand is refused: numpy flattens there, which is a reshape only when
the operand is contiguous.

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
