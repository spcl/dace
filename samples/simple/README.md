# Simple DaCe Samples

The files in this folder are samples of the DaCe Python interface (using mostly 
the explicit low-level interface). They are **unoptimized** and are meant to be
transformed through the SDFG API, the VSCode plugin, or through the console
command-line interface.

An example of how to use the SDFG API for transformations can be found in the
tutorials and in `matmul.py`.

The examples found in this folder are:
* `axpy.py`: Scaled vector addition (`a*x+y`), demonstrates the Map parallel scope.
* `mat_add.py`, `transpose.py`: Simple matrix addition and transposition, 
  demonstrate multi-dimensional Maps using the explicit syntax.
* `sum.py`: Simple sum, demonstrating the use of the Reduce library node.
* `ddot.py`: Dot product, demontrates explicit syntax for write-conflict resolution.
* `matmul.py`: Matrix-matrix multiplication, demonstrating how transformations
  and library nodes can be used to output high-performance code.
* `histogram.py`, `histogram_declarative.py`: Two versions of computing two-dimensional
  histograms. The declarative version demonstrates the use of the Reduce library node.
* `filter.py`: Predicate-based filtering, demonstrating the use of Streams.
* `fibonacci.py`: Fibonacci sequence, demonstrating the Consume scope for 
  handling dynamic tasks.
* `mandelbrot.py`: Mandelbrot set computation, demonstrating nested SDFGs 
  via control flow in a parallel scope.
* `spmv.py`: Sparse matrix-vector multiplication, demonstrating dynamic-range
  map scopes and indirect memory access.
