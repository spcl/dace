# Simple DaCe Samples

The files in this folder are samples of the DaCe Python interface (using mostly 
the explicit low-level interface). They are **unoptimized** and are meant to be
transformed through the SDFG API, the VSCode plugin, or through the console
command-line interface.

The examples found in this folder are:
* `axpy.py`: Scaled vector addition (`a*x+y`)
* `laplace.py`: Stencil operation, demonstrating the `dace.map` parallel scope.
* `mandelbrot.py`: Mandelbrot set computation, demonstrating control flow in a 
   parallel scope (nested SDFGs).
* `spmv.py`: Sparse matrix-vector multiplication, demonstrating dynamic-range 
  map scopes and indirect memory access.

An example of how to use the transformation API can be found in the
tutorials and in the [Matrix-Matrix Multiplication sample](samples/optimization/matmul.py).
