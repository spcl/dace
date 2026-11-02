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
* `adaptive_quadrature.py`: Adaptive quadrature, demonstrating consume loops
  (`dace.consume`) for tasks that split themselves until they are accurate
  enough, so that work concentrates where the integrand needs it. Runs on GPUs
  with `--gpu`.
* `sssp.py`: Single-source shortest paths as a worklist of vertices to relax,
  an asynchronous (not bulk-synchronous) algorithm, demonstrating consume loops
  with atomic updates that return the old value (`dace.atomic`), and reasoning
  about a benign race (reading values that only decrease). Runs on GPUs with
  `--gpu`.

An example of how to use the transformation API can be found in the
tutorials and in the [Matrix-Matrix Multiplication sample](samples/optimization/matmul.py).
