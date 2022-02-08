## DaCe Samples

This folder contains various samples that showcase the capabilities of the DaCe framework.

There are several sub-folders here:
* **simple**: Simple examples using the Python/NumPy interface
* **explicit**: Examples that use the explicit data-centric API (`dace.map`, `dace.tasklet`)
* **sdfg_api**: Programs that use the SDFG API to create the DaCe intermediate representation directly. Useful if you are writing your own frontend.
* **optimization**: Examples that use the transformation and instrumentation API to optimize programs to run fast on CPUs and GPUs
* **fpga**: FPGA programs with explicit circuit design patterns (e.g., systolic arrays), mostly using the SDFG API
* **distributed**: Python/NumPy and explicit applications that run on multiple machines
* **codegen**: Samples showing how to extend the code generator of DaCe to support new platforms (e.g., Tensor Cores)


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
