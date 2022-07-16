Working with Fast Linear Algebra (BLAS) Libraries
=================================================

DaCe natively supports linear algebra operations (such as matrix multiplication) by using Basic Linear Algebra
Subprogram (BLAS) libraries. They integrate into existing programs when using, for example, the ``@`` operator, or
calling NumPy functions such as ``numpy.dot``. In the resulting SDFG, :ref:`libnodes` will appear and can be then 
expanded to use one of the fast BLAS libraries, for example `Intel MKL <https://software.intel.com/en-us/intel-mkl>`_ 
or `NVIDIA CUBLAS <https://developer.nvidia.com/cublas>`_.

By default, to maintain compatibility with all platforms, DaCe may expand library nodes to the native 
(unoptimized) SDFG implementation. This could be useful for manual optimization (e.g., to fuse algebraic operations
together), but for most cases it is beneficial to call those libraries for their high-performance implementations.

The expansion target can be easily configured in :envvar:`library.blas.default_implementation` or manually on the graph,
for each node or for the entire ``blas`` library globally. For specific nodes, modify their ``implementation`` property.
For global configuration, change ``dace.libraries.blas.default_implementation``. You can also check if a specific BLAS
library is installed and can be found by DaCe using the ``is_installed`` static method of each library environment.

The example below shows how you can check whether a library is installed inside a Python script, as well as change the 
setting:

.. code-block:: python

    from dace.libraries import blas
    
    print('BLAS calls will expand by default to', blas.default_implementation)

    if blas.IntelMKL.is_installed():
        blas.default_implementation = 'MKL'
    elif blas.cuBLAS.is_installed():
        blas.default_implementation = 'cuBLAS'
    elif blas.OpenBLAS.is_installed():
        blas.default_implementation = 'OpenBLAS'
    elif not blas.BLAS.is_installed():
        # No BLAS library found, use the unoptimized native SDFG fallback
        blas.default_implementation = 'pure'

.. note::
    If a library is installed but cannot be found with the above method or during compilation, the environment variables
    ``CPATH`` / ``LIBRARY_PATH`` / ``LD_LIBRARY_PATH`` may be misconfigured. See :ref:`troubleshooting` for more information.
