.. _integration:

Integrating SDFGs in Existing Programs
======================================

Compiling an SDFG creates a standalone dynamic library (``.so``, ``.dll``, or ``.dylib``) file that can be reused in
existing programs, written in any language that supports calling C functions (or Foreign Function Interface, FFI). 
The library can be linked to the program at compile time, or loaded at runtime to Python using the :class:`~dace.codegen.compiled_sdfg.CompiledSDFG`
class.

The file itself will be placed in the *build folder* of the SDFG, which by default is the ``.dacecache`` subfolder in
the current working directory. This behavior is configurable through :envvar:`default_build_folder`.

Python
------

Calling the decorated DaCe function or SDFG object directly will compile the SDFG (if necessary) and execute it.

Using arrays with custom padding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When calling an SDFG from Python with standard NumPy or other array types, the strides are automatically set by the
array programming library. However, if the SDFG uses custom padding, the array must be created with the correct
dimensions and strides. DaCe provides a helper function to create such arrays: :func:`~dace.data.make_array_from_descriptor`.
Example usage:

.. code-block:: python

    import dace

    custom_desc = dace.data.Array(
        dace.float32, [10, 10],
        storage=dace.StorageType.GPU_Global,
        strides=[20, 1], start_offset=3)

    @dace.program
    def my_sdfg(A: custom_desc):
        ...

    # Create an array with custom padding
    a = dace.data.make_array_from_descriptor(custom_desc)
    # Alternatively, sdfg.arrays['A'] could have been used as an argument
    
    # A is a valid GPU (CuPy) array that can be used in the SDFG or outside of it
    A[:] = 1

    # Call the SDFG
    my_sdfg(A)



Compiled SDFG objects
~~~~~~~~~~~~~~~~~~~~~

.. note::
    If performance of each call is not a concern (for example, if the SDFG runs for a long time), it is always preferable 
    to :ref:`call the SDFG object itself <calling_sdfgs>` or the ``@dace`` function.


Compiled SDFG objects are usually returned from DaCe when compiling programs. You can also manually create one using
the :func:`~dace.sdfg.utils.load_precompiled_sdfg` function for ease of use. The following example shows how to load a
compiled SDFG directly:

.. code-block:: python

    import dace
    from dace.codegen.compiled_sdfg import CompiledSDFG
    from dace.sdfg.utils import load_precompiled_sdfg

    # Load a compiled SDFG from a build folder
    csdfg = load_precompiled_sdfg('.dacecache/my_sdfg')

    # Load the compiled SDFG from an .so file directly (low-level API)
    sdfg = dace.SDFG.from_file('my_sdfg.sdfg')
    compiled_sdfg = CompiledSDFG(sdfg, 'my_sdfg.so')

Internally, the :class:`~dace.codegen.compiled_sdfg.CompiledSDFG` class is a wrapper around ``ctypes`` that allows
you to call the SDFG's entry point function, perform basic type checking, and argument marshalling (i.e., array to pointer,
Python callback to function pointer, etc.).

Since the compiled SDFG is a low-level interface, it is much faster to call than the Python interface. 
`We show this behavior in the Benchmarking tutorial <https://nbviewer.org/github/spcl/dace/blob/master/tutorials/benchmarking.ipynb>`_. 
However, it requires caution as opposed to calling the ``@dace.program`` or the ``SDFG`` object because:

    * Each array return value is represented internally as a single array (not reallocated every call) and will be 
      **reused** across calls.
    * Less type checking is performed, so data may be reinterpreted if passed wrong.
    * The closure of the program (e.g., scalar fields in a class) will **not** be recomputed and thus may be stale.


Internal Structure and Functions
--------------------------------

The build folder contains the compiled SDFG file (``program.sdfg``), the exact used configuration file for reproducibility,
and several subfolders: ``src`` for source code, ``build`` for the linked library, ``include`` for
an auto-generated header file that can be used to call the library, ``profiling`` for profiling, ``perf`` and ``data`` for 
instrumentation, ``map`` for source maps (used in debugging), and ``sample``, which contains a short code sample that
demonstrates how to invoke the library from C.

A compiled SDFG library contains three functions, which are named after the SDFG:

    * ``__dace_init_<SDFG name>``: Initializes the SDFG, allocating all arrays and initializing all data descriptors.
      The function returns a handle to the state object, which is a struct containing all information that will persist
      between invocations of the SDFG. The other functions take this handle as their first argument. The arguments to
      this function are only the symbols used in the SDFG, ordered by name.
    * ``__dace_exit_<SDFG name>``: Deallocates all arrays and frees all data descriptors in the given handle.
    * ``__program_<SDFG name>``: The actual SDFG function, which takes the handle as its first argument, followed by
      the arguments to the SDFG, ordered by name, followed by the symbol arguments, also ordered by name.

The header file contains the function prototypes and the struct definition for the handle.


.. _integration_c:

C/C++ and C ABI-Compatible Languages
------------------------------------

The header file can be used to call the compiled SDFG from C, C++, or FORTRAN programs. The following example shows how 
to call a compiled SDFG from C:

.. code-block:: c

    #include "my_sdfg.h"

    int main() {
        int M = 1, N = 20, K = 3;
        double *A = malloc(100 * N * sizeof(double));
        double *B = malloc(100 * M * sizeof(double));
        int i;

        // Initialize the SDFG (note that only the symbols are passed)
        my_sdfg_t handle = __dace_init_my_sdfg(K, M, N);

        // ...

        // Call the SDFG with arguments and symbols
        for (i = 0; i < 10; ++i)
            __program_my_sdfg(handle, A, B, K, M, N);

        // ...

        // Finalize the SDFG, freeing its resources
        __dace_exit_my_sdfg(handle);

        free(A);
        free(B);
        return 0;
    }
