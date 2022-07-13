.. _integration:

Integrating SDFGs in Existing Programs
======================================

Compiling SDFGs yields a subfolder in .dacecache (or configured by ``default_build_folder``)

Python
------
calling directly, briefly mention type checking and marshalling

Discuss ``CompiledSDFG`` objects and calling them directly for performance.

.. note::
    if compiled, return values are cached


Using GPU Arrays
----------------

You can use CuPy arrays and torch arrays, or any array that implements ``__cuda_array_interface__``. In JIT mode
it will create a ``GPU_Global`` data descriptor.


C/C++ and C ABI-Compatible Languages
------------------------------------

ABI compatible, mention include/ and sample/ folders in dacecache

