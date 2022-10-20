.. _integration:

Integrating SDFGs in Existing Programs
======================================

Compiling SDFGs yields a subfolder in ``.dacecache`` (or configured by ``default_build_folder``)

Python
------
calling directly, briefly mention type checking and marshalling

Discuss ``CompiledSDFG`` objects and calling them directly for performance.

.. note::
    if compiled, return values are cached


Mention :func:`~dace.data.make_array_from_descriptor` and :func:`~dace.data.make_reference_from_descriptor`


Using GPU Arrays
----------------

You can use GPU arrays in Python directly via `CuPy <https://cupy.dev/>`_ arrays, PyTorch tensors, or any 
array that implements ``__cuda_array_interface__``. In DaCe, JIT mode will create a ``GPU_Global`` data
descriptor automatically and use that array on the GPU.

.. _integration_c:

C/C++ and C ABI-Compatible Languages
------------------------------------

ABI compatible, mention include/ and sample/ folders in dacecache

