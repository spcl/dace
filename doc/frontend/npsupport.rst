NumPy Support
=============

.. note::

   This section is a work in progress.  It does not reflect the capabilities provided in the latest version of DaCe.


The Data-Centric Python-Frontend currently supports a limited subset of NumPy:

- Python unary and binary operations among NumPy arrays, constants, and symbols. Binary operations mainly work between arrays that have the same shape. Operations between arrays of size 1 and arrays of any size are also supported.
- Array creation routines ``ndarray``, ``eye``
- Array manipulation routine ``transpose``
- Math routines ``eye``, ``exp``, ``sin``, ``cos``, ``sqrt``, ``log``, ``conj``, ``real``, ``imag`` (only the input positional argument supported)
- Reduction routines ``sum``, ``mean``, ``amax``, ``amin``, ``argmax``, ``argmin`` (input positional and ``axis`` keyword arguments supported)
- Type conversion routines, e.g., ``int32``, ``complex64``, etc.
- All built-in universal functions (ufunc):

  - Ufunc call with optional ``out``, ``where``, and ``dtype`` keyword arguments. Standard NumPy broadcasting rules are applied.
  - Ufunc ``reduce`` method with optional ``out``, ``keepdims``, ``axis``, and ``initial`` keyword arguments.
  - Ufunc ``accumulate`` method with optional ``out``, ``axis`` keyword arguments.
  - Ufunc ``outer`` method with optional ``out``, ``where``, and ``dtype`` keyword arguments.
