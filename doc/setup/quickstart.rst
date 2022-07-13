Quick-Start
===========

Follow the :ref:`installation` guide, or simply install the latest version of DaCe with pip via ``pip install dace``

.. note::
    Having issues? See :ref:`troubleshooting`.


Using DaCe in Python is as simple as adding a ``@dace`` decorator:

.. code-block:: python

    import dace
    import numpy as np

    @dace
    def myprogram(a):
        for i in range(a.shape[0]):
            a[i] += i
        return np.sum(a)


Calling ``myprogram`` with any NumPy array should return the same result as Python would, but compile the program with
DaCe under the hood.

.. note::
    GPU arrays that support the ``__cuda_array_interface__`` interface (e.g., PyTorch, Numba, CuPy) also
    work out of the box.

Internally, DaCe creates a shared library (DLL/SO file) that can readily 
be used in any C ABI compatible language, such as C++ or FORTRAN (See :ref:`integration`).

From here on out, you can optimize (:ref:`interactively <vscode>`, :ref:`programmatically <opt_sdfgapi>`, or
:ref:`automatically <opt_auto>`), :ref:`instrument <profiling>`, and distribute
your code. 


For more examples of how to use DaCe, see the `samples <https://github.com/spcl/dace/tree/master/samples>`_ and 
`tutorials <https://github.com/spcl/dace/tree/master/tutorials>`_ folders on GitHub.
