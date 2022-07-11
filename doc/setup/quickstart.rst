Quick-Start
===========

Follow the :ref:`installation` guide, or simply install the latest version of DaCe with pip: `pip install dace`

.. note::
    Having issues? See :ref:`troubleshooting`


Using DaCe in Python is as simple as adding a `@dace` decorator:
```python
import dace
import numpy as np

@dace
def myprogram(a):
    for i in range(a.shape[0]):
        a[i] += i
    return np.sum(a)
```

Calling `myprogram` with any NumPy array or 
`__{cuda_}array_interface__`-supporting tensor (e.g., PyTorch, Numba) will 
generate data-centric code, compile, and run it. 

From here on out, you can 
_optimize_ (interactively or automatically), _instrument_, and _distribute_ 
your code. The code creates a shared library (DLL/SO file) that can readily 
be used in any C ABI compatible language (C/C++, FORTRAN, etc.).




For more information on how to use DaCe, see the [samples](samples) or tutorials below:

* [Getting Started](https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/getting_started.ipynb)
* [Benchmarks, Instrumentation, and Performance Comparison with Other Python Compilers](https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/benchmarking.ipynb)
* [Explicit Dataflow in Python](https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/explicit.ipynb)
* [NumPy API Reference](https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/numpy_frontend.ipynb)
* [SDFG API](https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/sdfg_api.ipynb)
* [Using and Creating Transformations](https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/transformations.ipynb)
* [Extending the Code Generator](https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/codegen.ipynb)

