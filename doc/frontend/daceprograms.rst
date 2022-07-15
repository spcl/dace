Writing DaCe Programs in Python
===============================

The entry point to DaCe is ``dace.program`` (aliases ``dace``, ``dace.function``). It is a function that wraps around
an existing Python function or method for parsing.

Usage
-----

You can use :func:`~dace.frontend.python.interface.program` either as a wrapper or a decorator:

.. code-block:: python

    import dace

    def myfunction(a, b):
        return a + b
    
    dfunc = dace.program(myfunction)

    @dace.program
    def other_function(c, d):
        return c * d + myfunction(c, d)

    # Calling ``dfunc`` or ``other_function`` will trigger DaCe compilation


The Python frontend will then try to parse the entire function, including internal function calls. What it can parse
becomes part of the :ref:`data-centric intermediate representation <sdfg>`, and what it cannot parse will be encapsulated
(best-effort) as callbacks to the Python interpreter. Callbacks are useful because they allow programs to use the full
power of the Python ecosystem. For example, if you wish to read a file, compute something expensive and plot the result
during computation, you can!

.. code-block:: python

    import dace
    import seaborn as sns  # For plotting
    import matplotlib.pyplot as plt

    @dace.program
    def expensive_computation(inputs: dace.float64[N, M, K]):
        ...
        for i in range(iterations):
            ...
            sns.histplot(intermediate_result)
            plt.show()
            ...
        ...


A warning is issued for any such unintended callback, just in case you wanted to compile that function too.

If you are using classes, methods need to use the ``@dace.method`` decorator (due to how Python function binding works).
Fields and globals work directly in the Python frontend:

.. code-block:: python

    import numpy as np

    class MyClass:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        @dace.method
        def method(self, z):
            return self.x * (self.y + z)

    a = np.random.rand(20)
    b = np.random.rand(20)
    obj = MyClass(2, a)
    c = obj.method(b)
    assert np.allclose(c, 2 * (a + b))



Ahead-Of-Time vs. Just-In-Time Compilation
------------------------------------------

when you just need to compile once (running on an FPGA or a supercomputer)
type hints - AOT vs. JIT

As opposed to other frameworks, you can use symbols to avoid recompilation for every size. DaCe has a powerful symbolic
engine (powered by `SymPy <https://www.sympy.org>`_) that can perform checks and analysis 


.. code-block:: python
    
    @dace.program
    def func(A: dace.float64[N, K], B: dace.float64[M, K]):
        C = A @ B    # NOT OK - will raise an error for mismatching dimensions
        C = A @ B.T  # OK
        ...


Compile-Time Arguments
~~~~~~~~~~~~~~~~~~~~~~
dace.compiletime

Parallelization Hints
---------------------

dace.map (see SDFG IR)

distributed programming: data distribution

Explicit Dataflow Mode
----------------------

.. note::
    use sparingly

with dace.tasklet


Calling SDFGs Directly
----------------------

If you want to call SDFGs from a ``@dace.program`` you can do so directly. This is useful when you have a custom implementation
or when you want to use another frontend (for example, in a domain-specific language, more details about how it exactly
works can be found in :ref:`dsl`). Example:

.. code-block:: python

    import dace

    mysdfg = dace.SDFG(...)
    # ...

    @dace.program
    def function(a, b, c):
        mysdfg(A=a, B=c)
