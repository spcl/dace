.. _dsl:

Python DSL Support via SDFG-Convertible Objects
===============================================

Many domain-specific languages (DSLs) embedded in Python are themselves
data-centric and can be lowered to an SDFG. DaCe exposes a small protocol,
*SDFG-convertible objects*, which lets such DSLs participate transparently in
``@dace.program`` parsing without having to be rewritten as DaCe replacements.
Examples of frameworks that integrate with DaCe through this protocol include:

* the built-in ``@dace.program`` and ``@dace.method`` decorators (a
  ``@dace.program`` is itself convertible, which is why one DaCe program
  can call another);
* ``dace.ml.torch.module.DaceModule``, which exposes PyTorch
  ``nn.Module`` instances as SDFG-convertible objects so they can be
  invoked from any ``@dace.program``;
* `GT4Py <https://github.com/GridTools/gt4py>`_ stencils, whose backends
  produce SDFGs and register stencil objects as convertibles.

The ``SDFGConvertible`` protocol
--------------------------------

The protocol is defined in :mod:`dace.frontend.python.common` as the abstract
class :class:`~dace.frontend.python.common.SDFGConvertible`. Any object that
appears in the closure of a ``@dace.program`` and implements (some of) the
methods below is treated as a callable SDFG by the Python frontend.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Purpose
   * - ``__sdfg__(*args, **kwargs)``
     - Return the SDFG that should be invoked at the call site. The
       arguments are the same Python values that the caller passed in;
       implementations typically use them to specialize shapes,
       data types, or compile-time options before generating the SDFG.
       This is the minimum a convertible must implement.
   * - ``__sdfg_signature__()``
     - Return ``(arg_names, constant_args)``, describing the positional
       parameter names of the produced SDFG and the names whose values
       must be constant at parse time. The frontend uses this to bind the
       caller's arguments to the convertible's SDFG arguments.
   * - ``__sdfg_closure__(reevaluate=None)``
     - Return a dictionary of additional values (arrays, scalars, callbacks,
       nested convertibles) that should be merged into the parent program's
       closure. ``reevaluate`` is a list of names that the caller wants
       refreshed - implementations that cache arrays should re-read them
       in that case.
   * - ``closure_resolver(constant_args, given_args, parent_closure=None)``
     - Optional. Return an :class:`~dace.frontend.python.parser.SDFGClosure`
       built from the convertible's own captured state. Implementations
       that maintain a stateful closure (e.g., a neural network's weights)
       use this hook to wire those values into the SDFG.

Only ``__sdfg__`` and ``__sdfg_signature__`` are required for a basic
convertible. The other methods exist to expose stateful captures cleanly to
the parent program's closure.

Minimal example
~~~~~~~~~~~~~~~

The following stub shows a callable Python class that participates in
``@dace.program`` parsing as if it were itself a DaCe program:

.. code-block:: python

    import dace

    class MyOperator:
        def __init__(self, scale: float):
            self.scale = scale

        def __sdfg__(self, A):
            sdfg = dace.SDFG('myop')
            sdfg.add_array('A', A.shape, A.dtype)
            state = sdfg.add_state()
            ...
            return sdfg

        def __sdfg_signature__(self):
            return (['A'], [])

    op = MyOperator(scale=2.0)

    @dace.program
    def use(A: dace.float32[16]):
        op(A)             # parsed as a nested SDFG inside `use`

When ``use`` is parsed, the Python frontend recognizes ``op`` in the closure
as an :class:`~dace.frontend.python.common.SDFGConvertible`, calls
``op.__sdfg__(A)`` to obtain the operator's SDFG, and inlines it as a nested
SDFG inside ``use``.

Caveats and recommendations
---------------------------

* ``__sdfg__`` is called every time the parent program is parsed. If
  generating the SDFG is expensive, cache it on the object and key the cache
  on the relevant compile-time arguments.
* The returned SDFG must be self-contained - any state it depends on at
  runtime must either be passed through arguments or exposed via
  ``__sdfg_closure__`` / ``closure_resolver``.
* For DSLs that produce many SDFGs, consider returning a small SDFG that
  delegates to a library node; this keeps the parent program's IR
  navigable and allows the DSL to ship its own expansions.
* If your DSL already has a dedicated frontend (e.g., it parses its own
  AST), see :doc:`frontend` for guidelines on writing a separate frontend
  pipeline rather than extending the Python frontend.
