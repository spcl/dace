Extending DaCe
==============

The DaCe framework is designed to be user-extensible. Developers can externally extend DaCe by
adding new types of nodes, frontend languages, transformations, and hardware architecture backends.

The three key mechanisms of extensibility are class inheritance, :ref:`replacements <replacements>`, and :ref:`enums`.

   * Class inheritance is used for new :ref:`library nodes <libnodes>`, :ref:`data types <typeclasses>`, 
     :ref:`transformations <transformations>`, :ref:`passes <pass>`, :ref:`code generator targets <codegen>`, 
     :ref:`instrumentation providers <instrumentation>`, and others.
   * Replacements are used to extend the :ref:`frontends <python-frontend>` with new language constructs and specialized library
     functions (e.g., a `custom implementation <https://github.com/spcl/dace/blob/7cf31b318e54d5798ded29cbadcbcaf232b67282/dace/frontend/python/replacements.py#L234>`_ for ``numpy.eye``).
   * Enumerations can be extended to add new entries to device types, :ref:`storage locations <descriptors>`, and 
     others. See the enumerations in :mod:`dace.dtypes` for more examples.


For more examples of how to extend DaCe, see the following resources:

   * Library nodes: `Einsum specialization library node <https://github.com/spcl/dace/blob/master/dace/libraries/blas/nodes/einsum.py>`_
   * Transformations: `Using and Creating Transformations <https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/transformations.ipynb>`_
   * Code generators: `Extending the Code Generator <https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/codegen.ipynb>`_
   * Frontend extensions (enumerations and replacements): `Tensor Core code sample <https://github.com/spcl/dace/blob/master/samples/codegen/tensor_cores.py>`_

.. .. toctree
..    :maxdepth: 1

..    .. symbolic
..    .. libraries
..    .. frontend
..    .. sdfgconvertible
..    .. backend

