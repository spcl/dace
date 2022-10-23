.. _transforming:

SDFG Transformation Internals
=============================

Once created, stateful dataflow multigraphs can be transformed for performance, offloading to different architectures, 
and reducing elements for simplification and further transformations.

As a white-box approach, SDFG transformations can be written externally, and can be applied on graphs programmatically or 
interactively, through built-in passes, the transformation Python API, or the `Visual Studio Code <https://github.com/spcl/dace-vscode>`_
plugin.

Below you can find the different ways transformations are represented in DaCe and how they are used in the framework.

.. toctree::
   :maxdepth: 1

   passes
   transformations
   simplify
   auto_optimize
