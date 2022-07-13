.. _codegen:

Code Generation
===============

File structure (framecode etc.)

How it Works
------------

How it works (accompany gif + links to modules)
    * preprocess
    * CFG
    * framecode
    * allocation management + decisions regarding AllocationLifetime
    * defined variables
    * traversal + delegation (dispatching with example)
    * concurrency (e.g., OMP sections/tasking, GPU streams)

Link to Tensor Core tutorial

.. image:: codegen.gif

Important features


.. _debug_codegen:

Debugging Code Generation
-------------------------

conditional breakpoints

codegen lineinfo config


FPGA Code Generation
--------------------

Modules / kernels + illustration

Memory interfaces (in/out)
