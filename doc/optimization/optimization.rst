.. _optimization:

Optimizing Programs
===================

Once created, SDFGs can be optimized by a variety of means. 

TODO explain analysis via profiling

The most common is to use local transformations. 

For example, the :class:`~dace.transformation.interstate.sdfg_nesting.InlineSDFG` transformation inlines a nested SDFG
into its parent SDFG. This transformation is applied to a single SDFG, 
and does not require any information from other SDFGs.
it's an iterative process

.. image:: images/overview.png

TODO explain

TODO: Intro to what transformations are

Transforming a program is often a sequence of operations.
Some transformations do not necessarily improve the performance of an SDFG, but are a "stepping stone" for other
transformations, for example :class:`~dace.transformation.dataflow.tiling.MapTiling` on a map can lead to 
:class:`~dace.transformation.dataflow.local_storage.InLocalStorage` being available on the memlets.

If you are looking for a starting point, the experimental :ref:`automatic optimization heuristics <opt_auto>` can be
useful to provide a better-performing (but likely not optimal) SDFG.

The following resources are available to help you optimize your SDFG:

  * Using transformations: `Using and Creating Transformations <https://nbviewer.org/github/spcl/dace/blob/master/tutorials/transformations.ipynb>`
  * Matrix multiplication CPU and GPU optimization example: :ref:`optimizing the matrix multiplication example <optimizing_matrix_mult>`
  * Tuning data layouts: :ref:`tuning data layouts <tuning_data_layouts>`


.. toctree::
    :maxdepth: 1
    
    profiling
    blas
    vscode
    gpu
    fpga


.. interactive
.. guidelines
    
