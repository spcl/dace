Optimizing Programs in DaCe
===========================


Transforming a program is often a sequence of operations.
Some transformations do not necessarily improve the performance of an SDFG, but are a "stepping stone" for other
transformations, for example :class:`~dace.transformation.dataflow.tiling.MapTiling` on a map can lead to 
:class:`~dace.transformation.dataflow.local_storage.InLocalStorage` being available on the memlets.

If you are looking for a starting point, the experimental :ref:`automatic optimization heuristics <opt_auto>` can be
useful to provide a better-performing (but likely not optimal) SDFG.


.. toctree::
    :maxdepth: 1
    
    profiling
    interactive
    vscode
    guidelines
    gpu
    fpga

