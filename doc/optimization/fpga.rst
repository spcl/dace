FPGA Optimization Best Practices
================================

.. note::

   This document is a work in progress. Feel free to make any contributions or suggestions via Pull Requests.


This section provides guidance on leveraging DaCe functionalities to optimize DaCe programs targeting FPGAs.
Once the user program is parsed into an SDFG, we can optimize (transform) it to improve performance. In the case of FPGA programs, 
the user can apply transformations or follow best practices to reduce data movements, specialize operations implementation, and
increase spatial parallelism. 
In the following we start by presenting automatic transformations specifically helpful for FPGA programs. Then we discuss how to specialize
library node implementation. Finally we show how to control various low-level aspects, such as Maps scheduling and Memory Hierarchy.


.. _fpga_transformations:

Transformations for FPGA programs
---------------------------------

.. TODO: Structure this slightly differently (don't assume the user knows anything). Show an example of apply_fpga_transformation, 
.. and dedicate subsubsections for transformation types (streaming transformations, memory layout transformations) instead of just simple bullet points. 

Existing SDFGs can be transformed from a generic to an FPGA implementation using graph transformations. 
The resulting SDFGs can be can be further optimized using general-purpose transformations available in the DaCe toolbox. 
This includes platform-agnostic transformations (such as Trivial Map Elimination, Map Collapsing, Map tiling, ...) and more 
FPGA-oriented transformations, which we describe here.

* :py:func:`~dace.transformation.interstate.fpga_transform_sdfg.FPGATransformSDFG`: programmers can automatically offload a full
  SDFG using this transformation. This takes care of creating create additional pre- and post-states performing memory transfers 
  between host and device. The memories accessed by the transformed subgraph are replaced with their FPGA equivalents.
* :py:func:`~dace.transformation.dataflow.streaming_memory.StreamingMemory`: this transformation enables the automatic creation of 
  streaming memory accessors (see :ref:`fpga_streams`). The transformation analyzes accesses to data containers. If applicable,
  it converts an existing memory access to a streaming memory access: the data is read/written to/from a stream in a separate connected 
  component than the computation. If the `use_memory_buffering` option is set to ``True``, the transformation enables burst reads/write form/to memory, by
  using a wider data format (e.g., 512 bits), and then convert it on the fly to the right data type used by the computation.
* :py:func:`~dace.transformation.dataflow.streaming_memory.StreamingComposition`: in unoptimized SDFGs, intermediate data occuring between two consecutive computations
  is represented as data access nodes, pointing to off-chip memory by default. This off-chip accesses are undesirable, and in certain conditions can be completely avoided.
  This transformation converts two connected computations (nodes, map scopes) into two separate processing elements, with a stream connecting the results. 
  The transformation performs checks similar to the previous one, and applyes only if the memory access patterns of the two computations match.
* :py:func:`~dace.transformation.auto.fpga.fpga_global_to_local`: changes the storage of containers allocated in global memory to local memory when this is possible.
* :py:func:`~dace.transformation.auto.fpga.fpga_rr_interleave_containers_to_banks`: interleaved global memory containers on the available off-chip memory banks.
  Containers are allocated in a Round-Robin fashion.


Library Nodes and FPGA specialization
-------------------------------------

Library nodes are high-level nodes that represent specific functions (e.g., matrix multiplication). During compilation and optimization, 
Library Nodes are *expanded* by replacing them with a subgraph, *lowering* them towards a concrete
implementation of their behavior.

..  TODO: add links to the library node (rather than mention their name). For this, we need to enable their docs

Available FPGA expansions
^^^^^^^^^^^^^^^^^^^^^^^^^
DaCe provides FPGA-specific expansions for the principal numerical linear algebra or common operations:

* vector dot product (``dot``) can be specialized for FPGA using two expansions:  ``FPGA_Accumulate`` and ``FPGA_PartialSums``. The former assumes that 
  native single clock cycle accumulation of the data type is possible on the target architecture (e.g., 32-bit floating 
  point on Intel Stratix 10). The latter does not assume that native accumulation of the data type is possible. 
  Both expansions achieve an Initiation Interval of 1.
* matrix-vector multiplication (``gemv``) is available in two versions:
  
  * ``FPGA_Accumulate``: this FPGA-oriented expansion iterates over the input matrix in simple row-major order, with optional 
    tiling in both dimensions, where the tiles are also traversed in simple row-major order.
  * ``FPGA_TilesByColumn``: this expansion reads the input matrix in column-major order, such that consecutive values are accumulated into different
    registers. The matrix can optionally be tiled, where the tiles will be traversed in row-major order.

  These two expansions complement each other as they can be used to favor composability (pipeline-ability) with the rest of the computation.
  For example, if another library node produces the input matrix by row, it makes sense to use the first expansion so that the matrix values 
  can be streamed directly.
* outer product (``ger``) can be expanded for FPGA using the ``FPGA`` expansion. Input vectors can be optionally tiled.
* matrix-matrix multiplication(``gemm``) FPGA specialization is implemented by the ``FPGA1DSystolic`` expansion. This implements the matrix-matrix
  multiplication (with accumulation) using a 1D systolic array. The matrices can optionally be tiled along the result columns. 
  The user can specify the number of used processing elements and tile size according to her needs.
* Reduction library nodes can be inserted by the frontend. They "reduce" an array according to a binary operation (e.g., sum, max), starting 
  with initial value identity, over the given axis. Reductions can be specialized for FPGAs using the ``FPGAPartialReduction`` expansion.


How to specialized library node expansions for FPGA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users can target FPGA expansions either through VSCode plugin, or programmatically.
In the VSCode plugin, the user can select for each library node the desired expansion and apply it.

To do this programmatically, the user has two options:

* expand specific library nodes. This can be done by choosing the implementation, and manually expand it:
  
  .. code-block:: python

    # Get the library node that we want to expand, e.g., a gemv node
    gemv_node = ... 

    # Set the desired expansion, e.g., "FPGA_Accumulate"
    gemv_node.implementation = "FPGA_Accumulate"

    # Expand it by passing the SDFG and state that contains it together
    # with expansion arguments (if any).
    # For example, in this case we specify a tile size of 1024 x 1024 elements
    expansion_args = {
      "tile_size_x": 1024,
      "tile_size_y": 1024
    }
    gemv_node.expand(sdfg, state, **expansion_args)
    
* set a default expansion for all the library nodes of a given type:

  .. code-block:: python
    
    # Set a default expansion for all GEMM library node
    from dace.libraries.blas import Gemm
    Gemm.default_implementation = "FPGA1DSystolic"


Vectorization
-------------------------------------
TBD

Maps and parallelism
--------------------

In DaCe maps are used to express parallel scopes in SDFGs.
In the context of FPGAs, we distinguish between:

* *pipelined* maps, where iterations are executed in sequence, exploiting pipeline parallelism in the mapped computation; 
* *unrolled* maps, which represent parametrically replicated hardware, such as systolic arrays or SIMD-style vectorization.

By default, maps are code-generated as pipelined loops. The user can switch to unrolled maps by changing their schedule (either
programmatically or through the VSCode plugin). For pipelined maps, the schedule must be set to :py:data:`~dace.dtypes.ScheduleType.Default`, while
for unrolled maps it must be set to :py:data:`~dace.dtypes.ScheduleType.Unrolled`.

.. TODO: add a simple illustrative figure (or a snippet of generated code) -- probably it is better to add both of them

FPGA memory hierarchy
-----------------------------

Modern FPGAs are characterized by having small, fast on-chip memory and large, but slower, off-chip memory.

DaCe allows to specify for each FPGA container, where it should be allocated by specifying its :py:data:`~dace.dtypes.StorageType`, either programmatically
or through the VSCode plugin. We can distinguish between:

* *global* memory (:py:data:`~dace.dtypes.StorageType.FPGA_Global`), which represents data present in off-chip, memory-mapped storage such as DDR or HBM. 
  Containers in global memory can be created/accessed from both the host and the device side;
* *local* memory (:py:data:`~dace.dtypes.StorageType.FPGA_Local`), representing any on-chip memory implementation such as registers, BRAM/M20K, 
  LUTRAM, or UltraRAM. Which one will be actually used is left up to the HLS compiler;
* *register* memory (:py:data:`~dace.dtypes.StorageType.FPGA_Register`), which is a subset of local memory, but forces the compiler to implement it 
  as register (LUT), allowing parallel read/write to the container. This can be useful in the presence of unrolled maps.


.. TODO: also introduce Shift Register

.. _fpga_streams:

Streams and how to exploit them
-------------------------------
In DaCe, stream containers represent single or multidimensional arrays of First-In-First-Out (FIFO) queues (see :ref:`descriptors`).

In FPGAs, they are implemented in hardware (FIFOs) either using BRAM or registers. This implies that streams
cannot be unbounded and must be single-producer, single-consumer. 

Streams can be particularly useful in FPGA programs as:

* they facilitate the division of the program logic in processing elements. The different processing elements can be
  simultaneously in execution while communicating using fast on-chip resources, reducing more expensive off-chip memory
  accesses;
* they allow memory access extraction, enabling compute and memory accesses to be pipelined and optimized separately. 
  Creating streaming accessors has many benefits, including using burst mode in memory controllers, tailored buffering,
  or broadcasting off-chip memory to multiple processing elements.


While these opportunities can be exploited by carefully designing the SDFG, 
DaCe also provides transformations to automatically enabling them (see :ref:`fpga_transformations`).

.. TODO: add sample code




FPGA kernels and processing elements
------------------------------------

.. TODO: this is part of the general info (schedule, storage, dataflow structure)
.. an embedded SDFG example would go a long way

In DaCe, a state that only accesses containers situated on the FPGA will trigger FPGA code generation.

In DaCe, we hierarchically organize the code in *FPGA Kernels*, which can be further divided into multiple *Processing elements*.
These concepts will be mapped to different entities depending on the used FPGA backend (see :ref:`Code generating FPGA kernels and processing elements <codegen_fpga_kernels>`).



By default, an SDFG state with only FPGA containers is inferred as an FPGA kernel. Then, each of the weakly connected component
found in the state are treated as different Processing Elements, that can be executed in parallel.
The notion of partitioning the functionality of a kernel into multiple independently-scheduled modules is central to designing large FPGA architectures, and can be exploited to write systolic arrays.

If the :envvar:`compiler.fpga.concurrent_kernel_detection` configuration option is set to ``True``, 
a heuristic will further inspect each independent component for other parallelism opportunities (e.g., branches of the SDFG
that can be executed in parallel). If this is the case, multiple, possibly depending, FPGA Kernels are generated for the same state.



