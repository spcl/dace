Using GPUs and Optimization Best Practices
==========================================

.. note::
    Experiencing errors or unintended behavior during GPU optimization? Refer to :ref:`gpu-debugging` for information
    on how to pinpoint the issue.

Before reading this document, read the `CUDA programming guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>`_ 
for more information on programmable GPU architectures.

How are SDFGs mapped to GPU programs?
-------------------------------------

To utilize GPUs, DaCe provides two basic elements:

    * Storage locations for arrays, scalars, and streams on the GPU: :class:`~dace.dtypes.StorageType.GPU_Global` for
      global memory on the GPU DRAM, and :class:`~dace.dtypes.StorageType.GPU_Shared` for shared memory. Note that 
      register storage works normally inside GPU kernels
    * Map schedules for running GPU kernels: :class:`~dace.dtypes.ScheduleType.GPU_Device` for a GPU kernel (grid) map,
      and :class:`~dace.dtypes.ScheduleType.GPU_ThreadBlock` for a map of threads in a single thread-block.

The :class:`~dace.transformation.interstate.gpu_transform_sdfg.GPUTransformSDFG` transformation takes an existing SDFG
and transforms it into a GPU program. Call it on an SDFG with :func:`~dace.sdfg.sdfg.SDFG.apply_gpu_transformations`.
The transformation will automatically detect GPU kernels and thread-blocks, and will make copies for all the relevant 
sub-arrays used to the GPU.

**Threads**: Each Map scope that has a ``GPU_Device`` schedule will create a GPU kernel call. The number of blocks, 
and threads in each block, are determined by the Map's parameters. The number of elements in a ``GPU_Device`` map will
create the thread-block grid. If there is no ``GPU_ThreadBlock`` map inside the ``GPU_Device`` map, the threads per block
will be defined from the map's :attr:`~dace.sdfg.nodes.Map.gpu_block_size` attribute, or the :envvar:`compiler.cuda.default_block_size`
if a block size is not given. If there is a ``GPU_ThreadBlock`` map, the block size is determined by the thread-block map's parameters.
If multiple thread-block maps are present, the maximum of their parameters will be the number of threads per block, and each
smaller map will have an ``if`` condition predicating a subset of the threads to work. This enables optimizing programs
via thread/warp specialization.

Some examples of Example of an SDFG **without** a GPU thread-block map and its generated code:

.. raw:: html

  <div class="figure align-right" id="scalarsym" style="width: 30%">
    <iframe width="100%" height="320" frameborder="0" src="../_static/embed.html?url=sdfg/gpu-notb.sdfg"></iframe>
  </div>


.. code-block:: cpp

    __global__ void example(const double * __restrict__ gpu_A, 
                            double * __restrict__ gpu_B, int N) {
        // i defines the thread index
        int i = (blockIdx.x * 64 + threadIdx.x);
        if (i < N) {
            double a = gpu_A[i];
            double b;
            b = (a + 1);  // <-- Tasklet code
            gpu_B[i] = b;
        }
    }
    // ...
    cudaLaunchKernel((void*)example, 
                     dim3(int_ceil(N, 64), 1, 1), // Grid size
                     dim3(64, 1, 1),              // Block size
                     example_25_0_0_2_args, 0,
                     __state->gpu_context->streams[0]);



Example of an SDFG **with** a GPU thread-block map and its generated code:

.. raw:: html

  <div class="figure align-right" id="scalarsym" style="width: 30%">
    <iframe width="100%" height="360" frameborder="0" src="../_static/embed.html?url=sdfg/gpu-tb.sdfg"></iframe>
  </div>


.. code-block:: cpp

    __global__ void example(const double * __restrict__ gpu_A, 
                            double * __restrict__ gpu_B, int N) {
        // i defines the block index 
        // It is multiplied by 32 because of the map range
        int i = (32 * blockIdx.x);

        // j defines the thread index
        int j = threadIdx.x;
        double a = gpu_A[((32 * i) + j)];
        double b;
        b = (a + 1);  // <-- Tasklet code
        gpu_B[((32 * i) + j)] = b;
    }
    // ...
    cudaLaunchKernel((void*)example, 
                     dim3(int_ceil(N, 32), 1, 1), // Grid size
                     dim3(32, 1, 1),              // Block size
                     example_25_0_0_2_args, 0,
                     __state->gpu_context->streams[0]);


**Memory**: ``GPU_Global`` memory can be read and written to from inside the kernel, but is usually defined outside the
GPU maps. Copies from host to/from GPU is done by a memlet between a ``GPU_Global`` array and a ``CPU_Heap`` array. 
``GPU_Shared`` memory is allocated inside the kernel, and is only accessible from inside the kernel. An error will be
triggered if a ``GPU_Shared`` array is accessed from outside the kernel. The ``CPU_Pinned`` storage type is used for
host memory that is pinned for GPU access, and can be accessed from within a kernel (albeit much slower than a GPU array).

**Collaborative Copies**: If there exists a ``GPU_ThreadBlock`` map, and a ``GPU_Shared`` array is copied to/from a 
``GPU_Global`` array, the copy will be done collaboratively across all threads in the block. This requires that the access node
of the shared array will be outside the thread-block map (such that the copy is performed by the entire block).

**Streams**: Streams are used to overlap computation and data transfer. During code generation, DaCe will automatically
infer the streams that are needed for each kernel, and will create them.

**Synchronization**: DaCe will automatically insert synchronization points between SDFG states, and between thread-block
maps inside a kernel. This is a natural interpretation of the SDFG semantics, based on the fact that closing a map implies
synchronization across the map's scope. This behavior can be overridden by setting the ``nosync`` property in a state.
The number of GPU streams can be controlled with the :envvar:`compiler.cuda.max_concurrent_streams` configuration entry.
It is set to zero by default, which does not limit streams. If set to ``-1``, no streams will be created (the default
stream will be used). This is sometimes preferable for performance.

.. _amd:

Using AMD GPUs
--------------

AMD GPUs are supported in the same way as NVIDIA GPUs. By default, DaCe is set to autodetect which GPU is connected to the current system.
If this fails, or you have both and want to use AMD, the target should be changed from ``auto`` to
``cuda`` or ``hip``. The default block size would also be suboptimal, as AMD GPUs have a wavefront size of 64 rather than 32.

To run a program on an AMD GPU, you can configure the ``.dace.conf`` file and change the appropriate, but optional,
settings. For example:

.. code-block:: yaml

    compiler:
      cuda:
        # Change the backend to HIP (optional)
        backend: hip

        # Specify the AMD GPU architecture (optional)
        hip_arch: 'gfx906'

        # Override default block size (optional but important due to warp/wavefront size)
        default_block_size: 64,1,1

        # Override default HIP arguments (optional)
        hip_args: -std=c++17 -fPIC -O3 -ffast-math -Wno-unused-parameter


Subsequently, any GPU DaCe program will use HIP.

Note that if you are using CuPy, install its appropriate HIP/ROCm version.

.. note::
    Not every CUDA feature is directly supported by HIP. 
    Refer to the `HIP documentation <https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html>`_ for more information.
    If compilation fails, try to :ref:`manually edit the source code and recompile <recompilation>`,
    or use the HIP-provided tools to convert CUDA code to HIP code without changing the backend.
    If you find a feature that is not supported in DaCe, please open an issue on GitHub.
    

Optimizing GPU SDFGs
--------------------

When optimizing GPU SDFGs, there are a few things to keep in mind. Below is a non-exhaustive list of common GPU optimization
practices and how DaCe helps achieve them. To see some of these optimizations in action, check out the ``optimize_for_gpu``
function in the `Matrix Multiplication optimization example <https://github.com/spcl/dace/blob/master/samples/optimization/matmul.py>`_.

    * **Minimize host<->GPU transfers**: It is important to keep as much data as possible on the GPU across the application.
      This is especially true for data that is accessed frequently, such as data that is used in a loop.
      Copies to/from the GPU are generally much slower than the GPU computation itself.
      ``GPUTransformSDFG`` will automatically try to keep the arrays on the GPU for as long as possible, but it is not
      guaranteed. If you have a large array that can be stored on the GPU before the SDFG starts, you can use the
      ``GPU_Global`` storage type to store it on the GPU. This will prevent the array from being copied to the GPU.

    * **GPU kernel granularity**: GPU kernels cannot be too small nor arbitrarily large. On the one hand, kernels have
      a launch overhead and will not typically run shorter than 2-4 microseconds. On the other hand, an entire program
      cannot be represented by a single GPU kernel, as registers are limited and will spill. The optimal kernel size
      depends on the GPU architecture, but a good rule of thumb is to fuse small operations as much as possible into a
      single kernel (or run them on the host). If you have a large kernel, you can try to split it into multiple kernels
      using tools such as the :class:`~dace.transformation.dataflow.map_fission.MapFission` transformation.

    * **Atomic write-conflicts**: If you have multiple threads writing to the same memory location with a write-conflict
      resolution memlet, the code generator will emit atomic operations, which can be very slow.
      If you have a large number of threads writing to the same memory location, you can try to reduce the number of
      threads by re-scheduling the maps to make the writes conflict-free, or fuse multiple threads into a single thread.
      If the conflict is necessary, transformations such as :class:`~dace.transformation.dataflow.stream_transient.AccumulateTransient`
      will create local storage and reduce the number of atomic operations. Another approach is to use warp tiling via
      the :class:`~dace.transformation.dataflow.warp_tiling.WarpTiling` transformation, which will convert them to fast
      warp-level instructions.

    * **Block size**: The maximum number of threads per block is limited by the GPU architecture (usually to 1024 threads),
      if you have a kernel that uses more threads than the maximum, or allocates too much shared memory, you will get an
      error. The optimal block size depends on the degree of reuse that can be achieved by the threads in the block.
      Try to use transformations such as :class:`~dace.transformation.dataflow.tiling.MapTiling` to increase the
      work performed per thread, or move map dimensions to the ``GPU_Device`` map in order to make them part of the 
      thread-block grid, which is more accommodating.

    * **Kernel fusion and data movement**: Moving data between registers and global memory is very expensive. Try to
      reduce the amount of data that is moved between the two, even at the expense of recomputing values. Using transformations
      such as :class:`~dace.transformation.subgraph.subgraph_fusion.SubgraphFusion` to fuse without recomputation, or
      :class:`~dace.transformation.dataflow.otf_map_fusion.OTFMapFusion` for on-the-fly recomputation, usually helps.
      For data that is reused within a thread-block or a thread, use :class:`~dace.transformation.dataflow.tiling.MapTiling`
      combined with :class:`~dace.transformation.dataflow.local_storage.InLocalStorage` to block and cache the data.

    * **Persistent memory allocation**: If you have a transient array, by default DaCe will allocate it within the program.
      As the calls to ``cudaMalloc`` and ``cudaFree`` are expensive, it is better to allocate the array once and reuse it
      across the program (or multiple invocations). Use the :class:`~dace.dtypes.AllocationLifetime.Persistent` lifetime
      to allocate such arrays once. Note that this can only be used for arrays whose size is known at initialization time
      (e.g., constant or dependent on input symbols).

    * **Memory footprint reduction**: Passes such as :class:`~dace.transformation.passes.transient_reuse.TransientReuse`
      can help reduce the amount of bytes allocated by the SDFG. For dynamic memory reuse, use memory pooling by setting
      the ``pool`` attribute of a data descriptor to ``True``.

    * **Stream and synchronization overhead**: For mostly sequential programs, disabling concurrent GPU streams (see above)
      may help performance. The synchronization between states inside GPU kernels and between thread-block maps can similarly
      be disabled (if you know what you are doing) with the ``nosync`` property.

    * **Global memory access**: Try to keep memory accesses structured and coalesced across threads in a warp. It is also
      mostly better (if a thread is working on multiple elements) to use wide loads and stores of 128-bits. You can 
      verify that your loads/stores are structured and wide using the ``cuobjdump`` tool on the compiled SDFG (for example
      ``cuobjdump -sass .dacecache/<sdfg name>/build/lib<sdfg name>.so``). It is also important to keep **local loads/stores**
      (``LDL.*`` and ``STL.*`` instructions) to a minimum, as they are often a sign that registers were spilled onto local
      memory, which is much slower to access. You can ensure wide loads and stores are used with the 
      :class:`~dace.transformation.dataflow.vectorization.Vectorization` transformation, and reschedule the division of
      work to threads to reduce register pressure.

    * **Specialized hardware**: Specialized hardware, such as NVIDIA Tensor Cores or AMD's matrix instructions, can
      significantly improve performance. DaCe will not automatically emit such instructions, but you can use such operations
      in your code. See the `Tensor Core code sample <https://github.com/spcl/dace/blob/master/samples/codegen/tensor_cores.py>`_ 
      to see how to make use of such units.

    * **Advanced GPU Map schedules**: DaCe provides two additional built-in map schedules: :class:`~dace.dtypes.ScheduleType.GPU_ThreadBlock_Dynamic`
      and :class:`~dace.dtypes.ScheduleType.GPU_Persistent`. The former is useful for grids that have a varying number
      of work per thread-block (for example, in graph traversal and sparse computation), as it generates a dynamically 
      load-balanced thread-block schedule. The latter is useful for *persistent kernels*, where the same thread-blocks
      are kept alive and synchronized on the grid level without ending the kernel. You can set those schedules
      on the maps based on the workload characteristics.
