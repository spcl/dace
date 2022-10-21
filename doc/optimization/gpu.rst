GPU Optimization Best Practices
===============================

.. note::
    Experiencing errors or unintended behavior during GPU optimization? Refer to :ref:`gpu-debugging` for information
    on how to pinpoint the issue.


fusion and auto-optimize

memory allocation (shared, registers, global). lifetime (persistent), memory pooling

block size (+ block sizes that are too large will not run, tile your map)

disabling streams

WCR and atomics

coalescing

warp tiling

wide loads/stores (verify with ``cuobjdump``)

syncs at the end of every state and nosync

Using tensor cores - refer to tutorial


.. _amd:

Using AMD GPUs
--------------

AMD GPUs are supported in the same way as NVIDIA GPUs. The only difference is that the target should be changed from 
``cuda`` to ``hip``. To run a program on an AMD GPU, you can configure the ``.dace.conf`` file and change the appropriate
settings. For example:

.. code-block:: yaml

    compiler:
      cuda:
        # Change the backend from CUDA to HIP
        backend: hip

        # Specify the AMD GPU architecture (optional)
        hip_arch: '906'

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
    

