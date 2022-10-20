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

TODO: Configure HIP
install the HIP version of CuPy

if code is misbehaving see how to modify code and rerun in codegen.