GPU Optimization Best Practices
===============================

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
