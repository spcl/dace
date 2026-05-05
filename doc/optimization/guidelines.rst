.. _opt_guidelines:

Optimization Guidelines
=======================

This page collects the high-level principles that underlie effective DaCe
optimization. The other pages in :ref:`optimization` show *how* to apply
specific transformations; this one is about *what* to aim for.

Minimize data movement
----------------------

Data movement - between memory hierarchy levels, between devices, and
across processes - dominates the cost of most modern workloads. The single
most impactful optimization in DaCe is therefore to remove or shorten
memory transfers. In practice this means:

* **Fuse states and maps** so that producers and consumers of the same data
  share a scope. Map fusion (and the broader subgraph fusion family of
  transformations) eliminates the round-trip through main memory between
  fused nodes.
* **Stage data in fast memory** with
  :class:`~dace.transformation.dataflow.local_storage.InLocalStorage`,
  :class:`~dace.transformation.dataflow.local_storage.AccumulateTransient`,
  and tiling transformations. Once a tile is in shared/registers, reuse it
  for as many computations as possible before evicting.
* **Avoid redundant transients** by running the
  :class:`~dace.transformation.passes.transient_reuse.TransientReuse` pass.
  Transients that are not reused inflate working sets without speeding up
  the program.
* **Specialize storage location**. Set the
  :attr:`~dace.data.Data.storage` attribute of arrays to match where they
  are produced and consumed; the codegen will emit the right allocation
  and copy logic.

Expose parallelism, then specialize the schedule
------------------------------------------------

DaCe's IR separates *what* is parallel (maps, library nodes) from *how* it
is executed (schedules, storage, library implementations). The recommended
order of operations is:

1. **Expose** all the parallelism in the program. Use the Python frontend's
   :func:`dace.map` syntax or transformations like
   :class:`~dace.transformation.interstate.loop_to_map.LoopToMap`
   and :class:`~dace.transformation.dataflow.map_collapse.MapCollapse` to
   produce maximally-parallel loop nests.
2. **Specialize** the schedule to the target platform - use transformations
   such as :class:`~dace.transformation.interstate.gpu_transform_sdfg.GPUTransformSDFG`
   or choose a schedule manually (:attr:`~dace.dtypes.ScheduleType.GPU_Device` /
   :attr:`~dace.dtypes.ScheduleType.GPU_ThreadBlock` schedules for CUDA,
   :attr:`~dace.dtypes.ScheduleType.CPU_Multicore` for OpenMP, etc.)
3. **Pick library implementations** for any matrix multiplications, FFTs,
   reductions, or collectives. Library nodes default to a portable
   ``"pure"`` expansion; switching to ``"MKL"``, ``"cuBLAS"``,
   ``"NCCL"``, or another vendor implementation often delivers an
   order-of-magnitude speedup. See :ref:`blas`.

Profile, do not guess
---------------------

DaCe makes performance measurement cheap. Use it before deciding where to
optimize:

* :ref:`profiling` shows how to enable instrumentation on selected SDFG
  elements and how to interpret the resulting reports.
* :ref:`optimization_vscode` overlays runtime measurements onto the SDFG
  view so that hot regions are visually obvious.
* :ref:`opt_sdfgapi` describes how to drive the optimization loop from a
  notebook, applying transformations between profiling runs.

A good rule of thumb: the earliest and largest speedups typically come from
repeatedly looking for the "hottest" region and applying one or
two transformations to improve its scheduling.

When in doubt, start with the auto-optimizer
--------------------------------------------

The :ref:`automatic heuristics <opt_auto>` are a strong baseline on most
workloads. Calling ``auto_optimize`` once before doing manual optimization
saves time and makes it easier to tell whether a manual change is actually
helping. If the baseline is already close to peak (verifiable through
profiling and a roofline-style calculation), it is often more productive
to switch to algorithmic improvements than to keep tuning the schedule.
