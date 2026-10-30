# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loop parallelization: :class:`~.parallelize_loops.ParallelizeLoops` and its subpasses.

* ``parallelize_loops`` -- the driver: the probe (LoopToMap plus array facts), :meth:`accepts`, the lift sweep;
* ``wavefront`` -- loop skewing of 2-D nests whose anti-diagonal is parallel;
* ``contracts`` -- parallelism under a runtime contract: a versioned constraint, a guarded scatter.

Modules are imported where used, not here: eager imports would close an ``interstate`` import cycle.
"""
