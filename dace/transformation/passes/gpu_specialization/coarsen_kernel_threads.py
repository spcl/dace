# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Make every thread of a kernel compute several elements (thread coarsening).

After the offload a kernel runs one thread per iteration of its ``GPU_Device`` map::

    GPU_Device map (jk, jc):                 # one thread per (jk, jc), jc is threadIdx.x
        body(jk, jc)

This pass tiles the map along one parameter by ``factor`` and gives the tile to the thread as a serial loop::

    GPU_Device map (c_jk, jc), c_jk = 0, F, 2F, ...:     # one thread per F levels
        map jk in c_jk : c_jk + F (Sequential):
            body(jk, jc)

so a thread block covers ``factor`` times more of the coarsened axis: data the block reads independently of that
axis is read once for ``factor`` times more work (and a stage of it through shared memory,
:class:`~dace.transformation.passes.gpu_specialization.stage_thread_block_data.StageThreadBlockData`, is amortized
over it). The coarsened parameter defaults to the outermost one that is not ``threadIdx.x`` (the last): its
consecutive elements are a thread's, so the lanes of a wavefront still walk the unit-stride axis. ``factors``
coarsens several parameters at once (``{"jk": 6, "jc": 2}``: a thread computes a 6 x 2 tile). A map of one parameter
(by default), a kernel that already has thread-block maps, and a parameter the map does not have are left alone.
"""

from typing import Any

from dace import SDFG, SDFGState, dtypes, properties
from dace.sdfg import nodes
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.passes.gpu_specialization.block_tile_kernels import is_kernel

#: Prefix of a coarsened parameter's chunk index (``c_jk`` for ``jk``).
COARSE_PREFIX = "c"


@properties.make_properties
class CoarsenKernelThreads(ppl.Pass):
    """Tile every kernel map along one parameter by ``factor``, the tile a serial loop of the thread (see the module
    docstring)."""

    CATEGORY: str = "Device Specialization"

    factor = properties.Property(dtype=int, default=4, desc="Elements of the coarsened parameter per thread.")
    parameter = properties.Property(
        dtype=str,
        default="",
        desc="The parameter to coarsen; empty takes the outermost one that is not the last (threadIdx.x).",
    )
    factors = properties.DictProperty(
        key_type=str, value_type=int, desc="Elements per thread of several parameters; overrides factor/parameter."
    )

    def __init__(self, factor: int = 4, parameter: str = "", factors: dict[str, int] | None = None) -> None:
        super().__init__()
        self.factor = factor
        self.parameter = parameter
        self.factors = dict(factors or {})

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Memlets | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def choose(self, entry: nodes.MapEntry) -> dict[str, int]:
        """The factor of each parameter of ``entry`` to coarsen (by more than one), empty for none."""
        params = entry.map.params
        if self.factors:
            return {p: f for p, f in self.factors.items() if p in params and f > 1}
        if self.factor < 2:
            return {}
        if self.parameter:
            return {self.parameter: self.factor} if self.parameter in params else {}
        return {params[0]: self.factor} if len(params) > 1 else {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> dict[str, str] | None:
        """Coarsen every kernel of ``sdfg`` and its nested SDFGs.

        :returns: the factors applied per kernel label, or None if no kernel was coarsened.
        """
        kernels = [
            (state, node)
            for node, state in sdfg.all_nodes_recursive()
            if isinstance(node, nodes.MapEntry) and is_kernel(state, node)
        ]
        coarsened = {}
        for state, kernel in kernels:
            if xfh.gpu_map_has_explicit_threadblocks(state, kernel):
                continue
            chosen = self.choose(kernel)
            if not chosen:
                continue
            self.coarsen(state, kernel, chosen)
            coarsened[kernel.map.label] = chosen
        return coarsened or None

    def coarsen(self, state: SDFGState, kernel: nodes.MapEntry, factors: dict[str, int]) -> None:
        """Tile ``kernel`` by ``factors``; the original entry becomes the thread's serial loop over the tile."""
        tile_sizes = [factors.get(p, 1) for p in kernel.map.params]
        MapTiling.apply_to(
            state.sdfg,
            options={"prefix": COARSE_PREFIX, "tile_sizes": tile_sizes, "tile_trivial": False, "divides_evenly": False},
            map_entry=kernel,
        )
        grid, *_ = xfh.get_parent_map(state, kernel)
        # The kernel's launch properties and its timing move to the grid map; the tile is the thread's loop.
        grid.map.schedule = dtypes.ScheduleType.GPU_Device
        for name in ("gpu_block_size", "gpu_launch_bounds", "gpu_min_warps_per_eu", "gpu_maxnreg", "instrument"):
            setattr(grid.map, name, getattr(kernel.map, name))
        kernel.map.schedule = dtypes.ScheduleType.Sequential
        kernel.map.gpu_block_size = None
        kernel.map.instrument = dtypes.InstrumentationType.No_Instrumentation
