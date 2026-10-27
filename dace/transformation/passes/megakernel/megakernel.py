# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MegaKernelOptimization``: run each maximal hierarchical linear chain of GPU kernels as one persistent,
cooperative kernel. Stage 1 finds the chains; it does not rewrite the SDFG yet."""

from dace import SDFG, properties
from dace.sdfg.state import ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.megakernel.discovery import MegakernelChain, find_chains


@properties.make_properties
@transformation.explicit_cf_compatible
class MegaKernelOptimization(ppl.Pass):
    """Fuses the GPU kernels of every maximal hierarchical linear chain in ``region`` (default: the whole SDFG) into
    one persistent kernel.

    ``small_symbols`` names the symbols whose extents are too short to spread over the grid: a map dimension whose
    extent names only them stays a loop in every thread. The pass reads no symbol facts of the SDFG.
    """

    CATEGORY: str = "Optimization Preparation"

    small_symbols = properties.SetProperty(
        element_type=str, default=set(), desc="Symbols whose extents stay a loop in every thread"
    )

    def __init__(self, region: ControlFlowRegion | None = None, small_symbols: set[str] | None = None) -> None:
        super().__init__()
        self._region = region
        self.small_symbols = set(small_symbols or ())

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> list[MegakernelChain] | None:
        chains = find_chains(self._region if self._region is not None else sdfg, sdfg)
        return chains or None
