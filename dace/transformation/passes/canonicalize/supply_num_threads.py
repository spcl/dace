# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Register the reserved thread-count symbol so a later pass can size a buffer per thread."""

from typing import Any

from dace import SDFG, dtypes, properties, symbolic
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: One name, one dtype.
DTYPE = dtypes.int64


def symbol_dtype(sdfg: SDFG) -> dtypes.typeclass:
    """The width this SDFG gives the symbol, else :data:`DTYPE`."""
    return sdfg.symbols.get(symbolic.NUM_THREADS_SYMBOL, DTYPE)


@properties.make_properties
@transformation.explicit_cf_compatible
class SupplyNumThreads(ppl.Pass):
    """Declare ``__dace_num_threads`` on the top-level SDFG."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """:returns: 1 when the declaration was added or corrected, else ``None``."""
        name = symbolic.NUM_THREADS_SYMBOL
        # Only the top-level graph declares it
        if sdfg.parent_sdfg is not None or sdfg.symbols.get(name) == DTYPE:
            return None
        # Assigned, not ``add_symbol``: sizing a transient by the symbol already registered it at
        # ``DEFAULT_SYMBOL_TYPE``, so the name usually EXISTS here at a width nobody chose, and ``add_symbol`` refuses a
        # name it already holds.
        if name in sdfg.symbols:
            sdfg.symbol_repo.set_type(name, DTYPE)
        else:
            sdfg.add_symbol(name, DTYPE)
        return 1
