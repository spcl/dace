# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Register the reserved thread-count symbol so a later pass can size a buffer per thread.

Sizing a per-thread buffer needs the OpenMP thread count at ALLOCATION time, but the count must not
enter the ABI -- a caller cannot be asked for a machine property. The DEFINITION is emitted by frame
code (:data:`~dace.codegen.targets.framecode.NUM_THREADS_DECL`), at the top of the program function
ahead of every allocation, because that is the only point early enough: an allocation is emitted
before any state runs, so nothing that runs as part of the graph -- a tasklet in a preamble state,
an interstate assignment -- can define a symbol that an allocation above it already read.

What is left here is the DECLARATION: the name and its width, registered on the SDFG so a pass that
introduces a use does not have to arrange for the symbol to exist first, and so nested SDFGs inherit
it through ``symbol_mapping`` like any other symbol.
"""
from typing import Any, Dict, Optional

from dace import SDFG, dtypes, properties, symbolic
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: One name, one dtype. int64 because the value divides int64 extents -- a narrower declaration
#: makes the emitted ``int_ceil`` deduce a mixed-width result a loop counter cannot take. Every site
#: that declares the symbol reads THIS, so two of them cannot disagree.
DTYPE = dtypes.int64


def symbol_dtype(sdfg: SDFG):
    """The width this SDFG gives the symbol, else :data:`DTYPE`.

    A name denotes one value AND one width, so every site that emits or reads the symbol asks here
    rather than assuming. :class:`SupplyNumThreads` normalizes the entry to :data:`DTYPE`, which
    matters because sizing a transient by the symbol REGISTERS it as a side effect, at
    ``DEFAULT_SYMBOL_TYPE`` (int32) -- a width nobody chose and that would then be emitted.
    """
    return sdfg.symbols.get(symbolic.NUM_THREADS_SYMBOL, DTYPE)


@properties.make_properties
@transformation.explicit_cf_compatible
class SupplyNumThreads(ppl.Pass):
    """Declare ``__dace_num_threads`` on the top-level SDFG."""

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """:returns: 1 when the declaration was added or corrected, else ``None``."""
        name = symbolic.NUM_THREADS_SYMBOL
        # Only the top-level graph declares it; a nested SDFG receives the value through its
        # ``symbol_mapping``, and declaring it again there would shadow the parent's.
        if sdfg.parent_sdfg is not None or sdfg.symbols.get(name) == DTYPE:
            return None
        # Assigned, not ``add_symbol``: sizing a transient by the symbol already registered it at
        # ``DEFAULT_SYMBOL_TYPE``, so the name usually EXISTS here at a width nobody chose, and
        # ``add_symbol`` refuses a name it already holds. The reserved symbol has one width by
        # contract, so the accidental narrower declaration is corrected rather than deferred to.
        sdfg.symbols[name] = DTYPE
        return 1


def ensure_in_scope(nsdfg_node) -> None:
    """Make ``__dace_num_threads`` resolvable inside ``nsdfg_node``.

    A symbol reaches a nested SDFG only through its ``symbol_mapping``, so a transformation that
    introduces a USE of the thread count below the top level must top the mapping up or the inner
    graph names a symbol nothing defines. Idempotent, so calling it twice is free.
    """
    name = symbolic.NUM_THREADS_SYMBOL
    if name not in nsdfg_node.symbol_mapping:
        nsdfg_node.symbol_mapping[name] = symbolic.pystr_to_symbolic(name)
    inner = nsdfg_node.sdfg
    if inner is not None and name not in inner.symbols:
        inner.add_symbol(name, DTYPE)
