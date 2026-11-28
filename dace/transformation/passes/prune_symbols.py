# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

from dace import SDFG, dtypes, properties, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveUnusedSymbols(ppl.Pass):
    """
    Prunes unused symbols from the SDFG symbol repository (``sdfg.symbols``) and interstate edges.
    Also includes uses in Tasklets of all languages.
    """

    CATEGORY: str = "Simplification"

    recursive = properties.Property(
        dtype=bool, default=True, category="Applicability", desc="Prune nested SDFGs recursively"
    )
    symbols = properties.SetProperty(
        element_type=str, allow_none=True, category="Applicability", desc="Limit considered symbols to this set"
    )

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.States | ppl.Modifies.Edges | ppl.Modifies.Descriptors | ppl.Modifies.Tasklets)

    def apply_pass(self, sdfg: SDFG, _) -> set[tuple[int, str]] | None:
        """
        Removes unused symbols from the SDFG.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :param initial_symbols: If not None, sets values of initial symbols.
        :return: The symbols removed; an empty set when only dead interstate assignments were deleted; ``None``
                 only when the SDFG is untouched, since pipelines read ``None`` as "not modified".
        """
        result: set[str] = set()

        repository_symbols_to_consider = self.symbols or set(sdfg.symbols.keys())
        bound_symbols = {sym for isedge in sdfg.all_interstate_edges() for sym in isedge.data.assignments}

        # Only these names are ever acted on, so the walk stops once each is known to be used
        unused_symbols = repository_symbols_to_consider | bound_symbols
        for used in self.symbol_uses(sdfg) if unused_symbols else ():
            unused_symbols -= used
            if not unused_symbols:
                break

        # Remove unused symbols from interstate edge assignments.
        modified = False
        if unused_symbols & bound_symbols:
            for isedge in sdfg.all_interstate_edges():
                for sym in isedge.data.assignments.keys() & unused_symbols:
                    del isedge.data.assignments[sym]
                    modified = True

        # Remove unused symbols from the SDFG's symbols repository.
        for sym in repository_symbols_to_consider & unused_symbols:
            if sym in sdfg.symbols:
                sdfg.remove_symbol(sym)
                result.add(sym)

        if self.recursive:
            # Prune nested SDFGs recursively
            sid = sdfg.cfg_id
            result = {(sid, sym) for sym in result}

            for state in sdfg.states():
                for node in state.nodes():
                    if isinstance(node, nodes.NestedSDFG):
                        old_symbols = self.symbols
                        self.symbols = set()
                        nres = self.apply_pass(node.sdfg, _)
                        self.symbols = old_symbols
                        # An empty set still means the nested run modified something
                        if nres is not None:
                            modified = True
                            result.update(nres)

        # Return result
        if result:
            return result
        return set() if modified else None

    def report(self, pass_retval: set[str]) -> str:
        return f"Removed {len(pass_retval)} unused symbols: {pass_retval}."

    def used_symbols(self, sdfg: SDFG) -> set[str]:
        return set().union(*self.symbol_uses(sdfg))

    def symbol_uses(self, sdfg: SDFG) -> Iterator[set[str]]:
        """The names ``sdfg`` reads, in batches, the cheap sources first."""
        for e in sdfg.all_interstate_edges():
            yield e.data.free_symbols

        for desc in sdfg.arrays.values():
            yield set(map(str, desc.free_symbols))

        # Add symbols in global/init/exit code
        for code in itertools.chain(sdfg.global_code.values(), sdfg.init_code.values(), sdfg.exit_code.values()):
            yield symbolic.symbols_in_code(code.as_string)

        for relation in sdfg.symbol_repo.runtime_assumptions:
            yield symbolic.relation_names(relation)

        for block in sdfg.all_control_flow_blocks():
            # In addition to the standard free symbols, we are conservative with other tasklet languages by
            # tokenizing their code. Since this is intersected with `sdfg.symbols`, keywords such as "if" are
            # ok to include
            if isinstance(block, SDFGState):
                yield block.free_symbols
                for node in block.nodes():
                    if isinstance(node, nodes.Tasklet):
                        for code in (node.code, node.code_global, node.code_init, node.code_exit):
                            if code.language != dtypes.Language.Python:
                                yield symbolic.symbols_in_code(
                                    code.as_string, sdfg.symbols.keys(), node.ignored_symbols
                                )
            else:
                # The blocks inside a region are visited themselves: walking them again for every enclosing region
                # repeats the whole nest once per nesting level
                yield block.used_symbols(all_symbols=True, with_contents=False)
