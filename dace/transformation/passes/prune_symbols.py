# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import itertools
import re
from dataclasses import dataclass
from typing import Optional, Set, Tuple

from dace import SDFG, dtypes, properties
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl

_NAME_TOKENS = re.compile(r'[a-zA-Z_][a-zA-Z_0-9]*')


@dataclass(unsafe_hash=True)
@properties.make_properties
class RemoveUnusedSymbols(ppl.Pass):
    """
    Prunes unused symbols from the SDFG symbol repository (``sdfg.symbols``).
    Also includes uses in Tasklets of all languages.
    """

    CATEGORY: str = 'Simplification'

    recursive = properties.Property(dtype=bool, default=True, desc='Prune nested SDFGs recursively')
    symbols = properties.SetProperty(element_type=str, allow_none=True, desc='Limit considered symbols to this set')

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.States | ppl.Modifies.Edges | ppl.Modifies.Descriptors | ppl.Modifies.Tasklets)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[Tuple[int, str]]]:
        """
        Propagates constants throughout the SDFG.
        
        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :param initial_symbols: If not None, sets values of initial symbols.
        :return: A set of propagated constants, or None if nothing was changed.
        """
        result: Set[str] = set()

        symbols_to_consider = self.symbols or set(sdfg.symbols.keys())

        # Compute used symbols
        used_symbols = self.used_symbols(sdfg)

        # Remove unused symbols
        for sym in symbols_to_consider - used_symbols:
            if sym in sdfg.symbols:
                sdfg.remove_symbol(sym)
                result.add(sym)

        for e in sdfg.edges():
            for aname in list(e.data.assignments):
                if aname in result:
                    del e.data.assignments[aname]

        if self.recursive:
            # Prune nested SDFGs recursively
            sid = sdfg.sdfg_id
            result = set((sid, sym) for sym in result)

            for state in sdfg.nodes():
                for node in state.nodes():
                    if isinstance(node, nodes.NestedSDFG):
                        old_symbols = self.symbols
                        self.symbols = set()
                        nres = self.apply_pass(node.sdfg, _)
                        self.symbols = old_symbols
                        if nres:
                            result.update(nres)

        # Return result
        return result or None

    def report(self, pass_retval: Set[str]) -> str:
        return f'Removed {len(pass_retval)} unused symbols: {pass_retval}.'

    def used_symbols(self, sdfg: SDFG) -> Set[str]:
        result = set()

        # Add symbols in global/init/exit code
        for code in itertools.chain(sdfg.global_code.values(), sdfg.init_code.values(), sdfg.exit_code.values()):
            result |= _symbols_in_code(code.as_string)

        for desc in sdfg.arrays.values():
            result |= set(map(str, desc.used_symbols(False)))

        for state in sdfg.nodes():
            result |= state.used_symbols(False)
            # In addition to the standard free symbols, we are conservative with other tasklet languages by
            # tokenizing their code. Since this is intersected with `sdfg.symbols`, keywords such as "if" are
            # ok to include
            for node in state.nodes():
                if isinstance(node, nodes.Tasklet):
                    if node.code.language != dtypes.Language.Python:
                        result |= _symbols_in_code(node.code.as_string)
                    if node.code_global.language != dtypes.Language.Python:
                        result |= _symbols_in_code(node.code_global.as_string)
                    if node.code_init.language != dtypes.Language.Python:
                        result |= _symbols_in_code(node.code_init.as_string)
                    if node.code_exit.language != dtypes.Language.Python:
                        result |= _symbols_in_code(node.code_exit.as_string)


        for e in sdfg.edges():
            result |= e.data.used_symbols(False)

        return result

def _symbols_in_code(code: str) -> Set[str]:
    if not code:
        return set()
    return set(re.findall(_NAME_TOKENS, code))
