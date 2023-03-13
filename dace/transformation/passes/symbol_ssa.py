# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
from typing import Any, Dict, Optional, Set

from dace import SDFG, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes import analysis as ap


class StrictSymbolSSA(ppl.Pass):
    """
    Perform an SSA transformation on all symbols in the SDFG in a strict manner, i.e., without introducing phi nodes.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.Nodes | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.Nodes | ppl.Modifies.States

    def depends_on(self):
        return {ap.SymbolWriteScopes}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """
        Rename symbols in a restricted SSA manner.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary mapping the original name to a set of all new names created for each symbol.
        """
        results: Dict[str, Set[str]] = defaultdict(lambda: set())

        symbol_scope_dict: ap.SymbolScopeDict = pipeline_results[ap.SymbolWriteScopes.__name__][sdfg.sdfg_id]

        for name, scope_dict in symbol_scope_dict.items():
            # If there is only one scope, don't do anything.
            if len(scope_dict) <= 1:
                continue

            for write, shadowed_reads in scope_dict.items():
                if write is not None:
                    newname = sdfg.find_new_symbol(name)
                    sdfg.symbols[newname] = sdfg.symbols[name]

                    # Replace the write to this symbol with a write to the new symbol.
                    try:
                        write.data.assignments[newname] = write.data.assignments[name]
                        del write.data.assignments[name]
                    except KeyError:
                        # Ignore.
                        pass

                    # Replace all dominated reads.
                    for read in shadowed_reads:
                        if isinstance(read, SDFGState):
                            read.replace(name, newname)
                        else:
                            if read not in scope_dict:
                                read.data.replace(name, newname)
                            else:
                                read.data.replace(name, newname, replace_keys=False)

                    results[name].add(newname)

        if len(results) == 0:
            return None
        else:
            return results

    def report(self, pass_retval: Any) -> Optional[str]:
        return f'Renamed {len(pass_retval)} symbols: {pass_retval}.'
