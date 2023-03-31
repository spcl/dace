# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
from typing import Any, Dict, Optional, Set

from dace import SDFG, InterstateEdge
from dace.sdfg import nodes as nd
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes import analysis as ap


class ScalarFission(ppl.Pass):
    """
    Fission transient scalars or arrays of size 1 that are dominated by a write into separate data containers.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes

    def depends_on(self):
        return {ap.ScalarWriteShadowScopes}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """
        Rename scalars and arrays of size 1 based on dominated scopes.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary mapping the original name to a set of all new names created for each data container.
        """
        results: Dict[str, Set[str]] = defaultdict(lambda: set())

        shadow_scope_dict: ap.WriteScopeDict = pipeline_results[ap.ScalarWriteShadowScopes.__name__][sdfg.sdfg_id]

        for name, write_scope_dict in shadow_scope_dict.items():
            desc = sdfg.arrays[name]

            # If this isn't a scalar or an array of size 1, don't do anything.
            if desc.total_size != 1:
                continue

            # If there is only one scope, don't do anything.
            if len(write_scope_dict) <= 1:
                continue

            # Don't rename anything that's not transient, as it may be used externally.
            if not desc.transient:
                continue

            for write, shadowed_reads in write_scope_dict.items():
                if write is not None and len(shadowed_reads) > 0:
                    newdesc = desc.clone()
                    newname = sdfg.add_datadesc(name, newdesc, find_new_name=True)

                    # Replace the write and any connected memlets with writes to the new data container.
                    write_node = write[1]
                    write_node.data = newname
                    for iedge in write[0].in_edges(write_node):
                        if iedge.data.data == name:
                            iedge.data.data = newname
                    for oeade in write[0].out_edges(write_node):
                        if oeade.data.data == name:
                            oeade.data.data = newname

                    # Replace all dominated reads and connected memlets.
                    for read in shadowed_reads:
                        if isinstance(read[1], nd.AccessNode):
                            read_node = read[1]
                            read_node.data = newname
                            for iedge in read[0].in_edges(read_node):
                                if iedge.data.data == name:
                                    iedge.data.data = newname
                            for oeade in read[0].out_edges(read_node):
                                if oeade.data.data == name:
                                    oeade.data.data = newname
                        elif isinstance(read[1], InterstateEdge):
                            read[1].replace_dict({name: newname})

                    results[name].add(newname)
        return results

    def report(self, pass_retval: Any) -> Optional[str]:
        return f'Renamed {len(pass_retval)} scalars: {pass_retval}.'
