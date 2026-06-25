from dace import dtypes, properties, data
from dace.sdfg import nodes, SDFG, InterstateEdge
from dace.sdfg.state import SDFGState, ConditionalBlock, ControlFlowRegion, LoopRegion, ReturnBlock, ContinueBlock, BreakBlock, ControlFlowBlock
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible

from warnings import warn
from typing import Any, Dict, Tuple, List, Optional, Set, Type, Union


@properties.make_properties
@explicit_cf_compatible
class OffloadToAccelerator(ppl.Pass):
    """
    Docstring for OffloadToAccelerator
    """
    
    CATEGORY: str = 'Offload To Accelerator'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False
    
    #def depends_on(self) -> Set[Union[Type['Pass'], 'Pass']]:
    #    return set()
    
    #def report(self, pass_retval: Any) -> Optional[str]:
    #    """
    #    Returns a user-readable string report based on the results of this pass.
    #
    #    :param pass_retval: The return value from applying this pass.
    #    :return: A string with the user-readable report, or None if nothing to report.
    #    """
    #    return None

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        """
        Applies the pass to the given SDFG.

        :param sdfg: The SDFG to apply the pass to.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: Some object if pass was applied, or None if nothing changed.
        """

    def len1_array_to_scalar(self):
        pass

    def absorb_single_use_map_tasklets(self):
        pass

    def constant_input_duplication(self):
        pass

    def split_mixed_states(self):
        pass

# replace fp64 with fp32 in npbench repo (local)
# insert my offloading pass into dace_framework.py -> change branch of dace to mine, change result folder to avoid conflict (auto if)
# get numerical correctness on all npbench stencils
# run a) sequentially, b) old offloading pass, c) new offloading pass -> diagram
# see what breaks -> fix those bugs
# see what runs slow -> optimize those patterns
# mark if existing pass fails