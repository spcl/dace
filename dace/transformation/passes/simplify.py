# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import warnings

from dace import SDFG, config, properties
from dace.transformation import helpers as xfh, transformation
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.array_elimination import ArrayElimination
from dace.transformation.passes.consolidate_edges import ConsolidateEdges
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
from dace.transformation.passes.dead_state_elimination import DeadStateElimination
from dace.transformation.passes.fusion_inline import FuseStates, InlineControlFlowRegions, InlineSDFGs
from dace.transformation.passes.optional_arrays import OptionalArrayInference
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols
from dace.transformation.passes.reference_reduction import ReferenceToView
from dace.transformation.passes.simplification.control_flow_raising import ControlFlowRaising
from dace.transformation.passes.simplification.prune_empty_conditional_branches import PruneEmptyConditionalBranches

SIMPLIFY_PASSES = [
    InlineSDFGs,
    InlineControlFlowRegions,
    ScalarToSymbolPromotion,
    ControlFlowRaising,
    FuseStates,
    OptionalArrayInference,
    ConstantPropagation,
    DeadDataflowElimination,
    DeadStateElimination,
    PruneEmptyConditionalBranches,
    RemoveUnusedSymbols,
    ReferenceToView,
    ArrayElimination,
    ConsolidateEdges,
]

_nonrecursive_passes = [
    ScalarToSymbolPromotion,
    DeadDataflowElimination,
    DeadStateElimination,
    ArrayElimination,
    ConsolidateEdges,
    ReferenceToView,
]


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class SimplifyPass(ppl.FixedPointPipeline):
    """
    A pipeline that simplifies an SDFG by applying a series of simplification passes.
    """

    CATEGORY: str = 'Simplification'

    validate = properties.Property(dtype=bool,
                                   default=False,
                                   desc='Whether to validate the SDFG at the end of the pipeline.')
    validate_all = properties.Property(dtype=bool, default=False, desc='Whether to validate the SDFG after each pass.')
    skip = properties.SetProperty(element_type=str, default=set(), desc='Set of pass names to skip.')
    verbose = properties.Property(dtype=bool, default=False, desc='Whether to print reports after every pass.')

    no_inline_function_call_regions = properties.Property(dtype=bool,
                                                          default=False,
                                                          desc='Whether to prevent inlining function call regions.')
    no_inline_named_regions = properties.Property(dtype=bool,
                                                  default=False,
                                                  desc='Whether to prevent inlining named control flow regions.')

    def __init__(self,
                 validate: bool = False,
                 validate_all: bool = False,
                 skip: Optional[Set[str]] = None,
                 verbose: bool = False,
                 no_inline_function_call_regions: bool = False,
                 no_inline_named_regions: bool = False,
                 pass_options: Optional[Dict[str, Any]] = None):
        if skip:
            passes: List[ppl.Pass] = [p() for p in SIMPLIFY_PASSES if p.__name__ not in skip]
        else:
            passes: List[ppl.Pass] = [p() for p in SIMPLIFY_PASSES]

        super().__init__(passes=passes)
        self.validate = validate
        self.validate_all = validate_all
        self.skip = skip or set()
        if config.Config.get('debugprint') == 'verbose':
            self.verbose = True
        else:
            self.verbose = verbose

        self.no_inline_function_call_regions = no_inline_function_call_regions
        self.no_inline_named_regions = no_inline_named_regions

        pass_opts = {
            'InlineControlFlowRegions.no_inline_function_call_regions': self.no_inline_function_call_regions,
            'InlineControlFlowRegions.no_inline_named_regions': self.no_inline_named_regions,
        }
        if pass_options:
            pass_opts.update(pass_options)
        for p in passes:
            p.set_opts(pass_opts)

    def apply_subpass(self, sdfg: SDFG, p: ppl.Pass, state: Dict[str, Any]):
        """
        Apply a pass from the pipeline. This method is meant to be overridden by subclasses.
        """
        if sdfg.root_sdfg.using_explicit_control_flow:
            if (not hasattr(p, '__explicit_cf_compatible__') or p.__explicit_cf_compatible__ == False):
                warnings.warn(p.__class__.__name__ + ' is not being applied due to incompatibility with ' +
                              'experimental control flow blocks. If the SDFG does not contain experimental blocks, ' +
                              'ensure the top level SDFG does not have `SDFG.using_explicit_control_flow` set to ' +
                              'True. If ' + p.__class__.__name__ + ' is compatible with experimental blocks, ' +
                              'please annotate it with the class decorator ' +
                              '`@dace.transformation.explicit_cf_compatible`. see ' +
                              '`https://github.com/spcl/dace/wiki/Experimental-Control-Flow-Blocks` ' +
                              'for more information.')
                return None

        if type(p) in _nonrecursive_passes:  # If pass needs to run recursively, do so and modify return value
            ret: Dict[int, Any] = {}
            for sd in sdfg.all_sdfgs_recursive():
                subret = p.apply_pass(sd, state)
                if subret is not None:
                    ret[sd.cfg_id] = subret
            ret = ret or None
        else:
            ret = p.apply_pass(sdfg, state)
        if ret is not None:
            sdfg.reset_cfg_list()

        if self.verbose:
            if ret is not None:
                if type(p) not in _nonrecursive_passes:
                    rep = p.report(ret)
                else:
                    # Create report from recursive application
                    rep = []
                    for sdid, subret in ret.items():
                        if subret is not None:
                            rep.append(f'SDFG {sdid}: ' + p.report(subret))
                    rep = '\n'.join(rep)

                if rep:
                    print(rep)

        # If validate all is enabled, check after every pass
        if ret is not None and self.validate_all:
            sdfg.validate()
        return ret

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = super().apply_pass(sdfg, pipeline_results)

        if result is not None:
            # Split back edges with assignments and conditions to allow richer
            # control flow detection in code generation
            xfh.split_interstate_edges(sdfg)

            # If validate is enabled, check SDFG at the end
            if self.validate and not self.validate_all:
                sdfg.validate()

        return result
