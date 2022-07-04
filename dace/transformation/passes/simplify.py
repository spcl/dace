# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from copy import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Set

from dace import SDFG, config
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.array_elimination import ArrayElimination
from dace.transformation.passes.consolidate_edges import ConsolidateEdges
from dace.transformation.passes.constant_propagation import ConstantPropagation
from dace.transformation.passes.dead_dataflow_elimination import DeadDataflowElimination
from dace.transformation.passes.dead_state_elimination import DeadStateElimination
from dace.transformation.passes.fusion_inline import FuseStates, InlineSDFGs
from dace.transformation.passes.optional_arrays import OptionalArrayInference
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion

SIMPLIFY_PASSES = [
    InlineSDFGs,
    ScalarToSymbolPromotion,
    FuseStates,
    OptionalArrayInference,
    ConstantPropagation,
    DeadDataflowElimination,
    DeadStateElimination,
    ArrayElimination,
]

_nonrecursive_passes = [ScalarToSymbolPromotion, DeadDataflowElimination, DeadStateElimination, ArrayElimination]


@dataclass(unsafe_hash=True)
class SimplifyPass(ppl.FixedPointPipeline):
    validate: bool = False  #: Whether to validate the SDFG at the end of the pipeline
    validate_all: bool = False  #: Whether to validate the SDFG after each pass
    skip: Optional[Set[str]] = None  #: Names of passes to skip in this pipeline
    verbose: bool = False  #: Whether to print reports after every pass

    def __init__(self,
                 validate: bool = False,
                 validate_all: bool = False,
                 skip: Optional[Set[str]] = None,
                 verbose: bool = False):
        if skip:
            passes = [p() for p in SIMPLIFY_PASSES if p.__name__ not in skip]
        else:
            passes = [p() for p in SIMPLIFY_PASSES]

        super().__init__(passes=passes)
        self.validate = validate
        self.validate_all = validate_all
        self.skip = skip or set()
        self.verbose = verbose

    def apply_subpass(self, sdfg: SDFG, p: ppl.Pass, state: Dict[str, Any]):
        """
        Apply a pass from the pipeline. This method is meant to be overridden by subclasses.
        """
        if type(p) in _nonrecursive_passes:  # If pass needs to run recursively, do so and modify return value
            ret: Dict[int, Any] = {}
            for sd in sdfg.all_sdfgs_recursive():
                subret = p.apply_pass(sd, state)
                if subret is not None:
                    ret[sd.sdfg_id] = subret
            ret = ret or None
        else:
            ret = p.apply_pass(sdfg, state)

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
            else:
                print(type(p).__name__, 'did nothing')

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
