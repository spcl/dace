# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from dace import SDFG, config, properties
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.canonicalization.scope_consumer_producer import (
    ScopeConsumerProducerCanonicalization,
    ScopeIntermediateAccessesCanonicalization)

CANONICALIZATION_PASSES = [
    ScopeIntermediateAccessesCanonicalization,
    ScopeConsumerProducerCanonicalization,
]

_nonrecursive_canon_passes = [
]


@dataclass(unsafe_hash=True)
@properties.make_properties
class CanonicalizationPass(ppl.FixedPointPipeline):
    """
    TODO
    """

    CATEGORY: str = 'Canonicalization'

    validate = properties.Property(dtype=bool,
                                   default=False,
                                   desc='Whether to validate the SDFG at the end of the pipeline.')
    validate_all = properties.Property(dtype=bool, default=False, desc='Whether to validate the SDFG after each pass.')
    skip = properties.SetProperty(element_type=str, default=set(), desc='Set of pass names to skip.')
    verbose = properties.Property(dtype=bool, default=False, desc='Whether to print reports after every pass.')

    def __init__(self,
                 validate: bool = False,
                 validate_all: bool = False,
                 skip: Optional[Set[str]] = None,
                 verbose: bool = False):
        if skip:
            passes = [p() for p in CANONICALIZATION_PASSES if p.__name__ not in skip]
        else:
            passes = [p() for p in CANONICALIZATION_PASSES]

        super().__init__(passes=passes)
        self.validate = validate
        self.validate_all = validate_all
        self.skip = skip or set()
        if config.Config.get('debugprint') == 'verbose':
            self.verbose = True
        else:
            self.verbose = verbose

    def apply_subpass(self, sdfg: SDFG, p: ppl.Pass, state: Dict[str, Any]):
        """
        Apply a pass from the pipeline. This method is meant to be overridden by subclasses.
        """
        if type(p) in _nonrecursive_canon_passes:  # If pass needs to run recursively, do so and modify return value
            ret: Dict[int, Any] = {}
            for sd in sdfg.all_sdfgs_recursive():
                subret = p.apply_pass(sd, state)
                if subret is not None:
                    ret[sd.cfg_id] = subret
            ret = ret or None
        else:
            ret = p.apply_pass(sdfg, state)

        if self.verbose:
            if ret is not None:
                if type(p) not in _nonrecursive_canon_passes:
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
