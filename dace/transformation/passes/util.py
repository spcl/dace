# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Set, Type

from ..transformation import PatternTransformation, SubgraphTransformation, TransformationBase
from ..pass_pipeline import Pass, VisitorPass, StatePass, Pipeline, FixedPointPipeline, ScopePass


def available_passes(all_passes: bool = False) -> Set[Type['Pass']]:
    """
    Returns all available passes and pass pipelines as a set by recursing over Pass subclasses. 
    :param all_passes: Include all passes, e.g., including PatternTransformation and other base passes.
    """
    full_pass_set = Pass.subclasses_recursive()

    if not all_passes:
        reduced_pass_set = set()
        for p in full_pass_set:
            if not issubclass(p, TransformationBase) and not p.CATEGORY == 'Helper':
                reduced_pass_set.add(p)
        return reduced_pass_set

    return full_pass_set
