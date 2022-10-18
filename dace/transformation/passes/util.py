# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Set, Type

from ..transformation import PatternTransformation, SubgraphTransformation
from ..pass_pipeline import Pass


def _recursive_subclasses(cls) -> Set:
    subclasses = set(cls.__subclasses__())
    subsubclasses = set()
    for sc in subclasses:
        subsubclasses.update(_recursive_subclasses(sc))

    # Ignore abstract classes
    result = subclasses | subsubclasses
    result = set(sc for sc in result if not getattr(sc, '__abstractmethods__', False))

    return result


def available_passes(all_passes: bool = True) -> Set[Type['Pass']]:
    """
    Returns all available passes and pass pipelines as a set by recursing over Pass subclasses. 
    :param all_passes: Include all passes, including PatternTransformations and SubgraphTransformations.
    """
    full_pass_set = _recursive_subclasses(Pass)
    if not all_passes:
        reduced_pass_set = set()
        for p in full_pass_set:
            if not issubclass(p, (PatternTransformation, SubgraphTransformation)):
                reduced_pass_set.add(p)
        return reduced_pass_set
    return full_pass_set
