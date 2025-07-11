# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Optional, Set

import warnings

from dace import SDFG, SDFGState, properties, transformation
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow import MapFusionVertical
from dace.transformation.passes import analysis as ap, pattern_matching as pmp


@properties.make_properties
@transformation.explicit_cf_compatible
class FullMapFusion(ppl.Pass):
    """
    Pass that combines `MapFusionVertical` and `FindSingleUseData` into one.

    Essentially, this function runs `FindSingleUseData` before `MapFusionVertical`, this
    will speedup the fusion, as the SDFG has to be scanned only once.
    The pass accepts the same options as `MapFusionVertical`, for a detailed description
    see there.

    :param only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
    :param only_toplevel_maps: Only consider Maps that are at the top.
    :param strict_dataflow: Which dataflow mode should be used, see above.
    :param assume_always_shared: Assume that all intermediates are shared.
    :param validate: Validate the SDFG after the pass as finished.
    :param validate_all: Validate the SDFG after every transformation.

    :todo: Implement a faster matcher as the pattern is constant.
    """

    CATEGORY: str = 'Simplification'

    # Settings
    only_toplevel_maps = properties.Property(
        dtype=bool,
        default=False,
        desc='Only perform fusing if the Maps are in the top level.',
    )
    only_inner_maps = properties.Property(
        dtype=bool,
        default=False,
        desc='Only perform fusing if the Maps are inner Maps, i.e. does not have top level scope.',
    )
    strict_dataflow = properties.Property(
        dtype=bool,
        default=True,
        desc='If `True` then the transformation will ensure a more stricter data flow.',
    )
    assume_always_shared = properties.Property(
        dtype=bool,
        default=False,
        desc='If `True` then all intermediates will be classified as shared.',
    )

    validate = properties.Property(
        dtype=bool,
        default=True,
        desc='If True, validates the SDFG after all transformations have been applied.',
    )
    validate_all = properties.Property(dtype=bool,
                                       default=False,
                                       desc='If True, validates the SDFG after each transformation applies.')

    def __init__(
        self,
        only_inner_maps: Optional[bool] = None,
        only_toplevel_maps: Optional[bool] = None,
        strict_dataflow: Optional[bool] = None,
        assume_always_shared: Optional[bool] = None,
        validate: Optional[bool] = None,
        validate_all: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if only_toplevel_maps is not None:
            self.only_toplevel_maps = only_toplevel_maps
        if only_inner_maps is not None:
            self.only_inner_maps = only_inner_maps
        if strict_dataflow is not None:
            self.strict_dataflow = strict_dataflow
        if assume_always_shared is not None:
            self.assume_always_shared = assume_always_shared
        if validate is not None:
            self.validate = validate
        if validate_all is not None:
            self.validate_all = validate_all

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.Scopes | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.States)

    def depends_on(self):
        return {ap.FindSingleUseData}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """
        Fuses all Maps that can be fused in the SDFG, including its nested SDFGs.

        For driving the fusion the function will construct a `PatternMatchAndApplyRepeated`
        object.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: The result of previous pipeline steps. The pass expects
            at least the result of the `FindSingleUseData`.
        :return: The numbers of Maps that were fused or `None` if none were fused.
        """
        if ap.FindSingleUseData.__name__ not in pipeline_results:
            raise ValueError(f'Expected to find `FindSingleUseData` in `pipeline_results`.')

        fusion = MapFusionVertical(only_inner_maps=self.only_inner_maps,
                                   only_toplevel_maps=self.only_toplevel_maps,
                                   strict_dataflow=self.strict_dataflow,
                                   assume_always_shared=self.assume_always_shared)

        try:
            # The short answer why we do this is because  `fusion._pipeline_results` is
            #  only defined during `apply()` and not during `can_be_applied()`. For more
            #  information see the note in `MapFusionVertical.is_shared_data()` and/or [issue#1911](https://github.com/spcl/dace/issues/1911).
            assert fusion._single_use_data is None
            fusion._single_use_data = pipeline_results["FindSingleUseData"]
            pazz = pmp.PatternMatchAndApplyRepeated(
                [fusion],
                permissive=False,
                validate=False,
                validate_all=self.validate_all,
            )
            result = pazz.apply_pass(sdfg, pipeline_results)

        finally:
            fusion._single_use_data = None

        if self.validate:
            sdfg.validate()

        return result
