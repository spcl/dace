# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Optional, Set

import warnings

from dace import SDFG, SDFGState, properties, transformation
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow import MapFusion
from dace.transformation.passes import analysis as ap, pattern_matching as pmp


@properties.make_properties
@transformation.explicit_cf_compatible
class FullMapFusion(ppl.Pass):
    """Pass that combines `MapFusion` and `FindSingleUseData` into one.

    Essentially, this function runs `FindSingleUseData` before `MapFusion`, this
    will speedup the fusion, as the SDFG has to be scanned only once.
    The pass accepts the same options as `MapFusion`, for a detailed description
    see there.
    The main difference is that parallel map fusion is on by default and that
    the single use data can not be passed as an argument.

    :param only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
    :param only_toplevel_maps: Only consider Maps that are at the top.
    :param strict_dataflow: Which dataflow mode should be used, see above.
    :param assume_always_shared: Assume that all intermediates are shared.
    :param allow_serial_map_fusion: Allow serial map fusion, by default `True`.
    :param allow_parallel_map_fusion: Allow to merge parallel maps, by default `True`.
    :param only_if_common_ancestor: In parallel map fusion mode, only fuse if both map
        have a common direct ancestor.
    :param validate: Validate the SDFG after the pass as finished.
    :param validate_all: Validate the SDFG after every transformation.

    :todo: Implement a faster matcher as the pattern is constant.
    """

    CATEGORY: str = 'Simplification'

    # Settings
    only_toplevel_maps = properties.Property(
        dtype=bool,
        default=False,
        desc="Only perform fusing if the Maps are in the top level.",
    )
    only_inner_maps = properties.Property(
        dtype=bool,
        default=False,
        desc="Only perform fusing if the Maps are inner Maps, i.e. does not have top level scope.",
    )
    strict_dataflow = properties.Property(
        dtype=bool,
        default=True,
        desc="If `True` then the transformation will ensure a more stricter data flow.",
    )
    assume_always_shared = properties.Property(
        dtype=bool,
        default=False,
        desc="If `True` then all intermediates will be classified as shared.",
    )
    allow_serial_map_fusion = properties.Property(
        dtype=bool,
        default=True,
        desc="If `True`, the default, then allow serial map fusion.",
    )

    allow_parallel_map_fusion = properties.Property(
        dtype=bool,
        default=True,
        desc="If `True`, the default, then also perform parallel map fusion.",
    )
    only_if_common_ancestor = properties.Property(
        dtype=bool,
        default=False,
        desc="If `True` restrict parallel map fusion to maps that have a direct common ancestor.",
    )
    validate = properties.Property(
        dtype=bool,
        default=True,
        desc='If True, validates the SDFG after all transformations have been applied.',
    )
    validate_all = properties.Property(
        dtype=bool,
        default=False,
        desc='If True, validates the SDFG after each transformation applies.'
    )


    def __init__(
        self,
        only_inner_maps: Optional[bool] = None,
        only_toplevel_maps: Optional[bool] = None,
        strict_dataflow: Optional[bool] = None,
        assume_always_shared: Optional[bool] = None,
        allow_serial_map_fusion: Optional[bool] = None,
        allow_parallel_map_fusion: Optional[bool] = None,
        only_if_common_ancestor: Optional[bool] = None,
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
        if allow_serial_map_fusion is not None:
            self.allow_serial_map_fusion = allow_serial_map_fusion
        if allow_parallel_map_fusion is not None:
            self.allow_parallel_map_fusion = allow_parallel_map_fusion
        if only_if_common_ancestor is not None:
            self.only_if_common_ancestor = only_if_common_ancestor
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

        # We have to pass the single use data at construction. This is because that
        #  `fusion._pipeline_results` is only defined, i.e. not `None` during `apply()`
        #  but during `can_be_applied()` it is not available. Thus we have to set it here.
        fusion = MapFusion(
            only_inner_maps=self.only_inner_maps,
            only_toplevel_maps=self.only_toplevel_maps,
            strict_dataflow=self.strict_dataflow,
            assume_always_shared=self.assume_always_shared,
            allow_serial_map_fusion=self.allow_serial_map_fusion,
            allow_parallel_map_fusion=self.allow_parallel_map_fusion,
            only_if_common_ancestor=self.only_if_common_ancestor,
            single_use_data=pipeline_results["FindSingleUseData"]
        )

        try:
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
