# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Optional, Set

import warnings

from dace import SDFG, SDFGState, properties, transformation
from dace.transformation import pass_pipeline as ppl, dataflow as dftrans
from dace.transformation.passes import analysis as ap, pattern_matching as pmp


@properties.make_properties
@transformation.explicit_cf_compatible
class FullMapFusion(ppl.Pass):
    """Pass that combines `MapFusionVertical`, `MapFusionHorizonatl` and `FindSingleUseData` into one.

    Essentially, this function runs `FindSingleUseData` before `MapFusion`, this
    will speedup vertical fusion, as the SDFG has to be scanned only once.
    The pass accepts the combined options of `MapFusionVertical` and `MapFusionHorizontal`.
    In addition it also accepts `perform_vertical_map_fusion` and `perform_horizontal_map_fusion`
    flags, both default to `True`. They allow to enable disable the two fusion components.
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
    require_exclusive_intermediates = properties.Property(
        dtype=bool,
        default=False,
        desc="If `True` then all intermediates need to be 'exclusive', i.e. they will be removed by the fusion.",
    )

    perform_vertical_map_fusion = properties.Property(
        dtype=bool,
        default=True,
        desc="If `True`, the default, then allow vertical Map fusion, see `MapFusionVertical`.",
    )
    perform_horizontal_map_fusion = properties.Property(
        dtype=bool,
        default=True,
        desc="If `True`, the default, then also perform horizontal Map fusion, see `MapFusionHorizontal`.",
    )

    only_if_common_ancestor = properties.Property(
        dtype=bool,
        default=False,
        desc="If `True` restrict parallel map fusion to maps that have a direct common ancestor.",
    )

    never_consolidate_edges = properties.Property(
        dtype=bool,
        default=False,
        desc="If `True`, always create a new connector, instead of reusing one that referring to the same data.",
    )
    consolidate_edges_only_if_not_extending = properties.Property(
        dtype=bool,
        default=False,
        desc="Only consolidate if this does not lead to an extension of the subset.",
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
        require_exclusive_intermediates: Optional[bool] = None,
        perform_vertical_map_fusion: Optional[bool] = None,
        perform_horizontal_map_fusion: Optional[bool] = None,
        only_if_common_ancestor: Optional[bool] = None,
        consolidate_edges_only_if_not_extending: Optional[bool] = None,
        never_consolidate_edges: Optional[bool] = None,
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
        if require_exclusive_intermediates is not None:
            self.require_exclusive_intermediates = require_exclusive_intermediates
        if perform_vertical_map_fusion is not None:
            self.perform_vertical_map_fusion = perform_vertical_map_fusion
        if perform_horizontal_map_fusion is not None:
            self.perform_horizontal_map_fusion = perform_horizontal_map_fusion
        if only_if_common_ancestor is not None:
            self.only_if_common_ancestor = only_if_common_ancestor
        if validate is not None:
            self.validate = validate
        if validate_all is not None:
            self.validate_all = validate_all
        if never_consolidate_edges is not None:
            self.never_consolidate_edges = never_consolidate_edges
        if consolidate_edges_only_if_not_extending is not None:
            self.consolidate_edges_only_if_not_extending = consolidate_edges_only_if_not_extending

        # TODO(phimuell): Raise an error if a flag was specified and the component that
        #   needs it is disabled.

        if not (self.perform_vertical_map_fusion or self.perform_horizontal_map_fusion):
            raise ValueError('Neither perform `MapFusionVertical` nor `MapFusionHorizontal`')

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

        fusion_transforms = []
        if self.perform_vertical_map_fusion:
            # We have to pass the single use data at construction. This is because that
            #  `fusion._pipeline_results` is only defined, i.e. not `None` during `apply()`
            #  but during `can_be_applied()` it is not available. Thus we have to set it here.
            fusion_transforms.append(
                dftrans.MapFusionVertical(
                    only_inner_maps=self.only_inner_maps,
                    only_toplevel_maps=self.only_toplevel_maps,
                    strict_dataflow=self.strict_dataflow,
                    assume_always_shared=self.assume_always_shared,
                    require_exclusive_intermediates=self.require_exclusive_intermediates,
                    consolidate_edges_only_if_not_extending=self.consolidate_edges_only_if_not_extending,
                    never_consolidate_edges=self.never_consolidate_edges,
                    # TODO: Remove once issue#1911 has been solved.
                    _single_use_data=pipeline_results["FindSingleUseData"],
                ))

        if self.perform_horizontal_map_fusion:
            # NOTE: If horizontal Map fusion is enable it is important that it runs after vertical
            #   Map fusion. The reason is that it has to check any possible Map pair. Thus, the
            #   number of Maps should be as small as possible.
            fusion_transforms.append(
                dftrans.MapFusionHorizontal(
                    only_inner_maps=self.only_inner_maps,
                    only_toplevel_maps=self.only_toplevel_maps,
                    only_if_common_ancestor=self.only_if_common_ancestor,
                    consolidate_edges_only_if_not_extending=self.consolidate_edges_only_if_not_extending,
                    never_consolidate_edges=self.never_consolidate_edges,
                ))

        pazz = pmp.PatternMatchAndApplyRepeated(
            fusion_transforms,
            permissive=False,
            validate=False,
            validate_all=self.validate_all,
        )
        result = pazz.apply_pass(sdfg, pipeline_results)

        if self.validate and (not self.validate_all):
            sdfg.validate()

        return result if result > 0 else None
