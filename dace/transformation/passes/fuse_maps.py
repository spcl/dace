# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import collections
import warnings
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from dace import SDFG, SDFGState, properties, transformation
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion
from dace.sdfg.validation import validate_state
from dace.transformation import pass_pipeline as ppl, dataflow as dftrans
from dace.transformation import transformation as xf
from dace.transformation.passes import analysis as ap

#: Rounds of (vertical -> horizontal) fusion. Two, not a fixpoint: the second round is what the
#: first enables, and a fixpoint pays a third round that only confirms convergence.
FUSE_ROUNDS = 2

#: One expression: the pattern nodes in declaration order, and the edges required between them.
PatternShape = Tuple[List[xf.PatternNode], Set[Tuple[int, int]]]

#: Applied transformations, by transformation name -- the report `PatternMatchAndApply` returns.
AppliedMap = Dict[str, List[Any]]


def _pattern_shapes(xform: xf.PatternTransformation) -> List[PatternShape]:
    """The declared expressions of ``xform`` as (nodes, edges) index pairs."""
    shapes: List[PatternShape] = []
    for expr in xform.expressions():
        pnodes = list(expr.nodes())
        index = {pn: i for i, pn in enumerate(pnodes)}
        shapes.append((pnodes, {(index[e.src], index[e.dst]) for e in expr.edges()}))
    return shapes


def _induced_matches(state: SDFGState, pnodes: List[xf.PatternNode],
                     pedges: Set[Tuple[int, int]]) -> Iterator[List[nodes.Node]]:
    """Yield the induced matches of one small pattern in ``state``.

    Replaces the VF2 subgraph isomorphism the pattern matcher used to run for these patterns and
    reproduces its enumeration order: pattern node 0 ranges over ``state.nodes()``, every later
    node over the successors of the image of the pattern node that points at it, and a candidate
    is rejected unless the edges induced among the images are exactly the pattern's.
    """
    npat = len(pnodes)
    all_nodes = state.nodes()
    # Successor sets are read many times per enumeration; the generator is recreated for every
    # probe and the caller applies nothing until it is abandoned, so the state cannot change under
    # the cache -- it dies with the generator.
    succ_cache: Dict[nodes.Node, Dict[nodes.Node, None]] = {}

    def successors(node: nodes.Node) -> Dict[nodes.Node, None]:
        cached = succ_cache.get(node)
        if cached is None:
            cached = {e.dst: None for e in state.out_edges(node)}
            succ_cache[node] = cached
        return cached

    # The pattern node whose image supplies the candidates for each level, or None for a free level.
    parents = [next((i for i in range(j) if (i, j) in pedges), None) for j in range(npat)]

    def extend(j: int, images: List[nodes.Node]) -> Iterator[List[nodes.Node]]:
        if j == npat:
            yield list(images)
            return
        candidates = all_nodes if parents[j] is None else successors(images[parents[j]])
        node_type = pnodes[j].node
        for cand in candidates:
            if not isinstance(cand, node_type) or any(cand is img for img in images):
                continue
            if all((cand in successors(img)) == ((i, j) in pedges) and (img in successors(cand)) == ((j, i) in pedges)
                   for i, img in enumerate(images)):
                images.append(cand)
                yield from extend(j + 1, images)
                images.pop()

    yield from extend(0, [])


@properties.make_properties
@transformation.explicit_cf_compatible
class FuseMaps(ppl.Pass):
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
        desc="Only perform fusing if the Maps are inner Maps, i.e., does not have top level scope.",
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
        desc="If `True` then all intermediates need to be 'exclusive', i.e., they will be removed by the fusion.",
    )
    require_all_intermediates = properties.Property(
        dtype=bool,
        default=False,
        desc="If `True` all outputs of the first Map must be intermediate, i.e., going into the second Map.",
    )

    perform_vertical_map_fusion = properties.Property(
        dtype=bool,
        default=True,
        desc="If `True`, the default, then allow vertical Map fusion: `MapReduceFusion` for a Map "
        "feeding a Reduce, then `MapFusionVertical` for a Map feeding a Map.",
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
        perform_vertical_map_fusion: Optional[bool] = None,
        perform_horizontal_map_fusion: Optional[bool] = None,
        only_inner_maps: Optional[bool] = None,
        only_toplevel_maps: Optional[bool] = None,
        strict_dataflow: Optional[bool] = None,
        assume_always_shared: Optional[bool] = None,
        require_exclusive_intermediates: Optional[bool] = None,
        require_all_intermediates: Optional[bool] = None,
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
        if require_all_intermediates is not None:
            self.require_all_intermediates = require_all_intermediates
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

        if not (self.perform_vertical_map_fusion or self.perform_horizontal_map_fusion):
            raise ValueError('Neither perform `MapFusionVertical` nor `MapFusionHorizontal`')
        if not self.perform_vertical_map_fusion:
            unique_vertical_arguments = {
                "strict_dataflow": strict_dataflow,
                "assume_always_shared": assume_always_shared,
                "require_exclusive_intermediates": require_exclusive_intermediates,
                "require_all_intermediates": require_exclusive_intermediates,
            }
            specified_vertical_arguments = [arg for arg, val in unique_vertical_arguments.items() if val is not None]
            if specified_vertical_arguments:
                raise ValueError(
                    f'Used `FuseMaps` without vertical Map fusion, but speciefied: {", ".join(specified_vertical_arguments)}'
                )
        if not self.perform_horizontal_map_fusion:
            if only_if_common_ancestor is not None:
                raise ValueError(
                    f'Used `FuseMaps` without horizontal Map fusion, but speciefied: only_if_common_ancestor')

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.Scopes | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.States)

    def depends_on(self):
        return [ap.FindSingleUseData]

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """
        Fuses all Maps that can be fused in the SDFG, including its nested SDFGs.

        Candidates are enumerated directly -- a producer MapExit whose access node feeds a
        consumer MapEntry for the vertical phase, same-scope Map pairs for the horizontal one --
        instead of paying VF2 subgraph isomorphism to rediscover a structure that is already known.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: The result of previous pipeline steps. If the result of
            `FindSingleUseData`, which `depends_on()` declares, is missing then the pass was
            called standalone and runs the analysis itself.
        :return: The numbers of Maps that were fused or `None` if none were fused.
        """
        if ap.FindSingleUseData.__name__ not in pipeline_results:
            # Called outside a pipeline, which is a supported use; do not mutate the caller's dict.
            pipeline_results = dict(pipeline_results)
            pipeline_results[ap.FindSingleUseData.__name__] = ap.FindSingleUseData().apply_pass(sdfg, {})

        fusion_transforms = []
        if self.perform_vertical_map_fusion:
            # First in the vertical phase: the Map feeding the Reduce is this pattern's first node,
            # so folding it into a neighbour leaves the reduction reading a materialized array that
            # nothing removes afterwards.
            fusion_transforms.append(dftrans.MapReduceFusion())

            # The single-use data reaches `can_be_applied` through `pipeline_results`, installed
            #  on the match before every probe -- not threaded in at construction (issue#1911).
            fusion_transforms.append(
                dftrans.MapFusionVertical(
                    only_inner_maps=self.only_inner_maps,
                    only_toplevel_maps=self.only_toplevel_maps,
                    strict_dataflow=self.strict_dataflow,
                    assume_always_shared=self.assume_always_shared,
                    require_exclusive_intermediates=self.require_exclusive_intermediates,
                    require_all_intermediates=self.require_all_intermediates,
                    consolidate_edges_only_if_not_extending=self.consolidate_edges_only_if_not_extending,
                    never_consolidate_edges=self.never_consolidate_edges,
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

        explicit_cf = sdfg.root_sdfg.using_explicit_control_flow
        units: List[Tuple[xf.PatternTransformation, List[PatternShape]]] = []
        for xform in fusion_transforms:
            if explicit_cf and not xform.__explicit_cf_compatible__:
                warnings.warn(f'Map fusion is skipping {type(xform).__name__} due to incompatibility with '
                              'experimental control flow blocks.')
                continue
            units.append((xform, _pattern_shapes(xform)))

        applied: AppliedMap = collections.defaultdict(list)
        for _ in range(FUSE_ROUNDS):
            for cfg in sdfg.all_control_flow_regions(recursive=True):
                for state_id, state in enumerate(cfg.nodes()):
                    if not isinstance(state, SDFGState):
                        continue
                    for xform, shapes in units:
                        self._drain(xform, shapes, cfg, state, state_id, pipeline_results, applied)

        if self.validate and (not self.validate_all):
            sdfg.validate()

        return applied or None

    def _drain(self, xform: xf.PatternTransformation, shapes: List[PatternShape], cfg: ControlFlowRegion,
               state: SDFGState, state_id: int, pipeline_results: Dict[str, Any], applied: AppliedMap) -> None:
        """Apply ``xform`` in ``state`` until nothing matches there any more.

        The fusions are single-state rewrites reading one fixed `FindSingleUseData` result, so a
        match in another state can neither appear nor vanish here -- draining state by state costs
        one state rescan per application instead of one whole-SDFG rescan.
        """
        name = type(xform).__name__
        owner = cfg.sdfg
        progress = True
        while progress:
            progress = False
            node_id = None
            for expr_index, (pnodes, pedges) in enumerate(shapes):
                for images in _induced_matches(state, pnodes, pedges):
                    if node_id is None:
                        node_id = {node: i for i, node in enumerate(state.nodes())}
                    xform.setup_match(owner, cfg.cfg_id, state_id, dict(zip(pnodes, (node_id[n] for n in images))),
                                      expr_index)
                    # `setup_match` resets it, so the cached analysis is installed after the call.
                    xform._pipeline_results = pipeline_results
                    xform.permissive = False
                    try:
                        matched = xform.can_be_applied(state, expr_index, owner, permissive=False)
                    except Exception as exception:
                        if Config.get_bool('optimizer', 'match_exception'):
                            raise
                        print(f'WARNING: {name}::can_be_applied triggered a '
                              f'{type(exception).__name__} exception: {exception}')
                        continue
                    if not matched:
                        continue
                    applied[name].append(xform.apply(state, owner))
                    if self.validate_all:
                        validate_state(state, state_id, owner, initialized_transients=set(owner.arrays.keys()))
                    progress = True
                    break
                if progress:
                    break
