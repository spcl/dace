# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains functions related to pattern matching in transformations. """

import collections
from dataclasses import dataclass
import time
import warnings

from dace import properties
from dace.config import Config
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import graph as gr, nodes as nd
from dace.sdfg.state import AbstractControlFlowRegion, ControlFlowRegion
from dace import graphlib as nx
from dace.graphlib import isomorphism as iso
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Type, Union
from dace.sdfg.validation import InvalidSDFGError, validate_state
from dace.transformation import transformation as xf, pass_pipeline as ppl


@dataclass
@properties.make_properties
class PatternMatchAndApply(ppl.Pass):
    """
    Applies a list of pattern-matching transformations in sequence. For every given transformation, matches the first
    pattern in the SDFG and applies it.
    """

    CATEGORY: str = 'Helper'

    transformations = properties.ListProperty(element_type=xf.PatternTransformation,
                                              default=[],
                                              desc='The list of transformations to apply')

    permissive = properties.Property(
        dtype=bool,
        default=False,
        desc='Whether to apply in permissive mode, i.e., apply in more cases where it may be unsafe.')
    validate = properties.Property(dtype=bool,
                                   default=True,
                                   desc='If True, validates the SDFG after all transformations have been applied.')
    validate_all = properties.Property(dtype=bool,
                                       default=False,
                                       desc='If True, validates the SDFG after each transformation applies.')
    states = properties.ListProperty(element_type=SDFGState,
                                     default=None,
                                     allow_none=True,
                                     desc='If not None, only applies transformations to the given states.')

    print_report = properties.Property(dtype=bool,
                                       default=None,
                                       allow_none=True,
                                       desc='Whether to show debug prints (or None to use configuration file).')
    progress = properties.Property(dtype=bool,
                                   default=None,
                                   allow_none=True,
                                   desc='Whether to show progress printouts (or None to use configuration file).')

    def __init__(self,
                 transformations: Union[xf.PatternTransformation, Iterable[xf.PatternTransformation]],
                 permissive: bool = False,
                 validate: bool = True,
                 validate_all: bool = False,
                 states: Optional[List[SDFGState]] = None,
                 print_report: Optional[bool] = None,
                 progress: Optional[bool] = None) -> None:
        if isinstance(transformations, xf.TransformationBase):
            self.transformations = [transformations]
        else:
            self.transformations = list(transformations)

        # Precompute metadata on each transformation (how to apply it)
        self._metadata = get_transformation_metadata(self.transformations)

        self.permissive = permissive
        self.validate = validate
        self.validate_all = validate_all
        self.states = states
        self.print_report = print_report
        self.progress = progress

    def depends_on(self) -> List[Union[Type[ppl.Pass], ppl.Pass]]:
        return ppl.unique_dependencies(self.transformations)

    def modifies(self) -> ppl.Modifies:
        result = ppl.Modifies.Nothing
        for p in self.transformations:
            result |= p.modifies()
        return result

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return any(p.should_reapply(modified) for p in self.transformations)

    def validate_after_match(self, match: xf.PatternTransformation, graph: Union[SDFG, SDFGState], sdfg: SDFG) -> None:
        """Check what the transformation that just applied could actually have broken.

        A full ``sdfg.validate()`` after EVERY match is O(whole SDFG) per application, which
        makes ``validate_all`` cost more than the transformations it is watching. A
        `SingleStateTransformation` rewrites one state, so that state is checked on its own;
        anything that can move interstate edges, symbols or descriptors around still gets the
        full check. The end-of-pass ``validate`` (on by default) stays the whole-SDFG net, so
        a cross-state break is still caught, just at the end of the pass rather than at the
        match that caused it.
        """
        if not isinstance(match, xf.SingleStateTransformation) or match.state_id < 0:
            sdfg.validate()
            return
        # The match may live in a nested SDFG, and validate_state rejects a state whose .sdfg is
        #  not the one passed in, so the owning SDFG is the one to check against -- not the root.
        owner = graph.sdfg
        # A single state has no cross-state context: a transient another state initialized
        #  looks uninitialized here, which warns -- and raises outright for a Reference. Seed
        #  every descriptor as initialized so only the state-local invariants are checked
        #  (connectors, memlets, scopes, views, subsets), which is what a dataflow
        #  transformation can break.
        validate_state(graph, graph.parent_graph.node_id(graph), owner, initialized_transients=set(owner.arrays.keys()))

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict[str, List[Any]]:
        applied_transformations = collections.defaultdict(list)

        # For every transformation in the list, find first match and apply
        for xform in self.transformations:
            if sdfg.root_sdfg.using_explicit_control_flow:
                if not xform.__explicit_cf_compatible__:
                    warnings.warn('Pattern matching is skipping transformation ' + xform.__class__.__name__ +
                                  ' due to incompatibility with experimental control flow blocks. If the ' +
                                  'SDFG does not contain experimental blocks, ensure the top level SDFG does ' +
                                  'not have `SDFG.using_explicit_control_flow` set to True. If ' +
                                  xform.__class__.__name__ + ' is compatible with experimental blocks, ' +
                                  'please annotate it with the class decorator ' +
                                  '`@dace.transformation.explicit_cf_compatible`. see ' +
                                  '`https://github.com/spcl/dace/wiki/Experimental-Control-Flow-Blocks` ' +
                                  'for more information.')
                    continue

            # Find only the first match
            try:
                match = next(m for m in match_patterns(sdfg, [xform],
                                                       metadata=self._metadata,
                                                       permissive=self.permissive,
                                                       states=self.states,
                                                       pipeline_results=pipeline_results))
            except StopIteration:
                continue

            tcfg = sdfg.cfg_list[match.cfg_id]
            graph = tcfg.node(match.state_id) if match.state_id >= 0 else tcfg

            # Set previous pipeline results
            match._pipeline_results = pipeline_results
            match.permissive = self.permissive

            result = match.apply(graph, tcfg.sdfg)
            applied_transformations[type(match).__name__].append(result)
            if self.validate_all:
                self.validate_after_match(match, graph, sdfg)

        if self.validate:
            sdfg.validate()

        if (len(applied_transformations) > 0
                and (self.print_report or (self.print_report is None and Config.get_bool('debugprint')))):
            print('Applied {}.'.format(', '.join(['%d %s' % (len(v), k) for k, v in applied_transformations.items()])))

        if len(applied_transformations) == 0:  # Signal that no transformation was applied
            return None
        return applied_transformations


@dataclass
@properties.make_properties
class PatternMatchAndApplyRepeated(PatternMatchAndApply):
    """
    A fixed-point pipeline that applies a list of pattern-matching transformations in repeated succession until no
    more transformations match. The order in which the transformations are applied is configurable (through
    ``order_by_transformation``).
    """

    CATEGORY: str = 'Helper'

    order_by_transformation = properties.Property(dtype=bool,
                                                  default=True,
                                                  desc='Whether or not to order by transformation.')

    state_local = properties.Property(
        dtype=bool,
        default=False,
        desc='The transformations are single-state and STATE-LOCAL: a match is decided by its own state '
        'alone, and applying it rewrites only that state and adds blocks to its region. Then, after an '
        'application, every state the walk passed is unchanged and still refused, so the walk resumes in '
        'the region of the match instead of the root, skipping the states it already refused -- the same '
        'matches, applied in the same order, without re-walking the SDFG per application.')

    def __init__(self,
                 transformations: Union[xf.PatternTransformation, Iterable[xf.PatternTransformation]],
                 permissive: bool = False,
                 validate: bool = True,
                 validate_all: bool = False,
                 states: Optional[List[SDFGState]] = None,
                 print_report: Optional[bool] = None,
                 progress: Optional[bool] = None,
                 order_by_transformation: bool = True,
                 state_local: bool = False) -> None:
        super().__init__(transformations, permissive, validate, validate_all, states, print_report, progress)
        self.order_by_transformation = order_by_transformation
        self.state_local = state_local

    # Helper function for applying and validating a transformation
    def _apply_and_validate(self, match: xf.PatternTransformation, sdfg: SDFG, start: float,
                            pipeline_results: Dict[str, Any], applied_transformations: Dict[str, Any]):
        tcfg = sdfg.cfg_list[match.cfg_id]
        graph = tcfg.node(match.state_id) if match.state_id >= 0 else tcfg

        # Set previous pipeline results
        match._pipeline_results = pipeline_results
        match.permissive = self.permissive

        if self.validate_all:
            match_name = match.print_match(tcfg)

        applied_transformations[type(match).__name__].append(match.apply(graph, tcfg.sdfg))
        if self.progress or (self.progress is None and (time.time() - start) > 5):
            print('Applied {}.\r'.format(', '.join(['%d %s' % (len(v), k)
                                                    for k, v in applied_transformations.items()])),
                  end='')
        if self.validate_all:
            try:
                self.validate_after_match(match, graph, sdfg)
            except InvalidSDFGError as err:
                # ``match.state_id`` indexes ``tcfg``, not the SDFG.
                raise InvalidSDFGError(
                    f'Validation failed after applying {match_name}. '
                    f'{type(err).__name__}: {err}',
                    sdfg,
                    match.state_id,
                    cfg=tcfg) from err

    def _apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any], apply_once: bool) -> Dict[str, List[Any]]:
        """
        Internal apply pass method that can run once through the graph or repeatedly.
        """
        if self.progress is None and not Config.get_bool('progress'):
            self.progress = False

        start = time.time()

        applied_transformations = collections.defaultdict(list)
        xforms = self.transformations
        match: Optional[xf.PatternTransformation] = None

        # Ensure transformations are unique
        if len(xforms) != len(set(xforms)):
            raise ValueError('Transformation set must be unique')

        if self.order_by_transformation:
            # `match_patterns()` matches on `self._metadata`, which covers every transformation of
            # this pass, and ignores its `patterns` argument. A loop per transformation therefore
            # enumerates the same matches and applies them in the same order as the loop below, and
            # only adds enumerations that apply nothing: one per remaining transformation, plus a
            # full round of them once anything applied. The loop here keeps the warning those
            # enumerations would have emitted.
            for xform in xforms:
                if sdfg.root_sdfg.using_explicit_control_flow:
                    if not xform.__explicit_cf_compatible__:
                        warnings.warn('Pattern matching is skipping transformation ' + xform.__class__.__name__ +
                                      ' due to incompatibility with experimental control flow blocks. If the ' +
                                      'SDFG does not contain experimental blocks, ensure the top level SDFG does ' +
                                      'not have `SDFG.using_explicit_control_flow` set to True. If ' +
                                      xform.__class__.__name__ + ' is compatible with experimental blocks, ' +
                                      'please annotate it with the class decorator ' +
                                      '`@dace.transformation.explicit_cf_compatible`. see ' +
                                      '`https://github.com/spcl/dace/wiki/Experimental-Control-Flow-Blocks` ' +
                                      'for more information.')

        applied = not self.state_local
        if self.state_local:
            self.apply_state_local(sdfg, start, pipeline_results, applied_transformations)
        while applied:
            applied = False
            matched_pattern = next(
                match_patterns(sdfg,
                               permissive=self.permissive,
                               patterns=xforms,
                               states=self.states,
                               metadata=self._metadata,
                               pipeline_results=pipeline_results), None)
            if matched_pattern is not None:
                self._apply_and_validate(matched_pattern, sdfg, start, pipeline_results, applied_transformations)
                applied = True

        if self.validate:
            try:
                sdfg.validate()
            except InvalidSDFGError as err:
                if applied and matched_pattern is not None:
                    # Defensive: unreachable -- ``applied`` is always False here -- but kept correct.
                    tcfg = sdfg.cfg_list[matched_pattern.cfg_id]
                    raise InvalidSDFGError(f'Validation failed after applying {matched_pattern.print_match(tcfg)}.',
                                           sdfg,
                                           matched_pattern.state_id,
                                           cfg=tcfg) from err
                else:
                    raise err

        if len(applied_transformations) == 0:
            return None

        return applied_transformations

    def apply_state_local(self, sdfg: SDFG, start: float, pipeline_results: Dict[str, Any],
                          applied_transformations: Dict[str, Any]) -> None:
        """The ``state_local`` fixpoint: ``match_patterns``' walk, resumed after each application.

        The walk is ``all_control_flow_regions(recursive=True)`` order -- a region's own states, then
        the regions below it -- kept as a stack so it can resume at the region of the last match.
        """
        interstate, singlestate = self._metadata
        if interstate or not all(isinstance(x, xf.SingleStateTransformation) for x in self.transformations):
            raise ValueError('state_local matching takes single-state transformations only')
        refused: Dict[SDFGState, None] = {}
        # Each frame is ``[region, child regions or None while its own states are walked, next child]``.
        stack: List[list] = [[sdfg, None, 0]]
        while stack:
            frame = stack[-1]
            region, children, index = frame
            if children is None:
                match = self.first_state_match(sdfg, region, singlestate, refused, pipeline_results)
                if match is not None:
                    self._apply_and_validate(match, sdfg, start, pipeline_results, applied_transformations)
                else:
                    frame[1] = child_regions(region)
                continue
            if index == len(children):
                stack.pop()
                continue
            frame[2] = index + 1
            stack.append([children[index], None, 0])

    def first_state_match(self, sdfg: SDFG, region: ControlFlowRegion, singlestate: 'TransformationData',
                          refused: Dict[SDFGState,
                                        None], pipeline_results: Dict[str, Any]) -> Optional[xf.PatternTransformation]:
        """The first match among ``region``'s own states, marking every state that has none as refused."""
        cfg_ids = CfgIds(sdfg)
        for state_id, state in enumerate(region.nodes()):
            if not isinstance(state, SDFGState) or state in refused or (self.states is not None
                                                                        and state not in self.states):
                continue
            match = next(
                state_matches(state, state_id, region, singlestate, type_match, None, self.permissive, pipeline_results,
                              cfg_ids), None)
            if match is not None:
                return match
            refused[state] = None
        return None

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict[str, List[Any]]:
        return self._apply_pass(sdfg, pipeline_results, apply_once=False)


def child_regions(region: ControlFlowRegion) -> List[ControlFlowRegion]:
    """The regions ``all_control_flow_regions(recursive=True)`` descends into below ``region``, in order.

    :param region: The region whose children to list.
    :returns: Nested SDFGs held by the region's states and its sub-regions, in block order.
    """
    children = []
    for block in region.nodes():
        if isinstance(block, SDFGState):
            children.extend(node.sdfg for node in block.nodes() if isinstance(node, nd.NestedSDFG) and node.sdfg)
        elif isinstance(block, AbstractControlFlowRegion):
            children.append(block)
    return children


@dataclass
@properties.make_properties
class PatternApplyOnceEverywhere(PatternMatchAndApplyRepeated):
    """
    A pass pipeline that applies all given transformations once, in every location that their pattern matched.
    If match condition becomes False (e.g., as a result of applying a transformation), the transformation is not
    applied on that location.
    """

    CATEGORY: str = 'Helper'

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict[str, List[Any]]:
        return self._apply_pass(sdfg, pipeline_results, apply_once=True)


def collapse_multigraph_to_nx(graph: Union[gr.MultiDiGraph, gr.OrderedMultiDiGraph]) -> nx.DiGraph:
    """ Collapses a directed multigraph into a networkx directed graph.

        In the output directed graph, each node is a number, which contains
        itself as node_data['node'], while each edge contains a list of the
        data from the original edges as its attribute (edge_data[0...N]).

        :param graph: Directed multigraph object to be collapsed.
        :return: Collapsed directed graph object.
    """

    # Create the digraph nodes.
    digraph_nodes: List[Tuple[int, Dict[str, nd.Node]]] = ([None] * graph.number_of_nodes())
    node_id = {}
    for i, node in enumerate(graph.nodes()):
        digraph_nodes[i] = (i, {'node': node})
        node_id[node] = i

    # Create the digraph edges.
    digraph_edges = {}
    for edge in graph.edges():
        src = node_id[edge.src]
        dest = node_id[edge.dst]

        if (src, dest) in digraph_edges:
            edge_num = len(digraph_edges[src, dest])
            digraph_edges[src, dest].update({edge_num: edge.data})
        else:
            digraph_edges[src, dest] = {0: edge.data}

    # Create the digraph
    result = nx.DiGraph()
    result.add_nodes_from(digraph_nodes)
    result.add_edges_from(digraph_edges)

    return result


def type_match(graph_node, pattern_node):
    """ Checks whether the node types of the inputs match.

        :param graph_node: First node (in matched graph).
        :param pattern_node: Second node (in pattern subgraph).
        :return: True if the object types of the nodes match, False otherwise.
        :raise TypeError: When at least one of the inputs is not a dictionary
                          or does not have a 'node' attribute.
        :raise KeyError: When at least one of the inputs is a dictionary,
                         but does not have a 'node' key.
    """
    if isinstance(pattern_node['node'], xf.PatternNode):
        return isinstance(graph_node['node'], pattern_node['node'].node)
    return isinstance(graph_node['node'], type(pattern_node['node']))


def pattern_types_present(nxpattern: nx.DiGraph, present: set) -> bool:
    """Whether every node of ``nxpattern`` has a node of a matching type (per :func:`type_match`) among
    the node types ``present`` in a graph -- the precondition for ``nxpattern`` to match there.

    :param nxpattern: A collapsed pattern graph, as :func:`get_transformation_metadata` builds it.
    :param present: The set of ``type(node)`` over the graph's nodes.
    """
    for pnid in nxpattern:
        pnode = nxpattern.nodes[pnid]['node']
        required = pnode.node if isinstance(pnode, xf.PatternNode) else type(pnode)
        if not any(issubclass(t, required) for t in present):
            return False
    return True


def type_or_class_match(node_a, node_b):
    """
    Checks whether `node_a` is an instance of the same type as `node_b`, or
    if either `node_a`/`node_b` is a type and the other is an instance of that
    type. This is used in subgraph matching to allow the subgraph pattern to
    be either a graph of instantiated nodes, or node types.

    :param node_a: First node.
    :param node_b: Second node.
    :return: True if the object types of the nodes match according to the
             description, False otherwise.
    :raise TypeError: When at least one of the inputs is not a dictionary
                        or does not have a 'node' attribute.
    :raise KeyError: When at least one of the inputs is a dictionary,
                        but does not have a 'node' key.
    :see: enumerate_matches
    """
    if isinstance(node_b['node'], type):
        return issubclass(type(node_a['node']), node_b['node'])
    elif isinstance(node_a['node'], type):
        return issubclass(type(node_b['node']), node_a['node'])
    elif isinstance(node_b['node'], xf.PatternNode):
        return isinstance(node_a['node'], node_b['node'].node)
    elif isinstance(node_a['node'], xf.PatternNode):
        return isinstance(node_b['node'], node_a['node'].node)
    return isinstance(node_a['node'], type(node_b['node']))


class CfgIds:
    """``cfg_id`` of every region of one CFG tree, from one pass over its ``cfg_list``.

    ``ControlFlowRegion.cfg_id`` is ``cfg_list.index(self)``, a linear scan; asked per candidate it
    made a matcher sweep quadratic in the region count (warpx_field_gather: 13000 regions, 3960
    ``ConditionFusion`` candidates per sweep). A region holding another list resolves as before.
    """
    __slots__ = ('cfg_list', 'index')

    def __init__(self, sdfg: SDFG) -> None:
        self.cfg_list = sdfg.cfg_list
        #: Built on the first lookup: a sweep that matches its first candidate needs one scan at most.
        self.index: Optional[Dict[ControlFlowRegion, int]] = None

    def cfg_id(self, region: ControlFlowRegion) -> int:
        """``region.cfg_id``, without the scan when ``region`` shares the indexed list."""
        if region.cfg_list is not self.cfg_list:
            return region.cfg_id
        if self.index is None:
            self.index = {}
            for i, cfg in enumerate(self.cfg_list):
                self.index.setdefault(cfg, i)
        found = self.index.get(region)
        return region.cfg_id if found is None else found


def _try_to_match_transformation(graph: Union[ControlFlowRegion, SDFGState],
                                 collapsed_graph: nx.DiGraph,
                                 subgraph: Dict[int, int],
                                 sdfg: SDFG,
                                 xform: Union[xf.PatternTransformation, Type[xf.PatternTransformation]],
                                 expr_idx: int,
                                 nxpattern: nx.DiGraph,
                                 state_id: int,
                                 permissive: bool,
                                 options: Dict[str, Any],
                                 pipeline_results: Optional[Dict[str, Any]] = None,
                                 cfg_ids: Optional['CfgIds'] = None) -> Optional[xf.PatternTransformation]:
    """
    Helper function that tries to instantiate a pattern match into a
    transformation object.

    :param pipeline_results: Results of the passes this one declared through ``depends_on()``,
                             installed on the match BEFORE ``can_be_applied`` so a predicate can
                             read a cached analysis instead of recomputing it per candidate. This
                             is what issue#1911 is about; ``setup_match`` resets the member, so it
                             has to be set after that call.
    :param cfg_ids: The CFG list indexed once by the caller, to resolve the region's ``cfg_id``.
    """
    # `collapse_multigraph_to_nx` numbers the nodes in the order of `graph.nodes()`, so the index of
    # a node in the collapsed graph is its node ID; `graph.node_id` would find it by a linear scan.
    subgraph = {nxpattern.nodes[j]['node']: i for i, j in subgraph.items()}

    try:
        if isinstance(xform, xf.PatternTransformation):
            match = xform
        else:  # Construct directly from type with options
            opts = options or {}
            try:
                match = xform(**opts)
            except TypeError:
                # Backwards compatibility, transformation does not support ctor arguments
                match = xform()
                # Set manually
                for oname, oval in opts.items():
                    setattr(match, oname, oval)

        if sdfg.root_sdfg.using_explicit_control_flow:
            if not match.__explicit_cf_compatible__:
                warnings.warn('Pattern matching is skipping transformation ' + match.__class__.__name__ +
                              ' due to incompatibility with experimental control flow blocks. If the ' +
                              'SDFG does not contain experimental blocks, ensure the top level SDFG does ' +
                              'not have `SDFG.using_explicit_control_flow` set to True. If ' +
                              match.__class__.__name__ + ' is compatible with experimental blocks, ' +
                              'please annotate it with the class decorator ' +
                              '`@dace.transformation.explicit_cf_compatible`. see ' +
                              '`https://github.com/spcl/dace/wiki/Experimental-Control-Flow-Blocks` ' +
                              'for more information.')
                return None

        region = graph.parent_graph if isinstance(graph, SDFGState) else graph
        cfg_id = region.cfg_id if cfg_ids is None else cfg_ids.cfg_id(region)
        match.setup_match(sdfg, cfg_id, state_id, subgraph, expr_idx, options=options)
        # After setup_match, which resets it to None.
        match._pipeline_results = pipeline_results
        match_found = match.can_be_applied(graph, expr_idx, sdfg, permissive=permissive)
    except Exception as e:
        if Config.get_bool('optimizer', 'match_exception'):
            raise
        if not isinstance(xform, type):
            xft = type(xform)
        else:
            xft = xform
        print('WARNING: {p}::can_be_applied triggered a {c} exception:'
              ' {e}'.format(p=xft.__name__, c=e.__class__.__name__, e=e))
        return None

    if match_found:
        return match

    return None


TransformationData = List[Tuple[Type[xf.PatternTransformation], int, nx.DiGraph, Callable, Dict[str, Any]]]
PatternMetadataType = Tuple[TransformationData, TransformationData]


def get_transformation_metadata(patterns: List[Type[xf.PatternTransformation]],
                                options: Optional[List[Dict[str, Any]]] = None) -> PatternMetadataType:
    """
    Collect all transformation expressions and metadata once, for use when
    applying transformations repeatedly.

    :param patterns: PatternTransformation type (or list thereof) to compute.
    :param options: An optional list of transformation parameter dictionaries.
    :return: A tuple of inter-state and single-state pattern matching
             transformations.
    """
    if options is None:
        options = [None] * len(patterns)

    singlestate_transformations: TransformationData = []
    interstate_transformations: TransformationData = []
    for pattern, opts in zip(patterns, options):
        # Find if the transformation is inter-state
        is_interstate = (isinstance(pattern, xf.MultiStateTransformation)
                         or (isinstance(pattern, type) and issubclass(pattern, xf.MultiStateTransformation)))
        for i, expr in enumerate(pattern.expressions()):
            # Make a networkx-version of the match subgraph
            nxpattern = collapse_multigraph_to_nx(expr)
            if len(nxpattern.nodes) == 1:
                matcher = _node_matcher
            elif len(nxpattern.nodes) == 2 and len(nxpattern.edges) == 1:
                matcher = _edge_matcher
            elif len(nxpattern.nodes) == 2 and len(nxpattern.edges) == 0:
                matcher = _unconnected_pair_matcher
            else:
                matcher = _subgraph_isomorphism_matcher

            if is_interstate:
                interstate_transformations.append((pattern, i, nxpattern, matcher, opts))
            else:
                singlestate_transformations.append((pattern, i, nxpattern, matcher, opts))

    return interstate_transformations, singlestate_transformations


def _subgraph_isomorphism_matcher(digraph, nxpattern, node_pred, edge_pred):
    """ Match based on the VF2 algorithm for general SI. """
    graph_matcher = iso.DiGraphMatcher(digraph, nxpattern, node_match=node_pred, edge_match=edge_pred)
    yield from graph_matcher.subgraph_isomorphisms_iter()


def _node_matcher(digraph, nxpattern, node_pred, edge_pred):
    """ Match individual nodes. """
    pnid = next(iter(nxpattern))
    pnode = nxpattern.nodes[pnid]

    for nid in digraph:
        if node_pred(digraph.nodes[nid], pnode):
            yield {nid: pnid}


def _unconnected_pair_matcher(digraph, nxpattern, node_pred, edge_pred):
    """ Match two pattern nodes that are not connected by an edge.

        Yields the same matches in the same order as ``_subgraph_isomorphism_matcher``, whose
        subgraph isomorphisms are induced: ordered pairs of distinct nodes that satisfy the node
        predicate, are not adjacent in either direction and have no self-edge. VF2 runs its
        feasibility test on every graph node for the second pattern node, once per candidate for
        the first, which dominates the matching time on large states.
    """
    first, second = nxpattern
    first_pattern_node = nxpattern.nodes[first]
    second_pattern_node = nxpattern.nodes[second]

    def candidates(pattern_node):
        return [
            nid for nid in digraph if node_pred(digraph.nodes[nid], pattern_node) and not digraph.has_edge(nid, nid)
        ]

    second_candidates = candidates(second_pattern_node)
    for u in candidates(first_pattern_node):
        for v in second_candidates:
            if u is not v and not digraph.has_edge(u, v) and not digraph.has_edge(v, u):
                yield {u: first, v: second}


def _edge_matcher(digraph, nxpattern, node_pred, edge_pred):
    """ Match individual edges. """
    pedge = next(iter(nxpattern.edges))
    pu = nxpattern.nodes[pedge[0]]
    pv = nxpattern.nodes[pedge[1]]

    if edge_pred is None:
        for u, v in digraph.edges:
            if (node_pred(digraph.nodes[u], pu) and node_pred(digraph.nodes[v], pv)):
                if u is v:  # Skip self-edges
                    continue
                yield {u: pedge[0], v: pedge[1]}
    else:
        for u, v in digraph.edges:
            if (node_pred(digraph.nodes[u], pu) and node_pred(digraph.nodes[v], pv)
                    and edge_pred(digraph.edges[u, v], nxpattern.edges[pedge])):
                if u is v:  # Skip self-edges
                    continue
                yield {u: pedge[0], v: pedge[1]}


def match_patterns(sdfg: SDFG,
                   patterns: Union[Type[xf.PatternTransformation], List[Type[xf.PatternTransformation]]],
                   node_match: Callable[[Any, Any], bool] = type_match,
                   edge_match: Optional[Callable[[Any, Any], bool]] = None,
                   permissive: bool = False,
                   metadata: Optional[PatternMetadataType] = None,
                   states: Optional[List[SDFGState]] = None,
                   options: Optional[List[Dict[str, Any]]] = None,
                   pipeline_results: Optional[Dict[str, Any]] = None):
    """ Returns a generator of Transformations that match the input SDFG.
        Ordered by SDFG ID.

        :param sdfg: The SDFG to match in.
        :param patterns: PatternTransformation type (or list thereof) to match.
        :param node_match: Function for checking whether two nodes match.
        :param edge_match: Function for checking whether two edges match.
        :param permissive: Match transformations in permissive mode.
        :param metadata: Transformation metadata that can be reused.
        :param states: If given, only tries to match single-state
                       transformations on this list.
        :param options: An optional iterable of transformation parameter
                        dictionaries.
        :param pipeline_results: Results of previously-run passes, made visible to each match's
                                 ``can_be_applied`` so a predicate can read a cached analysis
                                 rather than rescanning the SDFG per candidate.
        :return: A list of PatternTransformation objects that match.
    """

    if isinstance(patterns, type):
        patterns = [patterns]
    if isinstance(options, dict):
        options = [options]

    # Collect transformation metadata
    if metadata is not None:
        # Transformation metadata can be evaluated once per apply loop
        interstate_transformations, singlestate_transformations = metadata
    else:
        # Otherwise, precompute all transformation data once
        (interstate_transformations, singlestate_transformations) = get_transformation_metadata(patterns, options)

    # Collect SDFG and nested SDFGs
    cfrs = sdfg.all_control_flow_regions(recursive=True)
    cfg_ids = CfgIds(sdfg)

    # Try to find transformations on each SDFG
    for cfr in cfrs:
        ###################################
        # Match inter-state transformations
        if len(interstate_transformations) > 0:
            # Collapse multigraph into directed graph in order to use VF2
            digraph = collapse_multigraph_to_nx(cfr)

        for xform, expr_idx, nxpattern, matcher, opts in interstate_transformations:
            for subgraph in matcher(digraph, nxpattern, node_match, edge_match):
                match = _try_to_match_transformation(cfr, digraph, subgraph, cfr.sdfg, xform, expr_idx, nxpattern, -1,
                                                     permissive, opts, pipeline_results, cfg_ids)
                if match is not None:
                    yield match

        ####################################
        # Match single-state transformations
        if len(singlestate_transformations) == 0:
            continue
        for state_id, state in enumerate(cfr.nodes()):
            if not isinstance(state, SDFGState) or (states is not None and state not in states):
                continue
            yield from state_matches(state, state_id, cfr, singlestate_transformations, node_match, edge_match,
                                     permissive, pipeline_results, cfg_ids)


def state_matches(state: SDFGState, state_id: int, cfr: ControlFlowRegion,
                  singlestate_transformations: 'TransformationData', node_match: Callable[[Any, Any], bool],
                  edge_match: Optional[Callable[[Any, Any], bool]], permissive: bool,
                  pipeline_results: Optional[Dict[str, Any]], cfg_ids: CfgIds) -> Iterator[xf.PatternTransformation]:
    """The single-state matches in ``state`` (``state_id`` in ``cfr``), in ``match_patterns`` order."""
    candidates = singlestate_transformations
    if node_match is type_match:
        # A pattern whose node types do not all occur in the state cannot match there. Checking that
        # first skips the collapse below, which is the whole cost of a scan over states that hold
        # nothing to match (a restart after every application re-scans every state before the next).
        present = {type(node) for node in state.nodes()}
        candidates = [entry for entry in candidates if pattern_types_present(entry[2], present)]
        if not candidates:
            return

    # Collapse multigraph into directed graph in order to use VF2
    digraph = collapse_multigraph_to_nx(state)

    for xform, expr_idx, nxpattern, matcher, opts in candidates:
        for subgraph in matcher(digraph, nxpattern, node_match, edge_match):
            match = _try_to_match_transformation(state, digraph, subgraph, cfr.sdfg, xform, expr_idx, nxpattern,
                                                 state_id, permissive, opts, pipeline_results, cfg_ids)
            if match is not None:
                yield match


def enumerate_matches(sdfg: SDFG,
                      pattern: gr.Graph,
                      node_match=type_or_class_match,
                      edge_match=None) -> Iterator[gr.SubgraphView]:
    """
    Returns a generator of subgraphs that match the given subgraph pattern.

    :param sdfg: The SDFG to search in.
    :param pattern: A subgraph to look for.
    :param node_match: An optional function to use for matching nodes.
    :param node_match: An optional function to use for matching edges.
    :return: Yields SDFG subgraph view objects.
    """
    if len(pattern.nodes()) == 0:
        raise ValueError('Subgraph pattern cannot be empty')

    # Find if the subgraph is within states or SDFGs
    is_interstate = (isinstance(pattern.node(0), SDFGState)
                     or (isinstance(pattern.node(0), type) and pattern.node(0) is SDFGState))

    # Collapse multigraphs into directed graphs
    pattern_digraph = collapse_multigraph_to_nx(pattern)

    # Find matches in all SDFGs and nested SDFGs
    for graph in sdfg.all_sdfgs_recursive():
        if is_interstate:
            graph_matcher = iso.DiGraphMatcher(collapse_multigraph_to_nx(graph),
                                               pattern_digraph,
                                               node_match=node_match,
                                               edge_match=edge_match)
            for subgraph in graph_matcher.subgraph_isomorphisms_iter():
                yield gr.SubgraphView(graph, [graph.node(i) for i in subgraph.keys()])
        else:
            for state in graph.nodes():
                graph_matcher = iso.DiGraphMatcher(collapse_multigraph_to_nx(state),
                                                   pattern_digraph,
                                                   node_match=node_match,
                                                   edge_match=edge_match)
                for subgraph in graph_matcher.subgraph_isomorphisms_iter():
                    yield gr.SubgraphView(state, [state.node(i) for i in subgraph.keys()])
