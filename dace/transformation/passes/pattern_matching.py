# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains functions related to pattern matching in transformations. """

import collections
import copy
from dataclasses import dataclass
import time

from dace import properties
from dace.config import Config
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import graph as gr, nodes as nd
import networkx as nx
from networkx.algorithms import isomorphism as iso
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Type, Union
from dace.sdfg.validation import InvalidSDFGError
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

    permissive = properties.Property(dtype=bool,
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

    def depends_on(self) -> Set[Type[ppl.Pass]]:
        result = set()
        for p in self.transformations:
            result.update(p.depends_on())
        return result

    def modifies(self) -> ppl.Modifies:
        result = ppl.Modifies.Nothing
        for p in self.transformations:
            result |= p.modifies()
        return result

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return any(p.should_reapply(modified) for p in self.transformations)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict[str, List[Any]]:
        applied_transformations = collections.defaultdict(list)

        # For every transformation in the list, find first match and apply
        for xform in self.transformations:
            # Find only the first match
            try:
                match = next(m for m in match_patterns(
                    sdfg, [xform], metadata=self._metadata, permissive=self.permissive, states=self.states))
            except StopIteration:
                continue

            tsdfg = sdfg.sdfg_list[match.sdfg_id]
            graph = tsdfg.node(match.state_id) if match.state_id >= 0 else tsdfg

            # Set previous pipeline results
            match._pipeline_results = pipeline_results

            result = match.apply(graph, tsdfg)
            applied_transformations[type(match).__name__].append(result)
            if self.validate_all:
                sdfg.validate()

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

    def __init__(self,
                 transformations: Union[xf.PatternTransformation, Iterable[xf.PatternTransformation]],
                 permissive: bool = False,
                 validate: bool = True,
                 validate_all: bool = False,
                 states: Optional[List[SDFGState]] = None,
                 print_report: Optional[bool] = None,
                 progress: Optional[bool] = None,
                 order_by_transformation: bool = True) -> None:
        super().__init__(transformations, permissive, validate, validate_all, states, print_report, progress)
        self.order_by_transformation = order_by_transformation

    # Helper function for applying and validating a transformation
    def _apply_and_validate(self, match: xf.PatternTransformation, sdfg: SDFG, start: float,
                            pipeline_results: Dict[str, Any], applied_transformations: Dict[str, Any]):
        tsdfg = sdfg.sdfg_list[match.sdfg_id]
        graph = tsdfg.node(match.state_id) if match.state_id >= 0 else tsdfg

        # Set previous pipeline results
        match._pipeline_results = pipeline_results

        if self.validate_all:
            match_name = match.print_match(tsdfg)

        if Config.get('debugpass') == True:
            original_sdfg = copy.deepcopy(sdfg)
            sdfg_name = f"{original_sdfg.label}_{str(time.time()).replace('.', '_')}.sdfg"
            try:
                applied_transformations[type(match).__name__].append(match.apply(graph, tsdfg))
            except Exception as e:
                original_sdfg.save(sdfg_name)
                print(f'Exception occured when applying {type(match).__name__} on SDFG {match.sdfg_id} and '
                      f'SDFGState{match.state_id}.')
                print(f'Last correct SDFG: {sdfg_name}')
                raise e
            finally:
                try:
                    sdfg.validate()
                except Exception as e:
                    original_sdfg.save(sdfg_name)
                    print(f'Validation failed after applying {type(match).__name__} on SDFG {match.sdfg_id} and '
                          f'SDFGState{match.state_id}.')
                    print(f'Last correct SDFG: {sdfg_name}')
                    raise e
        else:
            applied_transformations[type(match).__name__].append(match.apply(graph, tsdfg))
        if self.progress or (self.progress is None and (time.time() - start) > 5):
            print('Applied {}.\r'.format(', '.join(['%d %s' % (len(v), k)
                                                    for k, v in applied_transformations.items()])),
                  end='')
        if self.validate_all:
            try:
                sdfg.validate()
            except InvalidSDFGError as err:
                raise InvalidSDFGError(
                    f'Validation failed after applying {match_name}. '
                    f'{type(err).__name__}: {err}', sdfg, match.state_id) from err

    def _apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any], apply_once: bool, func=None, args=None) -> Dict[str, List[Any]]:
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
            applied_anything = True
            while applied_anything:
                applied_anything = False
                for xform in xforms:
                    applied = True
                    while applied:
                        applied = False
                        for match in match_patterns(sdfg,
                                                    permissive=self.permissive,
                                                    patterns=[xform],
                                                    states=self.states,
                                                    metadata=self._metadata):
                            if func is not None:
                                sdfg.save('before.sdfg')
                            self._apply_and_validate(match, sdfg, start, pipeline_results, applied_transformations)
                            if func is not None and not func(sdfg, *args):
                                sdfg.save('after.sdfg')
                                raise RuntimeError('Validation failed after applying {}.'.format(match.print_match(sdfg)))
                            applied = True
                            applied_anything = True
                            break
                if apply_once:
                    break
        else:
            applied = True
            while applied:
                applied = False
                # Find and apply one of the chosen transformations
                for match in match_patterns(sdfg,
                                            permissive=self.permissive,
                                            patterns=xforms,
                                            states=self.states,
                                            metadata=self._metadata):
                    if func is not None:
                        sdfg.save('before.sdfg')
                    self._apply_and_validate(match, sdfg, start, pipeline_results, applied_transformations)
                    if func is not None and not func(sdfg, *args):
                        sdfg.save('after.sdfg')
                        raise RuntimeError('Validation failed after applying {}.'.format(match.print_match(sdfg)))
                    applied = True
                    break
                if apply_once:
                    break

        if self.validate:
            try:
                sdfg.validate()
            except InvalidSDFGError as err:
                if applied and match is not None:
                    raise InvalidSDFGError("Validation failed after applying {}.".format(match.print_match(self)), self,
                                           match.state_id) from err
                else:
                    raise err

        if (len(applied_transformations) > 0
                and (self.progress or self.print_report or
                     ((self.progress is None or self.print_report is None) and Config.get_bool('debugprint')))):
            print('Applied {}.'.format(', '.join(['%d %s' % (len(v), k) for k, v in applied_transformations.items()])))

        if len(applied_transformations) == 0:  # Signal that no transformation was applied
            return None
        return applied_transformations

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any], func=None, args=None) -> Dict[str, List[Any]]:
        return self._apply_pass(sdfg, pipeline_results, apply_once=False, func=func, args=args)


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


def _try_to_match_transformation(graph: Union[SDFG, SDFGState], collapsed_graph: nx.DiGraph, subgraph: Dict[int, int],
                                 sdfg: SDFG, xform: Union[xf.PatternTransformation, Type[xf.PatternTransformation]],
                                 expr_idx: int, nxpattern: nx.DiGraph, state_id: int, permissive: bool,
                                 options: Dict[str, Any]) -> Optional[xf.PatternTransformation]:
    """ 
    Helper function that tries to instantiate a pattern match into a 
    transformation object. 
    """
    subgraph = {
        nxpattern.nodes[j]['node']: graph.node_id(collapsed_graph.nodes[i]['node'])
        for i, j in subgraph.items()
    }

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

        match.setup_match(sdfg, sdfg.sdfg_id, state_id, subgraph, expr_idx, options=options)
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
                   options: Optional[List[Dict[str, Any]]] = None):
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
    sdfgs = sdfg.all_sdfgs_recursive()

    # Try to find transformations on each SDFG
    for tsdfg in sdfgs:
        ###################################
        # Match inter-state transformations
        if len(interstate_transformations) > 0:
            # Collapse multigraph into directed graph in order to use VF2
            digraph = collapse_multigraph_to_nx(tsdfg)

        for xform, expr_idx, nxpattern, matcher, opts in interstate_transformations:
            for subgraph in matcher(digraph, nxpattern, node_match, edge_match):
                match = _try_to_match_transformation(tsdfg, digraph, subgraph, tsdfg, xform, expr_idx, nxpattern, -1,
                                                     permissive, opts)
                if match is not None:
                    yield match

        ####################################
        # Match single-state transformations
        if len(singlestate_transformations) == 0:
            continue
        for state_id, state in enumerate(tsdfg.nodes()):
            if states is not None and state not in states:
                continue

            # Collapse multigraph into directed graph in order to use VF2
            digraph = collapse_multigraph_to_nx(state)

            for xform, expr_idx, nxpattern, matcher, opts in singlestate_transformations:
                for subgraph in matcher(digraph, nxpattern, node_match, edge_match):
                    match = _try_to_match_transformation(state, digraph, subgraph, tsdfg, xform, expr_idx, nxpattern,
                                                         state_id, permissive, opts)
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
