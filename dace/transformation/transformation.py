# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Contains classes that represent data-centric transformations.

There are three general types of transformations:
  * Pattern-matching Transformations (extending Transformation): Transformations
    that require a certain subgraph structure to match.
  * Subgraph Transformations (extending SubgraphTransformation): Transformations
    that can operate on arbitrary subgraphs.
  * Library node expansions (extending ExpandTransformation): An internal class
    used for tracking how library nodes were expanded.
"""

import copy
from dace import dtypes, serialize
from dace.dtypes import ScheduleType
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes as nd, graph as gr, utils as sdutil, propagation, infer_types
from dace.properties import make_properties, Property, DictProperty, SetProperty
from dace.registry import make_registry
from typing import Any, Dict, List, Optional, Set, Type, Union
import pydoc


class TransformationBase(object):
    """ Base class for data-centric transformations. """
    pass


@make_registry
@make_properties
class Transformation(TransformationBase):
    """ Base class for pattern-matching transformations, as well as a static
        registry of transformations, where new transformations can be added in a
        decentralized manner.
        An instance of a Transformation represents a match of the transformation
        on an SDFG, complete with a subgraph candidate and properties.

        New transformations that extend this class must contain static
        `PatternNode` fields that represent the nodes in the pattern graph, and
        use them to implement at least three methods:
          * `expressions`: A method that returns a list of graph
                           patterns (SDFG or SDFGState objects) that match this
                           transformation.
          * `can_be_applied`: A method that, given a subgraph candidate,
                              checks for additional conditions whether it can
                              be transformed.
          * `apply`: A method that applies the transformation
                     on the given SDFG.

        For more information and optimization opportunities, see the respective
        methods' documentation.

        In order to be included in lists and apply through the
        `sdfg.apply_transformations` API, each transformation shouls be
        registered with ``Transformation.register`` (or, more commonly,
        the ``@dace.registry.autoregister_params`` class decorator) with two
        optional boolean keyword arguments: ``singlestate`` (default: False)
        and ``coarsening`` (default: False).
        If ``singlestate`` is True, the transformation is matched on subgraphs
        inside an SDFGState; otherwise, subgraphs of the SDFG state machine are
        matched.
        If ``coarsening`` is True, this transformation will be performed automatically
        as part of SDFG dataflow coarsening.
    """

    # Properties
    sdfg_id = Property(dtype=int, category="(Debug)")
    state_id = Property(dtype=int, category="(Debug)")
    _subgraph = DictProperty(key_type=int, value_type=int, category="(Debug)")
    expr_index = Property(dtype=int, category="(Debug)")

    def annotates_memlets(self) -> bool:
        """ Indicates whether the transformation annotates the edges it creates
            or modifies with the appropriate memlets. This determines
            whether to apply memlet propagation after the transformation.
        """
        return False

    def expressions(self) -> List[gr.SubgraphView]:
        """ Returns a list of Graph objects that will be matched in the
            subgraph isomorphism phase. Used as a pre-pass before calling
            `can_be_applied`.
            :see: Transformation.can_be_applied
        """
        raise NotImplementedError

    def can_be_applied(self,
                       graph: Union[SDFG, SDFGState],
                       candidate: Dict['PatternNode', int],
                       expr_index: int,
                       sdfg: SDFG,
                       permissive: bool = False) -> bool:
        """ Returns True if this transformation can be applied on the candidate
            matched subgraph.
            :param graph: SDFGState object if this Transformation is
                          single-state, or SDFG object otherwise.
            :param candidate: A mapping between node IDs returned from
                              `Transformation.expressions` and the nodes in
                              `graph`.
            :param expr_index: The list index from `Transformation.expressions`
                               that was matched.
            :param sdfg: If `graph` is an SDFGState, its parent SDFG. Otherwise
                         should be equal to `graph`.
            :param permissive: Whether transformation should run in permissive mode.
            :return: True if the transformation can be applied.
        """
        raise NotImplementedError

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        """
        Applies this transformation instance on the matched pattern graph.
        :param sdfg: The SDFG to apply the transformation to.
        :return: A transformation-defined return value, which could be used
                 to pass analysis data out, or nothing.
        """
        raise NotImplementedError

    def match_to_str(self, graph: Union[SDFG, SDFGState],
                     candidate: Dict['PatternNode', int]) -> str:
        """ Returns a string representation of the pattern match on the
            candidate subgraph. Used when identifying matches in the console
            UI.
        """
        return str(list(candidate.values()))

    def __init__(self,
                 sdfg_id: int,
                 state_id: int,
                 subgraph: Dict['PatternNode', int],
                 expr_index: int,
                 override: bool = False,
                 options: Optional[Dict[str, Any]] = None) -> None:
        """ Initializes an instance of Transformation match.
            :param sdfg_id: A unique ID of the SDFG.
            :param state_id: The node ID of the SDFG state, if applicable. If
                             transformation does not operate on a single state,
                             the value should be -1.
            :param subgraph: A mapping between node IDs returned from
                             `Transformation.expressions` and the nodes in
                             `graph`.
            :param expr_index: The list index from `Transformation.expressions`
                               that was matched.
            :param override: If True, accepts the subgraph dictionary as-is
                             (mostly for internal use).
            :param options: An optional dictionary of transformation properties
            :raise TypeError: When transformation is not subclass of
                              Transformation.
            :raise TypeError: When state_id is not instance of int.
            :raise TypeError: When subgraph is not a dict of
                              PatternNode : int.
        """

        self.sdfg_id = sdfg_id
        self.state_id = state_id
        if not override:
            expr = self.expressions()[expr_index]
            for value in subgraph.values():
                if not isinstance(value, int):
                    raise TypeError('All values of '
                                    'subgraph'
                                    ' dictionary must be '
                                    'instances of int.')
            self._subgraph = {expr.node_id(k): v for k, v in subgraph.items()}
        else:
            self._subgraph = {-1: -1}
        # Serializable subgraph with node IDs as keys
        self._subgraph_user = copy.copy(subgraph)
        self.expr_index = expr_index

        # Ease-of-use API: Set new pattern-nodes with information about this
        # instance.
        for pname, pval in self._get_pattern_nodes().items():
            # Create new pattern node from existing field
            new_pnode = PatternNode(
                pval.node if isinstance(pval, PatternNode) else type(pval))
            new_pnode.match_instance = self

            # Append existing values in subgraph dictionary
            if pval in self._subgraph_user:
                self._subgraph_user[new_pnode] = self._subgraph_user[pval]

            # Override static field with the new node in this instance only
            setattr(self, pname, new_pnode)

        # Set properties
        if options is not None:
            for optname, optval in options.items():
                setattr(self, optname, optval)

    @property
    def subgraph(self):
        return self._subgraph_user

    def apply_pattern(self,
                      sdfg: SDFG,
                      append: bool = True,
                      annotate: bool = True) -> Union[Any, None]:
        """
        Applies this transformation on the given SDFG, using the transformation
        instance to find the right SDFG object (based on SDFG ID), and applying
        memlet propagation as necessary.
        :param sdfg: The SDFG (or an SDFG in the same hierarchy) to apply the
                     transformation to.
        :param append: If True, appends the transformation to the SDFG
                       transformation history.
        :return: A transformation-defined return value, which could be used
                 to pass analysis data out, or nothing.
        """
        if append:
            sdfg.append_transformation(self)
        tsdfg: SDFG = sdfg.sdfg_list[self.sdfg_id]
        retval = self.apply(tsdfg)
        if annotate and not self.annotates_memlets():
            propagation.propagate_memlets_sdfg(tsdfg)
        return retval

    def __lt__(self, other: 'Transformation') -> bool:
        """
        Comparing two transformations by their class name and node IDs
        in match. Used for ordering transformations consistently.
        """
        if type(self) != type(other):
            return type(self).__name__ < type(other).__name__

        self_ids = iter(self.subgraph.values())
        other_ids = iter(self.subgraph.values())

        try:
            self_id = next(self_ids)
        except StopIteration:
            return True
        try:
            other_id = next(other_ids)
        except StopIteration:
            return False

        self_end = False

        while self_id is not None and other_id is not None:
            if self_id != other_id:
                return self_id < other_id
            try:
                self_id = next(self_ids)
            except StopIteration:
                self_end = True
            try:
                other_id = next(other_ids)
            except StopIteration:
                if self_end:  # Transformations are equal
                    return False
                return False
            if self_end:
                return True

    @classmethod
    def _get_pattern_nodes(cls) -> Dict[str, 'PatternNode']:
        """
        Returns a dictionary of pattern-matching node in this transformation
        subclass. Used internally for pattern-matching.
        :return: A dictionary mapping between pattern-node name and its type.
        """
        return {
            k: getattr(cls, k)
            for k in dir(cls)
            if isinstance(getattr(cls, k), PatternNode) or (k.startswith(
                '_') and isinstance(getattr(cls, k), (nd.Node, SDFGState)))
        }

    @classmethod
    def apply_to(cls,
                 sdfg: SDFG,
                 options: Optional[Dict[str, Any]] = None,
                 expr_index: int = 0,
                 verify: bool = True,
                 annotate: bool = True,
                 permissive: bool = False,
                 save: bool = True,
                 **where: Union[nd.Node, SDFGState]):
        """
        Applies this transformation to a given subgraph, defined by a set of
        nodes. Raises an error if arguments are invalid or transformation is
        not applicable.

        The subgraph is defined by the `where` dictionary, where each key is
        taken from the `PatternNode` fields of the transformation. For example,
        applying `MapCollapse` on two maps can pe performed as follows:

        ```
        MapCollapse.apply_to(sdfg, outer_map_entry=map_a, inner_map_entry=map_b)
        ```

        :param sdfg: The SDFG to apply the transformation to.
        :param options: A set of parameters to use for applying the
                        transformation.
        :param expr_index: The pattern expression index to try to match with.
        :param verify: Check that `can_be_applied` returns True before applying.
        :param annotate: Run memlet propagation after application if necessary.
        :param permissive: Apply transformation in permissive mode.
        :param save: Save transformation as part of the SDFG file. Set to
                     False if composing transformations.
        :param where: A dictionary of node names (from the transformation) to
                      nodes in the SDFG or a single state.
        """
        if len(where) == 0:
            raise ValueError('At least one node is required')
        options = options or {}

        # Check that all keyword arguments are nodes and if interstate or not
        sample_node = next(iter(where.values()))

        if isinstance(sample_node, SDFGState):
            graph = sdfg
            state_id = -1
        elif isinstance(sample_node, nd.Node):
            graph = next(s for s in sdfg.nodes() if sample_node in s.nodes())
            state_id = sdfg.node_id(graph)
        else:
            raise TypeError('Invalid node type "%s"' %
                            type(sample_node).__name__)

        # Check that all nodes in the pattern are set
        required_nodes = cls.expressions()[expr_index].nodes()
        required_node_names = {
            pname: pval
            for pname, pval in cls._get_pattern_nodes().items()
            if pval in required_nodes
        }
        required = set(required_node_names.keys())
        intersection = required & set(where.keys())
        if len(required - intersection) > 0:
            raise ValueError('Missing nodes for transformation subgraph: %s' %
                             (required - intersection))

        # Construct subgraph and instantiate transformation
        subgraph = {
            required_node_names[k]: graph.node_id(where[k])
            for k in required
        }
        instance = cls(sdfg.sdfg_id, state_id, subgraph, expr_index)

        # Construct transformation parameters
        for optname, optval in options.items():
            if not optname in cls.__properties__:
                raise ValueError('Property "%s" not found in transformation' %
                                 optname)
            setattr(instance, optname, optval)

        if verify:
            if not instance.can_be_applied(
                    graph, subgraph, expr_index, sdfg, permissive=permissive):
                raise ValueError('Transformation cannot be applied on the '
                                 'given subgraph ("can_be_applied" failed)')

        # Apply to SDFG
        return instance.apply_pattern(sdfg, annotate=annotate, append=save)

    def __str__(self) -> str:
        return type(self).__name__

    def print_match(self, sdfg: SDFG) -> str:
        """ Returns a string representation of the pattern match on the
            given SDFG. Used for printing matches in the console UI.
        """
        if not isinstance(sdfg, SDFG):
            raise TypeError("Expected SDFG, got: {}".format(
                type(sdfg).__name__))
        if self.state_id == -1:
            graph = sdfg
        else:
            graph = sdfg.nodes()[self.state_id]
        string = type(self).__name__ + ' in '
        string += self.match_to_str(graph, self.subgraph)
        return string

    def to_json(self, parent=None) -> Dict[str, Any]:
        props = serialize.all_properties_to_json(self)
        return {
            'type': 'Transformation',
            'transformation': type(self).__name__,
            **props
        }

    @staticmethod
    def from_json(json_obj: Dict[str, Any],
                  context: Dict[str, Any] = None) -> 'Transformation':
        xform = next(ext for ext in Transformation.extensions().keys()
                     if ext.__name__ == json_obj['transformation'])

        # Recreate subgraph
        expr = xform.expressions()[json_obj['expr_index']]
        subgraph = {
            expr.node(int(k)): int(v)
            for k, v in json_obj['_subgraph'].items()
        }

        # Reconstruct transformation
        ret = xform(json_obj['sdfg_id'], json_obj['state_id'], subgraph,
                    json_obj['expr_index'])
        context = context or {}
        context['transformation'] = ret
        serialize.set_properties_from_json(
            ret,
            json_obj,
            context=context,
            ignore_properties={'transformation', 'type'})
        return ret


class PatternNode(object):
    """
    Static field wrapper of a node or an SDFG state that designates it as part
    of a subgraph pattern. These objects are used in subclasses of
    `Transformation` to represent the subgraph patterns.

    Example use:
    ```
    @registry.autoregister_params(singlestate=True)
    class MyTransformation(Transformation):
        some_map_node = PatternNode(nodes.MapEntry)
        array = PatternNode(nodes.AccessNode)
    ```

    The two nodes can then be used in the transformation static methods (e.g.,
    `expressions`, `can_be_applied`) to represent the nodes, and in the instance
    methods to point to the nodes in the parent SDFG.
    """
    def __init__(self, nodeclass: Type[Union[nd.Node, SDFGState]]) -> None:
        """
        Initializes a pattern-matching node.
        :param nodeclass: The class of the node to match (can either be a state
                          or a node type in a state).
        """
        self.node = nodeclass
        self.match_instance: Optional[Transformation] = None

    def __call__(self, sdfg: SDFG) -> Union[nd.Node, SDFGState]:
        """
        Returns the matched node corresponding to this pattern node in the
        given SDFG. Requires the match (Transformation class) instance to
        be set.
        :param sdfg: The SDFG on which the transformation was applied.
        :return: The SDFG state or node that corresponds to this pattern node
                 in the given SDFG.
        :raise ValueError: If the transformation match instance is not set.
        """
        if self.match_instance is None:
            raise ValueError('Cannot query matched node. Transformation '
                             'instance not initialized')
        node_id: int = self.match_instance.subgraph[self]
        state_id: int = self.match_instance.state_id

        # Inter-state transformation
        if state_id == -1:
            return sdfg.node(node_id)

        # Single-state transformation
        return sdfg.node(state_id).node(node_id)


@make_properties
class ExpandTransformation(Transformation):
    """
    Base class for transformations that simply expand a node into a
    subgraph, and thus needs only simple matching and replacement
    functionality. Subclasses only need to implement the method
    "expansion".

    This is an internal interface used to track the expansion of library nodes.
    """
    @classmethod
    def expressions(clc):
        return [sdutil.node_path_graph(clc._match_node)]

    @staticmethod
    def can_be_applied(graph: gr.OrderedMultiDiConnectorGraph,
                       candidate: Dict[nd.Node, int],
                       expr_index: int,
                       sdfg,
                       permissive: bool = False):
        # All we need is the correct node
        return True

    @classmethod
    def match_to_str(clc, graph: gr.OrderedMultiDiConnectorGraph,
                     candidate: Dict[nd.Node, int]):
        node = graph.nodes()[candidate[clc._match_node]]
        return str(node)

    @staticmethod
    def expansion(node):
        raise NotImplementedError("Must be implemented by subclass")

    @staticmethod
    def postprocessing(sdfg, state, expansion):
        pass

    def apply(self, sdfg, *args, **kwargs):
        state = sdfg.nodes()[self.state_id]
        node = state.nodes()[self.subgraph[type(self)._match_node]]
        expansion = type(self).expansion(node, state, sdfg, *args, **kwargs)
        if isinstance(expansion, SDFG):
            expansion = state.add_nested_sdfg(expansion,
                                              sdfg,
                                              node.in_connectors,
                                              node.out_connectors,
                                              name=node.name,
                                              schedule=node.schedule,
                                              debuginfo=node.debuginfo)
        elif isinstance(expansion, nd.CodeNode):
            expansion.debuginfo = node.debuginfo
            if isinstance(expansion, nd.NestedSDFG):
                # Fix parent references
                nsdfg = expansion.sdfg
                nsdfg.parent = state
                nsdfg.parent_sdfg = sdfg
                nsdfg.update_sdfg_list([])
                nsdfg.parent_nsdfg_node = expansion

                # Update schedule to match library node schedule
                nsdfg.schedule = node.schedule

            elif isinstance(expansion, (nd.EntryNode, nd.LibraryNode)):
                if expansion.schedule is ScheduleType.Default:
                    expansion.schedule = node.schedule
        else:
            raise TypeError("Node expansion must be a CodeNode or an SDFG")

        # Fix nested schedules
        if isinstance(expansion, nd.NestedSDFG):
            infer_types._set_default_schedule_types(expansion.sdfg,
                                                    expansion.schedule, True)

        expansion.environments = copy.copy(
            set(map(lambda a: a.full_class_path(),
                    type(self).environments)))
        sdutil.change_edge_dest(state, node, expansion)
        sdutil.change_edge_src(state, node, expansion)
        state.remove_node(node)
        type(self).postprocessing(sdfg, state, expansion)

    def to_json(self, parent=None) -> Dict[str, Any]:
        props = serialize.all_properties_to_json(self)
        return {
            'type': 'ExpandTransformation',
            'transformation': type(self).__name__,
            'classpath': nd.full_class_path(self),
            **props
        }

    @staticmethod
    def from_json(json_obj: Dict[str, Any],
                  context: Dict[str, Any] = None) -> 'ExpandTransformation':
        xform = pydoc.locate(json_obj['classpath'])

        # Recreate subgraph
        expr = xform.expressions()[json_obj['expr_index']]
        subgraph = {
            expr.node(int(k)): int(v)
            for k, v in json_obj['_subgraph'].items()
        }

        # Reconstruct transformation
        ret = xform(json_obj['sdfg_id'], json_obj['state_id'], subgraph,
                    json_obj['expr_index'])
        context = context or {}
        context['transformation'] = ret
        serialize.set_properties_from_json(
            ret,
            json_obj,
            context=context,
            ignore_properties={'transformation', 'type', 'classpath'})
        return ret


@make_registry
@make_properties
class SubgraphTransformation(TransformationBase):
    """
    Base class for transformations that apply on arbitrary subgraphs, rather
    than matching a specific pattern.

    Subclasses need to implement the `can_be_applied` and `apply` operations,
    as well as registered with the subclass registry. See the `Transformation`
    class docstring for more information.
    """

    sdfg_id = Property(dtype=int, desc='ID of SDFG to transform')
    state_id = Property(
        dtype=int,
        desc='ID of state to transform subgraph within, or -1 to transform the '
        'SDFG')
    subgraph = SetProperty(element_type=int,
                           desc='Subgraph in transformation instance')

    def __init__(self,
                 subgraph: Union[Set[int], gr.SubgraphView],
                 sdfg_id: int = None,
                 state_id: int = None):
        if (not isinstance(subgraph, (gr.SubgraphView, SDFG, SDFGState))
                and (sdfg_id is None or state_id is None)):
            raise TypeError(
                'Subgraph transformation either expects a SubgraphView or a '
                'set of node IDs, SDFG ID and state ID (or -1).')

        # An entire graph is given as a subgraph
        if isinstance(subgraph, (SDFG, SDFGState)):
            subgraph = gr.SubgraphView(subgraph, subgraph.nodes())

        if isinstance(subgraph, gr.SubgraphView):
            self.subgraph = set(
                subgraph.graph.node_id(n) for n in subgraph.nodes())

            if isinstance(subgraph.graph, SDFGState):
                sdfg = subgraph.graph.parent
                self.sdfg_id = sdfg.sdfg_id
                self.state_id = sdfg.node_id(subgraph.graph)
            elif isinstance(subgraph.graph, SDFG):
                self.sdfg_id = subgraph.graph.sdfg_id
                self.state_id = -1
            else:
                raise TypeError('Unrecognized graph type "%s"' %
                                type(subgraph.graph).__name__)
        else:
            self.subgraph = subgraph
            self.sdfg_id = sdfg_id
            self.state_id = state_id

    def subgraph_view(self, sdfg: SDFG) -> gr.SubgraphView:
        graph = sdfg.sdfg_list[self.sdfg_id]
        if self.state_id != -1:
            graph = graph.node(self.state_id)
        return gr.SubgraphView(graph,
                               [graph.node(idx) for idx in self.subgraph])

    def can_be_applied(self, sdfg: SDFG, subgraph: gr.SubgraphView) -> bool:
        """
        Tries to match the transformation on a given subgraph, returning
        True if this transformation can be applied.
        :param sdfg: The SDFG that includes the subgraph.
        :param subgraph: The SDFG or state subgraph to try to apply the
                         transformation on.
        :return: True if the subgraph can be transformed, or False otherwise.
        """
        pass

    def apply(self, sdfg: SDFG):
        """
        Applies the transformation on the given subgraph.
        :param sdfg: The SDFG that includes the subgraph.
        """
        pass

    @classmethod
    def apply_to(cls,
                 sdfg: SDFG,
                 *where: Union[nd.Node, SDFGState, gr.SubgraphView],
                 verify: bool = True,
                 **options: Any):
        """
        Applies this transformation to a given subgraph, defined by a set of
        nodes. Raises an error if arguments are invalid or transformation is
        not applicable.

        To apply the transformation on a specific subgraph, the `where`
        parameter can be used either on a subgraph object (`SubgraphView`), or
        on directly on a list of subgraph nodes, given as `Node` or `SDFGState`
        objects. Transformation properties can then be given as keyword
        arguments. For example, applying `SubgraphFusion` on a subgraph of three
        nodes can be called in one of two ways:
        ```
        # Subgraph
        SubgraphFusion.apply_to(
            sdfg, SubgraphView(state, [node_a, node_b, node_c]))

        # Simplified API: list of nodes
        SubgraphFusion.apply_to(sdfg, node_a, node_b, node_c)
        ```

        :param sdfg: The SDFG to apply the transformation to.
        :param where: A set of nodes in the SDFG/state, or a subgraph thereof.
        :param verify: Check that `can_be_applied` returns True before applying.
        :param options: A set of parameters to use for applying the
                        transformation.
        """
        subgraph = None
        if len(where) == 1:
            if isinstance(where[0], (list, tuple)):
                where = where[0]
            elif isinstance(where[0], gr.SubgraphView):
                subgraph = where[0]
        if len(where) == 0:
            raise ValueError('At least one node is required')

        # Check that all keyword arguments are nodes and if interstate or not
        if subgraph is None:
            sample_node = where[0]

            if isinstance(sample_node, SDFGState):
                graph = sdfg
                state_id = -1
            elif isinstance(sample_node, nd.Node):
                graph = next(s for s in sdfg.nodes()
                             if sample_node in s.nodes())
                state_id = sdfg.node_id(graph)
            else:
                raise TypeError('Invalid node type "%s"' %
                                type(sample_node).__name__)

            # Construct subgraph and instantiate transformation
            subgraph = gr.SubgraphView(graph, where)
            instance = cls(subgraph, sdfg.sdfg_id, state_id)
        else:
            # Construct instance from subgraph directly
            instance = cls(subgraph)

        # Construct transformation parameters
        for optname, optval in options.items():
            if not optname in cls.__properties__:
                raise ValueError('Property "%s" not found in transformation' %
                                 optname)
            setattr(instance, optname, optval)

        if verify:
            if not instance.can_be_applied(sdfg, subgraph):
                raise ValueError('Transformation cannot be applied on the '
                                 'given subgraph ("can_be_applied" failed)')

        # Apply to SDFG
        return instance.apply(sdfg)

    def to_json(self, parent=None):
        props = serialize.all_properties_to_json(self)
        return {
            'type': 'SubgraphTransformation',
            'transformation': type(self).__name__,
            **props
        }

    @staticmethod
    def from_json(json_obj: Dict[str, Any],
                  context: Dict[str, Any] = None) -> 'SubgraphTransformation':
        xform = next(ext for ext in SubgraphTransformation.extensions().keys()
                     if ext.__name__ == json_obj['transformation'])

        # Reconstruct transformation
        ret = xform(json_obj['subgraph'], json_obj['sdfg_id'],
                    json_obj['state_id'])
        context = context or {}
        context['transformation'] = ret
        serialize.set_properties_from_json(
            ret,
            json_obj,
            context=context,
            ignore_properties={'transformation', 'type'})
        return ret


def coarsening_transformations() -> List[Type[Transformation]]:
    """ :return: List of all registered dataflow coarsening transformations.
    """
    return [
        k for k, v in Transformation.extensions().items()
        if v.get('coarsening', False)
    ]
