# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
This file contains classes that describe data-centric transformations.

All transformations extend the ``TransformationBase`` class. There are three built-in types of transformations in DaCe:

  * Pattern-matching Transformations (extending ``PatternTransformation``): Transformations
    that require a certain subgraph structure to match. Within this abstract class, there are two sub-classes:

    * ``SingleStateTransformation``: Patterns are limited to a single SDFG state.
    * ``MultiStateTransformation``: Patterns are given on a subgraph of an SDFG state machine.

    A pattern-matching must extend at least one of those two classes.

  * Subgraph Transformations (extending SubgraphTransformation): Transformations that can operate on arbitrary
    subgraphs.
  * Library node expansions (extending ExpandTransformation): An internal class used for tracking how library nodes
    are expanded.
"""

import abc
import copy
from dace import serialize
from dace.dtypes import ScheduleType
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion
from dace.sdfg import nodes as nd, graph as gr, utils as sdutil, propagation, infer_types, state as st
from dace.properties import make_properties, Property, DictProperty, SetProperty
from dace.transformation import pass_pipeline as ppl
from typing import Any, Dict, Generic, List, Optional, Set, Type, TypeVar, Union, Callable
import pydoc
import warnings
from typing import TypeVar

PassT = TypeVar('PassT', bound=ppl.Pass)


def explicit_cf_compatible(cls: PassT) -> PassT:
    cls.__explicit_cf_compatible__ = True
    return cls


class TransformationBase(ppl.Pass):
    """
    Base class for graph rewriting transformations. An instance of a TransformationBase object represents a match
    of the transformation (i.e., including a specific subgraph candidate to apply the transformation to), as well
    as properties of the transformation, which may affect if it can apply or not.

    A Transformation can also be seen as a Pass that, when applied, operates on the given subgraph.

    :see: PatternTransformation
    :see: SubgraphTransformation
    :see: ExpandTransformation
    """

    def modifies(self):
        # Unless otherwise mentioned, a transformation modifies everything
        return ppl.Modifies.Everything

    def should_reapply(self, _: ppl.Modifies) -> bool:
        return True


@make_properties
class PatternTransformation(TransformationBase):
    """
    Abstract class for pattern-matching transformations.
    Please extend either ``SingleStateTransformation`` or ``MultiStateTransformation``.

    :see: SingleStateTransformation
    :see: MultiStateTransformation
    :seealso: PatternNode
    """

    # Properties
    cfg_id = Property(dtype=int, category="(Debug)")
    state_id = Property(dtype=int, category="(Debug)")
    _subgraph = DictProperty(key_type=int, value_type=int, category="(Debug)")
    expr_index = Property(dtype=int, category="(Debug)")

    @classmethod
    def subclasses_recursive(cls, all_subclasses: bool = False) -> Set[Type['PatternTransformation']]:
        """
        Returns all subclasses of this class, including subclasses of subclasses.

        :param all_subclasses: Include all subclasses (e.g., including ``ExpandTransformation``).
        """
        if not all_subclasses and cls is PatternTransformation:
            subclasses = set(SingleStateTransformation.__subclasses__()) | set(
                MultiStateTransformation.__subclasses__())
        else:
            subclasses = set(cls.__subclasses__())
        subsubclasses = set()
        for sc in subclasses:
            subsubclasses.update(sc.subclasses_recursive())

        # Ignore abstract classes
        result = subclasses | subsubclasses
        result = set(sc for sc in result if not getattr(sc, '__abstractmethods__', False))

        return result

    def annotates_memlets(self) -> bool:
        """ Indicates whether the transformation annotates the edges it creates
            or modifies with the appropriate memlets. This determines
            whether to apply memlet propagation after the transformation.
        """
        return False

    @classmethod
    def expressions(cls) -> List[gr.SubgraphView]:
        """ Returns a list of Graph objects that will be matched in the
            subgraph isomorphism phase. Used as a pre-pass before calling
            `can_be_applied`.

            :see: PatternTransformation.can_be_applied
        """
        raise NotImplementedError

    def can_be_applied(self,
                       graph: Union[ControlFlowRegion, SDFGState],
                       expr_index: int,
                       sdfg: SDFG,
                       permissive: bool = False) -> bool:
        """ Returns True if this transformation can be applied on the candidate
            matched subgraph.

            :param graph: SDFGState object if this transformation is single-state, or ControlFlowRegion object
                          otherwise.
            :param expr_index: The list index from `PatternTransformation.expressions`
                               that was matched.
            :param sdfg: If `graph` is an SDFGState, its parent SDFG. Otherwise
                         should be equal to `graph`.
            :param permissive: Whether transformation should run in permissive mode.
            :return: True if the transformation can be applied.
        """
        raise NotImplementedError

    def apply(self, graph: Union[ControlFlowRegion, SDFGState], sdfg: SDFG) -> Union[Any, None]:
        """
        Applies this transformation instance on the matched pattern graph.

        :param sdfg: The SDFG to apply the transformation to.
        :return: A transformation-defined return value, which could be used
                 to pass analysis data out, or nothing.
        """
        raise NotImplementedError

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        # It is assumed that at the time of calling the transformation, all fields (e.g., subgraph) are already set
        self._sdfg = sdfg
        self._pipeline_results = pipeline_results
        return self.apply_pattern()

    def match_to_str(self, graph: Union[ControlFlowRegion, SDFGState]) -> str:
        """ Returns a string representation of the pattern match on the
            candidate subgraph. Used when identifying matches in the console
            UI.
        """
        candidate = []
        node_to_name = {v: k for k, v in self._get_pattern_nodes().items()}
        for cnode in self.subgraph.keys():
            cname = node_to_name[cnode]
            candidate.append(getattr(self, cname))
        return str(candidate)

    def setup_match(self,
                    sdfg: SDFG,
                    cfg_id: int,
                    state_id: int,
                    subgraph: Dict['PatternNode', int],
                    expr_index: int,
                    override: bool = False,
                    options: Optional[Dict[str, Any]] = None) -> None:
        """
        Sets the transformation to a given subgraph pattern.

        :param cfg_id: A unique ID of the SDFG.
        :param state_id: The node ID of the SDFG state, if applicable. If
                            transformation does not operate on a single state,
                            the value should be -1.
        :param subgraph: A mapping between node IDs returned from
                            `PatternTransformation.expressions` and the nodes in
                            `graph`.
        :param expr_index: The list index from `PatternTransformation.expressions`
                            that was matched.
        :param override: If True, accepts the subgraph dictionary as-is
                            (mostly for internal use).
        :param options: An optional dictionary of transformation properties
        :raise TypeError: When transformation is not subclass of
                            PatternTransformation.
        :raise TypeError: When state_id is not instance of int.
        :raise TypeError: When subgraph is not a dict of {PatternNode: int}.
        """

        self._sdfg = sdfg
        self.cfg_id = cfg_id
        self.state_id = state_id
        if not override:
            expr = self.expressions()[expr_index]
            for value in subgraph.values():
                if not isinstance(value, int):
                    raise TypeError('All values of subgraph dictionary must be instances of int.')
            self._subgraph = {expr.node_id(k): v for k, v in subgraph.items()}
        else:
            self._subgraph = {-1: -1}
        # Serializable subgraph with node IDs as keys
        self._subgraph_user = copy.copy(subgraph)
        self.expr_index = expr_index

        # Set properties
        if options is not None:
            for optname, optval in options.items():
                setattr(self, optname, optval)

        self._pipeline_results = None

    @property
    def subgraph(self):
        return self._subgraph_user

    def apply_pattern(self, append: bool = True, annotate: bool = True) -> Union[Any, None]:
        """
        Applies this transformation on the given SDFG, using the transformation instance to find the right control flow
        graph object (based on control flow graph ID), and applying memlet propagation as necessary.

        :param append: If True, appends the transformation to the SDFG transformation history.
        :param annotate: If True, applies memlet propagation as necessary.
        :return: A transformation-defined return value, which could be used to pass analysis data out, or nothing.
        """
        if append:
            self._sdfg.append_transformation(self)
        tcfg = self._sdfg.cfg_list[self.cfg_id]
        tsdfg = tcfg.sdfg if not isinstance(tcfg, SDFG) else tcfg
        tgraph = tcfg.node(self.state_id) if self.state_id >= 0 else tcfg
        retval = self.apply(tgraph, tsdfg)
        if annotate and not self.annotates_memlets():
            propagation.propagate_memlets_sdfg(tsdfg)
        return retval

    def __lt__(self, other: 'PatternTransformation') -> bool:
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
            for k in dir(cls) if isinstance(getattr(cls, k), PatternNode) or (
                k.startswith('_') and isinstance(getattr(cls, k), (nd.Node, SDFGState)))
        }

    @classmethod
    def _can_be_applied_and_apply(cls,
                                  verify: bool,
                                  apply: bool,
                                  sdfg: SDFG,
                                  options: Optional[Dict[str, Any]] = None,
                                  expr_index: int = 0,
                                  annotate: bool = True,
                                  permissive: bool = False,
                                  save: bool = True,
                                  **where: Union[nd.Node, SDFGState]):
        """
        Applies `can_be_applied()` and/or `apply()` to a given subgraph, defined by
        a set of nodes.

        The subgraph is defined by the `where` dictionary, where each key is
        taken from the `PatternNode` fields of the transformation. For example,
        applying `MapCollapse` on two maps can pe performed as follows:

        If `apply` is `True` then the function will apply the transformation, if `verify`
        is also `True` the function will first call `can_be_applied()` to ensure the
        transformation can be applied. If not, an error is generated.
        If `apply` is `False` the function will only call `can_be_applied()` and
        returns its result.

        :param sdfg: The SDFG to apply the transformation to.
        :param verify: Check that `can_be_applied` returns True before applying.
        :param apply: Also apply the transformation.
        :param options: A set of parameters to use for applying the
                        transformation.
        :param expr_index: The pattern expression index to try to match with.
        :param annotate: Run memlet propagation after application if necessary.
        :param permissive: Apply transformation in permissive mode.
        :param save: Save transformation as part of the SDFG file. Set to
                     False if composing transformations.
        :param where: A dictionary of node names (from the transformation) to
                      nodes in the SDFG or a single state.
        """

        if not (apply or verify):
            raise ValueError('Neither "apply" or "verify" must be specialized.')

        if len(where) == 0:
            raise ValueError('At least one node is required')
        options = options or {}

        # Check that all keyword arguments are nodes and if interstate or not
        sample_node = next(iter(where.values()))

        if isinstance(sample_node, ControlFlowBlock):
            graph = sample_node.parent_graph
            state_id = -1
            cfg_id = graph.cfg_id
        elif isinstance(sample_node, nd.Node):
            graph = next(s for s in sdfg.states() if sample_node in s.nodes())
            state_id = graph.block_id
            cfg_id = graph.parent_graph.cfg_id
        else:
            raise TypeError('Invalid node type "%s"' % type(sample_node).__name__)

        # Check that all nodes in the pattern are set
        required_nodes = cls.expressions()[expr_index].nodes()
        required_node_names = {
            pname: pval
            for pname, pval in cls._get_pattern_nodes().items() if pval in required_nodes
        }
        required = set(required_node_names.keys())
        intersection = required & set(where.keys())
        if len(required - intersection) > 0:
            raise ValueError('Missing nodes for transformation subgraph: %s' % (required - intersection))

        # Construct subgraph and instantiate transformation
        subgraph = {required_node_names[k]: graph.node_id(where[k]) for k in required}
        instance = cls()
        instance.setup_match(sdfg, cfg_id, state_id, subgraph, expr_index)

        # Construct transformation parameters
        for optname, optval in options.items():
            if not optname in cls.__properties__:
                raise ValueError('Property "%s" not found in transformation' % optname)
            setattr(instance, optname, optval)

        if verify:
            can_be_applied = instance.can_be_applied(graph, expr_index, sdfg, permissive=permissive)
            if apply and (not can_be_applied):
                raise ValueError('Transformation cannot be applied on the '
                                 'given subgraph ("can_be_applied" failed)')
            elif not apply:
                return can_be_applied

        # Apply to SDFG
        return instance.apply_pattern(annotate=annotate, append=save)

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
        return cls._can_be_applied_and_apply(
            verify=verify,
            apply=True,
            sdfg=sdfg,
            options=options,
            expr_index=expr_index,
            annotate=annotate,
            permissive=permissive,
            save=save,
            **where,
        )

    @classmethod
    def can_be_applied_to(cls,
                          sdfg: SDFG,
                          options: Optional[Dict[str, Any]] = None,
                          expr_index: int = 0,
                          permissive: bool = False,
                          **where: Union[nd.Node, SDFGState]) -> bool:
        """
        Checks if the given transformation can be applied to a subgraph, defined by
        a set of nodes.

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
        return cls._can_be_applied_and_apply(
            verify=True,
            apply=False,
            sdfg=sdfg,
            options=options,
            expr_index=expr_index,
            annotate=False,
            permissive=permissive,
            save=False,
            **where,
        )

    def __str__(self) -> str:
        return type(self).__name__

    def print_match(self, cfg: ControlFlowRegion) -> str:
        """ Returns a string representation of the pattern match on the
            given Control Flow Region. Used for printing matches in the console UI.
        """
        if not isinstance(cfg, ControlFlowRegion):
            raise TypeError("Expected ControlFlowRegion, got: {}".format(type(cfg).__name__))
        if self.state_id == -1:
            graph = cfg
        else:
            graph = cfg.nodes()[self.state_id]
        string = type(self).__name__ + ' in '
        string += self.match_to_str(graph)
        return string

    def to_json(self, parent=None) -> Dict[str, Any]:
        props = serialize.all_properties_to_json(self)
        return {'type': 'PatternTransformation', 'transformation': type(self).__name__, **props}

    @staticmethod
    def from_json(json_obj: Dict[str, Any], context: Dict[str, Any] = None) -> 'PatternTransformation':
        xform = next(ext for ext in PatternTransformation.subclasses_recursive(all_subclasses=True)
                     if ext.__name__ == json_obj['transformation'])

        # Recreate subgraph
        expr = xform.expressions()[json_obj.get('expr_index', 0)]
        subgraph = {expr.node(int(k)): int(v) for k, v in json_obj.get('_subgraph', {}).items()}

        # Reconstruct transformation
        ret = xform()
        ret.setup_match(None, json_obj.get('cfg_id', 0), json_obj.get('state_id', 0), subgraph,
                        json_obj.get('expr_index', 0))
        context = context or {}
        context['transformation'] = ret
        serialize.set_properties_from_json(ret, json_obj, context=context, ignore_properties={'transformation', 'type'})
        return ret


@make_properties
@explicit_cf_compatible
class SingleStateTransformation(PatternTransformation, abc.ABC):
    """
    Base class for pattern-matching transformations that find matches within a single SDFG state.
    New transformations that extend this class must contain static ``PatternNode`` fields that represent the
    nodes in the pattern graph, and use them to implement at least three methods:

        * ``expressions``: A method that returns a list of graph patterns (SDFGState objects) that match this transformation.
        * ``can_be_applied``: A method that, given a subgraph candidate, checks for additional conditions whether it can be transformed.
        * ``apply``: A method that applies the transformation on the given SDFG.

    For example:

    .. code-block:: python

        class MyTransformation(SingleStateTransformation):
            node_a = PatternNode(nodes.AccessNode)
            node_b = PatternNode(nodes.MapEntry)

            @classmethod
            def expressions(cls):
                return [node_path_graph(cls.node_a, cls.node_b)]


    For more information and optimization opportunities, see the respective
    methods' documentation.

    :see: PatternNode
    """

    @classmethod
    @abc.abstractmethod
    def expressions(cls) -> List[st.StateSubgraphView]:
        """ Returns a list of SDFG state subgraphs that will be matched in the
            subgraph isomorphism phase. Used as a pre-pass before calling
            ``can_be_applied``.
        """
        pass

    @abc.abstractmethod
    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        """ Returns True if this transformation can be applied on the candidate matched subgraph.

            :param graph: SDFGState object in which the match was found.
            :param candidate: A mapping between node IDs returned from
                              ``PatternTransformation.expressions`` and the nodes in
                              ``graph``.
            :param expr_index: The list index from ``PatternTransformation.expressions``
                               that was matched.
            :param sdfg: The parent SDFG of the matched state.
            :param permissive: Whether transformation should run in permissive mode.
            :return: True if the transformation can be applied.
        """
        pass


@make_properties
class MultiStateTransformation(PatternTransformation, abc.ABC):
    """
    Base class for pattern-matching transformations that find matches within an SDFG state machine.
    New transformations that extend this class must contain static ``PatternNode``-annotated fields that represent the
    nodes in the pattern graph, and use them to implement at least three methods:

        * ``expressions``: A method that returns a list of graph patterns (SDFG objects) that match this transformation.
        * ``can_be_applied``: A method that, given a subgraph candidate, checks for additional conditions whether it can be transformed.
        * ``apply``: A method that applies the transformation on the given SDFG.

    For example:

    .. code-block:: python

        class MyTransformation(MultiStateTransformation):
            state_a = PatternNode(SDFGState)
            state_b = PatternNode(SDFGState)

            @classmethod
            def expressions(cls):
                return [node_path_graph(cls.state_a, cls.state_b)]


    For more information and optimization opportunities, see the respective
    methods' documentation.

    :see: PatternNode
    """

    @classmethod
    @abc.abstractmethod
    def expressions(cls) -> List[gr.SubgraphView]:
        """ Returns a list of SDFG subgraphs that will be matched in the
            subgraph isomorphism phase. Used as a pre-pass before calling
            ``can_be_applied``.
        """
        pass

    @abc.abstractmethod
    def can_be_applied(self, graph: ControlFlowRegion, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        """ Returns True if this transformation can be applied on the candidate matched subgraph.

            :param graph: SDFG object in which the match was found.
            :param candidate: A mapping between node IDs returned from
                              ``PatternTransformation.expressions`` and the nodes in
                              ``graph``.
            :param expr_index: The list index from ``PatternTransformation.expressions``
                               that was matched.
            :param sdfg: The SDFG in which the match was found (equal to ``graph``).
            :param permissive: Whether transformation should run in permissive mode.
            :return: True if the transformation can be applied.
        """
        pass


T = TypeVar("T")


class PatternNode(Generic[T]):
    """
    Static field wrapper of a node or an SDFG state that designates it as part
    of a subgraph pattern. These objects are used in subclasses of
    ``PatternTransformation`` to represent the subgraph patterns.

    Example use:

    .. code-block:: python

        class MyTransformation(SingleStateTransformation):
            some_map_node = PatternNode(nodes.MapEntry)
            array = PatternNode(nodes.AccessNode)


    The two nodes can then be used in the transformation static methods (e.g.,
    ``expressions``, ``can_be_applied``) to represent the nodes, and in the instance
    methods to point to the nodes in the parent SDFG.
    """

    def __init__(self, nodeclass: Type[T]) -> None:
        """
        Initializes a pattern-matching node.

        :param nodeclass: The class of the node to match (can either be a state
                          or a node type in a state).
        """
        self.node = nodeclass

    def __get__(self, instance: Optional[PatternTransformation], owner) -> T:
        if instance is None:
            # Static methods (e.g., expressions()) get the pattern node itself
            return self

        # If an instance is used, we return the matched node
        node_id: int = instance.subgraph[self]
        state_id: int = instance.state_id
        t_graph: ControlFlowRegion = instance._sdfg.cfg_list[instance.cfg_id]

        if not isinstance(node_id, int):  # Node ID is already an object
            return node_id

        # Inter-state transformation
        if state_id == -1:
            return t_graph.node(node_id)

        # Single-state transformation
        state: SDFGState = t_graph.node(state_id)
        return state.node(node_id)


@make_properties
class ExpandTransformation(PatternTransformation):
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

    def can_be_applied(self, graph: gr.OrderedMultiDiConnectorGraph, expr_index: int, sdfg, permissive: bool = False):
        # All we need is the correct node
        return True

    def match_to_str(self, graph: gr.OrderedMultiDiConnectorGraph):
        return str(self._match_node)

    @staticmethod
    def expansion(node: nd.LibraryNode, parent_state: SDFGState, parent_sdfg: SDFG, *args, **kwargs):
        raise NotImplementedError("Must be implemented by subclass")

    @staticmethod
    def postprocessing(sdfg, state, expansion):
        pass

    def apply(self, state, sdfg, *args, **kwargs):
        node = state.node(self.subgraph[type(self)._match_node])
        expansion = type(self).expansion(node, state, sdfg, *args, **kwargs)
        if isinstance(expansion, SDFG):
            expansion = state.add_nested_sdfg(expansion,
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
                nsdfg.update_cfg_list([])
                nsdfg.parent_nsdfg_node = expansion

                # Update schedule to match library node schedule
                nsdfg.schedule = node.schedule

            elif isinstance(expansion, (nd.EntryNode, nd.LibraryNode)):
                if expansion.schedule is ScheduleType.Default:
                    expansion.schedule = node.schedule
        else:
            raise TypeError("Node expansion must be a CodeNode or an SDFG")

        expansion.environments = copy.copy(set(map(lambda a: a.full_class_path(), type(self).environments)))
        sdutil.change_edge_dest(state, node, expansion)
        sdutil.change_edge_src(state, node, expansion)
        state.remove_node(node)

        # Fix nested schedules
        if isinstance(expansion, nd.NestedSDFG):
            infer_types.set_default_schedule_and_storage_types(expansion.sdfg, [expansion.schedule], True)

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
    def from_json(json_obj: Dict[str, Any], context: Dict[str, Any] = None) -> 'ExpandTransformation':
        xform = pydoc.locate(json_obj['classpath'])

        # Recreate subgraph
        expr = xform.expressions()[json_obj.get('expr_index', 0)]
        subgraph = {expr.node(int(k)): int(v) for k, v in json_obj.get('_subgraph', {}).items()}

        # Reconstruct transformation
        ret = xform()
        ret.setup_match(None, json_obj.get('cfg_id', 0), json_obj.get('state_id', 0), subgraph,
                        json_obj.get('expr_index', 0))
        context = context or {}
        context['transformation'] = ret
        serialize.set_properties_from_json(ret,
                                           json_obj,
                                           context=context,
                                           ignore_properties={'transformation', 'type', 'classpath'})
        return ret


@make_properties
class SubgraphTransformation(TransformationBase):
    """
    Base class for transformations that apply on arbitrary subgraphs, rather
    than matching a specific pattern.

    Subclasses need to implement the ``can_be_applied`` and ``apply`` operations,
    as well as registered with the subclass registry. See the ``PatternTransformation``
    class docstring for more information.
    """

    cfg_id = Property(dtype=int, desc='ID of CFG to transform')
    state_id = Property(dtype=int, desc='ID of state to transform subgraph within, or -1 to transform the SDFG')
    subgraph = SetProperty(element_type=int, desc='Subgraph in transformation instance')

    def setup_match(self, subgraph: Union[Set[int], gr.SubgraphView], cfg_id: int = None, state_id: int = None):
        """
        Sets the transformation to a given subgraph.

        :param subgraph: A set of node (or state) IDs or a subgraph view object.
        :param cfg_id: A unique ID of the CFG.
        :param state_id: The node ID of the SDFG state, if applicable. If transformation does not operate on a single
                         state, the value should be -1.
        """
        if (not isinstance(subgraph, (gr.SubgraphView, SDFG, SDFGState)) and (cfg_id is None or state_id is None)):
            raise TypeError('Subgraph transformation either expects a SubgraphView or a '
                            'set of node IDs, control flow graph ID and state ID (or -1).')

        self._pipeline_results = None

        # An entire graph is given as a subgraph
        if isinstance(subgraph, (SDFG, SDFGState)):
            subgraph = gr.SubgraphView(subgraph, subgraph.nodes())

        if isinstance(subgraph, gr.SubgraphView):
            self.subgraph = set(subgraph.graph.node_id(n) for n in subgraph.nodes())

            if isinstance(subgraph.graph, SDFGState):
                self.cfg_id = subgraph.graph.parent_graph.cfg_id
                self.state_id = subgraph.graph.block_id
            elif isinstance(subgraph.graph, SDFG):
                self.cfg_id = subgraph.graph.cfg_id
                self.state_id = -1
            else:
                raise TypeError('Unrecognized graph type "%s"' % type(subgraph.graph).__name__)
        else:
            self.subgraph = subgraph
            self.cfg_id = cfg_id
            self.state_id = state_id

    @classmethod
    def subclasses_recursive(cls) -> Set[Type['PatternTransformation']]:
        """
        Returns all subclasses of this class, including subclasses of subclasses.

        :param all_subclasses: Include all subclasses (e.g., including ``ExpandTransformation``).
        """
        subclasses = set(cls.__subclasses__())
        subsubclasses = set()
        for sc in subclasses:
            subsubclasses.update(sc.subclasses_recursive())

        # Ignore abstract classes
        result = subclasses | subsubclasses
        result = set(sc for sc in result if not getattr(sc, '__abstractmethods__', False))

        return result

    def subgraph_view(self, sdfg: SDFG) -> gr.SubgraphView:
        graph = sdfg.cfg_list[self.cfg_id]
        if self.state_id != -1:
            graph = graph.node(self.state_id)
        return gr.SubgraphView(graph, [graph.node(idx) for idx in self.subgraph])

    def can_be_applied(self, sdfg: SDFG, subgraph: gr.SubgraphView) -> bool:
        """
        Tries to match the transformation on a given subgraph, returning
        ``True`` if this transformation can be applied.

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

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        self._pipeline_results = pipeline_results
        return self.apply(sdfg)

    @classmethod
    def _can_be_applied_and_apply(cls, *where: Union[nd.Node, SDFGState, gr.SubgraphView], verify: bool, apply: bool,
                                  sdfg: SDFG, **options: Any):
        """

        Applies `can_be_applied()` and/or `apply()` to a given subgraph, defined by
        a set of nodes.

        The subgraph is defined by the `where` dictionary, where each key is
        taken from the `PatternNode` fields of the transformation. For example,
        applying `MapCollapse` on two maps can pe performed as follows:

        The subgraph is defined by the ``where`` parameter can be used either
        on a subgraph object (``SubgraphView``), or on directly on a list of
        subgraph nodes, given as ``Node`` or ``SDFGState`` objects. Transformation
        properties can then be given as keyword arguments.

        If `apply` is `True` then the function will apply the transformation, if `verify`
        is also `True` the function will first call `can_be_applied()` to ensure the
        transformation can be applied. If not an error is genrated.
        If `apply` is `False` the function will only call `can_be_applied()` and
        returns its result.

        :param sdfg: The SDFG to apply the transformation to.
        :param verify: Check that ``can_be_applied`` returns True before applying.
        :param apply: Apply the transformation to the subgraph.
        :param where: A set of nodes in the SDFG/state, or a subgraph thereof.
        :param options: A set of parameters to use for applying the transformation.
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
                graph = next(s for s in sdfg.nodes() if sample_node in s.nodes())
                state_id = sdfg.node_id(graph)
            else:
                raise TypeError('Invalid node type "%s"' % type(sample_node).__name__)

            # Construct subgraph and instantiate transformation
            subgraph = gr.SubgraphView(graph, where)
            instance = cls()
            instance.setup_match(subgraph, sdfg.cfg_id, state_id)
        else:
            # Construct instance from subgraph directly
            instance = cls()
            instance.setup_match(subgraph)

        # Construct transformation parameters
        for optname, optval in options.items():
            if not optname in cls.__properties__:
                raise ValueError('Property "%s" not found in transformation' % optname)
            setattr(instance, optname, optval)

        if verify:
            can_be_applied = instance.can_be_applied(sdfg, subgraph)
            if apply and (not can_be_applied):
                raise ValueError('Transformation cannot be applied on the '
                                 'given subgraph ("can_be_applied" failed)')
            elif not apply:
                return can_be_applied

        # Apply to SDFG
        return instance.apply(sdfg)

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

        To apply the transformation on a specific subgraph, the ``where``
        parameter can be used either on a subgraph object (``SubgraphView``), or
        on directly on a list of subgraph nodes, given as ``Node`` or ``SDFGState``
        objects. Transformation properties can then be given as keyword
        arguments. For example, applying ``SubgraphFusion`` on a subgraph of three
        nodes can be called in one of two ways:

        .. code-block:: python

            # Subgraph
            SubgraphFusion.apply_to(
                sdfg, SubgraphView(state, [node_a, node_b, node_c]))

            # Simplified API: list of nodes
            SubgraphFusion.apply_to(sdfg, node_a, node_b, node_c)


        :param sdfg: The SDFG to apply the transformation to.
        :param where: A set of nodes in the SDFG/state, or a subgraph thereof.
        :param verify: Check that ``can_be_applied`` returns True before applying.
        :param options: A set of parameters to use for applying the transformation.
        """
        return cls._can_be_applied_and_apply(
            verify=verify,
            apply=True,
            sdfg=sdfg,
            *where,
            **options,
        )

    @classmethod
    def can_be_applied_to(cls, sdfg: SDFG, *where: Union[nd.Node, SDFGState, gr.SubgraphView], **options: Any) -> bool:
        """
        Checks if the transformation can be applied to a given subgraph, defined
        by a set of nodes.

        To apply the transformation on a specific subgraph, the ``where``
        parameter can be used either on a subgraph object (``SubgraphView``), or
        on directly on a list of subgraph nodes, given as ``Node`` or ``SDFGState``
        objects. Transformation properties can then be given as keyword arguments.
        For example, to check if ``SubgraphFusion`` can be applied on a subgraph
        of three nodes can be done either:

        .. code-block:: python

            # Subgraph
            SubgraphFusion.can_be_applied_to(
                sdfg, SubgraphView(state, [node_a, node_b, node_c]))

            # Simplified API: list of nodes
            SubgraphFusion.can_be_applied_to(sdfg, node_a, node_b, node_c)


        :param sdfg: The SDFG to apply the transformation to.
        :param where: A set of nodes in the SDFG/state, or a subgraph thereof.
        :param options: A set of parameters to use for applying the transformation.
        """
        return cls._can_be_applied_and_apply(
            verify=True,
            apply=False,
            sdfg=sdfg,
            *where,
            **options,
        )

    def to_json(self, parent=None):
        props = serialize.all_properties_to_json(self)
        return {'type': 'SubgraphTransformation', 'transformation': type(self).__name__, **props}

    @staticmethod
    def from_json(json_obj: Dict[str, Any], context: Dict[str, Any] = None) -> 'SubgraphTransformation':
        xform = next(ext for ext in SubgraphTransformation.subclasses_recursive()
                     if ext.__name__ == json_obj['transformation'])

        # Reconstruct transformation
        ret = xform()
        ret.setup_match(json_obj.get('subgraph', {}), json_obj.get('cfg_id', 0), json_obj.get('state_id', 0))
        context = context or {}
        context['transformation'] = ret
        serialize.set_properties_from_json(ret, json_obj, context=context, ignore_properties={'transformation', 'type'})
        return ret


def _make_function_blocksafe(cls: ppl.Pass, function_name: str, get_sdfg_arg: Callable[[Any], Optional[SDFG]]):
    if hasattr(cls, function_name):
        vanilla_method = getattr(cls, function_name)

        def blocksafe_wrapper(tgt, *args, **kwargs):
            if isinstance(tgt, SDFG):
                sdfg = tgt
            elif kwargs and 'sdfg' in kwargs:
                sdfg = kwargs['sdfg']
            else:
                sdfg = get_sdfg_arg(tgt, *args)
            if sdfg and isinstance(sdfg, SDFG):
                root_sdfg: SDFG = sdfg.cfg_list[0]
                if not root_sdfg.using_explicit_control_flow:
                    return vanilla_method(tgt, *args, **kwargs)
                else:
                    warnings.warn('Skipping ' + function_name + ' from ' + cls.__name__ +
                                  ' due to incompatibility with experimental control flow blocks')

        setattr(cls, function_name, blocksafe_wrapper)


def _subgraph_transformation_extract_sdfg_arg(*args) -> SDFG:
    subgraph = args[1]
    if isinstance(subgraph, SDFG):
        return subgraph
    elif isinstance(subgraph, SDFGState):
        return subgraph.sdfg
    elif isinstance(subgraph, gr.SubgraphView):
        if isinstance(subgraph.graph, SDFGState):
            return subgraph.graph.sdfg
        elif isinstance(subgraph.graph, SDFG):
            return subgraph.graph
        raise TypeError('Unrecognized graph type "%s"' % type(subgraph.graph).__name__)
    raise TypeError('Unrecognized graph type "%s"' % type(subgraph).__name__)


def single_level_sdfg_only(cls: PassT) -> PassT:

    for function_name in ['apply_pass', 'apply_to']:
        _make_function_blocksafe(cls, function_name, lambda *args: args[1])

    if issubclass(cls, SubgraphTransformation):
        _make_function_blocksafe(cls, 'apply', lambda *args: args[1])
        _make_function_blocksafe(cls, 'can_be_applied', lambda *args: args[1])
        _make_function_blocksafe(cls, 'setup_match', _subgraph_transformation_extract_sdfg_arg)
    elif issubclass(cls, ppl.StatePass):
        _make_function_blocksafe(cls, 'apply', lambda *args: args[1].sdfg)
    elif issubclass(cls, ppl.ScopePass):
        _make_function_blocksafe(cls, 'apply', lambda *args: args[2].sdfg)
    else:
        _make_function_blocksafe(cls, 'apply', lambda *args: args[2])
        _make_function_blocksafe(cls, 'can_be_applied', lambda *args: args[3])
        _make_function_blocksafe(cls, 'setup_match', lambda *args: args[1])

    if issubclass(cls, PatternTransformation):
        _make_function_blocksafe(cls, 'apply_pattern', lambda *args: args[0]._sdfg)

    return cls
