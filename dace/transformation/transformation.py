# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that represent data-centric transformations. """

import copy
from dace import serialize
from dace.dtypes import ScheduleType
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes as nd, graph as gr, utils as sdutil, propagation
from dace.sdfg.graph import SubgraphView
from dace.properties import make_properties, Property, DictProperty, SetProperty
from dace.registry import make_registry
from typing import Dict, List, Set, Union


@make_registry
@make_properties
class Transformation(object):
    """ Base class for transformations, as well as a static registry of
        transformations, where new transformations can be added in a
        decentralized manner.

        New transformations are registered with ``Transformation.register``
        (or ``dace.registry.autoregister_params``) with two optional boolean
        keyword arguments: ``singlestate`` (default: False) and ``strict``
        (default: False).
        If ``singlestate`` is True, the transformation is matched on subgraphs
        inside an SDFGState; otherwise, subgraphs of the SDFG state machine are
        matched.
        If ``strict`` is True, this transformation will be considered strict
        (i.e., always beneficial to perform) and will be performed automatically
        as part of SDFG strict transformations.
    """

    # Properties
    sdfg_id = Property(dtype=int, category="(Debug)")
    state_id = Property(dtype=int, category="(Debug)")
    _subgraph = DictProperty(key_type=int, value_type=int, category="(Debug)")
    expr_index = Property(dtype=int, category="(Debug)")

    @staticmethod
    def annotates_memlets():
        """ Indicates whether the transformation annotates the edges it creates
            or modifies with the appropriate memlets. This determines
            whether to apply memlet propagation after the transformation.
        """

        return False

    @staticmethod
    def expressions():
        """ Returns a list of Graph objects that will be matched in the
            subgraph isomorphism phase. Used as a pre-pass before calling
            `can_be_applied`.
            @see Transformation.can_be_applied
        """

        raise NotImplementedError

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
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
            :param strict: Whether transformation should run in strict mode.
            :return: True if the transformation can be applied.
        """
        raise NotImplementedError

    @staticmethod
    def match_to_str(graph, candidate):
        """ Returns a string representation of the pattern match on the
            candidate subgraph. Used when identifying matches in the console
            UI.
        """
        raise NotImplementedError

    def __init__(self, sdfg_id, state_id, subgraph, expr_index):
        """ Initializes an instance of Transformation.
            :param sdfg_id: A unique ID of the SDFG.
            :param state_id: The node ID of the SDFG state, if applicable.
            :param subgraph: A mapping between node IDs returned from
                             `Transformation.expressions` and the nodes in
                             `graph`.
            :param expr_index: The list index from `Transformation.expressions`
                               that was matched.
            :raise TypeError: When transformation is not subclass of
                              Transformation.
            :raise TypeError: When state_id is not instance of int.
            :raise TypeError: When subgraph is not a dict of
                              dace.sdfg.nodes.Node : int.
        """

        self.sdfg_id = sdfg_id
        self.state_id = state_id
        for value in subgraph.values():
            if not isinstance(value, int):
                raise TypeError('All values of '
                                'subgraph'
                                ' dictionary must be '
                                'instances of int.')
        # Serializable subgraph with node IDs as keys
        expr = self.expressions()[expr_index]
        self._subgraph = {expr.node_id(k): v for k, v in subgraph.items()}
        self._subgraph_user = subgraph
        self.expr_index = expr_index

    @property
    def subgraph(self):
        return self._subgraph_user

    def __lt__(self, other):
        """ Comparing two transformations by their class name and node IDs
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

    def apply_pattern(self, sdfg):
        """ Applies this transformation on the given SDFG. """
        sdfg.append_transformation(self)
        self.apply(sdfg)
        if not self.annotates_memlets():
            propagation.propagate_memlets_sdfg(sdfg)

    def __str__(self):
        return type(self).__name__

    def modifies_graph(self):
        return True

    def print_match(self, sdfg):
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
        string += type(self).match_to_str(graph, self.subgraph)
        return string

    def to_json(self, parent=None):
        props = serialize.all_properties_to_json(self)
        return {
            'type': 'Transformation',
            'transformation': type(self).__name__,
            **props
        }

    @staticmethod
    def from_json(json_obj, context=None):
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


class ExpandTransformation(Transformation):
    """Base class for transformations that simply expand a node into a
       subgraph, and thus needs only simple matching and replacement
       functionality. Subclasses only need to implement the method
       "expansion".
    """
    @classmethod
    def expressions(clc):
        return [sdutil.node_path_graph(clc._match_node)]

    @staticmethod
    def can_be_applied(graph: gr.OrderedMultiDiConnectorGraph,
                       candidate: Dict[nd.Node, int],
                       expr_index: int,
                       sdfg,
                       strict: bool = False):
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
            # Modify internal schedules according to node schedule
            if node.schedule != ScheduleType.Default:
                for nstate in expansion.nodes():
                    topnodes = nstate.scope_dict(node_to_children=True)[None]
                    for topnode in topnodes:
                        if isinstance(topnode, (nd.EntryNode, nd.LibraryNode)):
                            topnode.schedule = node.schedule

            expansion = state.add_nested_sdfg(expansion,
                                              sdfg,
                                              node.in_connectors,
                                              node.out_connectors,
                                              name=node.name,
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
        else:
            raise TypeError("Node expansion must be a CodeNode or an SDFG")
        expansion.environments = copy.copy(
            set(map(lambda a: a.__name__,
                    type(self).environments)))
        sdutil.change_edge_dest(state, node, expansion)
        sdutil.change_edge_src(state, node, expansion)
        state.remove_node(node)
        type(self).postprocessing(sdfg, state, expansion)


@make_registry
@make_properties
class SubgraphTransformation(object):
    """
    Base class for transformations that apply on arbitrary subgraphs, rather than
    matching a specific pattern. Subclasses need to implement the `match` and `apply`
    operations.
    """

    sdfg_id = Property(dtype=int, desc='ID of SDFG to transform')
    state_id = Property(
        dtype=int,
        desc='ID of state to transform subgraph within, or -1 to transform the '
        'SDFG')
    subgraph = SetProperty(element_type=int,
                           desc='Subgraph in transformation instance')

    def __init__(self,
                 subgraph: Union[Set[int], SubgraphView],
                 sdfg_id: int = None,
                 state_id: int = None):
        if (not isinstance(subgraph, (SubgraphView, SDFG, SDFGState))
                and (sdfg_id is None or state_id is None)):
            raise TypeError(
                'Subgraph transformation either expects a SubgraphView or a '
                'set of node IDs, SDFG ID and state ID (or -1).')

        # An entire graph is given as a subgraph
        if isinstance(subgraph, (SDFG, SDFGState)):
            subgraph = SubgraphView(subgraph, subgraph.nodes())

        if isinstance(subgraph, SubgraphView):
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

    def subgraph_view(self, sdfg: SDFG) -> SubgraphView:
        graph = sdfg.sdfg_list[self.sdfg_id]
        if self.state_id != -1:
            graph = graph.node(self.state_id)
        return SubgraphView(graph, [graph.node(idx) for idx in self.subgraph])

    @staticmethod
    def match(sdfg: SDFG, subgraph: SubgraphView) -> bool:
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

    def to_json(self, parent=None):
        props = serialize.all_properties_to_json(self)
        return {
            'type': 'SubgraphTransformation',
            'transformation': type(self).__name__,
            **props
        }

    @staticmethod
    def from_json(json_obj, context=None):
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
