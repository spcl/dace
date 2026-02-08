# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from collections import defaultdict
from typing import Any, Dict, Optional, Set, Tuple, Union

from dace import SDFG, Memlet, SDFGState
from dace.frontend.python import astutils
from dace.properties import CodeBlock
from dace.sdfg import nodes as nd
from dace.sdfg.graph import Edge, MultiConnectorEdge
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace import data as dt
from dace import dtypes

import sys
from typing import Literal

dirtype = Literal['in', 'out']


class RecodeAttributeNodes(ast.NodeTransformer):

    connector: str
    data: Union[dt.Structure, dt.ContainerView]
    tasklet: nd.Tasklet
    direction: dirtype
    memlet: Memlet
    data_node: nd.AccessNode
    state: SDFGState
    views_constructed: Set[str]

    def __init__(self, state: SDFGState, data_node: nd.AccessNode, connector: str,
                 data: Union[dt.Structure, dt.ContainerView], tasklet: nd.Tasklet, memlet: Memlet, direction: dirtype):
        self.connector = connector
        self.data = data
        self.tasklet = tasklet
        self.direction = direction
        self.memlet = memlet
        self.data_node = data_node
        self.state = state
        self.views_constructed = set()

    def _handle_simple_name_access(self, node: ast.Attribute, val: ast.Name) -> Any:
        struct: dt.Structure = self.data
        if not node.attr in struct.members:
            raise RuntimeError(
                f'Structure attribute {node.attr} is not a member of the structure {struct.name} type definition')

        # Gather a new connector name and add the appropriate connector.
        new_connector_name = val.id + '_' + node.attr
        new_connector_name = self.tasklet.next_connector(new_connector_name)
        if self.direction == 'in':
            if not self.tasklet.add_in_connector(new_connector_name):
                raise RuntimeError(f'Failed to add connector {new_connector_name}')
        else:
            if not self.tasklet.add_out_connector(new_connector_name):
                raise RuntimeError(f'Failed to add connector {new_connector_name}')

        # Construct the correct AST replacement node (direct access, i.e., name node).
        replacement = ast.Name(id=new_connector_name, ctx=ast.Load())

        # Insert the appropriate view, if it does not exist yet.
        view_name = 'v_' + self.data_node.data + '_' + node.attr
        try:
            view = self.state.sdfg.arrays[view_name]
        except KeyError:
            view = dt.View.view(struct.members[node.attr])
            view_name = self.state.sdfg.add_datadesc(view_name, view, find_new_name=True)
        self.views_constructed.add(view_name)

        # Add an access node for the view and connect it appropriately.
        view_node = self.state.add_access(view_name)
        if self.direction == 'in':
            self.state.add_edge(self.data_node, None, view_node, 'views',
                                Memlet.from_array(self.data_node.data + '.' + node.attr, self.data.members[node.attr]))
            self.state.add_edge(view_node, None, self.tasklet, new_connector_name, Memlet.from_array(view_name, view))
        else:
            self.state.add_edge(view_node, 'views', self.data_node, None,
                                Memlet.from_array(self.data_node.data + '.' + node.attr, self.data.members[node.attr]))
            self.state.add_edge(self.tasklet, new_connector_name, view_node, None, Memlet.from_array(view_name, view))
        return self.generic_visit(replacement)

    def _handle_sliced_access(self, node: ast.Attribute, val: ast.Slice) -> Any:
        struct = self.data.stype
        if not isinstance(struct, dt.Structure):
            raise ValueError('Invalid ContainerView, can only lift ContainerViews to Structures')
        if not node.attr in struct.members:
            raise RuntimeError(
                f'Structure attribute {node.attr} is not a member of the structure {struct.name} type definition')

        # Gather a new connector name and add the appropriate connector.
        new_connector_name = node.value.value.id + '_slice_' + node.attr
        new_connector_name = self.tasklet.next_connector(new_connector_name)
        if self.direction == 'in':
            if not self.tasklet.add_in_connector(new_connector_name):
                raise RuntimeError(f'Failed to add connector {new_connector_name}')
        else:
            if not self.tasklet.add_out_connector(new_connector_name):
                raise RuntimeError(f'Failed to add connector {new_connector_name}')

        # We first lift the slice into a separate view, and then the attribute access.
        slice_view_name = 'v_' + self.data_node.data + '_slice'
        attr_view_name = slice_view_name + '_' + node.attr
        try:
            slice_view = self.state.sdfg.arrays[slice_view_name]
        except KeyError:
            slice_view = dt.View.view(struct)
            slice_view_name = self.state.sdfg.add_datadesc(slice_view_name, slice_view, find_new_name=True)
        try:
            attr_view = self.state.sdfg.arrays[attr_view_name]
        except KeyError:
            member: dt.Data = struct.members[node.attr]
            attr_view = dt.View.view(member)
            attr_view_name = self.state.sdfg.add_datadesc(attr_view_name, attr_view, find_new_name=True)
        self.views_constructed.add(slice_view_name)
        self.views_constructed.add(attr_view_name)

        # Construct the correct AST replacement node (direct access, i.e., name node).
        replacement = ast.Name(id=new_connector_name, ctx=ast.Load())

        # Add access nodes for the views and connect them appropriately.
        slice_view_node = self.state.add_access(slice_view_name)
        attr_view_node = self.state.add_access(attr_view_name)
        if self.direction == 'in':
            idx = astutils.unparse(val.slice)
            if isinstance(val.slice, ast.Tuple):
                idx = idx.strip('()')
            slice_memlet = Memlet(self.data_node.data + '[' + idx + ']')
            self.state.add_edge(self.data_node, None, slice_view_node, 'views', slice_memlet)
            attr_memlet = Memlet.from_array(slice_view_name + '.' + node.attr, struct.members[node.attr])
            self.state.add_edge(slice_view_node, None, attr_view_node, 'views', attr_memlet)
            # TODO: determine the actual subset from the tasklet accesses.
            self.state.add_edge(attr_view_node, None, self.tasklet, new_connector_name,
                                Memlet.from_array(attr_view_name, attr_view))
        else:
            self.state.add_edge(self.tasklet, new_connector_name, attr_view_node, None,
                                Memlet.from_array(attr_view_name, attr_view))
            # TODO: determine the actual subset from the tasklet accesses.
            attr_memlet = Memlet.from_array(slice_view_name + '.' + node.attr, struct.members[node.attr])
            self.state.add_edge(attr_view_node, 'views', slice_view_node, None, attr_memlet)
            idx = astutils.unparse(val.slice)
            slice_memlet = Memlet(self.data_node.data + '[' + idx + ']')
            self.state.add_edge(slice_view_node, 'views', self.data_node, None, slice_memlet)
        return self.generic_visit(replacement)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if not node.value:
            return self.generic_visit(node)

        if isinstance(self.data, (dt.Structure, dt.StructureView, dt.StructureReference)):
            if not isinstance(node.value, ast.Name) or node.value.id != self.connector:
                return self.generic_visit(node)
            return self._handle_simple_name_access(node, node.value)
        elif isinstance(self.data, (dt.ContainerView, dt.ContainerArray, dt.ContainerArrayReference)):
            if isinstance(node.value, ast.Name) and node.value.id == self.connector:
                # We are directly accessing a slice of a container array / view. That needs an inserted view to the
                # container first.
                slice_view_name = 'v_' + self.data_node.data + '_slice'
                try:
                    slice_view = self.state.sdfg.arrays[slice_view_name]
                except KeyError:
                    slice_view = dt.View.view(self.data.stype)
                    slice_view_name = self.state.sdfg.add_datadesc(slice_view_name, slice_view, find_new_name=True)
                self.views_constructed.add(slice_view_name)
                slice_view_node = self.state.add_access(slice_view_name)
                if self.direction == 'in':
                    self.state.add_edge(self.data_node, None, slice_view_node, 'views', self.memlet)
                    self.state.add_edge(slice_view_node, None, self.tasklet, self.connector,
                                        Memlet.from_array(slice_view_name, slice_view))
                else:
                    self.state.add_edge(slice_view_node, 'views', self.data_node, None, self.memlet)
                    self.state.add_edge(self.tasklet, self.connector, slice_view_node, None,
                                        Memlet.from_array(slice_view_name, slice_view))
            elif (isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name)
                  and node.value.value.id == self.connector):
                return self._handle_sliced_access(node, node.value)
            return self.generic_visit(node)
        else:
            raise NotImplementedError()


class InterstateEdgeRecoder(ast.NodeTransformer):

    sdfg: SDFG
    element: Union[Edge[InterstateEdge], Tuple[ControlFlowBlock, CodeBlock]]
    data_name: str
    data: Union[dt.Structure, dt.ContainerArray]
    views_constructed: Set[str]
    _lifting_state: SDFGState

    def __init__(self,
                 sdfg: SDFG,
                 element: Union[Edge[InterstateEdge], Tuple[ControlFlowBlock, CodeBlock]],
                 data_name: str,
                 data: Union[dt.Structure, dt.ContainerArray],
                 lifting_state: Optional[SDFGState] = None):
        self.sdfg = sdfg
        self.element = element
        self.data_name = data_name
        self.data = data
        self.views_constructed = set()
        self._lifting_state = lifting_state

    def _handle_simple_name_access(self, node: ast.Attribute) -> Any:
        struct: dt.Structure = self.data
        if not node.attr in struct.members:
            raise RuntimeError(
                f'Structure attribute {node.attr} is not a member of the structure {struct.name} type definition')

        # Insert the appropriate view, if it does not exist yet.
        view_name = 'v_' + self.data_name + '_' + node.attr
        try:
            view = self.sdfg.arrays[view_name]
        except KeyError:
            view = dt.View.view(struct.members[node.attr])
            view_name = self.sdfg.add_datadesc(view_name, view, find_new_name=True)
        self.views_constructed.add(view_name)

        # Construct the correct AST replacement node (direct access, i.e., name node).
        replacement = ast.Name(id=view_name, ctx=ast.Load())

        # Add access nodes for the view and the original container and connect them appropriately.
        lift_state, data_node = self._get_or_create_lifting_state()
        view_node = lift_state.add_access(view_name)
        lift_state.add_edge(data_node, None, view_node, 'views',
                            Memlet.from_array(data_node.data + '.' + node.attr, self.data.members[node.attr]))
        return self.generic_visit(replacement)

    def _handle_sliced_access(self, node: ast.Attribute, val: ast.Subscript) -> Any:
        struct = self.data.stype
        if not isinstance(struct, dt.Structure):
            raise ValueError('Invalid ContainerArray, can only lift ContainerArrays to Structures')
        if not node.attr in struct.members:
            raise RuntimeError(
                f'Structure attribute {node.attr} is not a member of the structure {struct.name} type definition')

        # We first lift the slice into a separate view, and then the attribute access.
        slice_view_name = 'v_' + self.data_name + '_slice'
        attr_view_name = slice_view_name + '_' + node.attr
        try:
            slice_view = self.sdfg.arrays[slice_view_name]
        except KeyError:
            slice_view = dt.View.view(struct)
            slice_view_name = self.sdfg.add_datadesc(slice_view_name, slice_view, find_new_name=True)
        try:
            attr_view = self.sdfg.arrays[attr_view_name]
        except KeyError:
            member: dt.Data = struct.members[node.attr]
            attr_view = dt.View.view(member)
            attr_view_name = self.sdfg.add_datadesc(attr_view_name, attr_view, find_new_name=True)
        self.views_constructed.add(slice_view_name)
        self.views_constructed.add(attr_view_name)

        # Construct the correct AST replacement node (direct access, i.e., name node).
        replacement = ast.Name(id=attr_view_name, ctx=ast.Load())

        # Add access nodes for the views to the slice and attribute and connect them appropriately to the original data
        # container.
        lift_state, data_node = self._get_or_create_lifting_state()
        slice_view_node = lift_state.add_access(slice_view_name)
        attr_view_node = lift_state.add_access(attr_view_name)
        idx = astutils.unparse(val.slice)
        if isinstance(val.slice, ast.Tuple):
            idx = idx.strip('()')
        slice_memlet = Memlet(data_node.data + '[' + idx + ']')
        lift_state.add_edge(data_node, None, slice_view_node, 'views', slice_memlet)
        attr_memlet = Memlet.from_array(slice_view_name + '.' + node.attr, struct.members[node.attr])
        lift_state.add_edge(slice_view_node, None, attr_view_node, 'views', attr_memlet)
        return self.generic_visit(replacement)

    def _get_or_create_lifting_state(self) -> Tuple[SDFGState, nd.AccessNode]:
        # Add a state for lifting before the access, if there isn't one that was created already.
        if self._lifting_state is None:
            if isinstance(self.element, Edge):
                pre_node: ControlFlowBlock = self.element.src
                self._lifting_state = pre_node.parent_graph.add_state_after(pre_node, self.data_name + '_lifting')
            else:
                self._lifting_state = self.element[0].parent_graph.add_state_before(self.element[0])

        # Add a node for the original data container so the view can be connected to it. This may already be a view from
        # a previous iteration of lifting, but in that case it is already correctly connected to a root data container.
        data_node = None
        for dn in self._lifting_state.data_nodes():
            if dn.data == self.data_name:
                data_node = dn
                break
        if data_node is None:
            data_node = self._lifting_state.add_access(self.data_name)

        return self._lifting_state, data_node

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if not node.value:
            return self.generic_visit(node)

        if isinstance(self.data, dt.Structure):
            if isinstance(node.value, ast.Name) and node.value.id == self.data_name:
                return self._handle_simple_name_access(node)
            elif (isinstance(node.value, ast.Subscript) and isinstance(node.value.slice, ast.Constant)
                  and node.value.slice.value == 0 and isinstance(node.value.value, ast.Name)
                  and node.value.value.id == self.data_name):
                return self._handle_simple_name_access(node)
            return self.generic_visit(node)
        else:
            # ContainerArray case.
            if isinstance(node.value, ast.Name) and node.value.id == self.data_name:
                # We are directly accessing a slice of a container array / view. That needs an inserted view to the
                # container first.
                slice_view_name = 'v_' + self.data_name + '_slice'
                try:
                    slice_view = self.sdfg.arrays[slice_view_name]
                except KeyError:
                    slice_view = dt.View.view(self.data.stype)
                    slice_view_name = self.sdfg.add_datadesc(slice_view_name, slice_view, find_new_name=True)
                self.views_constructed.add(slice_view_name)

                # Add an access node for the slice view and connect it appropriately to the root data container.
                lift_state, data_node = self._get_or_create_lifting_state()
                slice_view_node = lift_state.add_access(slice_view_name)
                lift_state.add_edge(data_node, None, slice_view_node, 'views',
                                    Memlet.from_array(self.data_name, self.sdfg.data(self.data_name)))
            elif (isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name)
                  and node.value.value.id == self.data_name):
                return self._handle_sliced_access(node, node.value)
            return self.generic_visit(node)


def _data_containers_in_ast(node: ast.AST, arrnames: Set[str]) -> Set[str]:
    result: Set[str] = set()
    for subnode in ast.walk(node):
        if isinstance(subnode, (ast.Attribute, ast.Subscript)):
            data = astutils.rname(subnode.value)
            if data in arrnames:
                result.add(data)
    return result


class LiftStructViews(ppl.Pass):
    """
    Lift direct accesses to struct members to accesses to views pointing to that struct member.
    For example, an access to a struct A's member B inside of a tasklet (e.g., `A.B = 0`) may be represented with an
    access node to A being fed into the tasklet, and the dereferencing of the struct member happens inside the tasklet.
    This pass instead lifts this into an access node to A that dereferences into a view node to member A.B, which in
    turn is fed into the tasklet (e.g., `view_A_B = 0`). This correctly exposes the actual data accesses corresponding
    to structure members.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Tasklets | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes & ppl.Modifies.Tasklets & ppl.Modifies.Memlets

    def depends_on(self):
        return {}

    def _lift_control_flow_region_access(self, cfg: ControlFlowRegion, result: Dict[str, Set[str]]) -> bool:
        lifted_something = False
        lifting_state = None
        for code_block in cfg.get_meta_codeblocks():
            codes = code_block.code if isinstance(code_block.code, list) else [code_block.code]
            for code in codes:
                for data in _data_containers_in_ast(code, cfg.sdfg.arrays.keys()):
                    if '.' in data:
                        continue
                    container = cfg.sdfg.arrays[data]
                    if isinstance(container, (dt.Structure, dt.ContainerArray)):
                        if lifting_state is None:
                            lifting_state = cfg.parent_graph.add_state_before(cfg)
                        visitor = InterstateEdgeRecoder(cfg.sdfg, (cfg, code_block), data, container, lifting_state)
                        visitor.visit(code)
                        if visitor.views_constructed:
                            result[data].update(visitor.views_constructed)
                            lifted_something = True
        return lifted_something

    def _lift_isedge(self, cfg: ControlFlowRegion, edge: Edge[InterstateEdge], result: Dict[str, Set[str]]) -> bool:
        lifted_something = False
        for k in edge.data.assignments.keys():
            assignment = edge.data.assignments[k]
            assignment_str = str(assignment)
            assignment_ast = ast.parse(assignment_str)
            data_in_edge = _data_containers_in_ast(assignment_ast, cfg.sdfg.arrays.keys())
            for data in data_in_edge:
                if '.' in data:
                    continue
                container = cfg.sdfg.arrays[data]
                if isinstance(container, (dt.Structure, dt.ContainerArray)):
                    visitor = InterstateEdgeRecoder(cfg.sdfg, edge, data, container)
                    new_code = visitor.visit(assignment_ast)
                    edge.data.assignments[k] = astutils.unparse(new_code)
                    assignment_ast = new_code
                    if visitor.views_constructed:
                        result[data].update(visitor.views_constructed)
                        lifted_something = True
        if not edge.data.is_unconditional():
            condition_ast = edge.data.condition.code[0]
            data_in_edge = _data_containers_in_ast(condition_ast, cfg.sdfg.arrays.keys())
            for data in data_in_edge:
                if '.' in data:
                    continue
                container = cfg.sdfg.arrays[data]
                if isinstance(container, (dt.Structure, dt.ContainerArray)):
                    visitor = InterstateEdgeRecoder(cfg.sdfg, edge, data, container)
                    new_code = visitor.visit(condition_ast)
                    edge.data.condition = CodeBlock([new_code])
                    condition_ast = new_code
                    if visitor.views_constructed:
                        result[data].update(visitor.views_constructed)
                        lifted_something = True
        return lifted_something

    def _lift_tasklet(self, state: SDFGState, data_node: nd.AccessNode, tasklet: nd.Tasklet,
                      edge: MultiConnectorEdge[Memlet], data: dt.Structure, connector: str,
                      direction: dirtype) -> Set[str]:
        # Only handle Python at the moment.
        if not tasklet.language == dtypes.Language.Python:
            return

        new_names = set()

        # Perform lifting.
        code_list = tasklet.code.code if isinstance(tasklet.code.code, list) else [tasklet.code.code]
        for code in code_list:
            visitor = RecodeAttributeNodes(state, data_node, connector, data, tasklet, edge.data, direction)
            visitor.visit(code)
            new_names.update(visitor.views_constructed)

        # Clean up by removing the lifted connector and connected edges.
        state.remove_edge(edge)
        if direction == 'in':
            if len(list(state.in_edges_by_connector(tasklet, connector))) == 0:
                tasklet.remove_in_connector(connector)
        else:
            if len(list(state.out_edges_by_connector(tasklet, connector))) == 0:
                tasklet.remove_out_connector(connector)

        return new_names

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Dict[str, Set[str]]]:
        """
        Lift struct member accesses to explicit views, returning a dictionary that indicates what accesses were lifted.
        :param sdfg: The SDFG to modify.
        :param _: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass results as
                  ``{Pass subclass name: returned object from pass}``. If not run in a pipeline, an empty dictionary is
                  expected.
        :return: A dictionary mapping structure names to a set of view names created for accesses to their members,
                 or None if no accesses were lifted.
        """
        result = defaultdict(set)

        lifted_something = False
        while True:
            lifted_something_this_round = False
            for cfg in sdfg.all_control_flow_regions(recursive=True):
                for block in cfg.nodes():
                    if isinstance(block, SDFGState):
                        for node in block.data_nodes():
                            cont = cfg.sdfg.data(node.data)
                            if (isinstance(cont, (dt.Structure, dt.StructureView, dt.StructureReference))
                                    or (isinstance(cont,
                                                   (dt.ContainerView, dt.ContainerArray, dt.ContainerArrayReference))
                                        and isinstance(cont.stype, dt.Structure))):
                                for oedge in block.out_edges(node):
                                    if isinstance(oedge.dst, nd.Tasklet):
                                        res = self._lift_tasklet(block, node, oedge.dst, oedge, cont, oedge.dst_conn,
                                                                 'in')
                                        result[node.data].update(res)
                                        lifted_something_this_round = True
                                for iedge in block.in_edges(node):
                                    if isinstance(iedge.src, nd.Tasklet):
                                        res = self._lift_tasklet(block, node, iedge.src, iedge, cont, iedge.src_conn,
                                                                 'out')
                                        result[node.data].update(res)
                                        lifted_something_this_round = True
                for edge in cfg.edges():
                    lifted_something_this_round |= self._lift_isedge(cfg, edge, result)

                lifted_something_this_round |= self._lift_control_flow_region_access(cfg, result)
            if not lifted_something_this_round:
                break
            else:
                lifted_something = True

        if not lifted_something:
            return None
        else:
            return result

    def report(self, pass_retval: Optional[Dict[str, Set[str]]]) -> Optional[str]:
        if pass_retval is not None:
            total_lifted = 0
            all_lifted = set()
            for v in pass_retval.values():
                total_lifted += len(v)
                all_lifted.update(v)
            f'Lifted {total_lifted} accesses: {all_lifted}'
        else:
            return 'No modifications performed'
