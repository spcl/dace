# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from collections import defaultdict
from typing import Any, Dict, Optional, Set, Union

from dace import SDFG, Memlet, SDFGState
from dace.sdfg import nodes as nd
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl
from dace import data as dt
from dace import dtypes


import sys
if sys.version_info >= (3, 8):
    from typing import Literal
    dirtype = Literal['in', 'out']
else:
    dirtype = "Literal['in', 'out']"

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
                f'Structure attribute {node.attr} is not a member of the structure {struct.name} type definition'
            )

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
        replacement = ast.Name()
        replacement.ctx = ast.Load()
        replacement.id = new_connector_name

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
            self.state.add_edge(view_node, None, self.tasklet, new_connector_name,
                                Memlet.from_array(view_name, view))
        else:
            self.state.add_edge(view_node, 'views', self.data_node, None,
                                Memlet.from_array(self.data_node.data + '.' + node.attr, self.data.members[node.attr]))
            self.state.add_edge(self.tasklet, new_connector_name, view_node, None,
                                Memlet.from_array(view_name, view))
        return self.generic_visit(replacement)

    def _handle_sliced_access(self, node: ast.Attribute, val: ast.Slice) -> Any:
        struct = self.data.stype
        if not isinstance(struct, dt.Structure):
            raise ValueError('Invalid ContainerView, can only lift ContainerViews to Structures')
        if not node.attr in struct.members:
            raise RuntimeError(
                f'Structure attribute {node.attr} is not a member of the structure {struct.name} type definition'
            )

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
        replacement = ast.Name()
        replacement.ctx = ast.Load()
        replacement.id = new_connector_name

        # Add access nodes for the views and connect them appropriately.
        slice_view_node = self.state.add_access(slice_view_name)
        attr_view_node = self.state.add_access(attr_view_name)
        if self.direction == 'in':
            idx = ast.unparse(val.slice)
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
            idx = ast.unparse(val.slice)
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
            elif (isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name) and
                  node.value.value.id == self.connector):
                return self._handle_sliced_access(node, node.value)
            return self.generic_visit(node)
        else:
            raise NotImplementedError()


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

    def _lift_tasklet(self, state: SDFGState, data_node: nd.AccessNode, tasklet: nd.Tasklet,
                      edge: MultiConnectorEdge[Memlet], data: dt.Structure, connector: str,
                      direction: dirtype) -> Set[str]:
        # Only handle Python at the moment.
        if not tasklet.language == dtypes.Language.Python:
            return

        new_names = set()

        # Perform lifting.
        code_list = tasklet.code.code if isinstance(tasklet.code.code, list) else [tasklet.code.code]
        new_code_list = []
        for code in code_list:
            visitor = RecodeAttributeNodes(state, data_node, connector, data, tasklet, edge.data, direction)
            new_code = visitor.visit(code)
            new_code_list.append(new_code)
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
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for node in state.data_nodes():
                    cont = nsdfg.data(node.data)
                    if (isinstance(cont, (dt.Structure, dt.StructureView, dt.StructureReference)) or
                        (isinstance(cont, (dt.ContainerView, dt.ContainerArray, dt.ContainerArrayReference)) and
                        isinstance(cont.stype, dt.Structure))):
                        for oedge in state.out_edges(node):
                            if isinstance(oedge.dst, nd.Tasklet):
                                res = self._lift_tasklet(state, node, oedge.dst, oedge, cont, oedge.dst_conn, 'in')
                                result[node.data].update(res)
                                lifted_something = True
                        for iedge in state.in_edges(node):
                            if isinstance(iedge.src, nd.Tasklet):
                                res = self._lift_tasklet(state, node, iedge.src, iedge, cont, iedge.src_conn, 'out')
                                result[node.data].update(res)
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
