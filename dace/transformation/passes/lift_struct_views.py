# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from typing import Any, Dict, Optional, Set

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
    data: dt.Structure
    tasklet: nd.Tasklet
    direction: dirtype
    memlet: Memlet
    data_node: nd.AccessNode
    state: SDFGState

    def __init__(self, state: SDFGState, data_node: nd.AccessNode, connector: str, data: dt.Structure,
                 tasklet: nd.Tasklet, memlet: Memlet, direction: dirtype):
        self.connector = connector
        self.data = data
        self.tasklet = tasklet
        self.direction = direction
        self.memlet = memlet
        self.data_node = data_node
        self.state = state

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if not node.value or not isinstance(node.value, ast.Name) or node.value.id != self.connector:
            return self.generic_visit(node)

        if not node.attr in self.data.members:
            raise RuntimeError('Structure attribute is not a member of the structure type definition')

        # Gather a new connector name and add the appropriate connector.
        new_connector_name = node.value.id + '_' + node.attr
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
            member: dt.Data = self.data.members[node.attr]
            if isinstance(member, dt.Structure):
                view = dt.StructureView(member.members, view_name, True, member.storage, member.location,
                                        member.lifetime, member.debuginfo)
                self.state.sdfg.add_datadesc(view_name, view)
            else:
                view_name, view = self.state.sdfg.add_view(view_name, member.shape, member.dtype, member.storage,
                                                           member.strides,
                                                           member.offset if isinstance(member, dt.Array) else None)

        # Add an access node for the view and connect it appropriately.
        view_node = self.state.add_access(view_name)
        if self.direction == 'in':
            self.state.add_edge(self.data_node, None, view_node, 'views',
                                Memlet.from_array(self.data_node.data + '.' + node.attr, self.data.members[node.attr]))
            # TODO: determine the actual subset from the tasklet accesses.
            self.state.add_edge(view_node, None, self.tasklet, new_connector_name,
                                Memlet.from_array(view_name, view))
        else:
            self.state.add_edge(view_node, 'views', self.data_node, None,
                                Memlet.from_array(self.data_node.data + '.' + node.attr, self.data.members[node.attr]))
            # TODO: determine the actual subset from the tasklet accesses.
            self.state.add_edge(self.tasklet, new_connector_name, view_node, None,
                                Memlet.from_array(view_name, view))

        return self.generic_visit(replacement)


class LiftStructViews(ppl.Pass):
    """
    TODO
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Tasklets | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes & ppl.Modifies.Tasklets & ppl.Modifies.Memlets

    def depends_on(self):
        return {}

    def _lift_tasklet_accesses(self, state: SDFGState, data_node: nd.AccessNode, tasklet: nd.Tasklet,
                               edge: MultiConnectorEdge[Memlet], data: dt.Structure, connector: str,
                               direction: dirtype):
        # Only handle Python at the moment.
        if not tasklet.language == dtypes.Language.Python:
            return

        # Perform lifting.
        code_list = tasklet.code.code if isinstance(tasklet.code.code, list) else [tasklet.code.code]
        new_code_list = []
        for code in code_list:
            visitor = RecodeAttributeNodes(state, data_node, connector, data, tasklet, edge.data, direction)
            new_code = visitor.visit(code)
            new_code_list.append(new_code)

        # Clean up by removing the lifted connector and connected edges.
        state.remove_edge(edge)
        if direction == 'in':
            tasklet.remove_in_connector(connector)
        else:
            tasklet.remove_out_connector(connector)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """
        TODO
        """
        results = dict()

        lifted_something = False
        for state in sdfg.states():
            for node in state.data_nodes():
                container = sdfg.data(node.data)
                if isinstance(container, dt.Structure):
                    for oedge in state.out_edges(node):
                        if isinstance(oedge.dst, nd.Tasklet):
                            self._lift_tasklet_accesses(state, node, oedge.dst, oedge, container, oedge.dst_conn,
                                                        'in')
                            lifted_something = True
                    for iedge in state.in_edges(node):
                        if isinstance(iedge.src, nd.Tasklet):
                            self._lift_tasklet_accesses(state, node, iedge.src, iedge, container, iedge.src_conn,
                                                        'out')
                            lifted_something = True

        if not lifted_something:
            return None
        else:
            return results

    def report(self, pass_retval: Any) -> Optional[str]:
        # TODO
        return ''