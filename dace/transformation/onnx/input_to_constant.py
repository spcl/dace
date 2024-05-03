from typing import Dict

import dace
from dace import registry, dtypes, properties, memlet as mm, symbolic
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.replace import replace_properties
from dace.transformation import transformation as xf

from dace.libraries.onnx import ONNXModel
from dace.libraries.onnx.converters import clean_onnx_name


def forward_memlet_tree_with_nested_and_copies(state, edge) -> mm.MemletTree:
    # Obtain the full state (to work with paths that trace beyond a scope)
    state = state._graph

    # Find tree root
    curedge = edge
    while (isinstance(curedge.src, nodes.EntryNode)
           and curedge.src_conn is not None):
        assert curedge.src_conn.startswith('OUT_')
        cname = curedge.src_conn[4:]
        curedge = next(e for e in state.in_edges(curedge.src)
                       if e.dst_conn == 'IN_%s' % cname)

    tree_root = mm.MemletTree(curedge)
    tree_root.state = state

    # Collect children (recursively)
    def add_children(treenode):
        # HACK: store the parent state as a undocumented attribute of treenode
        state = treenode.state
        is_entry_node = (isinstance(treenode.edge.dst, nodes.EntryNode)
                         and treenode.edge.dst_conn
                         and treenode.edge.dst_conn.startswith('IN_'))

        def make_tree(e, parent, state):
            tree = mm.MemletTree(e, parent=treenode)
            tree.state = state
            return tree

        if is_entry_node:
            conn = treenode.edge.dst_conn[3:]
            treenode.children = [
                mm.MemletTree(e, parent=treenode)
                for e in state.out_edges(treenode.edge.dst)
                if e.src_conn == 'OUT_%s' % conn
            ]
            for c in treenode.children:
                c.state = state
        elif isinstance(treenode.edge.dst, nodes.NestedSDFG):

            # todo what about shadowing in nested SDFGS
            access_nodes = (
                (n, parent)
                for n, parent in treenode.edge.dst.sdfg.all_nodes_recursive()
                if isinstance(n, nodes.AccessNode)
                and n.data == treenode.edge.dst_conn)

            treenode.children = []
            for access_node, parent in access_nodes:
                treenode.children.extend(
                    make_tree(e, treenode, parent)
                    for e in parent.out_edges(access_node))
        elif isinstance(treenode.edge.dst, nodes.AccessNode):
            # this is ok if this is just a copy of all elements

            sdfg: dace.SDFG = state.parent
            copied_data_name = treenode.edge.dst.data

            # semi-hack: check that the subset is complete
            if edge.data.subset.num_elements() != sdfg.arrays[
                    edge.data.data].total_size:
                return

            # also check that the copy is never written to (except for here)
            if any(
                    parent.in_degree(n) > 0
                    for n, parent in sdfg.all_nodes_recursive()
                    if isinstance(n, nodes.AccessNode) and n.data ==
                    copied_data_name and n is not treenode.edge.dst):
                return

            if state.in_degree(treenode.edge.dst) != 1:
                return

            # todo what about shadowing in nested SDFGS (should not descend into nested SDFGs)
            access_nodes = ((n, parent)
                            for n, parent in sdfg.all_nodes_recursive()
                            if isinstance(n, nodes.AccessNode)
                            and n.data == copied_data_name)

            for access_node, parent in access_nodes:
                treenode.children.extend(
                    make_tree(e, treenode, parent)
                    for e in parent.out_edges(access_node))
        else:
            return

        for child in treenode.children:
            add_children(child)

    # Start from root node (obtained from above parent traversal)
    add_children(tree_root)

    # Find edge in tree
    def traverse(node):
        if node.edge == edge:
            return node
        for child in node.children:
            res = traverse(child)
            if res is not None:
                return res
        return None

    # Return node that corresponds to current edge
    return traverse(tree_root)


def print_tree(tree):
    return "{} -> {}".format(tree.edge.src, tree.edge.dst) + "".join(
        "\n |\n +- {}".format(print_tree(c)) for c in tree.children)


def _replace_in_tasklet(tasklet: nodes.Tasklet, name: str, new_name: str):
    symname = symbolic.symbol(name)
    symrepl = {
        symname:
        symbolic.pystr_to_symbolic(new_name)
        if isinstance(new_name, str) else new_name
    }

    # Replace in node properties
    replace_properties(tasklet, symrepl, name, new_name)


@properties.make_properties
class InputToConstant(xf.SingleStateTransformation):
    """ Convert constant inputs to dace compile time constants.
    """

    access_node = xf.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.access_node)]

    def can_be_applied(self,
                       state: dace.SDFGState,
                       expr_index: int,
                       sdfg,
                       permissive: bool = False):
        # SDFG must be imported from an ONNXModel
        if not hasattr(sdfg, "_parent_onnx_model"):
            return False

        node: nodes.AccessNode = self.access_node

        # check that the data is a onnx parameter
        if node.data not in {
                clean_onnx_name(w)
                for w in sdfg._parent_onnx_model.weights
        }:
            return False

        # check that the data is never written to
        if any(
                len(parent.in_edges(n)) > 0
                for n, parent in sdfg.all_nodes_recursive()
                if isinstance(n, nodes.AccessNode) and n.data == node.data):
            return False

        for out_edge in state.out_edges(node):
            # check that the memlet tree leaves are all tasklets
            tree = forward_memlet_tree_with_nested_and_copies(state, out_edge)
            for child in tree.traverse_children(include_self=True):
                if child.children != []:
                    continue
                if not isinstance(child.edge.dst, nodes.Tasklet):
                    return False
                if child.edge.dst.language not in [dtypes.Language.Python]:
                    return False

        return True

    def match_to_str(self, graph):
        node = self.access_node
        return "Convert '{}' to a compile time constant".format(node.data)

    def apply(self, state: dace.SDFGState, sdfg: dace.SDFG):
        parent: ONNXModel = sdfg._parent_onnx_model
        node = self.access_node
        data_name = node.data

        # add the weight as a dace constant
        unclean_onnx_name = {clean_onnx_name(w): w
                             for w in parent.weights}[node.data]
        from torch import Tensor
        data = parent.weights[unclean_onnx_name].numpy() if isinstance(
            parent.weights[unclean_onnx_name],
            Tensor) else parent.weights[unclean_onnx_name]
        sdfg.add_constant(data_name, data, sdfg.arrays[node.data])

        for out_edge in state.out_edges(node):
            tree = forward_memlet_tree_with_nested_and_copies(state, out_edge)

            while tree.parent is not None:
                tree = tree.parent

            for child in tree.traverse_children(include_self=True):
                if child.children != []:
                    continue

                # we have reached an edge that should go into a python tasklet
                root_edge = child.edge
                tasklet = root_edge.dst
                conn_name = root_edge.dst_conn
                assert isinstance(tasklet, nodes.Tasklet)

                # remove the input from the tasklet
                tasklet.remove_in_connector(conn_name)
                root_edge.dst_conn = None

                # add the constant access to the top of the tasklet
                if len(data.shape) > 0:
                    access_str = "{}[{}]".format(data_name,
                                                 root_edge.data.subset)
                    tasklet.code = properties.CodeBlock(
                        "{} = {}\n".format(conn_name, access_str) +
                        tasklet.code.as_string, tasklet.language)
                else:  # if scalar, inline constant into tasklet contents
                    _replace_in_tasklet(tasklet, conn_name, data_name)

                # wipe the memlets off the tree
                state.remove_memlet_path(root_edge)

            # remove in parent SDFGs
            for sub_tree in tree.traverse_children(include_self=True):
                edge = sub_tree.edge
                if isinstance(edge.dst, nodes.NestedSDFG):
                    del edge.dst.sdfg.arrays[edge.dst_conn]
                    try:
                        sub_tree.state.remove_memlet_path(edge)
                    except KeyError:
                        pass  # memlet path was already removed

        # if this was the last node, remove the array from the sdfg and the OnnxModel
        if not any(True for n, parent in sdfg.all_nodes_recursive()
                   if isinstance(n, nodes.AccessNode) and n.data == node.data):
            del sdfg.arrays[node.data]
            del parent.weights[unclean_onnx_name]
