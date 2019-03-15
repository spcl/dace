""" Transformation graph pane management, includes both the (deprecated) 
    GraphViz version and Tree-View version. """
import uuid
import copy
from .DaceState import DaceState
from dace.transformation.pattern_matching import Transformation
from dace.transformation.optimizer import SDFGOptimizer
from diode.rendered_graph import RenderedGraph
import re
import datetime
import gi
from gi.repository import Gtk, Gdk
gi.require_version('Gtk', '3.0')


class OptimizationGraphNode:
    """ Representation of a node (transformation) in the transformation 
        graph. """

    def __init__(self, label=None):

        self.uid = str(uuid.uuid4())
        self.label = label
        self.color = "#000000FF"
        self.dace_state = None
        self.pattern_match = None
        self.expanded = False
        self.parent = None

        if label == None:
            self.label = str(self.uid)
        if type(self.label) != str:
            raise (TypeError)

    def to_dot(self):
        return "\"" + self.uid + "\"" + " [color=\"" + self.color + \
               "\", label=\"" + self.label + "\"];"

    def set_color(self, r, g, b, a=1.0):
        self.color = "#%02x%02x%02x%02x" % (int(r * 255), int(g * 255),
                                            int(b * 255), int(a * 255))

    def set_dace_state(self, dace_state):
        self.dace_state = dace_state

    def get_dace_state(self):
        return self.dace_state

    def set_pattern_match(self, pattern_match):
        self.pattern_match = pattern_match

    def get_pattern_match(self):
        return self.pattern_match

    def get_expanded(self):
        return self.expanded

    def apply_pattern_match(self):
        if self.pattern_match is None: return
        if self.parent is None: raise ValueError
        if self.dace_state is None:
            self.dace_state = copy.deepcopy(self.parent.get_dace_state())
            top_level_sdfg = self.dace_state.get_sdfg()
            actual_sdfg = top_level_sdfg.sdfg_list[self.pattern_match.sdfg_id]
            self.pattern_match.apply_pattern(actual_sdfg)

    def get_label(self):
        return self.label

    def get_uid(self):
        return self.uid

    def set_expanded(self, status=True):
        self.expanded = status

    def is_expanded(self):
        return self.expanded

    def set_parent(self, node):
        self.parent = node

    def get_parent(self):
        return self.parent


class OptimizationGraphEdge:
    """ Representation of an edge (transformation application) in the 
        transformation graph. """

    def __init__(self, tail, head, label="", pattern=None):
        self.pattern = pattern
        if type(tail) != OptimizationGraphNode:
            print("tail argument to OptimizationGraphEdge is not a node")
        if type(head) != OptimizationGraphNode:
            print("head argument to OptimizationGraphEdge is not a node")

        self.tail = tail
        self.head = head
        self.label = label
        self.uid = str(uuid.uuid4())

    def set_label(self, label):
        self.label = label

    def to_dot(self):
        return "\"" + self.tail.get_uid() + "\"" + " -> " + "\"" + \
               self.head.get_uid() + "\"" + "[label=\"" + self.label + "\"];"


class OptimizationGraph:
    """ Representation of the transformation graph structure. """

    def __init__(self, graph_da, treeview_widget, expand_node_callback,
                 hover_node_callback, activate_node_callback):
        self.treestore = Gtk.TreeStore(str, Gdk.Color)
        self.nodes = []
        self.edges = []
        self.current = None
        self.expand_node_callback = expand_node_callback
        self.hover_node_callback = hover_node_callback
        self.activate_node_callback = activate_node_callback
        self.highlighted_elements = []
        self.treeview = treeview_widget

        self.last_click_time = 0

        # Initialize the graph view
        self.rendered_optgraph = RenderedGraph()
        self.optgraph_da = graph_da
        self.rendered_optgraph.set_drawing_area(graph_da)
        self.optgraph_da.connect("draw", self.OnDrawGraph)
        self.optgraph_da.connect("scroll-event", self.OnScrollGraph)
        self.optgraph_da.connect("button-press-event", self.OnButtonPressGraph)
        self.optgraph_da.connect("button-release-event",
                                 self.OnButtonReleaseGraph)
        self.optgraph_da.connect("motion-notify-event", self.OnMouseMoveGraph)

        # Initialize the treeview widget
        renderer = Gtk.CellRendererText()
        column = Gtk.TreeViewColumn(
            "Optimization", renderer, text=0, foreground_gdk=1)
        self.treeview.append_column(column)
        self.treeview.connect("row-activated", self.OnRowActivate, None)
        selection = self.treeview.get_selection()
        selection.connect("changed", self.OnRowChanged, None)

    def OnDrawGraph(self, widget, cr):
        self.rendered_optgraph.render(widget, cr)
        return False

    def OnOptgraphNodeExpand(self, nodename):
        self.ExpandNode(nodename)

    def OnButtonPressGraph(self, widget, ev):
        x, y = ev.x, ev.y
        elem = self.rendered_optgraph.get_element_by_coords(x, y)
        self.rendered_optgraph.handle_button_press(ev)
        time_now = int((datetime.datetime.utcnow() - datetime.datetime(
            2015, 1, 1)).total_seconds() * 1000)
        delta_t = time_now - self.last_click_time
        self.last_click_time = time_now
        if type(elem).__name__ == "Node":
            nodename = elem.id.decode('utf-8')
            clicked = self.find_node(nodename)
            self.rendered_optgraph.clear_highlights()
            self.rendered_optgraph.highlight_element(elem)
            self.activate_node_callback(nodename, clicked.pattern_match)
            if delta_t < 250:
                # Double click, expand, make sure to use the right pattern
                # match properties
                self.set_current(clicked)
                self.expand_node(clicked)
                self.expand_node_callback(nodename, clicked.pattern_match)
        return False

    def OnMouseMoveGraph(self, widget, ev):
        self.rendered_optgraph.handle_drag_motion(ev)
        x, y = ev.x, ev.y
        elem = self.rendered_optgraph.get_element_by_coords(x, y)
        if type(elem).__name__ == "Node":
            nodename = elem.id.decode('utf-8')
            hovered = self.find_node(nodename)
            self.hover_node_callback(nodename, hovered.pattern_match)
        else:
            self.hover_node_callback(None, None)
        return False

    def OnButtonReleaseGraph(self, widget, ev):
        self.rendered_optgraph.handle_button_release(ev)
        return False

    def OnScrollGraph(self, widget, ev):
        d = self.rendered_optgraph.determine_scroll_direction(ev)
        self.rendered_optgraph.zoom(d, pos=(ev.x, ev.y))
        widget.queue_draw()
        return False

    def transform_label_uc_to_num(self, label_with_underscores):
        m = re.search("^(.+?)(_*)$", label_with_underscores)
        nuc = len(m.group(2))
        return m.group(1) + "_(" + str(nuc) + ")"

    def transform_label_num_to_uc(self, label_with_underscores):
        m = re.search("^(.+?)_\((\d+)\)$", label_with_underscores)
        if m is None:
            raise ValueError(str(label_with_underscores) + \
                             " is not the kind of label I expected")
        nuc = int(m.group(2))
        uc = "_" * nuc
        return m.group(1) + uc

    def parse_color_to_gdk_color(self, color):
        # #00ff00ff
        m = re.match("#(..)(..)(..)", color)
        if m is None:
            raise ValueError(str(color) + " is not a recognized " "color")
        r, g, b = int(m.group(1), 16), int(m.group(2), 16), int(m.group(3), 16)
        # GDK colors are 16-bit
        r = r / 255 * 65535
        g = g / 255 * 65535
        b = b / 255 * 65535
        color = Gdk.Color(r, g, b)
        return color

    def update_treestore(self, parent, parent_path):
        # Add all the edges from n
        for e in self.edges:
            if e.tail == parent:
                label = self.transform_label_uc_to_num(e.head.label)
                color = self.parse_color_to_gdk_color(e.head.color)
                p = self.treestore.append(
                    parent=parent_path, row=[label, color])
                self.update_treestore(e.head, p)

    def OnChange(self, preserve_view=False):
        # This is called whenever something about the stored data changed.
        self.rendered_optgraph.set_dotcode(self.to_dot(), preserve_view)

        # Clear the treeview
        self.treestore.clear()

        # Do a graph traversal and add everything to the treeview
        for n in self.nodes:
            if n.parent == None:
                label = self.transform_label_uc_to_num(n.label)
                color = self.parse_color_to_gdk_color(n.color)
                p = self.treestore.append(parent=None, row=[label, color])
                self.update_treestore(n, p)
        self.treeview.set_model(self.treestore)

    def OnRowChanged(self, treeselection, userdata):
        model, paths = treeselection.get_selected_rows()
        for path in paths:
            titer = model.get_iter(path)
            if titer is not None:
                value = model.get_value(titer, 0)
                nodename = self.transform_label_num_to_uc(value)
                nodes = self.find_nodes_by_label(nodename)
                if len(nodes) != 1:
                    raise ValueError("More than one optgraph node named " + \
                                     str(nodename))
                pm = nodes[0].get_pattern_match()
                self.hover_node_callback(nodename, pm)

    def clear(self):
        self.nodes = []
        self.edges = []
        self.current = None

    def get_nodes(self):
        return self.nodes

    def OnRowActivate(self, treeview, path, view_column, userdata):
        model = treeview.get_model()
        titer = model.get_iter(path)
        if titer is not None:
            value = model.get_value(titer, 0)
            nodename = self.transform_label_num_to_uc(value)
            nodes = self.find_nodes_by_label(nodename)
            if len(nodes) != 1:
                raise ValueError("More than one optgraph node named " + \
                        str(nodename))
            pm = nodes[0].get_pattern_match()
            nodeid = nodes[0].get_uid()
            self.rendered_optgraph.clear_highlights()
            self.set_current(nodes[0])
            self.expand_node(nodes[0])
            self.expand_node_callback(nodeid, pm)

    def set_current(self, node):
        if type(node) == str:
            node = self.find_node(node)
        current = self.get_current()
        if current is not None: self.set_explored(current)
        self.current = node
        node.apply_pattern_match()
        node.get_dace_state().compile()
        node.set_color(0, 1, 0)

    def set_explored(self, node):
        if type(node) == str:
            node = self.find_node(node)
        node.set_color(0, 0, 0)

    def set_unexplored(self, node):
        if type(node) == str:
            node = self.find_node(node)
        node.set_color(0.6, 0.6, 0.6)

    def get_current(self):
        return self.current

    def add_node(self, parent=None, pattern=None, label=None):
        existing = self.find_nodes_by_label(label)
        while len(existing) > 0:
            label += "_"
            existing = self.find_nodes_by_label(label)
        node = OptimizationGraphNode(label=label)
        self.nodes.append(node)
        node.set_parent(parent)
        node.set_pattern_match(pattern)
        return node

    def add_edge(self, tail, head, label="", pattern=None):
        t = tail
        h = head
        if type(tail) == str:
            t = self.find_node(tail)
        if type(head) == str:
            h = self.find_node(head)
        self.edges.append(OptimizationGraphEdge(t, h, label, pattern))

    def find_node(self, uid):
        for n in self.nodes:
            if n.uid == uid:
                return n
        print("Node " + uid + " does not exist!")
        return None

    def find_nodes_by_label(self, label):
        res = [n for n in self.nodes if n.label == label]
        return res

    def to_dot(self):
        dotstr = "digraph G {\n"
        for n in self.nodes:
            dotstr += n.to_dot()
            dotstr += "\n"
        for e in self.edges:
            dotstr += e.to_dot()
            dotstr += "\n"
        dotstr += "}\n"
        return dotstr

    def clear_subtree(self, root):
        nodes = []
        for edge in list(self.edges):
            if edge.tail == root:
                self.edges.remove(edge)
                nodes.append(edge.head)
        for node in nodes:
            self.clear_subtree(node)
            self.nodes.remove(node)
        root.set_expanded(False)

    def expand_node(self, node):
        if type(node) == str: node = self.find_node(node)
        if node.expanded == True: return

        if node.pattern_match is not None:
            node.get_dace_state().set_sdfg(
                copy.deepcopy(node.parent.get_dace_state().get_sdfg()))
            top_level_sdfg = node.get_dace_state().get_sdfg()
            actual_sdfg = top_level_sdfg.sdfg_list[node.pattern_match.sdfg_id]
            node.pattern_match.apply_pattern(actual_sdfg)

        # Add optgraph nodes for matching patterns
        node_dace_state = node.get_dace_state()
        opt = SDFGOptimizer(node_dace_state.get_sdfg())
        ptrns = opt.get_pattern_matches()
        for p in ptrns:
            label = type(p).__name__
            nn = self.add_node(label=label, parent=node)
            nn.set_pattern_match(p)
            self.set_unexplored(nn)
            self.add_edge(tail=node, head=nn, label="", pattern=p)

        node.get_dace_state().set_is_compiled(False)
        node.get_dace_state().compile()
        node.set_expanded(True)
        self.OnChange()
