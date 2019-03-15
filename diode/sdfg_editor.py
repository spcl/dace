import gi
import re
from collections import OrderedDict
gi.require_version('Gtk', '3.0')
gi.require_version('GtkSource', '3.0')
from gi.repository import Gtk, GtkSource

import dace
from diode.rendered_graph import RenderedGraph
from diode.images import ImageStore
from diode.property_renderer import PropertyRenderer, _get_edge_label


class SDFGEditor:
    def __init__(self, builder):

        self.buttons = [
            {
                "image": "cursor.png",
                "type": "mouse",
                "tool": "Mouse"
            },
            {
                "image": "delete.png",
                "type": "delete",
                "tool": "Delete"
            },
            {
                "image": "array.png",
                "type": "node",
                "tool": "Array"
            },
            {
                "image": "edge_thin.png",
                "type": "edge",
                "tool": "Memlet"
            },
            {
                "image": "map.png",
                "type": "node",
                "tool": "Map"
            },
            {
                "image": "tasklet.png",
                "type": "node",
                "tool": "Tasklet"
            },
            {
                "image": "stream.png",
                "type": "node",
                "tool": "Stream"
            },
            {
                "image": "stream_map.png",
                "type": "node",
                "tool": "Consume"
            },
            {
                "image": "state.png",
                "type": "node",
                "tool": "State"
            },
            {
                "image": "state_trans.png",
                "type": "edge",
                "tool": "State Transition"
            },
            {
                "image": "edge_head_redir.png",
                "type": "edge_redir",
                "tool": "Head Redirection"
            },
            {
                "image": "edge_tail_redir.png",
                "type": "edge_redir",
                "tool": "Tail Redirection"
            },
        ]

        self.active_tool = None  # an element of self.buttons
        self.builder = builder
        self.current_editing_script = ""
        self.sdfg_changed = False

        # Initialize the SDFG to a valid one. Otherwise, we need
        # to check in all the functions that use it if it is None.
        self.sdfg = dace.SDFG("newsdfg", OrderedDict(), {})

        self.first_selected_node_for_edge = None
        self.first_selected_state_for_edge = None
        self.selected_edge_for_redir = None

        self.rendered_sdfg = RenderedGraph()
        self.sdfg_da = self.builder.get_object("sdfg_editor_da")
        self.rendered_sdfg.set_drawing_area(self.sdfg_da)

        plabel = self.builder.get_object("se_propertylabel")
        pgrid = self.builder.get_object("se_propertygrid")
        self.propren = PropertyRenderer(plabel, pgrid, self.OnSDFGUpdate)

        self.image_store = ImageStore()
        self.load_buttons()
        self.connect_signals()

    def emit_script_cmd(self, cmd):
        self.current_editing_script += "sdfg_edit." + cmd + "\n"
        self.sdfg_changed = True

    def reset_edit_script(self):
        self.sdfg_changed = False
        self.current_editing_script = ""

    def get_sdfg(self):
        return self.sdfg

    def get_edit_script(self):
        return self.current_editing_script

    def connect_signals(self):
        sdfg_da = self.builder.get_object("sdfg_editor_da")
        sdfg_da.connect("draw", self.OnDrawSDFG)
        sdfg_da.connect("scroll-event", self.OnScrollSDFG)
        sdfg_da.connect("button-press-event", self.OnButtonPressSDFG)
        sdfg_da.connect("button-release-event", self.OnButtonReleaseSDFG)
        sdfg_da.connect("motion-notify-event", self.OnMouseMoveSDFG)

    def sdfg_modified(self):
        """ Returns True if the SDFG has been edited. """
        return self.sdfg_changed

    def OnSDFGUpdate(self, sdfg, elemtype, *args):
        if elemtype == "node":
            nodeid, propname, newval = args
            self.emit_script_cmd("ChangeSDFGNodeProperties(\"" + \
                                 str(nodeid) +   "\", \"" + \
                                 str(propname) + "\", \"" + \
                                 str(newval) + "\"")
        elif elemtype == "memlet":
            tail, head, label, propname, newval = args
            self.emit_script_cmd("ChangeSDFGMemletProperties(\"" + \
                                 str(tail) +     "\", \"" + \
                                 str(head) +     "\", \"" + \
                                 str(label) +    "\", \"" + \
                                 str(propname) + "\", \"" + \
                                 str(newval) + "\"")
        self.sdfg_changed = True
        self.rendered_sdfg.set_dotcode(self.sdfg.draw())

    def load_buttons(self):
        toolbar = self.builder.get_object("sdfged_toolbar")
        for b in self.buttons:
            pixbuf = self.image_store.get_image(b["image"])
            image = Gtk.Image.new_from_pixbuf(pixbuf)
            button = Gtk.ToggleToolButton()
            button.set_icon_widget(image)
            toolbar.add(button)
            b["button"] = button
            if b["tool"] == "Mouse":
                self.active_tool = b
            button.connect("toggled", self.OnToggleTBButton, b)

    def init_syntax_highlighting(self, widgetname, language):
        tbuffer = self.builder.get_object(widgetname).get_buffer()
        lang_manager = GtkSource.LanguageManager()
        language = lang_manager.get_language(language)
        tbuffer.set_language(language)
        tbuffer.set_highlight_syntax(True)

    def set_sdfg(self, sdfg):
        self.sdfg = sdfg
        dotcode = sdfg.draw()
        self.rendered_sdfg.set_dotcode(dotcode)

    def OnDrawSDFG(self, widget, cr):
        self.rendered_sdfg.render(widget, cr)
        return False

    def OnToggleTBButton(self, widget, button):
        if button["tool"] != self.active_tool["tool"]:
            self.active_tool["button"].set_active(False)
        statuslabel = self.builder.get_object("run_status_text")

        if button["type"] == "node":
            statuslabel.set_text("Click the SDFG pane to add a " + \
                      button["tool"] + " node")
        elif button["type"] == "edge":
            statuslabel.set_text("Click two nodes between which you want " + \
                      "to add a " + button["tool"] + " edge")
        elif button["tool"] == "Delete":
            statuslabel.set_text("Click a node or edge to delete it")
        elif button["type"] == "edge_redir":
            statuslabel.set_text("Click an Edge followed by the new head/tail")
        self.active_tool = button
        return True

    def OnScrollSDFG(self, widget, ev):
        d = self.rendered_sdfg.determine_scroll_direction(ev)
        self.rendered_sdfg.zoom(d, pos=(ev.x, ev.y))
        widget.queue_draw()
        return False

    def parse_state_label(self, label):
        """ Take a state label as a string in the form "s0 (BEGIN)" and return
            zero as an integer. """
        if not isinstance(label, str): return None
        p = re.compile("s(\d+)")
        m = p.match(label)
        if m:
            return int(m.group(1))
        else:
            return None

    def show_error(self, errormsg):
        dialog = Gtk.MessageDialog(
            self.builder.get_object("main_window"), 0, Gtk.MessageType.ERROR,
            Gtk.ButtonsType.CANCEL, "User Error:")
        dialog.format_secondary_text(errormsg)
        dialog.run()
        dialog.destroy()

    def from_elem2sdfg(self, elem):
        sid, nid = self.split_nodeid_in_state_and_nodeid(elem)
        return self.sdfg.nodes()[sid].nodes()[nid]

### Scripting Interface

    def AddArray(self, arrayname, statelabel):
        state = self.parse_state_label(statelabel)
        if state is None:
            raise ValueError("State " + statelabel + " not found / parsable")
        arrtype = self.sdfg.add_array("newarray", dace.types.float64, (1, ))
        newarray = dace.graph.nodes.AccessNode(arrtype)
        self.sdfg.nodes()[state].add_node(newarray)
        self.rendered_sdfg.set_dotcode(self.sdfg.draw())
        self.sdfg_changed = True

    def AddStream(self, streamname, statelabel):
        state = self.parse_state_label(statelabel)
        if state == None:
            raise ValueError("State " + statelabel + " not found / parsable")
        streamtype = self.sdfg.add_stream("newstream", dace.types.float64, 1,
                                          0)
        newstream = dace.graph.nodes.AccessNode(streamtype)
        self.sdfg.nodes()[state].add_node(newstream)
        self.rendered_sdfg.set_dotcode(self.sdfg.draw())
        self.sdfg_changed = True

    def AddMemlet(self, memletname, srcnodeid, dstnodeid):
        srcnode = self.from_elem2sdfg(srcnodeid)
        dstnode = self.from_elem2sdfg(dstnodeid)

        # Create a dummy (but valid) array
        mdata = self.sdfg.add_array("newarray", dace.types.float64, (1, ))
        msubset = dace.subsets.Range([("0", "N-1", "1")])
        newmemlet = dace.memlet.Memlet(mdata, 1, msubset, 1)
        sid1, nid1 = self.split_nodeid_in_state_and_nodeid(srcnodeid)
        sid2, nid2 = self.split_nodeid_in_state_and_nodeid(dstnodeid)
        if sid1 != sid2:
            raise ValueError("You cannot create memlets between states.")
            return False
        else:
            # TODO: connectors
            self.sdfg.nodes()[sid1].add_edge(srcnode, None, dstnode, None,
                                             newmemlet)
        self.rendered_sdfg.set_dotcode(self.sdfg.draw())
        self.sdfg_changed = True

    def AddStateTransition(self, s1, s2):
        s1 = self.parse_state_label(s1)
        s2 = self.parse_state_label(s2)
        self.sdfg.add_edge(self.sdfg.nodes()[s1],
                           self.sdfg.nodes()[s2],
                           dace.graph.edges.InterstateEdge())
        self.rendered_sdfg.set_dotcode(self.sdfg.draw())
        self.sdfg_changed = True

    def AddMap(self, mapname, statelabel):
        newmap = dace.graph.nodes.Map(mapname, ["i"],
                                      dace.subsets.Range([("0", "N-1", "1")]))
        state = self.parse_state_label(statelabel)
        if state == None:
            raise ValueError("State " + statelabel + " not found / parsable")
        self.sdfg.nodes()[state].add_node(dace.graph.nodes.MapEntry(newmap))
        self.sdfg.nodes()[state].add_node(dace.graph.nodes.MapExit(newmap))
        self.rendered_sdfg.set_dotcode(self.sdfg.draw())
        self.sdfg_changed = True

    def AddConsume(self, consumename, statelabel):
        state = self.parse_state_label(statelabel)
        if state == None:
            raise ValueError("State " + statelabel + " not found / parsable")
        newconsume = dace.graph.nodes.Consume(
            consumename, ["i"], dace.subsets.Range([("0", "N-1", "1")]))
        self.sdfg.nodes()[state].add_node(
            dace.graph.nodes.ConsumeEntry(newconsume))
        self.sdfg.nodes()[state].add_node(
            dace.graph.nodes.ConsumeExit(newconsume))
        self.rendered_sdfg.set_dotcode(self.sdfg.draw())
        self.sdfg_changed = True

    def AddState(self, statelabel):
        newstate = dace.SDFGState(statelabel, self.sdfg)
        self.sdfg.add_node(newstate)
        self.rendered_sdfg.set_dotcode(self.sdfg.draw())
        self.sdfg_changed = True

    def AddTasklet(self, taskletname, statelabel):
        state = self.parse_state_label(statelabel)
        if state == None:
            self.show_error("You cannot put tasklets outside of states")
            return False
        newtasklet = dace.graph.nodes.Tasklet(taskletname)
        self.sdfg.nodes()[state].add_node(newtasklet)
        self.rendered_sdfg.set_dotcode(self.sdfg.draw())
        self.sdfg_changed = True

    def DeleteNode(self, nodeid):
        sid, nid = self.split_nodeid_in_state_and_nodeid(nodeid)
        rmnode = self.sdfg.nodes()[sid].nodes()[nid]
        self.sdfg.nodes()[sid].remove_node(rmnode)
        self.rendered_sdfg.set_dotcode(self.sdfg.draw())
        self.sdfg_changed = True

    def DeleteEdge(self, tailname, headname, edgelabel):
        sid1, nid1 = self.split_nodeid_in_state_and_nodeid(headname)
        sid2, nid2 = self.split_nodeid_in_state_and_nodeid(tailname)
        if sid1 != sid2:
            # The SDFG is an OrderedDiGraph -> max one edge between each
            # node pair
            tail = self.sdfg.nodes()[sid2]
            head = self.sdfg.nodes()[sid1]
            edges = self.sdfg.edges_between(tail, head)
            if len(edges) != 1:
                raise ValueError(
                    "There should be one edge between " + str(tailname) + \
                    " and " + str(headname) + ", found " + str(edges) + \
                    " instead")
            self.sdfg.remove_edge(edges[0])
            self.rendered_sdfg.set_dotcode(self.sdfg.draw())
            self.sdfg_changed = True
            return
        # An SDFGState is a MultiDiGraph, there could be more than one
        # edge between a node pair, thus we use the label to identify which
        # one we should delete.
        sid = int(sid1)
        srcnd = self.sdfg.nodes()[sid].nodes()[nid2]
        dstnd = self.sdfg.nodes()[sid].nodes()[nid1]
        for e in self.sdfg.nodes()[sid].edges_between(srcnd, dstnd):
            _, _, _, _, d = e
            if str(d) == edgelabel:
                self.sdfg.nodes()[sid].remove_edge(e)
                self.rendered_sdfg.set_dotcode(self.sdfg.draw())
                self.sdfg_changed = True
                return
        raise ValueError("No memlet with this label was found!")

    def DeleteState(self, statename):
        sid = self.parse_state_label(statename)
        state = self.sdfg.nodes()[sid]
        self.sdfg.remove_node(state)
        self.rendered_sdfg.set_dotcode(self.sdfg.draw())
        self.sdfg_changed = True


### End of scripting interface

    def add_array(self, x, y):
        state_label = self.rendered_sdfg.get_subgraph_by_coords(x, y)
        state = self.parse_state_label(state_label)
        if state == None:
            self.show_error("You cannot put arrays outside of states")
            return False
        self.AddArray("newarray", state_label)
        self.emit_script_cmd("AddArray(\"newarray\", \"" + state_label +\
                             "\")")
        return True

    def add_stream(self, x, y):
        streamtype = self.sdfg.add_stream("newstream", dace.types.float64, 1,
                                          0)
        newstream = dace.graph.nodes.AccessNode(streamtype)
        state_label = self.rendered_sdfg.get_subgraph_by_coords(x, y)
        state = self.parse_state_label(state_label)
        if state == None:
            self.show_error("You cannot put streams outside of states")
            return False
        self.sdfg.nodes()[state].add_node(newstream)
        self.emit_script_cmd("AddStream(\"newstream\", \"" + state_label + \
                             "\")")
        return True

    def add_memlet(self, x, y):
        elem = self.rendered_sdfg.get_element_by_coords(x, y)
        if self.first_selected_node_for_edge == None:
            if type(elem).__name__ == "Node":
                self.first_selected_node_for_edge = elem
            else:
                self.show_error("You cannot create memlets between a " + type(
                    self.first_selected_node_for_edge).__name__ + \
                    " and a " + type(elem).__name__ + \
                    " choose two nodes instead")
                self.first_selected_node_for_edge = None
            return False

        srcnodeid = self.first_selected_node_for_edge.id.decode('utf-8')
        dstnodeid = elem.id.decode('utf-8')

        sid1, nid1 = self.split_nodeid_in_state_and_nodeid(srcnodeid)
        sid2, nid2 = self.split_nodeid_in_state_and_nodeid(dstnodeid)
        if sid1 != sid2:
            self.show_error("You cannot create memlets between states.")
            return False
        else:
            self.AddMemlet("newarray", srcnodeid, dstnodeid)
            self.emit_script_cmd("AddMemlet(\"newarray\", \"" + srcnodeid + \
                                 "\", \"" + dstnodeid + "\")")
        self.first_selected_node_for_edge = None
        return True

    def add_state_trans(self, x, y):
        sg = self.rendered_sdfg.get_subgraph_by_coords(x, y)
        if sg == None:
            self.first_selected_state_for_edge = None
            self.show_error(
                "You need to select a state to add a state transition")
            return False
        if self.first_selected_state_for_edge == None:
            self.first_selected_state_for_edge = sg
            return False
        self.AddStateTransition(self.first_selected_state_for_edge, sg)
        self.emit_script_cmd("AddStateTransition(\"" + \
                             self.first_selected_state_for_edge + \
                             "\", \"" + sg + "\")")

        self.first_selected_state_for_edge = None
        return True

    def add_state_trans_by_label(self, taillabel, headlabel):
        taillabel = self.convert_dummy_to_state(taillabel)
        headlabel = self.convert_dummy_to_state(headlabel)
        self.AddStateTransition(taillabel, headlabel)
        self.emit_script_cmd("AddStateTransition(\"" + taillabel + "\", \"" + \
                             headlabel + "\")")
        return True

    def convert_dummy_to_state(self, nodeid):
        match = re.match("s(\d+)_(\d+)", nodeid)
        if match:
            return "s" + str(match.groups()[0]) + "_" + str(match.groups()[1])
        match = re.match("s(\d+)", nodeid)
        if match: return "s" + str(match.groups()[0])
        match = re.match("dummy_(\d+)", nodeid)
        if match: return "s" + str(match.groups()[0])
        else: raise ValueError(str(nodeid) + " is not a state label")

    def redirect(self, x, y, direction):
        # If the user didn't select an edge yet, store the selected edge
        # and reset in case they select a node
        if self.selected_edge_for_redir == None:
            elem = self.rendered_sdfg.get_element_by_coords(x, y)
            if (elem == None) or (type(elem).__name__ != "Edge"):
                self.selected_edge_for_redir = None
                return False
            self.selected_edge_for_redir = elem
            return False

        # At this point we have our selected edge (stored as xdot elem) and we
        # expect the user to select either an SDFG node or a state. We need to
        # check if the selection makes sense, i.e., if the user selected an
        # interstate edge, the selection now refers to the state, otherwise
        # an SDFG node.
        edge = self.selected_edge_for_redir
        self.selected_edge_for_redir = None
        elem = self.rendered_sdfg.get_element_by_coords(x, y)
        node = elem.id.decode('utf-8')
        sg = self.rendered_sdfg.get_subgraph_by_coords(x, y)
        tail = edge.src.id.decode('utf-8')
        head = edge.dst.id.decode('utf-8')
        edge_label = _get_edge_label(edge)
        sid1, nid1 = self.split_nodeid_in_state_and_nodeid(head)
        sid2, nid2 = self.split_nodeid_in_state_and_nodeid(tail)
        sid_redir, nid_redir = self.split_nodeid_in_state_and_nodeid(node)

        # We are redirecting an interstate edge, delete the old edge and add
        # the new one
        # TODO: handle properties on edge
        if sid1 != sid2:
            # tail / head could be a dummy node (dumm_X), but we (and the user)
            # expect states here (sX)
            self.delete_state_transition(tail, head)
            if direction == "head": self.add_state_trans_by_label(tail, sg)
            elif direction == "tail": self.add_state_trans_by_label(sg, head)
            else: raise ValueError("direction \"head\" or \"tail\" expected.")
            return True

        # We are redirecting a Memlet, find the memlet first, so that we can
        # attach it to the new edge

        # An SDFGState is a MultiDiGraph, there could be more than one
        # edge between a node pair, thus we use the label to identify which
        # one we should delete.
        sid = int(sid1)
        srcnd = self.sdfg.nodes()[sid].nodes()[nid2]
        dstnd = self.sdfg.nodes()[sid].nodes()[nid1]
        redir = self.sdfg.nodes()[sid_redir].nodes()[nid_redir]
        for e in self.sdfg.nodes()[sid].edges_between(srcnd, dstnd):
            u, uconn, v, vconn, d = e
            if str(d) == edge_label:
                if direction == 'head': v = redir
                elif direction == 'tail': u = redir
                else:
                    raise ValueError(
                        "direction \"head\" or \"tail\" expected.")

                self.DeleteEdge(tail, head, edge_label)
                self.sdfg.nodes()[sid].add_edge(u, uconn, v, vconn, d)
                print("Finished the redirection, deleted edge (" + str(tail) +
                      "," + str(head) + "," + str(edge_label) +
                      ") added new edge (" + str(u) + ", " + str(v) + ", " +
                      str(d) + ")")
                return True
        raise ValueError("No memlet with this label was found!")

    def add_map(self, x, y):
        state_label = self.rendered_sdfg.get_subgraph_by_coords(x, y)
        state = self.parse_state_label(state_label)
        if state == None:
            self.show_error("You cannot put maps outside of states")
            return False
        self.AddMap("newmap", state_label)
        self.emit_script_cmd("AddMap(\"newmap\", \"" + state_label + "\")")
        return True

    def add_consume(self, x, y):
        state_label = self.rendered_sdfg.get_subgraph_by_coords(x, y)
        state = self.parse_state_label(state_label)
        if state == None:
            self.show_error("You cannot put consumes outside of states")
            return False
        self.AddConsume("newconsume", state_label)
        self.emit_script_cmd("sdfg_editor.AddConsume(\"newconsume\"," + \
                             state_label + "\")")
        return True

    def add_state(self, x, y):
        num_states = len(self.sdfg.nodes())
        state_label = "s" + str(num_states)
        self.AddState(state_label)
        self.emit_script_cmd("AddState(\"" + state_label + "\")")
        return True

    def add_tasklet(self, x, y):
        state_label = self.rendered_sdfg.get_subgraph_by_coords(x, y)
        state = self.parse_state_label(state_label)
        if state == None:
            self.show_error("You cannot put tasklets outside of states")
            return False
        self.AddTasklet("newtasklet", state_label)
        self.emit_script_cmd("AddTasklet(\"newtasklet\", \"" +\
                             state_label + "\")")
        return True

    def delete_node(self, x, y):
        elem = self.rendered_sdfg.get_element_by_coords(x, y)
        if elem == None:
            return False
        nodeid = elem.id.decode('utf-8')
        self.DeleteNode(nodeid)
        self.emit_script_cmd("DeleteNode(\"" + nodeid + "\")")
        return True

    def DeleteStateTransition(self, s1_label, s2_label):
        s1 = self.parse_state_label(s1_label)
        s2 = self.parse_state_label(s2_label)
        tail = self.sdfg.nodes()[s1]
        head = self.sdfg.nodes()[s2]
        edges = self.sdfg.edges_between(tail, head)
        if len(edges) != 1:
            raise ValueError(
                "There should be one edge between s" + str(s1) + \
                " and s" + str(s2) + ", found " + str(edges) + \
                " instead")
        self.sdfg.remove_edge(edges[0])
        self.rendered_sdfg.set_dotcode(self.sdfg.draw())
        return True

    def delete_state_transition(self, s1_label, s2_label):
        # s1_label and s2_label could refer to a dummy node here, convert
        # to a proper state. We don't want to expose the user (or the rest of
        # the code) to dummy nodes.
        s1_label = self.convert_dummy_to_state(s1_label)
        s2_label = self.convert_dummy_to_state(s2_label)
        self.DeleteStateTransition(s1_label, s2_label)
        self.emit_script_cmd("DeleteStateTransition(\"" + s1_label + \
                             "\", \"" + s2_label + "\", \"\")")
        return True

    def delete_edge(self, x, y):
        elem = self.rendered_sdfg.get_element_by_coords(x, y)
        if elem == None:
            return False
        tail = elem.src.id.decode('utf-8')
        head = elem.dst.id.decode('utf-8')
        edge_label = _get_edge_label(elem)
        sid1, nid1 = self.split_nodeid_in_state_and_nodeid(head)
        sid2, nid2 = self.split_nodeid_in_state_and_nodeid(tail)
        if sid1 != sid2:
            self.delete_state_transition(tail, head)
            return True
        # An SDFGState is a MultiDiGraph, there could be more than one
        # edge between a node pair, thus we use the label to identify which
        # one we should delete.
        sid = int(sid1)
        srcnd = self.sdfg.nodes()[sid].nodes()[nid2]
        dstnd = self.sdfg.nodes()[sid].nodes()[nid1]
        for e in self.sdfg.nodes()[sid].edges_between(srcnd, dstnd):
            _, _, _, _, d = e
            if str(d) == edge_label:
                self.DeleteEdge(tail, head, edge_label)
                self.emit_script_cmd("DeleteEdge(\"" + tail + "\", \"" + head + \
                                     "\", \"" + edge_label + "\")")

                return True
        raise ValueError("No memlet with this label was found!")

    def delete_state(self, x, y):
        subgraph = self.rendered_sdfg.get_subgraph_by_coords(x, y)
        self.DeleteState(subgraph)
        self.emit_script_cmd("DeleteState(\"" + subgraph + "\")")
        return True

    def delete_element(self, x, y):
        elem = self.rendered_sdfg.get_element_by_coords(x, y)
        subgraph = self.rendered_sdfg.get_subgraph_by_coords(x, y)
        if type(elem).__name__ == "Node":
            modified = self.delete_node(x, y)
            return modified
        if type(elem).__name__ == "Edge":
            modified = self.delete_edge(x, y)
            return modified
        if subgraph is not None:
            modified = self.delete_state(x, y)
            return modified

    def OnButtonPressSDFG(self, widget, ev):
        x, y = ev.x, ev.y
        self.rendered_sdfg.clear_highlights()
        if ev.button != 1: return
        modified_sdfg = False
        if self.active_tool["tool"] == "Mouse":
            self.rendered_sdfg.handle_button_press(ev)
            elem = self.rendered_sdfg.get_element_by_coords(x, y)
            if elem is not None:
                self.rendered_sdfg.highlight_element(elem)
                self.propren.render_properties_for_element(self.sdfg, elem)
            else:
                self.propren.render_free_symbols(self.sdfg)
        if self.active_tool["tool"] == "Array":
            modified_sdfg = self.add_array(x, y)
        if self.active_tool["tool"] == "Stream":
            modified_sdfg = self.add_stream(x, y)
        if self.active_tool["tool"] == "Memlet":
            modified_sdfg = self.add_memlet(x, y)
        if self.active_tool["tool"] == "Map":
            modified_sdfg = self.add_map(x, y)
        if self.active_tool["tool"] == "Consume":
            modified_sdfg = self.add_consume(x, y)
        if self.active_tool["tool"] == "Tasklet":
            modified_sdfg = self.add_tasklet(x, y)
        if self.active_tool["tool"] == "State":
            modified_sdfg = self.add_state(x, y)
        if self.active_tool["tool"] == "Delete":
            modified_sdfg = self.delete_element(x, y)
        if self.active_tool["tool"] == "State Transition":
            modified_sdfg = self.add_state_trans(x, y)
        if self.active_tool["tool"] == "Tail Redirection":
            modified_sdfg = self.redirect(x, y, "tail")
        if self.active_tool["tool"] == "Head Redirection":
            modified_sdfg = self.redirect(x, y, "head")

        if modified_sdfg == True:
            self.set_sdfg(self.sdfg)
            self.sdfg_da.queue_draw()

        return False

    def OnButtonReleaseSDFG(self, widget, ev):
        self.rendered_sdfg.handle_button_release(ev)
        return False

    def OnMouseMoveSDFG(self, widget, ev):
        self.rendered_sdfg.handle_drag_motion(ev)
        return False

    def split_nodeid_in_state_and_nodeid(self, nodeid):
        match = re.match("s(\d+)_(\d+)", nodeid)
        if match:
            ids = match.groups()
            return int(ids[0]), int(ids[1])
        else:
            match = re.match("dummy_(\d+)", nodeid)
            if match:
                ids = match.groups()
                return int(ids[0]), None
            else:
                raise ValueError("Node ID " + nodeid + " has the wrong form")
                return None
