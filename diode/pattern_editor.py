import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GtkSource', '3.0')
from gi.repository import Gtk, GtkSource

from diode.rendered_graph import RenderedGraph
from diode.abstract_sdfg import AbstractSDFG
from diode.images import ImageStore
from diode.property_renderer import PropertyRenderer, _get_edge_label


class PatternEditor:
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
                "image": "unmap.png",
                "type": "node",
                "tool": "Unmap"
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
                "tool": "Stream Map"
            },
            {
                "image": "stream_unmap.png",
                "type": "node",
                "tool": "Stream Unmap"
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
        ]

        self.active_tool = None  # an element of self.buttons
        self.builder = builder
        self.main_sdfg = None
        self.first_selected_node_for_edge = None

        self.rendered_main_sdfg = RenderedGraph()
        sdfg_da = self.builder.get_object("patedmainsdfg")
        self.rendered_main_sdfg.set_drawing_area(sdfg_da)

        self.abstract_find_sdfg = AbstractSDFG()
        self.rendered_find_sdfg = RenderedGraph()
        find_da = self.builder.get_object("find_da")
        self.rendered_find_sdfg.set_drawing_area(find_da)

        self.abstract_replace_sdfg = AbstractSDFG()
        self.rendered_replace_sdfg = RenderedGraph()
        replace_da = self.builder.get_object("replace_da")
        self.rendered_replace_sdfg.set_drawing_area(replace_da)

        tbuffer = self.builder.get_object("pe_sourceview").get_buffer()
        self.init_syntax_highlighting("pe_sourceview", "python")
        self.image_store = ImageStore()

        plabel = self.builder.get_object("pe_propertylabel")
        pgrid = self.builder.get_object("pe_propertygrid")
        self.propren = PropertyRenderer(plabel, pgrid, self.OnSDFGUpdate)

        self.load_buttons()
        self.connect_signals()

    def OnSDFGUpdate(self, sdfg, nodeid, propname, newval):
        self.rendered_main_sdfg.set_dotcode(self.main_sdfg.draw())

    def connect_signals(self):
        find_da = self.builder.get_object("find_da")
        replace_da = self.builder.get_object("replace_da")
        sdfg_da = self.builder.get_object("patedmainsdfg")

        sdfg_da.connect("draw", self.OnDrawMainSDFG)
        find_da.connect("draw", self.OnDrawFindSDFG)
        replace_da.connect("draw", self.OnDrawReplaceSDFG)
        sdfg_da.connect("scroll-event", self.OnScrollMainSDFG)
        find_da.connect("scroll-event", self.OnScrollFindSDFG)
        replace_da.connect("scroll-event", self.OnScrollReplaceSDFG)
        sdfg_da.connect("button-press-event", self.OnButtonPressMainSDFG)
        sdfg_da.connect("button-release-event", self.OnButtonReleaseMainSDFG)
        sdfg_da.connect("motion-notify-event", self.OnMouseMoveMainSDFG)
        find_da.connect("button-press-event", self.OnButtonPressFindSDFG)
        replace_da.connect("button-press-event", self.OnButtonPressReplaceSDFG)

    def load_buttons(self):
        toolbar = self.builder.get_object("pated_toolbar")
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

    def set_main_sdfg(self, sdfg):
        self.main_sdfg = sdfg
        dotcode = sdfg.draw()
        self.rendered_main_sdfg.set_dotcode(dotcode)

    def OnDrawMainSDFG(self, widget, cr):
        self.rendered_main_sdfg.render(widget, cr)
        return False

    def OnDrawFindSDFG(self, widget, cr):
        self.rendered_find_sdfg.render(widget, cr)
        return False

    def OnDrawReplaceSDFG(self, widget, cr):
        self.rendered_replace_sdfg.render(widget, cr)
        return False

    def OnToggleTBButton(self, widget, button):
        self.active_tool["button"].set_active(False)
        statuslabel = self.builder.get_object("run_status_text")
        if button["type"] == "node":
            statuslabel.set_text("Click \"find\" or \"replace\" pane to " + \
                      "add a " + button["tool"] + " node.")
        elif button["type"] == "edge":
            statuslabel.set_text("In the \"find\" or \"replace\" pane, " + \
                      "click two nodes between which you want to add a " + \
                      button["tool"] + " edge.")
        elif button["type"] == "edge_redir":
            statuslabel.set_text("In the \"find\" or \"replace\" pane, " + \
                      "click an edge, followed by the new node it should " + \
                      "attach to.")
        elif button["tool"] == "Delete":
            statuslabel.set_text("Click a node or edge in the \"find\" or " + \
                      "\"replace\" pane in oder to delete it.")
        self.active_tool = button
        return True

    def OnScrollMainSDFG(self, widget, ev):
        d = self.rendered_main_sdfg.determine_scroll_direction(ev)
        self.rendered_main_sdfg.zoom(d, pos=(ev.x, ev.y))
        widget.queue_draw()
        return False

    def OnScrollFindSDFG(self, widget, ev):
        d = self.rendered_find_sdfg.determine_scroll_direction(ev)
        self.rendered_find_sdfg.zoom(d, pos=(ev.x, ev.y))
        widget.queue_draw()
        return False

    def OnScrollReplaceSDFG(self, widget, ev):
        d = self.rendered_replace_sdfg.determine_scroll_direction(ev)
        self.rendered_replace_sdfg.zoom(d, pos=(ev.x, ev.y))
        widget.queue_draw()
        return False

    def OnButtonPressMainSDFG(self, widget, ev):
        x, y = ev.x, ev.y
        elem = self.rendered_main_sdfg.get_element_by_coords(x, y)
        if ev.button == 1:
            self.rendered_main_sdfg.handle_button_press(ev)
            elem = self.rendered_main_sdfg.get_element_by_coords(x, y)
            if elem is not None:
                self.rendered_main_sdfg.highlight_element(elem)
                self.propren.render_properties_for_element(
                    self.main_sdfg, elem)

        elif ev.button == 3:
            if elem == None:
                self.rendered_main_sdfg.clear_highlights()
            else:
                self.rendered_main_sdfg.highlight_element(elem)

    def OnButtonReleaseMainSDFG(self, widget, ev):
        self.rendered_main_sdfg.handle_button_release(ev)
        return False

    def OnMouseMoveMainSDFG(self, widget, ev):
        self.rendered_main_sdfg.handle_drag_motion(ev)
        return False

    def OnRepFindNodePropsChanged(self, widget, data):
        elem_in_replace = False
        elem = self.abstract_find_sdfg.find_node(data)
        if elem == None:
            elem = self.abstract_replace_sdfg.find_node(data)
            elem_in_replace = True
        if elem == None:
            raise ValueError("Could not find node " + data)
            return
        newval = widget.get_text()
        elem.set_label(newval)
        new_dot = ""
        if elem_in_replace == False:
            new_dot = self.abstract_find_sdfg.to_dot()
            self.rendered_find_sdfg.set_dotcode(new_dot)
        else:
            new_dot = self.abstract_replace_sdfg.to_dot()
            self.rendered_replace_sdfg.set_dotcode(new_dot)

    def OnRepFindEdgePropsChanged(self, widget, data):
        elem_in_replace = False
        elem = self.abstract_find_sdfg.find_edge(data[0], data[1])
        if elem == None:
            elem = self.abstract_replace_sdfg.find_edge(data[0], data[1])
            elem_in_replace = True
        if elem == None:
            raise ValueError("Could not find node " + data)
            return
        newval = widget.get_text()
        elem.set_label(newval)
        new_dot = ""
        if elem_in_replace == False:
            new_dot = self.abstract_find_sdfg.to_dot()
            self.rendered_find_sdfg.set_dotcode(new_dot)
        else:
            new_dot = self.abstract_replace_sdfg.to_dot()
            self.rendered_replace_sdfg.set_dotcode(new_dot)

    def render_properties_for_repfind_node(self, elem, abstract_graph):
        nodeid = elem.id.decode('utf-8')
        node = abstract_graph.find_node(nodeid)
        grid = self.builder.get_object("pe_propertygrid")
        self.clear_property_list()

        rownum = 0
        label = Gtk.Label()
        label.set_label("Node Label")
        label.set_tooltip_text("set the label")
        grid.attach(label, 0, rownum, 1, 1)
        widget = Gtk.Entry()
        widget.set_text(node.get_label())
        nuid = node.get_uid()
        widget.connect("changed", self.OnRepFindNodePropsChanged, nuid)

        grid.attach(widget, 1, rownum, 1, 1)
        rownum += 1
        grid.show_all()

    def render_properties_for_repfind_edge(self, tailelem, headelem,
                                           abstract_graph):
        tail_nodeid = tailelem.id.decode('utf-8')
        tailnode = abstract_graph.find_node(tail_nodeid)
        head_nodeid = headelem.id.decode('utf-8')
        headnode = abstract_graph.find_node(head_nodeid)
        edge = abstract_graph.find_edge(tail_nodeid, head_nodeid)

        grid = self.builder.get_object("pe_propertygrid")
        self.clear_property_list()

        rownum = 0
        label = Gtk.Label()
        label.set_label("Edge Label")
        label.set_tooltip_text("set the label")
        grid.attach(label, 0, rownum, 1, 1)
        widget = Gtk.Entry()
        widget.set_text(_get_edge_label(edge))
        widget.connect("changed", self.OnRepFindEdgePropsChanged,
                       [tail_nodeid, head_nodeid])
        grid.attach(widget, 1, rownum, 1, 1)
        rownum += 1
        grid.show_all()

    def button_press_in_replace_or_find(self, widget, ev, graph):
        rendered_graph = None
        abstract_sdfg = None
        if graph == "replace":
            rendered_graph = self.rendered_replace_sdfg
            abstract_graph = self.abstract_replace_sdfg
        elif graph == "find":
            rendered_graph = self.rendered_find_sdfg
            abstract_graph = self.abstract_find_sdfg
        else:
            raise ValueError("graph must be find or replace")

        # if the active tool is the mouse, show properties of clicked elem
        if self.active_tool["tool"] == "Mouse":
            elem = rendered_graph.get_element_by_coords(ev.x, ev.y)
            rendered_graph.clear_highlights()
            rendered_graph.highlight_element(elem)
            label = self.builder.get_object("pe_propertylabel")
            self.clear_property_list()
            if type(elem).__name__ == "Node":
                label.set_text("Properties of: " + elem.id.decode('utf-8'))
                self.render_properties_for_repfind_node(elem, abstract_graph)
            elif type(elem).__name__ == "Edge":
                tailelem = elem.src
                headelem = elem.dst
                label.set_text("Properties of: " + tailelem.id.decode('utf-8') \
                               + " -> " + headelem.id.decode('utf-8'))
                self.render_properties_for_repfind_edge(
                    tailelem, headelem, abstract_graph)
            else:
                label.set_text("Properties of: (Nothing selected)")
            return False

        elif self.active_tool["type"] == "node":
            abstract_graph.add_node(self.active_tool["tool"])
            new_dot = abstract_graph.to_dot()
            rendered_graph.set_dotcode(new_dot)

        elif self.active_tool["type"] == "edge":
            elem = rendered_graph.get_element_by_coords(ev.x, ev.y)
            if elem == None:
                return
            if self.first_selected_node_for_edge == None:
                self.first_selected_node_for_edge = elem.id.decode('utf-8')
            else:
                second_selected_node_for_edge = elem.id.decode('utf-8')
                abstract_graph.add_edge(self.first_selected_node_for_edge,
                                        second_selected_node_for_edge)
                self.first_selected_node_for_edge = None
                new_dot = abstract_graph.to_dot()
                rendered_graph.set_dotcode(new_dot)

        elif self.active_tool["tool"] == "Delete":
            elem = rendered_graph.get_element_by_coords(ev.x, ev.y)
            abstract_graph.delete_node(elem.id.decode('utf-8'))
            new_dot = abstract_graph.to_dot()
            rendered_graph.set_dotcode(new_dot)

    def OnButtonPressFindSDFG(self, widget, ev):
        self.button_press_in_replace_or_find(widget, ev, "find")

    def OnButtonPressReplaceSDFG(self, widget, ev):
        self.button_press_in_replace_or_find(widget, ev, "replace")
