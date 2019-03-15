import re
import gi
import dace
gi.require_version('Gtk', '3.0')
from gi.repository import Gdk, Gtk, GtkSource

import astunparse
import xdot.ui.elements


def _get_edge_label(edge: xdot.ui.elements.Edge) -> str:
    """ Helper function to get the label of an xdot Edge from its shape. """
    labels = [
        s.t for s in edge.shapes if isinstance(s, xdot.ui.elements.TextShape)
    ]
    return "\n".join(labels)


class PropertyRenderer:
    """ Renders GUI for node, edge, state, and transformation properties. """

    def __init__(self, label, grid, on_update_cb):
        self.propertygrid = grid
        self.propertylabel = label
        self.on_update_cb = on_update_cb
        self.screen = Gdk.Screen.get_default()
        self.gtk_provider = Gtk.CssProvider()
        self.gtk_context = Gtk.StyleContext()

    def clear_properties(self):
        old = self.propertygrid.get_children()
        for w in old:
            w.destroy()

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

    def get_value_from_widget(self, widget):
        newval = ""
        if isinstance(widget, Gtk.Switch):
            newval = widget.get_active()
        elif isinstance(widget, Gtk.Entry):
            newval = widget.get_text()
        elif isinstance(widget, Gtk.ComboBoxText):
            newval = widget.get_active_text()
        elif isinstance(widget, Gtk.TextBuffer):
            start = widget.get_start_iter()
            end = widget.get_end_iter()
            newval = widget.get_text(start, end, True)
        else:
            print("Unhandled widget type \"{}\" found "
                  "while reading node properties".format(type(widget)))
        return newval

    def node_props_changed(self, widget, *data):
        # The callback of a switch looks a bit different than all other
        # objects, we handle this here
        sdfg, nodeid, prop = data[len(data)-1][0], \
                             data[len(data)-1][1], \
                             data[len(data)-1][2]
        string_val = self.get_value_from_widget(widget)
        sid, nid = self.split_nodeid_in_state_and_nodeid(nodeid)
        node = sdfg.nodes()[sid].nodes()[nid]
        dace.properties.set_property_from_string(prop, node, string_val, sdfg)
        self.on_update_cb(sdfg, "node", nodeid, prop.attr_name, string_val)

    def memlet_props_changed(self, widget, *data):
        # The callback of a switch looks a bit different than all other
        # objects, we handle this here
        (sdfg, memlet, tail, head, label,
         prop) = (data[len(data) - 1][0], data[len(data) - 1][1],
                  data[len(data) - 1][2], data[len(data) - 1][3],
                  data[len(data) - 1][4], data[len(data) - 1][5])
        string_val = self.get_value_from_widget(widget)
        try:
            dace.properties.set_property_from_string(prop, memlet, string_val,
                                                     sdfg)
            self.revert_render_property_error(widget)
        except ValueError as e:
            self.render_property_error(e, widget)
        self.on_update_cb(sdfg, "memlet", tail, head, label, prop.attr_name,
                          string_val)

    def state_props_changed(self, widget, *data):
        # The callback of a switch looks a bit different than all other
        # objects, we handle this here
        sdfg, stateid, prop = data[len(data)-1][0], \
                              data[len(data)-1][1], \
                              data[len(data)-1][2]

        newval = self.get_value_from_widget(widget)
        dace.properties.set_property_from_string(prop,
                                                 sdfg.nodes()[stateid], newval)
        self.on_update_cb(sdfg, "state", sdfg, stateid, prop.attr_name, newval)

    def interstate_props_changed(self, widget, *data):
        sdfg, tail, head, prop = data[len(data)-1][0], \
                                 data[len(data)-1][1], \
                                 data[len(data)-1][2], \
                                 data[len(data)-1][3]
        string_val = self.get_value_from_widget(widget)
        edges = sdfg.edges_between(sdfg.nodes()[tail], sdfg.nodes()[head])
        _, _, v = edges[0]

        try:
            dace.properties.set_property_from_string(prop, v, string_val, sdfg)
            self.revert_render_property_error(widget)
        except ValueError as e:
            self.render_property_error(e, widget)
            self.on_update_cb(sdfg, "statetrans", tail, head, prop.attr_name,
                              string_val)

    def render_property_error(self, exception, widget):
        if isinstance(widget, Gtk.Entry):
            widget.set_tooltip_text(str(exception))
            widget_style_context = widget.get_style_context()
            # TODO: this should probably be unique
            widget.set_name("name_entry")
            self.gtk_context.add_provider_for_screen(
                self.screen, self.gtk_provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
            data = ("#name_entry.red {background-image: "
                    "linear-gradient(0deg, #ffe8e8, #ff0000) ;}")
            self.gtk_provider.load_from_data(data.encode('utf-8'))
            widget_style_context.add_class("red")
        else:
            raise exception

    def revert_render_property_error(self, widget):
        if isinstance(widget, Gtk.Entry):
            widget_style_context = widget.get_style_context()
            widget_style_context.remove_class("red")
            widget.set_tooltip_text("")

    def pattern_props_changed(self, widget, *data):
        # The callback of a switch looks a bit different than all other
        # objects, we handle this here
        optgraph, nodeid, pattern_match, prop = data[len(data)-1][0], \
                                                data[len(data)-1][1], \
                                                data[len(data)-1][2], \
                                                data[len(data)-1][3]

        newval = self.get_value_from_widget(widget)
        dace.properties.set_property_from_string(prop, pattern_match, newval)
        self.on_update_cb(None, "pattern_match", optgraph, nodeid,
                          prop.attr_name, newval)

    def free_symbol_changed(self, widget, *data):
        sdfg, symname = data[len(data) - 1][0], data[len(data) - 1][1]
        value = self.get_value_from_widget(widget)
        for sym in sdfg.undefined_symbols(True):
            if str(sym) == symname:
                symbol = dace.symbolic.symbol(symname)
                symbol.set(value)

    def render_prop(self, prop, value, callback, callback_data, sdfg):
        widget = None
        if prop.dtype == bool and value is not None:
            widget = Gtk.Switch()
            widget.set_active(value)
            widget.connect("state-set", callback, callback_data)
        elif isinstance(prop, dace.properties.CodeProperty):
            buf = GtkSource.Buffer()
            value = prop.to_string(value)
            widget = GtkSource.View.new_with_buffer(buf)
            lang_manager = GtkSource.LanguageManager()
            language = lang_manager.get_language("python")
            buf.set_language(language)
            buf.set_text(value)
            buf.set_highlight_syntax(True)
            buf.connect("changed", callback, callback_data)
        elif prop.enum is not None:
            widget = Gtk.ComboBoxText()

            if isinstance(prop, dace.properties.DataProperty):
                enum = prop.enum(sdfg)
            else:
                enum = prop.enum

            for i, option in enumerate(enum):
                widget.append_text(prop.to_string(option))
                if option == value:
                    widget.set_active(i)
            widget.connect("changed", callback, callback_data)
        else:
            widget = Gtk.Entry()
            widget.set_text(prop.to_string(value))
            widget.connect("changed", callback, callback_data)
        return widget

    def node_props(self, sdfg, nodeid):
        sid, nid = self.split_nodeid_in_state_and_nodeid(nodeid)
        node = sdfg.nodes()[sid].nodes()[nid]
        properties = node.properties()
        for rownum, (prop, value) in enumerate(properties):
            name_label = Gtk.Label()
            name_label.set_label(prop.attr_name)
            name_label.set_tooltip_text(prop.desc)
            self.propertygrid.attach(name_label, 0, rownum, 1, 1)
            callback_data = [sdfg, nodeid, prop]
            widget = self.render_prop(prop, value, self.node_props_changed,
                                      callback_data, sdfg)
            self.propertygrid.attach(widget, 1, rownum, 1, 1)
        self.propertygrid.show_all()

    def memlet_props(self, sdfg, memlet, tail, head, label):
        properties = memlet.properties()
        for rownum, (prop, value) in enumerate(properties):
            name_label = Gtk.Label()
            name_label.set_label(prop.attr_name)
            name_label.set_tooltip_text(prop.desc)
            self.propertygrid.attach(name_label, 0, rownum, 1, 1)
            callback_data = [sdfg, memlet, tail, head, label, prop]
            #print("rendering prop for: " +str(prop)+" name: "+str(prop.attr_name)+" dtype: "+str(prop.dtype)+" value: "+str(value))
            widget = self.render_prop(prop, value, self.memlet_props_changed,
                                      callback_data, sdfg)
            self.propertygrid.attach(widget, 1, rownum, 1, 1)
        self.propertygrid.show_all()

    def state_props(self, sdfg, stateid):
        properties = sdfg.nodes()[stateid].properties()
        for rownum, (prop, value) in enumerate(properties):
            name_label = Gtk.Label()
            name_label.set_label(prop.attr_name)
            name_label.set_tooltip_text(prop.desc)
            self.propertygrid.attach(name_label, 0, rownum, 1, 1)
            callback_data = [sdfg, stateid, prop]
            widget = self.render_prop(prop, value, self.state_props_changed,
                                      callback_data, sdfg)
            self.propertygrid.attach(widget, 1, rownum, 1, 1)
        self.propertygrid.show_all()

    def interstate_props(self, sdfg, tail, head):
        edges = sdfg.edges_between(sdfg.nodes()[tail], sdfg.nodes()[head])
        _, _, v = edges[0]
        properties = v.properties()
        for rownum, (prop, value) in enumerate(properties):
            name_label = Gtk.Label()
            name_label.set_label(prop.attr_name)
            name_label.set_tooltip_text(prop.desc)
            self.propertygrid.attach(name_label, 0, rownum, 1, 1)
            callback_data = [sdfg, tail, head, prop]
            widget = self.render_prop(prop, value,
                                      self.interstate_props_changed,
                                      callback_data, sdfg)
            self.propertygrid.attach(widget, 1, rownum, 1, 1)
        self.propertygrid.show_all()

    def pattern_props(self, optgraph, nodeid, pattern_match):
        if pattern_match is None:
            return

        properties = pattern_match.properties()
        for rownum, (prop, value) in enumerate(properties):
            name_label = Gtk.Label()
            name_label.set_label(prop.attr_name)
            name_label.set_tooltip_text(prop.desc)
            self.propertygrid.attach(name_label, 0, rownum, 1, 1)
            callback_data = [optgraph, nodeid, pattern_match, prop]
            widget = self.render_prop(prop, value, self.pattern_props_changed,
                                      callback_data, None)
            self.propertygrid.attach(widget, 1, rownum, 1, 1)
        self.propertygrid.show_all()

    def is_dummy(self, nodeid):
        match = re.match("dummy_(\d+)")
        if match:
            return True
        return False

    def render_properties_for_node(self, sdfg, stateid, nodeid):
        self.propertylabel.set_text("Node Properties:")
        self.clear_properties()
        combined_nodeid = "s" + str(stateid) + "_" + str(nodeid)
        self.node_props(sdfg, combined_nodeid)

    def render_properties_for_state(self, sdfg, stateid):
        self.propertylabel.set_text("State Properties:")
        self.clear_properties()
        self.state_props(sdfg, stateid)

    def render_properties_for_element(self, sdfg, elem):
        if type(elem).__name__ == "Node":
            # if this is a dummy node, show properties for the state
            nodeid = elem.id.decode('utf-8')
            self.propertylabel.set_text("Node Properties:")
            self.clear_properties()
            self.node_props(sdfg, nodeid)

        elif type(elem).__name__ == "Edge":
            tail = elem.src.id.decode('utf-8')
            head = elem.dst.id.decode('utf-8')
            edge_label = _get_edge_label(elem)
            sid1, nid1 = self.split_nodeid_in_state_and_nodeid(head)
            sid2, nid2 = self.split_nodeid_in_state_and_nodeid(tail)
            if sid1 != sid2:
                self.clear_properties()
                self.propertylabel.set_text("State Transition Properties:")
                self.interstate_props(sdfg, sid2, sid1)
                return
            sid = int(sid1)
            srcnode = sdfg.nodes()[sid].nodes()[int(nid2)]
            dstnode = sdfg.nodes()[sid].nodes()[int(nid1)]
            mid = -1
            for i, (_, _, _, _,
                    d) in enumerate(sdfg.nodes()[sid].edges_between(
                        srcnode, dstnode)):
                if str(d) == edge_label:
                    mid = i
                    break
            if mid < 0:
                raise ValueError("No memlet with this label was found!")
            memlet = d
            self.propertylabel.set_text("Memlet Properties: " + str(memlet))
            self.clear_properties()
            self.memlet_props(sdfg, memlet, tail, head, edge_label)

        else:
            self.propertylabel.set_text("Properties: (Nothing selected)")

    def render_free_symbols(self, sdfg):
        label = self.propertylabel
        grid = self.propertygrid
        label.set_text("Symbols of the SDFG")
        self.clear_properties()
        rownum = 0
        for sym in sdfg.undefined_symbols(True):
            symname = str(sym)
            name_label = Gtk.Label()
            name_label.set_label(symname)
            name_label.set_tooltip_text("Value of the symbolic variable " +
                                        symname)
            grid.attach(name_label, 0, rownum, 1, 1)
            widget = Gtk.Entry()
            callback_data = [sdfg, symname]
            widget.connect("changed", self.free_symbol_changed, callback_data)
            grid.attach(widget, 1, rownum, 4, 1)
            rownum += 1

        data_label = Gtk.Label()
        data_label.set_text("Data of the SDFG")
        grid.attach(data_label, 0, rownum, 5, 1)
        rownum += 1

        for name, dtype in sdfg.arrays.items():

            label_name = Gtk.Label()
            label_name.set_text(str(name))
            label_name.set_tooltip_text("Name of the data element")
            grid.attach(label_name, 0, rownum, 1, 1)

            label_type = Gtk.Label()
            if isinstance(dtype, dace.data.Array):
                label_type.set_text("Array")
            elif isinstance(dtype, dace.data.Stream):
                label_type.set_text("Stream")
            elif isinstance(dtype, dace.data.Scalar):
                label_type.set_text("Scalar")
            else:
                label_type.set_text(str(type(dtype)))
            label_type.set_tooltip_text("Type of the data element")
            grid.attach(label_type, 1, rownum, 1, 1)

            label_shape = Gtk.Label()
            if dtype is not None:
                label_shape.set_text(str(dtype.shape))
            else:
                label_shape.set_text("None")
            label_shape.set_tooltip_text("Shape of the data element")
            grid.attach(label_shape, 2, rownum, 1, 1)

            button_edit = Gtk.Button(
                None, image=Gtk.Image(stock=Gtk.STOCK_EDIT))
            button_edit.set_tooltip_text("Edit this data element")
            button_edit.connect("clicked", self.render_properties_for_data,
                                [str(name), sdfg])
            grid.attach(button_edit, 3, rownum, 1, 1)

            button_delete = Gtk.Button(
                None, image=Gtk.Image(stock=Gtk.STOCK_DELETE))
            button_delete.set_tooltip_text("Delete this data element")
            button_delete.connect("clicked", self.delete_data_cb,
                                  [str(name), sdfg])
            grid.attach(button_delete, 4, rownum, 1, 1)
            rownum += 1

        adddata_label = Gtk.Label()
        adddata_label.set_text("Add Data to the SDFG")
        grid.attach(adddata_label, 0, rownum, 5, 1)
        rownum += 1

        adddata_box = Gtk.Grid()
        grid.attach(adddata_box, 0, rownum, 5, 1)
        namefield = Gtk.Entry()
        addscalar = Gtk.Button("Scalar")
        addscalar.connect("clicked", self.add_data_cb,
                          ["scalar", sdfg, namefield])
        addarray = Gtk.Button("Array")
        addarray.connect("clicked", self.add_data_cb,
                         ["array", sdfg, namefield])
        addstream = Gtk.Button("Stream")
        addstream.connect("clicked", self.add_data_cb,
                          ["stream", sdfg, namefield])
        adddata_box.attach(namefield, 0, 1, 2, 1)
        adddata_box.attach(addscalar, 2, 1, 1, 1)
        adddata_box.attach(addarray, 3, 1, 1, 1)
        adddata_box.attach(addstream, 4, 1, 1, 1)
        rownum += 1

        grid.show_all()

    def add_data_cb(self, button, cb_data):
        datatype = cb_data[0]
        sdfg = cb_data[1]
        namefield = cb_data[2]
        if datatype == "scalar":
            sdfg.add_scalar(namefield.get_text(), dace.types.int32)
        elif datatype == "array":
            sdfg.add_array(namefield.get_text(), [2, 2], dace.types.float32)
        elif datatype == "stream":
            sdfg.add_stream(namefield.get_text(), dace.types.float32, 1)
        self.render_free_symbols(sdfg)

    def show_delete_error(self, sdfg, state, element):
        dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                   Gtk.ButtonsType.OK,
                                   "Data cannot be deleted while in use!")
        dialog.format_secondary_text(
            "The data item you tried to delete is still in use and thus it "
            "cannot be deleted.")
        dialog.run()
        dialog.destroy()

    def delete_data_cb(self, button, data):
        name = data[0]
        sdfg = data[1]

        # Traverse the SDFG, find any occurance of data "name", if it exists
        # show an error and do not delete
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.graph.nodes.AccessNode):
                    if str(node.data) == name:
                        self.show_delete_error(sdfg, state, node)
                        return None
            for memlet in state.edges():
                if str(memlet.data) == name:
                    self.show_delete_error(sdfg, state, memlet)
                    return None
        sdfg.remove_data(name)
        self.render_free_symbols(sdfg)

    def render_properties_for_data(self, button, data):
        name = data[0]
        sdfg = data[1]
        self.clear_properties()
        self.propertylabel.set_text("Edit Data: " + name)
        self.data_props(sdfg, name)

    def data_props(self, sdfg, name):
        data = None
        for d in sdfg.arrays.items():
            if d[0] == name:
                data = d[1]
        if data is None:
            raise ValueError("Data item " + name + " not found in SDFG " +
                             sdfg)
        properties = data.properties()
        for rownum, (prop, value) in enumerate(properties):
            name_label = Gtk.Label()
            name_label.set_label(prop.attr_name)
            name_label.set_tooltip_text(prop.desc)
            self.propertygrid.attach(name_label, 0, rownum, 1, 1)
            callback_data = [sdfg, name, data, prop]
            widget = self.render_prop(prop, value, self.data_props_changed,
                                      callback_data, None)
            self.propertygrid.attach(widget, 1, rownum, 1, 1)
        self.propertygrid.show_all()

    def data_props_changed(self, widget, *data):
        sdfg, name, data, prop = data[len(data)-1][0], \
                                 data[len(data)-1][1], \
                                 data[len(data)-1][2], \
                                 data[len(data)-1][3]
        newval = self.get_value_from_widget(widget)
        dace.properties.set_property_from_string(prop, data, newval)

    def render_properties_for_pattern(self, optgraph, nodeid, pattern_match):
        label = self.propertylabel
        tname = optgraph.find_node(nodeid).get_label()
        label.set_text("Transformation properties for " + str(tname) + ":")
        self.clear_properties()
        self.pattern_props(optgraph, nodeid, pattern_match)
