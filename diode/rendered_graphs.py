import gi
import sys
import subprocess

gi.require_version('Gtk', '3.0')

from xdot.ui.elements import Graph
from xdot.dot.lexer import ParseError
from xdot.dot.parser import XDotParser

from diode.rendered_graph import RenderedGraph

from gi.repository import Gdk, Gtk


class RenderedGraphs:
    """ SDFG rendering engine management and abstraction class. """

    def __init__(self, builder):
        self.builder = builder
        self.container = None
        self.renderer = None
        self.rendered_graphs = []
        self.on_click_cb = None

    def set_container(self, container):
        # Add a notebook with a single page: "Main SDFG"
        self.container = container

    def set_render_engine(self, engine):
        if engine not in ["html5", "xdot"]:
            raise ValueError("Rendering engine " + str(engine) +
                             " is not supported.")
        else:
            self.renderer = engine

    def render_sdfgs(self, name_sdfg_tuples):
        # Delete all old pages
        notebook = self.builder.get_object("sdfg_notebook")
        num_pages = notebook.get_n_pages()
        for page_index in range(0, num_pages):
            notebook.remove_page(-1)
        # Delete the old `rendered_graph`s
        self.rendered_graphs = []
        for elem in name_sdfg_tuples:
            scroll = Gtk.ScrolledWindow()
            drawingarea = Gtk.DrawingArea()
            scroll.add(drawingarea)
            label = Gtk.Label(label=elem[0])
            rendered_graph = RenderedGraph()
            rendered_graph.set_click_cb(self.on_click_cb)
            rendered_graph.set_drawing_area(drawingarea)
            rendered_graph.render_sdfg(elem[1])
            self.rendered_graphs.append(rendered_graph)
            notebook.append_page(scroll, label)
        window = self.builder.get_object("main_window")
        window.show_all()

    def clear_highlights(self):
        for rg in self.rendered_graphs:
            rg.clear_highlights()

    def set_on_click_cb(self, cb):
        self.on_click_cb = cb
        for rg in self.rendered_graphs:
            rg.set_on_click_cb(cb)

    def get_element_by_id(self, sdfg, id):
        for rg in self.rendered_graphs:
            if rg.sdfg == sdfg:
                return rg.get_element_by_id(id)

    def get_element_by_sdfg_node(self, sdfg, stateid, nodeid):
        for rg in self.rendered_graphs:
            if rg.sdfg == sdfg:
                search_nid = "s" + str(stateid) + "_" + str(nodeid)
                for n in rg.xdot_graph.nodes:
                    nid = n.id.decode('utf-8')
                    if nid == search_nid:
                        return n
                raise ValueError("Node was not found!")
            raise ValueError("SDFG was not found!")

    def highlight_element(self, sdfg, elem):
        # Find the right pane, switch to that pane and call highlight_element
        # on that graph
        for idx, rg in enumerate(self.rendered_graphs):
            if rg.sdfg == sdfg:
                self.switch_to_sdfg(sdfg)
                rg.highlight_element(elem)
                rg.center_highlights()
                return
        raise ValueError("SDFG was not found!")

    def switch_to_sdfg(self, sdfg):
        pane = None
        for idx, rg in enumerate(self.rendered_graphs):
            if rg.sdfg == sdfg:
                pane = idx
        if pane is None:
            raise ValueError("SDFG was not found!")
        notebook = self.builder.get_object("sdfg_notebook")
        notebook.set_current_page(pane)

    def currently_displayed_sdfg(self):
        notebook = self.builder.get_object("sdfg_notebook")
        idx = notebook.get_current_page()
        return self.rendered_graphs[idx].sdfg

    def set_memspeed_target(self):
        pass

    def render_performance_data(self):
        pass
