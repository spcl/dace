#!/usr/bin/python3
""" Entry point to DIODE: The Data-centric Interactive Optimization Development
    Environment. """

import gi
import os
import re
import sys
import copy
import pickle
import argparse
import traceback
from six import StringIO
gi.require_version('Gtk', '3.0')
gi.require_version('GtkSource', '3.0')
gi.require_version('WebKit2', '4.0')
from gi.repository import Gtk, GtkSource, GObject, GLib, Gdk

from diode.rendered_graphs import RenderedGraphs

from diode.rendered_graph_html5 import RenderedGraphHTML5
from diode.optgraph.optgraph import OptimizationGraph
from diode.optgraph.DaceState import DaceState
from dace.transformation.pattern_matching import Transformation
from dace.transformation.optimizer import SDFGOptimizer
from diode.pattern_editor import PatternEditor
from diode.sdfg_editor import SDFGEditor
from diode.config_ui import DIODEConfig
from diode.performance_plot import PerformancePlot
from diode.remote_execution import Executor, AsyncExecutor
from diode.property_renderer import PropertyRenderer
from diode.images import ImageStore

import dace
import dace.properties
from dace.config import Config


class DIODE:
    """ GUI class for DIODE: The Data-centric Interactive Optimization 
        Development Environment. 

        @note: Written in Gtk+Glade, using pygobject and Webkit.
    """

    def __init__(self, headless=False):
        """ Initializes a DIODE environment.
            @param headless: If True, runs without a window. Opens a UI window
                             otherwise.
        """
        self.has_warned_about_multiple_sdfgs = False
        self.config = DIODEConfig()

        self.current_python_script = ""
        self.headless = headless
        self.filename = None

        # Initialize glade
        self.builder = Gtk.Builder()
        GObject.type_register(GtkSource.View)
        scriptdir = os.path.dirname(os.path.abspath(__file__))
        self.builder.add_from_file(os.path.join(scriptdir, "main.glade"))

        # Initialize transformation optimization graph
        optgraph_da = self.builder.get_object("optimizationsgraph")
        treeview = self.builder.get_object("optimizationtreeview")
        self.optimization_graph = OptimizationGraph(
            graph_da=optgraph_da,
            treeview_widget=treeview,
            expand_node_callback=self.OnOptgraphNodeExpand,
            hover_node_callback=self.OnOptgraphNodeHover,
            activate_node_callback=self.OnOptgraphNodeActivate)

        self.pattern_editor = PatternEditor(self.builder)
        self.sdfg_editor = SDFGEditor(self.builder)

        # Initialize rendered SDFGs
        self.rendered_sdfgs = RenderedGraphs(self.builder)
        if self.config["renderer"]["html5renderer"]:
            self.rendered_sdfgs.set_render_engine("html5")
        else:
            self.rendered_sdfgs.set_render_engine("xdot")
        self.rendered_sdfgs.set_container("sdfg_notebook")

        # Initialize performance plot
        self.perfplot = PerformancePlot(self.builder)
        self.perfplot.render()

        # Initialize DaCe program executor
        self.executor = AsyncExecutor(self.perfplot, self.headless,
                                      self.rendered_sdfgs, self)

        # Set up a property renderer, tell it where to render into and what
        # to do after an update
        proplabel = self.builder.get_object("propertylabel")
        propgrid = self.builder.get_object("propertygrid")
        self.propren = PropertyRenderer(proplabel, propgrid,
                                        self.OnSDFGPropChange)
        self.rendered_sdfgs.set_on_click_cb(self.on_sdfg_click)

        # Load pictures for buttons
        self.image_store = ImageStore()
        pixbuf = self.image_store.get_image("run.png")
        image = Gtk.Image.new_from_pixbuf(pixbuf)
        button = self.builder.get_object("RunToolbutton")
        button.set_icon_widget(image)

        dic = {
            "onDeleteMainWindow": self.OnExit,
            "onActivateQuitMenu": self.OnExit,
            "onActivateOpenMenu": self.OnActivateOpenMenu,
            "onActivateSavePythonMenu": self.OnActivateSavePythonMenu,
            "onActivateSaveAsPythonMenu": self.OnActivateSaveAsPythonMenu,
            "onActivatePreferences": self.OnActivatePreferences,
            "onLoadTrans": self.OnLoadTrans,
            "onViewHwinfo": self.OnViewHwinfo,
            "onReadPAPICounters": self.OnReadPAPICounters,
            "onReadSystemInfo": self.OnReadSystemInfo,
            "onClickRunTB": self.OnClickRunTB,
            "onStoreScript": self.OnStoreScript,
            "onLoadScript": self.OnLoadScript,
            "onLoadSDFG": self.OnLoadSDFG,
            "onSaveSDFG": self.OnStoreSDFG,
            "onSwitchPage": self.OnSwitchPage,
            "onScrollPythonPane": self.OnScrollPythonPane,
            "onScrollCodePane": self.OnScrollCodePane,
        }
        self.builder.connect_signals(dic)

        # We don't have access to the sourcebuffer from within glade,
        # thus, we connect the signals / configure it here.
        tbuffer = self.builder.get_object("sourceview").get_buffer()
        tbuffer.connect("changed", self.OnChangeTextbuffer)
        self.builder.get_object("resview").set_editable(False)
        self.init_syntax_highlighting("sourceview", "python")
        self.init_syntax_highlighting("resview", "cpp")

        if self.headless == False:
            self.load_interface_configuration()

            window = self.builder.get_object("main_window")
            window.set_title(
                "DIODE: Data-centric Integrated Optimization Development "
                "Environment")
            window.set_position(Gtk.WindowPosition.CENTER)
            window.show_all()

            if self.config["diode"]["general"]["show_transfed"] == False:
                self.remove_page_in_notebook("notebook",
                                             "Transformation Editor")

            if self.config["diode"]["general"]["show_sdfged"] == False:
                self.remove_page_in_notebook("notebook", "SDFG Editor")

            if self.config["diode"]["general"]["show_optgraph"] == False:
                self.remove_page_in_notebook("opts_notebook", "Graph")

    def remove_page_in_notebook(self, notebook_id, page_name):
        notebook = self.builder.get_object(notebook_id)
        num_hide = self.page_name2page_num(page_name, notebook_id)
        notebook.remove_page(num_hide)

    def page_num2page_name(self, page_num, notebook_id="notebook"):
        notebook = self.builder.get_object(notebook_id)
        page = notebook.get_nth_page(page_num)
        if page is None:
            raise ValueError("Page " + str(page_num) + " does not exist.")
        return notebook.get_tab_label_text(page)

    def page_name2page_num(self, page_name, notebook_id="notebook"):
        notebook = self.builder.get_object(notebook_id)
        npages = notebook.get_n_pages()
        for i in range(0, npages):
            page = notebook.get_nth_page(i)
            if page is None:
                raise ValueError("Page " + str(i) + " does not exist.")
            label = notebook.get_tab_label_text(page)
            if label == page_name:
                return i
        raise ValueError("No page with the name \"" + page_name + "\" found.")

    def switch_to_page(self, notebook, pagename):
        """ The Gtk notebook API only allows to go to the next or previous 
            page, but not to a specific one, so we implement this logic in
            this function. """
        notebook.handler_block_by_func(self.OnSwitchPage)

        target = self.page_name2page_num(pagename)
        while target < notebook.get_current_page():
            notebook.prev_page()
        while target > notebook.get_current_page():
            notebook.next_page()
        notebook.handler_unblock_by_func(self.OnSwitchPage)

    def on_sdfg_click(self, sdfg, elem):
        if elem is not None:
            self.propren.render_properties_for_element(sdfg, elem)
        else:
            self.propren.render_free_symbols(sdfg)

    def OnSwitchPage(self, notebook, page, page_num):
        pagename = self.page_num2page_name(page_num)
        self.SwitchPanes(pagename)
        self.emit_script_cmd("diode.SwitchPanes(\"" + pagename + "\")")

    def SwitchPanes(self, newpage):
        # If we switch from SDFG editor to DIODE, "import" the edited graph
        # into the optimizer, but only if there have been changes
        notebook = self.builder.get_object("notebook")
        active_page = self.page_num2page_name(notebook.get_current_page())
        if (active_page == "SDFG Editor") and (newpage == "Optimizer") and \
           (self.sdfg_editor.sdfg_modified() == True):
            new_sdfg = self.sdfg_editor.get_sdfg()
            code = "# The currently shown SDFG has been generated using the "
            code += "SDFG editor.\n"

            new_ds = DaceState(
                dace_code=code,
                fake_fname="edit",
                sdfg=new_sdfg,
                headless=self.headless)
            if new_ds.has_multiple_eligible_sdfgs:
                self.onMultipleSDFGs()

            # Set the code window without triggering recompilation
            sv = self.builder.get_object("sourceview")
            buf = sv.get_buffer()
            buf.handler_block_by_func(self.OnChangeTextbuffer)
            buf.set_text(code)
            buf.handler_unblock_by_func(self.OnChangeTextbuffer)
            current_node = self.optimization_graph.get_current()
            new_node = self.optimization_graph.add_node(
                parent=current_node, label="Manual Edit")
            new_node.set_dace_state(new_ds)
            if current_node is not None:
                self.optimization_graph.add_edge(current_node, new_node)
            self.optimization_graph.set_current(new_node)
            self.draw_sdfg_graph()
            self.update_generated_code()
            self.current_python_script += self.sdfg_editor.get_edit_script()
            self.sdfg_editor.reset_edit_script()
        # Switch the page, but do _not_ emit the corresponding signal, since
        # that would generate endless recursion.
        notebook = self.builder.get_object("notebook")
        self.switch_to_page(notebook, newpage)

    def load_interface_configuration(self):
        # Set window dimensions
        window = self.builder.get_object("main_window")
        window.resize(
            int(Config.get('diode', 'layout', 'window_width')),
            int(Config.get('diode', 'layout', 'window_height')))
        if bool(Config.get('diode', 'layout', 'window_maximized')):
            window.maximize()

        # Set pane relative sizes
        self.set_pane_relative_sizes()

    def set_pane_relative_sizes(self):
        window = self.builder.get_object("main_window")
        width, height = window.get_size()
        toppane = self.builder.get_object("TopPane")
        pypane = self.builder.get_object("TopLeftPane")
        optgraphpane = self.builder.get_object("TopRightPane")
        codepane = self.builder.get_object("BottomPane")
        perfpane = self.builder.get_object("BottomRightPane")

        # Top pane
        toppane_height = float(Config.get('diode', 'layout', 'toppane_height'))
        pypane_width = float(Config.get('diode', 'layout', 'pypane_width'))
        optgraph_width = float(Config.get('diode', 'layout', 'optpane_width'))
        toppane.set_position((toppane_height / 100.0) * height)
        pypane.set_position((pypane_width / 100.0) * width)
        optgraphpane.set_position((optgraph_width / 100.0) * width)

        # Bottom pane
        codepane_width = float(Config.get('diode', 'layout', 'codepane_width'))
        perfpane_width = float(Config.get('diode', 'layout', 'perfpane_width'))
        codepane.set_position((codepane_width / 100.0) * width)
        perfpane.set_position((perfpane_width / 100.0) * width)

    def onMultipleSDFGs(self):
        if self.has_warned_about_multiple_sdfgs:
            pass
        if self.headless:
            return  # Don't warn when GUI is not displayed.
        self.has_warned_about_multiple_sdfgs = True
        self.show_error_message(
            "Multiple SDFGs",
            "The currently loaded code contains multiple top-level SDFGs, thus it is not defined which one should be run!"
        )

    def save_interface_configuration(self):
        # Save window dimensions
        window = self.builder.get_object("main_window")
        width, height = window.get_size()

        Config.set(
            'diode', 'layout', 'window_maximized', value=window.is_maximized())
        Config.set('diode', 'layout', 'window_width', value=width)
        Config.set('diode', 'layout', 'window_height', value=height)

        # Save pane dimensions
        toppane = self.builder.get_object("TopPane")
        pypane = self.builder.get_object("TopLeftPane")
        optgraphpane = self.builder.get_object("TopRightPane")
        codepane = self.builder.get_object("BottomPane")
        perfpane = self.builder.get_object("BottomRightPane")

        Config.set(
            'diode',
            'layout',
            'toppane_height',
            value=(float(toppane.get_position()) / height * 100))
        Config.set(
            'diode',
            'layout',
            'pypane_width',
            value=(float(pypane.get_position()) / width * 100))
        Config.set(
            'diode',
            'layout',
            'codepane_width',
            value=(float(codepane.get_position()) / width * 100))
        Config.set(
            'diode',
            'layout',
            'perfpane_width',
            value=(float(perfpane.get_position()) / width * 100))
        Config.set(
            'diode',
            'layout',
            'optpane_width',
            value=(float(optgraphpane.get_position()) / width * 100))

        # Serialize
        Config.save()

    def find_single_node_in_optgraph_by_label(self, label):
        nodes = self.optimization_graph.find_nodes_by_label(label)
        if len(nodes) < 1:
            print("There is no optstate with the label " + label)
            return None
        if len(nodes) > 1:
            print("There is more than one optimization state labeled " + \
                  label + ", using the first one.")
        return nodes[0]

    def OnLoadSDFG(self, *args):
        dialog = Gtk.FileChooserDialog(
            "Please choose a file", None, Gtk.FileChooserAction.SAVE,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE,
             Gtk.ResponseType.OK))
        main_window = self.builder.get_object("main_window")
        dialog.set_transient_for(main_window)
        response = dialog.run()
        filename = None
        if response == Gtk.ResponseType.OK:
            filename = dialog.get_filename()
            filename = os.path.realpath(filename)
            self.LoadSDFG(filename)
            self.emit_script_cmd("diode.LoadSDFG(\"" + filename + "\")")
        dialog.destroy()

    def OnStoreSDFG(self, *args):
        dialog = Gtk.FileChooserDialog(
            "Please choose a file", None, Gtk.FileChooserAction.SAVE,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE,
             Gtk.ResponseType.OK))
        main_window = self.builder.get_object("main_window")
        dialog.set_transient_for(main_window)
        response = dialog.run()
        filename = None
        if response == Gtk.ResponseType.OK:
            filename = dialog.get_filename()
            filename = os.path.realpath(filename)
            self.SaveSDFG(filename)
            self.emit_script_cmd("diode.SaveSDFG(\"" + filename + "\")")
        dialog.destroy()

    def OnButtonPressSDFG(self, clicked_elem):
        sdfg = self.optimization_graph.get_current().get_dace_state().get_sdfg(
        )

        # Now try to find the element in the SDFG that was clicked
        if len(clicked_elem) == 0:
            return
        if clicked_elem[0]['type'] == "SDFGState":
            sid = int(clicked_elem[0]['id'])
            if len(clicked_elem) > 1:
                if clicked_elem[1]['type'] != "Memlet":
                    nid = int(clicked_elem[1]['id'])
                    self.propren.render_properties_for_node(sdfg, sid, nid)
            else:
                self.propren.render_properties_for_state(sdfg, sid)

    def OnSDFGPropChange(self, sdfg, elemtype, *args):
        """ This is the callback for the PropertyRenderer in the Optimizer.
            It gets called after all changes have been applied to the SDFG.
            We need to make sure that the displayed SDFG is refreshed,
            and the action that was performed is logged. """

        # The property renderer is also used to render/change transformation
        # properties
        if elemtype == "pattern_match":
            optgraph, nodeid, propname, newval = args
            tname = optgraph.find_node(nodeid).get_label()
            self.emit_script_cmd("diode.ChangePatternProperties(\"" + \
                str(tname) + "\", \"" + \
                str(propname) +  "\", \"" + \
                str(newval) + "\")")

            # If the node is not expanded, we don't need to do anything
            # but if it is, we need to delete the subtree below and expand
            # again
            node = self.optimization_graph.find_node(nodeid)

            optnode = self.optimization_graph.find_node(nodeid)
            if optnode.is_expanded():
                # Create a new subtree using the modified pattern match
                self.optimization_graph.clear_subtree(optnode)
                self.optimization_graph.expand_node(optnode)
                self.optimization_graph.set_current(optnode)
                self.draw_sdfg_graph()
                self.update_generated_code()
            return

        if elemtype == "node":
            nodeid, propname, newval = args
            self.emit_script_cmd("diode.ChangeSDFGNodeProperties(\"" + \
                                 str(nodeid) +   "\", \"" + \
                                 str(propname) + "\", \"" + \
                                 str(newval) + "\")")
        elif elemtype == "memlet":
            tail, head, label, propname, newval = args
            self.emit_script_cmd("diode.ChangeSDFGMemletProperties(\"" + \
                                 str(tail) +     "\", \"" + \
                                 str(head) +     "\", \"" + \
                                 str(label) +    "\", \"" + \
                                 str(propname) + "\", \"" + \
                                 str(newval) + "\")")
        dace_state = self.optimization_graph.get_current().get_dace_state()
        dace_state.set_sdfg(sdfg)
        self.draw_sdfg_graph()
        self.update_generated_code()


#
#  DIODE FRONTEND FUNCTIONS
#

    def SaveSDFG(self, filename):
        sdfg = self.optimization_graph.get_current().get_dace_state().get_sdfg(
        )
        with open(filename, 'wb') as f:
            f.write((pickle.dumps(sdfg)))

    def LoadSDFG(self, filename):
        with open(filename, 'rb') as f:
            sdfg = pickle.loads(f.read())
            # Clear all DaCe states
            self.optimization_graph.clear()
            # Create a new DaCe state, name it after the file we loaded from
            filename = os.path.relpath(filename)
            nn = self.optimization_graph.add_node(label=filename)
            if not hasattr(sdfg, 'sourcecode'):
                sdfg.sourcecode = 'N/A'
            ds = DaceState(
                fake_fname="deserialized",
                sdfg=sdfg,
                dace_code=None,
                source_code=sdfg.sourcecode,
                headless=self.headless)
            if ds.has_multiple_eligible_sdfgs:
                self.onMultipleSDFGs()

            nn.set_dace_state(ds)
            self.optimization_graph.set_current(nn)
            self.optimization_graph.expand_node(nn)
            self.draw_sdfg_graph()
            self.update_generated_code()
            self.update_source_code()
            self.propren.render_free_symbols(sdfg)

    def update_source_code(self):
        dace_state = self.optimization_graph.get_current().get_dace_state()
        notebook = self.builder.get_object('sourceview_notebook')
        num_pages = notebook.get_n_pages()
        for page_index in range(0, num_pages):
            notebook.remove_page(-1)

        if dace_state.dace_code is not None:
            sourceview = GtkSource.View()
            sourceview.set_editable(False)
            tbuffer = sourceview.get_buffer()
            lang_manager = GtkSource.LanguageManager()
            language = lang_manager.get_language("python")
            tbuffer.set_language(language)
            tbuffer.set_highlight_syntax(True)
            code = str(dace_state.dace_code)

            tbuffer.set_text(code)
            scroll = Gtk.ScrolledWindow()
            scroll.add(sourceview)
            label = Gtk.Label(label="DaCe Code")
            notebook.append_page(scroll, label)

        if dace_state.source_code is not None:
            sourceview = GtkSource.View()
            sourceview.set_editable(False)
            tbuffer = sourceview.get_buffer()
            lang_manager = GtkSource.LanguageManager()
            language = lang_manager.get_language("python")
            tbuffer.set_language(language)
            tbuffer.set_highlight_syntax(True)
            code = str(dace_state.source_code)

            tbuffer.set_text(code)
            scroll = Gtk.ScrolledWindow()
            scroll.add(sourceview)
            label = Gtk.Label(label="Source Code")
            notebook.append_page(scroll, label)

        notebook.show_all()

    def OpenPythonFile(self, filename):
        self.filename = filename
        with open(filename, 'r') as f:
            self.open_file(f)

    def SetCurrent(self, optstate):
        node = self.find_single_node_in_optgraph_by_label(optstate)
        current = self.optimization_graph.get_current()
        self.optimization_graph.set_explored(current)
        self.optimization_graph.set_current(node)
        self.draw_sdfg_graph()
        self.update_generated_code()
        self.propren.render_free_symbols(current.get_dace_state().get_sdfg())

    def ExpandNode(self, optstate):
        # Make sure the "optstate" node is compiled and all its children are
        # displayed, but the children do not need to have a DaceState.
        node = self.find_single_node_in_optgraph_by_label(optstate)
        if node == None:
            raise ValueError("Node " + optstate + " not found")
        self.optimization_graph.set_current(node)
        self.optimization_graph.expand_node(node)
        self.draw_sdfg_graph()
        self.update_generated_code()
        self.optimization_graph.OnChange()

    def LoadTransformation(self, filename):
        old_patterns = Transformation.patterns()
        Transformation.register_pattern_file(filename)
        new_patterns = Transformation.patterns()
        if len(new_patterns) == len(old_patterns):
            print("The pattern " + filename + " didn't load!")
            return
        # Find out the name of the newly loaded transformation (a file should
        # not contain more than one transformation)
        new_names = set(p.__name__ for p in new_patterns)
        old_names = set(p.__name__ for p in old_patterns)
        new_pattern = new_names.difference(old_names).pop()
        self.update_patterns_in_optgraph(new_pattern)
        self.optimization_graph.OnChange()

    def ClearHighlights(self):
        self.rendered_sdfgs.clear_highlights()

    def ChangePreferences(self, option, newval):
        cpath = tuple(option.split(':'))
        try:
            Config.get(*cpath)
        except KeyError:
            raise KeyError("Option " + option + " does not exist!")

        Config.set(*cpath, value=newval)

    def HighlightSDFGElement(self, elem):
        pass

    def ChangeSDFGProperties(self, elem, prop, newval):
        curr = self.optimization_graph.get_current().get_dace_state()
        sdfg = curr.get_sdfg()
        sid, nid = self.split_nodeid_in_state_and_nodeid(elem)
        node = sdfg.find_node(sid, nid)
        dace.properties.set_property_from_string(prop, node, newval, sdfg)
        curr.set_sdfg(sdfg)
        self.draw_sdfg_graph()
        self.update_generated_code()

    def ChangeSDFGMemletProperties(self, elem_head, elem_tail, mid, prop,
                                   newval):
        curr = self.optimization_graph.get_current().get_dace_state()
        sdfg = curr.get_sdfg()
        sid1, nid1 = self.split_nodeid_in_state_and_nodeid(elem_head)
        sid2, nid2 = self.split_nodeid_in_state_and_nodeid(elem_tail)

        sid = int(sid1)
        sdfg_state = sdfg.nodes()[sid]

        nid1 = sdfg_state.nodes()[int(nid1)]
        nid2 = sdfg_state.nodes()[int(nid2)]
        memlet = sdfg_state.edges_between(nid2, nid1)[mid].data
        memlet.set_property(prop, newval)
        dace_state = self.optimization_graph.get_current().get_dace_state()
        dace_state.compile()
        self.draw_sdfg_graph()
        self.update_generated_code()

    def Run(self, fail_on_nonzero=None):
        if self.optimization_graph.get_current() == None:
            return False
        if fail_on_nonzero is None:
            if self.headless:
                fail_on_nonzero = True
            else:
                fail_on_nonzero = False

        dace_state = self.optimization_graph.get_current().get_dace_state()
        res = self.executor.run_async(dace_state, fail_on_nonzero)
        return res

    def ChangePatternProperties(self, nodelabel, propname, newval):
        node = self.find_single_node_in_optgraph_by_label(nodelabel)
        pattern_match = node.get_pattern_match()
        dace.properties.set_property_from_string(propname, pattern_match,
                                                 newval, None)
        node.apply_pattern_match()
        node.get_dace_state().set_is_compiled(False)
        self.optimization_graph.clear_subtree(node)
        self.optimization_graph.expand_node(node)
        self.draw_sdfg_graph()
        self.update_generated_code()
        self.optimization_graph.OnChange()

    def OnMoveHandle(self, widget, position, widget_name):
        widget_pos = str(widget.get_position())
        window = self.builder.get_object("main_window")
        win_size_x = str(window.get_allocation().width)
        win_size_y = str(window.get_allocation().height)
        print("Handle moved: " + widget_name   + \
              " new position = " + widget_pos  + \
              " window size x = " + win_size_x + \
              " window size y = " + win_size_y)

    def OnLoadTrans(self, *args):
        dialog = Gtk.FileChooserDialog(
            "Please choose a file", None, Gtk.FileChooserAction.OPEN,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN,
             Gtk.ResponseType.OK))
        main_window = self.builder.get_object("main_window")
        dialog.set_transient_for(main_window)
        response = dialog.run()
        filename = None
        if response == Gtk.ResponseType.OK:
            filename = dialog.get_filename()
            filename = os.path.realpath(filename)
            self.LoadTransformation(filename)
            self.emit_script_cmd("diode.LoadTransformation(\"" + filename + \
                                 "\")")
        dialog.destroy()

    def OnOptgraphNodeHover(self, nodeid, pattern_match):
        self.rendered_sdfgs.clear_highlights()
        if pattern_match is not None:
            sid = pattern_match.state_id
            nodes = list(pattern_match.subgraph.values())
            for n in nodes:
                nodeid = "s" + str(sid) + "_" + str(n)
                elem = self.rendered_sdfgs.get_element_by_id(
                    self.rendered_sdfgs.currently_displayed_sdfg(), nodeid)
                if elem is not None:
                    self.rendered_sdfgs.highlight_element(
                        self.rendered_sdfgs.currently_displayed_sdfg(), elem)

    def OnOptgraphNodeExpand(self, nodeid, pattern_match):
        # Double click, apply the transformation
        self.rendered_sdfgs.clear_highlights()
        self.propren.clear_properties()
        self.propren.render_properties_for_pattern(self.optimization_graph,
                                                   nodeid, pattern_match)
        self.draw_sdfg_graph()
        self.update_generated_code()
        self.emit_script_cmd("diode.ExpandNode(\"" + nodeid + "\")")

    def OnOptgraphNodeActivate(self, nodeid, pattern_match):
        # Single click on a node, render properties and show where
        # the transformation applies
        self.rendered_sdfgs.clear_highlights()
        self.propren.clear_properties()
        self.update_generated_code()
        self.propren.render_properties_for_pattern(self.optimization_graph,
                                                   nodeid, pattern_match)
        self.emit_script_cmd("diode.ActivateNode(\"" + nodeid + "\")")

    def update_patterns_in_optgraph(self, new_pattern):
        # Re-evaluate all expanded nodes in the optimization graph to
        # see if new edges need to be added.
        nodes = self.optimization_graph.get_nodes()
        for node in nodes:
            if node.get_expanded() == False: continue
            ds = node.get_dace_state()
            sdfg = ds.get_sdfg()
            opt = SDFGOptimizer(sdfg)
            patterns = opt.get_pattern_matches()
            for p in patterns:
                if str(p.pattern) == str(new_pattern):
                    # create a new state / node for the new pattern
                    nds = copy.deepcopy(node.get_dace_state())
                    p.apply_pattern(nds.get_sdfg())
                    optgraph = self.optimization_graph
                    nn = optgraph.add_node(
                        parent=node.get_parent(), label=new_pattern, pattern=p)
                    nn.set_dace_state(nds)
                    optgraph.set_unexplored(nn)
                    optgraph.add_edge(tail=node, head=nn, label="", pattern=p)

    def emit_script_cmd(self, cmd):
        self.current_python_script += cmd + "\n"

    def optimize_optscript(self):
        # Some actions stored in the optscript are redundant, i.e., if pattern
        # properties are changed ("foo" -> "fo" -> "f" by deleting characters)
        # This function removes such redundancies by simple pattern matching
        nreps = 1
        while nreps != 0:
            regex = r'^diode\.(ChangePatternProperties)\((.*?),\s*(.*?),\s*(.*?)\)\s*^diode\.\1\(\2,\s*\3,\s*(.*?)\)\s*'
            p = re.compile(regex, re.MULTILINE)
            result, nreps = p.subn(
                "diode.ChangePatternProperties(\\2, \\3, \\5)\n",
                self.current_python_script)
            self.current_python_script = result

    def OnStoreScript(self, *args):
        # Open a dialog and save to a file
        dialog = Gtk.FileChooserDialog(
            "Please choose a file", None, Gtk.FileChooserAction.SAVE,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE,
             Gtk.ResponseType.OK))
        main_window = self.builder.get_object("main_window")
        dialog.set_transient_for(main_window)
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            with open(dialog.get_filename(), 'w') as f:
                self.optimize_optscript()
                f.write(self.current_python_script)
        dialog.destroy()

    def OnLoadScript(self, *args):
        dialog = Gtk.FileChooserDialog(
            "Please choose a file", None, Gtk.FileChooserAction.SAVE,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE,
             Gtk.ResponseType.OK))
        main_window = self.builder.get_object("main_window")
        dialog.set_transient_for(main_window)
        response = dialog.run()
        contents = ""
        if response == Gtk.ResponseType.OK:
            self.load_optscript(dialog.get_filename())
        dialog.destroy()

    def load_optscript(self, filename):
        with open(filename, 'r') as f:
            contents = f.read()
            # We are running from within DIODE, execute everything
            # in the current context, i.e., diode = self.
            self.current_python_script += contents
            contents = "diode = self\n" + \
                       "sdfg_edit = self.sdfg_editor\n" + \
                       contents
            exec(contents)

    def OnViewHwinfo(self, *args):
        print("show hwinfo graph")

    def OnReadPAPICounters(self, *args):
        from dace.codegen.instrumentation.perfsettings import PerfUtils
        nonderiv, deriv, num_hw_ctrs = PerfUtils.read_available_perfcounters()

        dialog = Gtk.MessageDialog(
            None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Available counters on %s" % Config.get(
                "execution", "general", "host"))
        dialog.format_secondary_text(
            "Number of hardware counters available: %d\nNon-Derived: %s\nDerived: %s\n"
            % (num_hw_ctrs, str(nonderiv), str(deriv)))
        main_window = self.builder.get_object("main_window")
        dialog.set_transient_for(main_window)
        dialog.run()
        dialog.destroy()

    def OnReadSystemInfo(self, *args):
        from dace.codegen.instrumentation.perfsettings import PerfUtils, PerfPAPIInfoStatic

        bandwidth = PerfUtils.gather_remote_metrics()

        dialog = Gtk.MessageDialog(
            None, 0, Gtk.MessageType.INFO, Gtk.ButtonsType.OK,
            "System info on %s" % Config.get("execution", "general", "host"))
        dialog.format_secondary_text(
            "Memory bandwidth: %s B/c" % str(bandwidth))
        main_window = self.builder.get_object("main_window")
        dialog.set_transient_for(main_window)
        dialog.run()

        PerfPAPIInfoStatic.info.memspeed = float(bandwidth)

        dialog.destroy()

    def OnClickRunTB(self, widget):
        self.emit_script_cmd("diode.Run()")
        self.Run()

    def OnActivatePreferences(self, widget, *args):
        self.config.render_config_dialog()
        return True

    def OnActivateSaveAsPythonMenu(self, widget, *args):
        dialog = Gtk.FileChooserDialog(
            "Please choose a file", None, Gtk.FileChooserAction.SAVE,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE,
             Gtk.ResponseType.OK))
        main_window = self.builder.get_object("main_window")
        dialog.set_transient_for(main_window)
        response = dialog.run()
        filename = None
        if response == Gtk.ResponseType.OK:
            filename = dialog.get_filename()
            filename = os.path.realpath(filename)
            self.SavePythonCode(filename)
            self.filename = filename
            self.emit_script_cmd("diode.SavePythonCode(\"" + filename + "\")")
        dialog.destroy()

    def OnActivateSavePythonMenu(self, widget, *args):
        if self.filename is None:
            raise ValueError(
                "Asked to save code but no filename selected, maybe use Save As?"
            )
        self.SavePythonCode(self.filename)
        self.emit_script_cmd("diode.SavePythonCode(\"" + self.filename + "\")")

    def SavePythonCode(self, filename):
        statuslabel = self.builder.get_object("run_status_text")
        statusprogress = self.builder.get_object("run_status")
        statuslabel.set_text("Saving DaCe program to " + str(filename))
        tbuffer = self.builder.get_object("sourceview").get_buffer()
        start = tbuffer.get_start_iter()
        end = tbuffer.get_end_iter()
        dace_code = tbuffer.get_text(start, end, True)
        with open(filename, "w") as f:
            f.write(dace_code)
        statuslabel.set_text("Saving DaCe program to " + str(filename) +
                             " ... done")

    def init_syntax_highlighting(self, widgetname, language):
        tbuffer = self.builder.get_object(widgetname).get_buffer()
        lang_manager = GtkSource.LanguageManager()
        language = lang_manager.get_language(language)
        tbuffer.set_language(language)
        tbuffer.set_highlight_syntax(True)

    def open_file(self, f):
        self.filename = f.name
        tbuffer = self.builder.get_object("sourceview").get_buffer()
        buf = f.read()
        f.close()
        tbuffer.set_text(buf)

    def OnActivateOpenMenu(self, *args):
        dialog = Gtk.FileChooserDialog(
            "Please choose a file", None, Gtk.FileChooserAction.OPEN,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN,
             Gtk.ResponseType.OK))
        main_window = self.builder.get_object("main_window")
        dialog.set_transient_for(main_window)
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            with open(dialog.get_filename(), "r") as f:
                self.open_file(f)
                self.emit_script_cmd("diode.OpenPythonFile(\"" + \
                    os.path.realpath(f.name) + "\")")
        dialog.destroy()

    def OnChangeTextbuffer(self, *args):
        # Take the code and execute it
        # Note: When users change the buffer this is sometimes called without
        #       any text, thus we must handle this case gracefully.
        statuslabel = self.builder.get_object("run_status_text")
        statusprogress = self.builder.get_object("run_status")
        statuslabel.set_text("Compiling DaCe program")
        statusprogress.pulse()

        tbuffer = self.builder.get_object("sourceview").get_buffer()
        start = tbuffer.get_start_iter()
        end = tbuffer.get_end_iter()
        dace_code = tbuffer.get_text(start, end, True)
        try:
            dace_state = DaceState(
                dace_code, self.filename, headless=self.headless)

            if dace_state.has_multiple_eligible_sdfgs:
                self.onMultipleSDFGs()
        except:
            exstr = StringIO()
            traceback.print_exc(file=exstr)

            if self.headless == True:
                print("Compilation failed:\n\n" + exstr.getvalue())
                exit(-1)
            else:
                print("Compilation failed")

            self.update_generated_code("Compilation failed:\n\n" + \
                                       exstr.getvalue())
            statuslabel.set_text("Compiling DaCe program ... failed")
            return

        self.optimization_graph.clear()
        n = self.optimization_graph.add_node(label="Unoptimized")
        n.set_dace_state(dace_state)
        self.optimization_graph.set_current(n)
        self.optimization_graph.expand_node(n)
        self.update_generated_code()
        statuslabel.set_text("Compiling DaCe program ... completed")

        self.draw_sdfg_graph()
        self.propren.render_free_symbols(dace_state.get_sdfg())

        # Now that we rendered the SDFG, we can show any errors that might
        # have occured while generating it
        for e in dace_state.errors:
            self.show_validation_error(e)

    def highlight_node(self, sdfg, stateid, nodeid):
        self.rendered_sdfgs.clear_highlights()
        elem = self.rendered_sdfgs.get_element_by_sdfg_node(
            sdfg, stateid, nodeid)
        self.rendered_sdfgs.highlight_element(sdfg, elem)

    def highlight_source_by_debuginfo(self, di):
        sourceview = self.builder.get_object("sourceview")
        iterator = sourceview.get_buffer().get_iter_at_line(di.start_line)
        if sourceview.scroll_to_iter(iterator, 0, False, 0.5, 0.5) == False:
            print("scrolling failed!!")

    def show_validation_error(self, err):
        # Take any of the DaCe defined exceptions and display in a useful way,
        # i.e., show the message, highlight the corresponding node/memlet,
        # LOC etc.
        if isinstance(err, dace.sdfg.InvalidSDFGError):
            self.show_error_message("Invalid SDFG", err.message)
        if isinstance(err, dace.sdfg.InvalidSDFGInterstateEdgeError):
            self.show_error_message("Invalid Interstate Edge", err.message)
        if isinstance(err, dace.sdfg.InvalidSDFGEdgeError):
            self.show_error_message("Invalid SDFG Edge", err.message)
        if isinstance(err, dace.sdfg.InvalidSDFGNodeError):
            self.show_error_message("Invalid SDFG Node", err.message)
            if (err.sdfg is not None) and (err.state_id is not None) and (
                    err.node_id is not None):
                self.highlight_node(err.sdfg, err.state_id, err.node_id)
                state = err.sdfg.nodes()[err.state_id]
                node = state.nodes()[err.node_id]
                if hasattr(node, 'debuginfo') and (node.debuginfo is not None):
                    self.highlight_source_by_debuginfo(node.debuginfo)
                if hasattr(node,
                           'debuginfo2') and (node.debuginfo2 is not None):
                    self.highlight_source_by_debuginfo(node.debuginfo2)

    def show_error_message(self, caption, message):
        main_window = self.builder.get_object("main_window")
        dialog = Gtk.MessageDialog(main_window, 0, Gtk.MessageType.WARNING,
                                   Gtk.ButtonsType.OK, caption)
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def update_generated_code(self, contents=None):
        try:
            dace_state = self.optimization_graph.get_current().get_dace_state()
            if contents is None:
                program_code = dace_state.get_generated_code()
            else:
                program_code = contents
        except:
            program_code = contents

        notebook = self.builder.get_object('resview_notebook')
        num_pages = notebook.get_n_pages()
        for page_index in range(0, num_pages):
            notebook.remove_page(-1)

        # A list of CodeObject codes
        if isinstance(program_code, list):
            if len(program_code) > 1:
                notebook.set_show_tabs(True)
            else:
                notebook.set_show_tabs(False)
            for codeobj in program_code:
                sourceview = GtkSource.View()
                sourceview.set_editable(False)
                tbuffer = sourceview.get_buffer()
                lang_manager = GtkSource.LanguageManager()
                language = codeobj.language
                if codeobj.language == "cu":
                    language = "cpp"
                language = lang_manager.get_language(language)
                tbuffer.set_language(language)
                tbuffer.set_highlight_syntax(True)
                code = str(codeobj.code)
                code = re.sub("\s*////__DACE.*", "", code, 0, re.MULTILINE)
                tbuffer.set_text(code)
                scroll = Gtk.ScrolledWindow()
                scroll.add(sourceview)
                label = Gtk.Label(label=str(codeobj.title))
                notebook.append_page(scroll, label)
        else:
            # Add a single page that shows contents.
            # Probably we are showing an error, so don't highlight syntax.
            notebook.set_show_tabs(False)
            sourceview = GtkSource.View()
            sourceview.set_editable(False)
            label = Gtk.Label(label="Generated Code")
            scroll = Gtk.ScrolledWindow()
            scroll.add(sourceview)
            notebook.append_page(scroll, label)
            tbuffer = sourceview.get_buffer()
            tbuffer.set_text(program_code)
            lang_manager = GtkSource.LanguageManager()
            language = lang_manager.get_language("cpp")
            tbuffer.set_language(language)
            tbuffer.set_highlight_syntax(True)
        notebook.show_all()

    def draw_sdfg_graph(self):
        current_state = self.optimization_graph.get_current().get_dace_state()
        sdfgs = current_state.get_sdfgs()
        self.rendered_sdfgs.render_sdfgs(sdfgs)

    def OnNodePropertiesChangedSwitch(self, widget, value, data):
        # We need this function only because the callback for a GtkSwitch
        # state-change is different from most other widgets: it also sends
        # the new value (which we can obtain from the widget anyway)
        self.OnNodePropertiesChanged(widget, data)

    def OnPatternPropertiesChangedSwitch(self, widget, value, data):
        # We need this function only because the callback for a GtkSwitch
        # state-change is different from most other widgets: it also sends
        # the new value (which we can obtain from the widget anyway)
        self.OnPatternPropertiesChanged(widget, data)

    def split_nodeid_in_state_and_nodeid(self, nodeid):
        match = re.match("s(\d+)_(\d+)", nodeid)
        if match:
            sid, nid = match.groups()
            return int(sid), int(nid)
        else:
            raise ValueError("Node ID " + nodeid + " has the wrong form")
            return None

    def OnScrollPythonPane(self, widget, ev):
        if ev.get_state() & Gdk.ModifierType.CONTROL_MASK:
            d = determine_scroll_direction(ev)
            code_pane = self.builder.get_object("sourceview")
            return True  # Cancel scroll event (no motion)
        return False

    def OnScrollCodePane(self, widget, ev):
        if ev.get_state() & Gdk.ModifierType.CONTROL_MASK:
            d = determine_scroll_direction(ev)
            code_pane = self.builder.get_object("resview")
            return True  # Cancel scroll event (no motion)

        return False

    def OnExit(self, *args):
        self.save_interface_configuration()
        exit(0)


def run_main():
    parser = argparse.ArgumentParser(
        description=
        "DIODE: Data-centric Integrated Optimization Development Environment")
    parser.add_argument(
        "--file",
        metavar='file',
        type=argparse.FileType('r'),
        help="Load the specifed Python code")
    parser.add_argument(
        "--sdfg",
        metavar='sdfg',
        type=argparse.FileType('r'),
        help="Load the specifed SDFG")
    parser.add_argument(
        "--optscript",
        metavar="optscript",
        type=argparse.FileType('r'),
        help="Run the specified Optimization Script")
    parser.add_argument(
        "--stop-after-run",
        metavar="N",
        type=int,
        help="Terminate after the Nth call to diode.Run()")
    parser.add_argument(
        "--headless", action="store_true", help="Run without showing the GUI")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of using ssh")
    args = parser.parse_args()
    diode = None

    if args.headless:
        diode = DIODE(headless=True)
    else:
        diode = DIODE(headless=False)

    if args.local:
        Config.set("execution", "general", "repetitions", value=1)
        Config.set("execution", "general", "host", value=" ")
        Config.set(
            "execution",
            "general",
            "copycmd_l2r",
            value='cp ${srcfile} ${dstfile}')
        Config.set(
            "execution",
            "general",
            "copycmd_r2l",
            value='cp ${srcfile} ${dstfile}')
        Config.set("execution", "general", "execcmd", value='${command}')

    if args.file:
        diode.open_file(args.file)
        diode.emit_script_cmd("diode.OpenPythonFile(\"" + \
              os.path.realpath(args.file.name) + "\")")

    if args.sdfg:
        diode.LoadSDFG(args.sdfg.name)
        diode.emit_script_cmd("diode.LoadSDFG(\"" + \
              os.path.realpath(args.sdfg.name) + "\")")

    if args.optscript:
        diode.load_optscript(args.optscript.name)

    if args.headless == False:
        try:
            GLib.MainLoop().run()
        except KeyboardInterrupt:
            sys.exit()


if __name__ == "__main__":
    run_main()
