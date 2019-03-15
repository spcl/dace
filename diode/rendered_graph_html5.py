import sys
import os
import subprocess
import asyncio
import time
import json
import websockets
import threading

import gi
gi.require_version('WebKit2', '4.0')
from gi.repository import Gdk, WebKit2, Gtk

from gi.repository import GObject


class RenderedGraphHTML5:
    """ HTML5-based SDFG renderer. """

    def __init__(self, on_click):
        self.drawing_area = None
        self.on_click_cb = on_click
        self.websocket = None
        self.render_queue = []
        self.restore_render_queue = []  # Render queue copy
        self.command_queue = [
        ]  # Similar to render_queue, but for everything except SDFGs.
        self.restore_command_queue = []  # Command queue copy
        threading.Thread(target=self.run_asyncio, daemon=True).start()

    # The following functions highlight nodes
    def get_element_by_id(self, id):
        return id

    def highlight_element(self, elem):
        state, node = elem.split('_')
        state = state[1:]  # strip the 's' from the state number.
        self.command_queue.append(
            '{ "type": "highlight-element", "node-id": %d, "sdfg-id": %d }' %
            (int(node), int(state)))
        return

    def clear_highlights(self):
        self.command_queue.append('{ "type": "clear-highlights" }')
        return

    # End highlight functions

    def run_asyncio(self):
        try:
            loop = asyncio.SelectorEventLoop()
            asyncio.set_event_loop(loop)
            asyncio.get_event_loop().run_until_complete(
                websockets.serve(self.ConnHandler, 'localhost', 8023))
            asyncio.get_event_loop().run_forever()
        except Exception as e:
            pass

    async def ConnHandler(self, websocket, path):
        self.websocket = websocket
        print("Client connected!")
        print("render queue is " + str(len(self.render_queue)) +
              " elements long")
        for sdfg in self.render_queue:
            json = sdfg.toJSON()
            await websocket.send(json)

        self.render_queue.clear()

        while (True):
            try:
                msg = await websocket.recv()
                self.message_handler(msg)

                for sdfg in self.render_queue:
                    json = sdfg.toJSON()
                    await websocket.send(json)
                self.render_queue.clear()
                for cmd in self.command_queue:
                    # The difference in the command queue: All data must
                    # already be in JSON format
                    await websocket.send(cmd)
                self.command_queue.clear()
            except websockets.ConnectionClosed:
                # If the connection was closed, probably a refresh was
                # requested. This also means that we have to re-queue
                print("Restoring render queue after abort")
                self.render_queue = self.restore_render_queue.copy()
                self.command_queue = self.restore_command_queue.copy()
                break

    def message_handler(self, msg):
        m = json.loads(msg)
        if m["msg_type"] == "info":
            pass
        elif m["msg_type"] == "click":

            def mainthread_click_cb():
                self.on_click_cb(m["clicked_elements"])
                return False

            GObject.idle_add(mainthread_click_cb)

        elif m["msg_type"] == "heartbeat":
            pass  # Safe to ignore
        else:
            print("Unknown/Unhandled message from renderer: " +
                  str(m["msg_type"]))

    def render_sdfg(self, sdfg):
        self.render_queue.append(sdfg)

        self.restore_render_queue = self.render_queue.copy()
        self.command_queue = []
        self.restore_command_queue = []

    def render_performance_data(self):
        try:
            with open("perf.json") as perfjson:
                data = perfjson.read()
                # This is about 5KB for a really small example. Not sure how well this scales tbh... We'll see I guess
                self.command_queue.append(data)
                self.restore_command_queue = self.command_queue.copy()
        except FileNotFoundError as e:
            # Well then not. Whatever floats your boat.
            pass

    def set_memspeed_target(self):
        from dace.codegen.instrumentation.perfsettings import PerfPAPIInfoStatic
        self.command_queue.append('{ "type": "MemSpeed", "payload": "%s" }' %
                                  str(PerfPAPIInfoStatic.info.memspeed))

    def set_drawing_area(self, drawing_area):
        self.drawing_area = drawing_area
        browserholder = WebKit2.WebView()
        browserholder.set_editable(False)
        scriptdir = os.path.dirname(os.path.abspath(__file__))
        renderer_html_file = os.path.join(scriptdir, "renderer.html")
        with open(renderer_html_file, 'r') as f:
            renderer_html = f.read()

        settings = browserholder.get_settings()
        settings.set_enable_write_console_messages_to_stdout(True)
        settings.set_javascript_can_open_windows_automatically(
            True)  # For popping windows out
        browserholder.set_settings(settings)
        browserholder.connect('create', RenderedGraphHTML5.new_window)

        # Add the scroll handler
        def ScrollBinder(widget, event):
            return RenderedGraphHTML5.onScroll(browserholder, widget, event)

        browserholder.add_events(Gdk.EventMask.SCROLL_MASK)
        browserholder.connect('scroll-event', ScrollBinder)

        browserholder.load_uri("file://" + renderer_html_file)
        self.drawing_area.add(browserholder)
        browserholder.show()

    @staticmethod
    def onScroll(webkit, widget, event):
        """ Handles scrolling (actually, only zoom-Scrolling (CTRL + MouseWheel)) """
        # Handles zoom in / zoom out on Ctrl+mouse wheel

        zoom_const = 0.1
        scale_mode = "internal"

        accel_mask = Gtk.accelerator_get_default_mod_mask()
        if event.state & accel_mask == Gdk.ModifierType.CONTROL_MASK:
            _dir = event.direction
            _zoom = None
            if _dir == Gdk.ScrollDirection.UP:
                _zoom = zoom_const
            elif _dir == Gdk.ScrollDirection.DOWN:
                _zoom = -zoom_const
            direction = event.get_scroll_deltas()[1]
            if direction == 0:
                direction = event.get_scroll_deltas()[2]
            if direction > 0:  # scrolling down -> zoom out
                _zoom = -zoom_const
            elif direction < 0:
                _zoom = zoom_const

            if scale_mode == "external":
                webkit.set_zoom_level(webkit.get_zoom_level() + _zoom)
            elif scale_mode == "internal":
                # We dispatch a script, yippie!
                webkit.run_javascript("window.zoom_func(" + str(_zoom) + ");")
            return True
        return False

    @staticmethod
    def subwindow_close(window, event):

        print("Subwindow-close called")
        for widget in window.get_children():
            widget.run_javascript(
                "window._thisclient.passMessage({type: 'close'}); console.log('close requested');"
            )
        return False

    @staticmethod
    def new_window(view, nav_action):
        print("new_window called")
        win = Gtk.Window()

        win.connect('delete-event', RenderedGraphHTML5.subwindow_close)

        browserwindow = WebKit2.WebView()
        browserwindow.set_editable(False)
        req = nav_action.get_request()
        uri = req.get_uri()

        settings = browserwindow.get_settings()
        settings.set_enable_write_console_messages_to_stdout(True)
        settings.set_javascript_can_open_windows_automatically(True)
        browserwindow.set_settings(settings)
        browserwindow.connect('create', RenderedGraphHTML5.new_window)
        print(str(uri))
        browserwindow.load_uri(uri)

        win.add(browserwindow)

        browserwindow.show()
        win.show()

        win.resize(600, 400)
        win.set_keep_above(True)

        return browserwindow
