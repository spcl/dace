import sys
import os
import subprocess
import asyncio
import time
import json
import websockets
import threading
import queue

import gi
gi.require_version('WebKit2', '4.0')
from gi.repository import Gdk, WebKit2, Gtk

from gi.repository import GObject

import sqlite3
from diode.db_scripts.sql_to_json import MetaFetcher
from dace.codegen.instrumentation.perfsettings import PerfSettings


class RenderedGraphHTML5:
    """ HTML5-based SDFG renderer. """

    def __init__(self, on_click):
        self.drawing_area = None
        self.on_click_cb = on_click
        self.websocket = None
        self.sdfg = None
        self.render_queue = []
        self.restore_render_queue = []  # Render queue copy
        self.command_queue = [
        ]  # Similar to render_queue, but for everything except SDFGs.
        self.command_queue_roofline = []
        self.comm_queue_lock_roofline = threading.Lock()

        self.restore_command_queue = []  # Command queue copy
        threading.Thread(target=self.run_asyncio, daemon=True).start()
        threading.Thread(target=self.run_asyncio_roofline, daemon=True).start()
        self.comm_queue_lock = threading.Lock()

        self.db_request_queue = queue.Queue()
        self.max_db_request_threads = 3  # Sets the amount of threads that may be committed to serve performance results.

        def wrapper():
            self.async_request_handler()

        threading.Thread(target=wrapper, daemon=True).start()

        # data_source determines if data should be rendered from canned data or fresh data.
        self.data_source = "fresh"  # Normally, use results directly from the full results

        self.canned_data_programid = None

    def set_click_cb(self, func):
        self.on_click_cb = func

    def center_highlights(self):
        # TODO
        return

    def save_buttons(self, default_name="recorded_"):
        """ Request the remote to download a summary file once all data is present """
        self.add_to_command_queue(
            '{ "type": "save-images", "name": "%s" }' % default_name)

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

    def open_canned_data(self, can_path, forProgramID=1):

        if can_path == None or can_path == "":
            raise ValueError("Cannot open can from path " + str(can_path))
        # Just set the data source accordingly
        self.data_source = can_path

        # Open the database
        with sqlite3.connect(can_path) as conn:
            c = conn.cursor()

            # Read and draw the SDFG with the lowest index
            c.execute(
                """
    SELECT
        forProgramID, json
    FROM
        `SDFGs`
    WHERE
        forProgramID = ?
            """, (forProgramID, ))

            f = c.fetchall()
            if len(f) == 0:
                # No sdfg has been found in the CAN
                raise ValueError(
                    "The file specified is corrupted or contains no valid data"
                )

            # Extract the json data
            forProgramID, sdfg_json = f[0]

            # Call for a rendering (will clear other pending data)
            self.render_sdfg(sdfg_json)

            # Now request drawing of the selected programID as well
            self.render_performance_data(
                mode="all",
                data_source_path=self.data_source,
                forProgramID=forProgramID)

    def run_asyncio_roofline(self):
        self.impl_run_asyncio(port=8024, handler=self.ConnHandler_roofline)

    def run_asyncio(self):
        self.impl_run_asyncio(port=8023, handler=self.ConnHandler)

    def impl_run_asyncio(self, port, handler):
        try:
            loop = asyncio.SelectorEventLoop()
            asyncio.set_event_loop(loop)
            asyncio.get_event_loop().run_until_complete(
                websockets.serve(handler, 'localhost', port, timeout=0.0001))
            asyncio.get_event_loop().run_forever()
        except Exception as e:
            pass

    async def ConnHandler_roofline(self, websocket, path):

        while (True):
            try:
                msg = ""
                try:
                    msg = await asyncio.wait_for(
                        websocket.recv(), timeout=0.01)
                except asyncio.TimeoutError:
                    # Timeouts are expected
                    pass

                if len(msg) > 0:
                    m = json.loads(msg)
                    self.message_handler_roofline(m)

                # Lock to ensure correct operation (commands can be added asynchronously)
                with self.comm_queue_lock_roofline:
                    for cmd in self.command_queue_roofline:
                        await websocket.send(cmd)
                    self.command_queue_roofline.clear()
            except websockets.ConnectionClosed:
                print("Roofline connection closed.")
                break

    async def ConnHandler(self, websocket, path):
        self.websocket = websocket
        print("Client connected!")
        print("render queue is " + str(len(self.render_queue)) +
              " elements long")
        for sdfg in self.render_queue:
            json = ""
            # Allow directly rendering strings
            if isinstance(sdfg, str):
                json = sdfg
            else:
                json = sdfg.toJSON()
            await websocket.send(json)

        self.render_queue.clear()

        while (True):
            try:
                msg = ""

                try:
                    msg = await asyncio.wait_for(
                        websocket.recv(), timeout=0.01)
                except asyncio.TimeoutError:
                    # Timeouts are expected to happen
                    pass

                if len(msg) > 0:
                    self.message_handler(msg)

                for sdfg in self.render_queue:
                    if isinstance(sdfg, str):
                        json = sdfg
                    else:
                        json = sdfg.toJSON()
                    await websocket.send(json)
                self.render_queue.clear()

                # Lock to ensure correct operation (commands can be added asynchronously)
                with self.comm_queue_lock:
                    for cmd in self.command_queue:
                        # The difference in the command queue: All data must already be in JSON format
                        await websocket.send(cmd)
                    self.command_queue.clear()
            except websockets.ConnectionClosed:
                # If the connection was closed, probably a refresh was
                # requested. This also means that we have to re-queue
                print("Restoring render queue after abort")
                self.render_queue = self.restore_render_queue.copy()
                self.command_queue = self.restore_command_queue.copy()
                break

    def add_to_command_queue(self, x):
        with self.comm_queue_lock as l:
            self.command_queue.append(x)

    def message_handler_roofline(self, m):
        """ Handles roofline-specific commands """

        if m["command"] == "connected" or m["command"] == "get-data":
            # We should respond with the data for each value
            pass
            print("Can data requested")
            if self.data_source == "fresh":
                print(
                    "[FAIL] Data source is fresh, there are no results to be compared"
                )
                return
            else:
                from dace.codegen.instrumentation.perfsettings import InstrumentationProvider
                retdict = InstrumentationProvider.get_roofline_data(
                    self.data_source)
                with self.comm_queue_lock_roofline:
                    self.command_queue_roofline.append(json.dumps(retdict))

        elif m["command"] == "select-program":
            # We should draw the requested program from the CAN.
            programID = m["programID"]

            # Request it.
            self.open_canned_data(self.data_source, programID)

        elif m["command"] == "":
            pass

    def run_async_request(self, task):
        """ Run a request asynchronously """
        self.db_request_queue.put(task)

    def async_request_handler(self):
        spawned_threads = []
        while True:
            request_handler = self.db_request_queue.get()

            transself = self

            def consumer_thread():
                request_handler()
                # Keep running if there are others waiting, otherwise, end the thread to free some resources
                try:
                    while True:
                        elem = transself.db_request_queue.get_nowait()
                        elem()
                except:
                    pass

            newthread = threading.Thread(target=consumer_thread, daemon=True)

            if len(spawned_threads) < self.max_db_request_threads:
                spawned_threads.append(newthread)
                spawned_threads[-1].start()
            else:
                # Wait until any of the previous tasks has finished
                while True:
                    started = False
                    for i, t in enumerate(spawned_threads):
                        if not t.is_alive():
                            # Thread is dead, replace it
                            spawned_threads[i] = newthread
                            spawned_threads[i].start()
                            started = True
                            break
                    if started:
                        break
                    else:
                        request_handler()
                        break

    def message_handler(self, msg):

        db_path = "perfdata.db"
        if self.data_source != "fresh":
            db_path = self.data_source

        local_data_source = self.data_source

        dbg_print = False
        m = json.loads(msg)
        if m["msg_type"] == "info":
            pass
        elif m["msg_type"] == "roofline":
            resp = self.message_handler_roofline(m)
            if resp != None:
                self.add_to_command_queue(json.dumps(resp))

        elif m["msg_type"] == "click":

            def mainthread_click_cb():
                self.on_click_cb(m["clicked_elements"])
                return False

            GObject.idle_add(mainthread_click_cb)

        elif m["msg_type"] == "heartbeat":
            pass  # Ignored, this unblocks the recv() call
        elif m["msg_type"] == "fetcher":
            if dbg_print:
                print("Lazy fetch request received")

            seqid = m["seqid"]
            method = m["msg"]["method"]
            params = m["msg"]["params"]

            synchronous_execution = method != "Analysis"

            if synchronous_execution:
                conn = sqlite3.connect(db_path, check_same_thread=False)
                c = conn.cursor()

                # Attach a default scratch space to every connection such that we can write temporary dbs concurrently
                c.execute("ATTACH DATABASE ':memory:' AS scratch_default;")

            if method == "getSuperSectionCount":

                ma = MetaFetcher(local_data_source, self.canned_data_programid)
                d = ma.getSuperSectionCount(c, *params)
                resp = '{ "type": "fetcher", "seqid": %d, "msg": %s }' % (
                    int(seqid), str(d))
                self.add_to_command_queue(resp)

            elif method == "getAllSectionStateIds":

                def taskrunner():
                    conn = sqlite3.connect(db_path, check_same_thread=False)
                    c = conn.cursor()

                    # Attach a default scratch space to every connection such that we can write temporary dbs concurrently
                    c.execute("ATTACH DATABASE ':memory:' AS scratch_default;")
                    ma = MetaFetcher(local_data_source,
                                     self.canned_data_programid)
                    if dbg_print:
                        print("params: " + str(params))
                    d = ma.getAllSectionStateIds(c, *params)
                    resp = '{ "type": "fetcher", "seqid": %d, "msg": %s }' % (
                        int(seqid), str(d))
                    self.add_to_command_queue(resp)

                    conn.close()

                # Run asynchronously
                self.run_async_request(taskrunner)

            elif method == "getAllSectionNodeIds":

                def taskrunner():
                    conn = sqlite3.connect(db_path, check_same_thread=False)
                    c = conn.cursor()

                    # Attach a default scratch space to every connection such that we can write temporary dbs concurrently
                    c.execute("ATTACH DATABASE ':memory:' AS scratch_default;")
                    ma = MetaFetcher(local_data_source,
                                     self.canned_data_programid)
                    if dbg_print:
                        print("params: " + str(params))
                    d = ma.getAllSectionNodeIds(c, *params)
                    resp = '{ "type": "fetcher", "seqid": %d, "msg": %s }' % (
                        int(seqid), str(d))
                    self.add_to_command_queue(resp)

                    conn.close()

                # Run asynchronously
                #threading.Thread(target=taskrunner).start()
                self.run_async_request(taskrunner)
            elif method == "Analysis":
                # Any analysis, name given in params[0]
                cname = params[0]
                params = params[1:]

                transself = self

                def taskrunner():
                    if dbg_print:
                        print("Running analysis " + cname)
                        print("Params: " + str(params))

                    conn = sqlite3.connect(db_path, check_same_thread=False)
                    c = conn.cursor()

                    # Attach a default scratch space to every connection such that we can write temporary dbs concurrently
                    c.execute("ATTACH DATABASE ':memory:' AS scratch_default;")

                    if local_data_source == "fresh":

                        def my_import(name):
                            components = name.split('.')
                            mod = __import__(components[0])
                            for comp in components[1:]:
                                mod = getattr(mod, comp)
                            return mod

                        _module = my_import("diode.db_scripts.sql_to_json")
                        _cl = getattr(_module, cname)
                        _instance = _cl()

                        d = _instance.query_values(c, *params)
                        respval = json.dumps(d)
                        if d == None:
                            # Special case of undefined
                            d = "null"

                    else:  # Canned data
                        # This is easier as in: The results can be read from a database directly

                        argparams = [*params]
                        query_ss = True
                        if cname == "CriticalPathAnalysis":
                            # This has split IDs (instead of the unified id)
                            tmp = [*params]

                            if tmp[0] == None:
                                tmp[0] = 0x0FFFF
                            # Recreate the correct pair and remove the supersection part from the query
                            argparams = [
                                (int(tmp[1]) << 16) | (int(tmp[0]) & 0xFFFF)
                            ]
                            query_ss = False

                        if argparams[0] == -1 or argparams[0] == "-1":
                            argparams[0] = 0x0FFFFFFFF

                        c.execute(
                            """
SELECT
    json
FROM
    `AnalysisResults`
WHERE
    {progid_q}
    AnalysisName = ?
    AND forUnifiedID = ?
    {ss_q}
--  AND forSection = ?
;""".format(ss_q="AND forSuperSection = ?" if query_ss else "",
                        progid_q="forProgramID = %d AND" % self.canned_data_programid
                        if self.canned_data_programid != None else ""), (
                        cname,
                        *argparams,
                        ))

                        result = c.fetchall()

                        if len(result) == 0:
                            # This can actually happen (and is validly caught at client-side)
                            respval = "null"
                        else:
                            # Unpack
                            respval, = result[0]

                    resp = '{ "type": "fetcher", "seqid": %d, "msg": %s }' % (
                        int(seqid), respval)

                    conn.close()

                    if dbg_print:
                        print("Analysis result:" + str(resp))
                    transself.add_to_command_queue(resp)

                # Run asynchronously
                self.run_async_request(taskrunner)
            else:
                # Arbitrary execution string
                transself = self

                def taskrunner():
                    conn = sqlite3.connect(db_path, check_same_thread=False)
                    c = conn.cursor()

                    # Attach a default scratch space to every connection such that we can write temporary dbs concurrently
                    c.execute("ATTACH DATABASE ':memory:' AS scratch_default;")

                    ma = MetaFetcher(local_data_source,
                                     self.canned_data_programid)
                    if dbg_print:
                        print("method: " + str(method))
                        print("params: " + str(params))
                    d = getattr(MetaFetcher, method)(ma, c, *params)

                    conn.close()
                    tstr = str(d)
                    d = tstr.replace("'", '"')
                    resp = '{ "type": "fetcher", "seqid": %d, "msg": %s }' % (
                        int(seqid), str(d))
                    self.add_to_command_queue(resp)

                # Run asynchronously
                self.run_async_request(taskrunner)

            if synchronous_execution:
                conn.close()

        else:
            print("Unknown/Unhandled message from renderer: " +
                  str(m["msg_type"]))

    def render_sdfg(self, sdfg):
        self.sdfg = sdfg
        self.render_queue.append(sdfg)

        self.restore_render_queue = self.render_queue.copy()
        self.command_queue = []
        self.restore_command_queue = []

    def render_performance_data(self,
                                mode="",
                                data_source_path="fresh",
                                forProgramID=None):
        from dace.codegen.instrumentation.perfsettings import PerfSettings

        self.data_source = data_source_path

        # Set the program ID used for lookups
        if forProgramID != None:
            self.canned_data_programid = int(forProgramID)
        elif data_source_path != "fresh":
            pid = 1
            with sqlite3.connect(data_source_path) as conn:
                c = conn.cursor()
                c.execute("SELECT MAX(forProgramID) FROM AnalysisResults;")
                res = c.fetchall()
                assert len(res) > 0
                pid, = res[0]
                print("res for pid: " + str(res))

            # Use the highest ID (newest program)
            self.canned_data_programid = int(pid)

        # Uncomment to get a button image
        #self.save_buttons()

        try:
            suffix = mode

            # If sql is enabled, use that
            if PerfSettings.perf_use_sql():
                # Don't just send data, but send a message allowing the JS client to request data
                self.add_to_command_queue("""
                {
                    "type": "DataReady",
                    "mode": "%s"
                }
                """ % mode)
                self.restore_command_queue = self.command_queue.copy()

            else:  # Otherwise, go legacy mode
                with open("perf_%s.json" % suffix) as perfjson:
                    data = perfjson.read()
                    self.add_to_command_queue(data)
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

        settings = browserholder.get_settings()
        settings.set_enable_write_console_messages_to_stdout(True)
        settings.set_javascript_can_open_windows_automatically(
            True)  # For popping windows out
        browserholder.set_settings(settings)
        browserholder.connect('create', RenderedGraphHTML5.new_window)
        # Download connector
        browserholder.get_context().connect("download-started",
                                            RenderedGraphHTML5.download)

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
                # We dispatch a script to handle scrolling on the JS side
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
    def download(context, download):
        print("Download was requested")
        print("destination: " + str(download.get_destination()))
        print("download: " + str(download))

        def desthandler(d, destname):
            print("destname: " + str(destname))
            wd = os.getcwd()
            print("current working directory: " + wd)
            d.set_destination("file://" + str(wd) + str("/perf_extracts/") +
                              str(destname))
            return False

        download.connect("decide-destination", desthandler)

        def finishhandler(d):
            print("Finished")
            print("final destination: " + str(d.get_destination()))

        download.connect("finished", finishhandler)

        def errorhandler(d, code):
            print("Error: " + str(code))

        download.connect("failed", errorhandler)
        return True

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
