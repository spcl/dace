import sys
import subprocess

from xdot.ui.elements import Graph
from xdot.dot.lexer import ParseError
from xdot.dot.parser import XDotParser

import xdot
print(xdot.ui.elements.__file__)

from gi.repository import Gdk


class RenderedGraph:
    """ Legacy xdot-based SDFG renderer. """

    def __init__(self):
        self.ZOOM_INCREMENT = 1.1

        self.xdot_graph = None
        self.drawing_area = None
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.drag_start_x = None
        self.drag_start_y = None
        self.drag_trans_x = 0.0
        self.drag_trans_y = 0.0
        self.dragging = False
        self.zoom_ratio = 1.0
        self.highlighted_elements = []
        self.click_cb = None
        self.sdfg = None

    def set_click_cb(self, cb):
        self.click_cb = cb

    def determine_scroll_direction(self, ev):
        if ev.direction == Gdk.ScrollDirection.SMOOTH:
            (t, x, y) = Gdk.Event.get_scroll_deltas(ev)
            if y < 0: return "in"
            else: return "out"
        else:
            if ev.direction == Gdk.ScrollDirection.UP:
                return "in"
            elif ev.direction == Gdk.ScrollDirection.DOWN:
                return "out"
        raise ValueError("Unrecognized scroll direction")

    def set_dotcode(self, dotcode, preserve_view=False):
        if self.drawing_area == None:
            raise ValueError("You need to assign a drawing area first.")
        xdotcode = self.run_xdot(dotcode)
        if xdotcode is None:
            raise ValueError("xdot didn't work on input:\n" + dotcode)
        xdotcb = xdotcode.encode('utf-8')
        parser = XDotParser(xdotcb)
        self.xdot_graph = parser.parse()
        if preserve_view == False:
            self.zoom("default", center=True)
        self.drawing_area.queue_draw()

    def set_drawing_area(self, drawing_area):
        self.drawing_area = drawing_area
        self.drawing_area.connect("draw", self.render)
        self.drawing_area.set_events(Gdk.EventMask.BUTTON_PRESS_MASK
                                     | Gdk.EventMask.SCROLL_MASK
                                     | Gdk.EventMask.BUTTON_MOTION_MASK)
        self.drawing_area.connect("scroll-event", self.OnScrollSDFG)
        self.drawing_area.connect("button-press-event", self.OnButtonPressSDFG)
        self.drawing_area.connect("button-release-event",
                                  self.OnButtonReleaseSDFG)
        self.drawing_area.connect("motion-notify-event", self.OnMouseMoveSDFG)

    def OnScrollSDFG(self, widget, ev):
        d = self.determine_scroll_direction(ev)
        self.zoom(d, pos=(ev.x, ev.y))
        widget.queue_draw()
        return False

    def OnButtonPressSDFG(self, widget, ev):
        x, y = ev.x, ev.y
        elem = self.get_element_by_coords(x, y)
        self.clear_highlights()
        if elem is not None:
            self.highlight_element(elem)
        self.handle_button_press(ev)
        if self.click_cb is not None:
            self.click_cb(self.sdfg, elem)
        else:
            print("No callback for clicks on element set!")
        return False

    def OnMouseMoveSDFG(self, widget, ev):
        self.handle_drag_motion(ev)
        return False

    def OnButtonReleaseSDFG(self, widget, ev):
        self.handle_button_release(ev)
        return False

    def render_sdfg(self, sdfg):
        self.sdfg = sdfg
        dotcode = sdfg.draw()
        self.set_dotcode(dotcode)

    def set_memspeed_target(self):
        pass

    def render_performance_data(self):
        pass

    def render(self, wid, cr):
        if self.drawing_area == None:
            raise ValueError("You need to assign a drawing area first.")
        cr.set_source_rgba(1.0, 1.0, 1.0, 1.0)
        cr.paint()
        cr.save()
        rect = self.drawing_area.get_allocation()
        cr.translate(0.5 * rect.width, 0.5 * rect.height)
        cr.scale(self.zoom_ratio, self.zoom_ratio)
        cr.translate(-self.trans_x, -self.trans_y)
        if self.xdot_graph != None:
            self.xdot_graph.draw(cr, highlight_items=self.highlighted_elements)
        cr.restore()

    def zoom(self, zoom_direction, center=False, pos=None):
        # Constrain zoom ratio to a sane range to prevent numerical
        # instability.
        zoom_ratio = min(self.zoom_ratio, 1E4)
        zoom_ratio = max(self.zoom_ratio, 1E-6)

        if zoom_direction == "in":
            zoom_ratio *= self.ZOOM_INCREMENT
        elif zoom_direction == "out":
            zoom_ratio /= self.ZOOM_INCREMENT
        elif zoom_direction == "default":
            zoom_ratio = 1.0
        else:
            return

        if center:
            self.trans_x = self.xdot_graph.width / 2
            self.trans_y = self.xdot_graph.height / 2
        elif pos is not None:
            rect = self.drawing_area.get_allocation()
            x, y = pos
            x -= 0.5 * rect.width
            y -= 0.5 * rect.height
            self.trans_x += x / self.zoom_ratio - x / zoom_ratio
            self.trans_y += y / self.zoom_ratio - y / zoom_ratio
        self.zoom_ratio = zoom_ratio

    def handle_button_press(self, ev):
        self.drag_start_x = ev.x
        self.drag_start_y = ev.y
        self.drag_trans_x = self.trans_x
        self.drag_trans_y = self.trans_y
        self.dragging = True

    def handle_drag_motion(self, ev):
        if self.dragging:
            delta_x = self.drag_start_x - ev.x
            delta_y = self.drag_start_y - ev.y
            self.trans_x = self.drag_trans_x + (delta_x / self.zoom_ratio)
            self.trans_y = self.drag_trans_y + (delta_y / self.zoom_ratio)
            self.drag_start = None
            self.drawing_area.queue_draw()

    def handle_button_release(self, ev):
        self.handle_drag_motion(ev)
        self.dragging = False

    def get_element_by_coords(self, x, y):
        if self.xdot_graph == None:
            return None
        x, y = self.window2graph(x, y)
        elem = self.xdot_graph.get_element(x, y)
        return elem

    def get_subgraph_by_coords(self, x, y):
        if self.xdot_graph == None:
            return None
        x, y = self.window2graph(x, y)
        subg_label = self.xdot_graph.get_subgraph(x, y)
        return subg_label

    def get_element_by_id(self, id):
        for n in self.xdot_graph.nodes:
            if n.id.decode('utf-8') == id:
                return n
        return None

    def highlight_element(self, elem):
        self.highlighted_elements.append(elem)
        self.drawing_area.queue_draw()

    def center_highlights(self):

        xmin, xmax = float('Inf'), -float('Inf')
        ymin, ymax = float('Inf'), -float('Inf')

        for e in self.highlighted_elements:
            x1, x2, y1, y2 = e.bounding
            if (x1 > -float('Inf')) & (x1 < float('Inf')):
                if x1 > xmax:
                    xmax = x1
                if x1 < xmin:
                    xmin = x1
            if (y1 > -float('Inf')) & (y1 < float('Inf')):
                if y1 > ymax:
                    ymax = y1
                if y1 < ymin:
                    ymin = y1
            if (x2 > -float('Inf')) & (x2 < float('Inf')):
                if x2 > xmax:
                    xmax = x2
                if x2 < xmin:
                    xmin = x2
            if (y2 > -float('Inf')) & (y2 < float('Inf')):
                if y2 > ymax:
                    ymax = y2
                if y2 < ymin:
                    ymin = y2

            self.trans_x = xmin
            self.trans_y = ymin

        self.drawing_area.queue_draw()

    def clear_highlights(self):
        self.highlighted_elements = []
        self.drawing_area.queue_draw()

    def window2graph(self, x, y):
        rect = self.drawing_area.get_allocation()
        x -= 0.5 * rect.width
        y -= 0.5 * rect.height
        x /= self.zoom_ratio
        y /= self.zoom_ratio
        x += self.trans_x
        y += self.trans_y
        return x, y

    def run_xdot(self, dotcode):
        try:
            p = subprocess.Popen(
                ["dot", '-Txdot'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                universal_newlines=False)
        except OSError as exc:
            error = '%s: %s' % (None, exc.strerror)
            p = subprocess.CalledProcessError(exc.errno, None, exc.strerror)
        else:
            xdotcode, error = p.communicate(dotcode.encode())
        error = error.rstrip()
        if error:
            sys.stderr.write(str(error) + '\n')
        if p.returncode != 0:
            return None
        return xdotcode.decode()
