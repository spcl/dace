import * as renderer from "./renderer_dir/renderer_main";
import {SDFG_Parser} from "./sdfg_parser";
import {ObjectHelper} from "./renderer_dir/datahelper";

class CanvasManager {
    // Manages translation and scaling of canvas rendering

    static counter() {
        return _canvas_manager_counter++;
    }
    constructor(ctx, renderer) {
        this.ctx = ctx;
        this.anim_id = null;
        this.drawables = [];
        this.renderer = renderer;
        this.indices = [];

        this.request_scale = false;
        this.scale_factor = {x: 1, y: 1};

        this._destroying = false;

        this.scale_origin = {x: 0, y: 0};

        this.contention = 0;

        this._svg = document.createElementNS("http://www.w3.org/2000/svg",'svg');

        this.user_transform = this._svg.createSVGMatrix();

        this.addCtxTransformTracking();
    }

    svgPoint(x, y) {
        let pt  = this._svg.createSVGPoint();
        pt.x=x; pt.y=y;
        return pt;
    }

    applyUserTransform() {
        let ut = this.user_transform;
        this.ctx.setTransform(ut.a, ut.b, ut.c, ut.d, ut.e, ut.f);
    }

    get translation() {
        return { x: this.user_transform.e, y: this.user_transform.f };
    }

    addCtxTransformTracking() {
        /* This function is a hack to provide the non-standardized functionality
        of getting the current transform from a RenderingContext.
        When (if) this is standardized, the standard should be used instead.
        This is made for "easy" transforms and does not support saving/restoring
        */

        let svg = document.createElementNS("http://www.w3.org/2000/svg",'svg');
        this.ctx._custom_transform_matrix = svg.createSVGMatrix();
        // Save/Restore is not supported.

        let checker = () => {
            console.assert(!isNaN(this.ctx._custom_transform_matrix.f));
        };
        let _ctx = this.ctx;
        let scale_func = _ctx.scale;
        _ctx.scale = function(sx,sy) {
            _ctx._custom_transform_matrix = _ctx._custom_transform_matrix.scaleNonUniform(sx,sy);
            checker();
            return scale_func.call(_ctx, sx, sy);
        };
        let translate_func = _ctx.translate;
        _ctx.translate = function(sx,sy) {
            _ctx._custom_transform_matrix = _ctx._custom_transform_matrix.translate(sx,sy);
            checker();
            return translate_func.call(_ctx, sx, sy);
        };
        let rotate_func = _ctx.rotate;
        _ctx.rotate = function(r) {
            _ctx._custom_transform_matrix = _ctx._custom_transform_matrix.rotate(r * 180.0 / Math.PI);
            checker();
            return rotate_func.call(_ctx, r);
        };
        let transform_func = _ctx.scale;
        _ctx.transform = function(a,b,c,d,e,f){
			let m2 = svg.createSVGMatrix();
			m2.a=a; m2.b=b; m2.c=c; m2.d=d; m2.e=e; m2.f=f;
            _ctx._custom_transform_matrix = _ctx._custom_transform_matrix.multiply(m2);
            checker();
			return transform_func.call(_ctx,a,b,c,d,e,f);
		};

        let setTransform_func = _ctx.setTransform;
		_ctx.setTransform = function(a,b,c,d,e,f){
			_ctx._custom_transform_matrix.a = a;
			_ctx._custom_transform_matrix.b = b;
			_ctx._custom_transform_matrix.c = c;
			_ctx._custom_transform_matrix.d = d;
			_ctx._custom_transform_matrix.e = e;
            _ctx._custom_transform_matrix.f = f;
            checker();
			return setTransform_func.call(_ctx,a,b,c,d,e,f);
		};

		_ctx.custom_inverseTransformMultiply = function(x,y){
            let pt  = svg.createSVGPoint();
            pt.x=x; pt.y=y;
            checker();
			return pt.matrixTransform(_ctx._custom_transform_matrix.inverse());
		}
    }

    destroy() {
        this._destroying = true;
        this.clearDrawables();
    }

    addDrawable(obj) {
        this.drawables.push(obj);
        this.indices.push({"c": CanvasManager.counter(), "d": obj});
    }

    removeDrawable(drawable) {
        this.drawables = this.drawables.filter(x => x != drawable);
    }

    clearDrawables() {
        for(let x of this.drawables) {
            x.destroy();
        }
        this.drawables = [];
        this.indices = [];
    }

    scale(diff, x=0, y=0) {

        if(this.request_scale || Math.abs(diff) < 0.0001 || this.contention > 0) {
            console.log("Blocking potential race");
            return;
        }
        this.contention++;
        this.request_scale = true;

        this.scale_origin.x = x;
        this.scale_origin.y = y;
        this.scale_factor.x += diff;
        this.scale_factor.y += diff;

        this.scale_factor.x = Math.max(0.001, this.scale_factor.x);
        this.scale_factor.y = Math.max(0.001, this.scale_factor.y);
        {
            let sv = diff < 0 ? 0.9 : 1.1;
            let pt = this.svgPoint(this.scale_origin.x, this.scale_origin.y).matrixTransform(this.user_transform.inverse());
            this.user_transform = this.user_transform.translate(pt.x, pt.y);
            this.user_transform = this.user_transform.scale(sv, sv, 1, 0, 0, 0);
            this.user_transform = this.user_transform.translate(-pt.x, -pt.y);
        }
        this.contention--;
    }

    translate(x, y) {
        this.user_transform = this.user_transform.translate(x / this.user_transform.a, y / this.user_transform.d);
    }

    mapPixelToCoordsX(xpos) {
        return this.svgPoint(xpos, 0).matrixTransform(this.user_transform.inverse()).x;
    }

    mapPixelToCoordsY(ypos) {
        return this.svgPoint(0, ypos).matrixTransform(this.user_transform.inverse()).y;
    }

    getScale() {
        ObjectHelper.assert("Uniform scale", this.scale_factor.x == this.scale_factor.y);
        return this.noJitter(this.scale_factor.x); // We don't allow non-uniform scaling.
    }

    noJitter(x) {
        x = parseFloat(x.toFixed(3));
        x = Math.round(x * 100) / 100;
        return x;
    }


    draw() {
        if(this._destroying)
            return;

        if(this.contention > 0) return;
        this.contention += 1;
        let ctx = this.ctx;

        // Clear with default transform
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.restore();

        let mx = 0;
        let my = 0;
        if(this.request_scale && this.contention == 1) {

            mx = this.mapPixelToCoordsX(this.scale_origin.x);
            my = this.mapPixelToCoordsY(this.scale_origin.y);

            // Reset the translation
            this.applyUserTransform();
            this.request_scale = false;
        }
        else
        {
            mx = this.mapPixelToCoordsX(this.scale_origin.x);
            my = this.mapPixelToCoordsY(this.scale_origin.y);
        }
        this.applyUserTransform();

        this.renderer.draw();
        this.contention -= 1;
    }

    draw_async() {
        this.anim_id = window.requestAnimationFrame(() => this.draw());
    }
}

function isWithinBB(x, y, w, h, layoutinfo) {
    return (x >= layoutinfo.x - layoutinfo.width / 2.0) &&
        (x + w <= layoutinfo.x + layoutinfo.width / 2.0) &&
        (y >= layoutinfo.y - layoutinfo.height / 2.0) &&
        (y + h <= layoutinfo.y + layoutinfo.height / 2.0);

}

function isWithinBBEdge(x, y, w, h, layoutinfo) {
    // TODO: Use w and h
    // Compute distance between point and line for each point in curve
    for (let i = 0; i < layoutinfo.points.length - 1; i++) {
        let dist = ptLineDistance({x: x, y: y}, layoutinfo.points[i], layoutinfo.points[i + 1]);
        if (dist < 5.0)
            return true;
    }

    // Bounding box method
    /*
    return (x >= layoutinfo.x) &&
        (x + w <= layoutinfo.x + layoutinfo.width) &&
        (y >= layoutinfo.y) &&
        (y + h <= layoutinfo.y + layoutinfo.height);
    */
}

function getQuadraticAngle(t, sx, sy, cp1x, cp1y, ex, ey) {
    let dx = 2*(1-t)*(cp1x-sx) + 2*t*(ex-cp1x);
    let dy = 2*(1-t)*(cp1y-sy) + 2*t*(ey-cp1y);
    return -Math.atan2(dx, dy) + 0.5*Math.PI;
}

// Returns the distance from point p to line defined by two points (line1, line2)
function ptLineDistance(p, line1, line2) {
    let dx = (line2.x - line1.x);
    let dy = (line2.y - line1.y);
    let res = dy * p.x - dx * p.y + line2.x * line1.y - line2.y * line1.x;

    return Math.abs(res) / Math.sqrt(dy*dy + dx*dx);
}

function equals(a, b) {
     return JSON.stringify(a) === JSON.stringify(b);
}


class SDFGRenderer {
    constructor(sdfg, diode_context, container) {
        // DIODE/SDFV-related fields
        this.sdfg = sdfg;
        this.diode = diode_context;
        this.analysis_provider = null;

        // Rendering-related fields
        this.container = container;
        this.ctx = null;
        this.canvas = null;
        this.last_visible_elements = null;
        this.last_hovered_elements = null;
        this.last_clicked_elements = null;

        // Mouse-related fields
        this.drag_start = null; // Null if the mouse/touch is not activated
        this.drag_second_start = null; // Null if two touch points are not activated

        this.init_elements();
    }

    // Initializes the DOM
    init_elements() {
        jQuery.when(
            jQuery.getScript('renderer_dir/global_vars.js'),
            jQuery.getScript('renderer_dir/dagre.js'),
            $.Deferred(function( deferred ){
                $( deferred.resolve );
            })
        ).done(() => {
            this.canvas = document.createElement('canvas');
            this.container.append(this.canvas);
            // TODO: Add buttons
            this.ctx = this.canvas.getContext("2d");

            // Translation/scaling management
            this.canvas_manager = new CanvasManager(this.ctx, this);

            // Create the initial SDFG layout
            this.relayout_sdfg();

            // Set mouse event handlers
            this.set_mouse_handlers();

            // Link the analysis provider from which to pull values (for performance data)
            this.analysis_provider = ((x, y) => this.diode.analysisProvider(x,y));

            // Queue first render
            this.draw_async();
        });
    }

    draw_async() {
        this.canvas_manager.draw_async();
    }

    // Set mouse events (e.g., click, drag, zoom)
    set_mouse_handlers() {
        let canvas = this.getCanvas();
        let br = () => canvas.getBoundingClientRect();

        let comp_x = event => this.canvas_manager.mapPixelToCoordsX(event.clientX - br().left);
        let comp_y = event => this.canvas_manager.mapPixelToCoordsY(event.clientY - br().top);

        // Mouse handler event types
        for (let evtype of ['mousedown', 'mousemove', 'mouseup', 'touchstart', 'touchmove', 'touchend',
                            'wheel', 'click', 'dblclick', 'contextmenu']) {
            canvas.addEventListener(evtype, x => {
                x.stopPropagation();
                x.preventDefault();
                this.on_mouse_event(x, comp_x, comp_y, evtype)
            });
        }

        // Prevent double clicking from selecting text (see https://stackoverflow.com/a/43321596/6489142)
        /*
        canvas.addEventListener('mousedown', function (event) {
            if (event.detail > 1)
                event.preventDefault();
        }, false);
        */

        /////////
        // TODO: Should move to event handler


        canvas.addEventListener('wheel', e => {
            e.stopPropagation();
            e.preventDefault();

            let canvas = () => this.getCanvas();
            let br = () => canvas().getBoundingClientRect();

            let comp_x = event => (event.clientX - br().left);
            let comp_y = event => (event.clientY - br().top);
            this.canvas_manager.scale(-e.deltaY / 1000.0, comp_x(e), comp_y(e));
            this.canvas_manager.draw_async();

        });

    }

    // Layout SDFG elements (states, nodes, scopes, nested SDFGs)
    relayout_sdfg() {

    }

    // Render SDFG
    draw() {
        let ctx = this.ctx;

        on_post_draw();
    }

    on_post_draw() {
        let ctx = this.ctx;
        ctx.save();
        ctx.fillText('aaa', 0, 0);
        ctx.restore();
    }

    // Returns a dictionary of SDFG elements in a given rectangle. Used for
    // selection, rendering, localized transformations, etc.
    // The output is a dictionary of lists of dictionaries. The top-level keys are:
    // states, nodes, connectors, edges, isedges (interstate edges). For example:
    // {'states': [{sdfg: sdfg_name, state: 1}, ...], nodes: [sdfg: sdfg_name, state: 1, node: 5],
    //              edges: [], isedges: [], connectors: []}
    elements_in_rect(x, y, w, h) {
        let elements = {states: [], nodes: [], connectors: [],
                        edges: [], isedges: []};

        function traverse_recursive(sdfg, elements) {
            let sdfg_name = sdfg.attributes.name;
            sdfg.nodes.forEach(function (state, state_id) {
                if (isWithinBB(x, y, w, h, state.attributes.layout)) {
                    // Selected states
                    elements.states.push({sdfg: sdfg_name, state: state_id});

                    state.nodes.forEach(function (node, node_id) {
                        if (isWithinBB(x, y, w, h, node.attributes.layout)) {
                            // Selected nodes
                            elements.nodes.push({
                                sdfg: sdfg_name, state: state_id, node: node_id
                            });

                            // NOTE: Connectors are sized LINEHEIGHT x LINEHEIGHT, spaced evenly with LINEHEIGHT
                            let starty = node.attributes.layout.y - node.attributes.layout.height / 2.0;
                            let endy = node.attributes.layout.y + node.attributes.layout.height / 2.0;

                            let i;
                            // Input connectors
                            if (y >= starty - LINEHEIGHT/2.0 && y <= starty + LINEHEIGHT/2.0) {
                                let conn_length = 2 * LINEHEIGHT * node.attributes.in_connectors.length - LINEHEIGHT;
                                let conn_startx = node.attributes.layout.x - conn_length / 2.0;
                                for (i = 0; i < node.attributes.in_connectors.length; i++) {
                                    if (x >= conn_startx + i * 2 * LINEHEIGHT &&
                                        x <= conn_startx + i * 2 * LINEHEIGHT + LINEHEIGHT) {
                                        elements.connectors.push({sdfg: sdfg_name, state: state_id,
                                                                  node: node_id, connector: i, conntype: "in"});
                                    }
                                }
                            }

                            // Output connectors
                            if (y >= endy - LINEHEIGHT/2.0 && y <= endy + LINEHEIGHT/2.0) {
                                let conn_length = 2 * LINEHEIGHT * node.attributes.out_connectors.length - LINEHEIGHT;
                                let conn_startx = node.attributes.layout.x - conn_length / 2.0;
                                for (i = 0; i < node.attributes.out_connectors.length; i++) {
                                    if (x >= conn_startx + i * 2 * LINEHEIGHT &&
                                        x <= conn_startx + i * 2 * LINEHEIGHT + LINEHEIGHT) {
                                        elements.connectors.push({sdfg: sdfg_name, state: state_id,
                                                                  node: node_id, connector: i, conntype: "out"});
                                    }
                                }
                            }
                            // END of connectors

                            // If nested SDFG, traverse recursively
                            if (node.type === "NestedSDFG")
                                traverse_recursive(node.sdfg, elements);
                        }
                    });

                    // Selected edges
                    state.edges.forEach((edge, edge_id) => {
                        if (isWithinBBEdge(x, y, w, h, edge.attributes.layout)) {
                            elements.edges.push({sdfg: sdfg_name, state: this.state_id, edge: edge_id});
                        }
                    });
                }
            });

            // Selected inter-state edges
            sdfg.edges.forEach((edge, isedge_id) => {
                if (isWithinBBEdge(x, y, w, h, edge.attributes.layout)) {
                    elements.isedges.push({sdfg: sdfg_name, edge: isedge_id});
                }
            });
        }

        // Traverse all SDFGs recursively
        traverse_recursive(this.sdfg, elements);

        return elements;
    }

    on_mouse_event(event, comp_x_func, comp_y_func, evtype="click") {
        let dirty = false; // Whether to redraw at the end

        if (evtype === "touchstart" || evtype === "mousedown") {
            let ev = (evtype === "touchstart") ? event.touches[0] : event;
            this.drag_start = ev;
            if (evtype === "touchstart" && e.targetTouches.length == 2)
                this.drag_second_start = event.touches[1];

        } else if (evtype === "touchend" || evtype === "mouseup") {
            this.drag_start = null;
            this.drag_second_start = null;

        } else if (evtype === "mousemove" || evtype === "touchmove") {
            // Zoom (pinching)
            if (evtype === "touchmove" && e.targetTouches.length == 2) {
                // Find distance between two points and center, zoom to that
                let centerX = (comp_x_func(this.drag_start) + comp_x_func(this.drag_second_start)) / 2.0;
                let centerY = (comp_y_func(this.drag_start) + comp_y_func(this.drag_second_start)) / 2.0;
                let x1 = comp_x_func(this.drag_start), x2 = comp_x_func(this.drag_second_start);
                let y1 = comp_y_func(this.drag_start), y2 = comp_y_func(this.drag_second_start);
                let initialDistance = Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);

                x1 = comp_x_func(event.touches[0]);
                x2 = comp_x_func(event.touches[1]);
                y1 = comp_y_func(event.touches[0]);
                y2 = comp_y_func(event.touches[1]);
                let currentDistance = Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);

                // TODO: Better scaling formula w.r.t. distance between touches
                this.canvas_manager.scale((currentDistance - initialDistance) / 30000.0,
                    comp_x_func(event), comp_y_func(event));

                this.drag_start = event.touches[0];
                this.drag_second_start = event.touches[1];

            } else if (evtype === 'mousemove' || event.targetTouches.length == 1) { // dragging
                let ev = (evtype === 'touchmove') ? event.touches[0] : event;
                if (evtype === 'touchmove') {
                    ev.buttons = 1;
                    ev.movementX = ev.clientX - this.drag_start.clientX;
                    ev.movementY = ev.clientY - this.drag_start.clientY;
                }


                if (ev.buttons & 1) {
                    // Only accept the primary (~left) mouse button as dragging source
                    let movement = [
                        e.movementX,
                        e.movementY
                    ];

                    this.canvas_manager.translate(...movement);
                }

                this.drag_start = position;
            }

            // Mark for redraw
            dirty = true;
        }
        // End of mouse-move/touch-based events

        if(mode === "click" || mode === "dblclick") {
            transmitter.send(JSON.stringify({"msg_type": mode, "clicked_elements": clicked_elements}));

            // Prevent text from being selected etc.
            event.preventDefault();
        }
        else if(mode === "contextmenu") {
            // Send a message (as with the click handler), but include all information here as well.
            transmitter.send(JSON.stringify({
                "msg_type": "contextmenu",
                "clicked_elements": clicked_elements,
                "lpos": {'x': x, 'y': y}, // Logical position (i.e. inside the graph),
                'spos': { 'x': event.x, 'y': event.y}, // Screen position, i.e. as reported by the event
            }));

        } else if(mode === "hover") {
            if (event.buttons & 1) // Dragging does not induce redaw/hover
                return;

            transmitter.send(JSON.stringify({
                "msg_type": "hover",
                "clicked_elements": clicked_elements,
                "lpos": {'x': x, 'y': y}, // Logical position (i.e. inside the graph),
                'spos': {'x': event.x, 'y': event.y}, // Screen position, i.e. as reported by the event
            }));
        }

        let omsg = JSON.parse(msg);
        if (omsg.msg_type.endsWith('click')) {
            // ok
        } else if (omsg.msg_type === 'contextmenu') {
            // ok
        } else if (omsg.msg_type === 'hover') {
            // ok
        } else {
            console.log("Unexpected message type '" + omsg.msg_type + "'");
            return;
        }
        let clicked_elems = omsg.clicked_elements;
        let clicked_states = clicked_elems.filter(x => x.type === 'SDFGState');
        let clicked_nodes = clicked_elems.filter(x => x.type !== 'SDFGState' && !x.type.endsWith("Edge"));
        let clicked_edges = clicked_elems.filter(x => x.type === "MultiConnectorEdge");
        let clicked_interstate_edges = clicked_elems.filter(x => x.type === "Edge");

        let state_id = null;
        let node_id = null;
        if (clicked_states.length > 1) {
            alert("Cannot determine clicked state!");
            return;
        }
        if (clicked_nodes.length > 1) {
            console.warn("Multiple nodes cannot be selected");
            //#TODO: Arbitrate this - for now, we select the element with the lowest id
        }

        let state_only = false;

        // Check if anything was clicked at all
        if (clicked_states.length == 0 && clicked_interstate_edges.length == 0 && omsg.msg_type !== 'hover') {
            // Nothing was selected
            let sstate = this.initialized_sdfgs[0];
            sstate.clearHighlights();
            this.render_free_variables();
            return;
        }
        if ((clicked_nodes.length + clicked_edges.length + clicked_interstate_edges.length) === 0) {
            // A state was selected
            if (clicked_states.length > 0)
                state_only = true;
        }

        if (clicked_states.length > 0)
            state_id = clicked_states[0].id;
        if (clicked_interstate_edges.length > 0)
            node_id = clicked_interstate_edges[0].id;

        if (clicked_nodes.length > 0)
            node_id = clicked_nodes[0].id;
        else if (clicked_edges.length > 0)
            node_id = clicked_edges[0].id;

        if (omsg.msg_type === "hover") {
            let sdfg = this.initialized_sdfgs[0].sdfg;

            // Position for tooltip
            let pos = omsg.lpos;

            sdfg.mousepos = pos;

            if (state_only) {
                sdfg.hovered = {'state': [state_id, pos]};
            } else if (clicked_nodes.length > 0)
                sdfg.hovered = {'node': [state_id, node_id, pos]};
            else if (clicked_edges.length > 0) {
                let edge_id = clicked_edges[0].true_id;
                sdfg.hovered = {'edge': [state_id, edge_id, pos]};
            } else if (clicked_interstate_edges.length > 0) {
                let isedge_id = clicked_interstate_edges[0].true_id;
                sdfg.hovered = {'interstate_edge': [isedge_id, pos]};
            } else {
                sdfg.hovered = {};
            }
            return;
        }

        if (omsg.msg_type === "dblclick") {
            let sdfg = this.initialized_sdfgs[0].sdfg;
            let elem;
            if (state_only) {
                elem = sdfg.nodes[state_id];
            } else if (clicked_nodes.length > 0) {
                elem = sdfg.nodes[state_id].nodes[node_id];
            } else {
                return false;
            }
            // Toggle collapsed state
            if ('is_collapsed' in elem.attributes) {
                elem.attributes.is_collapsed = !elem.attributes.is_collapsed;

                // Re-layout SDFG
                this.initialized_sdfgs[0].setSDFG(this.initialized_sdfgs[0].sdfg);
                this.initialized_sdfgs[0].init_SDFG();
            }
            return;
        }

        if (omsg.msg_type == "contextmenu") {
            // Context menu was requested
            let spos = omsg.spos;
            let lpos = omsg.lpos;

            let cmenu = new ContextMenu();
            cmenu.addOption("Show transformations", x => {
                console.log("'Show transformations' was clicked");

                this.project().request(['highlight-transformations-' + this.getState()['sdfg_name']], x => {

                }, {
                    params: {
                        state_id: state_id,
                        node_id: node_id
                    }
                });
            });
            cmenu.addOption("Apply transformation \u25B6", x => {
                console.log("'Apply transformation' was clicked");

                // Find available transformations for this node
                this.project().request(['get-transformations-' + this.getState()['sdfg_name']], x => {
                    console.log("get-transformations response: ", x);

                    let tmp = Object.values(x)[0];

                    // Create a sub-menu at the correct position


                    let submenu = new ContextMenu();

                    for (let y of tmp) {
                        submenu.addOption(y.opt_name, x => {
                            this.project().request(['apply-transformation-' + this.getState()['sdfg_name']], x => {
                                },
                                {
                                    params: y.id_name
                                });
                        });
                    }

                    submenu.show(spos.x + cmenu.width(), spos.y);
                }, {
                    params: {
                        state_id: state_id,
                        node_id: node_id
                    }
                });

                // Don't close the current context menu from this event
                x.preventDefault();
                x.stopPropagation();
            });
            cmenu.show(spos.x, spos.y);

            return;
        }

        if (omsg.msg_type !== "click")
            return;

        // Highlight selected items
        let sstate = this.initialized_sdfgs[0];
        sstate.clearHighlights();
        if (state_only) {
            clicked_states.forEach(n => {
                sstate.addHighlight({'state-id': state_id});
            });
        } else {
            clicked_nodes.forEach(n => {
                sstate.addHighlight({'state-id': state_id, 'node-id': n.id});
            });
            clicked_edges.forEach(e => {
                sstate.addHighlight({'state-id': state_id, 'edge-id': e.true_id});
            });
            clicked_interstate_edges.forEach(e => {
                sstate.addHighlight({'state-id': state_id, 'isedge-id': e.true_id});
            });
        }

        // Render properties asynchronously
        setTimeout(() => {
            // Get and render the properties from now on
            let sdfg_data = this.getSDFGDataFromState();
            let sdfg = sdfg_data.sdfg;
            let sdfg_name = sdfg_data.sdfg_name;

            console.log("sdfg", sdfg);

            let states = sdfg.nodes;
            let state = null;
            for (let x of states) {
                if (x.id == state_id) {
                    state = x;
                    break;
                }
            }
            let render_props = n => {
                let attr = n.attributes;

                let akeys = Object.keys(attr).filter(x => !x.startsWith("_meta_"));

                let proplist = [];
                for (let k of akeys) {

                    let value = attr[k];
                    let meta = attr["_meta_" + k];
                    if (meta == undefined) {
                        continue;
                    }

                    let pdata = JSON.parse(JSON.stringify(meta));
                    pdata.value = value;
                    pdata.name = k;

                    proplist.push(pdata);
                }
                let nid = parseInt(n.id);
                if (isNaN(nid) || node_id == null) {
                    nid = node_id
                }
                let propobj = {
                    node_id: nid,
                    state_id: state_id,
                    sdfg_name: sdfg_name,
                    data: () => ({props: proplist})
                };

                this.renderProperties(propobj);
            };
            if (clicked_interstate_edges.length > 0) {
                let edges = sdfg.edges;
                for (let e of edges) {
                    if (e.src == node_id.src && e.dst == node_id.dst) {
                        render_props(e.attributes.data);
                        break;
                    }
                }
                return;
            }
            if (state_only) {
                render_props(state);
                return;
            }

            let nodes = state.nodes;
            for (let ni = 0; ni < nodes.length; ++ni) {
                let n = nodes[ni];
                if (n.id != node_id)
                    continue;

                // Special case treatment for scoping nodes (i.e. Maps, Consumes, ...)
                if (n.type.endsWith("Entry")) {
                    // Find the matching exit node
                    let exit_node = this.find_exit_for_entry(nodes, n);
                    // Highlight both entry and exit nodes
                    sstate.addHighlight({'state-id': state_id, 'node-id': exit_node.id});
                    let tmp = this.merge_properties(n, 'entry_', exit_node, 'exit_');
                    render_props(tmp);

                    break;
                } else if (n.type.endsWith("Exit")) {
                    // Find the matching entry node and continue with that
                    let entry_id = parseInt(n.scope_entry);
                    let entry_node = nodes[entry_id];
                    // Highlight both entry and exit nodes
                    sstate.addHighlight({'state-id': state_id, 'node-id': entry_id});
                    let tmp = this.merge_properties(entry_node, 'entry_', n, 'exit_');
                    render_props(tmp);
                    break;
                } else if (n.type === "AccessNode") {
                    // Find matching data descriptor and show that as well
                    let ndesc = sdfg.attributes._arrays[n.attributes.data];
                    let tmp = this.merge_properties(n, '', ndesc, 'datadesc_');
                    render_props(tmp);
                    break;
                }

                render_props(n);
                break;
            }

            let edges = state.edges;
            for (let e of edges) {
                if (e.src == node_id.src && e.dst == node_id.dst) {
                    render_props(e.attributes.data);
                    break;
                }
            }
        }, 0);

        if (dirty)
            this.draw_async();
    }
}



export { SDFGRenderer }