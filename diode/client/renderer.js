import {ContextMenu} from "./diode.js";
import * as elements from "./renderer_elements.js";

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

function isWithinBB(x, y, layoutinfo) {
    return (x >= layoutinfo.x - layoutinfo.width / 2.0) &&
        (x <= layoutinfo.x + layoutinfo.width / 2.0) &&
        (y >= layoutinfo.y - layoutinfo.height / 2.0) &&
        (y <= layoutinfo.y + layoutinfo.height / 2.0);
}

function isBBoverlapped(x, y, w, h, layoutinfo) {
    return (x <= layoutinfo.x + layoutinfo.width / 2.0) &&
        (x + w >= layoutinfo.x - layoutinfo.width / 2.0) &&
        (y <= layoutinfo.y + layoutinfo.height / 2.0) &&
        (y + h >= layoutinfo.y - layoutinfo.height / 2.0);
}

function isWithinBBEdge(x, y, layoutinfo) {
    // Compute distance between point and line for each point in curve
    for (let i = 0; i < layoutinfo.points.length - 1; i++) {
        let dist = ptLineDistance({x: x, y: y}, layoutinfo.points[i], layoutinfo.points[i + 1]);
        if (dist < 5.0)
            return true;
    }

    // Bounding box method
    /*
    return (x >= layoutinfo.x) &&
        (x <= layoutinfo.x + layoutinfo.width) &&
        (y >= layoutinfo.y) &&
        (y <= layoutinfo.y + layoutinfo.height);
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

function calculateBoundingBox(g) {
    // iterate over all objects, calculate the size of the bounding box
    let bb = {};
    bb.width = 0;
    bb.height = 0;

    g.nodes().forEach(function (v) {
        let x = g.node(v).x + g.node(v).width / 2.0;
        let y = g.node(v).y + g.node(v).height / 2.0;
        if (x > bb.width) bb.width = x;
        if (y > bb.height) bb.height = y;
    });

    return bb;
}

function calculateEdgeBoundingBox(edge) {
    // iterate over all points, calculate the size of the bounding box
    let bb = {};
    bb.x1 = edge.points[0].x;
    bb.y1 = edge.points[0].y;
    bb.x2 = edge.points[0].x;
    bb.y2 = edge.points[0].y;

    edge.points.forEach(function (p) {
        bb.x1 = p.x < bb.x1 ? p.x : bb.x1;
        bb.y1 = p.y < bb.y1 ? p.y : bb.y1;
        bb.x2 = p.x > bb.x2 ? p.x : bb.x2;
        bb.y2 = p.y > bb.y2 ? p.y : bb.y2;
    });

    bb = {'x': bb.x1, 'y': bb.y1, 'width': (bb.x2 - bb.x1),
          'height': (bb.y2 - bb.y1)};
    if (bb.width <= 5) {
        bb.width = 10;
        bb.x -= 5;
    }
    if (bb.height <= 5) {
        bb.height = 10;
        bb.y -= 5;
    }
    return bb;
}

function calculateNodeSize(sdfg_state, node, ctx) {
    let labelsize = ctx.measureText(node.label).width;
    let inconnsize = 2 * LINEHEIGHT * node.attributes.in_connectors.length - LINEHEIGHT;
    let outconnsize = 2 * LINEHEIGHT * node.attributes.out_connectors.length - LINEHEIGHT;
    let maxwidth = Math.max(labelsize, inconnsize, outconnsize);
    let maxheight = 2*LINEHEIGHT;
    if (node.attributes.in_connectors.length + node.attributes.out_connectors.length > 0) {
        maxheight += 4*LINEHEIGHT;
    }

    let size = { width: maxwidth, height: maxheight }

    // add something to the size based on the shape of the node
    if (node.type === "AccessNode") {
        size.width += size.height;
    }
    else if (node.type.endsWith("Entry")) {
        size.width += 2.0 * size.height;
        size.height /= 1.75;
    }
    else if (node.type.endsWith("Exit")) {
        size.width += 2.0 * size.height;
        size.height /= 1.75;
    }
    else if (node.type === "Tasklet") {
        size.width += 2.0 * (size.height / 3.0);
        size.height /= 1.75;
    }
    else if (node.type === "EmptyTasklet") {
        size.width = 0.0;
        size.height = 0.0;
    }
    else if (node.type === "Reduce") {
        size.width *= 2;
        size.height = size.width / 3.0;
    }
    else {
    }

    return size
}

function check_and_redirect_edge(edge, drawn_nodes, sdfg_state) {
    // If destination is not drawn, no need to draw the edge
    if (!drawn_nodes.has(edge.dst))
        return null;
    // If both source and destination are in the graph, draw edge as-is
    if (drawn_nodes.has(edge.src))
        return edge;

    // If immediate scope parent node is in the graph, redirect
    let scope_src = sdfg_state.nodes[edge.src].scope_entry;
    if (!drawn_nodes.has(scope_src))
        return null;

    // Clone edge for redirection, change source to parent
    let new_edge = Object.assign({}, edge);
    new_edge.src = scope_src;

    return new_edge;
}

function equals(a, b) {
     return JSON.stringify(a) === JSON.stringify(b);
}

// Layout SDFG elements (states, nodes, scopes, nested SDFGs)
function relayout_sdfg(ctx, sdfg) {
    let STATE_MARGIN = 4*LINEHEIGHT;

    // Layout the SDFG as a dagre graph
    let g = new dagre.graphlib.Graph();
    g.setGraph({});
    g.setDefaultEdgeLabel(function (u, v) { return {}; });

    // layout each state to get its size
    sdfg.nodes.forEach((state) => {
        let stateinfo = {};

        stateinfo.label = state.id;
        let state_g = null;
        if (state.attributes.is_collapsed) {
            stateinfo.width = ctx.measureText(stateinfo.label).width;
            stateinfo.height = LINEHEIGHT;
        }
        else {
            state_g = relayout_state(ctx, state, sdfg);
            stateinfo = calculateBoundingBox(state_g);
        }
        stateinfo.width += 2*STATE_MARGIN;
        stateinfo.height += 2*STATE_MARGIN;
        g.setNode(state.id, new elements.State({state: state,
                                                layout: stateinfo,
                                                graph: state_g}, state.id, sdfg));
    });

    sdfg.edges.forEach((edge, id) => {
        let label = edge.attributes.data.label;
        let textmetrics = ctx.measureText(label);
        g.setEdge(edge.src, edge.dst, new elements.Edge({ name: label, label: label, height: LINEHEIGHT,
                                                                width: textmetrics.width}, id, sdfg));
    });

    dagre.layout(g);

    // Annotate the sdfg with its layout info
    sdfg.nodes.forEach(function (state) {
        let gnode = g.node(state.id);
        state.attributes.layout = {};
        state.attributes.layout.x = gnode.x;
        state.attributes.layout.y = gnode.y;
        state.attributes.layout.width = gnode.width;
        state.attributes.layout.height = gnode.height;
    });

    sdfg.edges.forEach(function (edge) {
        let gedge = g.edge(edge.src, edge.dst);
        let bb = calculateEdgeBoundingBox(gedge);
        edge.attributes = {};
        edge.attributes.data = {};
        edge.attributes.data.label = gedge.label;
        edge.attributes.label = gedge.label;
        edge.attributes.layout = {};
        edge.attributes.layout.width = bb.width;
        edge.attributes.layout.height = bb.height;
        edge.attributes.layout.x = bb.x;
        edge.attributes.layout.y = bb.y;
        edge.attributes.layout.points = gedge.points;
    });

    // Offset node and edge locations to be in state margins
    sdfg.nodes.forEach((s, sid) => {
        if (s.attributes.is_collapsed)
            return;

        let state = g.node(sid);
        let topleft = state.topleft();
        elements.offset_state(s, state, {x: topleft.x + STATE_MARGIN,
                                         y: topleft.y + STATE_MARGIN});
    });

    return g;
}

function relayout_state(ctx, sdfg_state, sdfg) {
    // layout the state as a dagre graph
    let g = new dagre.graphlib.Graph({multigraph: true});

    // Set an object for the graph label
    g.setGraph({ranksep: 15});

    g.setDefaultEdgeLabel(function (u, v) { return {}; });

    // Add nodes to the graph. The first argument is the node id. The
    // second is metadata about the node (label, width, height),
    // which will be updated by dagre.layout (will add x,y).

    // Process nodes hierarchically
    let toplevel_nodes = sdfg_state.scope_dict[-1];
    let drawn_nodes = new Set();

    function layout_node(node) {
        let nested_g = null;
        let nodesize = calculateNodeSize(sdfg_state, node, ctx);
        node.attributes.layout = {};
        node.attributes.layout.width = nodesize.width;
        node.attributes.layout.height = nodesize.height;
        node.attributes.layout.label = node.label;
        node.attributes.layout.type = node.type;
        node.attributes.layout.in_connectors = node.attributes.in_connectors;
        node.attributes.layout.out_connectors = node.attributes.out_connectors;
        node.attributes.layout.properties = node.attributes;
        node.attributes.layout.sdfg = sdfg;
        node.attributes.layout.state = sdfg_state;

        // Recursively lay out nested SDFGs
        if (node.type === "NestedSDFG") {
            nested_g = relayout_sdfg(ctx, node.attributes.sdfg);
            let sdfginfo = calculateBoundingBox(nested_g);
            node.attributes.layout.width = sdfginfo.width + 2*LINEHEIGHT;
            node.attributes.layout.height = sdfginfo.height + 2*LINEHEIGHT;
        }

        // Dynamically create node type
        let obj = new elements[node.type]({node: node, graph: nested_g}, node.id, sdfg, sdfg_state.id);

        // Add connectors
        let i = 0;
        for (let cname of node.attributes.in_connectors) {
            let conn = new elements.Connector({name: cname, type: 'in'}, i, sdfg, node.id);
            obj.connectors.push(conn);
            i += 1;
        }
        i = 0;
        for (let cname of node.attributes.out_connectors) {
            let conn = new elements.Connector({name: cname, type: 'out'}, i, sdfg, node.id);
            obj.connectors.push(conn);
            i += 1;
        }

        g.setNode(node.id, obj);
        drawn_nodes.add(node.id.toString());

        // Recursively draw nodes
        if (node.id in sdfg_state.scope_dict) {
            if (node.attributes.is_collapsed)
                return;
            sdfg_state.scope_dict[node.id].forEach(function (nodeid) {
                let node = sdfg_state.nodes[nodeid];
                layout_node(node);
            });
        }
    }


    toplevel_nodes.forEach(function (nodeid) {
        let node = sdfg_state.nodes[nodeid];
        layout_node(node);
    });

    sdfg_state.edges.forEach((edge, id) => {
        edge = check_and_redirect_edge(edge, drawn_nodes, sdfg_state);
        if (!edge) return;

        let label = edge.attributes.data.label;
        console.assert(label != undefined);
        let textmetrics = ctx.measureText(label);

        // Inject layout information analogous to state nodes
        edge.attributes.layout = {
            label: label,
            width: textmetrics.width,
            height: LINEHEIGHT
        };
        g.setEdge(edge.src, edge.dst, new elements.Edge(edge.attributes.layout, id, sdfg, sdfg_state.id), id);
    });

    dagre.layout(g);


    sdfg_state.edges.forEach(function (edge, id) {
        edge = check_and_redirect_edge(edge, drawn_nodes, sdfg_state);
        if (!edge) return;
        let gedge = g.edge(edge.src, edge.dst, id);
        let bb = calculateEdgeBoundingBox(gedge);
        edge.attributes.layout.width = bb.width;
        edge.attributes.layout.height = bb.height;
        edge.width = bb.width;
        edge.height = bb.height;
        edge.x = bb.x;
        edge.y = bb.y;
        edge.attributes.layout.x = bb.x;
        edge.attributes.layout.y = bb.y;
        edge.attributes.layout.points = gedge.points;
    });

    // Layout connectors and nested SDFGs
    sdfg_state.nodes.forEach(function (node, id) {       
        let gnode = g.node(id);
        let topleft = gnode.topleft();
        
        // Offset nested SDFG
        if (node.type === "NestedSDFG") {

            elements.offset_sdfg(node.attributes.sdfg, gnode.data.graph, {
                x: topleft.x + LINEHEIGHT, 
                y: topleft.y + LINEHEIGHT
            });
        }
        // Connector management 
        let SPACING = LINEHEIGHT;  
        let iconn_length = (LINEHEIGHT + SPACING) * node.attributes.in_connectors.length - SPACING;
        let oconn_length = (LINEHEIGHT + SPACING) * node.attributes.out_connectors.length - SPACING;
        let iconn_x = gnode.x - iconn_length / 2.0 + LINEHEIGHT/2.0;
        let oconn_x = gnode.x - oconn_length / 2.0 + LINEHEIGHT/2.0;
       
        for (let c of gnode.connectors) {
            c.width = LINEHEIGHT;
            c.height = LINEHEIGHT;
            if (c.data.type === 'in') {
                c.x = iconn_x;
                iconn_x += LINEHEIGHT + SPACING;
                c.y = topleft.y;
            } else {
                c.x = oconn_x;
                oconn_x += LINEHEIGHT + SPACING;
                c.y = topleft.y + gnode.height;
            }
        }
    });

    return g;
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
        this.mousepos = null; // Last position of the mouse pointer
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
            this.relayout();

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
        let canvas = this.canvas;
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
    }

    // Re-layout graph and nested graphs
    relayout() {
        this.graph = relayout_sdfg(this.ctx, this.sdfg);

        // Set canvas background and size according to SDFG size
        this.canvas.style.backgroundColor = "#ffffff";
        let bb = calculateBoundingBox(this.graph);
        this.canvas.width = Math.min(Math.max(bb.width + 1000, this.canvas.width), 16384);
        this.canvas.height = Math.min(Math.max(bb.height + 1000, this.canvas.height), 16384);

        return this.graph;
    }

    // Render SDFG
    draw() {
        let ctx = this.ctx;
        let g = this.graph;
        let curx = this.canvas_manager.translation.x;
        let cury = this.canvas_manager.translation.y;
        let curw = this.container.width();
        let curh = this.container.height();

        this.on_pre_draw();

        elements.draw_sdfg(ctx, g, {x: curx, y: cury, w: curw, h: curh}, this.mousepos);

        this.on_post_draw();
    }

    on_pre_draw() { }

    on_post_draw() { }

    // Returns a dictionary of SDFG elements in a given rectangle. Used for
    // selection, rendering, localized transformations, etc.
    // The output is a dictionary of lists of dictionaries. The top-level keys are:
    // states, nodes, connectors, edges, isedges (interstate edges). For example:
    // {'states': [{sdfg: sdfg_name, state: 1}, ...], nodes: [sdfg: sdfg_name, state: 1, node: 5],
    //              edges: [], isedges: [], connectors: []}
    elements_in_rect(x, y, w, h) {
        let elements = {states: [], nodes: [], connectors: [],
                        edges: [], isedges: []};

        function traverse_recursive(sdfg, elements, x, y) {
            let sdfg_name = sdfg.attributes.name;
            sdfg.nodes.forEach(function (state, state_id) {
                if (isWithinBB(x, y, state.attributes.layout)) {
                    // Selected states
                    elements.states.push({sdfg: sdfg_name, id: state_id});

                    state.nodes.forEach(function (node, node_id) {
                        if (isWithinBB(x, y, node.attributes.layout)) {
                            // Selected nodes
                            elements.nodes.push({
                                sdfg: sdfg_name, state: state_id, id: node_id
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
                                traverse_recursive(node.sdfg, elements, x - node.x, y - node.y);
                        }
                    });

                    // Selected edges
                    state.edges.forEach((edge, edge_id) => {
                        if (isWithinBBEdge(x, y, edge.attributes.layout)) {
                            elements.edges.push({sdfg: sdfg_name, state: state_id, id: edge_id});
                        }
                    });
                }
            });

            // Selected inter-state edges
            sdfg.edges.forEach((edge, isedge_id) => {
                if (isWithinBBEdge(x, y, edge.attributes.layout)) {
                    elements.isedges.push({sdfg: sdfg_name, id: isedge_id});
                }
            });
        }

        // Traverse all SDFGs recursively
        traverse_recursive(this.sdfg, elements, x, y);

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
            this.mousepos = {x: comp_x_func(event), y: comp_y_func(event)};

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

                // Mark for redraw
                dirty = true;
                this.draw_async();
                return;
            } else if (evtype === 'mousemove' || event.targetTouches.length === 1) { // dragging
                let ev = (evtype === 'touchmove') ? event.touches[0] : event;
                if (evtype === 'touchmove') {
                    ev.buttons = 1;
                    ev.movementX = ev.clientX - this.drag_start.clientX;
                    ev.movementY = ev.clientY - this.drag_start.clientY;
                }


                if (ev.buttons & 1) {
                    // Only accept the primary (~left) mouse button as dragging source
                    let movement = [
                        ev.movementX,
                        ev.movementY
                    ];

                    this.canvas_manager.translate(...movement);

                    // Mark for redraw
                    dirty = true;
                    this.draw_async();
                    return;
                }

                this.drag_start = ev;
            }

        } else if (evtype === 'wheel') {
            this.canvas_manager.scale(-event.deltaY / 1000.0, comp_x_func(event), comp_y_func(event));
            dirty = true;
            this.draw_async();
            return;
        }

        // End of mouse-move/touch-based events


        // Find elements under cursor
        let elements = this.elements_in_rect(comp_x_func(event), comp_y_func(event), 0, 0);
        let clicked_states = elements.states;
        let clicked_nodes = elements.nodes;
        let clicked_edges = elements.edges
        let clicked_interstate_edges = elements.isedges;
        let clicked_connectors = elements.connectors;

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
        if (clicked_states.length == 0 && clicked_interstate_edges.length == 0 &&
            (evtype === 'click' || evtype === 'contextmenu')) {
            // Nothing was selected
            //let sstate = this.sdfg;
            //sstate.clearHighlights();
            this.diode.render_free_variables();
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

        if (evtype === "mousemove") {
            let sdfg = this.sdfg;

            // Position for tooltip
            let pos = {x: comp_x_func(event), y: comp_y_func(event)};

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
            this.draw_async();
            return;
        }

        if (evtype === "dblclick") {
            let sdfg = this.sdfg;
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
                this.relayout();
                this.draw_async();
            }
            return;
        }

        if (evtype === "contextmenu") {
            // Context menu was requested
            let spos = {x: event.x, y: event.y};
            let sdfg_name = this.sdfg.attributes.name;

            let cmenu = new ContextMenu();
            cmenu.addOption("Show transformations", x => {
                console.log("'Show transformations' was clicked");

                this.diode.project().request(['highlight-transformations-' + sdfg_name], x => {

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
                this.diode.project().request(['get-transformations-' + sdfg_name], x => {
                    console.log("get-transformations response: ", x);

                    let tmp = Object.values(x)[0];

                    // Create a sub-menu at the correct position


                    let submenu = new ContextMenu();

                    for (let y of tmp) {
                        submenu.addOption(y.opt_name, x => {
                            this.project().request(['apply-transformation-' + sdfg_name], x => {
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

        if (evtype !== "click")
            return;

        // Render properties asynchronously
        setTimeout(() => {
            // Get and render the properties from now on
            let sdfg = this.sdfg;
            let sdfg_name = sdfg.attributes.name;

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

                this.diode.renderProperties(propobj);
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