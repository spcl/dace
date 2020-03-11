
class CanvasManager {
    // Manages translation and scaling of canvas rendering

    static counter() {
        return _canvas_manager_counter++;
    }
    constructor(ctx, renderer, canvas) {
        this.ctx = ctx;
        this.ctx.lod = true;
        this.canvas = canvas;
        this.anim_id = null;
        this.prev_time = null;
        this.drawables = [];
        this.renderer = renderer;
        this.indices = [];

        // Animation-related fields
        this.animating = false;
        this.animate_target = null;

        this.request_scale = false;
        this.scalef = 1.0;

        this._destroying = false;

        this.scale_origin = {x: 0, y: 0};

        this.contention = 0;

        this._svg = document.createElementNS("http://www.w3.org/2000/svg",'svg');

        this.user_transform = this._svg.createSVGMatrix();

        this.addCtxTransformTracking();
    }

    stopAnimation() {
        this.animating = false;
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
        this.stopAnimation();
        if(this.request_scale || this.contention > 0) {
            return;
        }
        this.contention++;
        this.request_scale = true;

        this.scale_origin.x = x;
        this.scale_origin.y = y;

        let sv = diff;
        let pt = this.svgPoint(this.scale_origin.x, this.scale_origin.y).matrixTransform(this.user_transform.inverse());
        this.user_transform = this.user_transform.translate(pt.x, pt.y);
        this.user_transform = this.user_transform.scale(sv, sv, 1, 0, 0, 0);
        this.scalef *= sv;
        this.user_transform = this.user_transform.translate(-pt.x, -pt.y);
        
        this.contention--;
    }

    // Sets the view to the square around the input rectangle
    set_view(rect) {
        this.stopAnimation();
        this.user_transform = this._svg.createSVGMatrix();
        let canvas_w = this.canvas.width;
        let canvas_h = this.canvas.height;
        if (canvas_w == 0 || canvas_h == 0)
            return;

        let scale = 1, tx = 0, ty = 0;
        if (rect.width > rect.height) {
            scale = canvas_w / rect.width;
            tx = -rect.x;
            ty = -rect.y - (rect.height/2) + (canvas_h / scale / 2);

            // Now other dimension does not fit, scale it as well
            if (rect.height * scale > canvas_h) {
                scale = canvas_h / rect.height;
                tx = -rect.x - (rect.width/2) + (canvas_w / scale / 2);
                ty = -rect.y;
            }
        } else {
            scale = canvas_h / rect.height;
            tx = -rect.x - (rect.width/2) + (canvas_w / scale / 2);
            ty = -rect.y;
        
            // Now other dimension does not fit, scale it as well
            if (rect.width * scale > canvas_w) {
                scale = canvas_w / rect.width;
                tx = -rect.x;
                ty = -rect.y - (rect.height/2) + (canvas_h / scale / 2);
            }
        }

        // Uniform scaling
        this.user_transform = this.user_transform.scale(scale, scale, 1, 0, 0, 0);
        this.user_transform = this.user_transform.translate(tx, ty);
        this.scale_origin = {x: 0, y: 0};
        this.scalef = 1.0;
    }

    translate(x, y) {
        this.stopAnimation();
        this.user_transform = this.user_transform.translate(x / this.user_transform.a, y / this.user_transform.d);
    }

    mapPixelToCoordsX(xpos) {
        return this.svgPoint(xpos, 0).matrixTransform(this.user_transform.inverse()).x;
    }

    mapPixelToCoordsY(ypos) {
        return this.svgPoint(0, ypos).matrixTransform(this.user_transform.inverse()).y;
    }

    noJitter(x) {
        x = parseFloat(x.toFixed(3));
        x = Math.round(x * 100) / 100;
        return x;
    }

    points_per_pixel() {
        // Since we are using uniform scaling, (bottom-top)/height and
        // (right-left)/width should be equivalent
        let left = this.mapPixelToCoordsX(0);
        let right = this.mapPixelToCoordsX(this.canvas.width);
        return (right - left) / this.canvas.width;
    }

    draw(now = null) {
        if(this._destroying)
            return;

        let dt = now - this.prev_time;
        if (!now || !this.prev_time)
            dt = null;
        if (now)
            this.prev_time = now;

        if(this.contention > 0) return;
        this.contention += 1;
        let ctx = this.ctx;

        // Clear with default transform
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.restore();

        if(this.request_scale && this.contention == 1) {
            // Reset the translation
            this.applyUserTransform();
            this.request_scale = false;
        }
        else
            this.applyUserTransform();

        this.renderer.draw(dt);
        this.contention -= 1;

        if (this.animating) {
            if (!this.animate_target)
                this.animating = false;
            this.draw_async();
        }
    }

    draw_async() {
        this.anim_id = window.requestAnimationFrame((now) => this.draw(now));
    }
}

function getQuadraticAngle(t, sx, sy, cp1x, cp1y, ex, ey) {
    let dx = 2*(1-t)*(cp1x-sx) + 2*t*(ex-cp1x);
    let dy = 2*(1-t)*(cp1y-sy) + 2*t*(ey-cp1y);
    return -Math.atan2(dx, dy) + 0.5*Math.PI;
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

function boundingBox(elements) {
    let bb = {x1: null, y1: null, x2: null, y2: null};

    elements.forEach(function (v) {
        let topleft = v.topleft();
        if (bb.x1 === null || topleft.x < bb.x1) bb.x1 = topleft.x;
        if (bb.y1 === null || topleft.y < bb.y1) bb.y1 = topleft.y;
        
        let x2 = v.x + v.width / 2.0;
        let y2 = v.y + v.height / 2.0;

        if (bb.x2 === null || x2 > bb.x2) bb.x2 = x2;
        if (bb.y2 === null || y2 > bb.y2) bb.y2 = y2;
    });

    return {x: bb.x1, y: bb.y1, width: bb.x2 - bb.x1, height: bb.y2 - bb.y1};
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
    let inconnsize = 2 * LINEHEIGHT * node.attributes.layout.in_connectors.length - LINEHEIGHT;
    let outconnsize = 2 * LINEHEIGHT * node.attributes.layout.out_connectors.length - LINEHEIGHT;
    let maxwidth = Math.max(labelsize, inconnsize, outconnsize);
    let maxheight = 2*LINEHEIGHT;
    maxheight += 4*LINEHEIGHT;

    let size = { width: maxwidth, height: maxheight }

    // add something to the size based on the shape of the node
    if (node.type === "AccessNode") {
        size.height -= 4*LINEHEIGHT;
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
    else if (node.type === "LibraryNode") {
        size.width += 2.0 * (size.height / 3.0);
        size.height /= 1.75;
    }
    else if (node.type === "Reduce") {
        size.height -= 4*LINEHEIGHT;
        size.width *= 2;
        size.height = size.width / 3.0;
    }
    else {
    }

    return size;
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
        g.setNode(state.id, new State({state: state,
                                                layout: stateinfo,
                                                graph: state_g}, state.id, sdfg));
    });

    sdfg.edges.forEach((edge, id) => {
        g.setEdge(edge.src, edge.dst, new Edge(edge.attributes.data, id, sdfg));
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
        // Convert from top-left to center
        bb.x += bb.width / 2.0;
        bb.y += bb.height / 2.0;

        gedge.x = bb.x;
        gedge.y = bb.y;
        gedge.width = bb.width;
        gedge.height = bb.height;
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
        offset_state(s, state, {x: topleft.x + STATE_MARGIN,
                                         y: topleft.y + STATE_MARGIN});
    });

    let bb = calculateBoundingBox(g);
    g.width = bb.width;
    g.height = bb.height;

    return g;
}

function relayout_state(ctx, sdfg_state, sdfg) {
    // layout the state as a dagre graph
    let g = new dagre.graphlib.Graph({multigraph: true});

    // Set an object for the graph label
    g.setGraph({ranksep: 30});

    g.setDefaultEdgeLabel(function (u, v) { return {}; });

    // Add nodes to the graph. The first argument is the node id. The
    // second is metadata about the node (label, width, height),
    // which will be updated by dagre.layout (will add x,y).

    // Process nodes hierarchically
    let toplevel_nodes = sdfg_state.scope_dict[-1];
    if (toplevel_nodes === undefined)
        toplevel_nodes = Object.keys(sdfg_state.nodes);
    let drawn_nodes = new Set();

    function layout_node(node) {
        let nested_g = null;
        node.attributes.layout = {};

        // Set connectors prior to computing node size
        node.attributes.layout.in_connectors = node.attributes.in_connectors;
        if ('is_collapsed' in node.attributes && node.attributes.is_collapsed && node.type !== "NestedSDFG")
            node.attributes.layout.out_connectors = find_exit_for_entry(sdfg_state.nodes, node).attributes.out_connectors;
        else
            node.attributes.layout.out_connectors = node.attributes.out_connectors;

        let nodesize = calculateNodeSize(sdfg_state, node, ctx);
        node.attributes.layout.width = nodesize.width;
        node.attributes.layout.height = nodesize.height;
        node.attributes.layout.label = node.label;

        // Recursively lay out nested SDFGs
        if (node.type === "NestedSDFG") {
            nested_g = relayout_sdfg(ctx, node.attributes.sdfg);
            let sdfginfo = calculateBoundingBox(nested_g);
            node.attributes.layout.width = sdfginfo.width + 2*LINEHEIGHT;
            node.attributes.layout.height = sdfginfo.height + 2*LINEHEIGHT;
        }

        // Dynamically create node type
        let obj = new SDFGElements[node.type]({node: node, graph: nested_g}, node.id, sdfg, sdfg_state.id);

        // Add input connectors
        let i = 0;
        for (let cname of node.attributes.layout.in_connectors) {
            let conn = new Connector({name: cname}, i, sdfg, node.id);
            obj.in_connectors.push(conn);
            i += 1;
        }

        // Add output connectors -- if collapsed, uses exit node connectors
        i = 0;
        for (let cname of node.attributes.layout.out_connectors) {
            let conn = new Connector({name: cname}, i, sdfg, node.id);
            obj.out_connectors.push(conn);
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
        g.setEdge(edge.src, edge.dst, new Edge(edge.attributes.data, id, sdfg, sdfg_state.id), id);
    });

    dagre.layout(g);


    // Layout connectors and nested SDFGs
    sdfg_state.nodes.forEach(function (node, id) {       
        let gnode = g.node(id);
        if (!gnode) return;
        let topleft = gnode.topleft();
        
        // Offset nested SDFG
        if (node.type === "NestedSDFG") {

            offset_sdfg(node.attributes.sdfg, gnode.data.graph, {
                x: topleft.x + LINEHEIGHT, 
                y: topleft.y + LINEHEIGHT
            });
        }
        // Connector management 
        let SPACING = LINEHEIGHT;  
        let iconn_length = (LINEHEIGHT + SPACING) * node.attributes.layout.in_connectors.length - SPACING;
        let oconn_length = (LINEHEIGHT + SPACING) * node.attributes.layout.out_connectors.length - SPACING;
        let iconn_x = gnode.x - iconn_length / 2.0 + LINEHEIGHT/2.0;
        let oconn_x = gnode.x - oconn_length / 2.0 + LINEHEIGHT/2.0;
       
        for (let c of gnode.in_connectors) {
            c.width = LINEHEIGHT;
            c.height = LINEHEIGHT;
            c.x = iconn_x;
            iconn_x += LINEHEIGHT + SPACING;
            c.y = topleft.y;
        }
        for (let c of gnode.out_connectors) {
            c.width = LINEHEIGHT;
            c.height = LINEHEIGHT;
            c.x = oconn_x;
            oconn_x += LINEHEIGHT + SPACING;
            c.y = topleft.y + gnode.height;
        }
    });

    sdfg_state.edges.forEach(function (edge, id) {
        edge = check_and_redirect_edge(edge, drawn_nodes, sdfg_state);
        if (!edge) return;
        let gedge = g.edge(edge.src, edge.dst, id);

        // Reposition first and last points according to connectors
        let src_conn = null, dst_conn = null;
        if (edge.src_connector) {
            let src_node = g.node(edge.src);
            let cindex = src_node.data.node.attributes.layout.out_connectors.indexOf(edge.src_connector);
            if (cindex >= 0) {
                gedge.points[0].x = src_node.out_connectors[cindex].x;
                gedge.points[0].y = src_node.out_connectors[cindex].y;
                src_conn = src_node.out_connectors[cindex];
            }
        }
        if (edge.dst_connector) {
            let dst_node = g.node(edge.dst);
            let cindex = dst_node.data.node.attributes.layout.in_connectors.indexOf(edge.dst_connector);
            if (cindex >= 0) {
                gedge.points[gedge.points.length - 1].x = dst_node.in_connectors[cindex].x;
                gedge.points[gedge.points.length - 1].y = dst_node.in_connectors[cindex].y;
                dst_conn = dst_node.in_connectors[cindex];
            }
        }

        let n = gedge.points.length - 1;
        if (src_conn !== null)
            gedge.points[0] = dagre.util.intersectRect(src_conn, gedge.points[n]);
        if (dst_conn !== null)
            gedge.points[n] = dagre.util.intersectRect(dst_conn, gedge.points[0]);

        if  (gedge.points.length == 3 && gedge.points[0].x == gedge.points[n].x)
            gedge.points = [gedge.points[0], gedge.points[n]];

        let bb = calculateEdgeBoundingBox(gedge);
        // Convert from top-left to center
        bb.x += bb.width / 2.0;
        bb.y += bb.height / 2.0;

        edge.width = bb.width;
        edge.height = bb.height;
        edge.x = bb.x;
        edge.y = bb.y;
        gedge.width = bb.width;
        gedge.height = bb.height;
        gedge.x = bb.x;
        gedge.y = bb.y;
    });


    return g;
}

class SDFGRenderer {
    constructor(sdfg, container, on_mouse_event = null) {
        // DIODE/SDFV-related fields
        this.sdfg = sdfg;

        // Rendering-related fields
        this.container = container;
        this.ctx = null;
        this.canvas = null;
        this.toolbar = null;
        this.menu = null;
        this.last_visible_elements = null;
        this.last_hovered_elements = null;
        this.last_clicked_elements = null;
        this.tooltip = null;
        this.tooltip_container = null;

        // Mouse-related fields
        this.mousepos = null; // Last position of the mouse pointer (in canvas coordinates)
        this.realmousepos = null; // Last position of the mouse pointer (in pixel coordinates)
        this.drag_start = null; // Null if the mouse/touch is not activated
        this.drag_second_start = null; // Null if two touch points are not activated
        this.external_mouse_handler = on_mouse_event;

        this.init_elements();
    }

    destroy() {
        try {
            if (this.menu)
                this.menu.destroy();
            this.canvas_manager.destroy();
            this.container.removeChild(this.canvas);
            this.container.removeChild(this.toolbar);
            this.container.removeChild(this.tooltip_container);
        } catch (ex) {
            // Do nothing
        }
    }

    // Initializes the DOM
    init_elements() {

        this.canvas = document.createElement('canvas');
        this.container.append(this.canvas);

        // Add buttons
        this.toolbar = document.createElement('div');
        this.toolbar.style = 'position:absolute; top:10px; left: 10px;';
        let d;

        // Menu bar
        try {
            ContextMenu;
            d = document.createElement('button');
            d.innerHTML = '<i class="material-icons">menu</i>';
            d.style = 'padding-bottom: 0px; user-select: none';
            let that = this;
            d.onclick = function () {
                if (that.menu && that.menu.visible()) {
                    that.menu.destroy();
                    return;
                }
                let rect = this.getBoundingClientRect();
                let cmenu = new ContextMenu();
                cmenu.addOption("Save view as PNG", x => that.save_as_png());
                cmenu.addOption("Save view as PDF", x => that.save_as_pdf());
                cmenu.addOption("Save all as PDF", x => that.save_as_pdf(true));
                that.menu = cmenu;
                cmenu.show(rect.left, rect.bottom);
            };
            d.title = 'Menu';
            this.toolbar.appendChild(d);
        } catch (ex) {}

        // Zoom to fit
        d = document.createElement('button');
        d.innerHTML = '<i class="material-icons">filter_center_focus</i>';
        d.style = 'padding-bottom: 0px; user-select: none';
        d.onclick = () => this.zoom_to_view();
        d.title = 'Zoom to fit SDFG';
        this.toolbar.appendChild(d);

        // Collapse all
        d = document.createElement('button');
        d.innerHTML = '<i class="material-icons">unfold_less</i>';
        d.style = 'padding-bottom: 0px; user-select: none';
        d.onclick = () => this.collapse_all();
        d.title = 'Collapse all elements';
        this.toolbar.appendChild(d);

        // Expand all
        d = document.createElement('button');
        d.innerHTML = '<i class="material-icons">unfold_more</i>';
        d.style = 'padding-bottom: 0px; user-select: none';
        d.onclick = () => this.expand_all();
        d.title = 'Expand all elements';
        this.toolbar.appendChild(d);

        this.container.append(this.toolbar);
        // End of buttons

        // Tooltip HTML container
        this.tooltip_container = document.createElement('div');
        this.tooltip_container.innerHTML = '';
        this.tooltip_container.className = 'tooltip';
        this.tooltip_container.onmouseover = () => this.tooltip_container.style.display = "none";
        this.container.appendChild(this.tooltip_container);

        this.ctx = this.canvas.getContext("2d");

        // Translation/scaling management
        this.canvas_manager = new CanvasManager(this.ctx, this, this.canvas);

        // Resize event for container
        let observer = new MutationObserver((mutations) => { this.onresize(); this.draw_async(); });
        observer.observe(this.container, { attributes: true });

        // Create the initial SDFG layout
        this.relayout();

        // Set mouse event handlers
        this.set_mouse_handlers();

        // Set initial zoom
        this.zoom_to_view();

        // Queue first render
        this.draw_async();
    }

    draw_async() {
        this.canvas_manager.draw_async();
    }

    set_sdfg(new_sdfg) {
        this.sdfg = new_sdfg;
        this.relayout();
        this.draw_async();
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
                let cancelled = this.on_mouse_event(x, comp_x, comp_y, evtype);
                if (cancelled)
                    return;
                x.stopPropagation();
                x.preventDefault();
            });
        }
    }

    onresize() {
        // Set canvas background and size
        this.canvas.style.backgroundColor = "#ffffff";
        this.canvas.style.width = '99%';
        this.canvas.style.height = '99%';
        this.canvas.width  = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
    }

    // Re-layout graph and nested graphs
    relayout() {
        this.graph = relayout_sdfg(this.ctx, this.sdfg);
        this.onresize();

        return this.graph;
    }

    // Change translation and scale such that the chosen elements
    // (or entire graph if null) is in view
    zoom_to_view(elements=null) {
        if (!elements)
            elements = this.graph.nodes().map(x => this.graph.node(x));
        
        let bb = boundingBox(elements);
        this.canvas_manager.set_view(bb);

        this.draw_async();
    }

    collapse_all() {
        this.for_all_sdfg_elements((otype, odict, obj) => {
            if ('is_collapsed' in obj.attributes && !obj.type.endsWith('Exit'))
                obj.attributes.is_collapsed = true;
        });
        this.relayout();
        this.draw_async();
    }

    expand_all() {
        this.for_all_sdfg_elements((otype, odict, obj) => {
            if ('is_collapsed' in obj.attributes && !obj.type.endsWith('Exit'))
                obj.attributes.is_collapsed = false;
        });
        this.relayout();
        this.draw_async();
    }

    // Save functions
    save(filename, contents) {
        var link = document.createElement('a');
        link.setAttribute('download', filename);
        link.href = contents;
        document.body.appendChild(link);

        // wait for the link to be added to the document
        window.requestAnimationFrame(function () {
            var event = new MouseEvent('click');
            link.dispatchEvent(event);
            document.body.removeChild(link);
        });
    }

    save_as_png() {
        this.save('sdfg.png', this.canvas.toDataURL('image/png'));
    }

    save_as_pdf(save_all=false) {
        let stream = blobStream();

        // Compute document size
        let curx = this.canvas_manager.mapPixelToCoordsX(0);
        let cury = this.canvas_manager.mapPixelToCoordsY(0);
        let size;
        if (save_all) {
            // Get size of entire graph
            let elements = this.graph.nodes().map(x => this.graph.node(x));
            let bb = boundingBox(elements);
            size = [bb.width, bb.height];
        } else {
            // Get size of current view
            let endx = this.canvas_manager.mapPixelToCoordsX(this.canvas.width);
            let endy = this.canvas_manager.mapPixelToCoordsY(this.canvas.height);
            let curw = endx - curx, curh = endy - cury;
            size = [curw, curh];
        }
        //

        let ctx = new canvas2pdf.PdfContext(stream, {
            size: size
        });
        let oldctx = this.ctx;
        this.ctx = ctx;
        this.ctx.lod = !save_all;
        this.ctx.pdf = true;
        // Center on saved region
        if (!save_all)
            this.ctx.translate(-curx, -cury);

        this.draw_async();

        ctx.stream.on('finish', () => {
            this.save('sdfg.pdf', ctx.stream.toBlobURL('application/pdf'));
            this.ctx = oldctx;
            this.draw_async();
        });
    }

    // Render SDFG
    draw(dt) {
        let ctx = this.ctx;
        let g = this.graph;
        let curx = this.canvas_manager.mapPixelToCoordsX(0);
        let cury = this.canvas_manager.mapPixelToCoordsY(0);
        let endx = this.canvas_manager.mapPixelToCoordsX(this.canvas.width);
        let endy = this.canvas_manager.mapPixelToCoordsY(this.canvas.height);
        let curw = endx - curx, curh = endy - cury;

        this.visible_rect = {x: curx, y: cury, w: curw, h: curh};

        this.on_pre_draw();

        draw_sdfg(this, ctx, g, this.mousepos);

        this.on_post_draw();
    }

    on_pre_draw() { }

    on_post_draw() {
        try {
            this.ctx.end();
        } catch (ex) {}
        
        if (this.tooltip) {
            let br = this.canvas.getBoundingClientRect();
            let pos = {x: this.realmousepos.x - br.x,
                       y: this.realmousepos.y - br.y};

            // Clear style and contents
            this.tooltip_container.style = '';
            this.tooltip_container.innerHTML = '';
            this.tooltip_container.style.display = 'block';
            
            // Invoke custom container         
            this.tooltip(this.tooltip_container);

            // Make visible near mouse pointer
            this.tooltip_container.style.top = pos.y + 'px';
            this.tooltip_container.style.left = (pos.x + 20) + 'px';
        } else {
            this.tooltip_container.style.display = 'none';
        }
    }

    // Returns a dictionary of SDFG elements in a given rectangle. Used for
    // selection, rendering, localized transformations, etc.
    // The output is a dictionary of lists of dictionaries. The top-level keys are:
    // states, nodes, connectors, edges, isedges (interstate edges). For example:
    // {'states': [{sdfg: sdfg_name, state: 1}, ...], nodes: [sdfg: sdfg_name, state: 1, node: 5],
    //              edges: [], isedges: [], connectors: []}
    elements_in_rect(x, y, w, h) {
        let elements = {
            states: [], nodes: [], connectors: [],
            edges: [], isedges: []
        };
        this.do_for_intersected_elements(x, y, w, h, (type, e, obj) => {
            e.obj = obj;
            elements[type].push(e);
        });
        return elements;
    }

    do_for_intersected_elements(x, y, w, h, func) {
        // Traverse nested SDFGs recursively
        function traverse_recursive(g, sdfg_name) {
            g.nodes().forEach(state_id => {
                let state = g.node(state_id);
                if (!state) return;
                
                if (state.intersect(x, y, w, h)) {
                    // States
                    func('states', {sdfg: sdfg_name, id: state_id}, state);

                    if (state.data.state.attributes.is_collapsed)
                        return;

                    let ng = state.data.graph;
                    if (!ng)
                        return;
                    ng.nodes().forEach(node_id => {
                        let node = ng.node(node_id);
                        if (node.intersect(x, y, w, h)) {
                            // Selected nodes
                            func('nodes', {sdfg: sdfg_name, state: state_id, id: node_id}, node);

                            // If nested SDFG, traverse recursively
                            if (node.data.node.type === "NestedSDFG")
                                traverse_recursive(node.data.graph, node.data.node.attributes.sdfg.attributes.name);
                        }
                        // Connectors
                        node.in_connectors.forEach((c, i) => {
                            if (c.intersect(x, y, w, h))
                                func('connectors', {sdfg: sdfg_name, state: state_id, node: node_id,
                                                    connector: i, conntype: "in"}, c);
                        });
                        node.out_connectors.forEach((c, i) => {
                            if (c.intersect(x, y, w, h))
                                func('connectors', {sdfg: sdfg_name, state: state_id, node: node_id,
                                                    connector: i, conntype: "out"}, c);
                        });
                    });

                    // Selected edges
                    ng.edges().forEach(edge_id => {
                        let edge = ng.edge(edge_id);
                        if (edge.intersect(x, y, w, h)) {
                            func('edges', {sdfg: sdfg_name, state: state_id, id: edge.id}, edge);
                        }
                    });
                }
            });

            // Selected inter-state edges
            g.edges().forEach(isedge_id => {
                let isedge = g.edge(isedge_id);
                if (isedge.intersect(x, y, w, h)) {
                    func('isedges', {sdfg: sdfg_name, id: isedge.id}, isedge);
                }
            });
        }

        // Start with top-level SDFG
        traverse_recursive(this.graph, this.sdfg.attributes.name);
    }

    for_all_sdfg_elements(func) {
        // Traverse nested SDFGs recursively
        function traverse_recursive(sdfg) {
            sdfg.nodes.forEach((state, state_id) => {
                // States
                func('states', {sdfg: sdfg, id: state_id}, state);

                state.nodes.forEach((node, node_id) => {
                    // Nodes
                    func('nodes', {sdfg: sdfg, state: state_id, id: node_id}, node);

                    // If nested SDFG, traverse recursively
                    if (node.type === "NestedSDFG")
                        traverse_recursive(node.attributes.sdfg);
                });

                // Edges
                state.edges.forEach((edge, edge_id) => {
                    func('edges', {sdfg: sdfg, state: state_id, id: edge_id}, edge);
                });
            });

            // Selected inter-state edges
            sdfg.edges.forEach((isedge, isedge_id) => {
                func('isedges', {sdfg: sdfg, id: isedge_id}, isedge);
            });
        }

        // Start with top-level SDFG
        traverse_recursive(this.sdfg);
    }

    for_all_elements(x, y, w, h, func) {
        // Traverse nested SDFGs recursively
        function traverse_recursive(g, sdfg_name) {
            g.nodes().forEach(state_id => {
                let state = g.node(state_id);
                if (!state) return;

                // States
                func('states', {sdfg: sdfg_name, id: state_id}, state, state.intersect(x, y, w, h));

                if (state.data.state.attributes.is_collapsed)
                    return;

                let ng = state.data.graph;
                if (!ng)
                    return;
                ng.nodes().forEach(node_id => {
                    let node = ng.node(node_id);
                    // Selected nodes
                    func('nodes', {sdfg: sdfg_name, state: state_id, id: node_id}, node, node.intersect(x, y, w, h));

                    // If nested SDFG, traverse recursively
                    if (node.data.node.type === "NestedSDFG")
                        traverse_recursive(node.data.graph, node.data.node.attributes.sdfg.attributes.name);

                    // Connectors
                    node.in_connectors.forEach((c, i) => {
                        func('connectors', {sdfg: sdfg_name, state: state_id, node: node_id,
                                            connector: i, conntype: "in"}, c, c.intersect(x, y, w, h));
                    });
                    node.out_connectors.forEach((c, i) => {
                        func('connectors', {sdfg: sdfg_name, state: state_id, node: node_id,
                                            connector: i, conntype: "out"}, c, c.intersect(x, y, w, h));
                    });
                });

                // Selected edges
                ng.edges().forEach(edge_id => {
                    let edge = ng.edge(edge_id);
                    func('edges', {sdfg: sdfg_name, state: state_id, id: edge.id}, edge, edge.intersect(x, y, w, h));
                });
            });

            // Selected inter-state edges
            g.edges().forEach(isedge_id => {
                let isedge = g.edge(isedge_id);
                func('isedges', {sdfg: sdfg_name, id: isedge.id}, isedge, isedge.intersect(x, y, w, h));
            });
        }

        // Start with top-level SDFG
        traverse_recursive(this.graph, this.sdfg.attributes.name);
    }

    on_mouse_event(event, comp_x_func, comp_y_func, evtype="click") {
        let dirty = false; // Whether to redraw at the end

        if (evtype === "mousedown" || evtype === "touchstart") {
            this.drag_start = event;
        } else if (evtype === "mouseup") {
            this.drag_start = null;
        } else if (evtype === "touchend") {
            if (event.touches.length == 0)
                this.drag_start = null;
            else
                this.drag_start = event;
        } else if (evtype === "mousemove") {
            this.mousepos = {x: comp_x_func(event), y: comp_y_func(event)};
            this.realmousepos = {x: event.clientX, y: event.clientY};

            if (this.drag_start && event.buttons & 1) {
                // Only accept the primary mouse button as dragging source
                this.canvas_manager.translate(event.movementX, event.movementY);

                // Mark for redraw
                dirty = true;
                this.draw_async();
                return false;
            } else {
                this.drag_start = null;
                if (event.buttons & 1)
                    return true; // Don't stop propagation
            }
        } else if (evtype === "touchmove") {
            if (this.drag_start.touches.length != event.touches.length) {
                // Different number of touches, ignore and reset drag_start
                this.drag_start = event;
            } else if (event.touches.length == 1) { // Move/drag
                this.canvas_manager.translate(event.touches[0].clientX - this.drag_start.touches[0].clientX,
                                              event.touches[0].clientY - this.drag_start.touches[0].clientY);
                this.drag_start = event;

                // Mark for redraw
                dirty = true;
                this.draw_async();
                return false;
            } else if (event.touches.length == 2) {
                // Find relative distance between two touches before and after.
                // Then, center and zoom to their midpoint.
                let touch1 = this.drag_start.touches[0];
                let touch2 = this.drag_start.touches[1];
                let x1 = touch1.clientX, x2 = touch2.clientX;
                let y1 = touch1.clientY, y2 = touch2.clientY;
                let oldCenter = [(x1 + x2) / 2.0, (y1 + y2) / 2.0];
                let initialDistance = Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
                x1 = event.touches[0].clientX; x2 = event.touches[1].clientX;
                y1 = event.touches[0].clientY; y2 = event.touches[1].clientY;
                let currentDistance = Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
                let newCenter = [(x1 + x2) / 2.0, (y1 + y2) / 2.0];

                // First, translate according to movement of center point
                this.canvas_manager.translate(newCenter[0] - oldCenter[0],
                                              newCenter[1] - oldCenter[1]);
                // Then scale
                this.canvas_manager.scale(currentDistance / initialDistance,
                                          newCenter[0], newCenter[1]);

                this.drag_start = event;

                // Mark for redraw
                dirty = true;
                this.draw_async();
                return false;
            }
        } else if (evtype === "wheel") {
            // Get physical x,y coordinates (rather than canvas coordinates)
            let br = this.canvas.getBoundingClientRect();
            let x = event.clientX - br.x;
            let y = event.clientY - br.y;
            this.canvas_manager.scale(event.deltaY > 0 ? 0.9 : 1.1, x, y);
            dirty = true;
        }
        // End of mouse-move/touch-based events


        if (!this.mousepos)
            return true;

        // Find elements under cursor
        let elements = this.elements_in_rect(this.mousepos.x, this.mousepos.y, 0, 0);
        let clicked_states = elements.states;
        let clicked_nodes = elements.nodes;
        let clicked_edges = elements.edges;
        let clicked_interstate_edges = elements.isedges;
        let clicked_connectors = elements.connectors;
        let total_elements = clicked_states.length + clicked_nodes.length + clicked_edges.length +
            clicked_interstate_edges.length + clicked_connectors.length;
        let foreground_elem = null, fg_surface = -1;

        // Find the top-most element under mouse cursor (the one with the smallest dimensions)
        for (let category of [clicked_states, clicked_interstate_edges, 
                              clicked_nodes, clicked_edges]) {
            for (let i = 0; i < category.length; i++) {
                let s = category[i].obj.width * category[i].obj.height;
                if (fg_surface < 0 || s < fg_surface) {
                        fg_surface = s;
                        foreground_elem = category[i].obj;
                }
            }
        }
        
        // Change mouse cursor accordingly
        if (total_elements > 0)
            this.canvas.style.cursor = 'pointer';
        else
            this.canvas.style.cursor = 'auto';

        this.tooltip = null;
        this.last_hovered_elements = elements;

        // Hovered elements get colored green (if they are not already colored)
        this.for_all_elements(this.mousepos.x, this.mousepos.y, 0, 0, (type, e, obj, intersected) => {
            if (intersected && obj.stroke_color === null)
                obj.stroke_color = 'green';
            else if(!intersected && obj.stroke_color === 'green')
                obj.stroke_color = null;
        });

        if (evtype === "mousemove") {
            // TODO: Draw only if elements have changed
            dirty = true;
        }

        if (evtype === "dblclick") {
            let sdfg = (foreground_elem ? foreground_elem.sdfg : null);
            let sdfg_elem = null;
            if (foreground_elem instanceof State)
                sdfg_elem = foreground_elem.data.state;
            else if (foreground_elem instanceof Node) {
                sdfg_elem = foreground_elem.data.node;

                // If a scope exit node, use entry instead
                if (sdfg_elem.type.endsWith("Exit"))
                    sdfg_elem = sdfg.nodes[foreground_elem.parent_id].nodes[sdfg_elem.scope_entry];
            } else
                sdfg_elem = null;

            // Toggle collapsed state
            if (sdfg_elem && 'is_collapsed' in sdfg_elem.attributes) {
                sdfg_elem.attributes.is_collapsed = !sdfg_elem.attributes.is_collapsed;

                // Re-layout SDFG
                this.relayout();
                dirty = true;
            }
        }

        if (this.external_mouse_handler)
            dirty |= this.external_mouse_handler(evtype, event, {x: comp_x_func(event), y: comp_y_func(event)}, elements,
                                                 this, foreground_elem);

        if (dirty)
            this.draw_async();

        return false;
    }
}

window.SDFGRenderer = SDFGRenderer;
