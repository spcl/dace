class SDFGElement {
    // Parent ID is the state ID, if relevant
    constructor(elem, elem_id, sdfg, parent_id = null) {
        this.data = elem;
        this.id = elem_id;
        this.parent_id = parent_id;
        this.sdfg = sdfg;
        this.in_connectors = [];
        this.out_connectors = [];
        this.stroke_color = null;
        this.set_layout();
    }

    set_layout() {
        // dagre does not work well with properties, only fields
        this.width = this.data.layout.width;
        this.height = this.data.layout.height;
    }

    draw(renderer, ctx, mousepos) {}

    attributes() {
        return this.data.attributes;
    }

    label() {
        return this.data.label;
    }

    long_label() {
        return this.label();
    }

    // Produces HTML for a hover-tooltip
    tooltip() {
        return null;
    }

    topleft() {
        return {x: this.x - this.width / 2, y: this.y - this.height / 2};
    }

    strokeStyle() {
        if (this.stroke_color)
            return this.stroke_color;
        return "black";
    }

    // General bounding-box intersection function. Returns true iff point or rectangle intersect element.
    intersect(x, y, w = 0, h = 0) {
        if (w == 0 || h == 0) {  // Point-element intersection
            return (x >= this.x - this.width / 2.0) &&
                   (x <= this.x + this.width / 2.0) &&
                   (y >= this.y - this.height / 2.0) &&
                   (y <= this.y + this.height / 2.0);
        } else {                 // Box-element intersection
            return (x <= this.x + this.width / 2.0) &&
                    (x + w >= this.x - this.width / 2.0) &&
                    (y <= this.y + this.height / 2.0) &&
                    (y + h >= this.y - this.height / 2.0);
        }
    }
}

class State extends SDFGElement {
    draw(renderer, ctx, mousepos) {
        let topleft = this.topleft();

        ctx.fillStyle = "#deebf7";
        ctx.fillRect(topleft.x, topleft.y, this.width, this.height);
        ctx.fillStyle = "#000000";

        ctx.fillText(this.label(), topleft.x, topleft.y + LINEHEIGHT);

        // If this state is selected or hovered
        if (this.stroke_color) {
            ctx.strokeStyle = this.strokeStyle();
            ctx.strokeRect(topleft.x, topleft.y, this.width, this.height);
        }

        // If collapsed, draw a "+" sign in the middle
        if (this.data.state.attributes.is_collapsed) {
            ctx.beginPath();
            ctx.moveTo(this.x, this.y - LINEHEIGHT);
            ctx.lineTo(this.x, this.y + LINEHEIGHT);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(this.x - LINEHEIGHT, this.y);
            ctx.lineTo(this.x + LINEHEIGHT, this.y);
            ctx.stroke();
        }

        ctx.strokeStyle = "black";
    }

    attributes() {
        return this.data.state.attributes;
    }

    label() {
        return this.data.state.label;
    }
}

class Node extends SDFGElement {
    draw(renderer, ctx, mousepos) {
        let topleft = this.topleft();
        ctx.fillStyle = "white";
        ctx.fillRect(topleft.x, topleft.y, this.width, this.height);
        ctx.strokeStyle = this.strokeStyle();
        ctx.strokeRect(topleft.x, topleft.y, this.width, this.height);
        ctx.fillStyle = "black";
        let textw = ctx.measureText(this.label()).width;
        ctx.fillText(this.label(), this.x - textw/2, this.y + LINEHEIGHT/4);
    }

    label() {
        return this.data.node.label;
    }

    attributes() {
        return this.data.node.attributes;
    }

    set_layout() {
        this.width = this.data.node.attributes.layout.width;
        this.height = this.data.node.attributes.layout.height;
    }
}

class Edge extends SDFGElement {
    draw(renderer, ctx, mousepos) {
        let edge = this;

        ctx.beginPath();
        ctx.moveTo(edge.points[0].x, edge.points[0].y);

        let i;
        for (i = 1; i < edge.points.length - 2; i++) {
            let xm = (edge.points[i].x + edge.points[i + 1].x) / 2.0;
            let ym = (edge.points[i].y + edge.points[i + 1].y) / 2.0;
            ctx.quadraticCurveTo(edge.points[i].x, edge.points[i].y, xm, ym);
        }
        ctx.quadraticCurveTo(edge.points[i].x, edge.points[i].y,
                             edge.points[i+1].x, edge.points[i+1].y);

        let style = this.strokeStyle();
        if (style !== 'black')
            renderer.tooltip = this.tooltip();
        if (this.parent_id == null && style === 'black') {  // Interstate edge
            style = 'blue';
        }
        ctx.fillStyle = ctx.strokeStyle = style;

        // CR edges have dashed lines
        if (this.parent_id != null && this.data.attributes.wcr != null)
            ctx.setLineDash([2, 2]);
        else
            ctx.setLineDash([1, 0]);
        

        ctx.stroke();

        ctx.setLineDash([1, 0]);

        if (edge.points.length < 2)
            return;
        drawArrow(ctx, edge.points[edge.points.length - 2], edge.points[edge.points.length - 1], 3);

        ctx.fillStyle = "black";
        ctx.strokeStyle = "black";
    }

    tooltip() {
        return this.label();
    }

    set_layout() {
        this.width = this.data.width;
        this.height = this.data.height;
    }

    intersect(x, y, w = 0, h = 0) {
        // First, check bounding box
        if(!super.intersect(x, y, w, h))
            return false;

        // Then (if point), check distance from line
        if (w == 0 || h == 0) {
            for (let i = 0; i < this.points.length - 1; i++) {
                let dist = ptLineDistance({x: x, y: y}, this.points[i], this.points[i + 1]);
                if (dist <= 5.0)
                    return true;
            }
            return false;
        }
        return true;
    }
}

class Connector extends SDFGElement {
    draw(renderer, ctx, mousepos) {
        let topleft = this.topleft();
        ctx.beginPath();
        drawEllipse(ctx, topleft.x, topleft.y, this.width, this.height);
        ctx.closePath();
        ctx.strokeStyle = this.strokeStyle();
        ctx.stroke();
        ctx.fillStyle = "#f0fdff";
        ctx.fill();
        ctx.fillStyle = "black";
        ctx.strokeStyle = "black";
        if (this.strokeStyle() !== 'black')
            renderer.tooltip = this.tooltip();
    }

    attributes() {
        return {};
    }

    set_layout() { }

    label() { return this.data.name; }

    tooltip() {
        return this.label();
    }
}

class AccessNode extends Node {
    draw(renderer, ctx, mousepos) {
        let topleft = this.topleft();
        ctx.beginPath();
        drawEllipse(ctx, topleft.x, topleft.y, this.width, this.height);
        ctx.closePath();
        ctx.strokeStyle = this.strokeStyle();

        let nodedesc = this.sdfg.attributes._arrays[this.data.node.attributes.data];
        // Streams have dashed edges
        if (nodedesc.type === "Stream") {
            ctx.setLineDash([5, 3]);
        } else {
            ctx.setLineDash([1, 0]);
        }

        if (nodedesc.attributes.transient === false) {
            ctx.lineWidth = 3.0;
        } else {
            ctx.lineWidth = 1.0;
        }


        ctx.stroke();
        ctx.lineWidth = 1.0;
        ctx.setLineDash([1, 0]);
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.fillStyle = "black";
        var textmetrics = ctx.measureText(this.label());
        ctx.fillText(this.label(), this.x - textmetrics.width / 2.0, this.y + LINEHEIGHT / 4.0);
    }
}

class ScopeNode extends Node {
    draw(renderer, ctx, mousepos) {
        if (this.data.node.attributes.is_collapsed) {
            drawHexagon(ctx, this.x, this.y, this.width, this.height);
        } else {
            let topleft = this.topleft();
            drawTrapezoid(ctx, this.topleft(), this, this.scopeend());
        }
        ctx.strokeStyle = this.strokeStyle();

        // Consume scopes have dashed edges
        if (this.data.node.type.startsWith("Consume"))
            ctx.setLineDash([5, 3]);
        else
            ctx.setLineDash([1, 0]);


        ctx.stroke();
        ctx.setLineDash([1, 0]);
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.fillStyle = "black";
        var textmetrics = ctx.measureText(this.label());
        ctx.fillText(this.label(), this.x - textmetrics.width / 2.0, this.y + LINEHEIGHT / 2.0);
    }
}

class EntryNode extends ScopeNode {
    scopeend() { return false; }
}

class ExitNode extends ScopeNode {
    scopeend() { return true; }
}

class MapEntry extends EntryNode { stroketype(ctx) { ctx.setLineDash([1, 0]); } }
class MapExit extends ExitNode {  stroketype(ctx) { ctx.setLineDash([1, 0]); } }
class ConsumeEntry extends EntryNode {  stroketype(ctx) { ctx.setLineDash([5, 3]); } }
class ConsumeExit extends ExitNode {  stroketype(ctx) { ctx.setLineDash([5, 3]); } }

class EmptyTasklet extends Node {
    draw(renderer, ctx, mousepos) {
        // Do nothing
    }
}

class Tasklet extends Node {
    draw(renderer, ctx, mousepos) {
        let topleft = this.topleft();
        drawOctagon(ctx, topleft, this.width, this.height);
        ctx.strokeStyle = this.strokeStyle();
        ctx.stroke();
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.fillStyle = "black";
        let textmetrics = ctx.measureText(this.label());
        ctx.fillText(this.label(), this.x - textmetrics.width / 2.0, this.y + LINEHEIGHT / 2.0);
    }
}

class Reduce extends Node {
    draw(renderer, ctx, mousepos) {
        let topleft = this.topleft();
        ctx.beginPath();
        ctx.moveTo(topleft.x, topleft.y);
        ctx.lineTo(topleft.x + this.width / 2, topleft.y + this.height);
        ctx.lineTo(topleft.x + this.width, topleft.y);
        ctx.lineTo(topleft.x, topleft.y);
        ctx.closePath();
        ctx.strokeStyle = this.strokeStyle();
        ctx.stroke();
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.fillStyle = "black";
        let textmetrics = ctx.measureText(this.label());
        ctx.fillText(this.label(), this.x - textmetrics.width / 2.0, this.y - this.height / 4.0 + LINEHEIGHT / 2.0);
    }
}

class NestedSDFG extends Node {
    draw(renderer, ctx, mousepos) {
        if (this.data.node.attributes.is_collapsed) {
            let topleft = this.topleft();
            drawOctagon(ctx, topleft, this.width, this.height);
            ctx.strokeStyle = this.strokeStyle();
            ctx.stroke();
            drawOctagon(ctx, {x: topleft.x + 2.5, y: topleft.y + 2.5}, this.width - 5, this.height - 5);
            ctx.strokeStyle = this.strokeStyle();
            ctx.stroke();
            ctx.fillStyle = 'white';
            ctx.fill();
            ctx.fillStyle = 'black';
            let label = this.data.node.attributes.label;
            let textmetrics = ctx.measureText(label);
            ctx.fillText(label, this.x - textmetrics.width / 2.0, this.y + LINEHEIGHT / 4.0);
            return;
        }

        // Draw square around nested SDFG
        super.draw(renderer, ctx, mousepos);

        // Draw nested graph
        draw_sdfg(renderer, ctx, this.data.graph, mousepos);
    }

    set_layout() { 
        if (this.data.node.attributes.is_collapsed) {
            let labelsize = this.data.node.attributes.label.length * LINEHEIGHT * 0.8;
            let inconnsize = 2 * LINEHEIGHT * this.data.node.attributes.in_connectors.length - LINEHEIGHT;
            let outconnsize = 2 * LINEHEIGHT * this.data.node.attributes.out_connectors.length - LINEHEIGHT;
            let maxwidth = Math.max(labelsize, inconnsize, outconnsize);
            let maxheight = 2*LINEHEIGHT;
            maxheight += 4*LINEHEIGHT;

            let size = { width: maxwidth, height: maxheight };
            size.width += 2.0 * (size.height / 3.0);
            size.height /= 1.75;

            this.width = size.width;
            this.height = size.height;
        } else {
            this.width = this.data.node.attributes.layout.width;
            this.height = this.data.node.attributes.layout.height;
        }
    }


    label() { return ""; }
}

//////////////////////////////////////////////////////

// Draw an entire SDFG
function draw_sdfg(renderer, ctx, sdfg_dagre, mousepos) {
    // Render state machine
    let g = sdfg_dagre;
    g.nodes().forEach( v => { g.node(v).draw(renderer, ctx, mousepos); });
    g.edges().forEach( e => { g.edge(e).draw(renderer, ctx, mousepos); });

    visible_rect = renderer.visible_rect;

    // Render each visible state's contents
    g.nodes().forEach( v => {
        let node = g.node(v);

        // Skip invisible states
        if (!node.intersect(visible_rect.x, visible_rect.y, visible_rect.w, visible_rect.h))
            return;

        let ng = node.data.graph;
        let layout = node.data.state.attributes.layout;

        
        
        if (!node.data.state.attributes.is_collapsed && ng)
        {
            ng.nodes().forEach(v => {
                let n = ng.node(v);

                if (!n.intersect(visible_rect.x, visible_rect.y, visible_rect.w, visible_rect.h))
                    return;

                n.draw(renderer, ctx, mousepos);
                n.in_connectors.forEach(c => { c.draw(renderer, ctx, mousepos); });
                n.out_connectors.forEach(c => { c.draw(renderer, ctx, mousepos); });
            });
            ng.edges().forEach(e => {
                let edge = ng.edge(e);
                if (!edge.intersect(visible_rect.x, visible_rect.y, visible_rect.w, visible_rect.h))
                    return;
                ng.edge(e).draw(renderer, ctx, mousepos);
            });
        }
    });
}

// Translate an SDFG by a given offset
function offset_sdfg(sdfg, sdfg_graph, offset) {
    sdfg.nodes.forEach((state, id) => {
        let g = sdfg_graph.node(id);
        g.x += offset.x;
        g.y += offset.y;
        if (!state.attributes.is_collapsed)
            offset_state(state, g, offset);
    });
    sdfg.edges.forEach((e, eid) => {
        let edge = sdfg_graph.edge(e.src, e.dst);
        edge.x += offset.x;
        edge.y += offset.y;
        edge.points.forEach((p) => {
            p.x += offset.x;
            p.y += offset.y;
        });
    });
}

// Translate nodes, edges, and connectors in a given SDFG state by an offset
function offset_state(state, state_graph, offset) {
    let drawn_nodes = new Set();
    
    state.nodes.forEach((n, nid) => {
        let node = state_graph.data.graph.node(nid);
        if (!node) return;
        drawn_nodes.add(nid.toString());

        node.x += offset.x;
        node.y += offset.y;
        node.in_connectors.forEach(c => {
            c.x += offset.x;
            c.y += offset.y;
        });
        node.out_connectors.forEach(c => {
            c.x += offset.x;
            c.y += offset.y;
        });

        if (node.data.node.type === 'NestedSDFG')
            offset_sdfg(node.data.node.attributes.sdfg, node.data.graph, offset);
    });
    state.edges.forEach((e, eid) => {
        e = check_and_redirect_edge(e, drawn_nodes, state);
        if (!e) return;
        let edge = state_graph.data.graph.edge(e.src, e.dst, eid);
        if (!edge) return;
        edge.x += offset.x;
        edge.y += offset.y;
        edge.points.forEach((p) => {
            p.x += offset.x;
            p.y += offset.y;
        });
    });
}


///////////////////////////////////////////////////////

function drawHexagon(ctx, x, y, w, h, offset) {
    let topleft = {x: x - w / 2.0, y: y - h / 2.0};
    let hexseg = h / 3.0;
    ctx.beginPath();
    ctx.moveTo(topleft.x, y);
    ctx.lineTo(topleft.x + hexseg, topleft.y);
    ctx.lineTo(topleft.x + w - hexseg, topleft.y);
    ctx.lineTo(topleft.x + w, y);
    ctx.lineTo(topleft.x + w - hexseg, topleft.y + h);
    ctx.lineTo(topleft.x + hexseg, topleft.y + h);
    ctx.lineTo(topleft.x, y);
    ctx.closePath();
}

function drawOctagon(ctx, topleft, width, height) {
    let octseg = height / 3.0;
    ctx.beginPath();
    ctx.moveTo(topleft.x, topleft.y + octseg);
    ctx.lineTo(topleft.x + octseg, topleft.y);
    ctx.lineTo(topleft.x + width - octseg, topleft.y);
    ctx.lineTo(topleft.x + width, topleft.y + octseg);
    ctx.lineTo(topleft.x + width, topleft.y + 2 * octseg);
    ctx.lineTo(topleft.x + width - octseg, topleft.y + height);
    ctx.lineTo(topleft.x + octseg, topleft.y + height);
    ctx.lineTo(topleft.x, topleft.y + 2 * octseg);
    ctx.lineTo(topleft.x, topleft.y + 1 * octseg);
    ctx.closePath();
}

// Adapted from https://stackoverflow.com/a/2173084/6489142
function drawEllipse(ctx, x, y, w, h) {
    var kappa = .5522848,
    ox = (w / 2) * kappa, // control point offset horizontal
    oy = (h / 2) * kappa, // control point offset vertical
    xe = x + w,           // x-end
    ye = y + h,           // y-end
    xm = x + w / 2,       // x-middle
    ym = y + h / 2;       // y-middle

    ctx.moveTo(x, ym);
    ctx.bezierCurveTo(x, ym - oy, xm - ox, y, xm, y);
    ctx.bezierCurveTo(xm + ox, y, xe, ym - oy, xe, ym);
    ctx.bezierCurveTo(xe, ym + oy, xm + ox, ye, xm, ye);
    ctx.bezierCurveTo(xm - ox, ye, x, ym + oy, x, ym);
}

function drawArrow(ctx, p1, p2, size, offset) {
    ctx.save();
    // Rotate the context to point along the path
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    ctx.translate(p2.x, p2.y);
    ctx.rotate(Math.atan2(dy, dx));

    // arrowhead
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(-2 * size, -size);
    ctx.lineTo(-2 * size, size);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
}

function drawTrapezoid(ctx, topleft, node, inverted=false) {
    ctx.beginPath();
    if (inverted) {
        ctx.moveTo(topleft.x, topleft.y);
        ctx.lineTo(topleft.x + node.width, topleft.y);
        ctx.lineTo(topleft.x + node.width - node.height, topleft.y + node.height);
        ctx.lineTo(topleft.x + node.height, topleft.y + node.height);
        ctx.lineTo(topleft.x, topleft.y);
    } else {
        ctx.moveTo(topleft.x, topleft.y + node.height);
        ctx.lineTo(topleft.x + node.width, topleft.y + node.height);
        ctx.lineTo(topleft.x + node.width - node.height, topleft.y);
        ctx.lineTo(topleft.x + node.height, topleft.y);
        ctx.lineTo(topleft.x, topleft.y + node.height);
    }
    ctx.closePath();
}

// Returns the distance from point p to line defined by two points (line1, line2)
function ptLineDistance(p, line1, line2) {
    let dx = (line2.x - line1.x);
    let dy = (line2.y - line1.y);
    let res = dy * p.x - dx * p.y + line2.x * line1.y - line2.y * line1.x;

    return Math.abs(res) / Math.sqrt(dy*dy + dx*dx);
}

var SDFGElements = {SDFGElement: SDFGElement, State: State, Node: Node,Edge: Edge, Connector: Connector, AccessNode: AccessNode,
                    ScopeNode: ScopeNode, EntryNode: EntryNode, ExitNode: ExitNode, MapEntry: MapEntry, MapExit: MapExit,
                    ConsumeEntry: ConsumeEntry, ConsumeExit: ConsumeExit, EmptyTasklet: EmptyTasklet, Tasklet: Tasklet, Reduce: Reduce,
                    NestedSDFG: NestedSDFG};
                    
// Save as globals
Object.keys(SDFGElements).forEach(function(elem) {
    window[elem] = SDFGElements[elem];
});