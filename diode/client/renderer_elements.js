export class SDFGElement {
    // Parent ID is the state ID, if relevant
    constructor(elem, elem_id, sdfg, parent_id = null) {
        this.data = elem;
        this.id = elem_id;
        this.parent_id = parent_id;
        this.sdfg = sdfg;
    }

    draw(ctx, highlighted, mousepos) {}

    label() {
        return this.data.attributes.label;
    }

    long_label() {
        return this.data.attributes.label;
    }

    // Produces HTML for a hover-tooltip
    tooltip() {
        return null;
    }
}

export class State extends SDFGElement {
    draw(ctx, highlighted, mousepos) {

    }

    get width() {
        return this.data.layout.width;
    }
    get height() {
        return this.data.layout.height;
    }
}

export class Node extends SDFGElement {
    draw(ctx, highlighted, mousepos) {

    }
}

export class Edge extends SDFGElement {
    draw(ctx, highlighted, mousepos) {

    }

    tooltip() {
        return this.label();
    }
}

export class Connector extends SDFGElement {
    draw(ctx, highlighted, mousepos) {

    }

    tooltip() {
        return this.label();
    }
}

export class AccessNode extends Node {
    draw(ctx, highlighted, mousepos) {

    }
}

export class EntryNode extends Node {
    draw(ctx, highlighted, mousepos) {

    }
}

export class ExitNode extends Node {
    draw(ctx, highlighted, mousepos) {

    }
}

export class MapEntry extends EntryNode { stroketype(ctx) { ctx.setLineDash([1, 0]); } }
export class MapExit extends ExitNode {  stroketype(ctx) { ctx.setLineDash([1, 0]); } }
export class ConsumeEntry extends EntryNode {  stroketype(ctx) { ctx.setLineDash([5, 3]); } }
export class ConsumeExit extends ExitNode {  stroketype(ctx) { ctx.setLineDash([5, 3]); } }

export class Tasklet extends Node {
    draw(ctx, highlighted, mousepos) {

    }
}

export class Reduce extends Node {
    draw(ctx, highlighted, mousepos) {

    }
}

export class NestedSDFG extends Node {
    draw(ctx, highlighted, mousepos) {

    }
}

//////////////////////////////////////////////////////

function drawHexagon(ctx, x, y, w, h) {
    var topleft_x = x - w / 2.0;
    var topleft_y = y - h / 2.0;
    var hexseg = h / 3.0;
    ctx.beginPath();
    ctx.moveTo(topleft_x, y);
    ctx.lineTo(topleft_x + hexseg, topleft_y);
    ctx.lineTo(topleft_x + w - hexseg, topleft_y);
    ctx.lineTo(topleft_x + w, y);
    ctx.lineTo(topleft_x + w - hexseg, topleft_y + h);
    ctx.lineTo(topleft_x + hexseg, topleft_y + h);
    ctx.lineTo(topleft_x, y);
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

function drawArrow(ctx, p1, p2, size) {
    "use strict";
    let _ctx = ctx;
    _ctx.save();
    // Rotate the context to point along the path
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let len = Math.sqrt(dx * dx + dy * dy);
    _ctx.translate(p2.x, p2.y);
    _ctx.rotate(Math.atan2(dy, dx));

    // arrowhead
    _ctx.beginPath();
    _ctx.moveTo(0, 0);
    _ctx.lineTo(-2 * size, -size);
    _ctx.lineTo(-2 * size, size);
    _ctx.closePath();
    _ctx.fill();
    _ctx.restore();
}


