// This class groups node-drawing to reduce redundancies and non-global state.
class DrawNodeState {

    constructor(ctx, stateid) {
        this.ctx = ctx;
        this.stateid = stateid;

    }

    highlights() {

        return global_state.highlights.filter(x => x['state-id'] == this.stateid).map(x => x['node-id']);
    }

    nodeColor(nodeid) {
        if(this.highlights().filter(x => x == nodeid).length > 0) {
            return "red";
        }
        else {
            return "black";
        }
    }

    drawArrow(ctx, p1, p2, size) {
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

    drawArrayNode(node, nodeid) {
        let ctx = this.ctx;
        var topleft_x = node.x - node.width / 2.0;
        var topleft_y = node.y - node.height / 2.0;
        ctx.beginPath();
        ctx.moveTo(topleft_x + node.height / 2.0, topleft_y);
        ctx.arc(topleft_x + node.height / 2.0, topleft_y + node.height / 2.0, node.height / 2.0, 1.5 * Math.PI, 0.5 * Math.PI, true);
        ctx.lineTo(topleft_x + node.width - node.height, topleft_y + node.height);
        ctx.arc(topleft_x + node.width - node.height / 2.0, topleft_y + node.height / 2.0, node.height / 2.0, 0.5 * Math.PI, 1.5 * Math.PI, true);
        ctx.lineTo(topleft_x + node.height / 2.0, topleft_y);
        ctx.closePath();
        ctx.strokeStyle = this.nodeColor(nodeid);
        ctx.stroke();
        var textmetrics = ctx.measureText(node.label);
        ctx.fillText(node.label, node.x - textmetrics.width / 2.0, node.y + LINEHEIGHT / 2.0);
    }

    drawMapEntryNode(node, nodeid) {
        let ctx = this.ctx;
        var topleft_x = node.x - node.width / 2.0;
        var topleft_y = node.y - node.height / 2.0;
        ctx.beginPath();
        ctx.moveTo(topleft_x, topleft_y + node.height);
        ctx.lineTo(topleft_x + node.width, topleft_y + node.height);
        ctx.lineTo(topleft_x + node.width - node.height, topleft_y);
        ctx.lineTo(topleft_x + node.height, topleft_y);
        ctx.lineTo(topleft_x, topleft_y + node.height);
        ctx.closePath();
        ctx.strokeStyle = this.nodeColor(nodeid);
        ctx.stroke();
        var textmetrics = ctx.measureText(node.label);
        ctx.fillText(node.label, node.x - textmetrics.width / 2.0, node.y + LINEHEIGHT / 2.0);

        this.drawConnectors(node.in_connectors, topleft_x+node.height, topleft_y);
        this.drawConnectors(node.out_connectors, topleft_x+node.height, topleft_y+node.height - 2*LINEHEIGHT);
    }

    drawMapExitNode(node, nodeid) {
        let ctx = this.ctx;
        var topleft_x = node.x - node.width / 2.0;
        var topleft_y = node.y - node.height / 2.0;
        ctx.beginPath();
        ctx.moveTo(topleft_x, topleft_y);
        ctx.lineTo(topleft_x + node.width, topleft_y);
        ctx.lineTo(topleft_x + node.width - node.height, topleft_y + node.height);
        ctx.lineTo(topleft_x + node.height, topleft_y + node.height);
        ctx.lineTo(topleft_x, topleft_y);
        ctx.closePath();
        ctx.strokeStyle = this.nodeColor(nodeid);
        ctx.stroke();
        var textmetrics = ctx.measureText(node.label);
        ctx.fillText(node.label, node.x - textmetrics.width / 2.0, node.y + LINEHEIGHT / 2.0);

        this.drawConnectors(node.in_connectors, topleft_x+node.height, topleft_y);
        this.drawConnectors(node.out_connectors, topleft_x+node.height, topleft_y+node.height-2*LINEHEIGHT);
    }

    drawTaskletNode(node, nodeid) {
        var ctx = this.ctx;
        var topleft_x = node.x - node.width / 2.0;
        var topleft_y = node.y - node.height / 2.0;
        var hexseg = node.height / 3.0;
        ctx.beginPath();
        ctx.moveTo(topleft_x, topleft_y + hexseg);
        ctx.lineTo(topleft_x + hexseg, topleft_y);
        ctx.lineTo(topleft_x + node.width - hexseg, topleft_y);
        ctx.lineTo(topleft_x + node.width, topleft_y + hexseg);
        ctx.lineTo(topleft_x + node.width, topleft_y + 2 * hexseg);
        ctx.lineTo(topleft_x + node.width - hexseg, topleft_y + node.height);
        ctx.lineTo(topleft_x + hexseg, topleft_y + node.height);
        ctx.lineTo(topleft_x, topleft_y + 2 * hexseg);
        ctx.lineTo(topleft_x, topleft_y + 1 * hexseg);
        ctx.closePath();
        ctx.strokeStyle = this.nodeColor(nodeid);
        ctx.stroke();
        var textmetrics = ctx.measureText(node.label);
        ctx.fillText(node.label, node.x - textmetrics.width / 2.0, node.y + LINEHEIGHT / 2.0);
        this.drawConnectors(node.in_connectors, topleft_x+hexseg, topleft_y);
        this.drawConnectors(node.out_connectors, topleft_x+hexseg, topleft_y+node.height-2*LINEHEIGHT);
    }

    drawReduceNode(node, nodeid) {
        let ctx = this.ctx;
        var topleft_x = node.x - node.width / 2.0;
        var topleft_y = node.y - node.height / 2.0;

        ctx.beginPath();
        ctx.moveTo(topleft_x, topleft_y);
        ctx.lineTo(topleft_x + node.width / 2, topleft_y + node.height);
        ctx.lineTo(topleft_x + node.width, topleft_y);
        ctx.lineTo(topleft_x, topleft_y);
        ctx.closePath();
        ctx.strokeStyle = this.nodeColor(nodeid);
        ctx.stroke();
        var textmetrics = ctx.measureText(node.label);
        ctx.fillText(node.label, node.x - textmetrics.width / 2.0, node.y - node.height / 4.0 + LINEHEIGHT / 2.0);
    }

    drawConnectors(labels, topleft_x, topleft_y, connarea_width) {
        let next_topleft_x = topleft_x + 5;
        let next_topleft_y = topleft_y;
        var ctx = this.ctx;
        labels.forEach(function(label) {
            let labelwidth = ctx.measureText(label).width;
            ctx.beginPath();
            ctx.moveTo(next_topleft_x, next_topleft_y);
            ctx.lineTo(next_topleft_x + labelwidth, next_topleft_y);
            ctx.lineTo(next_topleft_x + labelwidth, next_topleft_y + 2 * LINEHEIGHT);
            ctx.lineTo(next_topleft_x, next_topleft_y + 2 * LINEHEIGHT);
            ctx.lineTo(next_topleft_x, next_topleft_y);
            ctx.closePath();
            ctx.strokeStyle = "black";
            ctx.stroke();
            ctx.fillText(label, next_topleft_x, next_topleft_y + 1.5*LINEHEIGHT);
            next_topleft_x += labelwidth + 10;
        });
    }

    draw_node(node, nodeid) {
        // TODO: add all node types here, leave rectangle as fallback
        if (node.type == "ArrayNode") {
            this.drawArrayNode(node, nodeid)
        }
        else if (node.type == "MapEntry") {
            this.drawMapEntryNode(node, nodeid)
        }
        else if (node.type == "MapExit") {
            this.drawMapExitNode(node, nodeid)
        }
        else if (node.type == "Tasklet") {
            this.drawTaskletNode(node, nodeid)
        }
        else if (node.type == "Reduce") {
            this.drawReduceNode(node, nodeid)
        }
        else {
            let ctx = this.ctx;
            var topleft_x = node.x - node.width / 2.0;
            var topleft_y = node.y - node.height / 2.0;
            ctx.beginPath();
            ctx.moveTo(topleft_x, topleft_y);
            ctx.lineTo(topleft_x + node.width, topleft_y);
            ctx.lineTo(topleft_x + node.width, topleft_y + node.height);
            ctx.lineTo(topleft_x, topleft_y + node.height);
            ctx.lineTo(topleft_x, topleft_y);
            ctx.closePath();
            ctx.strokeStyle = this.nodeColor(nodeid);
            ctx.stroke();
            ctx.fillText(node.label, node.x - node.width / 2, node.y);
        }
    }

    draw_edge(edge) {
        let ctx = this.ctx;
        ctx.beginPath();
        ctx.moveTo(edge.points[0].x, edge.points[0].y);
        for(let elem of edge.points) {
            ctx.lineTo(elem.x, elem.y)
        };
        ctx.strokeStyle = "black";
        ctx.stroke();
        if (edge.points.length < 2) return;
        this.drawArrow(ctx, edge.points[edge.points.length - 2], edge.points[edge.points.length - 1], 5);
        ctx.fillText(edge.label, edge.x - edge.width, edge.y);
    }

}