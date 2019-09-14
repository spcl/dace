// This class groups node-drawing to reduce redundancies and non-global state.
class DrawNodeState {

    constructor(ctx, stateid, sdfg_state = undefined) {
        this.ctx = ctx;
        this.stateid = stateid;
        if(sdfg_state === undefined) {
            sdfg_state = global_state;
        }
        this.sdfg_state = sdfg_state;
        this.tooltip = null;
    }

    highlights() {
        return this.sdfg_state.highlights.filter(x => (x['state-id'] == this.stateid &&
                                                       'node-id' in x)).map(x => x['node-id']);
    }

    highlight_edges() {
        return this.sdfg_state.highlights.filter(x => (x['state-id'] == this.stateid &&
                                                       'edge-id' in x)).map(x => x['edge-id']);
    }

    highlight_interstate_edges() {
        return this.sdfg_state.highlights.filter(x => ('isedge-id' in x)).map(x => x['isedge-id']);
    }

    highlight_states() {
        return this.sdfg_state.highlights.filter(x => (Object.keys(x).length == 1 &&
                                                       'state-id' in x)).map(x => x['state-id']);
    }

    onStartDraw() {
        this.tooltip = null;
    }

    onEndDraw() {
        if (this.tooltip) {
            let pos = this.tooltip[0];
            let label = this.tooltip[1];
            let ctx = this.ctx;
            let x = pos.x + 10;
            let textmetrics = ctx.measureText(label);
            ctx.fillStyle = "black";
            ctx.fillRect(x, pos.y - LINEHEIGHT, textmetrics.width * 1.4, LINEHEIGHT * 1.2);
            ctx.fillStyle = "white";
            ctx.fillText(label, x + 0.2 * textmetrics.width, pos.y - 0.1 * LINEHEIGHT);
            ctx.fillStyle = "black";
        }
    }

    nodeColor(nodeid, hovered = null) {
        if(this.highlights().filter(x => x == nodeid).length > 0) {
            return "red";
        } else if(hovered) {
            return "green";
        } else {
            return "black";
        }
    }

    edgeColor(edgeid, hovered = null) {
        if (this.stateid < 0) { // Inter-state edge
            if (this.highlight_interstate_edges().filter(x => x == edgeid).length > 0)
                return "red";
            return hovered ? "green" : "blue";
        } else {
            if (this.highlight_edges().filter(x => x == edgeid).length > 0)
                return "red";
            return hovered ? "green" : "black";
        }
    }

    showTooltip(pos, label) {
        this.tooltip = [pos, label];
    }

    drawHexagon(ctx, x, y, w, h) {
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
    drawEllipse(ctx, x, y, w, h) {
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

    drawArrayNode(node, nodeid, hovered) {
        let ctx = this.ctx;
        var topleft_x = node.x - node.width / 2.0;
        var topleft_y = node.y - node.height / 2.0;
        ctx.beginPath();
        this.drawEllipse(ctx, topleft_x, topleft_y, node.width, node.height);
        ctx.closePath();
        ctx.strokeStyle = this.nodeColor(nodeid, hovered);
        
        let nodedesc = node.sdfg.attributes._arrays[node.properties.data];
        // Streams have dashed edges
        if (nodedesc.type === "Stream") {
            ctx.setLineDash([5, 3]);
        } else {
            ctx.setLineDash([1,0]);
        }
        
        if (nodedesc.attributes.transient === false) {
            ctx.lineWidth = 3.0;
        } else {
            ctx.lineWidth = 1.0;
        }
        
        
        ctx.stroke();
        ctx.lineWidth = 1.0;
        ctx.setLineDash([1,0]);
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.fillStyle = "black";
        var textmetrics = ctx.measureText(node.label);
        ctx.fillText(node.label, node.x - textmetrics.width / 2.0, node.y + LINEHEIGHT / 4.0);
    }

    drawEntryNode(node, nodeid, hovered) {
        let ctx = this.ctx;

        if (node.properties.is_collapsed) {
            this.drawHexagon(this.ctx, node.x, node.y, node.width, node.height);
        } else {
            var topleft_x = node.x - node.width / 2.0;
            var topleft_y = node.y - node.height / 2.0;
            ctx.beginPath();
            ctx.moveTo(topleft_x, topleft_y + node.height);
            ctx.lineTo(topleft_x + node.width, topleft_y + node.height);
            ctx.lineTo(topleft_x + node.width - node.height, topleft_y);
            ctx.lineTo(topleft_x + node.height, topleft_y);
            ctx.lineTo(topleft_x, topleft_y + node.height);
            ctx.closePath();
        }
        ctx.strokeStyle = this.nodeColor(nodeid, hovered);
        
        // Consume scopes have dashed edges
        if (node.type.startsWith("Consume"))
            ctx.setLineDash([5, 3]);
        else
            ctx.setLineDash([1, 0]);
        
        
        ctx.stroke();
        ctx.setLineDash([1, 0]);
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.fillStyle = "black";
        var textmetrics = ctx.measureText(node.label);
        ctx.fillText(node.label, node.x - textmetrics.width / 2.0, node.y + LINEHEIGHT / 2.0);

        this.drawConnectors(node.in_connectors, topleft_x+node.height, topleft_y - 0.5*LINEHEIGHT,
                node.width - 2*node.height, hovered);
        this.drawConnectors(node.out_connectors, topleft_x+node.height,
                     topleft_y+node.height - 0.5*LINEHEIGHT, node.width - 2*node.height, hovered);
    }

    drawExitNode(node, nodeid, hovered) {
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
        ctx.strokeStyle = this.nodeColor(nodeid, hovered);
        
        // Consume scopes have dashed edges
        if (node.type.startsWith("Consume"))
            ctx.setLineDash([5, 3]);
        else
            ctx.setLineDash([1, 0]);
        
        
        ctx.stroke();
        ctx.setLineDash([1, 0]);
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.fillStyle = "black";
        var textmetrics = ctx.measureText(node.label);
        ctx.fillText(node.label, node.x - textmetrics.width / 2.0, node.y + LINEHEIGHT / 2.0);

        this.drawConnectors(node.in_connectors, topleft_x+node.height, topleft_y - 0.5*LINEHEIGHT,
                node.width - 2*node.height, hovered);
        this.drawConnectors(node.out_connectors, topleft_x+node.height,
                     topleft_y+node.height - 0.5*LINEHEIGHT, node.width - 2*node.height, hovered);
    }

    drawTaskletNode(node, nodeid, hovered) {
        var ctx = this.ctx;
        var topleft_x = node.x - node.width / 2.0;
        var topleft_y = node.y - node.height / 2.0;
        var octseg = node.height / 3.0;
        ctx.beginPath();
        ctx.moveTo(topleft_x, topleft_y + octseg);
        ctx.lineTo(topleft_x + octseg, topleft_y);
        ctx.lineTo(topleft_x + node.width - octseg, topleft_y);
        ctx.lineTo(topleft_x + node.width, topleft_y + octseg);
        ctx.lineTo(topleft_x + node.width, topleft_y + 2 * octseg);
        ctx.lineTo(topleft_x + node.width - octseg, topleft_y + node.height);
        ctx.lineTo(topleft_x + octseg, topleft_y + node.height);
        ctx.lineTo(topleft_x, topleft_y + 2 * octseg);
        ctx.lineTo(topleft_x, topleft_y + 1 * octseg);
        ctx.closePath();
        ctx.strokeStyle = this.nodeColor(nodeid, hovered);
        ctx.stroke();
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.fillStyle = "black";
        var textmetrics = ctx.measureText(node.label);
        ctx.fillText(node.label, node.x - textmetrics.width / 2.0, node.y + LINEHEIGHT / 2.0);
        this.drawConnectors(node.in_connectors, topleft_x+octseg, topleft_y - 0.5*LINEHEIGHT,
                node.width - 2*octseg, hovered);
        this.drawConnectors(node.out_connectors, topleft_x+octseg, topleft_y+node.height - 0.5*LINEHEIGHT,
                node.width - 2*octseg, hovered);
    }

    drawReduceNode(node, nodeid, hovered) {
        let ctx = this.ctx;
        var topleft_x = node.x - node.width / 2.0;
        var topleft_y = node.y - node.height / 2.0;

        ctx.beginPath();
        ctx.moveTo(topleft_x, topleft_y);
        ctx.lineTo(topleft_x + node.width / 2, topleft_y + node.height);
        ctx.lineTo(topleft_x + node.width, topleft_y);
        ctx.lineTo(topleft_x, topleft_y);
        ctx.closePath();
        ctx.strokeStyle = this.nodeColor(nodeid, hovered);
        ctx.stroke();
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.fillStyle = "black";
        var textmetrics = ctx.measureText(node.label);
        ctx.fillText(node.label, node.x - textmetrics.width / 2.0, node.y - node.height / 4.0 + LINEHEIGHT / 2.0);
    }

    drawConnectors(labels, topleft_x, topleft_y, connarea_width, hovered2) {
        let next_topleft_x = topleft_x;
        let next_topleft_y = topleft_y;
        var ctx = this.ctx;
        var that = this;
        var hovered = this.sdfg_state.sdfg.mousepos;
        
        var spacing = 10;
        var connlength = (labels.length - 1) * spacing;
        labels.forEach(function(label) {
            connlength += LINEHEIGHT; //ctx.measureText(label).width * 1.1;
        });
        
        next_topleft_x += (connarea_width - connlength) / 2.0;
            
        labels.forEach(function(label) {
            //let textmetrics = ctx.measureText(label);
            //let labelwidth = textmetrics.width * 1.1;
            let labelwidth = LINEHEIGHT;
            ctx.beginPath();
            //that.drawEllipse(ctx, next_topleft_x, next_topleft_y - 0.1*LINEHEIGHT, labelwidth, LINEHEIGHT*1.2);
            that.drawEllipse(ctx, next_topleft_x, next_topleft_y, labelwidth, LINEHEIGHT);
            ctx.closePath();

            // If this connector is being hovered
            let tooltip = false;
            if (hovered && hovered.x >= next_topleft_x && hovered.x <= (next_topleft_x + labelwidth) &&
                hovered.y >= (next_topleft_y - 0.1*LINEHEIGHT) && hovered.y <= (next_topleft_y + LINEHEIGHT*1.1)) {
                tooltip = true;
                ctx.strokeStyle = "green";
            } else {
                ctx.strokeStyle = "black";
            }
            ctx.stroke();
            ctx.fillStyle = "#f0fdff";
            ctx.fill();
            ctx.fillStyle = "black";
            ctx.strokeStyle = "black";

            if (tooltip)
                that.showTooltip(hovered, label);

            next_topleft_x += labelwidth + 10;
        });
    }

    draw_node(node, nodeid) {
        // Find out whether this node is hovered on
        var hovered = null;
        if ('hovered' in node.sdfg) {
            let hovered_dict = node.sdfg.hovered;
            if ('node' in hovered_dict) {
                hovered = (node.state.id == hovered_dict['node'][0] &&
                    nodeid == hovered_dict['node'][1]) ? hovered_dict['node'][2] : null;
            }
        }

        if (node.type == "AccessNode") {
            this.drawArrayNode(node, nodeid, hovered)
        }
        else if (node.type.endsWith("Entry")) {
            this.drawEntryNode(node, nodeid, hovered)
        }
        else if (node.type.endsWith("Exit")) {
            this.drawExitNode(node, nodeid, hovered)
        }
        else if (node.type == "Tasklet") {
            this.drawTaskletNode(node, nodeid, hovered)
        } else if (node.type == "EmptyTasklet") {
            // Do nothing
        } else if (node.type == "Reduce") {
            this.drawReduceNode(node, nodeid, hovered)
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
            ctx.strokeStyle = this.nodeColor(nodeid, hovered);
            ctx.stroke();
            ctx.fillStyle = "white";
            ctx.fill();
            ctx.fillText(node.label, node.x - node.width / 2, node.y);
        }
    }

    draw_edge(edge, edgeid) {
        // Find out whether this edge is hovered
        let hovered = null;
        if ('hovered' in edge.sdfg) {
            let hovered_dict = edge.sdfg.hovered;
            if ('edge' in hovered_dict && this.stateid >= 0) {
                hovered = (edge.state.id == hovered_dict['edge'][0] &&
                    edgeid == hovered_dict['edge'][1]) ? hovered_dict['edge'][2] : null;
            } else if ('interstate_edge' in hovered_dict && this.stateid < 0) {
                hovered = (edgeid == hovered_dict['interstate_edge'][0]) ? hovered_dict['interstate_edge'][1] : null;
            }
        }

        let ctx = this.ctx;
        ctx.beginPath();

        // Consider connectors in edge drawing
        let src_offset = 0, dst_offset = 0;
        let sedge;
        if (this.stateid >= 0)
            sedge = edge.sdfg.nodes[this.stateid].edges[edgeid];
        else
            sedge = edge.sdfg.edges[edgeid];

        if ('src_connector' in sedge && sedge.src_connector)
            src_offset = 0.5*LINEHEIGHT;
        if ('dst_connector' in sedge && sedge.dst_connector)
            dst_offset = -0.5*LINEHEIGHT;

        ctx.moveTo(edge.points[0].x, edge.points[0].y + src_offset);

        let i;
        for (i = 1; i < edge.points.length - 2; i++) {
            let xm = (edge.points[i].x + edge.points[i + 1].x) / 2.0;
            let ym = (edge.points[i].y + edge.points[i + 1].y) / 2.0;
            ctx.quadraticCurveTo(edge.points[i].x, edge.points[i].y, xm, ym);
        }
        ctx.quadraticCurveTo(edge.points[i].x, edge.points[i].y, edge.points[i+1].x, edge.points[i+1].y  + dst_offset);


        ctx.strokeStyle = this.edgeColor(edgeid, hovered);
        ctx.fillStyle = this.edgeColor(edgeid, hovered);

        ctx.stroke();
        if (edge.points.length < 2) return;
        let lastpoint = Object.assign({}, edge.points[edge.points.length - 1]);
        lastpoint.y += dst_offset;
        this.drawArrow(ctx, edge.points[edge.points.length - 2], lastpoint, 3);

        ctx.fillStyle = "black";
        ctx.strokeStyle = "black";

        if (hovered)
            this.showTooltip(hovered, edge.label);
    }

    draw_state(state, state_id) {
        let topleft_x = state.x - state.width / 2.0;
        let topleft_y = state.y - state.height / 2.0;
        let hovered = this.sdfg_state.sdfg.hovered;
        if (hovered && 'state' in hovered)
            hovered = state_id == hovered['state'][0] ? hovered['state'][1] : null;
        else
            hovered = null;

        let ctx = this.ctx;
        ctx.beginPath();
        ctx.moveTo(topleft_x, topleft_y);
        ctx.lineTo(topleft_x + state.width, topleft_y);
        ctx.lineTo(topleft_x + state.width, topleft_y+state.height);
        ctx.lineTo(topleft_x, topleft_y+state.height);
        ctx.lineTo(topleft_x, topleft_y);
        ctx.closePath();
        ctx.fillStyle="#deebf7";

        ctx.fill();
        ctx.fillStyle="#000000";

        // If this state is highlighted or hovered
        if (hovered || this.highlight_states()[0] == state_id) {
            ctx.strokeStyle = (this.highlight_states()[0] == state_id) ? "red" : "green";
            ctx.stroke();
            ctx.strokeStyle = "black";
        }

    }


}

export { DrawNodeState };