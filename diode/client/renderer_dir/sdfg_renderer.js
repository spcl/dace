// This class groups node-drawing to reduce redundancies and non-global state.
class DrawNodeState {

    constructor(ctx, stateid, sdfg_state = undefined) {
        this.ctx = ctx;
        this.stateid = stateid;
        if(sdfg_state === undefined) {
            sdfg_state = global_state;
        }
        this.sdfg_state = sdfg_state;
    }

    highlights() {
        return this.sdfg_state.highlights.filter(x => x['state-id'] == this.stateid).map(x => x['node-id']);
    }

    nodeColor(nodeid, hovered = false) {
        if(this.highlights().filter(x => x == nodeid).length > 0) {
            return "red";
        } else if(hovered) {
            return "green";
        } else {
            return "black";
        }
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
        var topleft_x = node.x - node.width / 2.0;
        var topleft_y = node.y - node.height / 2.0;
        ctx.beginPath();
        ctx.moveTo(topleft_x, topleft_y + node.height);
        ctx.lineTo(topleft_x + node.width, topleft_y + node.height);
        ctx.lineTo(topleft_x + node.width - node.height, topleft_y);
        ctx.lineTo(topleft_x + node.height, topleft_y);
        ctx.lineTo(topleft_x, topleft_y + node.height);
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

        this.drawConnectors(node.in_connectors, topleft_x+node.height, topleft_y - 0.5*LINEHEIGHT, node.width - 2*node.height);
        this.drawConnectors(node.out_connectors, topleft_x+node.height, topleft_y+node.height - 0.5*LINEHEIGHT, node.width - 2*node.height);
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

        this.drawConnectors(node.in_connectors, topleft_x+node.height, topleft_y - 0.5*LINEHEIGHT, node.width - 2*node.height);
        this.drawConnectors(node.out_connectors, topleft_x+node.height, topleft_y+node.height - 0.5*LINEHEIGHT, node.width - 2*node.height);
    }

    drawTaskletNode(node, nodeid, hovered) {
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
        ctx.strokeStyle = this.nodeColor(nodeid, hovered);
        ctx.stroke();
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.fillStyle = "black";
        var textmetrics = ctx.measureText(node.label);
        ctx.fillText(node.label, node.x - textmetrics.width / 2.0, node.y + LINEHEIGHT / 2.0);
        this.drawConnectors(node.in_connectors, topleft_x+hexseg, topleft_y - 0.5*LINEHEIGHT, node.width - 2*hexseg);
        this.drawConnectors(node.out_connectors, topleft_x+hexseg, topleft_y+node.height - 0.5*LINEHEIGHT, node.width - 2*hexseg);
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

    drawConnectors(labels, topleft_x, topleft_y, connarea_width) {
        let next_topleft_x = topleft_x;
        let next_topleft_y = topleft_y;
        var ctx = this.ctx;
        var ellipse = this.drawEllipse;
        
        var spacing = 10;
        var connlength = (labels.length - 1) * spacing;
        labels.forEach(function(label) {
            connlength += ctx.measureText(label).width * 1.1;
        });
        
        next_topleft_x += (connarea_width - connlength) / 2.0;
            
        labels.forEach(function(label) {
            let textmetrics = ctx.measureText(label);
            let labelwidth = textmetrics.width * 1.1;
            ctx.beginPath();
            ellipse(ctx, next_topleft_x, next_topleft_y - 0.1*LINEHEIGHT, labelwidth, LINEHEIGHT*1.2);
            ctx.closePath();
            ctx.strokeStyle = "black";
            ctx.stroke();
            ctx.fillStyle = "#f0fdff";
            ctx.fill();
            ctx.fillStyle = "black";
            
            ctx.fillText(label, next_topleft_x + (labelwidth / 2.0) - (textmetrics.width / 2.0), 
                         next_topleft_y + 0.8*LINEHEIGHT);
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
        }
        else if (node.type == "Reduce") {
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
            if ('edge' in hovered_dict) {
                hovered = (edge.state.id == hovered_dict['edge'][0] &&
                    edgeid == hovered_dict['edge'][1]) ? hovered_dict['edge'][2] : null;
            }
        }

        let ctx = this.ctx;
        ctx.beginPath();
        ctx.moveTo(edge.points[0].x, edge.points[0].y);
        for(let elem of edge.points) {
            ctx.lineTo(elem.x, elem.y)
        }
        if (hovered)
            ctx.strokeStyle = "green";
        else
            ctx.strokeStyle = "black";

        ctx.stroke();
        if (edge.points.length < 2) return;
        this.drawArrow(ctx, edge.points[edge.points.length - 2], edge.points[edge.points.length - 1], 5);
        if (hovered) {
            let x = hovered.x + 10;
            let textmetrics = ctx.measureText(edge.label);
            ctx.fillStyle = "black";
            ctx.fillRect(x, hovered.y - LINEHEIGHT, textmetrics.width * 1.4, LINEHEIGHT * 1.2);
            ctx.fillStyle = "white";
            ctx.fillText(edge.label, x + 0.2*textmetrics.width, hovered.y - 0.1*LINEHEIGHT);
            ctx.fillStyle = "black";
        }
    }

}

export { DrawNodeState };