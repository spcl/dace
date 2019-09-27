import {ObjectHelper} from "../datahelper.js";


var roofline_socket = 0;


function drawMarkersY(ctx, margin, options) {

    let width = () => ctx.canvas.width;
    let height = () => ctx.canvas.height;

    let steps = options.steps;
    let step_value_func = options.step_value_func;

    let axis_len_x = height() - margin.top - margin.bottom;

    let pix_step = axis_len_x / steps;


    let start = {
        x: margin.left,
        y: height() - margin.bottom
    };

    let marker_height = 10;
    ctx.save();
    for(let i = 1; i < steps; ++i) {

        let cur_val = step_value_func(i);
        let pos = start.y - i * pix_step;

        // Markers
        ctx.beginPath();
        ctx.strokeStyle = "black";
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.moveTo(start.x - marker_height / 2.0, pos);
        ctx.lineTo(start.x + marker_height / 2.0, pos);
        ctx.stroke();


        // Gridlines
        ctx.beginPath();
        ctx.strokeStyle = "gray";
        ctx.setLineDash([5, 15]);
        ctx.lineWidth = 1;
        ctx.moveTo(margin.left, pos);
        ctx.lineTo(width() - margin.right, pos);
        ctx.stroke();

        // Draw value indices
        ctx.beginPath();
        ctx.strokeStyle = "black";
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
        ctx.font = "30px Arial";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";


        if(cur_val < 1) {
            cur_val = ObjectHelper.valueToSensibleString(cur_val, "fraction");
        }
        ctx.fillText(cur_val.toString(), start.x - marker_height / 2.0, pos);

    }
    
    let ypos = (height() - margin.bottom - margin.top) / 2.0 - margin.top;
    // Draw axis title
    ctx.beginPath();
    ctx.strokeStyle = "black";
    ctx.setLineDash([]);
    ctx.lineWidth = 1;
    ctx.font = "40px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.translate(margin.left, 0);
    ctx.rotate(Math.PI / -2.0);
    ctx.fillText("Performance [FLOP/c]", -height() / 2.0, -margin.left / 2);
    ctx.translate(-margin.left, 0);

    ctx.restore();
}

function drawMarkersX(ctx, margin, options) {

    let width = () => ctx.canvas.width;
    let height = () => ctx.canvas.height;

    let steps = options.steps;
    let step_value_func = options.step_value_func;

    let axis_len_x = width() - margin.left - margin.right;

    let pix_step = axis_len_x / steps;


    let start = {
        x: margin.left,
        y: height() - margin.bottom
    };

    let marker_height = 10;
    ctx.save();
    for(let i = 1; i < steps; ++i) {

        let cur_val = step_value_func(i);
        let pos = start.x + i * pix_step;

        // Markers
        ctx.beginPath();
        ctx.strokeStyle = "black";
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.moveTo(pos, start.y - marker_height / 2.0);
        ctx.lineTo(pos, start.y + marker_height / 2.0);
        ctx.stroke();


        // Gridlines
        ctx.beginPath();
        ctx.strokeStyle = "gray";
        ctx.setLineDash([5, 15]);
        ctx.lineWidth = 1;
        ctx.moveTo(pos, margin.top);
        ctx.lineTo(pos, height() - margin.bottom);
        ctx.stroke();

        // Draw value indices
        ctx.beginPath();
        ctx.strokeStyle = "black";
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
        ctx.font = "30px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";

        if(cur_val < 1) {
            cur_val = ObjectHelper.valueToSensibleString(cur_val, "fraction");
        }

        ctx.fillText(cur_val.toString(), pos, start.y + marker_height / 2.0);
    }
    // Draw axis title
    ctx.beginPath();
    ctx.strokeStyle = "black";
    ctx.setLineDash([]);
    ctx.lineWidth = 1;
    ctx.font = "40px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("Operational intensity [FLOP/byte]", margin.left + (width() - margin.left - margin.right) / 2.0, height() - margin.bottom / 2.0);

    ctx.restore();
}

function fillLine(ctx, margin, yoptions, gflops, mode="hard") {
    // For animations only
    let end_scale = 1.0;
    let start_scale = 1.0;

    // Width function
    let width = () => ctx.canvas.width;
    let height = () => ctx.canvas.height;

    // Draw the peak GFLOP/c line
    ctx.save();

    if(mode==="soft") {
        ctx.setLineDash([15, 15]);
    }
    else {
        ctx.setLineDash([]);
    }
    ctx.beginPath();
    ctx.strokeStyle = "black";
    ctx.lineWidth = 3;

    let axis_len_y = height() - margin.top - margin.bottom;
    let pix_step = axis_len_y / yoptions.steps;

    let logval = yoptions.inv_step_value_func(gflops);
    let y = height() - margin.bottom - pix_step * logval;

    ctx.moveTo(margin.left, y);
    ctx.lineTo(width() - margin.right, y);
    ctx.stroke();

    ctx.beginPath();

    ctx.font = "25px Arial";
    ctx.fillStyle = "black";

    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillText(gflops.toString() + " FLOP/c", width() - margin.right, y);

    ctx.restore();
}

function options_max_val(options) {
    return options.step_value_func((options.steps - 0) * options.value_per_step);
}

function options_min_val(options) {
    return options.step_value_func(0);
}

function getYPixel(ctx, yoptions, margin, value) {
    let height = () => ctx.canvas.height;
    let axis_len_y = height() - margin.top - margin.bottom;
    let pix_step_y = axis_len_y / yoptions.steps;

    let val = yoptions.inv_step_value_func(value);
    let y = height() - margin.bottom - pix_step_y * val;

    return y;
}

function getXPixel(ctx, xoptions, margin, value) {
    let width = () => ctx.canvas.width;
    let axis_len_x = width() - margin.left - margin.right;
    let pix_step_x = axis_len_x / xoptions.steps;

    let val = xoptions.inv_step_value_func(value);
    let x = margin.left + pix_step_x * val;

    return x;
}

function fillBandwidthBound(ctx, margin, xoptions, yoptions, bytes_per_cycle) {
    // For animations only
    let end_scale = 1.0;
    let start_scale = 1.0;

    // Width function
    let width = () => ctx.canvas.width;
    let height = () => ctx.canvas.height;

    ctx.save();

    ctx.strokeStyle = "black";
    ctx.lineWidth = 3;

    let axis_len_y = height() - margin.top - margin.bottom;
    let pix_step_y = axis_len_y / yoptions.steps;

    let axis_len_x = width() - margin.left - margin.right;
    let pix_step_x = axis_len_x / xoptions.steps;

    let maxyval = options_max_val(yoptions);
    let maxxval = options_max_val(xoptions);

    // y = x + q
    let yval = bytes_per_cycle * maxxval;
    let xval = maxxval;

    yval = Math.min(yval, maxyval);
    xval = Math.min(yval / bytes_per_cycle, maxxval);

    // Find the positions for yval and xval and draw lines to there.
    let logvaly_start = yoptions.inv_step_value_func(options_min_val(xoptions) * bytes_per_cycle);
    let y_start = height() - margin.bottom - pix_step_y * logvaly_start;

    let logvaly = yoptions.inv_step_value_func(yval);
    let y = height() - margin.bottom - pix_step_y * logvaly;

    let logvalx = xoptions.inv_step_value_func(xval);
    let x = margin.left + pix_step_x * logvalx;

    // Move to zero
    ctx.moveTo(margin.left, y_start);

    ctx.lineTo(x, y);
    ctx.stroke();


    ctx.translate(x, -y + margin.top + margin.bottom);
    let angle = Math.atan2(y - y_start, x - margin.left);
    ctx.rotate(angle);
    ctx.font = "25px Arial";
    ctx.fillStyle = "black";

    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillText(bytes_per_cycle.toString() + " B/c", 0, 0);


    ctx.restore();
}

var dotdata = [];
var flopbounds = [];
var bandwidthbounds = [];


function draw_full(ctx, painter)  {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    painter.drawAxes();

    painter.drawComputeBounds();
    painter.drawBandwidthBounds();

    painter.connectDots();

    painter.drawDots();

};

function drawDot(ctx, xoptions, yoptions, margin, flop_per_cyc, flop_per_byte, size = 5) {
    ctx.save();

    let x = getXPixel(ctx, xoptions, margin, flop_per_byte);
    let y = getYPixel(ctx, yoptions, margin, flop_per_cyc);

    let off = size;

    ctx.fillStyle = "red";
    ctx.fillRect(x - off, y - off, off * 2, off * 2);

    ctx.restore();
}

function setDot(ctx, xoptions, yoptions, margin, flop_per_cyc, flop_per_byte, id) {
    let x = getXPixel(ctx, xoptions, margin, flop_per_byte);
    let y = getYPixel(ctx, yoptions, margin, flop_per_cyc);
    dotdata.push({x: x, y: y, floppercyc: flop_per_cyc, flopperbyte: flop_per_byte, highlight: false, id: id});
}


function drawInfoBox(ctx, x, y, sidelen, id=undefined) {
    
    let triangle_size = 20;
    let boxleft = x - sidelen;
    let boxright = x + sidelen;
    let boxbottom = y - triangle_size;
    let boxtop = boxbottom - sidelen;

    ctx.save();
    ctx.fillStyle = "cyan";
    ctx.moveTo(x, y);
    
    ctx.beginPath();

    ctx.lineTo(x - triangle_size, boxbottom);
    ctx.lineTo(boxleft, boxbottom);
    ctx.lineTo(boxleft, boxtop);
    ctx.lineTo(boxright, boxtop);
    ctx.lineTo(boxright, boxbottom);
    ctx.lineTo(x + triangle_size, boxbottom);
    ctx.lineTo(x, y);

    ctx.closePath();
    ctx.stroke();
    ctx.fill();

    ctx.fillStyle = "black";
    ctx.font = "40px Arial";
    ctx.textBaseline = "top";
    if(id !== undefined) {
        ctx.fillText("id: " + id, boxleft + triangle_size, boxtop + triangle_size);
    }
    ctx.restore();
}



class RooflinePainter {

    constructor(ctx, margin, xopts, yopts) {
        this.ctx = ctx;
        this.margin = margin;
        this.xoptions = xopts;
        this.yoptions = yopts;
    }

    drawAxes() {

        let ctx = this.ctx;
        let margin = this.margin;

        let width = () => ctx.canvas.width;
        let height = () => ctx.canvas.height;

        // Draw the y-Axis
        ctx.strokeStyle = "black";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(margin().left, height() - margin().bottom);
        ctx.lineTo(margin().left, margin().top);
        ctx.stroke();
        ctx.closePath();

        // Draw the x-Axis
        ctx.beginPath();
        ctx.moveTo(margin().left, height() - margin().bottom);
        ctx.lineTo(width() - margin().right, height() - margin().bottom);
        ctx.stroke();
        ctx.closePath();

        drawMarkersX(ctx, margin(), this.xoptions);
        drawMarkersY(ctx, margin(), this.yoptions);
    }

    drawDots() {
        let highlights = [];
        for(let elem of dotdata) {
            drawDot(this.ctx, this.xoptions, this.yoptions, this.margin(), elem.floppercyc, elem.flopperbyte, elem.highlight ? 10 : 5);
            if(elem.highlight) {
                highlights.push(elem);
            }
        }

        for(let elem of highlights) {
            elem.x = getXPixel(this.ctx, this.xoptions, this.margin(), elem.flopperbyte);
            elem.y = getYPixel(this.ctx, this.yoptions, this.margin(), elem.floppercyc);
            drawInfoBox(this.ctx, elem.x, elem.y, 200, elem.id);
        }
    }

    drawComputeBounds() {
        let i = 0;
        for(let elem of flopbounds) {
            if(i == 0) {
                fillLine(this.ctx, this.margin(), this.yoptions, elem, "hard");
            }
            else {
                fillLine(this.ctx, this.margin(), this.yoptions, elem, "soft");
            }
            ++i;
        }
    }

    drawBandwidthBounds() {
        for(let elem of bandwidthbounds) {
            fillBandwidthBound(this.ctx, this.margin(), this.xoptions, this.yoptions, elem);
        }
    }

    connectDots() {
        let dot_x = undefined;
        let dot_y = undefined;
        this.ctx.beginPath();
        for(let dot of dotdata) {
            let x = dot.x;
            let y = dot.y;
            let first = dot_x === undefined;
            if(first) {
                this.ctx.moveTo(x, y);
            }
            else {
                this.ctx.lineTo(x, y);
                this.ctx.stroke();
            }
            dot_x = x;
            dot_y = y;
            
        }
    }
}



function main(canvas=undefined, message_source=undefined) {

    if(canvas == undefined) canvas = document.getElementById("myCanvas");
    let ctx = canvas.getContext("2d");

    

    let width = () => ctx.canvas.width;
    let height = () => ctx.canvas.height;

    let margin = () => { return {
            top: 100,
            bottom: 100,
            left: 200,
            right: 100
        };
    };

    let xoptions = {
        steps: 20,
        value_per_step: 1,
        step_value_func: (step) => {
            return Math.pow(2.0, -6.0 + step);
        },
        inv_step_value_func: (step) => {
            return 6.0 + Math.log2(step);
        }
    };
    let yoptions = {
        steps: 15,
        value_per_step: 1,
        step_value_func: (step) => {
            return Math.pow(2.0, -8.0 + step);
        },
        inv_step_value_func: (step) => {
            return 8.0 + Math.log2(step);
        }
    };

    let painter = new RooflinePainter(ctx, margin, xoptions, yoptions);

    canvas.addEventListener("mousemove", (event) => {
        let rect = canvas.getBoundingClientRect();
        let x = (event.clientX - rect.left) / (rect.right - rect.left) * canvas.width;
        let y = (event.clientY - rect.top) / (rect.bottom - rect.top) * canvas.height;

        let tol = 15;
        let pow2 = x => x*x;
        
        for(let elem of dotdata)
        {
            elem.x = getXPixel(ctx, xoptions, margin(), elem.flopperbyte);
            elem.y = getYPixel(ctx, yoptions, margin(), elem.floppercyc);
            if(pow2(elem.x - x) + pow2(elem.y - y) <= pow2(tol))
            {
                elem.highlight = true;
            }
            else
                elem.highlight = false;
        }

        draw_full(ctx, painter);
    });

    /*
    Clicking should cause the program with the given id to be highlighted. For that, a message has to be sent to the "host".
    */
    canvas.addEventListener("click", (event) => {
        let rect = canvas.getBoundingClientRect();
        let x = (event.clientX - rect.left) / (rect.right - rect.left) * canvas.width;
        let y = (event.clientY - rect.top) / (rect.bottom - rect.top) * canvas.height;

        let tol = 15;
        let pow2 = x => x*x;
        
        for(let elem of dotdata)
        {
            if(pow2(elem.x - x) + pow2(elem.y - y) <= pow2(tol))
            {
                roofline_socket.send(JSON.stringify({"msg_type": "roofline", "command": "select-program", "programID": elem.id}));
                return; // Dots sadly could be interleaving. To not send too many select requests, we have to end it here
            }
            
        }
    });



    fillLine(ctx, margin(), yoptions, 8.0);
    fillBandwidthBound(ctx, margin(), xoptions, yoptions, 8);

    flopbounds.push(64.0, 8.0);
    bandwidthbounds.push(8.0);

    setDot(ctx, xoptions, yoptions, margin(), 1, 1, 1);
    setDot(ctx, xoptions, yoptions, margin(), 2, 3, 2);
    setDot(ctx, xoptions, yoptions, margin(), 4, 6, 3);
    setDot(ctx, xoptions, yoptions, margin(), 7, 7, 4);
    setDot(ctx, xoptions, yoptions, margin(), 8, 8, 5);

    draw_full(ctx, painter);

    let data_proc = msg => {
        if(msg["msg_type"] == "roofline-data") {

            dotdata = [];

            let data = msg["data"];

            for(let x of data) {
                let pid = x["ProgramID"];
                let flop_per_c = x["FLOP_C"];
                let in_b_per_c = x["INPUT_B_C"];
                let proc_b_per_c = x["PROC_B_C"];
                let mem_b_per_c = x["MEM_B_C"];


                let flop_per_byte = flop_per_c / mem_b_per_c;

                // We'll use the in-memory (which is not quite accurate, since out-data usually has to be loaded as well)
                setDot(ctx, xoptions, yoptions, margin(), flop_per_c, flop_per_byte, pid);
            }
        }
    };
    
    let socksetup = () => {
        roofline_socket = new WebSocket('ws://localhost:8024/');

        roofline_socket.onopen = function (event) {
            roofline_socket.send(JSON.stringify({"msg_type": "roofline", "command": "connected"}));
        }
        roofline_socket.onmessage = function (event) {

            msg = JSON.parse(event.data);
            data_proc(msg);
        }
        roofline_socket.onclose = function (event) {
            console.log("ERROR: Connection closed!");
        }
        roofline_socket.onerror = function (event) {
            console.log("ERROR: There was an error with the connection.");
        }
        
    };

    if(message_source == undefined)
        socksetup();
    else {
        message_source(data_proc);
    }

    return () => draw_full(ctx, painter);
}

export {main as main};