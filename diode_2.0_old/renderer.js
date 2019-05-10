var editor = ace.edit("editor");
var MatlabMode = ace.require("ace/mode/matlab").Mode;
editor.session.setMode(new MatlabMode());

var remote = require('electron').remote;
const dialog = require('electron').remote.dialog

const ipcRenderer = require('electron').ipcRenderer;
ipcRenderer.on('getText', function (event, arg) {
    var text = editor.getValue()
    ipcRenderer.send('getTextResponse', text);
});

ipcRenderer.on('serverAlive', function (event, arg) {
    wait_widget = document.getElementById("wait-server");
    wait_widget.style.display = 'none';
});

ipcRenderer.on('renderSDFG', function (event, sdfg) {
    var sdfg = JSON.parse(sdfg);
    if (sdfg.type != "SDFG") {
        alert("The passed SDFG is not an SDFG!")
        return;
    }
    for (var i=0; i<sdfg.nodes.length; i++) {
        determineNodeSize(sdfg.nodes[i]);
        layoutSubgraph(sdfg.nodes[i]);
        //drawSubgraph(sdfg.nodes[i])
    }

});

ipcRenderer.on('openFile', function (event, arg) {
    dialog.showOpenDialog(function (fileNames) {
        if (fileNames === undefined) return;
        var fileName = fileNames[0];
        var fs = require('fs');
        fs.readFile(fileName, 'utf-8', function (err, data) {
            editor.setValue(data);
        });
    });
});



function get_node_by_id(subg, node_id) {
    for (var i=0; i<subg.attributes.nodes.length; i++) {
        var node = subg.attributes.nodes[i];
        if (node.id == node_id) return node;
    }
    return null;
}

function is_entry_node(node) {
    if (node.attributes.type == "MapEntry") return true;
    else return false;
}

function is_exit_node(node) {
    if (node.attributes.type == "MapExit") return true;
    if (node.attributes.type == "ConflictResolution") return true;
    else return false;
}


function determineNodeSize(subg, current_x=0, current_y=0) {
    // we need the context to measure text
    ctx = document.getElementById("sdfg").getContext('2d');

    // initially, when an element has no size yet, its size is only determined by the size of
    // its label

    if (subg.attributes.type == "SDFGState") {

        // for each node in the state, determine its size
        for (var i = 0; i < subg.attributes.nodes.length; i++) {
            node = subg.attributes.nodes[i];
            var text = ctx.measureText(subg.attributes.label);
            subg.attributes.size = [text.width, 20];
            determineNodeSize(node);
        }
    }
    else if (subg.attributes.type == "ArrayNode") {
        var text = ctx.measureText(subg.attributes.label);
        subg.attributes.size = [text.width, 20];
        // add space for the round corners
        subg.attributes.size[0] += subg.attributes.size[1];
    }
    else if (subg.attributes.type == "ConflictResolution") {
        //TODO solve triangle problem, for now just take a guess
        var text = ctx.measureText(subg.attributes.label);
        subg.attributes.size = [2 * text.width, 25];
    }
    else if (subg.attributes.type == "MapEntry") {
        var text = ctx.measureText(subg.attributes.label);
        subg.attributes.size = [text.width, 20];
        // add space for the corners
        subg.attributes.size[0] += 2 * subg.attributes.size[1];
    }
    else if (subg.attributes.type == "MapExit") {
        var text = ctx.measureText(subg.attributes.label);
        subg.attributes.size = [text.width, 20];
        // add space for the corners
        subg.attributes.size[0] += 2 * subg.attributes.size[1];
    }
    else if (subg.attributes.type == "Tasklet") {
        var text = ctx.measureText(subg.attributes.label);
        subg.attributes.size = [text.width, 20];
        // add space for the corners
        subg.attributes.size[0] += subg.attributes.size[1];
    }
    else {
        var text = ctx.measureText(subg.attributes.label);
        subg.attributes.size = [text.width, 20];
    }

} 

function add_node_to_scopes(node, scopes) {
    for (var s=0; s<scopes.length; s++) {
        if (scopes[s].entry_node == node.scope_entry) {
            scopes[s].nodes.push(node.id)
            return;
        }
    }

    // not inserted yet, insert
    var new_scope = {};
    new_scope.entry_node = node.scope_entry;
    new_scope.nodes = [ node.id ];
    new_scope.child_scopes = []; // fill in find_parent_scopes
    new_scope.parent_scope = null; // fill in find_parent_scopes
    scopes.push(new_scope);
}

function find_parent_scopes(scopes) {
    for (var s=0; s<scopes.length; s++) {
        entry = scopes[s].entry_node;
        for (var s2=0; s2<scopes.length; s2++) {
            for (var c=0; c<scopes[s2].nodes.length; c++) {
                if (scopes[s2].nodes[c] == entry) {
                    scopes[s2].child_scopes.push(s);
                    scopes[s].parent_scope = s2;
                }
            }
        }
    }
}


function topsort_general(vertices, edges) {

    // vertices: A set
    // edges: a set of [u, v] with u and v in vertices

    var heights = new Map();  // map each vertex to a height
    var in_edges = new Map(); // map each vertex to a set of sucessors

    //initialize
    for (var v of vertices) { 
        heights.set(v, 0);
        in_edges.set(v, new Set())
        for (const e of edges) {
            if (e[1] == v) in_edges.get(v).add(e[0]);
        } 
    }

    var change = true;
    while (change) {
        change = false;
        for (const u of vertices) {
            var maxl = 0;
            for (const v of in_edges.get(u)) {
                l = heights.get(v);
                if (l>maxl) maxl = l;
            }
            var origh = heights.get(u);
            if (origh < maxl+1) {
                heights.set(u, maxl + 1);
                change = true;
            }
        }
    }

    return heights;
}


function get_node_size(subg, node_id) {
    for (var i=0; i<subg.attributes.nodes.length; i++) {
        node = subg.attributes.nodes[i];
        if (node.id == node_id) {
            return node.attributes.size;
        }
    }
    console.log("Node " + node_id + " not found!");
    return null;
}

function compute_level_indices(levels) {
    // for now they are random, but they shouldn't be
    // input  Map vertex => level
    // output Map vertex => level_index

    var output = new Map();
    var level_fill = new Map();
    for (let [nodeid, level] of levels) {
        if (level_fill.has(level)) {
            index = level_fill.get(level);
            output.set(nodeid, index);
            level_fill.delete(level);
            level_fill.set(level, index+1);
        }
        else {
            output.set(nodeid, 0);
            level_fill.set(level, 1);
        }
    }

    return output;
}

function layout_scope(subg, scopes, sidx) {

    if (scopes[sidx].layout_done == true) return;
    
    // make sure all children are layed out before we start
    for (var c=0; c<scopes[sidx].child_scopes.length; c++) {
        var cs = scopes[sidx].child_scopes[c]
        if (!scopes[cs].layout_done) layout_scope(subg, scopes, cs);
    }

    // form a scope-graph of "independent" i.e., in this scope but not in a
    // child scope, nodes and scopes in this scope

    // iterate over nodes in scope, if it is a normal node, add it to vertices,
    // if it is an entry node add the corresponding scope to vertices
    var vertices = new Set();
    var edges = new Set();

    if (scopes[sidx].entry_node != "null") vertices.add(scopes[sidx].entry_node);
    for (var i = 0; i < scopes[sidx].nodes.length; i++) {
        node = get_node_by_id(subg, scopes[sidx].nodes[i]);
        vertices.add(node.id);
    }


    // add edges:
    for (var i = 0; i < subg.attributes.edges.length; i++) {
        edge = subg.attributes.edges[i];
        // simple case: connect two nodes (scope entries are nodes as well)
        if (vertices.has(edge.src) && vertices.has(edge.dst)) edges.add([edge.src, edge.dst]);

        // connect a scope exit to a node
        if (vertices.has(edge.dst) && is_exit_node(get_node_by_id(subg, edge.src))) {
            n = get_node_by_id(subg, edge.src);
            edges.add([n.scope_entry, edge.dst]);
        }

    }

    levels = topsort_general(vertices, edges);
    level_indices = compute_level_indices(levels);

    // for each level, determine its size
    var level_fill = [];     // how many nodes are in each level
    var level_sizes = [];    // what is the width/height of each level
    var max_level = 0;       // number of levels
    for (let [node, level] of levels) {
        if (level > max_level) max_level = level;
    }
    for (var i = 0; i < max_level + 1; i++) {
        level_sizes[i] = [0, 0];
        level_fill[i] = 0;
    }
    for (let [nodeid, level] of levels) {
        level_fill[level] += 1;
        var node = get_node_by_id(subg, nodeid);
        var width = 0;
        var height = 0;
        if (is_entry_node(node) && (level != 1)) {
            // this is a scope node, find it in scopes
            for (var i = 0; i < scopes.length; i++) {
                if (scopes[i].entry_node == node.id) {
                    width = scopes[i].scope_size[0];
                    height = scopes[i].scope_size[1];
                }
            }
        }
        else {
            width = node.attributes.size[0]
            height = node.attributes.size[1]
        }

        level_sizes[level][0] += width;
        level_sizes[level][1] = Math.max(level_sizes[level][1], height);
    }

    var level_positions = [0];
    var max_level_width = 0;
    for (var i = 0; i < max_level + 1; i++) {
        level_positions.push(level_positions[i] + level_sizes[i][1]);
        if (level_sizes[i][0] > max_level_width) max_level_width = level_sizes[i][0];
    }

    scopes[sidx].scope_size = [max_level_width, level_positions[max_level + 1]];

    // assign a _relative_ (to the parent bounding box) position to each node
    // its y position is given by level_positions[level]
    // its x position is (scope_width / level_fill)*level_index + margin
    //  where margin is ((scope_width / level_fill) - node_width)/2

    // if the relative position is 0,0
    if (scopes[sidx].entry_node == "null") scopes[sidx].rel_position = [0,0];

    for (let [nodeid, level] of levels) {
        var node = get_node_by_id(subg, nodeid);
        var rel_y_pos = level_positions[level];
        var column_size = scopes[sidx].scope_size[0] / level_fill[level];

        if (is_entry_node(node) && (level != 1)) {
            // this is a scope node, find it in scopes
            for (var i = 0; i < scopes.length; i++) {
                if (scopes[i].entry_node == node.id) {
                    // assign the position to scopes[i]
                    var node_width = scopes[i].scope_size[0];
                    var margin = (column_size - node_width)/2;
                    var rel_x_pos = column_size * level_indices.get(node.id) + margin;
                    scopes[i].rel_position = [rel_x_pos, rel_y_pos];
                }
            }
        }
        else {
            // assign the position to the node
            var node_width = node.attributes.size[0];
            var margin = (column_size - node_width)/2;
            var rel_x_pos = column_size * level_indices.get(node.id) + margin;
            node.rel_position = [rel_x_pos, rel_y_pos];
        }

    }

    scopes[sidx].layout_done = true;

}

function draw_scopes(subg, scopes, sidx) {
    // if this is the outermost scope, its absolute position is 0,0
    if (scopes[sidx].entry_node == "null") {
        scopes[sidx].position = [0,0];
    }

    if (!scopes[sidx].hasOwnProperty("position")) return;
    if (scopes[sidx].hasOwnProperty("is_drawn")) return;
    scopes[sidx].is_drawn = true;

    // draw all child scopes
    for (var cidx=0; cidx<scopes[sidx].child_scopes.length; cidx++) {
        var child = scopes[sidx].child_scopes[cidx];
        var cpos_x = scopes[sidx].position[0];
        var cpos_y = scopes[sidx].position[1];
        cpos_x += scopes[child].rel_position[0];
        cpos_y += scopes[child].rel_position[1];
        scopes[child].position = [cpos_x, cpos_y];
        draw_scopes(subg, scopes, child);
    }

    // draw a box for the scope
    var ctx = document.getElementById("sdfg").getContext('2d');
    
    ctx.beginPath();
    ctx.moveTo(scopes[sidx].position[0], scopes[sidx].position[1]);
    ctx.lineTo(scopes[sidx].position[0] + scopes[sidx].scope_size[0], scopes[sidx].position[1]);
    ctx.lineTo(scopes[sidx].position[0] + scopes[sidx].scope_size[0], scopes[sidx].position[1] + scopes[sidx].scope_size[1]);
    ctx.lineTo(scopes[sidx].position[0], scopes[sidx].position[1] + scopes[sidx].scope_size[1]);
    ctx.lineTo(scopes[sidx].position[0], scopes[sidx].position[1]);
    ctx.stroke();

    // draw the entry node (this is not in nodes)
    if (scopes[sidx].entry_node != "null") {
        node = get_node_by_id(subg, scopes[sidx].entry_node);
        var cpos_x = scopes[sidx].position[0];
        var cpos_y = scopes[sidx].position[1];
        cpos_x += node.rel_position[0];
        cpos_y += node.rel_position[1];
        node.attributes.position = [cpos_x, cpos_y];
        drawSubgraph(node);
    }

    // draw the nodes in this scope except entry nodes of child scopes (will be drawn by child scope draw function)
    for (var nidx=0; nidx<scopes[sidx].nodes.length; nidx++) {
        node = get_node_by_id(subg, scopes[sidx].nodes[nidx]);
        if (is_entry_node(node)) continue;
        var cpos_x = scopes[sidx].position[0];
        var cpos_y = scopes[sidx].position[1];
        cpos_x += node.rel_position[0];
        cpos_y += node.rel_position[1];
        node.attributes.position = [cpos_x, cpos_y];
        drawSubgraph(node);
    }
}

function layoutSubgraph(subg) {
    console.log("Need to layout: " + JSON.stringify(subg));
    
    scopes = [];
    for (var i=0; i<subg.attributes.nodes.length; i++) {
        node = subg.attributes.nodes[i];
        add_node_to_scopes(node, scopes);
    }
    find_parent_scopes(scopes);
    
    for (var i=0; i<scopes.length; i++) {
        layout_scope(subg, scopes, i);
    }

    for (var i=0; i<scopes.length; i++) {
        draw_scopes(subg, scopes, i);
    }

}




function drawTasklet(node) {
    ctx.beginPath();
    ctx.moveTo(node.position[0] + node.size[1]/2, node.position[1]); //1
    ctx.lineTo(node.position[0] + node.size[0]-node.size[1]/2, node.position[1]); //2
    ctx.lineTo(node.position[0] + node.size[0], node.position[1]+node.size[1]/3); //3
    ctx.lineTo(node.position[0] + node.size[0], node.position[1]+2*(node.size[1]/3)); //4
    ctx.lineTo(node.position[0] + node.size[0]-node.size[1]/2, node.position[1]+node.size[1]); //5
    ctx.lineTo(node.position[0] + node.size[1]/2, node.position[1]+node.size[1]); //6
    ctx.lineTo(node.position[0], node.position[1]+2*(node.size[1]/3)); //7
    ctx.lineTo(node.position[0], node.position[1]+1*(node.size[1]/3)); //8
    ctx.lineTo(node.position[0] + node.size[1]/2, node.position[1]); //1
    ctx.stroke();
    ctx.fillText(node.label,node.position[0]+node.size[1]/2,node.position[1]+node.size[1]*0.75);
}

function drawMapEntry(node) {
    ctx.beginPath();
    ctx.moveTo(node.position[0], node.position[1] + node.size[1]);
    ctx.lineTo(node.position[0]+node.size[0], node.position[1] + node.size[1]);
    ctx.lineTo(node.position[0]+node.size[0]- node.size[1], node.position[1]);
    ctx.lineTo(node.position[0]+node.size[1], node.position[1]);
    ctx.lineTo(node.position[0], node.position[1]+node.size[1]);
    ctx.stroke();
    //TODO the y position calculation is a wild guess
    ctx.fillText(node.label,node.position[0]+node.size[1],node.position[1]+node.size[1]*0.75);
}

function drawMapExit(node) {
    ctx.beginPath();
    ctx.moveTo(node.position[0], node.position[1]);
    ctx.lineTo(node.position[0]+node.size[0], node.position[1]);
    ctx.lineTo(node.position[0]+node.size[0] - node.size[1], node.position[1] + node.size[1]);
    ctx.lineTo(node.position[0]+node.size[1], node.position[1]+node.size[1]);
    ctx.lineTo(node.position[0], node.position[1]);
    ctx.stroke();
    // TODO the y position calculation is a wild guess
    ctx.fillText(node.label,node.position[0]+node.size[1],node.position[1]+node.size[1]*0.75);
}

function drawArrayNode(node) {
    ctx.beginPath();
    ctx.arc(node.position[0]+node.size[1]/2, node.position[1]+node.size[1]/2, node.size[1]/2, 0.5*Math.PI, 1.5*Math.PI);
    ctx.lineTo(node.position[0]+node.size[0]-node.size[1], node.position[1]);
    ctx.arc(node.position[0]+node.size[0]-node.size[1]/2, node.position[1]+node.size[1]/2, node.size[1]/2, 1.5*Math.PI, 0.5*Math.PI);
    ctx.lineTo(node.position[0]+node.size[1]/2, node.position[1]+node.size[1]);
    ctx.stroke();
    // TODO the y position calculation is a wild guess
    ctx.fillText(node.label,node.position[0]+node.size[1]/2,node.position[1]+node.size[1]*0.75);
}

function drawConflictResolution(node) {
    ctx.beginPath();
    ctx.moveTo(node.position[0], node.position[1]);
    ctx.lineTo(node.position[0]+node.size[0], node.position[1])
    ctx.lineTo(node.position[0]+node.size[0]/2, node.position[1]+node.size[1]);
    ctx.lineTo(node.position[0], node.position[1]);
    ctx.stroke();
    var text = ctx.measureText(node.label);
    ctx.fillText(node.label,node.position[0]+node.size[0]/2-text.width/2,node.position[1]+(node.size[1]/2)*0.75);
}

function drawSubgraph(subg) {
    if (subg.attributes.type == "SDFGState") {
        var rerun = false;
        for (var i=0; i<subg.attributes.nodes.length; i++) {
            node = subg.attributes.nodes[i];
            drawSubgraph(node);
        }
    }
    else {
        node = subg.attributes
        if (node.type == "Tasklet") drawTasklet(node);
        else if (node.type == "MapEntry") drawMapEntry(node);
        else if (node.type == "MapExit") drawMapExit(node);
        else if (node.type == "ArrayNode") drawArrayNode(node);
        else if (node.type == "ConflictResolution") drawConflictResolution(node);
        else console.log("Undrawn node type: " + node.type);
    }
}
