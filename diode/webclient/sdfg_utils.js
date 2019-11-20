function find_exit_for_entry(nodes, entry_node) {
    for(let n of nodes) {
        if(n.type.endsWith("Exit") && parseInt(n.scope_entry) == entry_node.id) {
            return n;
        }
    }
    console.warn("Did not find corresponding exit");
    return null;
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


function reviver(name, val) {
    if (name == 'sdfg' && val && typeof val === 'string' && val[0] === '{') {
        return JSON.parse(val, reviver);
    }
    return val;
}

// Recursively parse SDFG, including nested SDFG nodes
function parse_sdfg(sdfg_json) {
    return JSON.parse(sdfg_json, reviver);
}

function isDict(v) {
    return typeof v === 'object' && v !== null && !(v instanceof Array) && !(v instanceof Date);
}

function replacer(name, val, orig_sdfg) {
    if (val && isDict(val) && val !== orig_sdfg && 'type' in val && val.type === 'SDFG') {
        return JSON.stringify(val, (n,v) => replacer(n, v, val));
    }
    return val;
}

function stringify_sdfg(sdfg) {
    return JSON.stringify(sdfg, (name, val) => replacer(name, val, sdfg));
}
