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

// Includes various properties and returns their string representation
function sdfg_property_to_string(prop) {
    if (prop === null) return prop;
    if (prop.type === "Indices" || prop.type === "subsets.Indices") {
        let indices = prop.indices;
        let preview = '[';
        for (let index of indices) {
            preview += sdfg_property_to_string(index) + ', ';
        }
        return preview.slice(0, -2) + ']';
    } else if (prop.type === "Range" || prop.type === "subsets.Range") {
        let ranges = prop.ranges;

        // Generate string from range
        let preview = '[';
        for (let range of ranges) {
            if (range.start == range.end && range.step == 1 && range.tile == 1)
                preview += sdfg_property_to_string(range.start);
            else {
                let endp1 = sdfg_property_to_string(range.end) + ' + 1';
                // Try to simplify using math.js
                try {
                    endp1 = math.simplify(endp1).toString();
                } catch(e) {}

                preview += sdfg_property_to_string(range.start) + ':' +
                    endp1;
                if (range.step != 1) {
                    preview += ':' + sdfg_property_to_string(range.step);
                    if (range.tile != 1)
                        preview += ':' + sdfg_property_to_string(range.tile);
                } else if (range.tile != 1) {
                    preview += '::' + sdfg_property_to_string(range.tile);
                }
            }
            preview += ', ';
        }
        return preview.slice(0, -2) + ']';
    } else if (prop.language !== undefined && prop.string_data !== undefined) {
        // Code
        return '<pre class="w3-code">' + prop.string_data + '</pre>';
    } else if (prop.approx !== undefined && prop.main !== undefined) {
        // SymExpr
        return prop.main;
    } else if (prop.constructor == Object) {
        // General dictionary
        return JSON.stringify(prop);
    } else if (prop.constructor == Array) {
        // General array
        let result = '[ ';
        let first = true;
        for (let subprop of prop) {
            if (!first)
                result += ', ';
            result += sdfg_property_to_string(subprop);
            first = false;
        }
        return result + ' ]';
    } else {
        return prop;
    }
}
