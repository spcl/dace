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

function sdfg_range_elem_to_string(range, settings=null) {
    let preview = '';
    if (range.start == range.end && range.step == 1 && range.tile == 1)
        preview += sdfg_property_to_string(range.start, settings);
    else {
        if (settings && settings.inclusive_ranges) {
            preview += sdfg_property_to_string(range.start, settings) + '..' +
                sdfg_property_to_string(range.end, settings);
        } else {
            let endp1 = sdfg_property_to_string(range.end, settings) + ' + 1';
            // Try to simplify using math.js
            var mathjs = undefined;
            try {
                mathjs = window.math;
            } catch(e) {
                try { mathjs = math; } catch(e) {}
            }
            try {
                endp1 = mathjs.simplify(endp1).toString();
            } catch(e) {}
            preview += sdfg_property_to_string(range.start, settings) + ':' +
                endp1;
        }
        if (range.step != 1) {
            preview += ':' + sdfg_property_to_string(range.step, settings);
            if (range.tile != 1)
                preview += ':' + sdfg_property_to_string(range.tile, settings);
        } else if (range.tile != 1) {
            preview += '::' + sdfg_property_to_string(range.tile, settings);
        }
    }
    return preview;
}

// Includes various properties and returns their string representation
function sdfg_property_to_string(prop, settings=null) {
    if (prop === null) return prop;
    if (typeof prop === 'boolean') {
        if (prop)
            return 'True';
        return 'False';
    } else if (prop.type === "Indices" || prop.type === "subsets.Indices") {
        let indices = prop.indices;
        let preview = '[';
        for (let index of indices) {
            preview += sdfg_property_to_string(index, settings) + ', ';
        }
        return preview.slice(0, -2) + ']';
    } else if (prop.type === "Range" || prop.type === "subsets.Range") {
        let ranges = prop.ranges;

        // Generate string from range
        let preview = '[';
        for (let range of ranges) {
            preview += sdfg_range_elem_to_string(range, settings) + ', ';
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
            result += sdfg_property_to_string(subprop, settings);
            first = false;
        }
        return result + ' ]';
    } else {
        return prop;
    }
}

/**
 * Receives a callback that accepts (node, parent graph) and returns a value.
 * This function is invoked recursively per scope (including scope nodes), unless the return
 * value is false, upon which the sub-scope will not be visited.
 * The function also accepts an optional post-subscope callback (same signature as `func`).
 **/
function traverse_sdfg_scopes(sdfg, func, post_subscope_func=null) {
    function scopes_recursive(graph, nodes, processed_nodes=null) {
        if (processed_nodes === null)
            processed_nodes = new Set();

        for (let nodeid of nodes) {
            let node = graph.node(nodeid);
            if (node !== undefined && processed_nodes.has(node.id.toString()))
                continue;

            // Invoke function
            let result = func(node, graph);

            // Skip in case of e.g., collapsed nodes
            if (result !== false) {
                // Traverse scopes recursively (if scope_dict provided)
                if (node.type().endsWith('Entry')) {
                    let state = node.sdfg.nodes[node.parent_id];
                    if (state.scope_dict[node.id] !== undefined)
                        scopes_recursive(graph, state.scope_dict[node.id], processed_nodes);
                }

                // Traverse states or nested SDFGs
                if (node.data.graph) {
                    let state = node.data.state;
                    if (state !== undefined && state.scope_dict[-1] !== undefined)
                        scopes_recursive(node.data.graph, state.scope_dict[-1]);
                    else // No scope_dict, traverse all nodes as a flat hierarchy
                        scopes_recursive(node.data.graph, node.data.graph.nodes());
                }
            }
            
            if (post_subscope_func)
                post_subscope_func(node, graph);

            processed_nodes.add(node.id.toString());
        }
    }
    scopes_recursive(sdfg, sdfg.nodes());
}
