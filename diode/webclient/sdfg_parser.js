class SDFG_Parser {
    constructor(sdfg) {
        this._sdfg = sdfg;
    }

    getStates() {

        return this._sdfg.nodes.map(x => new SDFG_State_Parser(x));
    }

    static lookup_symbols(sdfg, state_id, elem, symbols_to_resolve, depth=0) {
        // Resolve used symbols by following connectors in reverse order
        let state = sdfg.nodes[state_id];

        let syms = [];

        if(elem.constructor == Object) {
            // Memlet
            let memlets = state.edges.filter(x => x.dst == elem.dst && x.src == elem.src);

            // Recurse into parent (since this a multigraph, all edges need to be looked at)
            for(let m of memlets) {

                // Find symbols used (may be Indices or Range)
                let mdata = m.attributes.data.attributes.subset;
                // Check for indices
                if(mdata.type == "subsets.Indices") {
                    // These are constants or variables
                    // Reverse to have smallest unit first
                    let tmp = mdata.indices.map(x => x).reverse();
                    for(let x of tmp) {
                        // Add the used variables as null and hope that they will be resolved
                        depth += 1;
                        syms.push({var: x, val: null, depth: depth});
                    }
                }
                else if(mdata.type == "subsets.Range") {
                    // These are ranges
                    
                    // These ranges are not of interest, as they specify what is copied and don't define new variables 
                }

                // Find parent nodes
                let parent = sdfg.nodes[state_id].nodes[m.src];
                let tmp = SDFG_Parser.lookup_symbols(sdfg, state_id, m.src, symbols_to_resolve, depth + 1);
                syms = [...syms, ...tmp];
            }
        }
        else {
            // Node
            let node = state.nodes[elem];

            // Maps (and Consumes) define ranges, extract symbols from there
            try {
                let rngs = node.attributes.range.ranges.map(x => x); // The iterator ranges
                let params = node.attributes.params.map(x => x); // The iterators

                console.assert(rngs.length == params.length, "Ranges and params should have the same count of elements");

                // Reverse from big -> little to little -> big (or outer -> inner to inner -> outer)
                rngs.reverse();
                params.reverse();

                for(let i = 0; i < rngs.length; ++i) {
                    // Check first if the variable is already defined, and if yes, if the value is the same
                    let fltrd = syms.filter(x => x.var == params[i]);
                    if(fltrd.length == 0) {
                        depth += 1;
                        syms.push({var: params[i], val: rngs[i], depth: depth});
                    }
                    else {
                        if(JSON.stringify(fltrd[0].val) != JSON.stringify(rngs[i])) {
                            console.warn("Colliding definitions for var " + params[i], fltrd[0].val, rngs[i]);
                        }
                    }
                }
            }
            catch(e) {
                // Not a node defining ranges (every node except maps / consumes)
            }
            // Find all incoming edges
            let inc_edges = state.edges.filter(x => x.dst == elem);
            for(let e of inc_edges) {
                let tmp = SDFG_Parser.lookup_symbols(sdfg, state_id, e, symbols_to_resolve, depth + 1);
                syms = [...syms, ...tmp];
            }
        }
        
        return syms;
    }
}

class SDFG_State_Parser {
    constructor(sdfg_state) {
        this._sdfg_state = sdfg_state;
    }

    getNodes() {
        return this._sdfg_state.nodes.map(x => new SDFG_Node_Parser(x));
    }
}

class SDFG_Node_Parser {
    constructor(node) {
        this._node = node;
    }

    isNodeType(node_type) {
        return this._node.attributes.type === node_type;
    }
}

class SDFG_PropUtil {
    static getMetaFor(obj, attr_name) {
        return obj.attributes['_meta_' + attr_name];
    }

    static getAttributeNames(obj) {
        let keys = Object.keys(obj.attributes);
        let list = keys.filter(x => keys.includes('_meta_' + x));
        return list;
    }
}

export {SDFG_Parser, SDFG_State_Parser, SDFG_Node_Parser, SDFG_PropUtil}