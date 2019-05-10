class SDFG_Parser {
    constructor(sdfg) {
        this._sdfg = sdfg;
    }

    getStates() {

        return this._sdfg.nodes.map(x => new SDFG_State_Parser(x));
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

export {SDFG_Parser, SDFG_State_Parser, SDFG_Node_Parser}