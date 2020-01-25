import {REST_request, FormBuilder, setup_drag_n_drop} from "./main.js"
import {Appearance} from "./diode_appearance.js"
import {SDFG_Parser, SDFG_PropUtil} from "./sdfg_parser.js"
import * as DiodeTables from "./table.js"
import * as Roofline from "./renderer_dir/Roofline/main.js"

class DIODE_Settings {
    constructor(denormalized = {}) {
        this.settings_values = denormalized;
        this.changed = {};
    }

    load() {
        // Load the settings from localStorage
        this.settings_values = window.localStorage.getItem("DIODE/Settings/confirmed");
        this.changed = window.localStorage.getItem("DIODE/Settings/changed");
    }

    store() {
        // Store the settings to localStorage
        window.localStorage.setItem("DIODE/Settings/confirmed", this.settings_values);
        window.localStorage.setItem("DIODE/Settings/changed", this.changed);
    }

    change(setting, value) {
        this.changed[setting.join('/')] = value;
    }

    hasChanged() {
        return this.changed != {};
    }

    changedValues() {
        return this.changed;
    }

    clearChanged() {
        this.changed = {};
    }

    values() {
        return this.settings_values;
    }

};

class DIODE_Context {
    constructor(diode, gl_container, state) {
        this.diode = diode;
        this.container = gl_container;
        this.initial_state = state;

        this.created = state.created;
        if (this.created == undefined) {
            this.created = this.diode.getPseudorandom();
            this.saveToState({created: this.created});
        }
        this._project = null;
        if (state.project_id != undefined && state.project_id != null && state.project_id != "null") {
            this._project = new DIODE_Project(this.diode, state.project_id);
        }

        this._event_listeners = [];
        this._event_listeners_closed = []; // Listeners that are installed on closed windows (NOTE: These are NOT active on open windows)

        this._no_user_destroy = false; // When true, all destroy()-events are assumed to be programmatic
    }

    project() {
        console.assert(this._project != null, "Project invalid");
        return this._project;
    }

    on(name, data, keep_alive_when_closed = false) {
        let eh = this.diode.goldenlayout.eventHub;

        let params = [name, data];
        eh.on(...params);
        this._event_listeners.push(params);

        if (keep_alive_when_closed) {
            // NOTE: The new function has to be created because the function cannot be identical
            // (This is, because the handler is removed by (name, function) pairs)
            this.closed_on(name, (x) => data(x));
        }
    }

    // same as on(), but only active when the window is closed (in the closed windows list)
    closed_on(name, data) {


        let params = [name, data];
        this._event_listeners_closed.push(params);
    }

    removeClosedWindowEvents() {
        // This function has to be called when reopening from the closed windows list
        // DO NOT call inside destroy()!
        let eh = this.diode.goldenlayout.eventHub;
        for (let x of this._event_listeners_closed) {
            eh.unbind(...x);
        }
        this._event_listeners_closed = [];
    }

    destroy() {

        console.log("destroying", this);
        // Unlink all event listeners
        let eh = this.diode.goldenlayout.eventHub;
        for (let x of this._event_listeners) {
            eh.unbind(...x);
        }
        this._event_listeners = [];
    }

    close() {
        /*
            The difference to destroy: close() is called when the USER clicks close,
            destroy will be called afterwards when the element is actually removed
        */

        // Add to closed-windows list
        console.log("closing", this);

        this.project().addToClosedWindowsList(this.container._config.componentName, this.getState());

        // Install the event listeners to listen when the window is closed
        let eh = this.diode.goldenlayout.eventHub;
        for (let params of this._event_listeners_closed) {
            eh.on(...params);
        }
    }

    setupEvents(project) {
        if (this._project == null) {
            this._project = project;
        }

        this.container.extendState({'project_id': this._project.eventString('')})

        this.on('destroy-' + this.getState().created, msg => {
            if (!this._no_user_destroy) {
                // _might_ be a user destroy - call close
                this.close();
            }
            this.destroy();
        });

        this.on('enter-programmatic-destroy', msg => {
            this._no_user_destroy = true;
            console.log("Entering programmatic reordering", this);
        });
        this.on('leave-programmatic-destroy', msg => {
            this._no_user_destroy = false;
            console.log("Leaving programmatic reordering", this);
        });

        this.closed_on('window-reopened-' + this.getState().created, x => {
            this.removeClosedWindowEvents();
        });
    }

    getState() {
        return this.container.getState();
    }

    saveToState(dict_value) {
        this.container.extendState(dict_value);
    }

    resetState(dict_value = {}) {
        this.container.setState(dict_value);
    }

    saveToPersistentState(key, value) {
        localStorage.setItem(key, value);
    }

    getPersistentState(key) {
        return localStorage.getItem(key);
    }
}


class DIODE_Context_SDFG extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);

        this._message_handler = x => alert(x);

        this.renderer_pane = null;

        this._analysis_values = {};

        console.log("state", state);
    }

    saveToState(dict_value) {
        let renamed_dict = {};
        let json_list = ['sdfg_data', 'sdfg'];
        for (let x of Object.entries(dict_value)) {
            renamed_dict[x[0]] = (json_list.includes(x[0]) && typeof (x[1]) != 'string') ? JSON.stringify(x[1]) : x[1];
        }
        super.saveToState(renamed_dict);

        console.assert(this.getState().sdfg == undefined);
    }

    resetState(dict_value) {
        let renamed_dict = {};
        let json_list = ['sdfg_data', 'sdfg'];
        for (let x of Object.entries(dict_value)) {
            renamed_dict[x[0]] = (json_list.includes(x[0]) && typeof (x[1]) != 'string') ? JSON.stringify(x[1]) : x[1];
        }
        super.resetState(renamed_dict);
        console.assert(this.getState().sdfg == undefined);
    }

    getState() {
        let _state = super.getState();
        let _transformed_state = {};
        let json_list = ['sdfg_data', 'sdfg'];
        for (let x of Object.entries(_state)) {
            _transformed_state[x[0]] = (typeof (x[1]) == 'string' && json_list.includes(x[0])) ? JSON.parse(x[1]) : x[1];
        }
        return _transformed_state;
    }

    setupEvents(project) {
        super.setupEvents(project);

        let transthis = this;

        let eh = this.diode.goldenlayout.eventHub;
        this.on(this._project.eventString('-req-new-sdfg'), (msg) => {

            if (typeof (msg) == 'string') msg = parse_sdfg(msg);
            if (msg.sdfg_name === this.getState()['sdfg_name']) {
                // Ok
            } else {
                // Names don't match - don't replace this one then.
                // #TODO: This means that renamed SDFGs will not work as expected.
                return;
            }
            setTimeout(() => eh.emit(this.project().eventString('new-sdfg'), 'ok'), 1);
            this.create_renderer_pane(msg, true);
        });

        // #TODO: When multiple sdfgs are present, the event name
        // should include a hash of the target context
        this.on(this._project.eventString('-req-sdfg-msg'), msg => {

            let ret = this.message_handler_filter(msg);
            if (ret === undefined) {
                ret = 'ok';
            }
            setTimeout(() => eh.emit(transthis._project.eventString('sdfg-msg'), ret), 1);
        });

        this.on(this._project.eventString('-req-sdfg_props'), msg => {
            // Echo with data
            if (msg != undefined) {
                this.discardInvalidated(msg);
            }
            let resp = this.getChangedSDFGPropertiesFromState();
            let named = {};
            named[this.getState()['sdfg_name']] = resp;
            setTimeout(() => eh.emit(transthis._project.eventString('sdfg_props'), named), 1);
        }, true);

        this.on(this.project().eventString('-req-property-changed-' + this.getState().created), (msg) => {
            // Emit ok directly (to avoid caller timing out)
            setTimeout(() => eh.emit(this.project().eventString("property-changed-" + this.getState().created), "ok"), 1);

            if (msg.type == "symbol-properties") {
                this.symbolPropertyChanged(msg.element, msg.name, msg.value);
            } else
                this.propertyChanged(msg.element, msg.name, msg.value);

        }, true);

        this.on(this.project().eventString('-req-delete-data-symbol-' + this.getState().created), (msg) => {
            setTimeout(() => eh.emit(this.project().eventString("delete-data-symbol-" + this.getState().created), "ok"), 1);

            this.removeDataSymbol(msg);
        });

        this.on(this.project().eventString('-req-add-data-symbol-' + this.getState().created), (msg) => {
            setTimeout(() => eh.emit(this.project().eventString("add-data-symbol-" + this.getState().created), "ok"), 1);

            this.addDataSymbol(msg.type, msg.name);
        });

        this.on(this.project().eventString('-req-draw-perfinfo'), (msg) => {
            setTimeout(() => eh.emit(transthis._project.eventString('draw-perfinfo'), "ok"), 1);
            this._analysis_values = msg.map(x => ({
                forProgram: x[0],
                AnalysisName: x[1],
                runopts: x[2],
                forUnifiedID: x[3],
                forSuperSection: x[4],
                forSection: x[5],
                data: JSON.parse(x[6]),
            }));
            this.renderer_pane.drawAllPerfInfo();
        });

        this.on(this.project().eventString('-req-sdfg_object'), msg => {
            // Return the entire serialized SDFG
            let _state = this.getSDFGDataFromState();
            let sdfg = _state.type == 'SDFG' ? _state : _state.sdfg;
            let named = {};
            named[this.getState()['sdfg_name']] = sdfg;
            //named = JSON.stringify(named);
            setTimeout(() => eh.emit(this.project().eventString("sdfg_object"), named), 1);
        }, true);
    }

    removeDataSymbol(aname) {

        let o = this.getSDFGDataFromState();
        let sdfg = o['sdfg'];

        let found = false;
        for (let x of Object.keys(sdfg.attributes._arrays)) {
            if (x == aname) {
                // Matching name
                delete sdfg.attributes._arrays[x];
                found = true;
                break;
            }
        }
        if (!found) console.error("Did not find symbol " + name + " in SDFG, this is a fatal error");

        let old = this.getState();
        if (old.type == "SDFG")
            console.error("Defensive programming no longer allowed; change input");
        else
            old.sdfg_data.sdfg = sdfg;

        this.resetState(old);

        this.diode.refreshSDFG();
    }

    addDataSymbol(type, aname) {

        if (aname == "") {
            alert("Invalid symbol name. Enter a symbol name in the input field");
            return;
        }

        // Create a dummy element, then allow changing later
        let typestr = "";
        if (type == "Scalar") typestr = "Scalar";
        else if (type == "Array") typestr = "Array";
        let data = {
            type: typestr,
            attributes: {
                dtype: "int32",
            }
        };

        let o = this.getSDFGDataFromState();
        let sdfg = o['sdfg'];

        let found = false;
        for (let x of Object.keys(sdfg.attributes._arrays)) {
            if (x == aname) {
                // Matching name
                found = true;
                break;
            }
        }
        if (found) {
            this.diode.toast("Cannot add symbol", "A symbol with name " + aname + " does already exist.", "error", 3000);
            return;
        }

        sdfg.attributes._arrays[aname] = data;
        let old = this.getState();
        if (old.type == "SDFG")
            console.error("Defensive programming no longer allowed; change input");
        else
            old.sdfg_data.sdfg = sdfg;

        this.resetState(old);


        this.diode.refreshSDFG();
    }

    analysisProvider(aname, nodeinfo) {

        let unified_id = null;
        if (nodeinfo != null) {
            unified_id = (parseInt(nodeinfo.stateid) << 16) | parseInt(nodeinfo.nodeid);
        }
        console.log("analysisProvider", aname, nodeinfo);
        if (aname == "getstates") {

            let states = this._analysis_values.map(x => (x.forUnifiedID >> 16) & 0xFFFF);
            return states;
        } else if (aname == "getnodes") {
            let nodes = this._analysis_values.map(x => (x.forUnifiedID) & 0xFFFF);
            return nodes;
        } else if (aname == "all_vec_analyses") {
            let vec_analyses = this._analysis_values.filter(x => x.AnalysisName == 'VectorizationAnalysis');
            let fltrd_vec_analyses = vec_analyses.filter(x => x.forUnifiedID == unified_id);
            return fltrd_vec_analyses;
        } else if (aname == 'CriticalPathAnalysis') {
            let cpa = this._analysis_values.filter(x => x.AnalysisName == 'CriticalPathAnalysis');
            let filtered = cpa.filter(x => x.forUnifiedID == unified_id);
            return filtered;
        } else if (aname == 'ParallelizationAnalysis') {
            let pa = this._analysis_values.filter(x => x.AnalysisName == 'ThreadAnalysis');
            let filtered = pa.filter(x => x.forUnifiedID == unified_id);
            return filtered;
        } else if (aname == 'MemoryAnalysis') {
            let ma = this._analysis_values.filter(x => x.AnalysisName == 'MemoryAnalysis');
            let filtered = ma.filter(x => x.forUnifiedID == unified_id);
            return filtered;
        } else if (aname == 'MemOpAnalysis') {
            let moa = this._analysis_values.filter(x => x.AnalysisName == 'MemoryOpAnalysis');
            let filtered = moa.filter(x => x.forUnifiedID == unified_id);
            return filtered;
        } else if (aname == 'CacheOpAnalysis') {
            let coa = this._analysis_values.filter(x => x.AnalysisName == 'CacheOpAnalysis');
            let filtered = coa.filter(x => x.forUnifiedID == unified_id);
            return filtered;
        } else if (aname == "defaultRun") {
            // This pseudo-element returns a filter function that returns only elements from the "default" run configuration
            // #TODO: Make the default run configurable.
            // For now, the default run is the run with the most committed cores
            //return x => x.filter(y => y.runopts == '# ;export OMP_NUM_THREADS=4; Running in multirun config');
            return x => {
                let tmp = x.map(y => {
                    let r = [];
                    y.runopts.replace(/OMP_NUM_THREADS=(\d+)/gm, (m, p) => r.push(p));
                    return r;
                });
                let max_num = Math.max(...tmp.map(x => parseInt(x)));
                return x.filter(z => z.runopts == '# ;export OMP_NUM_THREADS=' + max_num + '; Running in multirun config');
            };
        } else {
            throw "#TODO";
        }
    }

    message_handler_filter(msg) {
        /* 
            This function is a compatibility layer
        */
        msg = JSON.parse(msg);
        if (msg.sdfg_name != this.getState()['sdfg_name']) {
            return;
        }
        if (msg.type == 'clear-highlights') {
            if (this.highlighted_elements)
                this.highlighted_elements.forEach(e => {
                    if (e) e.stroke_color = null;
                });
            this.highlighted_elements = [];
            this.renderer_pane.draw_async();
        }
        if (msg.type == 'highlight-elements') {
            // Clear previously highlighted elements
            if (this.highlighted_elements)
                this.highlighted_elements.forEach(e => {
                    if (e) e.stroke_color = null;
                });
            this.highlighted_elements = [];

            let graph = this.renderer_pane.graph;

            // The input contains a list of multiple elements
            for (let x of msg.elements) {
                let sid = x[0], nid = x[1];
                let elem = null;
                if (sid == -1)
                    elem = graph.node(nid);
                else
                    elem = graph.node(sid).data.graph.node(nid);
                this.highlighted_elements.push(elem);
            }
            this.highlighted_elements.forEach(e => {
                if (e) e.stroke_color = "#D35400";
            });
            this.renderer_pane.draw_async();
        } else {
            // Default behavior is passing through (must be an object, not JSON-string)
            //this._message_handler(msg);
        }
    }

    // Returns a goldenlayout component if exists, or null if doesn't
    has_component(comp_name, parent = null) {
        // If parent component not provided, use root window
        if (!parent)
            parent = this.diode.goldenlayout.root;
        if ('componentName' in parent && parent.componentName === comp_name)
            return parent;
        if ('contentItems' in parent) {
            for (let ci of parent.contentItems) {
                let result = this.has_component(comp_name, ci);
                if (result) return result;
            }
        }
        return null;
    }

    render_free_variables(force_open) {
        let sdfg_dat = this.getSDFGDataFromState();
        if (sdfg_dat.type != "SDFG") sdfg_dat = sdfg_dat.sdfg;
        this.diode.replaceOrCreate(['render-free-vars'], 'PropWinComponent', {
                data: sdfg_dat,
                calling_context: this.created
            },
            () => {
                console.log("Calling recreation function");
                let config = {
                    type: 'component',
                    componentName: 'PropWinComponent',
                    componentState: {}
                };

                this.diode.addContentItem(config);
                setTimeout(() => this.render_free_variables(force_open), 1);
            });
    }

    merge_properties(node_a, aprefix, node_b, bprefix) {
        /*  Merges node_a and node_b into a single node, such that the rendered properties are identical
            when selecting either node_a or node_b.
        */
        let en_attrs = SDFG_PropUtil.getAttributeNames(node_a);
        let ex_attrs = SDFG_PropUtil.getAttributeNames(node_b);

        let new_attrs = {};

        for (let na of en_attrs) {
            let meta = SDFG_PropUtil.getMetaFor(node_a, na);
            if (meta.indirected) {
                // Most likely shared. Don't change.
                new_attrs['_meta_' + na] = meta;
                new_attrs[na] = node_a.attributes[na];
            } else {
                // Private. Add, but force-set a new Category (in this case, MapEntry)
                let mcpy = JSON.parse(JSON.stringify(meta));
                mcpy['category'] = node_a.type + " - " + mcpy['category'];
                new_attrs['_meta_' + aprefix + na] = mcpy;
                new_attrs[aprefix + na] = node_a.attributes[na];
            }

        }
        // Same for ex_attrs, but don't add if shared
        for (let xa of ex_attrs) {
            let meta = SDFG_PropUtil.getMetaFor(node_b, xa);
            if (!meta.indirected) {
                let mcpy = JSON.parse(JSON.stringify(meta));
                mcpy['category'] = node_b.type + " - " + mcpy['category'];
                mcpy['_noderef'] = node_b.node_id;
                new_attrs['_meta_' + bprefix + xa] = mcpy;
                new_attrs[bprefix + xa] = node_b.attributes[xa];
            }
        }
        // Copy the first node for good measure
        // TODO: Inhibits property update for map/array
        let ecpy = JSON.parse(JSON.stringify(node_a));
        ecpy.attributes = new_attrs;
        return ecpy;
    }

    getSDFGPropertiesFromState() {
        let o = this.getSDFGDataFromState();
        let props = o['sdfg_props'];

        return props;
    }

    getSDFGDataFromState() {
        let _state = this.getState();
        let o = null;
        if (_state.sdfg != undefined) {
            o = _state;
        } else {
            o = _state['sdfg_data'];
        }
        if ((typeof o) == 'string') {
            o = JSON.parse(o);
        }
        while (typeof o.sdfg == 'string') {
            o.sdfg = JSON.parse(o.sdfg);
        }
        return o;
    }

    renderProperties(node) {
        /*
            node: object, duck-typed
                
        */

        let params = node.data;
        let transthis = this;

        // Render in the (single, global) property window
        this.diode.renderPropertiesInWindow(transthis, node, params);
    }

    getSDFGElementReference(node_id, state_id) {
        if (node_id != null && node_id.constructor == Object) {
            return this.getEdgeReference(node_id, state_id);
        } else {
            return this.getNodeReference(node_id, state_id);
        }
    }

    getEdgeReference(node_id, state_id) {
        let o = this.getSDFGDataFromState();
        let sdfg = o['sdfg'];

        if (state_id == undefined) {
            // Look for sdfg-level edges
            for (let e of sdfg.edges) {
                if (e.src == node_id.src && e.dst == node_id.dst) {
                    return [e.attributes.data, sdfg];
                }
            }
        }

        for (let x of sdfg.nodes) {
            if (x.id == state_id) {
                for (let e of x.edges) {

                    if (e.src == node_id.src && e.dst == node_id.dst) {
                        return [e.attributes.data, sdfg];
                    }
                }

                break;
            }
        }
    }

    getNodeReference(node_id, state_id) {
        let o = this.getSDFGDataFromState();
        let sdfg = o['sdfg'];

        for (let x of sdfg.nodes) {
            if (x.id == state_id) {

                if (node_id == null) return [x, sdfg];
                for (let n of x.nodes) {

                    if (n.id == node_id) {
                        return [n, sdfg];
                    }
                }

                break;
            }
        }
    }


    symbolPropertyChanged(node, name, value) {
        /*
            A data symbol was changed.
        */
        console.log("symbolPropertyChanged", name, value);

        // Search arrays first
        let o = this.getSDFGDataFromState();
        let sdfg = o['sdfg'];

        let found = false;
        let d = node.data();
        for (let x of Object.keys(sdfg.attributes._arrays)) {
            if (x == d[0]) {
                // Matching name
                sdfg.attributes._arrays[x].attributes[name] = value;
                found = true;
                break;
            }
        }
        if (!found) console.error("Did not find symbol " + name + " in SDFG, this is a fatal error");

        let old = this.getState();
        if (old.type == "SDFG")
            console.error("Defensive programming no longer allowed; change input");
        else
            old.sdfg_data.sdfg = sdfg;

        this.resetState(old);

        this.diode.refreshSDFG();
    }

    propertyChanged(node, name, value) {
        /*
            When a node-property is changed, the changed data is written back
            into the state.
        */
        let nref = node.element;
        let sdfg = node.sdfg;

        nref.attributes[name] = value;

        let old = this.getState();
        if (old.type == "SDFG")
            old = sdfg;
        else
            old.sdfg_data.sdfg = sdfg;

        this.resetState(old);

        this.diode.refreshSDFG();
    }


    create_renderer_pane(sdfg_data = undefined, update = false) {
        if (sdfg_data == undefined) {
            sdfg_data = this.getState()["sdfg_data"];
        }
        let tmp = sdfg_data;
        if ((typeof tmp) === 'string') {
            tmp = parse_sdfg(sdfg_data);
        } else {
            if ('sdfg' in sdfg_data)
                tmp = sdfg_data;
            else
                tmp = {sdfg: sdfg_data};
        }

        {
            // Load the properties from json instead of loading the old properties
            // This means deleting any property delivered
            // This is just so we don't accidentally use the old format
            console.assert(tmp.sdfg_props === undefined);

        }

        {
            // Reset the state to avoid artifacts
            let s = this.getState();
            console.assert(s.sdfg_data != undefined);
            delete s.sdfg_data;
            this.resetState(s);
        }
        this.saveToState({
            "sdfg_data": tmp
        });

        if (this.renderer_pane !== null)
            this.renderer_pane.set_sdfg(tmp.sdfg);
        else {
            let sdfv = new SDFGRenderer(tmp.sdfg, this.container.getElement()[0],
                (et, e, c, el, r, fge) => this.on_renderer_mouse_event(et, e, c, el, r, fge));
            this.renderer_pane = sdfv;
        }

        // Display data descriptors by default (in parallel to the creation of the renderer)
        this.render_free_variables(true);
    }

    on_renderer_mouse_event(evtype, event, canvas_coords, elements, renderer, foreground_elem) {
        let state_only = false;
        let clicked_states = elements.states;
        let clicked_nodes = elements.nodes;
        let clicked_edges = elements.edges;
        let clicked_interstate_edges = elements.isedges;
        let clicked_connectors = elements.connectors;
        let total_elements = clicked_states.length + clicked_nodes.length + clicked_edges.length +
            clicked_interstate_edges.length + clicked_connectors.length;

        // Clear context menu
        if (evtype === 'click' || evtype === 'doubleclick' || evtype === 'mousedown' || evtype === 'contextmenu' ||
            evtype === 'wheel') {
            if (this.contextmenu) {
                this.contextmenu.destroy();
                this.contextmenu = null;
            }
        }

        // Check if anything was clicked at all
        if (total_elements == 0 && evtype === 'click') {
            // Clear highlighted elements
            if (this.highlighted_elements)
                this.highlighted_elements.forEach(e => {
                    if (e) e.stroke_color = null;
                });

            // Nothing was selected
            this.render_free_variables(false);
            return true;
        }
        if (total_elements == 0 && evtype === 'contextmenu') {
            let cmenu = new ContextMenu();
            cmenu.addOption("SDFG Properties", x => {
                this.render_free_variables(true);
            });
            cmenu.show(event.x, event.y);
            this.contextmenu = cmenu;
            return false;
        }

        if ((clicked_nodes.length + clicked_edges.length + clicked_interstate_edges.length) === 0) {
            // A state was selected
            if (clicked_states.length > 0)
                state_only = true;
        }

        let state_id = null, node_id = null;
        if (clicked_states.length > 0)
            state_id = clicked_states[0].id;
        if (clicked_interstate_edges.length > 0)
            node_id = clicked_interstate_edges[0].id;

        if (clicked_nodes.length > 0)
            node_id = clicked_nodes[0].id;
        else if (clicked_edges.length > 0)
            node_id = clicked_edges[0].id;


        if (evtype === "contextmenu") {
            // Context menu was requested
            let spos = {x: event.x, y: event.y};
            let sdfg_name = renderer.sdfg.attributes.name;

            let cmenu = new ContextMenu();

            ///////////////////////////////////////////////////////////
            // Collapse/Expand
            let sdfg = (foreground_elem ? foreground_elem.sdfg : null);
            let sdfg_elem = null;
            if (foreground_elem instanceof State)
                sdfg_elem = foreground_elem.data.state;
            else if (foreground_elem instanceof Node) {
                sdfg_elem = foreground_elem.data.node;

                // If a scope exit node, use entry instead
                if (sdfg_elem.type.endsWith("Exit"))
                    sdfg_elem = sdfg.nodes[foreground_elem.parent_id].nodes[sdfg_elem.scope_entry];
            } else
                sdfg_elem = null;

            // Toggle collapsed state
            if (sdfg_elem && 'is_collapsed' in sdfg_elem.attributes) {
                cmenu.addOption((sdfg_elem.attributes.is_collapsed) ? 'Expand' : 'Collapse',
                    x => {
                        sdfg_elem.attributes.is_collapsed = !sdfg_elem.attributes.is_collapsed;
                        this.renderer_pane.relayout();
                        this.renderer_pane.draw_async();
                    });
            }
            ///////////////////////////////////////////////////////////


            cmenu.addOption("Show transformations", x => {
                console.log("'Show transformations' was clicked");

                this.project().request(['highlight-transformations-' + sdfg_name], x => {
                }, {
                    params: {
                        state_id: state_id,
                        node_id: node_id
                    }
                });
            });
            cmenu.addOption("Apply transformation \u25B6", x => {
                console.log("'Apply transformation' was clicked");

                // Find available transformations for this node
                this.project().request(['get-transformations-' + sdfg_name], x => {
                    console.log("get-transformations response: ", x);

                    let tmp = Object.values(x)[0];

                    // Create a sub-menu at the correct position


                    let submenu = new ContextMenu();

                    for (let y of tmp) {
                        submenu.addOption(y.opt_name, x => {
                            this.project().request(['apply-transformation-' + sdfg_name], x => {
                                },
                                {
                                    params: y.id_name
                                });
                        });
                    }

                    submenu.show(spos.x + cmenu.width(), spos.y);
                }, {
                    params: {
                        state_id: state_id,
                        node_id: node_id
                    }
                });


                // Don't close the current context menu from this event
                x.preventDefault();
                x.stopPropagation();
            });
            cmenu.addOption("Show Source Code", x => {
                console.log("go to source code");
            });
            cmenu.addOption("Show Generated Code", x => {
                console.log("go to generated code");
            });
            cmenu.addOption("Properties", x => {
                console.log("Force-open property pane");
            });

            cmenu.show(spos.x, spos.y);
            this.contextmenu = cmenu;

            return false;
        }

        if (evtype !== "click")
            return false;

        // Clear highlighted elements
        if (this.highlighted_elements)
            this.highlighted_elements.forEach(e => {
                if (e) e.stroke_color = null;
            });
        // Mark this element red
        this.highlighted_elements = [foreground_elem];

        // Render properties asynchronously
        setTimeout(() => {
            // Get and render the properties from now on
            console.log("sdfg", foreground_elem.sdfg);

            let dst_nodeid = null;
            if (foreground_elem instanceof Edge && foreground_elem.parent_id !== null) {
                let edge = foreground_elem.sdfg.nodes[state_id].edges[foreground_elem.id];
                dst_nodeid = edge.dst;
            }

            let render_props = element_list => {
                let properties = [];
                element_list.forEach(element => {
                    // Collect all properties and metadata for each element
                    let attr = element.attributes;
                    let akeys = Object.keys(attr).filter(x => !x.startsWith("_meta_"));

                    for (let k of akeys) {
                        let value = attr[k];
                        let meta = attr["_meta_" + k];
                        if (meta == undefined)
                            continue;

                        let pdata = JSON.parse(JSON.stringify(meta));
                        pdata.value = value;
                        pdata.name = k;

                        properties.push({
                            property: pdata, element: element, sdfg: foreground_elem.sdfg,
                            category: element.type + ' - ' + pdata.category
                        });
                    }
                });
                this.renderProperties({
                    data: properties
                });
            };

            if (foreground_elem instanceof Edge)
                render_props([foreground_elem.data]);
            else if (foreground_elem instanceof Node) {
                let n = foreground_elem.data.node;
                // Set state ID, if exists
                n.parent_id = foreground_elem.parent_id;
                let state = foreground_elem.sdfg.nodes[foreground_elem.parent_id];
                // Special case treatment for scoping nodes (i.e. Maps, Consumes, ...)
                if (n.type.endsWith("Entry")) {
                    // Find the matching exit node
                    let exit_node = find_exit_for_entry(state.nodes, n);

                    // Highlight both entry and exit nodes
                    let gstate = renderer.graph.node(foreground_elem.parent_id);
                    let rnode = gstate.data.graph.node(exit_node.id);
                    this.highlighted_elements.push(rnode);

                    render_props([n, exit_node]);
                } else if (n.type.endsWith("Exit")) {
                    // Find the matching entry node and continue with that
                    let entry_id = parseInt(n.scope_entry);
                    let entry_node = state.nodes[entry_id];

                    // Highlight both entry and exit nodes
                    let gstate = renderer.graph.node(foreground_elem.parent_id);
                    let rnode = gstate.data.graph.node(entry_node.id);
                    this.highlighted_elements.push(rnode);

                    render_props([entry_node, n]);
                } else if (n.type === "AccessNode") {
                    // Find matching data descriptor and show that as well
                    let ndesc = foreground_elem.sdfg.attributes._arrays[n.attributes.data];
                    render_props([n, ndesc]);
                } else
                    render_props([n]);
            } else if (foreground_elem instanceof State)
                render_props([foreground_elem.data.state]);

            this.highlighted_elements.forEach(e => {
                if (e) e.stroke_color = "red";
            });
            renderer.draw_async();
        }, 0);

        // Timeout handler draws asynchronously
        return false;
    }
}


class DIODE_Context_TransformationHistory extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);

    }

    setupEvents(project) {
        super.setupEvents(project);

        let eh = this.diode.goldenlayout.eventHub;

        this.on(this.project().eventString('-req-update-tfh'), msg => {

            // Load from project
            let hist = this.project().getTransformationHistory();
            this.create(hist);
            setTimeout(() => eh.emit(this.project().eventString('update-tfh'), 'ok'), 1);
        });
    }

    create(hist = []) {
        let parent_element = this.container.getElement();
        $(parent_element).css('overflow', 'auto');
        $(parent_element)[0].setAttribute("data-hint", '{"type": "DIODE_Element", "name": "TransformationHistory"}');

        parent_element = $(parent_element)[0];

        parent_element.innerHTML = '';

        let history_base_div = document.createElement("div");
        history_base_div.classList = "transformation_history_base";

        let history_scoll_div = document.createElement("div");
        history_scoll_div.classList = "transformation_history_list";

        this._history_scroll_div = history_scoll_div;

        history_base_div.appendChild(history_scoll_div);

        parent_element.innerHTML = "";
        parent_element.appendChild(history_base_div);

        let i = 0;
        for (let x of hist) {
            this.addElementToHistory(x, i);
            ++i;
        }
    }

    addElementToHistory(simple_node, index) {
        let hsd = this._history_scroll_div;

        let elem = document.createElement("div");
        elem.classList = "transformation_history_list_element";

        let title = document.createElement("div");
        title.classList = "transformation_history_list_element_title";
        title.innerText = Object.values(simple_node)[0][0].name;

        let ctrl = document.createElement("div");
        ctrl.classList = "flex_row transformation_history_list_element_control";

        {
            let revert = document.createElement("div");
            revert.classList = "revert-button";
            revert.title = "revert";
            revert.innerHTML = "<i class='material-icons'>undo</i>";
            $(revert).hover(() => {
                elem.classList.add("revert-hovered");
            }, () => {
                elem.classList.remove("revert-hovered");
            })

            revert.addEventListener('click', _x => {
                // Reset to the associated checkpoint
                let tsh = this.project().getTransformationSnapshots()[index];
                this.diode.multiple_SDFGs_available({compounds: tsh[1]});

                // Remove the descending checkpoints
                this.project().discardTransformationsAfter(index);

                if (true) {
                    this.diode.gatherProjectElementsAndCompile(this, {}, {
                        sdfg_over_code: true,

                    });
                }
            });

            ctrl.appendChild(revert);
        }

        elem.appendChild(title);
        elem.appendChild(ctrl);
        hsd.appendChild(elem);

    }
}


class DIODE_Context_AvailableTransformations extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);

        this._tree_view = null;
        this._current_root = null;

        // Allow overflow
        let parent_element = this.container.getElement();
        $(parent_element).css('overflow', 'auto');

        this.operation_running = false;
    }

    setupEvents(project) {
        super.setupEvents(project);

        let transthis = this;

        let eh = this.diode.goldenlayout.eventHub;
        this.on(this._project.eventString('-req-extend-optgraph'), (msg) => {

            let o = msg;
            if (typeof (o) == "string") {
                JSON.parse(msg);
            }
            let sel = o[this.getState()['for_sdfg']];
            if (sel === undefined) {
                return;
            }
            setTimeout(() => eh.emit(transthis._project.eventString('extend-optgraph'), 'ok'), 1);

            this.create(sel);
        });

        this.on(this._project.eventString('-req-optpath'), msg => {
            let named = {};
            named[this.getState()['for_sdfg']] = [];
            setTimeout(() => eh.emit(transthis._project.eventString('optpath'), named), 1);
        });

        let sname = this.getState()['for_sdfg'];
        this.on(this._project.eventString('-req-new-optgraph-' + sname), msg => {
            // In any case, inform the requester that the request will be treated
            let o = JSON.parse(msg);
            let sel = o.matching_opts;
            if (sel === undefined) {
                //eh.emit(transthis._project.eventString('new-optgraph-' + sname), 'not ok');
                return;
            }
            setTimeout(() => eh.emit(transthis._project.eventString('new-optgraph-' + sname), 'ok'), 1);

            this.create(o);
        });

        this.on(this.project().eventString('-req-highlight-transformations-' + sname), msg => {

            this.getTransformations(msg).forEach(x => this.highlightTransformation(x));

        });

        this.on(this.project().eventString('-req-get-transformations-' + sname), msg => {

            let transforms = this.getTransformations(msg);

            setTimeout(() => eh.emit(transthis._project.eventString('get-transformations-' + sname), transforms), 1);
        });

        this.on(this.project().eventString('-req-apply-transformation-' + sname), msg => {

            let children = this.getState()['optstruct'];
            for (let c of Object.values(children)) {
                for (let d of c) {

                    if (d === undefined) continue;
                    if (d.id_name == msg) {
                        // Call directly.
                        // (The click handler invokes the simple transformation)
                        d.representative.dispatchEvent(new Event('click'));
                    }
                }
            }
        });

        this.on(this.project().eventString('-req-locate-transformation-' + sname), msg => {
            this.locateTransformation(...JSON.parse(msg));
        });

        this.on(this.project().eventString('-req-apply-adv-transformation-' + sname), msg => {

            let x = JSON.parse(msg);
            this.applyTransformation(...x);
        });
        this.on(this.project().eventString('-req-append-history'), msg => {
            this.appendHistoryItem(msg.new_sdfg, msg.item_name);
        });

        this.on(this.project().eventString('-req-property-changed-' + this.getState().created), (msg) => {
            this.propertyChanged(msg.element, msg.name, msg.value);
            setTimeout(() => eh.emit(this.project().eventString("property-changed-" + this.getState().created), "ok"), 1);
        });
    }

    getTransformations(affecting) {
        let ret = [];
        let selstring = "s" + affecting.state_id + "_" + affecting.node_id;
        let children = this.getState()['optstruct'];
        for (let c of Object.values(children)) {
            for (let d of c) {

                if (d === undefined) continue;
                let affects = d.affects;

                if (affects.includes(selstring)) {
                    ret.push(d);
                }
            }
        }
        return ret;
    }

    highlightTransformation(node) {
        let repr = node.representative;
        let s = repr.parentNode;
        while (s) {
            s.classList.remove("at_collapse");
            s = s.previousElementSibling;
        }

        $(repr).css('color', 'red');
        setTimeout(x => {
            $(repr).css('color', '');
        }, 5000);
    }

    propertyChanged(node, name, value) {
        return this.propertyChanged2(node, name, value);
    }

    propertyChanged2(node, name, value) {
        node.element[name] = value;
    }

    renderProperties(node, pos, apply_params) {
        /*
            node: The TreeNode for which to draw the properties.
        */

        let params = node.opt_params;

        let transthis = this;

        let reduced_node = {};
        reduced_node.data = () => node.opt_params;
        this.diode.renderPropertiesInWindow(transthis, reduced_node, params, {
            type: "transformation",
            sdfg_name: this.getState()['for_sdfg'],
            opt_name: node.opt_name,
            pos: pos,
            apply_params: apply_params,
        });
    }


    sendHighlightRequest(idstring_list) {
        this._project.request(['sdfg-msg'], resp => {

        }, {
            params: JSON.stringify({
                type: 'highlight-elements',
                sdfg_name: this.getState()['for_sdfg'],
                elements: idstring_list
            }),
            timeout: 1000
        });
    }

    sendClearHighlightRequest() {
        this._project.request(['sdfg-msg'], resp => {

        }, {
            params: JSON.stringify({
                sdfg_name: this.getState()['for_sdfg'],
                type: 'clear-highlights',
            }),
            timeout: 1000
        });
    }

    addNodes(og) {

        let full = {};
        // Load the available data
        for (let x of og) {
            if (full[x.opt_name] === undefined) {
                full[x.opt_name] = []
            }
            full[x.opt_name].push(x);
        }
        let arrayed = [];
        for (let x of Object.entries(full)) {
            let k = x[0];
            let v = x[1];
            arrayed.push([k, v]);
        }
        let sorted = arrayed.sort((a, b) => a[0].localeCompare(b[0]));
        for (let z of sorted) {
            let y = z[0];
            let x = z[1];

            let i = 0;
            let container_node = undefined;
            if (x.length > 1) {

                container_node = document.createElement('div');
                container_node.classList = "flex_column";

                let c_title = document.createElement('div');
                {
                    let c_title_span = document.createElement("span");
                    c_title_span.innerText = y;
                    c_title.classList = "at_group_header";
                    c_title.appendChild(c_title_span);

                    c_title.addEventListener('click', x => {
                        c_title.classList.toggle("at_collapse");
                    });
                }

                container_node.appendChild(c_title);

                this._transformation_list.appendChild(container_node);
            }
            for (let n of x) {
                this.addNode(n, i, container_node);
                ++i;
            }
        }
        let _s = this.getState();
        _s.optstruct = full;
        this.saveToState(_s);
    }

    locateTransformation(opt_name, opt_pos, affects) {
        console.log("locateTransformation", arguments);

        this.sendHighlightRequest(affects);

        let _state = this.getState();
        let _repr = _state.optstruct[opt_name][opt_pos].representative;
        $(_repr).css("background", "green");

        setTimeout(() => {
            this.sendClearHighlightRequest();
            $(_repr).css("background", '');
        }, 1000);
    }

    applyTransformation(x, pos, _title) {
        if (this.operation_running) return;
        this.operation_running = true;
        let _state = this.getState();
        let optstruct = _state['optstruct'];
        let named = {};
        let cpy = JSON.parse(JSON.stringify(optstruct[x.opt_name][pos]));
        named[this.getState()['for_sdfg']] = [{
            name: _title,
            params: {
                props: cpy['opt_params']
            }
        }];

        let tmp = () => {
            // Compile after the transformation has been saved
            this.diode.gatherProjectElementsAndCompile(this, {
                optpath: named
            }, {
                sdfg_over_code: true
            });
        };

        this.project().request(['sdfg_object'], x => {
            console.log("Got snapshot", x);
            if (typeof (x.sdfg_object) == 'string')
                x.sdfg_object = JSON.parse(x.sdfg_object);

            this.project().saveSnapshot(x['sdfg_object'], named);

            this.project().request(['update-tfh'], x => {
                this.operation_running = false;
            }, {
                on_timeout: () => {
                    this.operation_running = false;
                }
            });

            setTimeout(tmp, 10);
        }, {});
    }

    appendHistoryItem(new_sdfg, item_name) {
        if (this.operation_running) return;
        this.operation_running = true;
        let named = {};
        named[this.getState()['for_sdfg']] = [{
            name: item_name,
            params: {}
        }];

        let tmp = () => {
            // Update SDFG
            this.diode.gatherProjectElementsAndCompile(this, {
                code: stringify_sdfg(new_sdfg)
            }, {});
        };

        this.project().request(['sdfg_object'], x => {
            console.log("Got snapshot", x);
            if (typeof (x.sdfg_object) == 'string')
                x.sdfg_object = JSON.parse(x.sdfg_object);

            this.project().saveSnapshot(x['sdfg_object'], named);

            this.project().request(['update-tfh'], x => {
                this.operation_running = false;
            }, {
                on_timeout: () => {
                    this.operation_running = false;
                }
            });

            setTimeout(tmp, 10);
        }, {});
    }

    addNode(x, pos = 0, parent_override = undefined) {

        let _title = x.opt_name;

        // Add a suffix
        if (pos != 0) {
            _title += "$" + pos;
        }
        x.id_name = _title;

        let at_list = (parent_override === undefined) ?
            this._transformation_list : parent_override;

        // Create the element
        let list_elem = document.createElement("div");
        list_elem.classList = "flex_row at_element";

        // Add a title-div
        let title = document.createElement("div");
        title.innerText = _title;

        title.addEventListener('mouseenter', _x => {
            this.sendHighlightRequest(x.affects);
        });
        title.addEventListener('mouseleave', _x => {
            this.sendClearHighlightRequest();
        });

        title.addEventListener('click', _x => {

            this.applyTransformation(x, pos, _title);
        });

        title.setAttribute('data-hint', '{"type": "transformation", "name": "' + x.opt_name + '"}');
        x.representative = title;

        // Add a control-div
        let ctrl = document.createElement("div");
        // Advanced button
        {
            let adv_button = document.createElement('b');
            adv_button.classList = "";
            adv_button.innerText = '...';

            adv_button.addEventListener('click', _x => {
                // Clicking this reveals the transformation properties
                this.renderProperties(x, pos, [x, pos, _title]);
            });

            ctrl.appendChild(adv_button);
        }
        // Help button
        /*
        {
            let help_button = document.createElement('i');
            help_button.classList = "";
            help_button.innerText = '?';
            help_button.setAttribute("data-hint", '{"type": "transformation", "name": "' + x.opt_name + '"}');
            help_button.addEventListener("click", _ev => this.diode.hint(_ev));
            ctrl.appendChild(help_button);
        }*/

        list_elem.appendChild(title);
        list_elem.appendChild(ctrl);


        at_list.appendChild(list_elem);
    }

    create(newstate = undefined) {
        if (newstate != undefined) {

            let _state = this.getState();
            Object.assign(_state, newstate);
            this.resetState(_state);
        }
        let _state = this.getState();
        if (typeof (_state) == 'string') {
            _state = JSON.parse(_state);
        }
        let matching_opts = undefined;
        if (_state.matching_opts != undefined) {
            matching_opts = _state.matching_opts;
        } else if (_state.optgraph_data != undefined) {
            let _data = JSON.parse(_state.optgraph_data);
            matching_opts = _data.matching_opts;
        }
        let parent = (this.container.getElement())[0];
        parent.innerHTML = '';

        let at_div = document.createElement('div');
        at_div.classList = "at_container";
        let at_list = document.createElement('div');

        this._transformation_list = at_list;

        at_div.appendChild(at_list);

        parent.appendChild(at_div);

        if (matching_opts != undefined) {
            this.addNodes(matching_opts);
        }
    }
}


class DIODE_Context_CodeIn extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);
        this.editor = null;
        this._terminal_identifer = null;

        this._markers = [];
    }

    setupEvents(project) {
        super.setupEvents(project);

        let transthis = this;

        let eh = this.diode.goldenlayout.eventHub;
        this.on(this._project.eventString('-req-input_code'), (msg) => {
            // Echo with data
            setTimeout(() => eh.emit(transthis._project.eventString('input_code'), this.getState()['code_content']), 1);
            transthis.editor.clearSelection();
        }, true);

        this.on(this.project().eventString('-req-new_error'), msg => {
            // Echo with data
            setTimeout(() => eh.emit(transthis._project.eventString('new_error'), 'ok'), 1);
            this.highlight_error(msg);
        });

        this.on(this.project().eventString('-req-highlight-code'), msg => {
            setTimeout(() => eh.emit(transthis._project.eventString('highlight-code'), 'ok'), 1);
            this.highlight_code(msg);
        });

        this.on(this.project().eventString('-req-set-inputcode'), msg => {
            setTimeout(() => eh.emit(transthis._project.eventString('set-inputcode'), 'ok'), 1);

            this.editor.setValue(msg);
            this.editor.clearSelection();
        });

        this.on(this.project().eventString('-req-clear-errors'), msg => {
            setTimeout(() => eh.emit(transthis._project.eventString('clear-errors'), 'ok'), 1);
            this.clearErrors();
        });

    }

    highlight_code(dbg_info) {
        let s_c = dbg_info.start_col;
        let e_c = dbg_info.end_col;
        if (e_c <= s_c) {
            // The source data is broken; work-around this limitation by setting the end-column has high as possible
            e_c = 2000;
        }
        let markerrange = new ace.Range(dbg_info.start_line - 1, s_c, dbg_info.end_line - 1, e_c);
        // Create a unique class to be able to select the marker later
        let uc = "chm_" + this.diode.getPseudorandom();

        let marker = this.editor.session.addMarker(
            markerrange,
            "code_highlight " + uc
        );

        this.editor.resize(true);
        this.editor.scrollToLine(dbg_info.start_line, true, true, function () {
        });
        this.editor.gotoLine(dbg_info.start_line, 10, true);

        setTimeout(() => {
            this.editor.getSession().removeMarker(marker);
        }, 5000);
    }

    clearErrors() {
        for (let m of this._markers) {
            this.editor.getSession().removeMarker(m);
        }
        this._markers = [];
    }

    highlight_error(error) {

        if (error.type == "SyntaxError") {
            let lineno = parseInt(error.line);
            let offset = parseInt(error.offset);
            let text = error.text;

            lineno -= 1;

            let lineval = this.editor.session.getLine(lineno);


            let start = lineval.indexOf(text.substring(0, text.length - 1));

            let markerrange = new ace.Range(lineno, start, lineno, start + text.length - 1);

            // Create a unique class to be able to select the marker later
            let uc = "sem_" + this.diode.getPseudorandom();

            let marker = this.editor.session.addMarker(
                markerrange,
                "syntax_error_highlight " + uc
            );

            this._markers.push(marker);

            // #TODO: Either find a way to display the error information directly as a tooltip (which ace does not seem to support trivially)
            // #TODO: or add a dedicated error-view.
        } else {
            console.log("Untreated error type", error);
        }
    }

    terminal_identifier() {
        return this._terminal_identifer;
    }

    compile(code) {
        for (let m of this._markers) {
            this.editor.getSession().removeMarker(m);
        }
        this._markers = [];
        this.diode.compile(this, code);
    }

    setEditorReference(editor) {
        this.editor = editor;

        let elem = this.container.getElement()[0];
        elem.addEventListener('resize', x => {
            this.editor.resize();
        });
    }


    compile_and_run(code) {

        let millis = this.diode.getPseudorandom();

        let terminal_identifier = "terminal_" + millis;

        // create a new terminal
        let terminal_config = {
            title: "Terminal",
            type: 'component',
            componentName: 'TerminalComponent',
            componentState: {created: millis}
        };
        this.diode.addContentItem(terminal_config);

        console.log("Server emitting to ", terminal_identifier);

        this._terminal_identifer = terminal_identifier;

        this.diode.gatherProjectElementsAndCompile(this, {}/*{ 'code': code}*/, {
            run: true,
            term_id: terminal_identifier
        });
    }

}

class DIODE_Context_CodeOut extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);

        this.editor = null;
    }

    setupEvents(project) {
        super.setupEvents(project);

        let transthis = this;

        let eh = this.diode.goldenlayout.eventHub;
        this.on(this._project.eventString('-req-new-codeout'), (msg) => {

            if (msg.sdfg_name != this.getState()['sdfg_name']) {
                // Name mismatch; ignore
                return;
            }
            // See DIODE Errata "GoldenLayout:EventResponses"
            //eh.emit(transthis.project().eventString('new-codeout'), 'ok')
            setTimeout(x => eh.emit(transthis.project().eventString('new-codeout'), 'ok'), 1);
            let extracted = msg;
            this.setCode(extracted);
        });
    }

    cleanCode(codestr) {
        // Removes '////DACE:'-comments in the output code
        return codestr.replace(/\s*\/\/\/\/\_\_DACE:[^\n]*/gm, "");
    }

    setCode(extracted) {
        let input = extracted;
        if (typeof extracted === "string") {
            extracted = JSON.parse(extracted);
        }

        if (typeof extracted.generated_code == "string") {
            this.editor.setValue(this.cleanCode(extracted.generated_code));
            this.editor.clearSelection();
        } else {
            // Probably an array type
            this.editor.setValue("");
            this.editor.clearSelection();
            for (let c of extracted.generated_code) {
                let v = c;
                if (extracted.generated_code.length > 1) {
                    v = "\n\n\n" + "#########  NEXT CODE FILE ############\n\n\n" + v;
                }
                let session = this.editor.getSession();
                session.insert({
                    row: session.getLength(),
                    column: 0
                }, this.cleanCode(v));
                this.editor.clearSelection();
            }
        }
        this.saveToState({'code': input});
    }

    setEditorReference(editor) {
        this.editor = editor;

        let elem = this.container.getElement()[0];
        elem.addEventListener('resize', x => {
            this.editor.resize();
        });
    }
}

class DIODE_Context_Error extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);

        this.editor = null;
    }

    setupEvents(project) {
        super.setupEvents(project);

        let transthis = this;

        let eh = this.diode.goldenlayout.eventHub;
        this.on(this._project.eventString('-req-new-error'), (msg) => {

            setTimeout(x => eh.emit(transthis.project().eventString('new-error'), 'ok'), 1);
            let extracted = msg;
            this.setError(extracted);
        });
    }

    setError(error) {
        console.log("error", error);

        let error_string = "";
        if (typeof (error) == "string")
            this.editor.setValue(error);
        else if (Array.isArray(error)) {
            for (let e of error) {
                if (e.msg != undefined) {
                    error_string += e.msg;
                }
                console.log("Error element", e);
            }
            this.editor.setValue(error_string);
        }
        this.saveToState({'error': error});
    }

    setEditorReference(editor) {
        this.editor = editor;

        let elem = this.container.getElement()[0];
        elem.addEventListener('resize', x => {
            this.editor.resize();
        });
    }
}


class DIODE_Context_PerfTimes extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);
        this._chart = null;
    }

    setupEvents(project) {
        super.setupEvents(project);

        let eh = this.diode.goldenlayout.eventHub;
        let transthis = this;

        this.on(this.project().eventString('-req-new-time'), (msg) => {
            setTimeout(x => eh.emit(transthis.project().eventString('new-time'), 'ok'), 1);
            this.addTime(msg.time);
        });
    }

    create() {
        let elem = this.container.getElement()[0];
        elem.innerHTML = "";

        // Create the graph
        let canvas = document.createElement("canvas");
        elem.appendChild(canvas);

        let oldstate = this.getState();
        if (oldstate.runtimes === undefined) {
            oldstate.runtimes = [];
        }

        console.log("Execution times loaded", oldstate.runtimes)

        let labels = [];
        for (let i = 0; i < oldstate.runtimes.length; ++i) {
            labels.push(i);
        }

        this._chart = new Chart(canvas.getContext("2d"), {
            type: 'bar',

            data: {
                labels: labels,
                datasets: [{
                    label: 'Exec. times in s',
                    backgroundColor: "blue",
                    data: oldstate.runtimes.map(x => x)
                }]
            },
            options: {
                responsive: true,
                scales: {
                    yAxes: [{
                        display: true,
                        ticks: {
                            beginAtZero: true
                        }
                    }],
                    xAxes: [{
                        display: true,
                        ticks: {
                            autoSkip: true
                        }
                    }]
                },
                legend: {
                    //display: false,
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Execution times'
                }
            }
        });
        this._chart.update();
    }

    addTime(runtime) {
        let oldstate = this.getState();
        if (oldstate.runtimes === undefined) {
            oldstate.runtimes = [];
        }
        oldstate.runtimes.push(runtime);
        this.resetState(oldstate);

        this.create();
    }

}

class DIODE_Context_Roofline extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);
        this._proc_func = null;
    }

    setupEvents(project) {
        super.setupEvents(project);

        let eh = this.diode.goldenlayout.eventHub;
        let transthis = this;

        /*this.on(this.project().eventString('-req-new-time'), (msg) => {
            setTimeout(x => eh.emit(transthis.project().eventString('new-time'), 'ok'), 1);
            this.addTime(msg.time);
        });*/
    }

    create() {
        let parent = this.container.getElement()[0];
        parent.style.width = "100%";
        parent.style.height = "100%";

        let canvas = document.createElement("canvas");
        canvas.width = 1920;
        canvas.height = 1080;

        let redraw_func = Roofline.main(canvas, proc_func => {
            // Setup code, called on init. Incoming data must be passed to proc_func
            this._proc_func = proc_func;
        });

        let on_resize = () => {
            console.log("Resizing");
            canvas.width = parseInt(parent.style.width) - 20;
            canvas.height = parseInt(parent.style.height) - 20;

            // Reset then
            redraw_func();
        };

        parent.addEventListener("resize", on_resize);

        if (window.ResizeObserver) {
            new ResizeObserver(on_resize).observe(parent);
        } else {
            console.warn("ResizeObserver not available");
        }

        parent.appendChild(canvas);

        REST_request('/dace/api/v1.0/perfdata/roofline/', {
            client_id: this.diode.getClientID()
        }, xhr => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                this._proc_func(JSON.parse(xhr.response));
            }
        });
    }


}

class DIODE_Context_InstrumentationControl extends DIODE_Context {
    /*
        This context shows status and controls for the current instrumentation run.
        It is an interface and does not store data other than that needed to provide popouts.
        In particular, it does not display instrumentation results (this is done in the SDFG component).
    */
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);

    }

    setupEvents(project) {
        super.setupEvents(project);

        let started = false;

        // Start updating interval
        this._update_timeout = setInterval(() => {
            REST_request("/dace/api/v1.0/dispatcher/list/", {}, xhr => {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    // Got response
                    let done = true;
                    let resp = JSON.parse(xhr.response);
                    let elems = resp.elements;
                    for (let e of elems) {
                        let o = e.options;
                        if (typeof (o) == 'string') {
                            console.log("o is ", o);
                            if (o == "endgroup") {
                                // At least started, not done yet
                                done = false;
                                started = true;
                            }
                        }
                    }

                    if (started && done) {
                        started = false;
                        this.diode.load_perfdata();
                    }
                }

            });
        }, 2000);
    }

    close() {
        clearInterval(this._update_timeout);
        this._update_timeout = null;
        super.close();
    }

    destroy() {
        super.destroy();
    }

    create() {
        let parent = this.container.getElement()[0];

        parent.innerHTML = "<h2>Instrumentation control</h2><p>Do not close this window while instrumented programs are running</p>";

        // Functionality provided in this context
        // - Download perfdata database
        // - Delete remote perfdata database (e.g. to run a new / different program)
        // - Wait for tasks to be done (to auto-update performance information)

        let download_but = document.createElement("a");
        download_but.innerText = "Download perfdata database";
        download_but.href = base_url + "/dace/api/v1.0/perfdata/download/" + this.diode.getClientID() + "/";
        download_but.download = "perfdata.sqlite3";

        let download_can_but = document.createElement("a");
        download_can_but.innerText = "Download CAN";
        download_can_but.href = base_url + "/dace/api/v1.0/can/download/" + this.diode.getClientID() + "/";
        download_can_but.download = "current.sqlite3";

        let delete_but = document.createElement("button");
        delete_but.innerText = "Delete remote database";
        delete_but.addEventListener("click", () => {
            REST_request("/dace/api/v1.0/perfdata/reset/", {
                client_id: this.diode.getClientID()
            }, x => {
            })
        });

        let delete_can_but = document.createElement("button");
        delete_can_but.innerText = "Delete remote CAN";
        delete_can_but.addEventListener("click", () => {
            REST_request("/dace/api/v1.0/can/reset/", {
                client_id: this.diode.getClientID()
            }, x => {
            })
        });

        let render_but = document.createElement("button");
        render_but.innerText = "Display instrumentation results";
        render_but.addEventListener("click", () => {
            this.diode.load_perfdata();
        });

        let roofline_but = document.createElement("button");
        roofline_but.innerText = "Show roofline";
        roofline_but.addEventListener("click", () => {
            this.diode.show_roofline();
        });

        let celem = document.createElement("div");
        celem.classList = "flex_column";

        celem.appendChild(download_but);
        celem.appendChild(download_can_but);
        celem.appendChild(delete_but);
        celem.appendChild(delete_can_but);
        celem.appendChild(render_but);
        celem.appendChild(roofline_but);

        parent.appendChild(celem);
    }

}

class DIODE_Context_Terminal extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);
    }

    setEditorReference(editor) {
        this.editor = editor;
    }

    append(output) {
        let session = this.editor.getSession();
        session.insert({
            row: session.getLength(),
            column: 0
        }, output);

        let curr_str = session.getValue();

        // Extract performance information if available
        let re = /~#~#([^\n]+)/gm;
        let matches = [...curr_str.matchAll(re)];
        for (let m of matches) {
            console.log("Got match", m);
            // We want to access the second element (index 1) because it contains the list
            let perflist = m[1];
            // Because this is a python list, it may contain "'" (single quotes), which is invalid json
            perflist = perflist.replace(/\'/g, '');
            perflist = JSON.parse(perflist);
            perflist.sort((a, b) => a - b);
            let median_val = perflist[Math.floor(perflist.length / 2)];

            console.log("Got median execution time", median_val);
            this.project().request(['new-time'], () => {
            }, {
                params: {
                    time: median_val
                }
            });
        }

        this.container.extendState({
            "current_value": curr_str
        });
        this.editor.clearSelection();
    }

}

class DIODE_Context_StartPage extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);
    }

    create() {
        let plus = `<svg width="50" height="50" version="1.1" viewBox="0 0 13.2 13.2" xmlns="http://www.w3.org/2000/svg"><g transform="translate(0 -284)"><g fill="none" stroke="#008000" stroke-width="2.65"><path d="m6.61 285v10.6"/><path d="m1.32 290h10.6"/></g></g></svg>`
        plus = "data:image/svg+xml;base64," + btoa(plus);

        this.container.setTitle("Start Page");


        let parent = $(this.container.getElement())[0];

        let header = document.createElement('h1');
        header.id = "startpage_header";
        header.innerText = "DIODE";
        parent.appendChild(header);

        let startpage_container = document.createElement('div');
        startpage_container.id = 'startpage_container';
        startpage_container.classList = "flex_row";
        startpage_container.style = "width: 100%;height:100%;"

        let startpage_recent = document.createElement('div');
        startpage_recent.id = 'startpage_recent';
        {
            let file_title = document.createElement('div');
            file_title.innerText = "New";
            file_title.classList = "startpage_title";
            startpage_recent.appendChild(file_title);

            startpage_recent.appendChild(this.createStartpageListElement("Create a new Project", null, null, plus, x => {
                this.container.close();

                // Force creation of a new "project" instance (since we are explicitly creating a new project)
                // (NOTE: Functionality moved to "newFile")
                //this.diode.createNewProject();

                this.diode.openUploader("code-python");
            }));


            let recent_title = document.createElement('div');
            recent_title.innerText = "Recent";
            recent_title.classList = "startpage_title";
            startpage_recent.appendChild(recent_title);
        }
        let startpage_resources = document.createElement('div');
        startpage_resources.id = 'additional_resources';
        {
            let resource_title = document.createElement('div');
            resource_title.innerText = "Resources";
            resource_title.classList = "startpage_title";
            startpage_resources.appendChild(resource_title);
        }


        // Load elements from list
        {
            let projects = DIODE_Project.getSavedProjects();
            for (let p of projects) {
                console.log("p", p);

                let pdata = DIODE_Project.getProjectData(p);

                startpage_recent.appendChild(this.createStartpageListElement(p, pdata.last_saved, pdata.description, undefined, x => {
                    DIODE_Project.load(this.diode, p);
                }));
            }

        }

        let dace_logo = `
        <svg xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:svg="http://www.w3.org/2000/svg" xmlns="http://www.w3.org/2000/svg" xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" id="svg13" height="60.891094" width="57.565113" version="1.0" viewBox="0 0 143.91279 152.22773" inkscape:version="0.92.3 (2405546, 2018-03-11)">
          
          <metadata id="metadata17">
            
          </metadata>
          <defs id="defs3">
            <pattern y="0" x="0" height="6" width="6" patternUnits="userSpaceOnUse" id="EMFhbasepattern"/>
          </defs>
          <path id="path5" d="m 0,0 h 71.95639 c 39.75591,0 71.95639,34.079345 71.95639,76.11387 0,42.05451 -32.20048,76.11387 -71.95639,76.11387 H 0 Z" style="fill:#0070c0;fill-opacity:1;fill-rule:evenodd;stroke:none"/>
          <path id="path7" d="M 76.913385,27.183525 115.29013,75.154451 76.913385,123.12538 Z" style="fill:#ffffff;fill-opacity:1;fill-rule:evenodd;stroke:none"/>
          <path id="path9" d="M 28.622652,27.183525 66.999394,50.049666 V 100.27923 L 28.622652,123.12538 Z" style="fill:#ffffff;fill-opacity:1;fill-rule:evenodd;stroke:none"/>
          <path id="path11" d="m 67.079345,75.234403 h 9.93398" style="fill:none;stroke:#ffffff;stroke-width:3.99757719px;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:8;stroke-dasharray:none;stroke-opacity:1"/>
        </svg>`;

        dace_logo = "data:image/svg+xml;base64," + btoa(dace_logo);

        startpage_resources.appendChild(this.createStartpageListElement("Visit DaCe on GitHub", null, null, "external_lib/GitHub-Mark.png", x => {
            window.open("https://github.com/spcl/dace", "_blank");
        }));
        startpage_resources.appendChild(this.createStartpageListElement("Visit project page", null, null, dace_logo, x => {
            window.open("https://spcl.inf.ethz.ch/Research/DAPP/", "_blank");
        }));


        startpage_container.appendChild(startpage_recent);
        startpage_container.appendChild(startpage_resources);

        parent.appendChild(startpage_container);
    }

    createStartpageListElement(name, time, info, image = undefined, onclick = x => x) {

        let diode_image = `<svg width="50" height="50" version="1.1" viewBox="0 0 13.229 13.229" xmlns="http://www.w3.org/2000/svg"><g transform="translate(0 -283.77)" fill="none" stroke="#000" stroke-linecap="round" stroke-width=".68792"><path d="m3.3994 287.29v6.9099l6.5603-3.7876-6.5644-3.7899z" stroke-linejoin="round"/><g><path d="m3.3191 290.39h-2.6127"/><path d="m12.624 290.41h-2.6647v-3.3585"/><path d="m9.9597 290.41v2.9962"/></g></g></svg>`
        if (image == undefined) {
            image = "data:image/svg+xml;base64," + btoa(diode_image);
        }
        let elem = document.createElement("div");
        elem.classList = "startpage_list_element";

        let cols = document.createElement('div');
        {
            cols.classList = "flex_row";
            // Col 1: Image
            let img = document.createElement('img');
            img.src = image;
            img.width = "50";
            img.height = "50";
            cols.appendChild(img);

            // Col 2: Row
            let col2 = document.createElement('div');
            {
                col2.classList = "flex_column";
                // This row includes project name and details
                let proj_name = document.createElement('span');
                proj_name.innerText = name;
                let proj_detail = document.createElement('span');
                proj_detail.innerText = info;

                col2.appendChild(proj_name);
                if (info != null)
                    col2.appendChild(proj_detail);
                else {
                    // We have space to use - use a bigger font
                    proj_name.style.fontSize = "1.2rem";


                    col2.style.justifyContent = "center";
                }
            }
            cols.appendChild(col2);

            let col3 = document.createElement('div');
            {
                col3.classList = "flex_column";
                // This row includes project date
                let proj_date = document.createElement('span');
                proj_date.innerText = time;
                let proj_unused = document.createElement('span');

                if (proj_date != null) {
                    col3.appendChild(proj_date);
                    col3.appendChild(proj_unused);
                }
            }
            cols.appendChild(col3);
        }
        elem.appendChild(cols);

        elem.addEventListener('click', onclick);

        return elem;
    }
}


class DIODE_Context_DIODESettings extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);

        this._settings_container = null;

        this._editor_themes = this.getThemes();
    }

    getThemes() {
        REST_request('/dace/api/v1.0/diode/themes', undefined, xhr => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                this._editor_themes = JSON.parse(xhr.response);
                console.log("Got editor themes", this._editor_themes);

                this.create();
            }
        }, "GET");
    }

    themeNames() {
        return this._editor_themes.map(x => x.substr("theme-".length).slice(0, -3));
    }

    setContainer(elem) {
        this._settings_container = elem;
    }

    create() {
        // Editor theme
        {
            let cont = FormBuilder.createContainer(undefined);
            let label = FormBuilder.createLabel(undefined, "editor theme", "Sets the ace editor theme");
            let theme_names = this.themeNames();
            let input = FormBuilder.createSelectInput(undefined, x => {
                let val = x.value;

                DIODE.setTheme(val);
                DIODE.loadTheme().then(x => {
                        this.diode._appearance.setFromAceEditorTheme(val);
                    }
                );
            }, theme_names, DIODE.editorTheme());

            cont.append(label);
            cont.append(input);

            this._settings_container.append(cont);
        }
        // Auto-Compile
        {
            let cont = FormBuilder.createContainer(undefined);
            let label = FormBuilder.createLabel(undefined, "Compile on property change", "When false, the program is not recompiled after a property change");

            let input = FormBuilder.createToggleSwitch(undefined, x => {
                let val = x.checked;
                DIODE.setRecompileOnPropertyChange(val);
            }, DIODE.recompileOnPropertyChange());

            cont.append(label);
            cont.append(input);

            this._settings_container.append(cont);
        }
        // (Debug) mode
        {
            let cont = FormBuilder.createContainer(undefined);
            let label = FormBuilder.createLabel(undefined, "DaCe Debug mode", "When true, the program shows elements primarily useful for debugging and developing DaCe/DIODE.");

            let input = FormBuilder.createToggleSwitch(undefined, x => {
                let val = x.checked;
                DIODE.setDebugDevMode(val);
            }, DIODE.debugDevMode());

            cont.append(label);
            cont.append(input);

            this._settings_container.append(cont);
        }
        // UI font
        {
            let cont = FormBuilder.createContainer(undefined);
            let label = FormBuilder.createLabel(undefined, "UI Font", "Select the font used in the UI (does not affect code panes and SDFG renderers)");

            let current = Appearance.getClassProperties("diode_appearance")['fontFamily'];

            let input = FormBuilder.createSelectInput(undefined, x => {
                let val = x.value;

                this.diode._appearance.setFont(val);
                this.diode._appearance.apply();
            }, Appearance.fonts(), current);

            cont.append(label);
            cont.append(input);

            this._settings_container.append(cont);
        }
    }

}


class DIODE_Context_RunConfig extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);

        this._settings_container = null;

    }

    create() {
        let parent = this.container.getElement()[0];

        let runopts_container = document.createElement("div");

        let runopts_general_container = document.createElement("div");

        let values = {
            "Configuration name": "",
            "Host": "localhost",
            "Use SSH": true,
            "SSH Key": this.diode.pubSSH(),
            "SSH Key override": "",
            "Instrumentation": "off",
            "Number of threads": "[0]"
        };

        let params = [];
        let node = null;
        let transthis = null;

        // Build the callback object
        transthis = {
            propertyChanged: (node, name, value) => {
                if (name == "Configuration name") {
                    if (this.diode.getRunConfigs().map(x => x['Configuration name']).includes(value)) {
                        // Load values and reset inputs
                        let copy = this.diode.getRunConfigs(value);
                        for (let x of Object.keys(copy)) {
                            let v = copy[x];
                            let ps = params.find(y => y.name == x);
                            ps.value = v;
                        }
                        values = copy;
                        runopts_general_container.innerHTML = "";
                        this.diode.renderProperties(transthis, node, params, runopts_general_container, {});
                        return;
                    }
                }
                values[name] = value;
            }
        };
        /*
        element structure:
        {
            name: <name>
            desc: <description> (as tooltip)
            category: <Category name>
            type: <Type used to render>
            value: <Value to store>
        }
        */
        {
            params = [{
                name: "Configuration name",
                type: "combobox",
                value: values['Configuration name'],
                options: this.diode.getRunConfigs().map(x => x['Configuration name']),
                desc: "Name of this configuration",
            }, {
                name: "Host",
                type: "hosttype",
                value: values['Host'],
                desc: "Host executing the programs",
            },
            ]; // Array of elements

            // Add category (common to all elements)
            params.forEach(x => x.category = "General");

            let remoteparams = [{
                name: "Use SSH",
                type: "bool",
                value: values['Use SSH'],
                desc: "Use SSH. Mandatory for remote hosts, optional for localhost.",
            }, {
                name: "SSH Key",
                type: "str",
                value: values['SSH Key'],
                desc: "Public SSH key (id_rsa.pub) to add to remote authorized_keys. This key must not be password-protected!",
            }, {
                name: "SSH Key override",
                type: "str",
                value: values['SSH Key override'],
                desc: "Override the identity key file (ssh option -i) with this value if your password-free key is not id_rsa"
            }
            ];

            // Add category (common to all elements)
            remoteparams.forEach(x => x.category = "Remote");

            let instrumentationparams = [{
                name: "Instrumentation",
                type: "selectinput",
                value: values['Instrumentation'],
                options: ['off', 'minimal', 'full'],
                desc: "Set instrumentation mode (CPU only)",
            },
                {
                    name: "Number of threads",
                    type: "list",
                    value: values['Number of threads'],
                    desc: "Sets the number of OpenMP threads." +
                        "If multiple numbers are specified, the program is executed once for every number of threads specified. Specify 0 to use system default"
                },

            ];

            // Add category (common to all elements)
            instrumentationparams.forEach(x => x.category = "Profiling");


            // Merge params
            params = [...params, ...remoteparams, ...instrumentationparams];

            let node = {
                data: () => params
            };

            // Build the settings
            this.diode.renderProperties(transthis, node, params, runopts_general_container, {});
        }


        runopts_container.appendChild(runopts_general_container);


        parent.appendChild(runopts_container);

        let apply_button = document.createElement("button");
        apply_button.innerText = "Save";
        apply_button.addEventListener('click', _x => {
            this.diode.addToRunConfigs(values);
        });
        parent.appendChild(apply_button);
    }


}

class DIODE_Context_Settings extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);
    }

    settings_change_callback(type, path, value) {

        console.assert(value !== undefined, "Undefined value");

        console.log("Setting changed", path, value);
        this.diode.settings().change(path, value);

        this.set_settings();
    }

    link_togglable_onclick(element, toToggle) {
        let toggleclassname = "collapsed_container";
        element.on('click', () => {
            if (toToggle.hasClass(toggleclassname)) {
                toToggle.removeClass(toggleclassname)
            } else {
                toToggle.addClass(toggleclassname)
            }
        });
    }

    parse_settings2(settings, parent = undefined, path = []) {
        let is_topmost = false;
        if (parent === undefined) {
            parent = new ValueTreeNode("Settings", null);
            is_topmost = true;
        }

        let dicts = [];
        let values = [];


        Object.entries(settings).forEach(
            ([key, value]) => {
                let meta = value.meta;
                let val = value.value;

                if (meta.type == 'dict') {
                    dicts.push([key, value]);
                } else {
                    values.push([key, value]);
                }
            });

        let settings_lookup = {};
        // Create the elements that are not in subcategories (=dicts) here
        let dt = new DiodeTables.Table();
        {
            let params = JSON.parse(JSON.stringify(values));
            params = params.map(x => {
                let key = x[0];
                x = x[1];
                let tmp = x.meta;
                tmp.value = x.value;
                tmp.name = tmp.title;
                tmp.category = "General";
                tmp.key = key;
                return tmp;
            });

            let cur_dt = dt;

            let dtc = null;
            let categories = {};
            for (let x of params) {

                let cat = x.category;
                if (categories[cat] == undefined) {
                    categories[cat] = [];
                }
                categories[cat].push(x);
            }
            if (!DIODE.debugDevMode()) {
                delete categories["(Debug)"]
            }
            // INSERTED
            let transthis = {
                propertyChanged: (path, name, value) => {
                    console.log("PropertyChanged", path, name, value);
                    this.settings_change_callback(undefined, path, value);
                }
            };
            // !INSERTED
            for (let z of Object.entries(categories)) {

                // Sort within category
                let cat_name = z[0];
                let y = z[1].sort((a, b) => a.name.localeCompare(b.name));


                // Add Category header
                cur_dt = dt;
                let sp = document.createElement('span');
                sp.innerText = cat_name;
                let tr = cur_dt.addRow(sp);
                tr.childNodes[0].colSpan = "2";

                dtc = new DiodeTables.TableCategory(cur_dt, tr);

                for (let x of y) {

                    settings_lookup[path] = x.value;

                    let value_part = diode.getMatchingInput(transthis, x, [...path, x.key]);
                    let cr = cur_dt.addRow(x.name, value_part);
                    if (dtc != null) {
                        dtc.addContentRow(cr);
                    }
                }
            }
            dt.setCSSClass("diode_property_table");
        }
        // Link handlers
        parent.setHandler("activate", (node, level) => {
            if (level == 1) {
                let repr = node.representative();
                // Clear all selections in this tree
                node.head().asPreOrderArray(x => x.representative()).forEach(x => x.classList.remove("selected"));
                repr.classList.add("selected");
                let cont = $("#diode_settings_props_container")[0];
                cont.innerHTML = "";
                dt.createIn(cont);
            }
        });

        // Recurse into (sorted) dicts
        dicts = dicts.sort((a, b) => a[1].meta.title.localeCompare(b[1].meta.title));

        for (let d of dicts) {
            let key = d[0];
            d = d[1];
            let setting_path = path.concat(key);
            // Create the list element first
            let newparent = parent.addNode(d.meta.title, null);
            // The representative DOM node does not exist yet - add hints after half a second.
            setTimeout(() => newparent.representative().title = d.meta.description, 500);

            // Recurse
            console.log("Setting path", setting_path);
            let tmp = this.parse_settings2(d.value, newparent, setting_path);
            settings_lookup = {...settings_lookup, ...tmp};
        }

        if (is_topmost) {
            let _tree = new TreeView(parent);
            _tree.create_html_in($("#diode_settings_container")[0]);
        }
        return settings_lookup;
    }

    get_settings() {
        let post_params = {
            client_id: this.diode.getClientID()
        }
        REST_request("/dace/api/v1.0/preferences/get", post_params, (xhr) => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                let settings = this.parse_settings2(JSON.parse(xhr.response));

                this.diode._settings = new DIODE_Settings(settings);
                //this.diode._settings = null;

            }
        });
    }

    set_settings() {
        if (!this.diode.settings().hasChanged()) {
            // Do not update if nothing has changed
            return;
        }
        // Find settings that changed and write them back
        let changed_values = this.diode.settings().changedValues();
        this.saveToState({
            "changed": changed_values,
            "confirmed": this.diode.settings().values()
        });
        // #TODO: Maybe only update when a "save"-Button is clicked?

        let transthis = this;
        let client_id = this.diode.getClientID();
        let post_params = changed_values;
        post_params['client_id'] = client_id;
        // Debounce
        let debounced = this.diode.debounce("settings-changed", function () {

            REST_request("/dace/api/v1.0/preferences/set", post_params, (xhr) => {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    transthis.diode.settings().clearChanged();
                    this.diode.toast("Settings changed", "The changed settings were applied at remote server", "info", 3000);
                }
            });
        }, 1000);
        debounced();
    }
}

/*
Coordinates windows belonging to the same project.

*/
class DIODE_Project {
    constructor(diode, project_id = undefined) {
        this._diode = diode;
        if (project_id === undefined || project_id === null) {
            this._project_id = diode.getPseudorandom();
        } else {
            this._project_id = project_id;
        }
        this.setup();
        this._callback = null;
        this._rcvbuf = {};
        this._waiter = {};

        this._listeners = {};

        this._closed_windows = [];
    }

    clearTransformationHistory() {
        // Reset transformation history
        sessionStorage.removeItem("transformation_snapshots");
    }

    getTransformationSnapshots() {
        let sdata = sessionStorage.getItem("transformation_snapshots");
        if (sdata == null) {
            sdata = [];
        } else
            sdata = JSON.parse(sdata);

        return sdata;
    }

    getTransformationHistory() {
        let sdata = this.getTransformationSnapshots();
        return sdata.map(x => x[0]);
    }

    discardTransformationsAfter(index) {
        let sdata = this.getTransformationSnapshots();

        // Cut the tail off and resave
        sessionStorage.setItem("transformation_snapshots", JSON.stringify(sdata.slice(0, index)));
        // Send the update notification
        this.request(['update-tfh'], x => x, {});
    }

    saveSnapshot(sdfgs, changing_transformation) {
        /*
            Saves the current snapshot, defined by sdfgs and the current (new) transformation.

        */

        let sdata = this.getTransformationSnapshots();
        sdata.push([changing_transformation, sdfgs]);
        sessionStorage.setItem("transformation_snapshots", JSON.stringify(sdata));
    }

    reopenClosedWindow(name) {
        let window = this.getConfigForClosedWindow(name, true);
        this._diode.addContentItem(window);

        // Emit the reopen event
        this._diode.goldenlayout.eventHub.emit('window-reopened-' + name);
    }

    getConfigForClosedWindow(name, remove = true) {
        let list = this.getClosedWindowsList();
        let new_list = [];

        let rets = [];

        for (let x of list) {
            let cname = x[0];
            let state = x[1];

            if (state.created === name) {
                // Found the requested element

                rets.push([cname, state]);

                if (remove) {
                    // Don't put into the new list
                } else {
                    new_list.push([cname, state]);
                }
            } else {
                // Not found
                new_list.push([cname, state]);
            }
        }

        // Store back
        this.setClosedWindowsList(new_list);

        console.assert(rets.length === 1, "Expected only 1 match!");
        let ret = rets[0];

        // Build a config for this
        let config = {
            type: 'component',
            componentName: ret[0],
            componentState: ret[1]
        };

        return config;
    }

    setClosedWindowsList(new_list) {
        this._closed_windows = new_list;
        sessionStorage.setItem(this._project_id + "-closed-window-list", JSON.stringify(this._closed_windows));
    }

    clearClosedWindowsList() {
        this._closed_windows = [];

        sessionStorage.setItem(this._project_id + "-closed-window-list", JSON.stringify(this._closed_windows));
    }

    addToClosedWindowsList(componentName, state) {
        this._closed_windows = this.getClosedWindowsList();
        this._closed_windows.push([componentName, state]);

        sessionStorage.setItem(this._project_id + "-closed-window-list", JSON.stringify(this._closed_windows));
    }

    getClosedWindowsList() {
        let tmp = sessionStorage.getItem(this._project_id + "-closed-window-list");
        if (typeof (tmp) === "string") {
            tmp = JSON.parse(tmp);
        }

        if (tmp === null) {
            return [];
        }
        return tmp;
    }

    eventString(suffix) {
        console.assert(this._project_id != null, "project id valid");
        return this._project_id + suffix;
    }

    startListening(event, id) {
        let hub = this._diode.goldenlayout.eventHub;

        let transthis = this;
        let cb = (msg) => {
            let tmp = transthis._rcvbuf[id][event];
            if (tmp instanceof Array) {
                transthis._rcvbuf[id][event].push(msg);
            } else if (tmp instanceof Object) {
                Object.assign(transthis._rcvbuf[id][event], msg);
            } else {
                transthis._rcvbuf[id][event] = msg;
            }
        };
        let params = [this.eventString(event), cb, this];
        hub.on(...params);

        this._listeners[id].push(params);
    }

    stopListening(id) {
        let hub = this._diode.goldenlayout.eventHub;
        for (let x of this._listeners[id]) {
            hub.unbind(...x);
        }
        delete this._listeners[id];
    }


    setup() {

    }

    static load(diode, project_name) {
        let pdata = DIODE_Project.getProjectData(project_name);

        let ret = new DIODE_Project(diode, pdata.project_id);


        // For simplicity, we copy the saved config over the current config

        // First, destroy the current layout
        diode.goldenlayout.destroy();

        // Then, we copy over the config
        sessionStorage.setItem("savedState", JSON.stringify(pdata.data));

        // ... and the project id
        sessionStorage.setItem("diode_project", ret._project_id);

        // ... and the transformation snapshots
        sessionStorage.setItem("transformation_snapshots", JSON.stringify(pdata.snapshots));

        // Reload the page (This will create a new goldenlayout with the specified data)
        window.location.reload();

        return ret;
    }

    static getProjectData(project_name) {
        let pdata = localStorage.getItem("project_" + project_name);
        if (pdata == null)
            throw "Project must exist";

        return JSON.parse(pdata);
    }

    static getSavedProjects() {
        let tmp = localStorage.getItem("saved_projects");
        if (tmp == null)
            return [];
        return JSON.parse(tmp);
    }

    save() {
        /*
            Saves all elements of this project to its own slot in the local storage
            (such that it can be opened again even if the window was closed).
            
        */

        let snapshots = this.getTransformationSnapshots();
        if (typeof (snapshots) == 'string')
            snapshots = JSON.parse(snapshots);
        let y = {
            project_id: this._project_id,
            data: this._diode.goldenlayout.toConfig(),
            snapshots: snapshots,
            last_saved: new Date().toLocaleDateString(),
            description: "<No description>"
        };
        let save_val = JSON.stringify(y);

        // The sdfg is not sufficiently unique.
        let save_name = prompt("Enter project name");

        window.localStorage.setItem("project_" + save_name, save_val);

        let sp = window.localStorage.getItem("saved_projects");
        if (sp == null) {
            sp = [];
        } else {
            sp = JSON.parse(sp);
        }

        sp = [save_name, ...sp];
        window.localStorage.setItem("saved_projects", JSON.stringify(sp));


    }

    request(list, callback, options = {}) {
        /*
            options:
                timeout: Number                 ms to wait until on_timeout is called
                on_timeout: [opt] Function      Function called on timeout
                params: [opt] object            Parameters to pass with the request
        */
        let tmp = new DIODE_Project(this._diode, this._project_id);
        return tmp.__impl_request(list, callback, options);
    }

    __impl_request(list, callback, options = {}) {
        /*
            options:
                timeout: Number                 ms to wait until on_timeout is called
                on_timeout: [opt] Function      Function called on timeout
                params: [opt] object            Parameters to pass with the request
        */
        this._callback = callback;
        let params = options.params;
        let reqid = "id" + this._diode.getPseudorandom();
        // Clear potentially stale values
        this._rcvbuf[reqid] = {};
        this._listeners[reqid] = [];
        for (let x of list) {
            this.startListening(x, reqid);
            this._diode.goldenlayout.eventHub.emit(this.eventString("-req-" + x), params, this);
        }

        let transthis = this;
        const interval_step = 100;
        let timeout = options.timeout;

        this._waiter[reqid] = setInterval(() => {
            let missing = false;

            for (let x of list) {
                if (!(x in transthis._rcvbuf[reqid])) {
                    missing = true;
                    break;
                }
            }
            if (!missing) {
                clearInterval(transthis._waiter[reqid]);
                transthis.stopListening(reqid);
                transthis._waiter[reqid] = null;
                let tmp = transthis._rcvbuf[reqid];
                delete transthis._rcvbuf[reqid];
                return transthis._callback(tmp, options.timeout_id);
            } else if (timeout !== null) {
                timeout -= interval_step;
                if (timeout <= 0) {
                    // Timed out - fail silently
                    clearInterval(transthis._waiter[reqid]);
                    transthis.stopListening(reqid);
                    if (options.on_timeout != undefined) {
                        options.on_timeout(transthis._rcvbuf[reqid]);
                    }
                    transthis._waiter[reqid] = null;
                    delete transthis._rcvbuf[reqid];
                }
            }
        }, interval_step);
    }
}

class DIODE_Context_PropWindow extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);

        this._html_container = null;

        this.container.setTitle("Properties");
    }

    setupEvents(project) {
        super.setupEvents(project);

        let eh = this.diode.goldenlayout.eventHub;
        this.on(this.project().eventString('-req-display-properties'), (msg) => {
            setTimeout(() => eh.emit(this.project().eventString("display-properties"), 'ok'), 1);
            this.getHTMLContainer().innerHTML = "";
            let p = msg.params;
            if (typeof (p) == 'string') p = JSON.parse(p);
            this.diode.renderProperties(msg.transthis, msg.node, p, this.getHTMLContainer(), msg.options);
        });

        this.on(this.project().eventString('-req-render-free-vars'), msg => {
            setTimeout(() => eh.emit(this.project().eventString("render-free-vars"), 'ok'), 1);
            this.renderDataSymbols(msg.calling_context, msg.data);
        });
    }


    renderDataSymbolProperties(caller_id, symbol) {
        /*
            caller_id: .created of calling context (SDFG Context, mainly)
            symbol: [sym_name, {
                    .attributes: <attr_obj>
                    .type: <type name>
                }]
        */
        let reduced_node = {};
        reduced_node.data = () => symbol;
        this.diode.renderPropertiesInWindow(caller_id, reduced_node, symbol[1].attributes, {
            backaction: () => {
                // #TODO: Implement a quick way of getting back from here
            },
            type: "symbol-properties"
        });
    }

    removeDataSymbol(calling_context, data_name) {
        this.project().request(["delete-data-symbol-" + calling_context], x => {
        }, {
            params: data_name
        });
    }

    addDataSymbol(calling_context, data_type, data_name) {
        this.project().request(["add-data-symbol-" + calling_context], x => {
        }, {
            params: {
                name: data_name,
                type: data_type
            }
        });
    }

    renderDataSymbols(calling_context, data) {
        // #TODO: This creates the default state (as in same as render_free_symbols() in the old DIODE)
        if (data == null) {
            console.warn("Data has not been set - creating empty window");
            return;
        }
        let free_symbol_table = new DiodeTables.Table();
        free_symbol_table.setHeaders("Symbol", "Type", "Dimensions", "Controls");

        // Go over the undefined symbols first, then over the arrays (SDFG::arrays)
        let all_symbols = [...data.undefined_symbols, "SwitchToArrays", ...Object.entries(data.attributes._arrays)];

        let caller_id = calling_context;
        console.assert(caller_id != undefined && typeof (caller_id) == 'string');

        for (let x of all_symbols) {

            if (x == "SwitchToArrays") {
                // Add a delimiter
                let col = free_symbol_table.addRow("Arrays");
                col.childNodes.forEach(x => {
                    x.colSpan = 4;
                    x.style = "text-align:center;";
                });
                continue;
            }
            if (x[0] == "null" || x[1] == null) {
                continue;
            }
            let edit_but = document.createElement('button');
            edit_but.addEventListener('click', _x => {
                this.renderDataSymbolProperties(caller_id, x);
            });
            edit_but.innerText = "Edit";
            let del_but = document.createElement('button');
            del_but.addEventListener('click', _x => {
                this.removeDataSymbol(caller_id, x[0]);
            });
            del_but.innerText = "Delete";
            let but_container = document.createElement('div');
            but_container.appendChild(edit_but);
            but_container.appendChild(del_but);
            free_symbol_table.addRow(x[0], x[1].type, x[1].attributes.dtype + "[" + x[1].attributes.shape + "]", but_container);
        }

        free_symbol_table.addRow("Add data symbols").childNodes.forEach(x => {
            x.colSpan = 4;
            x.style = "text-align:center;";
        });
        {
            let input_name = document.createElement("input");
            input_name.type = "text";
            input_name.placeholder = "Symbol name";
            let add_scalar = document.createElement("button");
            add_scalar.innerText = "Add Scalar";
            add_scalar.addEventListener("click", () => {
                this.addDataSymbol(caller_id, "Scalar", input_name.value);
            });
            let add_array = document.createElement("button");
            add_array.addEventListener("click", () => {
                this.addDataSymbol(caller_id, "Array", input_name.value);
            })
            add_array.innerText = "Add Array";

            let but_container = document.createElement("div");
            but_container.appendChild(add_scalar);
            but_container.appendChild(add_array);

            free_symbol_table.addRow(input_name, but_container).childNodes.forEach(x => {
                x.colSpan = 2;
                x.style = "text-align:center;";
            });

            let libnode_container = document.createElement("div");
            let expand_all = document.createElement("button");
            expand_all.addEventListener("click", () => {
                // Expand all library nodes
                REST_request("/dace/api/v1.0/expand/", {
                        sdfg: data,
                    }, (xhr) => {
                        if (xhr.readyState === 4 && xhr.status === 200) {
                            let resp = parse_sdfg(xhr.response);
                            if (resp.error !== undefined) {
                                // Propagate error
                                this.diode.handleErrors(this, resp);
                            }

                            // Add to history
                            this.project().request(["append-history"], x => {
                            }, { params: {
                                    new_sdfg: resp.sdfg,
                                    item_name: "Expand library nodes"
                                }
                            });
                        }
                    });
            });
            expand_all.innerText = "Expand all library nodes";
            libnode_container.appendChild(expand_all);
            free_symbol_table.addRow(libnode_container);
        }

        this.getHTMLContainer().innerHTML = "";

        free_symbol_table.setCSSClass('free_symbol_table');
        free_symbol_table.createIn(this.getHTMLContainer());
    }

    getHTMLContainer() {
        let parent = $(this.container.getElement()).children(".sdfgpropdiv");
        return parent[0];
    }

    createFromState() {
        let p = this.getHTMLContainer();
        p.setAttribute("data-hint", '{"type": "DIODE", "name": "Property_Window"}');
        let state = this.getState();
        if (state.params != undefined && state.params.params != null) {
            let p = state.params;
            this.diode.renderProperties(p.transthis, p.node, JSON.parse(p.params), this.getHTMLContainer());
        }
    }

}


class DIODE_Context_Runqueue extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);

    }

    setupEvents(project) {
        super.setupEvents(project);

        let transthis = this;

        let eh = this.diode.goldenlayout.eventHub;

        this._autorefresher = null;

    }

    destroy() {
        clearInterval(this._autorefresher);
        super.destroy();
    }

    refreshUI(data) {

        if (typeof (data) == 'string') {
            data = JSON.parse(data);
        }

        if (data.elements == undefined) {
            data.elements = [];
        }
        let base_element = $(this.container.getElement())[0];
        base_element.innerHTML = "";
        let container = document.createElement("div");
        $(container).css("overflow", "auto");
        $(container).width("100%");
        $(container).height("100%");
        let table = document.createElement("table");

        // Build the header
        let header = document.createElement("thead");
        let header_row = document.createElement("tr");
        let header_titles = ['position', 'clientID', 'state', 'options'];
        header_titles.map(x => {
            let h = document.createElement("th");
            h.innerText = x;
            return h;
        }).forEach(x => header_row.appendChild(x));

        header.appendChild(header_row);
        table.appendChild(header);

        let tbody = document.createElement("tbody");
        for (let x of data.elements) {

            let optparse = x.options;
            if (typeof (optparse) == 'string') {

            } else if (optparse == undefined) {
                if (x.output != undefined && x.type == "orphan") {
                    optparse = document.createElement("button");
                    optparse.onclick = click => {
                        this.diode.addContentItem({
                            'type': 'component',
                            'componentName': 'TerminalComponent',
                            'componentState': {
                                current_value: x.output
                            },
                            'title': 'Output'
                        });
                    }
                    optparse.innerText = "Output";
                }
            } else {
                if (optparse.type == undefined) {
                    optparse = optparse.perfopts;
                }
                optparse = optparse.mode + ", coresets " + optparse.core_counts;
            }
            let values = [
                x['index'],
                x['client_id'],
                x['state'],
                optparse
            ];
            let row = document.createElement("tr");
            values.map(y => {
                let c = document.createElement("td");
                if (typeof (y) == 'string' || typeof (y) == 'number')
                    c.innerText = y;
                else {
                    c.appendChild(y);
                }
                return c;
            }).forEach(y => row.appendChild(y));
            tbody.appendChild(row);
        }

        table.appendChild(tbody);


        container.appendChild(table);
        base_element.appendChild(container);
        $(table).DataTable();
    }

    create() {
        this._autorefresher = setInterval(x => {
            this.getCurrentQueue();
        }, 2000);
        $(this.container.getElement()).css("overflow", "auto");
        this.refreshUI({});
    }

    getCurrentQueue() {

        let post_params = {};
        REST_request("/dace/api/v1.0/dispatcher/list/", post_params, xhr => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                // Got response
                this.refreshUI(xhr.response);
            }
        });
    }

}


class DIODE {
    constructor() {
        this._settings = new DIODE_Settings();
        this._debouncing = {};

        this._background_projects = [];
        this._current_project = null;

        this._stale_data_button = null;

        this._shortcut_functions = {
            /*
            format:
            key: {
                .alt: Trigger if altKey is pressed
                .ctrl: Trigger if ctrlKey is pressed
                .function: Function to run
                .state: Multi-key only: state in state machine
                .expect: The state transitions (without the first state transition)
            }
            */
        };

        this._creation_counter = 0;

        // Load a client_id
        this._client_id = localStorage.getItem("diode_client_id");
        if (this._client_id == null) {
            this._client_id = this.getPseudorandom();
            localStorage.setItem("diode_client_id", this._client_id);
        }

        // Initialize appearance
        this._appearance = new Appearance(localStorage.getItem("DIODE/Appearance"));
        this._appearance.setOnChange(x => {
            localStorage.setItem("DIODE/Appearance", JSON.stringify(x.toStorable()))
        });
    }

    setupEvents() {
        this.goldenlayout.eventHub.on(this.project().eventString('-req-show_stale_data_button'), x => {
            this.__impl_showStaleDataButton();
        });
        this.goldenlayout.eventHub.on(this.project().eventString('-req-remove_stale_data_button'), x => {
            this.__impl_removeStaleDataButton();
        });
        this.goldenlayout.eventHub.on(this.project().eventString('-req-show_loading'), x => {
            this.__impl_showIndeterminateLoading();
        });
        this.goldenlayout.eventHub.on(this.project().eventString('-req-hide_loading'), x => {
            this.__impl_hideIndeterminateLoading();
        });

        // Install the hint mechanic on the whole window
        window.addEventListener('contextmenu', ev => {
            console.log("contextmenu requested on", ev.target);

            this.hint(ev);
        });
    }

    getRunConfigs(name = undefined) {
        let tmp = localStorage.getItem("diode_run_configs");
        if (tmp != null) {
            tmp = JSON.parse(tmp);
        } else {
            // Create a default
            tmp = [{
                "Configuration name": "default",
                "Host": "localhost",
                "Use SSH": true,
                "SSH Key": this.pubSSH(),
                "SSH Key override": "",
                "Instrumentation": "off",
                "Number of threads": "[0]"
            }];
        }

        if (name != undefined) {
            let ret = tmp.filter(x => x['Configuration name'] == name);
            if (ret.length == 0) {
                // Error
                console.error("Could not find a configuration with that name", name);
            } else {
                return ret[0];
            }
        }
        return tmp;
    }

    addToRunConfigs(config) {
        delete config['SSH Key']; // Don't save large, unnecessary data
        let existing = this.getRunConfigs();

        let i = 0;
        for (let x of existing) {
            if (x['Configuration name'] == config['Configuration name']) {
                // Replace
                existing[i] = config;
                break;
            }
            ++i;
        }
        if (i >= existing.length) {
            existing.push(config);
        }
        existing.sort((a, b) => a['Configuration name'].localeCompare(b['Configuration name']));
        localStorage.setItem("diode_run_configs", JSON.stringify(existing));
    }

    setCurrentRunConfig(name) {
        sessionStorage.setItem("diode_current_run_config", name);
    }

    getCurrentRunConfigName() {
        let tmp = sessionStorage.getItem("diode_current_run_config");
        if (tmp == null) {
            return "default";
        } else {
            return tmp;
        }
    }

    getCurrentRunConfig() {
        let config = this.getRunConfigs(this.getCurrentRunConfigName());
        return config;
    }

    applyCurrentRunConfig() {
        let config = this.getCurrentRunConfig();

        let new_settings = {};

        new_settings = {
            ...new_settings, ...{
                "execution/general/host": config['Host']
            }
        };

        // Apply the runconfig values to the dace config
        if (config['Use SSH']) {


            let keyfile_string = /\S/.test(config['SSH Key override']) ? (" -i " + config['SSH Key override'] + " ") : " ";
            new_settings = {
                ...new_settings, ...{
                    "execution/general/execcmd": ("ssh -oBatchMode=yes" + keyfile_string + "${host} ${command}"),
                    "execution/general/copycmd_r2l": ("scp -B" + keyfile_string + " ${host}:${srcfile} ${dstfile}"),
                    "execution/general/copycmd_l2r": ("scp -B" + keyfile_string + " ${srcfile} ${host}:${dstfile}"),
                }
            };
        } else {
            // Use standard / local commands
            new_settings = {
                ...new_settings, ...{
                    "execution/general/execcmd": "${command}",
                    "execution/general/copycmd_r2l": "cp ${srcfile} ${dstfile}",
                    "execution/general/copycmd_l2r": "cp ${srcfile} ${dstfile}",
                }
            };
        }

        // Instrumentation settings are not to be applied here, but later when the run request is actually sent

        let ret = new Promise((resolve, reject) => {
            let post_params = {
                client_id: this.getClientID(),
                ...new_settings
            };
            REST_request("/dace/api/v1.0/preferences/set", post_params, (xhr) => {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    resolve(config);
                } else if (xhr.status !== 0 && !(xhr.status + "_").startsWith("2")) {
                    reject();
                }
            });
        });

        return ret;
    }

    pubSSH() {
        let cached = localStorage.getItem('diode_pubSSH');
        if (cached != null) {
            return cached;
        }
        REST_request("/dace/api/v1.0/getPubSSH/", undefined, xhr => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                let j = JSON.parse(xhr.response);
                if (j.error == undefined) {
                    let t = j.pubkey;
                    localStorage.setItem('diode_pubSSH', t);
                } else {
                    alert(j.error);
                }
            }
        }, 'GET');
    }

    static getHostList() {
        let tmp = localStorage.getItem("diode_host_list");
        if (tmp == null) return ['localhost'];
        else return JSON.parse(tmp);
    }

    static setHostList(list) {
        localStorage.setItem("diode_host_list", JSON.stringify(list));
    }

    hint(ev) {
        /*
            ev: Event triggering this hint.
        */

        let create_overlay = (_h, elem) => {
            // Found hint data
            let fulldata = JSON.parse(_h);

            // #TODO: Link to documentation instead of using this placeholder
            $(elem).w2overlay("<div><h2>Help for category " + fulldata.type + "</h2>" + fulldata.name + "</div>");
        };

        let target = ev.target;
        let _h = target.getAttribute("data-hint");
        if (_h == null) {
            // Iterate chain
            if (!ev.composed) return;
            let x = ev.composedPath();
            for (let e of x) {
                if (e.getAttribute != undefined) {
                    _h = e.getAttribute("data-hint");
                } else _h = null;

                if (_h != null) {
                    create_overlay(_h, e);
                    ev.stopPropagation();
                    ev.preventDefault();
                    break;
                }
            }
            return;
        } else {
            create_overlay(_h, target);
            ev.stopPropagation();
            ev.preventDefault();
        }

        console.log("Got hint data", _h);

    }

    openUploader(purpose = "") {

        w2popup.open({
            title: "Upload a code file",
            body: `
<div class="w2ui-centered upload_flexbox">
    <label for="file-select" style="flex-grow: 1">
        <div class="diode_uploader" id='upload_box'>
            <div class="uploader_text">
                Drop file here or click to select a file
            </div>
        </div>
    </label>
    <input id="file-select" type="file"  accept=".py,.m,.sdfg" style="position:absolute;"/>
</div>
`,
            buttons: '',
            showMax: true
        });
        let x = $('#upload_box');
        if (x.length == 0) {
            throw "Error: Element not available";
        }
        x = x[0];

        let file_handler = (data) => {
            if (purpose == "code-python") {
                this.newFile(data);
            }
        };

        setup_drag_n_drop(x, (mime, data) => {
            console.log("upload mime", mime);

            file_handler(data);

            // Close the popup
            w2popup.close();
        }, null, {
            readMode: "text"
        });

        let fuploader = $('#file-select');
        if (fuploader.length == 0) {
            throw "Error: Element not available";
        }
        fuploader = fuploader[0];

        fuploader.style.opacity = 0;

        fuploader.addEventListener("change", x => {
            let file = fuploader.files[0];

            let reader = new FileReader();
            reader.onload = y => {
                file_handler(y.target.result);
                // Close the popup
                w2popup.close();
            };
            reader.readAsText(file);
        });
    }

    getClientID() {
        return this._client_id;
    }

    static hash(data) {
        return btoa(crypto.subtle.digest('SHA-256', Uint8Array.from(data)));
    }


    initEnums() {
        this.getEnum("ScheduleType");
        this.getEnum("StorageType");
        this.getEnum("AccessType");
        this.getEnum("Language");
    }

    // Closes all open windows
    closeAll() {
        if (!this.goldenlayout.root)
            return;
        let comps = this.goldenlayout.root.getItemsByFilter(x => x.config.type == "component");
        comps.forEach((comp) => comp.close());
        this.project().clearClosedWindowsList();
    }

    addContentItem(config) {
        // Remove all saved instances of this component type from the closed windows list
        if (config.componentName) {
            let cw = this.project()._closed_windows;
            this.project().setClosedWindowsList(cw.filter(x => x[0] != config.componentName));
        }

        let root = this.goldenlayout.root;

        // In case goldenlayout was not yet initialized, fail silently
        if (!root)
            return;

        if (root.contentItems.length === 0) {
            // Layout is completely missing, need to add one (row in this case)
            let layout_config = {
                type: 'row',
                content: []
            };
            root.addChild(layout_config);

            // retry with recursion
            this.addContentItem(config);
        } else {
            if (this.goldenlayout.isSubWindow) {
                // Subwindows don't usually have layouts, so send a request that only the main window should answer
                this.goldenlayout.eventHub.emit('create-window-in-main', JSON.stringify(config));
            } else {
                for (let ci of root.contentItems) {
                    if (ci.config.type != "stack") {
                        ci.addChild(config);
                        return;
                    }

                }
                let copy = root.contentItems[0].contentItems.map(x => x.config);
                root.contentItems[0].remove();
                // retry with recursion
                for (let ci of copy) {
                    this.addContentItem(ci);
                }
                this.addContentItem(config);
                //root.contentItems[0].addChild(config);
            }
        }
    }

    newFile(content = "") {
        // Reset project state
        this.closeAll();
        this.createNewProject();

        let millis = this.getPseudorandom();

        // Assuming SDFG files start with {
        if (content[0] == '{') {
            // Prettify JSON object, if not pretty
            if (content.split('\n').length == 1)
                content = JSON.stringify(JSON.parse(content), null, 2);
        }


        let config = {
            title: "Source Code",
            type: 'component',
            componentName: 'CodeInComponent',
            componentState: {created: millis, code_content: content}
        };

        this.addContentItem(config);

        // Compile automatically after loading
        this.gatherProjectElementsAndCompile(this, {}, {sdfg_over_code: true});
    }

    open_diode_settings() {
        let millis = this.getPseudorandom();

        let config = {
            title: "Settings",
            type: 'component',
            componentName: 'SettingsComponent',
            componentState: {created: millis}
        };

        this.addContentItem(config);
    }

    open_runqueue() {
        let millis = this.getPseudorandom();

        let config = {
            title: "Run Queue",
            type: 'component',
            componentName: 'RunqueueComponent',
            componentState: {created: millis}
        };

        this.addContentItem(config);
    }

    getEnum(name) {
        let cached = localStorage.getItem('Enumeration:' + name);
        if (cached == null || cached == undefined) {
            // Request the enumeration from the server

            REST_request("/dace/api/v1.0/getEnum/" + name, undefined, xhr => {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    console.log(name, xhr.response);
                    let tmp = JSON.parse(xhr.response);
                    if (name == "Language") {
                        tmp.enum.push("NoCode");
                    }
                    localStorage.setItem('Enumeration:' + name, JSON.stringify(tmp));
                }
            }, 'GET');

            return null;
        }

        return JSON.parse(cached)['enum'];
    }

    renderProperties(transthis, node, params, parent, options = undefined) {
        /*
            Creates property visualizations in a 2-column table.
        */
        if (params == null) {
            console.warn("renderProperties as nothing to render");
            return;
        }
        if (!Array.isArray(params)) {
            let realparams = params;

            // Format is different (diode to_json with seperate meta / value - fix before passing to renderer)
            let params_keys = Object.keys(params).filter(x => !x.startsWith('_meta_'));
            params_keys = params_keys.filter(x => Object.keys(params).includes('_meta_' + x));

            let _lst = params_keys.map(x => {
                let mx = JSON.parse(JSON.stringify(params['_meta_' + x]));

                mx.name = x;
                mx.value = params[x];

                return {property: mx, category: mx.category, element: realparams, data: node.data};
            });

            params = _lst;
        }

        if (typeof (transthis) == 'string') {
            // Event-based
            let target_name = transthis;
            transthis = {
                propertyChanged: (element, name, value) => {
                    // Modify in SDFG object first
                    this.project().request(['property-changed-' + target_name], x => {

                    }, {
                        timeout: 200,
                        params: {
                            element: element,
                            name: name,
                            value: value,
                            type: options ? options.type : options
                        }
                    });

                    // No need to refresh SDFG if transformation
                    if (options && options.type === 'transformation')
                        return;

                    this.refreshSDFG();
                },
                applyTransformation: () => {
                    this.project().request(['apply-adv-transformation-' + target_name], x => {

                    }, {
                        timeout: 200,
                        params: options == undefined ? undefined : options.apply_params
                    })
                },
                locateTransformation: (opt_name, opt_pos, affects) => {
                    this.project().request(['locate-transformation-' + options.sdfg_name], x => {

                    }, {
                        timeout: 200,
                        params: JSON.stringify([opt_name, opt_pos, affects]),
                    })
                },
                project: () => this.project()
            };
        }
        let dt = new DiodeTables.Table();
        let cur_dt = dt;

        let dtc = null;
        let categories = {};
        for (let x of params) {

            let cat = x.category;
            if (categories[cat] == undefined) {
                categories[cat] = [];
            }
            categories[cat].push(x);
        }
        if (!DIODE.debugDevMode()) {
            delete categories["(Debug)"]
        }
        for (let z of Object.entries(categories)) {

            // Sort within category
            let cat_name = z[0];
            let y = z[1].sort((a, b) => a.property.name.localeCompare(b.property.name));


            // Add Category header
            cur_dt = dt;
            let sp = document.createElement('span');
            sp.innerText = cat_name;
            let tr = cur_dt.addRow(sp);
            tr.childNodes[0].colSpan = "2";

            dtc = new DiodeTables.TableCategory(cur_dt, tr);

            for (let propx of y) {
                let title_part = document.createElement("span");
                let x = propx.property;
                title_part.innerText = x.name;
                title_part.title = x.desc;
                let value_part = diode.getMatchingInput(transthis, x, propx);
                let cr = cur_dt.addRow(title_part, value_part);
                if (dtc != null) {
                    dtc.addContentRow(cr);
                }
            }
        }
        dt.setCSSClass("diode_property_table");

        if (options && options.type == "transformation") {
            // Append a title
            let title = document.createElement("span");
            title.classList = "";
            title.innerText = options.opt_name;
            parent.appendChild(title);

            // Append a "locate" button
            let locate_button = document.createElement("span");
            locate_button.innerText = "location_on";
            locate_button.classList = "material-icons";
            locate_button.style = "cursor: pointer;";

            locate_button.addEventListener("click", () => {
                // Highlight the affected elements (both transformation and nodes)
                transthis.locateTransformation(options.opt_name, options.pos, options.apply_params[0].affects);
            });
            parent.appendChild(locate_button);
        }
        dt.createIn(parent);
        if (options && options.type == "transformation") {
            // Append an 'apply-transformation' button
            let button = document.createElement('button');
            button.innerText = "Apply advanced transformation";
            button.addEventListener('click', _x => {
                button.disabled = true;
                this.project().request(['apply-adv-transformation-' + options.sdfg_name], _y => {
                }, {
                    params: JSON.stringify(options.apply_params)
                });
            });
            parent.appendChild(button);

        }
    }

    create_visual_access_representation(elems, data) {

        let __access_indices = elems[0];
        let __ranges = elems[1];
        let __user_input = elems[2];
        let __additional_defines = {};

        // Add standard functions
        __additional_defines['Min'] = "(...x) => Math.min(...x)";
        __additional_defines['int_ceil'] = "(a,b) => Math.ceil(a/b)";

        for (let x of __user_input) {
            let _r = window.prompt("Enter a value for symbol " + x);
            if (_r != null) {

                // Write the value 
                __additional_defines[x] = _r;
            } else {
                // Aborted
                break;
            }
        }

        // Read the data object properties
        console.log("data", data);
        if (data.type != "Array") {
            console.warn("Non-Array accessed", data);
            // #TODO: What to do here?
        }
        let __mem_dims = data.attributes.shape;
        console.assert(__mem_dims != undefined);
        // Try to eval the functions
        let __eval_func = () => {
            let __defs = Object.keys(__additional_defines).map(x => "let " + x + " = " + __additional_defines[x]).join(";") + ";";
            __mem_dims = __mem_dims.map(x => eval(__defs + x));
        };
        __eval_func();


        let __tbl_container = document.createElement("div");
        let __tbl_hor = document.createElement("div");
        __tbl_hor.classList = "flex_row";
        __tbl_hor.style = "flex-wrap: nowrap; justify-content: flex-start;";
        let __axis_x_info = document.createElement("div");
        {
            __axis_x_info.classList = "flex_row";
            __axis_x_info.style = "justify-content: space-between;";


            for (let __i = 0; __i < 2; ++__i) {
                let __tmp = document.createElement("div");
                // Take the second dimension
                try {
                    __tmp.innerText = (__i == 0) ? "0" : __mem_dims[1];
                } catch (__e) {
                    __tmp.innerText = (__i == 0) ? "Start" : "End";
                }
                __axis_x_info.appendChild(__tmp);
            }
        }

        let __axis_y_info = document.createElement("div");
        {
            __axis_y_info.classList = "flex_column";
            __axis_y_info.style = "justify-content: space-between;";


            for (let __i = 0; __i < 2; ++__i) {
                let __tmp = document.createElement("div");
                // Take the first dimension
                try {
                    __tmp.innerText = (__i == 0) ? "0" : __mem_dims[0];
                } catch (__e) {
                    __tmp.innerText = (__i == 0) ? "Start" : "End";
                }
                __axis_y_info.appendChild(__tmp);
            }
        }

        let __tbl_vert = document.createElement("div");
        __tbl_vert.classList = "flex_col";
        if (data != null) {
            __tbl_vert.appendChild(__axis_x_info);
            __tbl_hor.appendChild(__axis_y_info);
        }
        // Now create a table with the according cells for this
        let __size = 10;
        if (__mem_dims.some(x => x > 128)) __size = 5;

        if (__mem_dims.length < 2) __mem_dims.push(1); // Force at least 2 dimensions (2nd dim size is trivial: 1)

        console.log("access indices", __access_indices);
        console.log("ranges", __ranges);

        // This is very limited; future work can take place here
        // The current implementation works only by fixing all but one range per dimension.
        // This is done for 2 main reasons:
        // 1) Performance. It is not possible to determine the access patterns in O(n) without either using an LSE
        //        with range side conditions (which is hard to solve in "naked" JS). The only alternative is to actually
        //        scan __all__ possible values, which is infeasible on a browser client.
        // 2) Visual Cluttering. Seeing too much at once is not helpful. Implementing an easy-to-use UI solving this problem
        //        is beyond the scope of the initial PoC.

        // Obtain fixed ranges (all but smallest)
        let __fixed_rngs = __ranges.map(x => x).sort((a, b) => a.depth - b.depth).slice(1).reverse();
        // Get the variable range (smallest)
        let __var_rng = __ranges[0];

        let __rng_inputs = [];

        let __create_func = () => null;

        let __main = x => x.main != undefined ? x.main : x;
        // Add inputs for every fixed range
        {
            let input_cont = document.createElement("div");
            let __defs = Object.keys(__additional_defines).map(x => "let " + x + " = " + __additional_defines[x]).join(";") + ";";

            let __global_slider = document.createElement("input");
            {
                __global_slider.type = "range";
                __global_slider.min = "0";
                __global_slider.value = "0";
                __global_slider.step = "1";

                input_cont.appendChild(__global_slider);
            }

            let __total_rng_count = 1;
            let __locked_range = false;

            input_cont.classList = "flex_column";
            for (let __r of __fixed_rngs) {
                let _lbl = document.createElement("label");
                let _in = document.createElement("input");
                _in.type = "number";
                _in.min = "0";
                _in.step = "1";

                _in.addEventListener("change", _click => {
                    // Trigger update

                    // Move the slider position
                    let __spos = 0;
                    let __base = 1;
                    for (let __r of __rng_inputs.map(x => x).reverse()) {
                        let __s = parseInt(__r.max) - parseInt(__r.min) + 1;

                        let __v = __r.value;

                        __spos += parseInt(__v) * __base;
                        __base *= parseInt(__s);
                    }
                    __global_slider.value = __spos;

                    __create_func();
                });
                // Set limits
                try {
                    _in.min = eval(__defs + __main(__r.val.start));
                } catch (e) {
                    console.warn("Got error when resolving expression");
                }
                try {
                    _in.max = eval(__defs + __main(__r.val.end));
                } catch (e) {
                    console.warn("Got error when resolving expression");
                }
                try {
                    _in.value = eval(__defs + __main(__r.val.start));
                } catch (e) {
                    console.warn("Got error when resolving expression");
                }

                // Add the starting value as an expression to defs
                __defs += "let " + __r.var + " = " + __main(__r.val.start) + ";";

                _lbl.innerText = "Range iterator " + __r.var + " over [" + __main(__r.val.start) + ", " + __main(__r.val.end) + "] in steps of " + __main(__r.val.step);
                _in.setAttribute("data-rname", __r.var);
                _lbl.appendChild(_in);
                __rng_inputs.push(_in);

                input_cont.appendChild(_lbl);

                if (__total_rng_count == 0) __total_rng_count = 1;

                let __e_size = ((__x) => eval(__defs + "(" + __main(__x.val.end) + " - " + __main(__x.val.start) + "+1) / " + __main(__x.val.step)))(__r);
                if (__e_size == 0 || __locked_range) {
                    __locked_range = true;
                } else {
                    __total_rng_count *= __e_size;
                }
            }
            console.log("__total_rng_count", __total_rng_count);
            {
                __global_slider.max = __total_rng_count - 1; // Inclusive range

                __global_slider.addEventListener("input", __ev => {
                    let __v = parseInt(__global_slider.value);

                    for (let __r of __rng_inputs.map(x => x).reverse()) {
                        let __s = parseInt(__r.max) - parseInt(__r.min) + 1;

                        let __subval = __v % __s;

                        __r.value = __subval;

                        __v = Math.floor(__v / __s);
                    }
                });

                let r = __var_rng;
                let _lbl = document.createElement("label");
                let _in = document.createElement("span");

                __global_slider.addEventListener("input", _ev => {
                    __create_func();
                });

                _in.innerText = "(whole range)";
                _lbl.innerText = "Range iterator " + r.var + " over [" + __main(r.val.start) + ", " + __main(r.val.end) + "] in steps of " + __main(r.val.step);
                _lbl.appendChild(_in);

                input_cont.appendChild(_lbl);
            }
            __tbl_container.appendChild(input_cont);
        }

        __create_func = () => {
            __tbl_vert.innerHTML = "";
            __tbl_vert.appendChild(__axis_x_info);
            let __all_fixed = {};
            Object.assign(__all_fixed, __additional_defines);
            // Get the fixed values
            {
                for (let i of __rng_inputs) {
                    let rname = i.getAttribute("data-rname");
                    let val = i.value;

                    __all_fixed[rname] = val;
                }
            }

            let __defstring = Object.keys(__all_fixed).map(x => "let " + x + " = " + __all_fixed[x] + ";").join("");

            const __ellision_thresh_y = 64;
            const __ellision_thresh_x = 128;

            let __mark_cells = {};

            // Evaluate the range
            {
                let feval = (x) => {
                    return eval(__defstring + x);
                };
                let __it = __var_rng.var;
                let __r_s = __main(__var_rng.val.start);
                let __r_e = __main(__var_rng.val.end);
                let __r_step = __main(__var_rng.val.step);

                // Remember: Inclusive ranges
                for (let __x = feval(__r_s); __x <= feval(__r_e); __x += feval(__r_step)) {
                    // Add this to the full evaluation
                    let __a_i = __access_indices.map((x, i) => [x, i]).sort((a, b) => a[1] - b[1]).map(x => x[0]);


                    __a_i = __a_i.map(__y => feval("let " + __it + " = " + __x + ";" + __y.var));

                    let __tmp = __mark_cells[__a_i[1]];
                    if (__tmp == undefined) {
                        __mark_cells[__a_i[1]] = [];
                    }
                    __mark_cells[__a_i[1]].push(__a_i[0]);
                }
            }

            for (let __dim_2 = 0; __dim_2 < __mem_dims[0]; ++__dim_2) {

                // Check ellision
                if (__mem_dims[0] > __ellision_thresh_y && __dim_2 > __ellision_thresh_y / 2 && __dim_2 < __mem_dims[0] - __ellision_thresh_y / 2) {
                    // Elide
                    if (__dim_2 - 1 == __ellision_thresh_y / 2) {
                        // Add ellision info _once_
                        let __row = document.createElement("div");
                        __row.classList = "flex_row";
                        __row.style = "justify-content: flex-start;flex-wrap: nowrap;"
                        __row.innerText = "...";
                        __tbl_vert.appendChild(__row);
                    }
                    continue;
                }
                let __row = document.createElement("div");
                __row.classList = "flex_row";
                __row.style = "justify-content: flex-start;flex-wrap: nowrap;"

                for (let __i = 0; __i < __mem_dims[1]; ++__i) {
                    // Check ellision
                    if (__mem_dims[1] > __ellision_thresh_x && __i > __ellision_thresh_x / 2 && __i < __mem_dims[1] - __ellision_thresh_x / 2) {
                        // Elide
                        if (__i - 1 == __ellision_thresh_x / 2) {
                            // Add ellision info _once_
                            let __cell = document.createElement('div');
                            __cell.style = "line-height: 1px;";
                            //let __colorstr = "background: white;";
                            //__cell.style = "min-width: " + __size + "px; min-height: " + __size + "px;border: 1px solid black;" + __colorstr;
                            __cell.innerText = "...";
                            __row.appendChild(__cell);
                        }
                        continue;
                    }

                    let __set_marking = false;
                    {
                        let __tmp = __mark_cells[__dim_2];
                        if (__tmp != undefined) {
                            if (__tmp.includes(__i)) __set_marking = true;
                        }
                    }

                    let __cell = document.createElement('div');
                    let __colorstr = "background: white;";
                    if (__set_marking) {
                        __colorstr = "background: red;";
                    }

                    __cell.style = "min-width: " + __size + "px; min-height: " + __size + "px;border: 1px solid darkgray;" + __colorstr;
                    __row.appendChild(__cell);
                }

                __tbl_vert.appendChild(__row);
            }
            __tbl_hor.appendChild(__tbl_vert);
        };
        __tbl_container.appendChild(__tbl_hor);

        __create_func();

        return __tbl_container;
    }

    create_visual_range_representation(__starts, __ends, __steps, __tiles, __mayfail = true, __data = null) {
        let __obj_to_arr = x => (x instanceof Array) ? x : [x];

        __starts = __obj_to_arr(__starts);
        __ends = __obj_to_arr(__ends);
        __steps = __obj_to_arr(__steps);
        __tiles = __obj_to_arr(__tiles);

        // NOTE: This context is eval()'d. This means that all variables must be in the forbidden namespace (__*) or they might be erroneously evaluated.
        let __symbols = {};

        let __e_starts = [];
        let __e_ends = [];
        let __e_steps = [];
        let __e_tiles = [];
        let __e_sizes = [];

        for (let __r_it = 0; __r_it < __starts.length; ++__r_it) {

            let __start = __starts[__r_it];
            let __end = __ends[__r_it];
            let __step = __steps[__r_it];
            let __tile = __tiles[__r_it];

            let __e_start = null;
            let __e_end = null;
            let __e_step = null;
            let __e_tile = null;

            let __mem_dims = [];
            while (true) {
                let __failed = false;
                let __symbol_setter = Object.entries(__symbols).map(x => "let " + x[0] + "=" + x[1] + ";").join("");
                try {
                    // Define a couple of dace-functions that are used here
                    let Min = (...x) => Math.min(...x);
                    let int_ceil = (a, b) => Math.ceil(a / b);

                    __e_start = eval(__symbol_setter + __start);
                    __e_end = eval(__symbol_setter + __end) + 1; // The DaCe ranges are inclusive - we want to exclusive here.
                    __e_step = eval(__symbol_setter + __step);
                    __e_tile = eval(__symbol_setter + __tile);

                    if (__data != null) {
                        let __shapedims = __data.attributes.shape.length;
                        __mem_dims = [];
                        for (let __s = 0; __s < __shapedims; ++__s) {
                            __mem_dims.push(eval(__symbol_setter + __data.attributes.shape[__s]));
                        }
                    }
                } catch (e) {
                    if (e instanceof ReferenceError) {
                        // Expected, let the user provide inputs
                        if (__mayfail) {
                            __failed = true;
                            break;
                        } else {
                            // Prompt the user and retry
                            let __sym_name = e.message.split(" ")[0];
                            let __ret = window.prompt("Enter a value for Symbol `" + __sym_name + "`");
                            if (__ret == null) throw e;
                            __symbols[__sym_name] = parseInt(__ret);
                            __failed = true;
                        }
                    } else {
                        // Unexpected error, rethrow
                        throw e;
                    }
                }
                if (!__failed) {
                    break;
                }
            }

            let __e_size = __e_end - __e_start;

            __e_starts.push(__e_start);
            __e_ends.push(__e_end);
            __e_steps.push(__e_step);
            __e_tiles.push(__e_tile);
            __e_sizes.push(__e_size);
        }

        let __tbl_container = document.createElement("div");
        let __tbl_hor = document.createElement("div");
        __tbl_hor.classList = "flex_row";
        __tbl_hor.style = "flex-wrap: nowrap;";
        let __axis_x_info = document.createElement("div");
        {
            __axis_x_info.classList = "flex_row";
            __axis_x_info.style = "justify-content: space-between;";


            for (let __i = 0; __i < 2; ++__i) {
                let __tmp = document.createElement("div");
                // Take the second dimension
                try {
                    __tmp.innerText = (__i == 0) ? "0" : __mem_dims[1];
                } catch (__e) {
                    __tmp.innerText = (__i == 0) ? "Start" : "End";
                }
                __axis_x_info.appendChild(__tmp);
            }
        }

        let __axis_y_info = document.createElement("div");
        {
            __axis_y_info.classList = "flex_column";
            __axis_y_info.style = "justify-content: space-between;";


            for (let __i = 0; __i < 2; ++__i) {
                let __tmp = document.createElement("div");
                // Take the first dimension
                try {
                    __tmp.innerText = (__i == 0) ? "0" : __mem_dims[0];
                } catch (__e) {
                    __tmp.innerText = (__i == 0) ? "Start" : "End";
                }
                __axis_y_info.appendChild(__tmp);
            }
        }

        if (__data != null) {
            __tbl_container.appendChild(__axis_x_info);
            __tbl_hor.appendChild(__axis_y_info);
        }
        // Now create a table with the according cells for this
        // Since this is done on a per-dimension basis, the table only has to be 1D, so we use a flexbox for this (easier)
        let __row = document.createElement("div");
        __row.classList = "flex_row";
        __row.style = "justify-content: flex-start;flex-wrap: nowrap;"

        let __size = 10;
        let __e_size = __e_sizes[0]; // #TODO: Adapt for multi-dim ranges if those are requested
        let __e_step = __e_steps[0];
        let __e_tile = __e_tiles[0];

        if (__e_size > 512) __size = 5;

        for (let __i = 0; __i < __e_size; ++__i) {
            let __cell = document.createElement('div');
            let __colorstr = "background: white;";
            if (Math.floor(__i / __e_step) % 2 == 0) {
                if (__i % __e_step < __e_tile) {
                    __colorstr = "background: SpringGreen;";
                }
            } else {
                if (__i % __e_step < __e_tile) {
                    __colorstr = "background: Darkorange;";
                }
            }
            __cell.style = "min-width: " + __size + "px; min-height: " + __size + "px;border: 1px solid black;" + __colorstr;
            __row.appendChild(__cell);
        }

        __tbl_hor.appendChild(__row);
        __tbl_container.appendChild(__tbl_hor);
        return __tbl_container;
    }

    getMatchingInput(transthis, x, node) {

        let create_language_input = (value, onchange) => {
            if (value == undefined) {
                value = x.value;
            }
            if (onchange == undefined) {
                onchange = (elem) => {
                    transthis.propertyChanged(node, x.name, elem.value);
                };
            }
            let language_types = this.getEnum('Language');
            let qualified = value;
            if (!language_types.includes(qualified)) {
                qualified = "Language." + qualified;
            }
            let elem = FormBuilder.createSelectInput("prop_" + x.name, onchange, language_types, qualified);
            return elem;
        };

        let __resolve_initials = (__initials, __syms) => {
            "use strict";
            delete window.i; // Whoever thought it was a good idea to define a global variable named 'i'...
            // We have to operate in the forbidden namespace (__*)

            // Filter out all constants first
            __initials = __initials.filter(x => isNaN(x.var));

            // Add a merger function
            let __merger = (a, b) => {
                let acpy = a.map(x => x);
                for (let y of b) {
                    if (acpy.filter(x => (x == y) || (x.var != undefined && (JSON.stringify(x.var) == JSON.stringify(y.var)))).length > 0) {
                        continue;
                    } else {
                        acpy.push(y);
                    }
                }
                return acpy;
            };

            let __needed_defs = [];
            let __placeholder_defines = [];
            let __user_input_needed = [];
            while (true) {
                let __retry = false;
                let __placeholder_def_str = __placeholder_defines.map(x => "let " + x + " = 1").join(";");
                // Inject the known functions as well
                __placeholder_def_str += ";let Min = (...e) => Math.min(...e); let int_ceil = (a, b) => Math.ceil(a/b);"
                for (let __i of __initials) {
                    // For every initial, find the first defining element (element with the same name that assigns an expression)
                    try {
                        let __test = eval(__placeholder_def_str + __i.var);
                    } catch (e) {
                        if (e instanceof ReferenceError) {
                            let __sym_name = e.message.split(" ")[0];
                            let __defs = __syms.filter(x => x.var == __sym_name && x.val != null);
                            if (__defs.length > 0) {
                                // Found a matching definition
                                __placeholder_defines.push(__sym_name);
                                __needed_defs = __merger(__needed_defs, [__defs[0]]);

                                let __j = 0;
                                for (; __j < __syms.length; ++__j) {
                                    if (JSON.stringify(__syms[__j]) == JSON.stringify(__defs[0])) {
                                        break;
                                    }
                                }
                                let __f = x => x.main != undefined ? x.main : x;

                                // Recurse into range subelements (if applicable)
                                if (__defs[0].val != null && __defs[0].val.start != undefined) {
                                    // find the starting node

                                    let __tmp = __resolve_initials([{
                                        var: __f(__defs[0].val.start),
                                        val: null
                                    }], __syms.slice(__j));
                                    __needed_defs = __merger(__needed_defs, __tmp[1]);
                                    __user_input_needed = __merger(__user_input_needed, __tmp[2]);
                                    __tmp = __resolve_initials([{
                                        var: __f(__defs[0].val.end),
                                        val: null
                                    }], __syms.slice(__j));
                                    __needed_defs = __merger(__needed_defs, __tmp[1]);
                                    __user_input_needed = __merger(__user_input_needed, __tmp[2]);
                                    __tmp = __resolve_initials([{
                                        var: __f(__defs[0].val.step),
                                        val: null
                                    }], __syms.slice(__j));
                                    __needed_defs = __merger(__needed_defs, __tmp[1]);
                                    __user_input_needed = __merger(__user_input_needed, __tmp[2]);
                                    __tmp = __resolve_initials([{
                                        var: __f(__defs[0].val.tile),
                                        val: null
                                    }], __syms.slice(__j));
                                    __needed_defs = __merger(__needed_defs, __tmp[1]);
                                    __user_input_needed = __merger(__user_input_needed, __tmp[2]);
                                } else {
                                    // Recurse into the found value.
                                    let __tmp = __resolve_initials([{
                                        var: __f(__defs[0].val),
                                        val: null
                                    }], __syms.slice(__j));
                                    console.log("rec", __tmp);
                                    // Add elements to lists
                                    __needed_defs = __merger(__needed_defs, __tmp[1]);
                                    __user_input_needed = __merger(__user_input_needed, __tmp[2]);
                                }
                            } else {
                                // Need user input for this Symbol (defer actually requesting that info from the user)
                                __user_input_needed.push(__sym_name);
                                // Also promise to define the symbol later
                                __placeholder_defines.push(__sym_name);
                            }
                            __retry = true;
                            break;
                        } else {
                            // Rethrow unknown exceptions
                            throw e;
                        }
                    }
                    if (__retry) break;
                }
                if (__retry) continue;


                break;
            }
            // Return a (cleaned) list of the required elements
            return [__initials, __needed_defs, __user_input_needed];
        };

        let create_index_subset_input = (transthis, x, node) => {
            // Similar to range, but actually binding values
            // This therefore occurs in memlets inside Maps mostly
            // (Except when accessing using constants)

            // Because the indices are used to access data (arrays),
            // there needs to be lookup by finding the parent nodes (potentially using connectors).
            // A lookup may traverse to top-level and throw if the symbols are not resolved yet.

            let cont = document.createElement("div");

            if (node.data === undefined)
                return $(cont);


            let indices = x.value.indices;

            // Generate string from indices
            let preview = '[';
            for (let index of indices) {
                preview += index + ', ';
            }
            preview = preview.slice(0, -2) + ']';

            cont.innerText = preview + '  ';

            let elem = document.createElement("button");
            elem.style.float = "right";
            elem.innerText = "Edit";
            cont.appendChild(elem);

            elem.addEventListener("click", (_click) => {
                this.project().request(['sdfg_object'], resp => {
                    let tmp = resp['sdfg_object'];
                    let syms = [];
                    for (let v of Object.values(tmp)) {
                        let tsyms = SDFG_Parser.lookup_symbols(v, node.state_id, node.node_id, null);
                        syms = [...syms, ...tsyms];
                        console.log("syms", syms);

                        // Got the symbols, now resolve.

                        // Resolve (ltr is inner-to-outer)
                    }
                    // Rationale here: Render the underlying data as a basis,
                    // then use index and range information to find access patterns

                    // Find the initial values
                    let initials = [];
                    for (let x of syms) {
                        if (x.val != null) break;
                        initials.push(x);
                    }
                    // initials contains the indices used. Resolve all ranges defining those

                    let newelems = __resolve_initials(initials, syms);
                    console.log("newelems", newelems);

                    let data = node.data().props.filter(x => x.name == "data")[0];

                    let data_objs = [];
                    for (let x of Object.values(tmp)) {
                        data_objs.push(x.attributes._arrays[data.value]);
                    }
                    data_objs = data_objs.filter(x => x != undefined);
                    if (data_objs.length > 0) {
                        data = data_objs[0];
                    }

                    let popup_div = document.createElement('div');

                    let popup_div_body = document.createElement('div');

                    let value_input = document.createElement("input");
                    value_input.type = "text";
                    value_input.value = JSON.stringify(x.value);

                    let e = this.create_visual_access_representation(
                        newelems, data
                    );

                    let apply_but = document.createElement("button");
                    apply_but.innerText = "Apply changes";
                    apply_but.addEventListener("click", _click => {
                        transthis.propertyChanged(node, x.name, JSON.parse(value_input.value));
                        w2popup.close();
                    });
                    popup_div_body.appendChild(value_input);

                    popup_div_body.appendChild(e);
                    popup_div.appendChild(popup_div_body);

                    w2popup.open({
                        title: "Data access / Indices property",
                        body: popup_div,
                        buttons: apply_but,
                        width: 1280,
                        height: 800,
                    });
                }, {});
            });

            return $(cont);
        }

        let create_range_input = (transthis, x, node) => {

            // As ranges _usually_ operate on data, check if a property named "data" is in the same object.
            // If it is, we can inform the design of visualizations with the shape of the data object (array)
            // #TODO: Always update this when changed (in the current implementation, it is possible that stale values are read for different properties)
            let data_obj = null;
            if (node.data != undefined) {
                let tmp = node.data().props.filter(x => x.name == "data");
                if (tmp.length > 0) {
                    // Found data (name only, will resolve when rendering is actually requested)
                    data_obj = tmp[0];
                }
            }


            let cont = document.createElement("div");

            if (node.data === undefined)
                return $(cont);


            let ranges = x.value.ranges;
            let popup_div = document.createElement('div');

            // Generate string from range
            let preview = '[';
            for (let range of ranges) {
                preview += range.start + ':' + range.end;
                if (range.step != 1) {
                    preview += ':' + range.step;
                    if (range.tile != 1)
                        preview += ':' + range.tile;
                } else if (range.tile != 1) {
                    preview += '::' + range.tile;
                }
                preview += ', ';
            }
            preview = preview.slice(0, -2) + ']';

            cont.innerText = preview + '  ';

            let elem = document.createElement("button");
            elem.style.float = "right";
            elem.innerText = "Edit";
            cont.appendChild(elem);


            let popup_div_body = document.createElement('div');


            let range_elems = [];
            for (let r of ranges) {
                // Add a row for every range
                let r_row = document.createElement('div');
                r_row.classList = "flex_row";
                r_row.style = "flex-wrap: nowrap;";
                if (typeof (r.start) != 'string') r.start = r.start.main;
                if (typeof (r.end) != 'string') r.end = r.end.main;
                if (typeof (r.step) != 'string') r.step = r.step.main;
                if (typeof (r.tile) != 'string') r.tile = r.tile.main;

                {
                    let input_refs = [];
                    // Generate 4 text inputs and add them to the row
                    for (let i = 0; i < 4; ++i) {
                        // Generate the label first
                        let lbl = document.createElement('label');
                        let ti = document.createElement('input');
                        ti.style = "width:100px;";
                        ti.type = "text";
                        switch (i) {
                            case 0:
                                ti.value = r.start;
                                lbl.textContent = "Start";
                                break;
                            case 1:
                                ti.value = r.end;
                                lbl.textContent = "End";
                                break;
                            case 2:
                                ti.value = r.step;
                                lbl.textContent = "Step";
                                break;
                            case 3:
                                ti.value = r.tile;
                                lbl.textContent = "Tile";
                                break;
                        }
                        input_refs.push(ti);
                        lbl.appendChild(ti);
                        r_row.appendChild(lbl);
                    }
                    range_elems.push(input_refs);
                    let visbut = document.createElement('div');
                    visbut.style = "min-width: 200px; min-height: 1rem;flex-grow: 1;display: flex;";
                    visbut.addEventListener('click', () => {

                        // Resolve the data name and set the object accordingly
                        if (data_obj != null) {

                            this.project().request(['sdfg_object'], sdfg_obj => {
                                if (typeof sdfg_obj.sdfg_object === 'object')
                                    sdfg_obj = sdfg_obj.sdfg_object;
                                else
                                    sdfg_obj = JSON.parse(sdfg_obj.sdfg_object);
                                console.log("got sdfg object", sdfg_obj);
                                // Iterate over all SDFGs, checking arrays and returning matching data elements

                                let data_objs = [];
                                for (let x of Object.values(sdfg_obj)) {
                                    data_objs.push(x.attributes._arrays[data_obj.value]);
                                }
                                data_objs = data_objs.filter(x => x != undefined);
                                if (data_objs.length > 0) {
                                    data_obj = data_objs[0];
                                }

                                let vis_elem = this.create_visual_range_representation(...input_refs.map(x => x.value), false, data_obj);
                                visbut.innerHTML = "";
                                visbut.appendChild(vis_elem);
                            }, {});

                        } else {
                            let vis_elem = this.create_visual_range_representation(...input_refs.map(x => x.value), false, data_obj);
                            visbut.innerHTML = "";
                            visbut.appendChild(vis_elem);
                        }

                    });
                    visbut.innerText = "Click here for visual representation";
                    r_row.appendChild(visbut);
                }


                popup_div_body.appendChild(r_row);
            }

            popup_div.appendChild(popup_div_body);

            let apply_but = document.createElement("button");
            apply_but.innerText = "Apply";
            apply_but.addEventListener("click", () => {
                let ret = {
                    ranges: [],
                    type: x.value.type,
                }
                for (let re of range_elems) {
                    ret.ranges.push({
                        start: re[0].value,
                        end: re[1].value,
                        step: re[2].value,
                        tile: re[3].value
                    })
                }
                transthis.propertyChanged(node, x.name, ret);
                w2popup.close();
            });

            elem.onclick = () => {
                w2popup.open({
                    title: "Range property",
                    body: popup_div,
                    buttons: apply_but,
                    width: 1280,
                    height: 800,
                });
            };
            return $(cont);
        };

        // TODO: Handle enumeration types better
        let elem = document.createElement('div');
        if (x.metatype == "bool") {
            let val = x.value;
            if (typeof (val) == 'string') val = val == 'True';
            elem = FormBuilder.createToggleSwitch("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.checked);

            }, val);
        } else if (
            x.metatype == "str" || x.metatype == "float" || x.metatype == "LambdaProperty"
        ) {
            elem = FormBuilder.createTextInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, x.value);
        } else if (
            x.metatype == "tuple" || x.metatype == "dict" ||
            x.metatype == "list" || x.metatype == "set"
        ) {
            elem = FormBuilder.createTextInput("prop_" + x.name, (elem) => {
                let tmp = elem.value;
                try {
                    tmp = JSON.parse(elem.value);
                } catch (e) {
                    tmp = elem.value;
                }
                transthis.propertyChanged(node, x.name, tmp);
            }, JSON.stringify(x.value));
        } else if (x.metatype == "Range") {
            elem = create_range_input(transthis, x, node);
        } else if (x.metatype == "DataProperty") {
            // The data list has to be fetched from the SDFG.
            // Therefore, there needs to be a placeholder until data is ready
            elem = document.createElement("span");
            elem.innerText = x.value;

            elem = $(elem);
            let cb = d => {
                // Only show data for the inner SDFG (it's possible to input an arbitrary string, still)
                let sdfg = node.sdfg;
                let arrays = sdfg.attributes._arrays;
                let array_names = Object.keys(arrays);

                let new_elem = FormBuilder.createComboboxInput("prop_" + x.name, (elem) => {
                    transthis.propertyChanged(node, x.name, elem.value);
                }, array_names, x.value);

                // Replace the placeholder
                elem[0].parentNode.replaceChild(new_elem[0], elem[0]);
            };
            this.project().request(['sdfg_object'], cb, {
                on_timeout: cb,
                timeout: 300
            });

        } else if (x.metatype == "LibraryImplementationProperty") {
            // The list of implementations has to be fetched from
            // the server directly.
            elem = document.createElement("span");
            elem.innerText = x.value;

            elem = $(elem);
            $.getJSON("/dace/api/v1.0/getLibImpl/" + node.element.classpath, available_implementations => {
                let cb = d => {
                    let new_elem = FormBuilder.createComboboxInput("prop_" + x.name, (elem) => {
                        transthis.propertyChanged(node, x.name, elem.value);
                    }, available_implementations, x.value);

                    // Add Expand button to transform the node
                    let button = FormBuilder.createButton("prop_" + x.name + "_expand",
                        (elem) => {

                            // Expand library node
                            REST_request("/dace/api/v1.0/expand/", {
                                    sdfg: node.sdfg,
                                    nodeid: [0, node.element.parent_id, node.element.id]
                                }, (xhr) => {
                                    if (xhr.readyState === 4 && xhr.status === 200) {
                                        let resp = parse_sdfg(xhr.response);
                                        if (resp.error !== undefined) {
                                            // Propagate error
                                            this.handleErrors(this, resp);
                                        }

                                        // Add to history
                                        this.project().request(["append-history"], x => {
                                        }, { params: {
                                                new_sdfg: resp.sdfg,
                                                item_name: "Expand " + node.element.label
                                            }
                                        });
                                    }
                                });
                        }, "Expand");


                    // Replace the placeholder
                    elem[0].parentNode.replaceChild(new_elem[0], elem[0]);
                    new_elem[0].parentNode.appendChild(button[0]);
                };
                this.project().request(['sdfg_object'], cb, {
                    on_timeout: cb,
                    timeout: 300
                });
            });
        } else if (x.metatype == "CodeProperty") {
            let codeelem = null;
            let langelem = null;
            let onchange = (elem) => {
                transthis.propertyChanged(node, x.name, {
                    'string_data': codeelem[0].value,
                    'language': langelem[0].value
                });
            };
            if (x.value == null) {
                x.value = {};
                x.value.language = "NoCode";
                x.value.string_data = "";
            }
            codeelem = FormBuilder.createLongTextInput("prop_" + x.name, onchange, x.value.string_data);
            elem.appendChild(codeelem[0]);
            langelem = create_language_input(x.value.language, onchange);
            elem.appendChild(langelem[0]);
            elem.classList.add("flex_column");

            return elem;
        } else if (x.metatype == "int") {
            elem = FormBuilder.createIntInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, parseInt(elem.value));
            }, x.value);
        } else if (x.metatype == 'ScheduleType') {
            let schedule_types = this.getEnum('ScheduleType');
            let qualified = x.value;
            if (!schedule_types.includes(qualified)) {
                qualified = "ScheduleType." + qualified;
            }
            elem = FormBuilder.createSelectInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, schedule_types, qualified);
        } else if (x.metatype == 'AccessType') {
            let access_types = this.getEnum('AccessType');
            let qualified = x.value;
            if (!access_types.includes(qualified)) {
                qualified = "AccessType." + qualified;
            }
            elem = FormBuilder.createSelectInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, access_types, qualified);
        } else if (x.metatype == 'Language') {
            elem = create_language_input();
        } else if (x.metatype == 'None') {
            // Not sure why the user would want to see this
            console.log("Property with type 'None' ignored", x);
            return elem;
        } else if (x.metatype == 'object' && ['identity', 'wcr_identity'].includes(x.name)) {
            // This is an internal property - ignore
            return elem;
        } else if (x.metatype == 'OrderedDiGraph') {
            // #TODO: What should we do with this?
            elem = FormBuilder.createTextInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, x.value);
        } else if (x.metatype == 'DebugInfo') {
            // Special case: The DebugInfo contains information where this element was defined
            // (in the original source).
            let info_obj = x.value;
            if (typeof (info_obj) == 'string')
                info_obj = JSON.parse(info_obj);
            elem = FormBuilder.createCodeReference("prop_" + x.name, (elem) => {
                // Clicked => highlight the corresponding code
                transthis.project().request(['highlight-code'], msg => {
                }, {
                    params: info_obj
                });
            }, info_obj);
        } else if (x.metatype == 'ListProperty') {
            // #TODO: Find a better type for this
            elem = FormBuilder.createTextInput("prop_" + x.name, (elem) => {
                let tmp = elem.value;
                try {
                    tmp = JSON.parse(elem.value);
                } catch (e) {
                    tmp = elem.value;
                }
                transthis.propertyChanged(node, x.name, tmp);
            }, JSON.stringify(x.value));
        } else if (x.metatype == "StorageType") {
            let storage_types = this.getEnum('StorageType');
            let qualified = x.value;
            if (!storage_types.includes(qualified)) {
                qualified = "StorageType." + qualified;
            }
            elem = FormBuilder.createSelectInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, storage_types, qualified);
        } else if (x.metatype == "InstrumentationType") {
            let storage_types = this.getEnum('InstrumentationType');
            let qualified = x.value;
            if (!storage_types.includes(qualified)) {
                qualified = "InstrumentationType." + qualified;
            }
            elem = FormBuilder.createSelectInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, storage_types, qualified);
        } else if (x.metatype == "typeclass") {
            // #TODO: Type combobox
            elem = FormBuilder.createTextInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, x.value);
        } else if (x.metatype == "hosttype") {
            elem = FormBuilder.createHostInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, DIODE.getHostList(), x.value);
        } else if (x.metatype == "selectinput") {
            elem = FormBuilder.createSelectInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, x.options, x.value);
        } else if (x.metatype == "combobox") {
            elem = FormBuilder.createComboboxInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, x.options, x.value);
        } else if (x.metatype == "font") {
            console.warn("Ignoring property type ", x.metatype);
            return elem;
        } else if (x.metatype == "SubsetProperty") {
            if (x.value == null) {
                elem = FormBuilder.createTextInput("prop_" + x.name, (elem) => {
                    transthis.propertyChanged(node, x.name, JSON.parse(elem.value));
                }, JSON.stringify(x.value));
            } else if (x.value.type == "subsets.Indices") {
                elem = create_index_subset_input(transthis, x, node);
            } else {
                elem = create_range_input(transthis, x, node);
            }
        } else if (x.metatype == "SymbolicProperty") {
            elem = FormBuilder.createTextInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, JSON.parse(elem.value));
            }, JSON.stringify(x.value));
        } else {
            console.log("Unimplemented property type: ", x);
            alert("Unimplemented property type: " + x.metatype);

            return elem;
        }
        return elem[0];
    }

    renderPropertiesInWindow(transthis, node, params, options) {
        let dobj = {
            transthis: typeof (transthis) == 'string' ? transthis : transthis.created,
            node: node,
            params: params,
            options: options
        };
        this.replaceOrCreate(['display-properties'], 'PropWinComponent', dobj,
            () => {
                let millis = this.getPseudorandom();
                let config = {
                    type: 'component',
                    componentName: 'PropWinComponent',
                    componentState: {
                        created: millis,
                        params: dobj
                    }
                };

                this.addContentItem(config);
            });
    }

    showStaleDataButton() {
        this.project().request(['show_stale_data_button'], x => {
        }, {});
    }

    removeStaleDataButton() {
        this.project().request(['remove_stale_data_button'], x => {
        }, {});
    }

    refreshSDFG() {
        this.gatherProjectElementsAndCompile(diode, {}, {sdfg_over_code: true});
    }

    __impl_showStaleDataButton() {
        /*
            Show a hard-to-miss button hinting to recompile.
        */

        if (DIODE.recompileOnPropertyChange()) {
            // Don't show a warning, just recompile directly
            this.gatherProjectElementsAndCompile(this, {}, {sdfg_over_code: true});
            return;
        }
        if (this._stale_data_button != null) {
            return;
        }
        let stale_data_button = document.createElement("div");
        stale_data_button.classList = "stale_data_button";
        stale_data_button.innerHTML = "Stale project data. Click here or press <span class='key_combo'>Alt-R</span> to synchronize";

        stale_data_button.addEventListener('click', x => {
            this.gatherProjectElementsAndCompile(diode, {}, {sdfg_over_code: true});
        })

        document.body.appendChild(stale_data_button);

        this._stale_data_button = stale_data_button;
    }

    __impl_removeStaleDataButton() {
        if (this._stale_data_button != null) {
            let p = this._stale_data_button.parentNode;
            p.removeChild(this._stale_data_button);
            this._stale_data_button = null;
        }
    }


    showIndeterminateLoading() {
        this.project().request(['show_loading'], x => {
        }, {});
    }

    hideIndeterminateLoading() {
        this.project().request(['hide_loading'], x => {
        }, {});
    }

    __impl_showIndeterminateLoading() {
        $("#loading_indicator").show();
    }

    __impl_hideIndeterminateLoading() {
        $("#loading_indicator").hide();
    }

    hideIndeterminateLoading() {
        this.project().request(['hide_loading'], x => {
        }, {});
    }

    static filterComponentTree(base, filterfunc = x => x) {
        let ret = [];
        for (let x of base.contentItems) {

            ret.push(...this.filterComponentTree(x, filterfunc));
        }

        return ret.filter(filterfunc);
    }

    static filterComponentTreeByCname(base, componentName) {
        let filterfunc = x => x.config.type == "component" && x.componentName == componentName;
        return base.getItemsByFilter(filterfunc);
    }

    groupSDFGs() {
        this.generic_group(x => x.config.type == "component" && x.componentName == "SDFGComponent");
    }

    groupCodeOuts() {
        this.generic_group(x => x.config.type == "component" && x.componentName == "CodeOutComponent");
    }

    groupOptGraph() {
        this.generic_group(x => x.config.type == "component" && x.componentName == "OptGraphComponent");
    }

    groupSDFGsAndCodeOutsTogether() {
        let comps = this.goldenlayout.root.getItemsByFilter(x => x.config.type == "component" && x.componentName == 'SDFGComponent');

        let names = []
        for (let x of comps) {
            names.push(x.config.componentState.sdfg_name);
        }

        for (let n of names) {
            this.generic_group(x => x.config.type == "component" &&
                (x.componentName == "SDFGComponent" || x.componentName == "CodeOutComponent") &&
                x.config.componentState.sdfg_name == n);
        }

        // Get the SDFG elements again to set them active
        comps = this.goldenlayout.root.getItemsByFilter(x => x.config.type == "component" && x.componentName == 'SDFGComponent');
        comps.forEach(x => x.parent.setActiveContentItem(x));

    }

    groupLikeDIODE1() {
        /*
        |---------------------------------------------
        | CodeIn  | Opt   |         SDFG             |
        |         | Tree  |       Renderer           |
        |---------------------------------------------
        |         |          |                       |
        | CodeOut | (Perf)   | Prop Renderer         |
        |         |          |                       |
        |         |          |                       |
        ----------------------------------------------
        */

        this.goldenlayout.eventHub.emit("enter-programmatic-destroy", "");

        // Collect the components to add to the layout later
        let code_ins = DIODE.filterComponentTreeByCname(this.goldenlayout.root, "CodeInComponent");
        let opttrees = DIODE.filterComponentTreeByCname(this.goldenlayout.root, "AvailableTransformationsComponent");
        let opthists = DIODE.filterComponentTreeByCname(this.goldenlayout.root, "TransformationHistoryComponent");
        let sdfg_renderers = DIODE.filterComponentTreeByCname(this.goldenlayout.root, "SDFGComponent");
        let code_outs = DIODE.filterComponentTreeByCname(this.goldenlayout.root, "CodeOutComponent");
        let property_renderers = DIODE.filterComponentTreeByCname(this.goldenlayout.root, "PropWinComponent");

        // Note that this only collects the _open_ elements and disregards closed or invalidated ones
        // Base goldenlayout stretches everything to use the full space available, this makes stuff look bad in some constellations
        // We compensate some easily replacable components here
        if (property_renderers.length == 0) {
            // Add an empty property window for spacing

            let c = this.getPseudorandom();
            property_renderers.push({
                config: {
                    type: 'component',
                    componentName: "PropWinComponent",
                    componentState: {
                        created: c
                    }
                }
            });
        }

        // Remove the contentItems already as a workaround for a goldenlayout bug(?) that calls destroy unpredictably
        let to_remove = [code_ins, code_outs, opttrees, opthists, sdfg_renderers, property_renderers];
        for (let y of to_remove) {
            for (let x of y) {
                if (x.componentName != undefined) {
                    x.remove();
                }
                // Otherwise: Might be a raw config
            }
        }

        // Remove existing content
        let c = [...this.goldenlayout.root.contentItems];
        for (let x of c) {
            this.goldenlayout.root.removeChild(x);
        }


        // Create the stacks (such that elements are tabbed)
        let code_in_stack = this.goldenlayout.createContentItem({
            type: 'stack',
            content: [/*...code_ins.map(x => x.config)*/]
        });
        let opttree_stack = this.goldenlayout.createContentItem({
            type: 'stack',
            content: [/*...opttrees.map(x => x.config)*/]
        });
        let sdfg_stack = this.goldenlayout.createContentItem({
            type: 'stack',
            content: [/*...sdfg_renderers.map(x => x.config)*/]
        });
        let code_out_stack = this.goldenlayout.createContentItem({
            type: 'stack',
            content: [/*...code_outs.map(x => x.config)*/]
        });
        let property_stack = this.goldenlayout.createContentItem({
            type: 'stack',
            content: [/*...property_renderers.map(x => x.config)*/]
        });

        let top_row = this.goldenlayout.createContentItem({
            type: 'row',
            content: [/*code_in_stack, opttree_stack, sdfg_stack*/]
        });

        let bottom_row = this.goldenlayout.createContentItem({
            type: 'row',
            content: [/*code_out_stack, property_stack*/]
        });


        let top_bottom = this.goldenlayout.createContentItem({
            type: 'column',
            content: [/*top_row, bottom_row*/]
        });
        // Now add the new layout construction
        this.goldenlayout.root.addChild(top_bottom);

        top_bottom.addChild(top_row);
        top_bottom.addChild(bottom_row);


        top_row.addChild(code_in_stack);
        top_row.addChild(opttree_stack);
        top_row.addChild(sdfg_stack);

        bottom_row.addChild(code_out_stack)
        bottom_row.addChild(property_stack);

        sdfg_renderers.forEach(x => sdfg_stack.addChild(x.config));
        property_renderers.forEach(x => property_stack.addChild(x.config));
        code_outs.forEach(x => code_out_stack.addChild(x.config));
        opttrees.forEach(x => opttree_stack.addChild(x.config));
        opthists.forEach(x => opttree_stack.addChild(x.config));
        code_ins.forEach(x => code_in_stack.addChild(x.config));

        // Everything has been added, but maybe too much: There might be empty stacks.
        // They should be removed to keep a "clean" appearance
        for (let x of [opttree_stack, code_in_stack, sdfg_stack, property_stack]) {
            if (x.contentItems.length == 0) {
                x.remove();
            }
        }


        this.goldenlayout.eventHub.emit("leave-programmatic-destroy", "");
    }

    generic_group(predicate) {
        /*
            Groups the SDFGs into their own Stack

        */
        this.goldenlayout.eventHub.emit("enter-programmatic-destroy", "");
        let sdfg_components = this.goldenlayout.root.getItemsByFilter(predicate);

        if (sdfg_components.length == 0) {
            console.warn("Cannot group, no elements found");
        }
        let new_container = this.goldenlayout.createContentItem({
            type: 'stack',
            contents: []
        });

        for (let x of sdfg_components) {
            let config = x.config;
            x.parent.removeChild(x);
            new_container.addChild(config);
        }

        this.addContentItem(new_container);

        this.goldenlayout.eventHub.emit("leave-programmatic-destroy", "");
    }


    addKeyShortcut(key, func, alt = true, ctrl = false) {
        let keys = [...key];
        let c = {
            'alt': alt,
            'ctrl': ctrl,
            'func': func,
            'state': 0,
            'expect': keys.slice(1)
        };
        if (this._shortcut_functions[keys[0]] === undefined) {
            this._shortcut_functions[keys[0]] = [c];
        } else {
            this._shortcut_functions[keys[0]].push(c);
        }
    }

    onKeyUp(ev) {
        if (ev.altKey == false && ev.key == 'Alt') {
            for (let cs of Object.values(this._shortcut_functions)) {
                for (let c of cs) {
                    c.state = 0;
                }
            }
        }
    }

    onKeyDown(ev) {
        for (let cs of Object.values(this._shortcut_functions)) {
            for (let c of cs) {
                if (ev.altKey == false) {
                    c.state = 0;
                    continue;
                }
                if (c.state > 0) {
                    // Currently in a combo-state
                    if (c.expect[c.state - 1] == ev.key) {
                        c.state += 1;
                        if (c.state > c.expect.length) {
                            // Found multi-key combo, activate
                            c.func();
                            ev.stopPropagation();
                            ev.preventDefault();

                            // Clear the states
                            this.onKeyUp({altKey: false, key: 'Alt'});
                            return;
                        }
                    }
                }
            }
        }
        let cs = this._shortcut_functions[ev.key];
        if (cs === undefined) return;

        let i = 0;
        for (let c of cs) {
            if (c.alt == ev.altKey && c.ctrl == ev.ctrlKey) {

                if (c.expect.length > 0) {
                    // Multi-key combo expected
                    c.state += 1;
                    console.log("dict value: ", this._shortcut_functions[ev.key][i]);
                    ++i;
                    continue;
                }
                c.func();

                ev.stopPropagation();
                ev.preventDefault();
                // Clear the states
                this.onKeyUp({altKey: false, key: 'Alt'});
            }
            ++i;
        }
    }

    createNewProject() {
        this._current_project = new DIODE_Project(this);
        this._current_project.clearTransformationHistory();
        sessionStorage.clear();
        window.sessionStorage.setItem("diode_project", this._current_project._project_id);
        this.setupEvents();
    }

    getProject() {
        let proj_id = window.sessionStorage.getItem("diode_project");
        this._current_project = new DIODE_Project(this, proj_id);
        if (proj_id == null || proj_id == undefined) {
            // There was a new project ID assigned, which is stored again in the session storage
            window.sessionStorage.setItem("diode_project", this.getCurrentProject()._project_id);
            this.setupEvents();
        }
    }

    project() {
        // Alias function to emulate context behavior
        return this.getCurrentProject();
    }

    getCurrentProject() {
        return this._current_project;
    }

    setLayout(gl) {
        this.goldenlayout = gl;
    }

    settings() {
        return this._settings;
    }

    getPseudorandom() {
        let date = new Date();
        let millis = date.getTime().toString() + Math.random().toFixed(10).toString() + this._creation_counter.toString();

        ++this._creation_counter;

        console.assert(millis !== undefined, "Millis well-defined");

        return millis;
    }

    multiple_SDFGs_available(sdfgs) {

        let sdfgs_obj = (typeof (sdfgs) == 'string') ? parse_sdfg(sdfgs) : sdfgs;

        for (let x of Object.keys(sdfgs_obj.compounds)) {
            // New sdfg
            let value = sdfgs_obj.compounds[x];
            this.SDFG_available(value, x);
        }
    }

    SDFG_available(sdfg, name = "sdfg") {
        // We create a new component and add it to the layout

        // To provide some distinguation, milliseconds since epoch are used.
        let millis = () => this.getPseudorandom();

        sdfg.sdfg_name = name;

        let create_sdfg_func = () => {
            let new_sdfg_config = {
                title: name,
                type: 'component',
                componentName: 'SDFGComponent',
                componentState: {created: millis(), sdfg_data: sdfg, sdfg_name: name}
            };
            this.addContentItem(new_sdfg_config);
        };
        this.replaceOrCreate(['new-sdfg'], 'SDFGComponent', JSON.stringify(sdfg), create_sdfg_func);

        let create_codeout_func = () => {
            let new_codeout_config = {
                title: "Generated Code",
                type: 'component',
                componentName: 'CodeOutComponent',
                componentState: {created: millis(), code: sdfg, sdfg_name: name}
            };
            this.addContentItem(new_codeout_config);
        }
        if (sdfg.generated_code != undefined) {
            console.log("requesting using ID", this.project());
            this.replaceOrCreate(['new-codeout'], 'CodeOutComponent', sdfg, create_codeout_func);
        }
    }

    Error_available(error) {
        let create_error_func = () => {
            let new_error_config = {
                title: "Error",
                type: 'component',
                componentName: 'ErrorComponent',
                componentState: {error: error}
            };
            this.addContentItem(new_error_config);
        };
        this.replaceOrCreate(['new-error'], 'ErrorComponent', error, create_error_func);
    }

    OptGraph_available(optgraph, name = "") {

        if (typeof optgraph != "string") {
            optgraph = JSON.stringify(optgraph);
        }

        // To provide some distinction, milliseconds since epoch are used.
        let millis = this.getPseudorandom();

        let create_optgraph_func = () => {
            let new_optgraph_config = {
                type: "column",
                content: [{
                    title: name == "" ? "Transformations" : "Transformations for `" + name + "`",
                    type: 'component',
                    componentName: 'AvailableTransformationsComponent',
                    componentState: {created: millis, for_sdfg: name, optgraph_data: optgraph}
                }]
            };
            this.addContentItem(new_optgraph_config);
        };
        this.replaceOrCreate(['new-optgraph-' + name], 'AvailableTransformationsComponent',
            optgraph, create_optgraph_func);

        let create_history_func = () => {
            let new_optgraph_config = {
                type: "column",
                content: [{
                    title: "History",
                    type: 'component',
                    componentName: 'TransformationHistoryComponent',
                    componentState: {created: millis, for_sdfg: name}
                }]
            };
            this.addContentItem(new_optgraph_config);
        };
        this.replaceOrCreate(['new-history-' + name], 'TransformationHistoryComponent',
            optgraph, create_history_func);

    }

    OptGraphs_available(optgraph) {
        let o = optgraph;
        if (typeof (o) == "string") {
            o = JSON.parse(optgraph);
        }

        for (let x of Object.keys(o)) {
            let elem = o[x];
            this.OptGraph_available(elem, x);
        }
    }

    gatherProjectElementsAndCompile(calling_context, direct_passing = {}, options = {}) {
        /*
            This method collects all available elements that can be used for compilation.

            direct_passing: Elements that do not have to be requested, but are available at call time
                .code:          Input code
                .sdfg_props:    SDFG properties deviating from default or changed
                .optpath:       Optimization path to be applied

            options
                .run:           If set to true, the Program is run after compilation
                .no_optgraph:   If set to true, the optgraph is not updated/created
                .term_id        The terminal identifier (if output should be shown), required when run is `true`
                .sdfg_over_code Use the existing SDFG-Serialization if available instead of recompiling from source
                .collect_cb     If set, the collected elements are passed to this function
                .dry_run        If `true`, no operation is passed to the server. This is useful when collecting elements for e.g. saving.
        */

        let code = direct_passing.code;
        let sdfg_props = direct_passing.sdfg_props;
        let optpath = direct_passing.optpath;

        let reqlist = [];
        if (code === undefined) {
            if (options.sdfg_over_code) {
                reqlist.push('sdfg_object');
            }
            reqlist.push('input_code');
        }

        if (optpath === undefined) reqlist.push('optpath');
        /*if(optpath != undefined) {
            optpath = undefined;
            reqlist.push('optpath');
        }*/


        let on_collected = (values) => {
            if (code != undefined) values['input_code'] = code;
            if (sdfg_props != undefined) values['sdfg_props'] = sdfg_props;
            if (optpath != undefined) values['optpath'] = optpath;

            if (options.collect_cb != undefined)
                options.collect_cb(values);

            if (options.dry_run === true)
                return;

            let cis = values['sdfg_object'] != undefined;
            let cval = values['input_code'];

            // Assuming SDFG files start with {
            if (cval[0] == '{') {
                let sd = parse_sdfg(cval);
                values['sdfg_object'] = {};
                values['sdfg_object'][sd.attributes.name] = cval;

                cis = true;
            }

            if (cis) {
                cval = values['sdfg_object'];
                if (typeof (cval) == 'string')
                    cval = parse_sdfg(cval);
            }

            calling_context.project().request(["clear-errors"], () => {
            });

            if (options.run === true) {
                let runopts = {};
                if (options['perfmodes']) {
                    runopts['perfmodes'] = options['perfmodes'];
                }
                runopts['repetitions'] = 5; // TODO(later): Allow users to configure number
                runopts['code_is_sdfg'] = cis;
                runopts['runnercode'] = values['input_code'];
                this.compile_and_run(calling_context, options.term_id, cval, values['optpath'], values['sdfg_props'], runopts);
            } else {
                let cb = (resp) => {
                    this.replaceOrCreate(['extend-optgraph'], 'AvailableTransformationsComponent', resp, (_) => {
                        this.OptGraphs_available(resp);
                    });
                };
                if (options['no_optgraph'] === true) {
                    cb = undefined;
                }

                this.compile(calling_context, cval, values['optpath'], values['sdfg_props'],
                    {
                        optpath_cb: cb,
                        code_is_sdfg: cis,
                    });

            }
        }

        calling_context.project().request(reqlist, on_collected, {timeout: 500, on_timeout: on_collected});
    }

    compile(calling_context, code, optpath = undefined, sdfg_node_properties = undefined, options = {}) {
        /*
            options:
                .code_is_sdfg: If true, the code parameter is treated as a serialized SDFG
                .runnercode: [opt] Provides the python code used to invoke the SDFG program if needed
        */
        let post_params = {};
        if (options.code_is_sdfg === true) {
            post_params = {"sdfg": stringify_sdfg(code)};

            post_params['code'] = options.runnercode;
        } else {
            post_params = {"code": code};
        }

        if (optpath != undefined) {
            post_params['optpath'] = optpath;
        }
        if (sdfg_node_properties != undefined) {
            post_params['sdfg_props'] = sdfg_node_properties;
        }
        post_params['client_id'] = this.getClientID();
        let version_string = "1.0";
        REST_request("/dace/api/v" + version_string + "/compile/dace", post_params, (xhr) => {
            if (xhr.readyState === 4 && xhr.status === 200) {

                let peek = parse_sdfg(xhr.response);
                if (peek['error'] != undefined) {
                    // There was at least one error - handle all of them
                    this.handleErrors(calling_context, peek);
                } else {
                    // Data is no longer stale
                    this.removeStaleDataButton();

                    let o = parse_sdfg(xhr.response);
                    this.multiple_SDFGs_available(xhr.response);
                    if (options.optpath_cb === undefined) {
                        this.OptGraphs_available(o['compounds']);
                    } else {
                        options.optpath_cb(o['compounds']);
                    }
                }
            }
        });
    }

    handleErrors(calling_context, object) {
        let errors = object['error'];
        if ('traceback' in object)
            errors += '\n\n' + object.traceback;

        this.Error_available(errors);

        if (typeof (errors) == "string") {
            console.warn("Error: ", errors);
            //alert(JSON.stringify(errors));
            return;
        }
        for (let error of errors) {

            if (error.type === "SyntaxError") {
                // This error is most likely caused exclusively by input code

                calling_context.project().request(['new_error'], msg => {
                    },
                    {
                        params: error,
                        timeout: 100,
                    });
            } else {
                console.warn("Error: ", error);
                //alert(JSON.stringify(error));
            }
        }
    }

    ui_compile_and_run(calling_context) {

        let millis = this.getPseudorandom();

        let terminal_identifier = "terminal_" + millis;

        // create a new terminal
        let terminal_config = {
            title: "Terminal",
            type: 'component',
            componentName: 'TerminalComponent',
            componentState: {created: millis}
        };
        this.addContentItem(terminal_config);


        this.gatherProjectElementsAndCompile(this, {}, {run: true, term_id: terminal_identifier, sdfg_over_code: true});
    }

    load_perfdata() {
        this.showIndeterminateLoading();
        let client_id = this.getClientID();


        let post_params = {
            client_id: client_id
        };
        REST_request("/dace/api/v1.0/perfdata/get/", post_params, (xhr) => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                let pd = JSON.parse(xhr.response);
                console.log("Got result", pd);

                let ondone = () => {
                    this.hideIndeterminateLoading();
                };

                this.project().request(['draw-perfinfo'], x => {
                    ondone();
                }, {
                    params: pd,
                    on_timeout: ondone
                });
            }
        });
    }

    show_exec_times() {
        let config = {
            type: 'component',
            componentName: 'PerfTimesComponent',
            componentState: {},
            title: "Execution times",
        };

        this.addContentItem(config);
    }

    show_run_options(calling_context) {
        let newconf = {
            type: "component",
            componentName: "RunConfigComponent",
            title: "Run Configuration"
        };

        this.addContentItem(newconf);
    }

    show_inst_options(calling_context) {
        let newconf = {
            type: "component",
            componentName: "InstControlComponent",
            title: "Instrumentation control"
        };

        this.addContentItem(newconf);
    }

    show_roofline(calling_context) {
        let newconf = {
            type: "component",
            componentName: "RooflineComponent",
            title: "Roofline"
        };

        this.addContentItem(newconf);
    }

    compile_and_run(calling_context, terminal_identifier, code, optpath = undefined, sdfg_node_properties = undefined, options = {}) {
        /*
            .runnercode: [opt] Code provided with SDFG to invoke the SDFG program. 
        */
        let post_params = {};
        if (options.code_is_sdfg === true) {
            post_params = {"sdfg": stringify_sdfg(code)};
            post_params['code'] = options.runnercode;
        } else {
            post_params = {"code": code};
        }
        if (optpath != undefined) {
            post_params['optpath'] = optpath;
        }
        if (sdfg_node_properties != undefined) {
            post_params['sdfg_props'] = sdfg_node_properties;
        }
        this.applyCurrentRunConfig().then((remaining_settings) => {
            let client_id = this.getClientID();
            post_params['client_id'] = client_id;
            if (remaining_settings['Instrumentation'] == 'off') {
                post_params['perfmodes'] = undefined;
            } else if (remaining_settings['Instrumentation'] == 'minimal') {
                post_params['perfmodes'] = ["default"];
            } else if (remaining_settings['Instrumentation'] == 'full') {
                post_params['perfmodes'] = ["default", "vectorize", "memop", "cacheop"];
            } else {
                alert("Error! Check console");
                console.error("Unknown instrumentation mode", remaining_settings['Instrumentation']);
            }
            //post_params['perfmodes'] = ["default", "vectorize", "memop", "cacheop"];
            let not = remaining_settings['Number of threads'];
            if (typeof (not) == "string") {
                not = JSON.parse(not);
            }
            post_params['corecounts'] = not.map(x => parseInt(x));
            post_params['repetitions'] = 5; // TODO(later): Allow users to configure number
            //post_params['corecounts'] = [1,2,3,4];
            let version_string = "1.0";
            REST_request("/dace/api/v" + version_string + "/run/", post_params, (xhr) => {
                if (xhr.readyState === 4 && xhr.status === 200) {

                    let tmp = xhr.response;
                    if (typeof (tmp) == 'string') tmp = JSON.parse(tmp);
                    if (tmp['error']) {
                        // Normal, users should poll on a different channel now.
                        this.display_current_execution_status(calling_context, terminal_identifier, client_id);
                    }
                }
            });
        });
    }

    display_current_execution_status(calling_context, terminal_identifier, client_id, perf_mode = undefined) {
        let post_params = {};
        post_params['client_id'] = client_id;
        post_params['perf_mode'] = perf_mode;
        let version_string = "1.0";
        REST_request("/dace/api/v" + version_string + "/run/status/", post_params, (xhr) => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                // Show success/error depending on the exit code
                if (xhr.response.endsWith(" 0"))
                    this.toast("Execution ended", "Run ended successfully", 'info');
                else
                    this.toast("Execution ended", "Run failed", 'error');

                // Flush remaining outputs
                let newdata = xhr.response.substr(xhr.seenBytes);
                this.goldenlayout.eventHub.emit(terminal_identifier, newdata);
                xhr.seenBytes = xhr.responseText.length;
            }
            if (xhr.readyState === 3) {
                let newdata = xhr.response.substr(xhr.seenBytes);
                this.goldenlayout.eventHub.emit(terminal_identifier, newdata);
                xhr.seenBytes = xhr.responseText.length;
            }
        });
    }

    toast(title, text, type = 'info', timeout = 10000, icon = undefined, callback = undefined) {
        let toast = VanillaToasts.create({
            title: title,
            text: text,
            type: type, // #TODO: Maybe add check for exit codes as well? (to show success/error)
            icon: icon,
            timeout: timeout,
            callback: callback
        });

        VanillaToasts.setTimeout(toast.id, timeout * 1.1);
    }

    optimize(calling_context, optpath) {
        // The calling_context might be necessary when multiple, different programs are allowed

        if (optpath === undefined) {
            optpath = [];
        }
        let transthis = this;

        let on_data_available = (code_data, sdfgprop_data, from_code) => {
            let code = null;
            if (from_code) {
                code = code_data;
            } else {
                code = code_data;
            }

            let props = null;
            if (sdfgprop_data != undefined)
                props = sdfgprop_data.sdfg_props;
            else
                props = undefined;

            let cb = (resp) => {
                transthis.replaceOrCreate(['extend-optgraph'], 'AvailableTransformationsComponent', resp, (_) => {
                    transthis.OptGraphs_available(resp);
                });
            };

            transthis.compile(calling_context, code, optpath, props, {
                optpath_cb: cb,
                code_is_sdfg: !from_code
            });

        }


        calling_context.project().request(['input_code', 'sdfg_object'], (data) => {

            let from_code = true;
            if (data['sdfg_object'] != undefined) {
                from_code = false;
                data = data['sdfg_object'];
                data = JSON.parse(data);
            } else {
                data = data['input_code'];
            }
            on_data_available(data, undefined, from_code);

        });
    }

    /*
        Tries to talk to a pre-existing element to replace the contents.
        If the addressed element does not exist, a new element is created.
    */
    replaceOrCreate(replace_request, window_name, replace_params, recreate_func) {
        let open_windows = null;
        if (this.goldenlayout.root)
            open_windows = this.goldenlayout.root.getItemsByFilter(x => (x.config.type == "component" &&
                x.componentName == window_name));

        if (open_windows && open_windows.length > 0) {  // Replace
            this.getCurrentProject().request(replace_request, (resp, tid) => {
                },
                {
                    timeout: null,
                    params: replace_params,
                    timeout_id: null
                });
        } else {  // Create
            recreate_func(replace_params);
        }
    }

    /*
        This function is used for debouncing, i.e. holding a task for a small amount of time
        such that it can be replaced with a newer function call which would otherwise get queued.
    */
    debounce(group, func, timeout) {

        if (this._debouncing === undefined) {
            // The diode parent object was not created. The function cannot be debounced in this case.
            return func;
        }
        let transthis = this;
        let debounced = function () {
            if (transthis._debouncing[group] !== undefined) {
                clearTimeout(transthis._debouncing[group]);
            }
            transthis._debouncing[group] = setTimeout(func, timeout);
        };

        return debounced;
    }

    static editorTheme() {
        let theme = localStorage.getItem('diode_ace_editor_theme');
        if (theme === null) {
            return "github";
        }
        return theme;
    }

    static themeString() {
        return "ace/theme/" + DIODE.editorTheme();
    }

    static loadTheme() {
        return $.getScript("external_lib/ace/theme-" + DIODE.editorTheme() + ".js");
    }

    static setTheme(name) {
        localStorage.setItem('diode_ace_editor_theme', name);
    }

    static recompileOnPropertyChange() {
        // Set a tendency towards 'false' 
        return localStorage.getItem('diode_recompile_on_prop_change') == "true";
    }

    static setRecompileOnPropertyChange(boolean_value) {
        if (boolean_value) {
            localStorage.setItem('diode_recompile_on_prop_change', "true");
        } else {
            localStorage.setItem('diode_recompile_on_prop_change', "false");
        }
    }

    static setDebugDevMode(boolean_value) {
        if (boolean_value) {
            localStorage.setItem('diode_DebugDevMode', "true");
        } else {
            localStorage.setItem('diode_DebugDevMode', "false");
        }
    }

    static debugDevMode() {
        /*
            The DebugDev mode determines if internal, not-crucial-for-user properties are shown.
        */
        let v = localStorage.getItem("diode_DebugDevMode");
        return v === "true";
    }
}


export {
    DIODE,
    DIODE_Context_SDFG,
    DIODE_Context_CodeIn,
    DIODE_Context_CodeOut,
    DIODE_Context_Settings,
    DIODE_Context_Terminal,
    DIODE_Context_DIODESettings,
    DIODE_Context_PropWindow,
    DIODE_Context,
    DIODE_Context_Runqueue,
    DIODE_Context_StartPage,
    DIODE_Context_TransformationHistory,
    DIODE_Context_AvailableTransformations,
    DIODE_Context_Error,
    DIODE_Context_RunConfig,
    DIODE_Context_PerfTimes,
    DIODE_Context_InstrumentationControl,
    DIODE_Context_Roofline
}
