import {REST_request, FormBuilder, setup_drag_n_drop} from "./main.js"
import * as renderer from "./renderer_dir/renderer_main.js"
import { Appearance } from "./diode_appearance.js"
import { SDFG_Parser, SDFG_State_Parser, SDFG_Node_Parser } from "./sdfg_parser.js"
import * as DiodeTables from "./table.js"


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
        if(this.created == undefined) {
            this.created = this.diode.getPseudorandom();
            this.saveToState({created: this.created});
        }
        this._project = null;
        if(state.project_id != undefined && state.project_id != null && state.project_id != "null") {
            this._project = new DIODE_Project(this.diode, state.project_id);
        }

        this._event_listeners = [];

        this._no_user_destroy = false; // When true, all destroy()-events are assumed to be programmatic
    }

    project() {
        console.assert(this._project != null, "Project invalid");
        return this._project;
    }

    on(name, data) {
        let eh = this.diode.goldenlayout.eventHub;

        let params = [name, data];
        eh.on(...params);
        this._event_listeners.push(params);
    }

    destroy() {

        console.log("destroying", this);
        // Unlink all event listeners
        let eh = this.diode.goldenlayout.eventHub;
        for(let x of this._event_listeners) {
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
    }

    setupEvents(project) {
        if(this._project == null) {
            this._project = project;
        }

        this.container.extendState({'project_id': this._project.eventString('')})

        this.on('destroy-' + this.getState().created, msg => {
            if(!this._no_user_destroy) {
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

        this.initialized_sdfgs = [];

        this._analysis_values = {};

        console.log("state", state);
    }

    saveToState(dict_value) {
        let renamed_dict = {};
        let json_list = ['sdfg_data', 'sdfg'];
        for(let x of Object.entries(dict_value)) {
            renamed_dict[x[0]] = (json_list.includes(x[0]) && typeof(x[1]) != 'string') ? JSON.stringify(x[1]) : x[1];
        }
        super.saveToState(renamed_dict);

        console.assert(this.getState().sdfg == undefined);
    }

    resetState(dict_value) {
        let renamed_dict = {};
        let json_list = ['sdfg_data', 'sdfg'];
        for(let x of Object.entries(dict_value)) {
            renamed_dict[x[0]] = (json_list.includes(x[0]) && typeof(x[1]) != 'string') ? JSON.stringify(x[1]) : x[1];
        }
        super.resetState(renamed_dict);
        console.assert(this.getState().sdfg == undefined);
    }

    getState() {
        let _state = super.getState();
        let _transformed_state = {};
        let json_list = ['sdfg_data', 'sdfg'];
        for(let x of Object.entries(_state)) {
            _transformed_state[x[0]] = (typeof(x[1]) == 'string' && json_list.includes(x[0])) ? JSON.parse(x[1]) : x[1];
        }
        return _transformed_state;
    }

    setupEvents(project) {
        super.setupEvents(project);

        let transthis = this;

        let eh = this.diode.goldenlayout.eventHub;
        this.on(this._project.eventString('-req-new-sdfg'), (msg) => {

            if(typeof(msg) == 'string') msg = JSON.parse(msg);
            if(msg.sdfg_name === this.getState()['sdfg_name']) {
                // Ok
            }
            else {
                // Names don't match - don't replace this one then.
                // #TODO: This means that renamed SDFGs will not work as expected.
                return;
            }
            eh.emit(this.project().eventString('new-sdfg'), 'ok');
            this.render_sdfg(msg, true);
        });

        // #TODO: When multiple sdfgs are present, the event name
        // should include a hash of the target context
        this.on(this._project.eventString('-req-sdfg-msg'), msg => {

            let ret = this.message_handler_filter(msg);
            if(ret === undefined) {
                ret = 'ok';
            }
            eh.emit(transthis._project.eventString('sdfg-msg'), ret);
        });

        this.on(this._project.eventString('-req-sdfg_props'), msg => {
            // Echo with data
            if(msg != undefined) {
                this.discardInvalidated(msg);
            }
            let resp = this.getChangedSDFGPropertiesFromState();
            let named = {};
            named[this.getState()['sdfg_name']] = resp;
            eh.emit(transthis._project.eventString('sdfg_props'), named);
        });

        this.on(this.project().eventString('-req-property-changed-' + this.getState().created), (msg) => {
            // Emit ok directly (to avoid caller timing out)
            eh.emit(this.project().eventString("property-changed-" + this.getState().created), "ok");
            
            if(msg.type == "symbol-properties") {
                this.symbolPropertyChanged(msg.node, msg.name, msg.value);
            }
            else
                this.propertyChanged(msg.node, msg.name, msg.value);
            
        });

        this.on(this.project().eventString('-req-draw-perfinfo'), (msg) => {
            this._analysis_values = msg.map(x => ({
                forProgram: x[0],
                AnalysisName: x[1],
                runopts: x[2],
                forUnifiedID: x[3],
                forSuperSection: x[4],
                forSection: x[5],
                data: JSON.parse(x[6]),
            }));
            for(let x of this.initialized_sdfgs) {
                x.drawAllPerfInfo();
            }
        });

        this.on(this.project().eventString('-req-sdfg_object'), msg => {
            // Return the entire serialized SDFG
            let _state = this.getSDFGDataFromState();
            let sdfg = _state.type == 'SDFG' ? _state : _state.sdfg;
            let named = {};
            named[this.getState()['sdfg_name']] = sdfg;
            named = JSON.stringify(named);
            eh.emit(this.project().eventString("sdfg_object"), named);
        });
    }

    analysisProvider(aname, nodeinfo) {

        let unified_id = null;
        if(nodeinfo != null) {
            unified_id = (parseInt(nodeinfo.stateid) << 16) | parseInt(nodeinfo.nodeid);
        }
        console.log("analysisProvider", aname, nodeinfo);
        if(aname == "getstates") {

            let states = this._analysis_values.map(x => (x.forUnifiedID >> 16) & 0xFFFF);
            return states;
        }
        else if(aname == "getnodes") {
            let nodes = this._analysis_values.map(x => (x.forUnifiedID) & 0xFFFF);
            return nodes;
        }
        else if(aname == "all_vec_analyses") {
            let vec_analyses = this._analysis_values.filter(x => x.AnalysisName == 'VectorizationAnalysis');
            let fltrd_vec_analyses = vec_analyses.filter(x => x.forUnifiedID == unified_id);
            return fltrd_vec_analyses;
        }
        else if(aname == 'CriticalPathAnalysis') {
            let cpa = this._analysis_values.filter(x => x.AnalysisName == 'CriticalPathAnalysis');
            let filtered = cpa.filter(x => x.forUnifiedID == unified_id);
            return filtered;
        }
        else if(aname == 'ParallelizationAnalysis') {
            let pa = this._analysis_values.filter(x => x.AnalysisName == 'ThreadAnalysis');
            let filtered = pa.filter(x => x.forUnifiedID == unified_id);
            return filtered;
        }
        else if(aname == 'MemoryAnalysis') {
            let ma = this._analysis_values.filter(x => x.AnalysisName == 'MemoryAnalysis');
            let filtered = ma.filter(x => x.forUnifiedID == unified_id);
            return filtered;
        }
        else if(aname == 'MemOpAnalysis') {
            let moa = this._analysis_values.filter(x => x.AnalysisName == 'MemoryOpAnalysis');
            let filtered = moa.filter(x => x.forUnifiedID == unified_id);
            return filtered;
        }
        else if(aname == 'CacheOpAnalysis') {
            let coa = this._analysis_values.filter(x => x.AnalysisName == 'CacheOpAnalysis');
            let filtered = coa.filter(x => x.forUnifiedID == unified_id);
            return filtered;
        }
        else if(aname == "defaultRun") {
            // This pseudo-element returns a filter function that returns only elements from the "default" run configuration
            // #TODO: Make the default run configurable.
            // For now, the default run is the run with the most committed cores
            return x => x.filter(y => y.runopts == '# ;export OMP_NUM_THREADS=4; Running in multirun config');
        }
        else {
            throw "#TODO";
        }
    }

    message_handler_filter(msg) {
        /* 
            This function is a compatibility layer
        */
        msg = JSON.parse(msg);
        if(msg.sdfg_name != this.getState()['sdfg_name']) {
            return;
        }
        if(msg.type == 'highlight-elements') {
            // The input contains a list of multiple elements
            for(let x of msg.elements) {

                let split = x.split('_');


                let modmsg = {
                    type: 'highlight-element',
                    'sdfg-id': split[0].slice(1), // Cut off the leading 's'. #TODO: Fix the misnomer sdfg-id => state-id
                    'node-id': split[1]
                };
                this._message_handler(modmsg);
            }
        }
        else {
            // Default behavior is passing through (must be an object, not JSON-string)
            this._message_handler(msg);
        }
    }

    render_free_variables() {
        let sdfg_dat = this.getSDFGDataFromState();
        if(sdfg_dat.type != "SDFG") sdfg_dat = sdfg_dat.sdfg;
        this.project().request(['render-free-vars'], x => {

        }, {
            params: {
                data: sdfg_dat,
                calling_context: this.created
            }
        });
    }

    sdfg_element_selected(msg) {
        console.log("selected sdfg element", msg);

        let omsg = JSON.parse(msg);
        if(omsg.msg_type == 'click') {
            // ok
        }
        else if(omsg.msg_type == 'contextmenu') {
            // ok
        }
        else {
            alert("Unexpected message type '" + omsg.type + "'");
            return;
        }
        let clicked_elems = omsg.clicked_elements;
        let clicked_states = clicked_elems.filter(x => x.type == 'SDFGState');
        let clicked_nodes = clicked_elems.filter(x => x.type != 'SDFGState'); // #DISCUSS: Is it guaranteed that the naming is dual here (SDFGState or everything else)?

        let state_id = null;
        let node_id = null;
        if(clicked_states.length > 1) {
            alert("Cannot determine clicked state!");
            return;
        }
        if(clicked_nodes.length > 1) {
            console.warning("Multiple nodes could be selected - #TODO: Arbitrate");
            //#TODO: Arbitrate this - for now, we select the element with the lowest id
        }

        let state_only = false;

        // Check if anything was clicked at all
        if(clicked_states.length == 0) {
            // Nothing was selected
            this.render_free_variables();
            return;
        }
        if(clicked_nodes.length == 0) {
            // A state was selected
            state_only = true;
        }

        state_id = clicked_states[0].id;
        if(!state_only)
            node_id = clicked_nodes[0].id;
        

        if(omsg.msg_type == "contextmenu") {
            // Context menu was requested
            let spos = omsg.spos;
            let lpos = omsg.lpos;

            let cmenu = new ContextMenu();
            cmenu.addOption("Show transformations", x => {
                console.log("'Show transformations' was clicked");

                this.project().request(['highlight-transformations-' + this.getState()['sdfg_name']], x => {
                    
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
                this.project().request(['get-transformations-' + this.getState()['sdfg_name']], x => {
                    console.log("get-transformations response: ", x);

                    let tmp = Object.values(x)[0];

                    // Create a sub-menu at the correct position


                    let submenu = new ContextMenu();
                    
                    for(let y of tmp) {
                        submenu.addOption(y.opt_name, x => {
                            this.project().request(['apply-transformation-' + this.getState()['sdfg_name']], x => {},
                            { params: y.id_name
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
            cmenu.show(spos.x, spos.y);

            return;
        }

        // Get and render the properties from now on
        let sdfg = this.getSDFGDataFromState().sdfg;

        let unified_id = (parseInt(state_id) << 16) | (parseInt(node_id));
        console.log("sdfg", sdfg);

        let states = sdfg.nodes;
        let state = null;
        for(let x of states) {
            if(x.id == state_id) {
                state = x;
                break;
            }
        }
        let render_props = n => {
            let attr = n.attributes;

            let akeys = Object.keys(attr).filter(x => !x.startsWith("_meta_"));

            let proplist = [];
            for(let k of akeys) {

                let value = attr[k];
                let meta = attr["_meta_" + k];
                if(meta == undefined) {
                    continue;
                }

                let pdata = JSON.parse(JSON.stringify(meta));
                pdata.value = value;
                pdata.name = k;

                proplist.push(pdata);
            }
            let propobj = {
                node_id: node_id,
                state_id: state_id,
                data: () => ({props: proplist})
            };

            this.renderProperties(propobj);
        };
        if(state_only) {
            render_props(state);
            return;
        }

        let nodes = state.nodes;
        for(let n of nodes) {

            if(n.id != node_id)
                continue;

            render_props(n);
            break;
        }
    }

    getSDFGPropertiesFromState() {
        let o = this.getSDFGDataFromState();
        let props = o['sdfg_props'];

        return props;
    }

    getSDFGDataFromState() {
        let _state = this.getState();
        let o = null;
        if(_state.sdfg != undefined) {
            o = _state;
        }
        else {
            o = _state['sdfg_data'];
        }
        if((typeof o) == 'string') {
            o = JSON.parse(o);
        }
        while(typeof o.sdfg == 'string') {
            o.sdfg = JSON.parse(o.sdfg);
        }
        return o;
    }

    renderProperties(node) {
        /*
            node: object, duck-typed
                
        */

        let params = node.data().props;
        let transthis = this;

        // Render in the (single, global) property window
        this.diode.renderPropertiesInWindow(transthis, node, params);
    }

    getNodeReference(node_id, state_id) {
        let o = this.getSDFGDataFromState();
        let sdfg = o['sdfg'];

        for(let x of sdfg.nodes) {
            if(x.id == state_id) {

                if(node_id == null) return [x, sdfg];
                for(let n of x.nodes) {

                    if(n.id == node_id) {
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
        console.log("symbolPropertyChanged", node.data(), name, value);
        
    }

    propertyChanged(node, name, value) {
        /*
            When a node-property is changed, the changed data is written back
            into the state.
        */
        let nref = this.getNodeReference(node.node_id, node.state_id);

        nref[0].attributes[name] = value;

        let old = this.getState();
        if(old.type == "SDFG")
            old = nref[1];
        else
            old.sdfg_data.sdfg = nref[1];

        this.resetState(old);

        this.diode.showStaleDataButton();
    }


    render_sdfg(sdfg_data = undefined, update = false) {
        if(sdfg_data == undefined) {
            sdfg_data = this.getState()["sdfg_data"];
        }
        let tmp = sdfg_data;
        if((typeof tmp) === 'string') {
            tmp = JSON.parse(sdfg_data);
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

        let _dbg_state = this.getState();

        console.assert(JSON.stringify(_dbg_state.sdfg_data) == JSON.stringify(tmp));


        // The HTML5 Renderer originally has been written for a WebSocket interface
        // Not all functionality will work as expected in this environment!

        let create = () => {

            for(let x of this.initialized_sdfgs) {
                this.container.getElement()[0].removeChild(x.getCanvas());
                x.destroy();
            }
            this.initialized_sdfgs = [];
            let sdfg_state = renderer.create_local();
            let cnvs = document.createElement('canvas');
            this.container.getElement().append(cnvs);
            sdfg_state.setCtx(cnvs.getContext("2d"));
            let sdfg = null;
            if(tmp.type == "SDFG") {
                sdfg = tmp;
            } else {
                sdfg = tmp.sdfg;
            }

            
            let transmitter = {
                send: x => this.sdfg_element_selected(x)
            };

            sdfg_state.setSDFG(sdfg);
            sdfg_state.init_SDFG();

            // Link the new message handler
            this._message_handler = msg => renderer.message_handler(msg, sdfg_state);

            // Link a new onclick handler
            sdfg_state.setOnclickHandler(transmitter, true);

            // Enable dragging
            sdfg_state.setDragHandler();

            // Enable scroll-zooming
            sdfg_state.setZoomHandler();

            // Link the analysis provider from which to pull values
            sdfg_state.setAnalysisProvider((x, y) => this.analysisProvider(x,y));

            this.initialized_sdfgs.push(sdfg_state);
        };

        jQuery.when(
            jQuery.getScript('renderer_dir/global_vars.js'),
            jQuery.getScript('renderer_dir/dagre.js'),
            jQuery.getScript('renderer_dir/Chart.bundle.min.js'),
            $.Deferred(function( deferred ){
                $( deferred.resolve );
            })
        ).done(function() {
            create();
        });

        // Prepare to display the access nodes for this sdfg (this is parallel to the creation of the sdfg, ideally)
        let state = new SDFG_Parser(tmp['type'] == 'SDFG' ? tmp : tmp.sdfg);
        let access_nodes = state.getStates().map(x => x.getNodes()).map(x => x.filter(y => y.isNodeType("AccessNode"))).filter(x => x.length > 0);
        console.log("Access nodes", access_nodes);
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
            eh.emit(this.project().eventString('update-tfh'), 'ok');
        });
    }

    create(hist=[]) {
        let parent_element = this.container.getElement();
        $(parent_element).css('overflow', 'auto');

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
        for(let x of hist) {
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
                this.diode.multiple_SDFGs_available({ compounds: tsh[1] });

                // Remove the descending checkpoints
                this.project().discardTransformationsAfter(index);

                if(true) {
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
    }

    setupEvents(project) {
        super.setupEvents(project);

        let transthis = this;

        let eh = this.diode.goldenlayout.eventHub;
        this.on(this._project.eventString('-req-extend-optgraph'), (msg) => {
            
            let o = msg;
            if(typeof(o) == "string") {
                JSON.parse(msg);
            }
            let sel = o[this.getState()['for_sdfg']];
            if(sel === undefined) {
                return;
            }
            eh.emit(transthis._project.eventString('extend-optgraph'), 'ok');
            
            this.create(sel);
        });

        this.on(this._project.eventString('-req-optpath'), msg => {
            let named = {};
            named[this.getState()['for_sdfg']] = [];
            eh.emit(transthis._project.eventString('optpath'), named);
        });

        let sname = this.getState()['for_sdfg'];
        this.on(this._project.eventString('-req-new-optgraph-' + sname), msg => {
            // In any case, inform the requester that the request will be treated
            let o = JSON.parse(msg);
            let sel = o.matching_opts;
            if(sel === undefined) {
                eh.emit(transthis._project.eventString('new-optgraph-' + sname), 'not ok');
                return;
            }
            eh.emit(transthis._project.eventString('new-optgraph-' + sname), 'ok');

            this.create(o);
        });

        this.on(this.project().eventString('-req-highlight-transformations-' + sname), msg => {

            this.getTransformations(msg).forEach(x => this.highlightTransformation(x));
            
        });

        this.on(this.project().eventString('-req-get-transformations-' + sname), msg => {

            let transforms = this.getTransformations(msg);

            eh.emit(transthis._project.eventString('get-transformations-' + sname), transforms);
        });
        
        this.on(this.project().eventString('-req-apply-transformation-' + sname), msg => {
            
            let children = this.getState()['optstruct'];
            for(let c of Object.values(children)) {
                for(let d of c) {
                    
                    if(d === undefined) continue;
                    if(d.id_name == msg) {
                        // Call directly.
                        // (The click handler invokes the simple transformation)
                        d.representative.dispatchEvent(new Event('click'));
                    }
                }
            }
        });

        this.on(this.project().eventString('-req-apply-adv-transformation-' + sname), msg => {

            let x = JSON.parse(msg);
            this.applyTransformation(...x);
        });


        this.on(this.project().eventString('-req-property-changed-' + this.getState().created), (msg) => {
            this.propertyChanged(msg.node, msg.name, msg.value);
            eh.emit(this.project().eventString("property-changed-" + this.getState().created), "ok");
        });
    }

    getTransformations(affecting) {
        let ret = [];
        let selstring = "s" + affecting.state_id + "_" + affecting.node_id;
        let children = this.getState()['optstruct'];
        for(let c of Object.values(children)) {
            for(let d of c) {
                
                if(d === undefined) continue;
                let affects = d.affects;

                if(affects.includes(selstring)) {
                    ret.push(d);
                }
            }
        }
        return ret;
    }

    highlightTransformation(node) {
        let repr = node.representative;
        let s = repr.parentNode;
        while(s) {
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
        /*
            The new transformation model does not allow changing properties
            of an already-applied transformation. This means there is no recompilation
        */

        let tmp = node.data();
        
        tmp[name] = value;
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
        for(let x of og) {
            if(full[x.opt_name] === undefined) {
                full[x.opt_name] = []
            }
            full[x.opt_name].push(x);
        }
        let arrayed = [];
        for(let x of Object.entries(full)) {
            let k = x[0];
            let v = x[1];
            arrayed.push([k, v]);
        }
        let sorted = arrayed.sort((a, b) => a[0].localeCompare(b[0]));
        for(let z of sorted) {
            let y = z[0];
            let x = z[1];

            let i = 0;
            let container_node = undefined;
            if(x.length > 1) {
                
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
            for(let n of x) {
                this.addNode(n, i, container_node);
                ++i;
            }
        }
        let _s = this.getState();
        _s.optstruct = full;
        this.saveToState(_s);
    }

    applyTransformation(x, pos, _title) {
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
            x.sdfg_object = JSON.parse(x.sdfg_object);

            this.project().saveSnapshot(x['sdfg_object'], named);

            this.project().request(['update-tfh'], x => {

                
            }, {

            });

            setTimeout(tmp, 10);
        }, {});
    }

    addNode(x, pos=0, parent_override=undefined) {

        let _title = x.opt_name;

        // Add a suffix
        if(pos != 0) {
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
        
        // Single click show properties (disabled)
        /*title.addEventListener('click', _x => {

            this.renderProperties(x, pos, [x, pos, _title]);
        });*/
        title.addEventListener(/*'dblclick'*/'click', _x => {

            this.applyTransformation(x, pos, _title);
        });
        x.representative = title;

        // Add a control-div
        let ctrl = document.createElement("div");
        // Advanced button
        {
            let adv_button = document.createElement('b');
            adv_button.classList = "";
            adv_button.innerText = '...';

            adv_button.addEventListener('click', _x => {
                // Clicking this should reveal the transformation properties
                // #TODO
                this.renderProperties(x, pos, [x, pos, _title]);
            });

            ctrl.appendChild(adv_button);
        }
        // Help button
        {
            let help_button = document.createElement('i');
            help_button.classList = "";
            help_button.innerText = '?';
            ctrl.appendChild(help_button);
        }

        list_elem.appendChild(title);
        list_elem.appendChild(ctrl);


        at_list.appendChild(list_elem);
    }

    create(newstate=undefined) {
        if(newstate != undefined) {

            let _state = this.getState();
            Object.assign(_state, newstate);
            this.resetState(_state);
        }
        let _state = this.getState();
        if(typeof(_state) == 'string') {
            _state = JSON.parse(_state);
        }
        let matching_opts = undefined;
        if(_state.matching_opts != undefined) {
            matching_opts = _state.matching_opts;
        }
        else if(_state.optgraph_data != undefined) {
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

        if(matching_opts != undefined) {
            this.addNodes(matching_opts);
        }
    }
}

class DIODE_Context_OptGraph extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);

        this._tree_view = null;
        this._current_root = null;

        // Allow overflow
        let parent_element = this.container.getElement();
        $(parent_element).css('overflow', 'auto');
    }

    setupEvents(project) {
        super.setupEvents(project);

        let transthis = this;

        let eh = this.diode.goldenlayout.eventHub;
        this.on(this._project.eventString('-req-extend-optgraph'), (msg) => {
            
            let o = msg;
            if(typeof(o) == "string") {
                JSON.parse(msg);
            }
            let sel = o[this.getState()['for_sdfg']];
            if(sel === undefined) {
                return;
            }
            eh.emit(transthis._project.eventString('extend-optgraph'), 'ok');
            transthis.extendOptGraph(JSON.stringify(sel));
        });

        this.on(this._project.eventString('-req-optpath'), msg => {
            let named = {};
            let _s = this.getState();
            let n = _s['for_sdfg'];
            named[n] = this._current_root.path();
            eh.emit(transthis._project.eventString('optpath'), named);
        });

        let sname = this.getState()['for_sdfg'];
        this.on(this._project.eventString('-req-new-optgraph-' + sname), msg => {
            // In any case, inform the requester that the request will be treated
            let o = JSON.parse(msg);
            let sel = o[sname];
            if(sel === undefined) {
                return;
            }
            eh.emit(transthis._project.eventString('new-optgraph-' + sname), 'ok');

            this.resetOptGraph(msg);            
        });

        this.on(this.project().eventString('-req-highlight-transformations-' + sname), msg => {

            let selstring = "s" + msg.state_id + "_" + msg.node_id;
            let children = this._current_root.children().filter(x => x.pathLabel() != " <virtual>");
            for(let c of children) {
                let d = c.data();
                if(d === undefined) continue;
                let affects = d.affects;

                if(affects.includes(selstring)) {
                    console.log("Found elem!");
                    this.highlightTransformation(c);
                }
            }
        });

        this.on(this.project().eventString('-req-get-transformations-' + sname), msg => {

            let selstring = "s" + msg.state_id + "_" + msg.node_id;
            let children = this._current_root.children().filter(x => x.pathLabel() != " <virtual>");
            let ret = [];
            for(let c of children) {
                let d = c.data();
                if(d === undefined) continue;
                let affects = d.affects;

                if(affects.includes(selstring)) {
                    ret.push([c.label(), c.path()]);
                }
            }

            eh.emit(transthis._project.eventString('get-transformations-' + sname), ret);
        });
        
        this.on(this.project().eventString('-req-apply-transformation-' + sname), msg => {
            let c = this._current_root.getChild(x => x.label() == msg);
            if(c == null) { 
                throw "Node not found";
            }
            this._current_root = c;
            let obj = {};
            obj[this.getState()['for_sdfg']] = c.path();

            this.diode.optimize(this, obj);
        });


        this.on(this.project().eventString('-req-property-changed-' + this.getState().created), (msg) => {
            this.propertyChanged(msg.node, msg.name, msg.value);
            eh.emit(this.project().eventString("property-changed-" + this.getState().created), "ok");
        });
    }

    highlightTransformation(node) {
        let repr = node.representative();

        // Expand all parents
        for(let x = node; x != null; x = x.parent()) {
            let p = x.representative().parent;
            if(p != undefined)
                p.classList.remove("collapsed_sublist");
        }
        let sel = $('.tree_view');
        sel.removeClass('collapsed_sublist');

        $(repr).css('color', 'red');
        setTimeout(x => {
            $(repr).css('color', '');
        }, 5000);
    }

    propertyChanged(node, name, value) {
        return this.propertyChanged2(node, name, value);
        /*
            When a property is changed, the data associated to the transformation node
            must be updated.
            In the old DIODE, the program would be recompiled as well.
            This function debounces calls to a relatively large timeout (2000ms)
            to alleviate overhead. Alternatively, this timeout can be disabled and a
            compile button must be pressed to request compilation from the server.
        */
    }

    propertyChanged2(node, name, value) {
        /*
            When a property is changed, the data associated to the transformation node
            must be updated.
            In the old DIODE, the program would be recompiled as well.
            This function debounces calls to a relatively large timeout (2000ms)
            to alleviate overhead. Alternatively, this timeout can be disabled and a
            compile button must be pressed to request compilation from the server.
        */

        let dbg_tmp = node.data();
        
        node.data()['props'][name] = value;

        // Apply values to saved representation
        

        let f = this.diode.debounce("optproperty-change", () => {
            let named = {};
            named[this.getState()['for_sdfg']] = this._current_root.path();
            //this.diode.optimize(this, this._current_root.path());
            this.diode.optimize(this, named);
        }, 2000);
        if(DIODE.recompileOnPropertyChange()) {
            f();
        }
    }

    renderProperties(node) {
        /*
            node: The TreeNode for which to draw the properties.
        */

        let params = node.data().props;
        let parent = $(this.container.getElement()).children(".optpropdiv");

        parent[0].innerHTML = "";

        let transthis = this;

        // The default node is a (doubly-linked) tree and therefore unsuitable for sending
        let reduced_node = {};
        reduced_node.data = () => node.data();
        this.diode.renderPropertiesInWindow(transthis, reduced_node, params);

    }

    resetOptGraph(optgraph) {
        /*
            This function ensures that on completion, the displayed OptTree is equivalent
            to the structure provided in the function parameter.

            #TODO: Since currently, compilation does not echo the provided OptTree used for compilation,
            #TODO: this function only has to deal with last-level changes (i.e. available Transformations in the current SDFG)
            #TODO: If the compile()-call returns a full OptTree at some point, this has to be adjusted!
        */

        // #TODO: As explained in the function description, this is a shortcut that may have to change in the future.
        this.extendOptGraph(optgraph);
    }

    extendOptGraph(extension) {
        let og = JSON.parse(extension);
        if(this._current_root == null) {
            // No root available
            alert("Did not find valid root");
        }
        else {
            let parent = this._current_root.representative().parentNode;
            
            this._current_root.clearChildren();
            this.addNodes(og);
            
            //Remove all (potentially already existing) sublists of this selection
            let children = Array.from(parent.childNodes);
            children.forEach(x => {
                if(x.nodeName == 'ul' || x.nodeName == 'UL') {
                    // Remove the subelements
                    parent.removeChild(x);
                }
            });

            for(let x of this._current_root.children()) {
                this._tree_view.create_html_in(parent, 0, x);
            }
            // Auto-expand
            parent.classList.remove("collapsed_sublist");

            // Add the expanded element to the state
            let diode_compat = this.treeToDIODEcompatible(this._tree_view._tree);

            // Also store the current path
            let current_path = this._current_root.path(x => x.label());
            this.saveToState({
                'optgraph_data': JSON.stringify({
                    'matching_opts': diode_compat,
                    'extend_path': current_path
                })
            });
        }
    }

    treeToDIODEcompatible(node) {
        /*
            Transforms the tree with root `node` into the DIODE-format
        */

        let tmp = node.data();
        let o = null;
        
        if(tmp === null) {
            o = {
                opt_name: node.label(),
                children: []
            };
        }
        else {
            o = {
                opt_name: node.label(),
                affects: tmp.affects,
                opt_params: tmp.props,
                children: []
            };
        }

        for(let x of node.children()) {
            if(x.pathLabel() != " <virtual>") {
                o.children.push(this.treeToDIODEcompatible(x));
            }
        }

        if(o.opt_name === 'Unoptimized') {
            // This is implicit, only return the children
            return o.children;
        }
        return o;
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

    addNodes(og, to_node = undefined) {
        let transthis = this;

        if(to_node === undefined) {
            to_node = this._current_root;
        }

        
        
        //Legacy behavior: Add $<num> to transformation names to disambiguate
        // #TODO: Maybe change only local names? (requires server adjustment)
        let conflict_resolution = (node, all_labels) => {
            let count = all_labels.filter(x => x.startsWith(node.label())).length;

            if(count > 0) {
                node._label += "$" + count;
            }
            return node;
        };

        let cont = og.matching_opts;
        if(cont === undefined) {
            cont = og;
        }

        // Group homonyms together
        let grouped = {};
        let i = 0;
        for(let x of cont) {
            if(grouped[x.opt_name] === undefined) {
                grouped[x.opt_name] = [];
            }
            grouped[x.opt_name].push([i, x]);
            ++i;
        }
        // Now group homonyms to the first index
        let ordered = {};
        for(let x of Object.keys(grouped)) {
            let v = grouped[x];

            let index = v[0][0];
            ordered[index] = v.map(x => x[1]);
        }
        let ordered_list = [];
        let sorted_keys = Object.keys(ordered).sort();
        for(let x of sorted_keys) {
            let v = ordered[x];
            ordered_list.push(v);
        }

        let adder_func = (target_node, transthis, x) => {
            let new_node = target_node.addNode(x.opt_name, { 'affects': x.affects, 'props': x.opt_params}, {LabelConflictGlobal: conflict_resolution});
            new_node.setHandler("activate", function(elem, level) {

                if(level == -1) {
                    // Stopped hovering - clear highlights
                    transthis.sendClearHighlightRequest();
                }
                if(level == 0) {
                    // Hover - if available, highlight affected elements
                    let affects = elem.data().affects;
                    if(affects != undefined && affects.length > 0) {
                        transthis.sendHighlightRequest(affects);
                    }
                }
                if(level == 1) {
                    // Click
                    transthis.renderProperties(elem);
                }
                else if(level == 2) {
                    // If a doubleclick happened, ask DIODE to manage further steps
                    transthis._current_root = elem;
                    let obj = {};
                    obj[transthis.getState()['for_sdfg']] = new_node.path();

                    transthis.diode.optimize(transthis, obj);
                }
            });

            // Children may be provided already - recurse
            this.addNodes(x.children, new_node);
        };

        for(let z of ordered_list) {
            let x = null;
            if(z instanceof Array) {
                if(z.length == 1) {
                    x = z[0];
                }
                else {
                    // (Locally) conflicting names, group into own subtree
                    let subtree_node = to_node.addNode(z[0].opt_name + "...", undefined);
                    subtree_node.setPathLabel(" <virtual>");

                    for(let y of z) {
                        adder_func(subtree_node, transthis, y);
                    }
                    console.assert(subtree_node.pathLabel() === ' <virtual>', "Node must be marked as virtual");
                    continue;
                }
            }
            else {
                x = z;
            }
            adder_func(to_node, transthis, x);
        }
    }

    openPath(path) {
        /*
            Expands the currently active opttree path.
        */
        let current = this._tree_view._tree;
        current.representative().parentNode.classList.remove('collapsed_sublist');
        for(let x of path) {
            for(let c of current.children()) {
                if(c.label() == x) {
                    c.representative().parentNode.classList.remove('collapsed_sublist');
                    current = c;
                    break;
                }
            }
        }
        this._current_root = current;
    }

    createOptGraph() {
        let rootnode = new ValueTreeNode("Unoptimized");
        this._current_root = rootnode;
        let transthis = this;


        let base_element = $(this.container.getElement());
        let treediv = $('<div class="treediv"></div>');
        let optpropdiv = $('<div class="optpropdiv"></div>');
        base_element.append(treediv);
        base_element.append(optpropdiv);

        rootnode.setHandler("activate", function(elem, level) {

            if(level == 2) {
                // If a doubleclick happened, reset the entire path
                transthis._current_root = elem;

                let obj = {};
                obj[transthis.getState()['for_sdfg']] = []
                transthis.diode.optimize(transthis, obj);
            }
        });
        

        let og = JSON.parse(this.getState().optgraph_data);

        this.addNodes(og);

        this._tree_view = new TreeView(rootnode);
        this._tree_view.setDebouncing(this.diode);
        this._tree_view.create_html_in(treediv);

        let expandpath = og.extend_path;
        if(expandpath != undefined) {
            this.openPath(expandpath);
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
        this.on(this._project.eventString('-req-input_code'), function(msg) {
            // Echo with data
            eh.emit(transthis._project.eventString('input_code'),transthis.editor.getValue());
            transthis.editor.clearSelection();
        });

        this.on(this.project().eventString('-req-new_error'), msg => {
            // Echo with data
            eh.emit(transthis._project.eventString('new_error'), 'ok');
            this.highlight_error(msg);
        });
        
        this.on(this.project().eventString('-req-highlight-code'), msg => {
            eh.emit(transthis._project.eventString('highlight-code'), 'ok');
            this.highlight_code(msg);
        });
        
        this.on(this.project().eventString('-req-set-inputcode'), msg => {
            eh.emit(transthis._project.eventString('set-inputcode'), 'ok');
            
            this.editor.setValue(msg);
            this.editor.clearSelection();
        });
        
        
    }

    highlight_code(dbg_info) {
        let s_c  = dbg_info.start_col;
        let e_c = dbg_info.end_col;
        if(e_c <= s_c) {
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
        this.editor.scrollToLine(dbg_info.start_line, true, true, function () {});
        this.editor.gotoLine(dbg_info.start_line, 10, true);

        setTimeout(() => {
            this.editor.getSession().removeMarker(marker);
        }, 5000);
    }

    highlight_error(error) {

        if(error.type == "SyntaxError") {
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
        }
        else {
            console.log("Untreated error type", error);
        }
    }

    terminal_identifier() {
        return this._terminal_identifer;
    }

    compile(code) {
        for(let m of this._markers) {
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
            title: "Compilation terminal",
            type: 'component',
            componentName: 'TerminalComponent',
            componentState: { created: millis }
        };
        this.diode.addContentItem(terminal_config);

        console.log("Server emitting to ", terminal_identifier);

        this._terminal_identifer = terminal_identifier;

        this.diode.gatherProjectElementsAndCompile(this, { 'code': code}, { run: true, term_id: terminal_identifier });
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

            if(msg.sdfg_name != this.getState()['sdfg_name']) {
                // Name mismatch; ignore
                return;
            }
            eh.emit(transthis._project.eventString('new-codeout'), 'ok');
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
        if(typeof extracted === "string") {
            extracted = JSON.parse(extracted);
        }

        if(typeof extracted.generated_code == "string") {
            this.editor.setValue(this.cleanCode(extracted.generated_code));
            this.editor.clearSelection();
        }
        else {
            // Probably an array type
            this.editor.setValue("");
            this.editor.clearSelection();
            for(let c of extracted.generated_code) {
                let v = c;
                if(extracted.generated_code.length > 1) {
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
        this.saveToState({ 'code': input });
    }

    setEditorReference(editor) {
        this.editor = editor;

        let elem = this.container.getElement()[0];
        elem.addEventListener('resize', x => {
            this.editor.resize();
        });
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

        this.container.extendState({
             "current_value": session.getValue()
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
        header.innerText = "DIODE2";
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
            //#TODO: Add "New" item
            startpage_recent.appendChild(this.createStartpageListElement("Create a new Project", null, null, plus, x => {
                this.container.close();

                // Force creation of a new "project" instance (since we are explicitly creating a new project, not a file)
                this.diode.createNewProject();
                //this.diode.newFile(); // We could do this, but it's not necessary - let the user create/load at his own discretion


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
            for(let p of projects) {
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

        startpage_resources.appendChild(this.createStartpageListElement("Visit DaCe on GitHub", null, null, "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png", x => {
            window.open("https://github.com/spcl/dace", "_blank");
        }));
        startpage_resources.appendChild(this.createStartpageListElement("Visit project page", null, null, dace_logo, x => {
            window.open("https://spcl.inf.ethz.ch/Research/DAPP/", "_blank");
        }));


        startpage_container.appendChild(startpage_recent);
        startpage_container.appendChild(startpage_resources);
     
        parent.appendChild(startpage_container);
    }

    createStartpageListElement(name, time, info, image=undefined, onclick = x => x) {

        let diode_image = `<svg width="50" height="50" version="1.1" viewBox="0 0 13.229 13.229" xmlns="http://www.w3.org/2000/svg"><g transform="translate(0 -283.77)" fill="none" stroke="#000" stroke-linecap="round" stroke-width=".68792"><path d="m3.3994 287.29v6.9099l6.5603-3.7876-6.5644-3.7899z" stroke-linejoin="round"/><g><path d="m3.3191 290.39h-2.6127"/><path d="m12.624 290.41h-2.6647v-3.3585"/><path d="m9.9597 290.41v2.9962"/></g></g></svg>`
        if(image == undefined) {
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
                if(info != null)
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

                if(proj_date != null) {
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


class DIODE_Context_DIODE2Settings extends DIODE_Context {
    constructor(diode, gl_container, state) {
        super(diode, gl_container, state);
        
        this._settings_container = null;

        this._editor_themes = this.getThemes();
    }

    getThemes() {
        REST_request('/dace/api/v1.0/diode2/themes', undefined, xhr => {
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
            let label = FormBuilder.createLabel(undefined, "DaCe Debug mode", "When true, the program shows elements primarily useful for debugging and developing DaCe/DIODE2.");
            
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
        element.on('click', function() {
            if(toToggle.hasClass(toggleclassname)) {
                toToggle.removeClass(toggleclassname)
            }
            else {
                toToggle.addClass(toggleclassname)
            }
        });
    }

    parse_settings(settings, parent=undefined, path = []) {
        if(parent === undefined) {
            parent = $("#diode_settings_container");
            //parent.parent().parent().css("overflow-y", "auto");
        }

        let categories = [];
        let settings_lookup = {};

        Object.entries(settings).forEach(
            ([key, value]) => {
                let meta = value.meta;
                let val = value.value;

                let setting_path = path.concat(key);

                if(meta.type == "dict") {
                    if(parent.attr('id') == 'diode_settings_container') {
                        // Top level
                        // Category, add an h2 to the settings window and register in categories
                        categories.push(key);
                        let cat_header = document.createElement("h2");
                        cat_header.id = "category_header_" + key;
                        cat_header.classList = "togglable";
                        cat_header.innerHTML = meta.title;
                        cat_header.title = meta.description;

                        parent.append(cat_header);

                        let cat_container = document.createElement("div");
                        cat_container.id = "sub_header_" + key + "_container";
                        cat_container.classList = "settings_container level_0 collapsed_container"
                        $(cat_header).after(cat_container);

                        this.link_togglable_onclick($(cat_header), $(cat_container));

                        // Recurse for sub-elements
                        let tmp = this.parse_settings(val, $(cat_container), path.concat([key]));
                        settings_lookup = {...settings_lookup, ...tmp};
                    }
                    else if(parent.attr('id').startsWith("category_header")) {
                        // One level under categories
                        let sub_header = document.createElement("h3");
                        sub_header.id = "sub_header_" + key;
                        sub_header.innerHTML = meta.title;
                        sub_header.title = meta.description;
                        sub_header.classList = "togglable";

                        parent.append(sub_header);

                        let sub_container = document.createElement("div");
                        sub_container.id = "sub_header_" + key + "_container";
                        sub_container.classList = "settings_container level_1 collapsed_container"
                        $(sub_header).after(sub_container);

                        this.link_togglable_onclick($(sub_header), $(sub_container));

                        // Recurse for sub-elements
                        let tmp = this.parse_settings(val, $(sub_container), path.concat([key]));                    
                        settings_lookup = {...settings_lookup, ...tmp};
                    }
                    else if(parent.attr('id').startsWith("sub_header")) {
                        // One level under categories
                        let subsub_header = document.createElement("h4");
                        subsub_header.id = "subsub_header_" + key;
                        subsub_header.innerHTML = meta.title;
                        subsub_header.title = meta.description;
                        subsub_header.classList = "togglable";

                        parent.append(subsub_header);

                        let subsub_container = document.createElement("div");
                        subsub_container.id = "sub_header_" + key + "_container";
                        subsub_container.classList = "settings_container level_1 collapsed_container"
                        $(subsub_header).after(subsub_container);

                        this.link_togglable_onclick($(subsub_header), $(subsub_container));

                        // Recurse for sub-elements
                        let tmp = this.parse_settings(val, $(subsub_container), path.concat([key]));
                        settings_lookup = {...settings_lookup, ...tmp};
                    }
                }
                else if(meta.type == "bool") {

                    let idstr = "bool_switch_" + key;
                    let label = FormBuilder.createLabel(idstr + "_label", meta.title, meta.description);
                    let tswitch = FormBuilder.createToggleSwitch(idstr, elem => { 

                        this.settings_change_callback("bool", setting_path, elem.checked);
                    }, val);

                    let container = FormBuilder.createContainer("");
                    container.append(label);
                    container.append(tswitch);
                    //container.append("<br />");
                    parent.append(container);

                    settings_lookup[setting_path] = val;
                }
                else if(meta.type == "str") {
                    let idstr = "string_field_" + key;
                    let label = FormBuilder.createLabel(idstr + "_label", meta.title, meta.description);
                    let text_input = FormBuilder.createTextInput(idstr, elem => {
                        this.settings_change_callback("str", setting_path, elem.value);
                    });
                    text_input.val(val);

                    let container = FormBuilder.createContainer("");
                    container.append(label);
                    container.append(text_input);
                    //container.append("<br />");
                    parent.append(container);

                    settings_lookup[setting_path] = val;
                }
                else if(meta.type == "int") {
                    let idstr = "int_field_" + key;
                    let label = FormBuilder.createLabel(idstr + "_label", meta.title, meta.description);
                    let int_input = FormBuilder.createIntInput(idstr, elem => {
                        this.settings_change_callback("int", setting_path, elem.value);
                    });
                    int_input.val(val);

                    let container = FormBuilder.createContainer("");
                    container.append(label);
                    container.append(int_input);
                    //container.append("<br />");
                    parent.append(container);

                    settings_lookup[setting_path] = val;
                }
                else if(meta.type == "float") {
                    let idstr = "float_field_" + key;
                    let label = FormBuilder.createLabel(idstr + "_label", meta.title, meta.description);
                    let float_input = FormBuilder.createFloatInput(idstr, elem => { 
                        this.settings_change_callback("float", setting_path, elem.value);
                    });
                    float_input.val(val);

                    let container = FormBuilder.createContainer("");
                    container.append(label);
                    container.append(float_input);
                    //container.append("<br />");
                    parent.append(container);

                    settings_lookup[setting_path] = val;
                }
                else if(meta.type == "font") {
                    // Not sure what to do here yet
                }
                else {
                    alert("Unknown settings type: \"" + meta.type + "\"");
                }
            }
        );        


        return settings_lookup;
    }

    get_settings() {
        let post_params = {
            client_id: this.diode.getClientID()
        }
        REST_request("/dace/api/v1.0/preferences/get", post_params, (xhr) => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                let settings = this.parse_settings(JSON.parse(xhr.response));

                this.diode._settings = new DIODE_Settings(settings);
    
            }
        });
    }

    set_settings() {
        if(!this.diode.settings().hasChanged()) {
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
        let debounced = this.diode.debounce("settings-changed", function() {
            
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
    constructor(diode, project_id=undefined) {
        this._diode = diode;
        if(project_id === undefined || project_id === null) {
            this._project_id = diode.getPseudorandom();
        }
        else {
            this._project_id = project_id;
        }
        this.setup();
        this._callback = null;
        this._rcvbuf = {};
        this._waiter = {};

        this._listeners = {};

        this._closed_windows = [];
    }

    getTransformationSnapshots() {
        let sdata = sessionStorage.getItem("transformation_snapshots");
        if(sdata == null) {
            sdata = [];
        }
        else
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

    getConfigForClosedWindow(name, remove = true) {
        let list = this.getClosedWindowsList();
        let new_list = [];

        let rets = [];

        for(let x of list) {
            let cname = x[0];
            let state = x[1];

            if(state.created === name) {
                // Found the requested element

                rets.push([cname, state]);

                if(remove) {
                    // Don't put into the new list
                }
                else {
                    new_list.push([cname, state]);
                }
            }
            else {
                // Not found
                new_list.push([cname, state]);
            }
        }

        // Store back
        this._closed_windows = new_list;
        sessionStorage.setItem(this._project_id + "-closed-window-list", JSON.stringify(this._closed_windows));

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
        if(typeof(tmp) === "string") {
            tmp = JSON.parse(tmp);
        }

        if(tmp === null) {
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
        let cb = function(msg) {
            let tmp = transthis._rcvbuf[id][event];
            if(tmp instanceof Array) {
                transthis._rcvbuf[id][event].push(msg);
            }
            else if(tmp instanceof Object) {
                Object.assign(transthis._rcvbuf[id][event], msg);
            }
            else {
                transthis._rcvbuf[id][event] = msg;
            }
        };
        let params = [this.eventString(event), cb, this];
        hub.on(...params);

        this._listeners[id].push(params);
    }

    stopListening(id) {
        let hub = this._diode.goldenlayout.eventHub;
        for(let x of this._listeners[id]) {
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
        if(pdata == null)
            throw "Project must exist";

        return JSON.parse(pdata);
    }

    static getSavedProjects() {
        let tmp = localStorage.getItem("saved_projects");
        if(tmp == null)
            return [];
        return JSON.parse(tmp);
    }

    save() {
        /*
            Saves all elements of this project to its own slot in the local storage
            (such that it can be opened again even if the window was closed).
            
        */

        let snapshots = sessionStorage.getItem("transformation_snapshots");
        if(typeof(snapshots) == 'string')
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
        if(sp == null) {
            sp = [];
        }
        else {
            sp = JSON.parse(sp);
        }

        sp = [save_name, ...sp];
        window.localStorage.setItem("saved_projects", JSON.stringify(sp));

            

    }

    request(list, callback, options={}) {
        /*
            options:
                timeout: Number                 ms to wait until on_timeout is called
                on_timeout: [opt] Function      Function called on timeout
                params: [opt] object            Parameters to pass with the request
        */
       let tmp = new DIODE_Project(this._diode, this._project_id);
       return tmp.__impl_request(list, callback, options);
    }

    __impl_request(list, callback, options={}) {
        /*
            options:
                timeout: Number                 ms to wait until on_timeout is called
                on_timeout: [opt] Function      Function called on timeout
                params: [opt] object            Parameters to pass with the request
        */
        this._callback = callback;
        let params = options.params;
        let reqid = "id"+this._diode.getPseudorandom();
        // Clear potentially stale values
        this._rcvbuf[reqid] = {};
        this._listeners[reqid] = [];
        for(let x of list) {
            this.startListening(x, reqid);
            this._diode.goldenlayout.eventHub.emit(this.eventString("-req-" + x), params, this);
        }

        let transthis = this;
        const interval_step = 100;
        let timeout = options.timeout;
        
        this._waiter[reqid] = setInterval(() => {
            let missing = false;

            for(let x of list) {
                if(!(x in transthis._rcvbuf[reqid])) {
                    missing = true;
                    break;
                }
            }
            if(!missing) {
                clearInterval(transthis._waiter[reqid]);
                transthis.stopListening(reqid);
                transthis._waiter[reqid] = null;
                let tmp = transthis._rcvbuf[reqid];
                delete transthis._rcvbuf[reqid];
                return transthis._callback(tmp, options.timeout_id);
            }
            else {
                timeout -= interval_step;
                if(timeout <= 0) {
                    // Timed out - fail silently
                    clearInterval(transthis._waiter[reqid]);
                    transthis.stopListening(reqid);
                    if(options.on_timeout != undefined) {
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
            eh.emit(this.project().eventString("display-properties"), 'ok');
            this.getHTMLContainer().innerHTML = "";
            this.diode.renderProperties2(msg.transthis, msg.node, msg.params, this.getHTMLContainer(), msg.options);
        });

        this.on(this.project().eventString('-req-render-free-vars'), msg => {
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

    renderDataSymbols(calling_context, data) {
        // #TODO: This creates the default state (as in same as render_free_symbols() in the old DIODE)
        
        let free_symbol_table = new DiodeTables.Table();
        free_symbol_table.setHeaders("Symbol", "Type", "Dimensions", "Controls");

        // Go over the undefined symbols first, then over the arrays (SDFG::arrays)
        let all_symbols = [...Object.entries(data.undefined_symbols), "SwitchToArrays", ...Object.entries(data.attributes._arrays)];
        
        let caller_id = calling_context;
        console.assert(caller_id != undefined && typeof(caller_id) == 'string');

        for(let x of all_symbols) {

            if(x == "SwitchToArrays") {
                // Add a delimiter
                let col = free_symbol_table.addRow("Arrays");
                col.childNodes.forEach(x => {
                    x.colSpan = 4;
                    x.style = "text-align:center;";
                });
                continue;
            }
            if(x[0] == "null" || x[1] == null) {
                continue;
            }
            let edit_but = document.createElement('button');
            edit_but.addEventListener('click', _x => {
                this.renderDataSymbolProperties(caller_id, x);
            });
            edit_but.innerText = "Edit";
            let del_but = document.createElement('button');
            del_but.addEventListener('click', _x => alert("del_but clicked"));
            del_but.innerText = "Delete";
            let but_container = document.createElement('div');
            but_container.appendChild(edit_but);
            but_container.appendChild(del_but);
            free_symbol_table.addRow(x[0], x[1].type, x[1].attributes.dtype + "[" + x[1].attributes.shape + "]", but_container);
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
        let state = this.getState();
        if(state.params != undefined) {
            let p = state.params;
            this.diode.renderProperties2(p.transthis, p.node, p.params, this.getHTMLContainer());
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

        if(typeof(data) == 'string') {
            data = JSON.parse(data);
        }

        if(data.elements == undefined) {
            data.elements = [];
        }
        let base_element = $(this.container.getElement())[0];
        base_element.innerHTML = "";
        let container = document.createElement("div");
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
        for(let x of data.elements) {

            let optparse = x.options;
            if(typeof(optparse) == 'string') {

            }
            else {
                if(optparse.type == undefined) {
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
                c.innerText = y;
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
        this._client_id = localStorage.getItem("diode2_client_id");
        if(this._client_id == null) {
            this._client_id = this.getPseudorandom();
            localStorage.setItem("diode2_client_id", this._client_id);
        }

        // Initialize appearance
        this._appearance = new Appearance(localStorage.getItem("DIODE2/Appearance"));
        this._appearance.setOnChange(x => {
            localStorage.setItem("DIODE2/Appearance", JSON.stringify(x.toStorable()))
        });
    }

    setupEvents() {
        this.goldenlayout.eventHub.on(this.project().eventString('-req-show_stale_data_button'), x => {
            this.__impl_showStaleDataButton();
        });
        this.goldenlayout.eventHub.on(this.project().eventString('-req-remove_stale_data_button'), x => {
            this.__impl_removeStaleDataButton();
        });

        // Install the hint mechanic on the whole window
        window.addEventListener('contextmenu', ev => {
            console.log("contextmenu requested on", ev.target);
        });
    }

    openUploader(purpose="") {
        
        w2popup.open({
            title: purpose == "pickle-sdfg" ? "Upload the pickled SDFG" : "Upload a code file",
            body: `
<div class="w2ui-centered upload_flexbox">
    <label for="file-select" style="flex-grow: 1">
        <div class="diode_uploader" id='upload_box'>
            <div class="uploader_text">
                Drop file here or click to select a file
            </div>
        </div>
    </label>
    <input id="file-select" type="file" style="position:absolute;"/>
</div>
`,
            buttons: '',
            showMax: true
        });
        let x = $('#upload_box');
        if(x.length == 0) {
            throw "Error: Element not available";
        }
        x = x[0];

        let file_handler = (data) => {
            if(purpose == "code-python") {
                this.newFile(data);
            }
            else if(purpose == "pickle-sdfg"){
                let b64_data = btoa(String.fromCharCode(...new Uint8Array(data)));
                this.load_from_binary_sdfg(b64_data);
            }
        };

        setup_drag_n_drop(x, (mime, data) => {
            console.log("upload mime", mime);

            file_handler(data);

            // Close the popup
            w2popup.close();
        }, null, {
            readMode: purpose == "pickle-sdfg" ? "binary" : "text"
        });

        let fuploader = $('#file-select');
        if(fuploader.length == 0) {
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
            if(purpose == "pickle-sdfg") {
                reader.readAsBinaryString(file);
            }
            else {
                reader.readAsText(file);
            }
        });
    }

    load_from_binary_sdfg(sdfg_data) {
        let post_params = {
            binary: sdfg_data
        };
        REST_request("/dace/api/v1.0/decompile/SDFG/", post_params, xhr => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                // The reponse is similar to the one of compile()
                let resp = xhr.response;
                let respdata = JSON.parse(resp);
                let compounds = respdata['compounds'];

                let input_code = "";

                for(let x of Object.entries(compounds)) {
                    let t = x[1].input_code;
                    if(t != undefined) {
                        input_code = t;
                    }
                }

                console.log("input code: ", input_code);
                this.project().request(['set-inputcode'], x => {
                    
                }, {
                    params: input_code
                });
                this.multiple_SDFGs_available(resp);
                this.OptGraphs_available(compounds);
            }
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


    addContentItem(config) {
        let root = this.goldenlayout.root;
        if(root.contentItems.length === 0) {
            // Layout is completely missing, need to add one (row in this case)
            let layout_config = {
                type: 'row',
                content: []
            };
            root.addChild(layout_config);

            // retry with recursion
            this.addContentItem(config);
        }
        else {
            // #TODO: To get the old DIODE layout, this needs to be smarter
            if(this.goldenlayout.isSubWindow) {
                // Subwindows don't usually have layouts, so send a request that only the main window should answer
                this.goldenlayout.eventHub.emit('create-window-in-main', JSON.stringify(config));
            }
            else {
                root.contentItems[0].addChild(config);
            }
        }
    }

    newFile(content="") {
        let millis = this.getPseudorandom();

        let config = {
            title: "CodeIn",
            type: 'component',
            componentName: 'CodeInComponent',
            componentState: { created: millis, code_content: content }
        };

        this.addContentItem(config);
    }

    open_diode2_settings() {
        let millis = this.getPseudorandom();

        let config = {
            title: "DIODE2 settings",
            type: 'component',
            componentName: 'DIODE2SettingsComponent',
            componentState: { created: millis }
        };

        this.addContentItem(config);
    }

    open_diode_settings() {
        let millis = this.getPseudorandom();

        let config = {
            title: "DIODE settings",
            type: 'component',
            componentName: 'SettingsComponent',
            componentState: { created: millis }
        };

        this.addContentItem(config);
    }

    open_runqueue() {
        let millis = this.getPseudorandom();

        let config = {
            title: "Runqueue",
            type: 'component',
            componentName: 'RunqueueComponent',
            componentState: { created: millis }
        };

        this.addContentItem(config);
    }

    getEnum(name) {
        let cached = localStorage.getItem('Enumeration:' + name);
        if(cached == null || cached == undefined) {
            // Request the enumeration from the server

            REST_request("/dace/api/v1.0/getEnum/" + name, undefined, xhr => {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    console.log(name, xhr.response);
                    localStorage.setItem('Enumeration:' + name, xhr.response);
                }
            }, 'GET');

            return null;
        }

        return JSON.parse(cached)['enum'];
    }

    renderProperties2(transthis, node, params, parent, options=undefined) {
        /*
            Creates property visualizations like Visual Studio
        */

        if(!Array.isArray(params)) {
            // Format is different (diode to_json with seperate meta / value - fix before passing to renderer)
            let params_keys = Object.keys(params).filter(x => !x.startsWith('_meta_'));
            params_keys = params_keys.filter(x => Object.keys(params).includes('_meta_' + x));

            let _lst = params_keys.map(x => {
                let mx = JSON.parse(JSON.stringify(params['_meta_' + x]));

                mx.name = x;
                mx.value = params[x];

                return mx;
            });

            params = _lst;
        }

        if(typeof(transthis) == 'string') {
            // Event-based
            let target_name = transthis;
            transthis = {
                propertyChanged: (node, name, value) => {
                    this.project().request(['property-changed-' + target_name], x => {

                    }, {
                        timeout: 200,
                        params: {
                            node: node,
                            name: name,
                            value: value,
                            type: options ? options.type : options
                        }
                    })
                },
                applyTransformation: () => {
                    this.project().request(['apply-adv-transformation' + target_name], x => {

                    }, {
                        timeout: 200,
                        params: options == undefined ? undefined : options.apply_params
                    })
                },
                project: () => this.project()
            };
        }
        let dt = new DiodeTables.Table();
        let i = 0;
        let cur_dt = dt;

        let dtc = null;
        let categories = {};
        for(let x of params) {
            
            let cat = x.category;
            if(categories[cat] == undefined) {
                categories[cat] = [];
            }
            categories[cat].push(x);
        }
        if(!DIODE.debugDevMode()) {
            delete categories["(Debug)"]
        }
        for(let z of Object.entries(categories)) {
            
            // Sort within category
            let cat_name = z[0];
            let y = z[1].sort((a, b) => a.name.localeCompare(b.name));


            // Add Category header
            cur_dt = dt;
            let sp  = document.createElement('span');
            sp.innerText = cat_name;
            let tr = cur_dt.addRow(sp);
            tr.childNodes[0].colSpan = "2";

            dtc = new DiodeTables.TableCategory(cur_dt, tr);

            for(let x of y) {
            
                let value_part = diode.getMatchingInput(transthis, x, node);
                let cr = cur_dt.addRow(x.name, value_part);
                if(dtc != null) {
                    dtc.addContentRow(cr);
                }
            }
        }
        dt.setCSSClass("diode_property_table");

        if(options && options.type == "transformation") {
            // Append a title
            let title = document.createElement("span");
            title.classList = "";
            title.innerText = options.opt_name;
            parent.appendChild(title);
        }
        dt.createIn(parent);
        if(options && options.type == "transformation") {
            // Append an 'apply-transformation' button
            let button = document.createElement('button');
            button.innerText = "Apply advanced transformation";
            button.addEventListener('click', _x => {
                this.project().request(['apply-adv-transformation-' + options.sdfg_name], _y => {}, {
                    params: JSON.stringify(options.apply_params)
                })
            });
            parent.appendChild(button);

        }
    }

    renderProperties(transthis, node, params, parent) {
        /*
            Creates property visualizations like DIODE1
        */
        if(typeof(transthis) == 'string') {
            // Event-based
            let target_name = transthis;
            transthis = {
                propertyChanged: (node, name, value) => {
                    this.project().request(['property-changed-' + target_name], x => {

                    }, {
                        timeout: 200,
                        params: {
                            node: node,
                            name: name,
                            value: value
                        }
                    })
                },
                project: () => this.project()
            };
        }
        for(let x of params) {
            let cont = document.createElement("div");
            cont.classList.add("settings_key_value");
            let label = document.createElement('span');
            label.classList.add("title");
            label.innerText = x.name;
            label.title = x.desc;
            cont.append(label);
            $(cont).append(this.getMatchingInput(transthis, x, node));
            parent.append(cont);
        }
    }

    getMatchingInput(transthis, x, node) {

        let create_language_input = (value, onchange) => {
            if(value == undefined) {
                value = x.value;
            }
            if(onchange == undefined) {
                onchange = (elem) => {
                    transthis.propertyChanged(node, x.name, elem.value);
                };
            }
            let language_types = this.getEnum('Language');
            let qualified = value;
            if(!language_types.includes(qualified)) {
                qualified = "Language." + qualified;
            }
            let elem = FormBuilder.createSelectInput("prop_" + x.name, onchange, language_types, qualified);
            return elem;
        };

        let elem = document.createElement('div');
        if(x.type == "bool") {
            let val = x.value;
            if(typeof(val) == 'string') val = val == 'True';
            elem = FormBuilder.createToggleSwitch("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.checked);

            }, val);
        }
        else if(
            x.type == "str" || x.type == "tuple" || x.type == "dict" ||
            x.type == "list" || x.type == "set"
        ) {
            elem = FormBuilder.createTextInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, x.value);
        }
        else if(x.type == "Range") {
            // #TODO: How to visualize/edit this?
            elem = FormBuilder.createTextInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, JSON.loads(elem.value));
            }, JSON.stringify(x.value));
        }
        else if(x.type == "CodeProperty") {
            let codeelem = null;
            let langelem = null;
            let onchange = (elem) => {
                transthis.propertyChanged(node, x.name, {
                    'string_data': codeelem[0].value,
                    'language': langelem[0].value
                });
            };
            codeelem = FormBuilder.createTextInput("prop_" + x.name, onchange, x.value.string_data);
            elem.appendChild(codeelem[0]);
            langelem = create_language_input(x.value.language, onchange);
            elem.appendChild(langelem[0]);

            return elem;
        }
        else if(x.type == "int") {
            elem = FormBuilder.createIntInput("prop_" + x.name, (elem) => {
                this.propertyChanged(node, x.name, elem.value);
            }, x.value);
        }
        else if(x.type == 'ScheduleType') {
            let schedule_types = this.getEnum('ScheduleType');
            let qualified = x.value;
            if(!schedule_types.includes(qualified)) {
                qualified = "ScheduleType." + qualified;
            }
            elem = FormBuilder.createSelectInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, schedule_types, qualified);
        }
        else if(x.type == 'AccessType') {
            let access_types = this.getEnum('AccessType');
            let qualified = x.value;
            if(!access_types.includes(qualified)) {
                qualified = "AccessType." + qualified;
            }
            elem = FormBuilder.createSelectInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, access_types, qualified);
        }
        else if(x.type == 'Language') {
            elem = create_language_input();
        }
        else if(x.type == 'None') {
            // Not sure why the user would want to see this
            console.log("Property with type 'None' ignored", x);
            return elem;
        }
        else if(x.type == 'object' && x.name == 'identity') {
            // This is an internal property - ignore
            return elem;
        }
        else if(x.type == 'OrderedDiGraph') {
            // #TODO: What should we do with this?
            elem = FormBuilder.createTextInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, x.value);
        }
        else if(x.type == 'DebugInfo') {
            // Special case: The DebugInfo contains information where this element was defined
            // (in the original source).
            let info_obj = JSON.parse(x.value);
            elem = FormBuilder.createCodeReference("prop_" + x.name, (elem) => {
                // Clicked => highlight the corresponding code
                transthis.project().request(['highlight-code'], msg => {}, {
                    params: info_obj
                });
            }, info_obj);
        }
        else if(x.type == 'ListProperty') {
            // #TODO: Find a better type for this
            elem = FormBuilder.createTextInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, x.value);
        }
        else if(x.type == "StorageType") {
            let storage_types = this.getEnum('StorageType');
            let qualified = x.value;
            if(!storage_types.includes(qualified)) {
                qualified = "StorageType." + qualified;
            }
            elem = FormBuilder.createSelectInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, storage_types, qualified);
        }
        else if(x.type == "typeclass") {
            // #TODO: Find a better type for this
            elem = FormBuilder.createTextInput("prop_" + x.name, (elem) => {
                transthis.propertyChanged(node, x.name, elem.value);
            }, x.value);
        }
        else {
            console.log("Unimplemented property type: ", x);
            alert("Unimplemented property type: " + x.type);

            return elem;
        }
        return elem[0];
    }

    renderPropertiesInWindow(transthis, node, params, options) {
        let dobj = {
            transthis: typeof(transthis) == 'string' ? transthis : transthis.created,
            node: node,
            params: params,
            options: options
        };
        this.replaceOrCreate(['display-properties'], dobj,
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
        this.project().request(['show_stale_data_button'], x=>{}, {});
    }
    removeStaleDataButton() {
        this.project().request(['remove_stale_data_button'], x=>{}, {});
    }

    __impl_showStaleDataButton() {
        /*
            Show a hard-to-miss button hinting to recompile.
        */

        if(this._stale_data_button != null) {
            return;
        }
        let stale_data_button = document.createElement("div");
        stale_data_button.classList = "stale_data_button";
        stale_data_button.innerHTML = "Stale project data. Click here or press <span class='key_combo'>Alt-R</span> to synchronize";

        stale_data_button.addEventListener('click', x => {
            this.gatherProjectElementsAndCompile(diode, {}, { sdfg_over_code: true });
        })

        document.body.appendChild(stale_data_button);

        this._stale_data_button = stale_data_button;
    }

    __impl_removeStaleDataButton() {
        if(this._stale_data_button != null) {
            let p = this._stale_data_button.parentNode;
            p.removeChild(this._stale_data_button);
            this._stale_data_button = null;
        }
    }

    static filterComponentTree(base, filterfunc = x => x) {
        let ret = [];
        for(let x of base.contentItems) {

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
        for(let x of comps) {
            names.push(x.config.componentState.sdfg_name);
        }

        for(let n of names) {
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
        let opttrees = DIODE.filterComponentTreeByCname(this.goldenlayout.root, "OptGraphComponent");
        let sdfg_renderers = DIODE.filterComponentTreeByCname(this.goldenlayout.root, "SDFGComponent");
        let code_outs = DIODE.filterComponentTreeByCname(this.goldenlayout.root, "CodeOutComponent");
        let property_renderers = DIODE.filterComponentTreeByCname(this.goldenlayout.root, "PropWinComponent");

        // Note that this only collects the _open_ elements and disregards closed or invalidated ones
        // Base goldenlayout stretches everything to use the full space available, this makes stuff look bad in some constellations
        // We compensate some easily replacable components here
        if(property_renderers.length == 0) {
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
        let to_remove = [code_ins, code_outs, opttrees, sdfg_renderers, property_renderers];
        for(let y of to_remove) {
            for(let x of y) {
                if(x.componentName != undefined) {
                    x.remove();
                }
                // Otherwise: Might be a raw config
            }
        }

        // Remove existing content
        let c = [...this.goldenlayout.root.contentItems];
        for(let x of c) {
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

        code_ins.forEach(x => code_in_stack.addChild(x.config));

        // Everything has been added, but maybe too much: There might be empty stacks.
        // They should be removed to keep a "clean" appearance
        for(let x of [opttree_stack, code_in_stack, sdfg_stack, property_stack]) {
            if(x.contentItems.length == 0) {
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
        let sdfg_components = this.goldenlayout.root.getItemsByFilter( predicate );

        if(sdfg_components.length == 0) {
            console.warn("Cannot group, no elements found");
        }
        let new_container = this.goldenlayout.createContentItem({
            type: 'stack',
            contents: []
        });

        for(let x of sdfg_components) {
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
        if(this._shortcut_functions[keys[0]] === undefined) {
            this._shortcut_functions[keys[0]] = [c];
        }
        else {
            this._shortcut_functions[keys[0]].push(c);
        }
    }

    onKeyUp(ev) {
        if(ev.altKey == false && ev.key == 'Alt') {
            for(let cs of Object.values(this._shortcut_functions)) {
                for(let c of cs) {
                    c.state = 0;
                }
            }
        }
    }
    onKeyDown(ev) {
        for(let cs of Object.values(this._shortcut_functions)) {
            for(let c of cs) {
                if(ev.altKey == false) {
                    c.state = 0;
                    continue;
                }
                if(c.state > 0) {
                    // Currently in a combo-state
                    if(c.expect[c.state-1] == ev.key) {
                        c.state += 1;
                        if(c.state > c.expect.length) {
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
        if(cs === undefined) return;

        let i = 0;
        for(let c of cs) {
            if(c.alt == ev.altKey && c.ctrl == ev.ctrlKey) {

                if(c.expect.length > 0) {
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
        window.sessionStorage.setItem("diode_project", this._current_project._project_id);
        this.setupEvents();
    }

    getProject() {
        let proj_id = window.sessionStorage.getItem("diode_project");
        this._current_project = new DIODE_Project(this, proj_id);
        if(proj_id == null || proj_id == undefined) {
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
        //let millis = date.getTime() * 10000 + Math.random() * 10000;
        let millis = date.getTime().toString() + Math.random().toFixed(10).toString() + this._creation_counter.toString();

        ++this._creation_counter;

        console.assert(millis !== undefined, "Millis well-defined");

        return millis;
    }
    
    multiple_SDFGs_available(sdfgs) {

        let sdfgs_obj = (typeof(sdfgs) == 'string') ? JSON.parse(sdfgs) : sdfgs;

        for(let x of Object.keys(sdfgs_obj.compounds)) {
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
                componentState: { created: millis(), sdfg_data: sdfg, sdfg_name: name }
            };
            this.addContentItem(new_sdfg_config);
        };
        this.replaceOrCreate(['new-sdfg'], JSON.stringify(sdfg), create_sdfg_func);

        let create_codeout_func = () => {
            let new_codeout_config = {
                title: "Output code for `" + name + "`",
                type: 'component',
                componentName: 'CodeOutComponent',
                componentState: { created: millis(), code: sdfg, sdfg_name: name }
            };
            this.addContentItem(new_codeout_config);
        }
        if(sdfg.generated_code != undefined) {
            this.replaceOrCreate(['new-codeout'], sdfg, create_codeout_func);
        }
    }

    OptGraph_available(optgraph, name="") {
        
        if(typeof optgraph != "string") {
            optgraph = JSON.stringify(optgraph);
        }

        // To provide some distinction, milliseconds since epoch are used.
        let millis = this.getPseudorandom();

        let create_optgraph_func = () => {
            let new_optgraph_config = {
                title: name == "" ? "OptGraph" : "OptGraph for `" + name + "`",
                type: 'component',
                //componentName: 'OptGraphComponent',
                componentName: 'AvailableTransformationsComponent',
                componentState: { created: millis, for_sdfg: name, optgraph_data: optgraph }
            };
            this.addContentItem(new_optgraph_config);
        };
        this.replaceOrCreate(['new-optgraph-' + name], optgraph, create_optgraph_func);
    }

    OptGraphs_available(optgraph) {
        let o = optgraph;
        if(typeof(o) == "string") {
            o = JSON.parse(optgraph);
        }
        
        for(let x of Object.keys(o)) {
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
        if(code === undefined) {
            if(options.sdfg_over_code) {
                reqlist.push('sdfg_object');
            }
            reqlist.push('input_code');
        }

        if(optpath === undefined) reqlist.push('optpath');
        /*if(optpath != undefined) {
            optpath = undefined;
            reqlist.push('optpath');
        }*/
        

        let on_collected = (values) => {
            if(code != undefined) values['input_code'] = code;
            if(sdfg_props != undefined) values['sdfg_props'] = sdfg_props;
            if(optpath != undefined) values['optpath'] = optpath;

            if(options.collect_cb != undefined) 
                options.collect_cb(values);

            if(options.dry_run === true)
                return;

            if(options.run === true) {
                let runopts = {};
                if(options['perfmodes']) {
                    runopts['perfmodes'] = options['perfmodes'];
                }
                this.compile_and_run(calling_context, options.term_id, values['input_code'], values['optpath'], values['sdfg_props'], runopts);
            }
            else {
                let cb = (resp) => {
                    this.replaceOrCreate(['extend-optgraph'], resp, (_) => { this.OptGraphs_available(resp);});
                };
                if(options['no_optgraph'] === true) {
                    cb = undefined;
                }
                let cis = values['sdfg_object'] != undefined;
                let cval = values['input_code'];

                if(cis) {
                    cval = values['sdfg_object'];
                    cval = JSON.parse(cval);
                }
                this.compile(calling_context, cval, values['optpath'], values['sdfg_props'],
                {
                    optpath_cb: cb,
                    code_is_sdfg: cis,
                });

            }
        }

        calling_context.project().request(reqlist, on_collected, { timeout: 500, on_timeout: on_collected });
    }

    compile(calling_context, code, optpath = undefined, sdfg_node_properties = undefined, options = {}) {
        /*
            options:
                .code_is_sdfg: If true, the code parameter is treated as a serialized SDFG
        */
        let post_params = {};
        if(options.code_is_sdfg === true) {
            post_params = { "sdfg": code };
        }
        else {
            post_params = { "code": code };
        }

        if(optpath != undefined) {
            post_params['optpath'] = optpath;
        }
        if(sdfg_node_properties != undefined) {
            post_params['sdfg_props'] = sdfg_node_properties;
        }
        post_params['client_id'] = this.getClientID();
        let version_string = "1.0";
        REST_request("/dace/api/v" + version_string + "/compile/dace", post_params, (xhr) => {
            if (xhr.readyState === 4 && xhr.status === 200) {

                let peek = JSON.parse(xhr.response);
                if(peek['error'] != undefined) {
                    // There was at least one error - handle all of them
                    this.handleErrors(calling_context, peek);
                }
                else {
                    // Data is no longer stale
                    this.removeStaleDataButton();

                    let o = JSON.parse(xhr.response);
                    this.multiple_SDFGs_available(xhr.response);
                    if(options.optpath_cb === undefined) {
                        this.OptGraphs_available(o['compounds']);
                    }
                    else {
                        options.optpath_cb(o['compounds']);
                    }
                }
            }
        });
    }

    handleErrors(calling_context, object) {
        let errors = object['error'];

        for(let error of errors) {

            if(error.type === "SyntaxError") {
                // This error is most likely to be caused exclusively by input code
                
                calling_context.project().request(['new_error'], msg => {},
                {
                    params: error,
                    timeout: 100,
                });
            }
            else {
                console.error("Error: ", error);
                alert(JSON.stringify(error));
            }
        }
    }

    ui_compile_and_run(calling_context) {

        let millis = this.getPseudorandom();

        let terminal_identifier = "terminal_" + millis;

        // create a new terminal
        let terminal_config = {
            title: "Compilation terminal",
            type: 'component',
            componentName: 'TerminalComponent',
            componentState: { created: millis }
        };
        this.addContentItem(terminal_config);


        this.gatherProjectElementsAndCompile(this, {}, { run: true, term_id: terminal_identifier });
    }

    load_perfdata() {
        let client_id = this.getClientID();


        let post_params = {
            client_id: client_id
        };
        REST_request("/dace/api/v1.0/perfdata/get/", post_params, (xhr) => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                let pd = JSON.parse(xhr.response);
                console.log("Got result", pd);

                this.project().request(['draw-perfinfo'], x => {
                    
                }, {
                    params: pd
                });
            }
        });
    }

    compile_and_run(calling_context, terminal_identifier, code, optpath = undefined, sdfg_node_properties = undefined, options={}) {
        let post_params = { "code": code };
        if(optpath != undefined) {
            post_params['optpath'] = optpath;
        }
        if(sdfg_node_properties != undefined) {
            post_params['sdfg_props'] = sdfg_node_properties;
        }
        let client_id = this.getClientID();
        post_params['client_id'] = client_id;
        post_params['perfmodes'] = ["default", "vectorize", "memop", "cacheop"];
        post_params['corecounts'] = [1,2,3,4];
        let version_string = "1.0";
        REST_request("/dace/api/v" + version_string + "/run/", post_params, (xhr) => {
            if (xhr.readyState === 4 && xhr.status === 200) {

                let tmp = xhr.response;
                if(typeof(tmp) == 'string') tmp = JSON.parse(tmp);
                if(tmp['error']) {
                    // Normal, users should poll on a different channel now.
                    this.display_current_execution_status(calling_context, terminal_identifier, client_id);
                }
            }
        });
    }

    display_current_execution_status(calling_context, terminal_identifier, client_id, perf_mode=undefined) {
        let post_params = {};
        post_params['client_id'] = client_id;
        post_params['perf_mode'] = perf_mode;
        let version_string = "1.0";
        REST_request("/dace/api/v" + version_string + "/run/status/", post_params, (xhr) => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                // #TODO: Show success/error depending on the exit code

                this.toast("Execution ended", "The execution of the last run has ended", 'info');
            }
            if (xhr.readyState === 3) {
                let newdata = xhr.response.substr(xhr.seenBytes);
                this.goldenlayout.eventHub.emit(terminal_identifier, newdata);
                xhr.seenBytes = xhr.responseText.length;
            }
        });
    }

    get_pattern_matches(calling_context, code, optpath = undefined, sdfg_props = undefined, callback_override = undefined) {
        let post_params = { "code": [code] };
        if(optpath != undefined) {
            post_params['optpath'] = optpath;
        }
        if(sdfg_props != undefined) {
            post_params['sdfg_props'] = sdfg_props;
        }
        post_params['client_id'] = this.getClientID();
        let version_string = "1.0";
        REST_request("/dace/api/v" + version_string + "/match_optimizer_patterns/", post_params, (xhr) => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                let peek = JSON.parse(xhr.response);
                if(peek['error'] != undefined) {
                    // There was at least one error - handle all of them
                    this.handleErrors(calling_context, peek);
                }
                else {
                    if(callback_override != undefined) {
                        callback_override(xhr.response);
                    }
                    else {
                        this.OptGraphs_available(xhr.response);
                        
                    }
                }
            }
        });
    }

    toast(title, text, type='info', timeout=10000, icon=undefined, callback=undefined) {
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

        if(optpath === undefined) {
            optpath = [];
        }
        // #DISCUSS: Reusing the compile-command seems to be the best option here. Any objections?
        let transthis = this;

        let on_data_available = (code_data, sdfgprop_data, from_code) => {
            let code = null;
            if(from_code) {
                code = code_data;
            }
            else {
                code = code_data;
            }
            
            let props = null;
            if(sdfgprop_data != undefined)
                props = sdfgprop_data.sdfg_props;
            else
                props = undefined;

            let cb = function(resp) {
                transthis.replaceOrCreate(['extend-optgraph'], resp, function(_) {transthis.OptGraphs_available(resp);});
            };

            transthis.compile(calling_context, code, optpath, props, {
                optpath_cb: cb,
                code_is_sdfg: !from_code
            });

            // Now get the next level of optimizations as well
            // (i.e. apply the specified transformations and get the new pattern matches)
            // #DISCUSS: This again is not very efficient and could be done directly when compiling.
            
            //transthis.get_pattern_matches(calling_context, code, optpath, props, cb);
        }


        calling_context.project().request(['input_code', 'sdfg_object'], function(data) {

            let from_code = true;
            if(data['sdfg_object'] != undefined) {
                from_code = false;
                data = data['sdfg_object'];
                data = JSON.parse(data);
            }
            else {
                data = data['input_code'];
            }
            on_data_available(data, undefined, from_code);
            
        });
    }

    /*
        Tries to talk to a pre-existing element to replace the contents.
        If the addressed element does not respond within a given threshold,
        a new element created.
    */
    replaceOrCreate(replace_request, replace_params, recreate_func) {

        //#TODO: Find out if it is a bug that one cannot use clearTimeout(recreation_timeout) instead of clearTimeout(tid) when replaceOrCreate-Calls are nested
        //(the let/const-Variables should be scope-local)
        const recreation_timeout = setTimeout(function() {
            // This should be executed only if the replace request was not answered
            recreate_func(replace_params);
        }, 1000);
        this.getCurrentProject().request(replace_request, function(resp, tid) {
            clearTimeout(tid);
        }, {
            timeout: 500,
            params: replace_params,
            timeout_id: recreation_timeout
        });
    }
    
    /*
        This function is used for debouncing, i.e. holding a task for a small amount of time
        such that it can be replaced with a newer function call which would otherwise get queued.
    */
    debounce(group, func, timeout) {

        if(this._debouncing === undefined) {
            // The diode parent object was not created. The function cannot be debounced in this case.
            return func;
        }
        let transthis = this;
        let debounced = function () {
            if(transthis._debouncing[group] !== undefined) {
                clearTimeout(transthis._debouncing[group]);
            }
            transthis._debouncing[group] = setTimeout(func, timeout);
        };

        return debounced;
    }

    static editorTheme() {
        let theme = localStorage.getItem('diode2_ace_editor_theme');
        if(theme === null) {
            return "monokai";
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
        localStorage.setItem('diode2_ace_editor_theme', name);
    }

    static recompileOnPropertyChange() {
        // Set a tendency towards 'false' 
        return localStorage.getItem('diode2_recompile_on_prop_change') != "true";
    }

    static setRecompileOnPropertyChange(boolean_value) {
        if(boolean_value) {
            localStorage.setItem('diode2_recompile_on_prop_change', "true");
        }
        else {
            localStorage.setItem('diode2_recompile_on_prop_change', "false");
        }
    }

    static setDebugDevMode(boolean_value) {
        if(boolean_value) {
            localStorage.setItem('diode2_DebugDevMode', "true");
        }
        else {
            localStorage.setItem('diode2_DebugDevMode', "false");
        }
    }
    static debugDevMode() {
        /*
            The DebugDev mode determines if internal, not-crucial-for-user properties are shown.
        */
        let v = localStorage.getItem("diode2_DebugDevMode");
        return v === "true";
    }
}


class ContextMenu {
    /*
        Implementation for a custom context menu
    */
    constructor(html_content = null) {

        this._html_content = html_content;
        this._click_close_handlers = [];

        this._ = setTimeout(() => {
            this._click_close_handlers = [
                ['click', x => {
                    this.destroy();
                }],
                ['contextmenu', x => {
                    this.destroy();
                }]
                
            ];

            for(let x of this._click_close_handlers) {
                window.addEventListener(...x);
            }
        }, 30);

        this._options = [];
    }

    width() {
        return this._cmenu_elem.offsetWidth;
    }
    
    addOption(name, onselect, onhover=null) {
        this._options.push({
            name: name,
            func: onselect,
            onhover: onhover
        });
    }

    destroy() {
        // Clear everything

        // Remove the context menu

        document.body.removeChild(this._cmenu_elem);

        for(let x of this._click_close_handlers) {
            window.removeEventListener(...x);
        }
    }

    show(x, y) {
        /*
            Shows the context menu originating at point (x,y)
        */

        let cmenu_div = document.createElement('div');
        cmenu_div.id = "contextmenu";
        $(cmenu_div).css('left', x + "px");
        $(cmenu_div).css('top', y + "px");
        cmenu_div.classList.add("context_menu");
        

        if(this._html_content == null) {
            // Set default context menu
            
            for(let x of this._options) {

                let elem = document.createElement('div');
                elem.addEventListener('click', x.func);
                elem.classList.add("context_menu_option");

                elem.innerText = x.name;
                cmenu_div.appendChild(elem);
            }

        }
        else {
            cmenu_div.innerHTML = this._html_content;
        }

        this._cmenu_elem = cmenu_div;
        document.body.appendChild(cmenu_div);
    }
}

export {DIODE, DIODE_Context_SDFG, DIODE_Context_CodeIn, DIODE_Context_CodeOut, DIODE_Context_Settings, DIODE_Context_Terminal, DIODE_Context_OptGraph, DIODE_Context_DIODE2Settings,
    DIODE_Context_PropWindow, DIODE_Context, DIODE_Context_Runqueue, DIODE_Context_StartPage, DIODE_Context_TransformationHistory, DIODE_Context_AvailableTransformations}