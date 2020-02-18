var base_url = "//" + window.location.host;

window.base_url = base_url;

import {
    DIODE, DIODE_Context_Settings,
    DIODE_Context_DIODESettings,
    DIODE_Context_CodeIn,
    DIODE_Context_CodeOut,
    DIODE_Context_Terminal,
    DIODE_Context_SDFG,
    DIODE_Context_PropWindow,
    DIODE_Context_Runqueue,
    DIODE_Context_StartPage,
    DIODE_Context_TransformationHistory,
    DIODE_Context_AvailableTransformations,
    DIODE_Context_Error,
    DIODE_Context_RunConfig,
    DIODE_Context_PerfTimes,
    DIODE_Context_InstrumentationControl,
    DIODE_Context_Roofline
} from "./diode.js"

function find_object_cycles(obj) {
    let found = [];

    let detect = (x, path) => {
        if(typeof(x) == "string") {

        }
        else if(x instanceof Array) {
            let index = 0;
            for(let y of x) {
                detect(y, [...path, index])
                ++index;
            }
        }
        else if(x instanceof Object) {
            if(found.indexOf(x) !== -1) {
                // Cycle found
                throw ["Cycle", path, x];
            }
            found.push(x);
            for(let y of Object.keys(x)) {
                if(x.hasOwnProperty(y)) {
                    detect(x[y], [...path, y]);
                }
            }
        }
    }

    return detect(obj, []);
    
}

function setup_drag_n_drop(elem, callbackSingle, callbackMultiple, options={ readMode: "text", condition: (elem) => true }) {

    /*
        callbackSingle: (mimetype: string, content: string) => mixed
            Takes the file contents (text) and the mimetype.
        callbackMultiple: reserved
        options:
            .readmode: "text" or "binary"
            .condition: Function called with parameter "elem" determining if the current element should have the handler active
    */

    let drag_enter = (e) => {
        if(!options.condition(elem)) return;
        e.stopPropagation();
        e.preventDefault();
    };

    let drag_over = e => {
        if(!options.condition(elem)) return;
        e.stopPropagation();
        e.preventDefault();
    };

    let drag_drop = e => {
        if(!options.condition(elem)) return;
        let files = Array.from(e.dataTransfer.files);
        
        if(files.length === 1) {
            e.stopPropagation();
            e.preventDefault();

            // A single file was provided
            let file = files[0];

            let mime = file.type;

            let reader = new FileReader();
            reader.onload = ev => {
                callbackSingle(mime, ev.target.result);
            };
            if(options.readMode == "text") {
                reader.readAsText(file);
            }
            else if(options.readMode == "binary") {
                reader.readAsArrayBuffer(file);
            }
            else {
                throw "Unimplemented read mode " + options.readMode;
            }
            
        }
        else if(files.length > 1) {
            e.stopPropagation();
            e.preventDefault();
            
            // #TODO: Deferred 
            alert("Cannot handle more than 1 input file at this point");
            throw "Previous alert caused here";
        }
        else {
            alert("Can only drop files at this point - everything else is user-agent-specific!")
            throw "Previous alert caused here";
        }

    };
    
    elem.addEventListener("dragenter", drag_enter, false);
    elem.addEventListener("dragover", drag_over, false);
    elem.addEventListener("drop", drag_drop, false);
}

function REST_request(command, payload, callback, method="POST") {
    let xhr = new XMLHttpRequest();
    let url = base_url + command;
    xhr.open(method, url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = () => {
        callback(xhr);
    };
    xhr.onerror = (e) => {
        console.warn("Connection error", e);
        alert("Connection error");
    };
    if(payload != undefined) {
        let data = JSON.stringify(payload);
        xhr.send(data);
    }
    else {
        xhr.send();
    }
}

class FormBuilder {

    static createContainer(idstr) {
        let elem = document.createElement("div");
        elem.id = idstr;
        elem.classList = "settings_key_value";
        return $(elem);
    }

    static createHostInput(id, onchange, known_list = ['localhost'], initial="localhost") {
        let elem = document.createElement('input');
        elem.type = "list";
        elem.id = id;
        let dlist = document.getElementById("hosttype-dlist");
        if(!dlist) {
            dlist = document.createElement("datalist");
            dlist.id = "hosttype-dlist";
            document.body.appendChild(dlist);
        }
        $(elem).attr("list", "hosttype-dlist");
        dlist.innerHTML = "";
        for(let x of known_list) {
            dlist.innerHTML += '<option value="' + x + '">'
        }

        elem.value = initial;
        elem.onchange = () => {
            onchange(elem);
        };

        return $(elem);
    }

    static createComboboxInput(id, onchange, known_list, initial) {
        let elem = document.createElement('div');
        let inputelem = document.createElement('input');
        inputelem.type = "list";
        inputelem.id = id;
        inputelem.onfocus = () => {
            // Clear (this will make it act more like a select)
            let oldvalue = inputelem.value;
            inputelem.onblur = () => {
                inputelem.value = oldvalue;
            }
            inputelem.value = "";
            
        }
        let dlist = document.createElement("datalist");
        dlist.id = id + "-dlist";
        elem.appendChild(dlist);
        
        $(inputelem).attr("list", id + "-dlist");
        dlist.innerHTML = "";
        for(let x of known_list) {
            dlist.innerHTML += '<option value="' + x + '">'
        }

        inputelem.value = initial;
        inputelem.onchange = () => {
            inputelem.onblur = null;
            onchange(inputelem);
        };

        elem.appendChild(inputelem);

        return $(elem);
    }

    static createCodeReference(id, onclick, obj) {
        let elem = document.createElement('span');
        elem.id = id;
        elem.addEventListener('click', x => {
            onclick(x);
        });
        elem.classList.add("code_ref");

        if(obj == null) {
            elem.innerText = "N/A";
            elem.title = "The DebugInfo for this element is not defined";
        }
        else {

            let split = obj.filename.split("/");
            let fname = split[split.length - 1];

            elem.innerText = fname + ":" + obj.start_line;
            elem.title = obj.filename;
        }

        return $(elem);
    }

    static createLabel(id, labeltext, tooltip) {
        let elem = document.createElement("span");
        elem.id = id;
        elem.innerHTML = labeltext;
        elem.title = tooltip;
        elem.classList = "title";

        return $(elem);
    }

    static createToggleSwitch(id, onchange, initial=false) {
        let legacy = false;
        let elem = document.createElement("input");
        elem.onchange = () => {
            onchange(elem);
        }
        elem.type = "checkbox";
        elem.id = id;
        elem.checked = initial;


        if(!legacy) {
            // Add CSS "toggle-slider" elements
            // This requires more HTML.
            let styled_elem = document.createElement("label");
            styled_elem.classList = "switch";
            $(styled_elem).append(elem);
            $(styled_elem).append($('<span class="slider round"></span>'));
            return $(styled_elem);
        }

        return $(elem);
    }


    static createTextInput(id, onchange, initial = '') {
        let elem = document.createElement("input");
        // oninput triggers on every change (as opposed to onchange, which only changes on deselection)
        elem.onchange = () => {
            onchange(elem);
        }
        elem.type = "text";
        elem.id = id;
        elem.value = initial
        return $(elem);
    }

    static createLongTextInput(id, onchange, initial = '') {
        let elem = document.createElement("textarea");
        // oninput triggers on every change (as opposed to onchange, which only changes on deselection)
        elem.onchange = () => {
            onchange(elem.innerHTML);
        }
        elem.id = id;
        elem.innerHTML = initial
        return $(elem);
    }

    static createSelectInput(id, onchange, options, initial = '') {
        let elem = document.createElement("select");

        for(let option of options) {
            let option_elem = document.createElement('option');
            option_elem.innerText = option;
            elem.append(option_elem);
        }
        
        // oninput triggers on every change (as opposed to onchange, which only changes on deselection)
        elem.oninput = () => {
            onchange(elem);
        }
        elem.id = id;
        elem.value = initial;
        return $(elem);
    }

    static createIntInput(id, onchange, initial = 0) {
        let elem = document.createElement("input");
        elem.oninput = () => {
            onchange(elem);
        }
        elem.type = "number";
        elem.step = 1;
        elem.id = id;
        elem.value = initial;
        return $(elem);
    }


    static createFloatInput(id, onchange) {
        let elem = document.createElement("input");
        elem.oninput = () => {
            onchange(elem);
        }
        elem.type = "number";
        elem.id = id;
        return $(elem);
    }

    static createButton(id, onclick, label) {
        let elem = document.createElement("button");
        elem.onclick = () => {
            onclick(elem);
        };
        elem.innerHTML = label;
        return $(elem);
    }
}

function start_DIODE() {
    var diode = new DIODE();
    window.diode = diode;
    diode.initEnums();
    diode.pubSSH(true);

    $("#toolbar").w2toolbar({
        name: "toolbar",
        items: [
            { type: 'menu',   id: 'file-menu', caption: 'File', icon: 'material-icons-outlined gmat-folder', items: [
                { text: 'Start', icon: 'material-icons-outlined gmat-new_folder', id: 'start' },
                { text: 'Open', icon: 'material-icons-outlined gmat-open', id: 'open-file' },
                { text: 'Save', icon: 'material-icons-outlined gmat-save', id: 'save' },
            ]},
            { type: 'break',  id: 'break0' },
            { type: 'menu',   id: 'settings-menu', caption: 'Settings', icon: 'material-icons-outlined gmat-settings', items: [
                { text: 'DACE settings', icon: 'material-icons-outlined gmat-settings-cloud', id: 'diode-settings' }, 
                { text: 'DIODE settings', icon: 'material-icons-outlined gmat-settings-application', id: 'diode-settings' }, 
                { text: 'Run Configurations', icon: 'material-icons-outlined gmat-playlist_play', id: 'runoptions' }, 
                { text: 'Runqueue', icon: 'material-icons-outlined gmat-view_list', id: 'runqueue' }, 
                { text: 'Perfdata', id: 'perfdata' },
                { text: 'Perftimes', id: 'perftimes' },
            ]},
            { type: 'menu',  icon: 'material-icons-outlined gmat-build', id: 'compile-menu', caption: 'Compile', items: [
                { text: 'Compile', id: 'compile' , icon: 'material-icons-outlined gmat-gavel' }, 
                { text: 'Run', id: 'run', icon: 'material-icons-outlined gmat-play' }, 
                { text: 'Discard changes and compile source', id: 'compile-clean', icon: 'material-icons-outlined gmat-clear' }, 
            ]},
            { type: 'menu-radio', id: 'runconfig', text: function(item) {
                let t = (typeof(item.selected) == 'string') ? item.selected : item.selected(); let el = this.get('runconfig:' + t); return "Config: " + ((el == null) ? diode.getCurrentRunConfigName() : el.text);
            }, selected: function(item) {
                return diode.getCurrentRunConfigName();
            }, items: [{
                id: 'default', text: "default"
            }
            ]},
            { type: 'menu',  id: 'transformation-menu', caption: 'Transformations', items: [
                { text: 'History', id: 'history' },
                { text: 'Available Transformations', id: 'available' }, 
                
            ]},
            { type: 'menu',   id: 'group-menu', caption: 'Group', icon: 'material-icons-outlined gmat-apps', items: [
                //{ text: 'Group by SDFGs', id: 'group-sdfgs' }, 
                { text: 'Group default', id: 'group-diode1' }
            ]},
            { type: 'menu',   id: 'closed-windows', caption: 'Closed windows', icon: 'material-icons-outlined gmat-reopen', items: []},
        ],
        onClick: function (event) {
            if(event.target === 'file-menu:open-file') {
                diode.openUploader("code-python");
            }
            if(event.target === 'file-menu:start') {
                // Close all windows before opening Start component
                diode.closeAll();

                let config = {
                    type: 'component',
                    componentName: 'StartPageComponent',
                    componentState: {}
                };
        
                diode.addContentItem(config);
            }
            if(event.target === 'file-menu:save') {
                diode.project().save();
            }
            if(event.target == "settings-menu:diode-settings") {
                diode.open_diode_settings();
            }
            if(event.target == "settings-menu:runqueue") {
                diode.open_runqueue();
            }
            if(event.target == "settings-menu:perfdata") {
                //diode.load_perfdata();
                diode.show_inst_options();
            }
            if(event.target == "settings-menu:perftimes") {
                diode.show_exec_times();
            }
            if(event.target == "group-menu:group-sdfgs") {
                diode.groupOptGraph(); diode.groupSDFGsAndCodeOutsTogether();
            }
            if(event.target == "group-menu:group-diode1") {
                diode.groupLikeDIODE1();
            }
            if(event.target == "runconfig") {
                let m = this.get(event.target);

                let configs = diode.getRunConfigs();

                m.items = [];

                for(let c of configs) {
                    let cname = c['Configuration name'];
                    m.items.push({id: cname, text: cname});
                }
            }
            if(event.target.startsWith("runconfig:")) {
                let name = event.target.substr("runconfig:".length);
                diode.setCurrentRunConfig(name);
            }
            if(event.target == "transformation-menu:history") {
                diode.addContentItem({
                    type: 'component',
                    componentName: 'TransformationHistoryComponent',
                    title: "Transformation History",
                    componentState: {}
                });
            }
            if(event.target == "transformation-menu:available") {
                diode.addContentItem({
                    type: 'component',
                    componentName: 'AvailableTransformationsComponent',
                    componentState: {}
                });
            }
            if(event.target == "compile-menu:compile") {
                // "Normal" recompilation
                diode.gatherProjectElementsAndCompile(diode, {}, {
                    sdfg_over_code: true
                });
            }
            if(event.target == "compile-menu:compile-clean") {
                diode.project().request(["clear-errors"], () => {});
                diode.project().discardTransformationsAfter(0);
                // Compile, disregarding everything but the input code
                diode.project().request(['input_code'], msg => {
                    diode.compile(diode, msg['input_code']);
                }, {
                    timeout: 300,
                    on_timeout: () => alert("No input code found, open a new file")
                });
            }
            if(event.target == "settings-menu:runoptions") {
                diode.show_run_options(diode);
            }
            if(event.target == "compile-menu:run") {
                // Running

                diode.ui_compile_and_run(diode);
            }
            if(event.target == "closed-windows") {
                let m = this.get(event.target);

                // Clear the items first (they will be re-read from the project)
                m.items = [];

                // Add a "clear all"
                m.items.push({text: "Clear all", id: 'clear-closed-windows', icon: 'material-icons-outlined gmat-clear'});

                let elems = diode.project().getClosedWindowsList();
                for(let x of elems) {
                    let name = x[0];

                    m.items.push({text: name, id: 'open-closed-' + x[1].created});
                }

                this.refresh();
                
            }
            if(event.target == 'closed-windows:clear-closed-windows') {
                diode.project().clearClosedWindowsList();
            }
            if(event.target.startsWith("closed-windows:open-closed-")) {
                // This is a request to re-open a closed window
                let name = event.target;
                name = name.substr("closed-windows:open-closed-".length);

                diode.project().reopenClosedWindow(name);

            }
        }
    });
    

    var goldenlayout_config = {
        content: [{
            type: 'row',
            content:[{
                type: 'component',
                componentName: 'StartPageComponent',
                componentState: {}
            }]
        }]
    };

    let saved_config = sessionStorage.getItem('savedState');
    //saved_config = null; // Don't save the config during development
    var goldenlayout = null;
    if(saved_config !== null) {
        goldenlayout = new GoldenLayout(JSON.parse(saved_config), $('#diode_gl_container'));
    }
    else {
        goldenlayout = new GoldenLayout(goldenlayout_config, $('#diode_gl_container'));
    }

    goldenlayout.on('stateChanged', diode.debounce("stateChanged", function() {
        if(!(goldenlayout.isInitialised && goldenlayout.openPopouts.every(popout => popout.isInitialised))) {
            return;
        }
        // Don't serialize SubWindows
        if(goldenlayout.isSubWindow)
            return;
        let tmp = goldenlayout.toConfig();
        //find_object_cycles(tmp);
        let state = JSON.stringify( tmp );
  	    sessionStorage.setItem( 'savedState', state );
    }, 500));

    if(!goldenlayout.isSubWindow) {
        goldenlayout.eventHub.on('create-window-in-main', x => {
            let config = JSON.parse(x);

            diode.addContentItem(config);
        });
    }

    goldenlayout.registerComponent( 'testComponent', function( container, componentState ){
        container.getElement().html( '<h2>' + componentState.label + '</h2>' );
    });
    goldenlayout.registerComponent( 'SettingsComponent', function( container, componentState ){
        // Wrap the component in a context 
        let diode_context = new DIODE_Context_Settings(diode, container, componentState);
        $(container.getElement()).load("settings_view.html", function() {
            diode_context.get_settings();
        });
        
    });
    goldenlayout.registerComponent( 'PerfTimesComponent', function( container, componentState ){
        // Wrap the component in a context 
        let diode_context = new DIODE_Context_PerfTimes(diode, container, componentState);
        diode_context.setupEvents(diode.getCurrentProject());
        diode_context.create();
    });
    goldenlayout.registerComponent( 'InstControlComponent', function( container, componentState ){
        // Wrap the component in a context 
        let diode_context = new DIODE_Context_InstrumentationControl(diode, container, componentState);
        diode_context.setupEvents(diode.getCurrentProject());
        diode_context.create();
    });
    goldenlayout.registerComponent( 'RooflineComponent', function( container, componentState ){
        // Wrap the component in a context 
        let diode_context = new DIODE_Context_Roofline(diode, container, componentState);
        diode_context.setupEvents(diode.getCurrentProject());
        diode_context.create();
    });
    goldenlayout.registerComponent( 'SDFGComponent', function( container, componentState ){
        // Wrap the component in a context 
        let diode_context = new DIODE_Context_SDFG(diode, container, componentState);
                
        diode_context.create_renderer_pane(componentState["sdfg_data"]);
        diode_context.setupEvents(diode.getCurrentProject());
    });
    goldenlayout.registerComponent( 'TransformationHistoryComponent', function( container, componentState ){
        // Wrap the component in a context 
        let diode_context = new DIODE_Context_TransformationHistory(diode, container, componentState);
        diode_context.setupEvents(diode.getCurrentProject());
        let hist = diode_context.project().getTransformationHistory();
        diode_context.create(hist);
        
    });
    goldenlayout.registerComponent( 'AvailableTransformationsComponent', function( container, componentState ){
        // Wrap the component in a context 
        let diode_context = new DIODE_Context_AvailableTransformations(diode, container, componentState);
        diode_context.setupEvents(diode.getCurrentProject());
        diode_context.create();
        
    });
    goldenlayout.registerComponent( 'CodeInComponent', function( container, componentState ){
        // Wrap the component in a context 
        let diode_context = new DIODE_Context_CodeIn(diode, container, componentState);
        let editorstring = "code_in_" + diode_context.created;
        let parent_element = $(container.getElement());
        let new_element = $("<div id='" + editorstring + "' style='height: 100%; width: 100%; overflow-y:auto'></div>");
        
        
        parent_element.append(new_element);
        parent_element.hide().show(0);
        
        (function(){
            let editor_div = new_element;
            editor_div.attr("id", editorstring);
            editor_div.text(componentState.code_content);
            editor_div.hide().show(0);
            let editor = ace.edit(new_element[0]);
            editor.setTheme(DIODE.themeString());
            editor.session.setMode("ace/mode/python");
            editor.getSession().on('change', () => {
                container.extendState({ "code_content": editor.getValue() });
            });

            setup_drag_n_drop(new_element[0], (mime, content) => {
                // #TODO: Set session mode from mime type - but we need a switch to manually do that first
                console.log("File dropped", mime, content);

                editor.setValue(content);
                editor.clearSelection();
            });

            editor.resize();

            editor.commands.addCommand({
                name: 'Compile',
                bindKey: {win: 'Ctrl-P',  mac: 'Command-P'},
                exec: function(editor) {
                    alert("Compile pressed");
                    diode_context.compile(editor.getValue());
                },
                readOnly: true // false if this command should not apply in readOnly mode
            });
            editor.commands.addCommand({
                name: 'Compile and Run',
                bindKey: {win: 'Alt-R',  mac: 'Alt-R'},
                exec: function(editor) {
                    alert("Compile & Run pressed");
                    diode_context.compile_and_run(editor.getValue());
                },
                readOnly: true // false if this command should not apply in readOnly mode
            });
            diode_context.setEditorReference(editor);
            diode_context.setupEvents(diode.getCurrentProject());
        })
    ()
    ;
    
        
    });

    goldenlayout.registerComponent( 'CodeOutComponent', function( container, componentState ){
        // Wrap the component in a context 
        let diode_context = new DIODE_Context_CodeOut(diode, container, componentState);
        let editorstring = "code_out_" + diode_context.created;
        let parent_element = $(container.getElement());
        let new_element = $("<div id='" + editorstring + "' style='height: 100%; width: 100%; overflow:auto'></div>");
        
        
        parent_element.append(new_element);
        parent_element.hide().show(0);
        
        (() => {
            let editor_div = new_element;
            editor_div.attr("id", editorstring);
            editor_div.hide().show(0);
            let editor = ace.edit(new_element[0]);
            editor.setTheme(DIODE.themeString());
            editor.session.setMode("ace/mode/c_cpp");
            editor.setReadOnly(true);


            diode_context.setEditorReference(editor);
            diode_context.setupEvents(diode.getCurrentProject());

            let extracted = diode_context.getState().code;
            diode_context.setCode(extracted);
            editor.resize();
        })
    ()
    ;
        
        
    });

    // Create an error component which is used for all errors originating in python.
    // As such, the errors are usually tracebacks. The current implementation
    // (just displaying the output) is rudimentary and can/should be improved.
    // #TODO: Improve the error-out formatting
    goldenlayout.registerComponent( 'ErrorComponent', function( container, componentState ){
        // Wrap the component in a context 
        let diode_context = new DIODE_Context_Error(diode, container, componentState);
        let editorstring = "error_" + diode_context.created;
        let parent_element = $(container.getElement());
        let new_element = $("<div id='" + editorstring + "' style='height: 100%; width: 100%; overflow:auto'></div>");
        
        
        parent_element.append(new_element);
        parent_element.hide().show(0);
        
        (() => {
            let editor_div = new_element;
            editor_div.attr("id", editorstring);
            editor_div.hide().show(0);
            let editor = ace.edit(new_element[0]);
            editor.setTheme(DIODE.themeString());
            editor.session.setMode("ace/mode/python");


            diode_context.setEditorReference(editor);
            diode_context.setupEvents(diode.getCurrentProject());

            let extracted = diode_context.getState().error;
            diode_context.setError(extracted);
            editor.resize();
        })
    ()
    ;
        
        
    });

    goldenlayout.registerComponent( 'TerminalComponent', function( container, componentState ){
        // Wrap the component in a context 
        let diode_context = new DIODE_Context_Terminal(diode, container, componentState);
        let editorstring = "terminal_" + diode_context.created;
        let parent_element = $(container.getElement());
        let new_element = $("<div id='" + editorstring + "' style='height: 100%; width: 100%; overflow:auto'></div>");
        
        parent_element.append(new_element);
        parent_element.hide().show(0);
        
    
        let editor_div = new_element;
        editor_div.hide().show(0);
        let editor = ace.edit(new_element[0]);
        editor.setTheme(DIODE.themeString());
        editor.session.setMode("ace/mode/sh");
        editor.setReadOnly(true);

        let firstval = diode_context.getState().current_value;
        if(firstval !== undefined)
            editor.setValue(firstval);
        editor.clearSelection();

        diode_context.setEditorReference(editor);

        console.log("Client listening to", editorstring);

        goldenlayout.eventHub.on(editorstring, function(e) {
            diode_context.append(e);
        });

        diode_context.setupEvents(diode.getCurrentProject());
    });

    goldenlayout.registerComponent( 'DIODESettingsComponent', function( container, componentState ){
        let diode_context = new DIODE_Context_DIODESettings(diode, container, componentState);
        let divstring = "diode_settings" + diode_context.created;
        let parent_element = $(container.getElement());
        let new_element = $("<div id='" + divstring + "' style='height: 100%; width: 100%; overflow:auto'></div>");

        new_element.append("<h1>DIODE settings</h1>");
        diode_context.setContainer(new_element);
        parent_element.append(new_element);

    });

    goldenlayout.registerComponent( 'RunConfigComponent', function( container, componentState ){
        let diode_context = new DIODE_Context_RunConfig(diode, container, componentState);

        diode_context.setupEvents(diode.getCurrentProject());
        diode_context.create();
    });

    goldenlayout.registerComponent( 'PropWinComponent', function( container, componentState ){
        // Wrap the component in a context 
        let diode_context = new DIODE_Context_PropWindow(diode, container, componentState);
        
        let elem = document.createElement('div');
        elem.classList.add("sdfgpropdiv");
        elem.style = "width: 100%; height: 100%";
        $(container.getElement()).append(elem);

        diode_context.setupEvents(diode.getCurrentProject());
        diode_context.createFromState();
    });

    goldenlayout.registerComponent( 'StartPageComponent', function( container, componentState){
        let diode_context = new DIODE_Context_StartPage(diode, container, componentState);



        diode_context.setupEvents(diode.getCurrentProject());
        diode_context.create();
    });

    goldenlayout.registerComponent( 'RunqueueComponent', function( container, componentState ){
        // Wrap the component in a context 
        let diode_context = new DIODE_Context_Runqueue(diode, container, componentState);
        

        diode_context.setupEvents(diode.getCurrentProject());
        diode_context.create();
    });

    goldenlayout.on('itemDestroyed', e => {
        if(e.config.componentState === undefined) {
            // Skip non-components
            return;
        }
        let x = e.config.componentState.created;
        goldenlayout.eventHub.emit('destroy-' + x);
        console.log("itemDestroyed", e);
    });

    diode.setLayout(goldenlayout);
    diode.getProject();


    goldenlayout.init();

    window.addEventListener('resize', x => {
        // goldenlayout does not listen to resize events if it is not full-body
        // So it must be notified manually
        goldenlayout.updateSize();
    });
    

    document.body.addEventListener('keydown', (ev) => {
        diode.onKeyDown(ev);
    });
    document.body.addEventListener('keyup', (ev) => {
        diode.onKeyUp(ev);
    });
    diode.addKeyShortcut('gg', () => { diode.groupOptGraph(); diode.groupSDFGsAndCodeOutsTogether(); } );
    diode.addKeyShortcut('gd', () => { diode.groupLikeDIODE1(); } );
    diode.addKeyShortcut('0', () => {
        diode.open_diode_settings();
    });
    diode.addKeyShortcut('r', () => { diode.gatherProjectElementsAndCompile(diode, {}, { sdfg_over_code: true }); });

    diode.setupEvents();

    // Add drag & drop for the empty goldenlayout container
    let dgc = $("#diode_gl_container");
    let glc = dgc[0].firstChild;
    setup_drag_n_drop(glc, (mime, content) => {
        console.log("File dropped", mime, content);

        let config = {
            type: "component",
            componentName: "CodeInComponent",
            componentState: {
                code_content: content
            }
        };

        diode.addContentItem(config);
    }, undefined, {
        readMode: "text",
        condition: (elem) => elem.childNodes.length == 0 // Only if empty
    });
};

export {start_DIODE, REST_request, 
    FormBuilder, find_object_cycles, setup_drag_n_drop}
