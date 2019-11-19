import {CanvasDrawManager, Bracket, min_func, max_func, Pos} from "./renderer_util.js"
import {DrawNodeState} from "./sdfg_renderer.js"
import {ObjectHelper, MathHelper, CriticalPathAnalysis, MemoryAnalysis} from "./datahelper.js"
import { VectorizationButton, AutoSuperSectionVectorizationAnalysis, SuperSectionVectorizationAnalysis } from "./vectorization_button.js";
import { ParallelizationButton } from "./parallelization_button.js";
import { MemoryButton } from "./memory_button.js";
import { MemoryOpButton } from "./memop_button.js";
import { CacheOpButton } from "./cache_button.js";



class SdfgState {
    constructor() {
        this.ctx = null;
        this.sdfg = null;
        this.perfdata = null;
        this.perfdata_mode = {};
        this.graph = null;
        this.top_level_graph = null;
        this.state_graphs = {};
        this.canvas_manager = null;
        this.brackets = {};

        this.performance_references = {};

        this.target_memory_speed = 20.0;

        this.highlights = []; // List of tuples of stateid and nodeid to highlight.

        this.graphcache = {}; // Cache for the graphs

        if (typeof socket !== 'undefined') {
            this.communicator = new Communicator(socket);
        }
        else {
            this.communicator = null;
        }

        this.request_download = "";

        this._canvas_click_event_listener = null;
        this._canvas_hover_event_listener = null;
        this._canvas_dblclick_event_listener = null;
        this._canvas_oncontextmenu_handler = null;


        this._use_lean_interface = true;
        this._analysis_provider = null;
    }

    setAnalysisProvider(func) {
        this._analysis_provider = func
    }

    destroy() {
        this.clearHighlights();
        this.clearPerfData();

        this.canvas_manager.destroy();
    }


    setTargetMemBandwidth(target_bandwidth) {
        this.target_memory_speed = target_bandwidth;
    }
    defaultRun() {
        if(this.perfdataIsLazy()) {
            // Return a lazy object
            return new SuperSectionFetcher(this.communicator, this.perfdata.type, undefined, "meta:most_cores");
        }
        else {
            return new SuperSectionFetcher(this.communicator, this.perfdata.type, this.perfdata.payload.find(x => x.runopts == "# ;export OMP_NUM_THREADS=4; Running in multirun config").data);
        }
    }
    
    getPerformanceReferences(id) {
        return this.performance_references[id];
    }

    getCanvas() {
        return this.ctx.canvas;
    }

    setCtx(ctx) { this.ctx = ctx; this.canvas_manager = new CanvasDrawManager(ctx, this); this.canvas_manager.draw(); }
    setSDFG(sdfg) { this.sdfg = sdfg; this.graphcache = {}; this.canvas_manager.draw(); }
    perfdataIsLazy() {
        return this.perfdata != null && this.perfdata.type == "DataReady";
    }
    setPerfData(pd) {
        if(pd.type == "DataReady") {
            // Clear and set to lazy.
            this.clearPerfData();
            this.perfdata = pd;
            this.perfdata_mode[pd.mode] = "avail";
            
            return;
        }
        let mode = pd.mode;
        if(mode == "default") 
        {
            this.perfdata = null;
            this.clearPerfData();
        }
        
        this.perfdata = ObjectHelper.mergeRecursive(this.perfdata, pd, {
            // Special case: Merge the different runs
            "payload": (o1, o2) => {
                let ret = o1;
                
                for(let i = 0; i < o2.length; ++i) {
                    for(let j = 0; j < ret[i].data.length; ++j) {
                        ObjectHelper.assert("Same node", ret[i].data[j].supernode == o2[i].data[j].supernode);
                        ret[i].data[j] = ObjectHelper.mergeRecursive(ret[i].data[j], o2[i].data[j], {
                            // Special case: Merge Supersections
                            "sections": (o1, o2) => {
                            let ret = o1;
                            for(let i = 0; i < ret.length; ++i) {
                                let b_found = false;
                                for(let j = 0; j < o2.length; ++j) {
                                    if(ret[i].entry_node == o2[j].entry_node && ret[i].entry_core == o2[j].entry_core) {
                                        b_found = true;
                                        ObjectHelper.assert("Same node", ret[i].entry_node == o2[j].entry_node);
                                        ObjectHelper.assert("Same core", ret[i].entry_core == o2[j].entry_core);
                                        
                                        ret[i] = ObjectHelper.mergeRecursive(ret[i], o2[j], {
                                            // Special case: Merge Sections
                                            "entries": (o1, o2) => {
                                                let ret = o1;

                                                for(let i = 0; i < ret.length; ++i) {
                                                    let vi = ret[i];
                                                    let b_found = false;
                                                    for(let j = 0; j < o2.length; ++j) {
                                                        let vj = o2[j];

                                                        if(vi.flags == vj.flags && vi.node == vj.node && vi.iteration == vj.iteration && vi.thread == vj.thread) {
                                                            ret[i] = ObjectHelper.mergeRecursive(ret[i], vj);
                                                            b_found = true;
                                                        }
                                                    }
                                                }

                                                return ret;
                                            }
                                        });
                                    }
                                }
                                ObjectHelper.assert("found", b_found);
                            }
                            return ret;
                        }});
                    }
                }
                return ret;
            },
        });

        this.perfdata_mode[mode] = "avail";
        console.log("Got new mode " + mode);
    }
    
    clearPerfData() { 
        this.perfdata = null;
        this.perfdata_mode = {}; 
        this.brackets = {}; 
    }
    addHighlight(hl) {
        this.highlights.push(hl);
    }
    clearHighlights() {
        this.highlights = [];
    }

    setGraph(stateid, g) { this.state_graphs[stateid] = g; }

    setTopLevelGraph(g) { ObjectHelper.assert("g valid", g != null); this.top_level_graph = g; }

    nodeid(unified) {
        return unified & 0xFFFF;
    }

    // Passed through from graph
    node(stateid, nodeid) {
        return this.state_graphs[stateid].node(this.nodeid(nodeid));
    }

    addBracket(b, stateid, nodeid) {
        if(nodeid == undefined) nodeid = 0;
        stateid = new Number(stateid);
        nodeid = new Number(nodeid);
        let unified_id = (stateid << 16) | nodeid;

        let ks = ObjectHelper.listKeys(this.brackets);
        if(ks.includes(unified_id.toString())) {
            this.canvas_manager.removeDrawable(this.brackets[unified_id]);
            this.brackets[unified_id].destroy();
        }
        this.brackets[unified_id] = b;
    }

    drawSDFG() {
        let g = this.top_level_graph;
        if(g == null) return; // Nothing to draw (yet)
        var renderer = new DrawNodeState(this.ctx, -1, this);
        renderer.onStartDraw();
        paint_sdfg(g, this.sdfg, renderer);

        // Draw what is inside the state boxes, offset by the top 
        // left corner of the state box
        this.sdfg.nodes.forEach(state => {
            let state_x_offs = g.node(state.id).x - g.node(state.id).width / 2.0;
            let state_y_offs = g.node(state.id).y - g.node(state.id).height / 2.0;
            let ctx = this.ctx;
            ctx.fillText(state.id, state_x_offs+1.0*LINEHEIGHT, state_y_offs+1.0*LINEHEIGHT);
            let state_g = null;
            if(cache_graphs && (state.id in this.graphcache)) {
                state_g = this.graphcache[state.id];
            }
            else {
                if (state.attributes.is_collapsed == true) {
                    state_g = new dagre.graphlib.Graph();
                    state_g.setGraph({});
                    state_g.setDefaultEdgeLabel(function (u, v) { return {}; });
                    dagre.layout(state_g);

                    // Draw "+" sign that signifies that this state is collapsed
                    let x = g.node(state.id).x;
                    let y = g.node(state.id).y;
                    this.ctx.beginPath();
                    this.ctx.moveTo(x, y - LINEHEIGHT);
                    this.ctx.lineTo(x, y + LINEHEIGHT);
                    this.ctx.stroke();
                    this.ctx.moveTo(x - LINEHEIGHT, y);
                    this.ctx.lineTo(x + LINEHEIGHT, y);
                    this.ctx.stroke();
                }
                else {
                    state_g = layout_state(state, this.sdfg, this);
                    addXYOffset(state_g, state_x_offs + 2*LINEHEIGHT, state_y_offs+2*LINEHEIGHT);
                    this.graphcache[state.id] = state_g;
                }

            }

            renderer.stateid = state.id;
            paint_state(state_g, renderer);
        });

        renderer.onEndDraw();
    }

    init_SDFG() {
        let sdfg = this.sdfg;
        // draw the state boxes
        let g = layout_sdfg(sdfg, this);
        let bb = calculateBoundingBox(g);
        let cnvs = this.ctx.canvas;
        cnvs.style.backgroundColor = "#ffffff";
        cnvs.width = Math.min(Math.max(bb.width + 1000, cnvs.width), 16384);
        cnvs.height = Math.min(Math.max(bb.height + 1000, cnvs.height), 16384);
        paint_sdfg(g, null, new DrawNodeState(this.ctx, -1, this));

        this.setTopLevelGraph(g);
        let transthis = this;
        // draw what is inside the state boxes, offset by the top left corner of the state box
        sdfg.nodes.forEach(state => {
            let state_x_offs = g.node(state.id).x - g.node(state.id).width / 2.0;
            let state_y_offs = g.node(state.id).y - g.node(state.id).height / 2.0;
            let ctx = transthis.ctx;
            ctx.fillText(state.id, state_x_offs+1.0*LINEHEIGHT, state_y_offs+1.0*LINEHEIGHT);
            let state_g = layout_state(state, sdfg, this);
            addXYOffset(state_g, state_x_offs + 2*LINEHEIGHT, state_y_offs+2*LINEHEIGHT);
            paint_state(state_g, new DrawNodeState(ctx, state.id, transthis));
            transthis.setGraph(state.id, state_g);
        });
    }

    async createBracket(ctx, stateid, nodeid, sdfg_state = undefined) {
        // Hook into the old function to provide the new function without changing much of the old code
        if(this._use_lean_interface) return await this.createBracket2(ctx, stateid, nodeid);

        let global_measurement = stateid == 0xFFFF || stateid == 65535;
        let k = nodeid;
        let b = new Bracket(ctx);
        if(sdfg_state === undefined) {
            sdfg_state = global_state;
        }
        sdfg_state.addBracket(b, stateid, nodeid);

        let sections = this.defaultRun();
        let targetsection = null;
        // Only selects one single section!
        await sections.wait_ready();
        for(let section of sections) {
            let clsec = section.realize();
            ObjectHelper.assert("correct type", this.perfdataIsLazy() || (clsec instanceof SuperSection));
            
            let tmp = null;
            if(clsec instanceof LazySuperSection) tmp = await clsec.containsSection(k, stateid);
            else tmp = clsec.containsSection(k, stateid);
            if(tmp) {
                let tmp = null;
                if(clsec instanceof LazySuperSection) 
                    tmp = await clsec.toSection(k, stateid);
                else
                    tmp = clsec.toSection(k, stateid);

                targetsection = await ObjectHelper.valueFromPromise(tmp);
                
                if(targetsection === undefined) continue;
                break; 
            }
        }
        if(global_measurement) {
            let clsec = sections.elem(0).realize();
            let t1 = await ObjectHelper.valueFromPromiseFull(clsec.sections());
            ObjectHelper.assert("Correct type", t1 instanceof Array);
            ObjectHelper.assert("Correct len", t1.length == 1);
            targetsection = t1[0];
            ObjectHelper.logObject("targetsection", targetsection);

            // Set the overhead numbers
            this.performance_references["overhead_numbers"] = this.perfdata.overhead_numbers;
        }

        if(targetsection == null) {
            console.log("Failed to obtain a valid section!");
            if(targetsection_ignore_error) return undefined; 
        }
        ObjectHelper.assert("targetsection valid", targetsection != null);
        
        let path_analysis = AutoCriticalPathAnalysis(this.communicator, this.perfdata.payload, k, stateid).analyze();
        
        // Same thing as above, now for multiple runs.
        let t_atd1 = sections.map(x => x.realize().toSection(k, stateid));
        t_atd1 = await ObjectHelper.waitArray(t_atd1);
        let all_threads_data = t_atd1.filter(x => x != undefined && (x._entries != undefined || x.communicator != undefined)).map(x => AutoThreadAnalysis(this.communicator, x).analyze());
        let t_atd2 = await ObjectHelper.waitArray(all_threads_data);
        let all_analyses = new DataBlock(t_atd2, "all_threads");

        path_analysis = await ObjectHelper.valueFromPromise(path_analysis);

        if(this.perfdata.default_depth == -1 && this.perfdata.mode == "default") {
            // Reference (no inner instrumentation, only the whole frame).
            this.performance_references["critical_paths"] = path_analysis.data.critical_paths;
        }
        else {
            // Normal mode (nodes, not global)
        }


        let targetsection_analysis = AutoThreadAnalysis(this.communicator, targetsection).analyze();
        targetsection_analysis = await ObjectHelper.valueFromPromise(targetsection_analysis);
        
        // Get repetitions (we'll need this later)
        let repquery = this.communicator.getRepetitions().get();
        let repcount =  await ObjectHelper.valueFromPromise(repquery);
        
        all_analyses.repcount = repcount;

        let but1 = new ParallelizationButton(ctx, targetsection_analysis, all_analyses, path_analysis, this.communicator);
        b.addButton(but1);

        let supersections = sections.map(x => x.realize());

        let memsubsel = undefined;
        if(global_measurement) {
            memsubsel = supersections.map(x => x.sections()).filter(x => x != undefined);
        }
        else {
            memsubsel = supersections.map(x => x.getSections(k, stateid));
            for(let i = 0; i < memsubsel.length; ++i) {
                memsubsel[i] = await ObjectHelper.valueFromPromise(memsubsel[i]);
            }
            memsubsel = memsubsel.filter(x => x != undefined);
        }
        memsubsel = ObjectHelper.flatten(memsubsel);
        
        // Legacy
        //let all_mem_analyses = new DataBlock(memsubsel.map(x => new MemoryAnalysis(new Section(x)).analyze()).filter(x => x != undefined), "all_thread_mem");
        
        let tmp_ssama = sections.map(x => AutoSuperSectionMemoryAnalysis(this.communicator, x.realize(), k, stateid, sdfg_state.target_memory_speed).analyze());
        tmp_ssama = await tmp_ssama;
        let supersection_all_mem_analyses = new DataBlock(tmp_ssama.filter(x => x != null), "all_thread_mem");

        
        let all_mem_analyses = supersection_all_mem_analyses;
        for(let i = 0; i < all_mem_analyses.data.length; ++i) {
            all_mem_analyses.data[i] = await ObjectHelper.valueFromPromise(all_mem_analyses.data[i]);
        }
        all_mem_analyses.data = all_mem_analyses.data.filter(x => x != null);
        all_mem_analyses.repcount = repcount;

        let but2 = new MemoryButton(ctx, all_mem_analyses, sdfg_state.target_memory_speed);
        b.addButton(but2);


        but1.setOnDoubleClick(p => {
            let newwin = new DiodeWindow(window);
            newwin.setSenderData({ 
                className: "ParallelizationButton",
                dataParams: but1.dataparams
            });
            let subwin = newwin.open("subwindow.html", "_blank");
            if(!subwin) {
                console.log("Failed to open subwindow");
                alert("failed to open subwindow");
            }
            
            return true;
        });
        but2.setOnDoubleClick(p => {
            let newwin = new DiodeWindow(window);
            newwin.setSenderData({ 
                className: "MemoryButton",
                dataParams: but2.dataparams
            });
            let subwin = newwin.open("subwindow.html", "_blank");
            if(!subwin) {
                console.log("Failed to open subwindow");
                alert("failed to open subwindow");
            }
            
            return true;
        });


        if(ObjectHelper.listKeys(this.perfdata_mode).includes("all")) {
            let waited_data = sections.map(x => AutoSuperSectionVectorizationAnalysis(this.communicator, x.realize(), k, stateid, path_analysis));
            
            for(let i = 0; i < waited_data.length; ++i) {
                let tmp = waited_data[i].analyze();
                tmp = await tmp;
                waited_data[i] = tmp;
                if(waited_data[i] != undefined)
                    waited_data[i] = await ObjectHelper.valueFromPromiseFull(waited_data[i]);
            }

            let supersection_all_vec_analyses = new DataBlock(waited_data.filter(x => x != null && x !== undefined), "all_thread_vec");
            supersection_all_vec_analyses.repcount = repcount;
            let but4 = new VectorizationButton(ctx, supersection_all_vec_analyses);
            b.addButton(but4);

            but4.setOnDoubleClick(p => {
                let newwin = new DiodeWindow(window);
                newwin.setSenderData({ 
                    className: "VectorizationButton",
                    dataParams: but4.dataparams
                });
                let subwin = newwin.open("subwindow.html", "_blank");
                if(!subwin) {
                    console.log("Failed to open subwindow");
                    alert("failed to open subwindow");
                }
                
                return true;
            });
        }
        if(ObjectHelper.listKeys(this.perfdata_mode).includes("all")) {
            let waited_data = sections.map(x => AutoSuperSectionCacheOpAnalysis(this.communicator, x.realize(), k, stateid, path_analysis));
            
            for(let i = 0; i < waited_data.length; ++i) {
                let tmp = waited_data[i].analyze();
                tmp = await tmp;
                waited_data[i] = tmp;
                if(waited_data[i] != undefined)
                    waited_data[i] = await ObjectHelper.valueFromPromiseFull(waited_data[i]);
            }

            let supersection_all_vec_analyses = new DataBlock(waited_data.filter(x => x != null && x !== undefined), "all_thread_cacheop");
            supersection_all_vec_analyses.repcount = repcount;
            let but4 = new CacheOpButton(ctx, supersection_all_vec_analyses);

            but4.setOnDoubleClick(p => {
                let newwin = new DiodeWindow(window);
                newwin.setSenderData({ 
                    className: "CacheOpButton",
                    dataParams: but4.dataparams
                });
                let subwin = newwin.open("subwindow.html", "_blank");
                if(!subwin) {
                    console.log("Failed to open subwindow");
                    alert("failed to open subwindow");
                }
                
                return true;
            });

            b.addButton(but4);
        }
        if(ObjectHelper.listKeys(this.perfdata_mode).includes("all")) {
            let waited_data = sections.map(x => AutoSuperSectionMemoryOpAnalysis(this.communicator, x.realize(), k, stateid, path_analysis));
            
            for(let i = 0; i < waited_data.length; ++i) {
                let tmp = waited_data[i].analyze();
                tmp = await tmp;
                waited_data[i] = tmp;
                if(waited_data[i] != undefined)
                    waited_data[i] = await ObjectHelper.valueFromPromiseFull(waited_data[i]);
            }

            let supersection_all_vec_analyses = new DataBlock(waited_data.filter(x => x != null && x !== undefined), "all_thread_memop");
            supersection_all_vec_analyses.repcount = repcount;
            let but4 = new MemoryOpButton(ctx, supersection_all_vec_analyses);

            but4.setOnDoubleClick(p => {
                let newwin = new DiodeWindow(window);
                newwin.setSenderData({ 
                    className: "MemoryOpButton",
                    dataParams: but4.dataparams
                });
                let subwin = newwin.open("subwindow.html", "_blank");
                if(!subwin) {
                    console.log("Failed to open subwindow");
                    alert("failed to open subwindow");
                }
                
                return true;
            });

            b.addButton(but4);
        }


        b.setupEventListeners();

        return b;
    }

    async createBracket2(ctx, stateid, nodeid) {
        console.assert(this._analysis_provider != null, "Analysis provider set");
        let global_measurement = stateid == 0xFFFF || stateid == 65535;
        let k = nodeid;
        let b = new Bracket(ctx);
        
        this.addBracket(b, stateid, nodeid);

        let nodeinfo = {
            stateid: stateid,
            nodeid: nodeid
        };

        let ap = analysis_name => this._analysis_provider(analysis_name, nodeinfo);

        let ava = ap("all_vec_analyses");
        let cpa = ap("CriticalPathAnalysis");
        let apa = ap("ParallelizationAnalysis");
        let ama = ap("MemoryAnalysis");
        let amoa= ap("MemOpAnalysis");
        let acoa= ap("CacheOpAnalysis");

        let build_param = (a, j) => ({
            data: a.map(x => ({ 
                data: x.data,
                judgement: j(x.data) 
            })),
        });

        let build_param2 = (a, j) => ({
            data: a.map(x => x.data),
        });

        let def_run_filter = ap("defaultRun");

        let ava_param = build_param(ava, SuperSectionVectorizationAnalysis.sjudgement);
        let cpa_param = build_param(cpa, x => CriticalPathAnalysis.sjudgement({data: x}));
        let apa_param = build_param2(apa, x => undefined);
        let tpa_param = build_param(apa, x => undefined);
        let def_ama_param = build_param(def_run_filter(ama), x => MemoryAnalysis.sjudgement({data: x}));
        let ama_param = build_param(ama, x => MemoryAnalysis.sjudgement({data: x}));

        let amoa_param= build_param(def_run_filter(amoa), x => undefined);
        let acoa_param= build_param(def_run_filter(acoa), x => undefined);


        let par_but = new ParallelizationButton(ctx, tpa_param.data[0], apa_param, cpa_param.data[0], undefined);
        b.addButton(par_but);

        let mem_but = new MemoryButton(ctx, def_ama_param, 20.0);
        b.addButton(mem_but);

        let vec_but = new VectorizationButton(ctx, ava_param, cpa_param);
        b.addButton(vec_but);

        let mop_but = new MemoryOpButton(ctx, amoa_param, cpa_param);
        b.addButton(mop_but);
        
        let cop_but = new CacheOpButton(ctx, acoa_param, cpa_param);
        b.addButton(cop_but);

        b.setupEventListeners(this.canvas_manager);

        return b;
    }

    // drawPerfInfo for all states
    async drawAllPerfInfo() {

        if(this.perfdata != null && this.perfdata.overhead_percentage != undefined) {

            if(max_func(this.perfdata.overhead_percentage, x => x) > max_overhead_percentage)
            {
                alert("Warning: Instrumentation overhead exceeded limit (" + max_overhead_percentage + ") at least once.\nPercentages: " + JSON.stringify(this.perfdata.overhead_percentage));
            }
        }

        if(this.getPerformanceReferences("overhead_numbers") != undefined) {
            let overhead = this.getPerformanceReferences("overhead_numbers");

            if(auto_compensate_overhead) {
                // Subtract the overhead numbers from all measurements
                let affected_keys = ObjectHelper.listKeys(overhead);
                for(let k of affected_keys) {
                    let v = overhead[k];
                    ObjectHelper.modifyingMapRecursive(this.perfdata.payload, k, x => {
                        if(Number(x) - Number(v) > 0) {
                            console.log("Changing overhead...");
                            return Number(x) - Number(v);
                        }
                        else {
                            console.log("Error: Would underflow!");
                            return 0;
                        }
                    });
                }
            }
        }

        let states = null;

        if(!this._use_lean_interface) {
            let defaultRun = this.defaultRun();
            await defaultRun.wait_ready();
            let tmp = await ObjectHelper.valueFromPromise(defaultRun.allSectionStateIds());
            states = ObjectHelper.flatten(tmp);
        }
        else {
            states = this._analysis_provider("getstates", null);
        }
        states = MathHelper.unique(states);
        for(let x of states) {
            if(x == 0xFFFF || x == 65535)
            {
                // This is the global state
                let bb = calculateBoundingBox(this.top_level_graph);

                let right = bb.width;
                let top = 0;
                let bot = bb.height;
                
                let b_tmp = this.createBracket(this.ctx, x, undefined);
                let b = await b_tmp;
                ObjectHelper.assert("Global measurement valid", b != undefined);

                b.drawEx(new Pos(right, top), new Pos(right, bot), 50, 20, true, () => {});
                this.canvas_manager.addDrawable(b);

                continue;
            }
            await this.drawPerfInfo(x);
        }

        if(ObjectHelper.listKeys(this.perfdata_mode).includes("all")) {
            if(this.request_download != "") {
                // A download of all buttons was requested
                
                createImageDownload(this.brackets, this.request_download);
                this.request_download = ""; // Clear
            }
        }
    }

    // multistate-unaware!
    async drawPerfInfo(stateid) {
        let ctx = this.ctx;
        let scopedict = {};

        if(!this._use_lean_interface) {
            if(this.perfdata.type != "DataReady") {
                let payload = this.perfdata.payload;

                // Check correctness of data
                let x = new ResultVerifier(payload);
            }
        }

        // There are 2 types of nodes: Those that open a scope 
        // (MapEntry) and those that don't (the rest).
        // We search first for all section entry nodes.
        let all_entry_nodes = null;
        if(!this._use_lean_interface) {
            let tmp = await this.defaultRun().allSectionNodeIds(stateid);
            all_entry_nodes = ObjectHelper.flatten(tmp);
        }
        else {
            all_entry_nodes = this._analysis_provider("getnodes", null);
        }
        // Filter that down to uniques.
        all_entry_nodes = MathHelper.unique(all_entry_nodes);
        // For every entry node, we create an array.
        for(let tmp of all_entry_nodes) {
            scopedict[tmp] = [];
        }
        // Add an extra one for the top-level
        scopedict[null] = [];


        for(let sdfgnode of this.sdfg.nodes) {
            if(sdfgnode.id != stateid) continue;
            for(let node of sdfgnode.nodes) {
                let gnode = this.node(stateid, node.id);
                if(gnode == undefined) {
                    console.log("Welp. How did this happen? (undefined gnode for (" + stateid + ", " + node.id + "))");
                }
                else {
                    let anchor_x = gnode.x + gnode.width / 2.;
                    let anchor_y = gnode.y - gnode.height / 2;

                    // We are only interested in scopes for now.
                    // Ordering scopes "upside down", because an 
                    // entry node is not marked as scope entry, but 
                    // nodes inside that scope have a link to the 
                    // starting node.
                    if(node.scope_entry != null) {
                        // This node has a scope entry. Add this node to the dict with the key of the entry node (reverse the mapping basically)
                        if(scopedict[node.scope_entry] == undefined) {
                            scopedict[node.scope_entry] = [];
                            console.log("Had to create an array for unknown entry node " + node.scope_entry);
                        }
                        scopedict[node.scope_entry].push(this.nodeid(node.id));
                    }
                }
            }
        }

        // Topologically sort scopes
        let sorted_array = [];
        {
            let changed = true;
            while(changed) {
                changed = false;
                
                for(let key in scopedict) {
                    // Skip objects already taken
                    if(sorted_array.some(e => e == key)) continue; 


                    // For each array, check if the selected key is a dependency. If it is, check next (we are going outside -> inside)
                    let retry = false;
                    Object.keys(scopedict).forEach(e => {
                        if(scopedict[e].some(o => o == key && !sorted_array.some(x => x == e))) {
                            retry = true;
                        }
                    });
                    if(retry) continue;
                    
                    // Otherwise, we'll have a change
                    changed = true;

                    // Add the key to the sorted array
                    sorted_array.push(key);
                }

            }
        }

        // Since we now have the sorted array, but in the wrong 
        // order, reverse it.
        let sa = sorted_array.reverse();

        // Draw the performance information for each node in the sorted list
        for(let k of sa) {
            if(k == null) continue;
            if(k == "null") continue;

            if(!all_entry_nodes.map(x => x.toString()).includes(k.toString())) {
                continue; // Skip nodes for which we don't have performance data.
            }

            // From the key, we can read the affected nodes
            let affected = [];
            affected.push(k);
            let tmp = scopedict[k];
            affected.push(...tmp);


            // Now get the maximum and minimum y-positions
            let top = min_func(affected, x => {

                let gnode = this.node(stateid, x);

                return gnode.y - gnode.height / 2;
            });
            let bot = max_func(affected, x => {
                let gnode = this.node(stateid, x);

                return gnode.y + gnode.height / 2;
            });

            // Now we just have to do the same for the right side :)
            let right = max_func(affected, x => {
                let gnode = this.node(stateid, x);

                return gnode.x + gnode.width / 2;
            });

            this.createBracket(ctx, stateid, k).then(resval => {
                if(resval === undefined) return;

                resval.drawEx(new Pos(right, top), new Pos(right, bot), 50, 20, true, () => {});
                this.canvas_manager.addDrawable(resval);
            });
            
        }
    }


    setMouseHandlers(transmitter, contextmenu = false) {
        /*
            transmitter: class instance or object with the following duck-typed properties:
                .send(data):
                    Sends data to the host.

            contextmenu: Whether or not to subscribe to the contextmenu event as well
        */

        let canvas = () => this.getCanvas();
        let br = () => canvas().getBoundingClientRect();
        
        let comp_x = event => this.canvas_manager.mapPixelToCoordsX(event.clientX - br().left);
        let comp_y = event => this.canvas_manager.mapPixelToCoordsY(event.clientY - br().top);
        
        let zoom_comp = x => x;

        // Click
        if(this._canvas_click_event_listener != null) {
            canvas().removeEventListener('click', this._canvas_click_event_listener);
        }
        this._canvas_click_event_listener = x => canvas_mouse_handler(x, comp_x, comp_y, zoom_comp, this, transmitter);
        canvas().addEventListener('click', this._canvas_click_event_listener);

        // Mouse-move (hover)
        if(this._canvas_hover_event_listener != null) {
            canvas().removeEventListener('mousemove', this._canvas_hover_event_listener);
        }
        this._canvas_hover_event_listener = x => canvas_mouse_handler(x, comp_x, comp_y, zoom_comp, this, transmitter,
            'hover');
        canvas().addEventListener('mousemove', this._canvas_hover_event_listener);

        // Double-click
        if(this._canvas_dblclick_event_listener != null) {
            canvas().removeEventListener('dblclick', this._canvas_dblclick_event_listener);
        }
        this._canvas_dblclick_event_listener = x => {
            canvas_mouse_handler(x, comp_x, comp_y, zoom_comp, this,
                transmitter, 'dblclick');
        }
        canvas().addEventListener('dblclick', this._canvas_dblclick_event_listener);

        // Prevent double clicking from selecting text (see https://stackoverflow.com/a/43321596/6489142)
        canvas().addEventListener('mousedown', function (event) {
            if (event.detail > 1)
                event.preventDefault();
        }, false);


        if(contextmenu) {

            if(this._canvas_oncontextmenu_handler != null) {
                canvas().removeEventListener('contextmenu', this._canvas_oncontextmenu_handler);
            }
            this._canvas_oncontextmenu_handler = x => {
                x.preventDefault();
                canvas_mouse_handler(x, comp_x, comp_y, zoom_comp, this, transmitter, "contextmenu");
            };
            canvas().addEventListener('contextmenu', this._canvas_oncontextmenu_handler);
        }

    }


    setDragHandler() {
        /*
            The drag handler is used to drag the entire view around.
            This function is needed only for the new DIODE user concept.
        */

        let canvas = this.getCanvas();

        canvas.addEventListener('mousemove', e => {
            if(e.buttons === 0) {
                // Not our event, but keep track of the scale origin
                let br = canvas.getBoundingClientRect();
                this.canvas_manager.scale_origin.x = e.clientX - br.left;
                this.canvas_manager.scale_origin.y = e.clientY - br.top;
                return;
            }
            if(e.buttons & 1) {
                // Only accept the primary (~left) mouse button as dragging source
                // #TODO: Should there be an option to adjust this?

                let movement = [
                    e.movementX,
                    e.movementY
                ];

                this.canvas_manager.translate(...movement);
                this.canvas_manager.draw_async();
            }
        });

        // Touch event management
        var lastTouch = null, secondTouch = null;
        canvas.addEventListener("touchstart", function (e) {
            let touch = e.touches[0];
            lastTouch = touch;
            if (e.targetTouches.length > 1)
                secondTouch = e.touches[1];
            let mouseEvent = new MouseEvent("mousedown", {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        }, false);
        canvas.addEventListener("touchend", function (e) {
            let mouseEvent = new MouseEvent("mouseup", {});
            canvas.dispatchEvent(mouseEvent);
        }, false);
        canvas.addEventListener("touchmove", e => {
            if (e.targetTouches.length == 2) { // zoom (pinching)
                e.stopPropagation();
                e.preventDefault();

                // Find distance between two points and center, zoom to that
                let centerX = (lastTouch.clientX + secondTouch.clientX) / 2.0;
                let centerY = (lastTouch.clientY + secondTouch.clientY) / 2.0;
                let initialDistance = Math.sqrt((lastTouch.clientX - secondTouch.clientX) ** 2 +
                                                (lastTouch.clientY - secondTouch.clientY) ** 2);
                let currentDistance = Math.sqrt((e.touches[0].clientX - e.touches[1].clientX) ** 2 +
                                                (e.touches[0].clientY - e.touches[1].clientY) ** 2);

                let br = () => canvas.getBoundingClientRect();

                let comp_x = event => (centerX - br().left);
                let comp_y = event => (centerY - br().top);
                // TODO: Better scaling formula w.r.t. distance between touches
                this.canvas_manager.scale((currentDistance - initialDistance) / 30000.0, comp_x(e), comp_y(e));

                lastTouch = e.touches[0];
                secondTouch = e.touches[1];
            } else if (e.targetTouches.length == 1) { // dragging
                let touch = e.touches[0];
                if (!lastTouch)
                    lastTouch = touch;
                let mouseEvent = new MouseEvent("mousemove", {
                    clientX: touch.clientX,
                    clientY: touch.clientY,
                    movementX: touch.clientX - lastTouch.clientX,
                    movementY: touch.clientY - lastTouch.clientY,
                    buttons: 1
                });
                lastTouch = touch;
                canvas.dispatchEvent(mouseEvent);
            }
        }, false);
        // End of touch-based events
    }

    setZoomHandler() {
        let canvas = this.getCanvas();

        canvas.addEventListener('wheel', e => {

            e.stopPropagation();
            e.preventDefault();

            let canvas = () => this.getCanvas();
            let br = () => canvas().getBoundingClientRect();
        
            let comp_x = event => (event.clientX - br().left);
            let comp_y = event => (event.clientY - br().top);
            this.canvas_manager.scale(-e.deltaY / 1000.0, comp_x(e), comp_y(e));
            this.canvas_manager.draw_async();

        });
    }
}

function canvas_mouse_handler( event,
                                comp_x_func,
                                comp_y_func,
                                zoom_comp_func,
                                sdfg_state,
                                transmitter,
                                mode="click") {
                                    
    
    let x = comp_x_func(event);
    let y = comp_y_func(event);
    
    if(window.get_zoom != undefined) {
        x = zoom_comp_func(x);
        y = zoom_comp_func(y);
    }

    let clicked_elements = [];
    sdfg_state.sdfg.nodes.forEach(function (state) {
        if (isWithinBB(x,y, state.attributes.layout)) {
            let elem = {'type': state.type, 'id': state.id};
            clicked_elements.push(elem);
            state.nodes.forEach(function (node) {
                if (isWithinBB(x,y, node.attributes.layout)) {
                    let elem = {'type': node.type, 'id': node.id};
                    clicked_elements.push(elem);
                }
            });
            // Check edges (Memlets). A memlet is considered "clicked" if the label is clicked.
            state.edges.forEach((edge, id) => {
                if (isWithinBBEdge(x, y, edge.attributes.layout)) {
                    let elem = {'type': edge.type, 'true_id': id,
                        'id': {src: edge.src, dst: edge.dst }};
                    clicked_elements.push(elem);
                }
            });
        }
    });
    sdfg_state.sdfg.edges.forEach((edge, id) => {
        if (isWithinBBEdge(x, y, edge.attributes.layout)) {
            let elem = {'type': edge.type, 'true_id': id,
                'id': {src: edge.src, dst: edge.dst }};
            clicked_elements.push(elem);
        }
    });
    if(mode === "click" || mode === "dblclick") {
        transmitter.send(JSON.stringify({"msg_type": mode, "clicked_elements": clicked_elements}));

        // Prevent text from being selected etc.
        event.preventDefault();
    }
    else if(mode === "contextmenu") {
        // Send a message (as with the click handler), but include all information here as well.
        transmitter.send(JSON.stringify({
            "msg_type": "contextmenu", 
            "clicked_elements": clicked_elements,
            "lpos": {'x': x, 'y': y}, // Logical position (i.e. inside the graph),
            'spos': { 'x': event.x, 'y': event.y}, // Screen position, i.e. as reported by the event
        }));

    } else if(mode === "hover") {
        if (event.buttons & 1) // Dragging does not induce redaw/hover
            return;

        transmitter.send(JSON.stringify({
            "msg_type": "hover",
            "clicked_elements": clicked_elements,
            "lpos": {'x': x, 'y': y}, // Logical position (i.e. inside the graph),
            'spos': {'x': event.x, 'y': event.y}, // Screen position, i.e. as reported by the event
        }));
    }

    // TODO: Draw only if changed
    sdfg_state.canvas_manager.draw_async();
}

function message_handler(msg, sdfg_state = undefined) {
    if(sdfg_state === undefined) {
        sdfg_state = global_state;
    }
    let sdfg = null;
    if (typeof msg === 'string' || msg instanceof String) {
        sdfg = JSON.parse(msg);
    }
    else {
        sdfg = msg;
    }
    if (sdfg.type == "SDFG") {
        sdfg_state.clearPerfData();
        sdfg_state.canvas_manager.clearDrawables();
        sdfg_state.setSDFG(sdfg); // Set the sdfg to a global helper

        // Clear all brackets
        for(let x of Object.values(sdfg_state.brackets)) {
            x.destroy();
        }
        sdfg_state.brackets = {};
        sdfg_state.graphcache = {};
        sdfg_state.canvas_manager.clearDrawables();
    }
    else if(sdfg.type == "PerfInfo") {
        sdfg_state.setPerfData(sdfg);
        sdfg_state.canvas_manager.clearDrawables();
        sdfg_state.drawAllPerfInfo();
        return;
    }
    else if(sdfg.type == "DataReady") {

        // Do the same as when directly getting "PerfInfo" (as above), but with the twist that data has to be requested lazily.
        sdfg_state.setPerfData(sdfg);
        sdfg_state.canvas_manager.clearDrawables();
        sdfg_state.drawAllPerfInfo();
        return;
    }
    else if(sdfg.type == "fetcher") {
        // Lazy evaluation stuff.
        sdfg_state.communicator.receive(sdfg);
        return;
    }
    else if(sdfg.type == "MemSpeed") {
        sdfg_state.setTargetMemBandwidth(sdfg.payload);
        return;
    }
    else if(sdfg.type == "save-images") {
        sdfg_state.request_download = sdfg.name;
        return;
    }
    else if(sdfg.type == "highlight-element") {
        let tmp = {
            "state-id": sdfg['sdfg-id'],
            "node-id": sdfg['node-id']
        };
        sdfg_state.addHighlight(tmp);
        return;
    }
    else if(sdfg.type == "clear-highlights") {
        sdfg_state.clearHighlights();
        return;
    }
    else {
        console.log("Expected to receive an SDFG, but I got " + msg);
        return;
    }

    

    sdfg_state.init_SDFG();

}


function addXYOffset(g, x_offs, y_offs) {
    "use strict";
    g.nodes().forEach(function (v) {
        g.node(v).x += x_offs;
        g.node(v).y += y_offs;
    });
    g.edges().forEach(function (e) {
        let edge = g.edge(e);
        edge.x += x_offs;
        edge.y += y_offs;
        edge.points.forEach(function (p) {
            p.x += x_offs;
            p.y += y_offs;
        }); 
    });
}

function paint_sdfg(g, sdfg, drawnodestate) {

    ObjectHelper.assert("drawnodestate must be defined", drawnodestate != undefined);

    g.nodes().forEach( v => {
        drawnodestate.draw_state(g.node(v), v);
    });
    g.edges().forEach((e, id) => {
        drawnodestate.draw_edge(g.edge(e), id);
    });

}

function layout_sdfg(sdfg, sdfg_state = undefined) {


    // layout the sdfg as a dagre graph
    let g = new dagre.graphlib.Graph();

    if(sdfg_state === undefined) {
        sdfg_state = global_state;
    }
    
    let ctx = sdfg_state.ctx;

    // Set an object for the graph label
    g.setGraph({});

    // Default to assigning a new object as a label for each new edge.
    g.setDefaultEdgeLabel(function (u, v) { return {}; });

    // layout each state to get its size
    sdfg.nodes.forEach(function (state) {
        let stateinfo = {};
        stateinfo.label = state.id;
        if (state.attributes.is_collapsed == true) {
            stateinfo.width = ctx.measureText(stateinfo.label).width;
            stateinfo.height = LINEHEIGHT;
        } 
        else {
            let state_g = layout_state(state, sdfg, sdfg_state);
            stateinfo = calculateBoundingBox(state_g);
        }
        stateinfo.width += 4*LINEHEIGHT;
        stateinfo.height += 4*LINEHEIGHT;
        g.setNode(state.id, stateinfo);
    });

    sdfg.edges.forEach(function (edge) {
        let label = edge.attributes.data.label;
        let textmetrics = ctx.measureText(label);
        g.setEdge(edge.src, edge.dst, { name: label, label: label, height: LINEHEIGHT, width: textmetrics.width,
                                        sdfg: sdfg });
    });

    dagre.layout(g);

    // annotate the sdfg with its layout info
    sdfg.nodes.forEach(function (state) {
        let gnode = g.node(state.id);
        state.attributes.layout = {};
        state.attributes.layout.x = gnode.x;
        state.attributes.layout.y = gnode.y;
        state.attributes.layout.width = gnode.width;
        state.attributes.layout.sdfg = sdfg;
        state.attributes.layout.height = gnode.height;
    });

    sdfg.edges.forEach(function (edge) {
        let gedge = g.edge(edge.src, edge.dst);
        let bb = calculateEdgeBoundingBox(gedge);
        edge.attributes = {};
        edge.attributes.data = {};
        edge.attributes.data.label = gedge.label;
        edge.attributes.label = gedge.label;
        edge.attributes.layout = {};
        edge.attributes.layout.width = bb.width;
        edge.attributes.layout.height = bb.height;
        edge.attributes.layout.x = bb.x;
        edge.attributes.layout.y = bb.y;
        edge.attributes.layout.sdfg = sdfg;
        edge.attributes.layout.points = gedge.points;
    });

    return (g);

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

function layout_state(sdfg_state, sdfg, controller_state = undefined) {
    // layout the state as a dagre graph

    if(controller_state === undefined) controller_state = global_state;
    let g = new dagre.graphlib.Graph({multigraph: true});

    // Set an object for the graph label
    g.setGraph({ranksep: 15});

    // Default to assigning a new object as a label for each new edge.
    g.setDefaultEdgeLabel(function (u, v) { return {}; });

    // Add nodes to the graph. The first argument is the node id. The 
    // second is metadata about the node (label, width, height),
    // which will be updated by dagre.layout (will add x,y).

    // Process nodes hierarchically
    let toplevel_nodes = sdfg_state.scope_dict[-1];
    let drawn_nodes = new Set();

    function draw_node(node) {
        let nodesize = calculateNodeSize(sdfg_state, node, controller_state);
        node.attributes.layout = {}
        node.attributes.layout.width = nodesize.width;
        node.attributes.layout.height = nodesize.height;
        node.attributes.layout.label = node.label;
        node.attributes.layout.in_connectors = node.attributes.in_connectors;
        node.attributes.layout.out_connectors = node.attributes.out_connectors;
        node.attributes.layout.sdfg = sdfg;
        node.attributes.layout.state = sdfg_state;
        g.setNode(node.id, node.attributes.layout);
        drawn_nodes.add(node.id.toString());

        // Recursively draw nodes
        if (node.id in sdfg_state.scope_dict) {
            if (node.attributes.is_collapsed)
                return;
            sdfg_state.scope_dict[node.id].forEach(function (nodeid) {
                let node = sdfg_state.nodes[nodeid];
                draw_node(node);
            });
        }
    }


    toplevel_nodes.forEach(function (nodeid) {
        let node = sdfg_state.nodes[nodeid];
        draw_node(node);
    });

    let ctx = controller_state.ctx;
    sdfg_state.edges.forEach((edge, id) => {
        edge = check_and_redirect_edge(edge, drawn_nodes, sdfg_state);
        if (!edge) return;

        let label = edge.attributes.data.label;
        console.assert(label != undefined);
        let textmetrics = ctx.measureText(label);

        // Inject layout information analogous to state nodes
        edge.attributes.layout = {
            label: label,
            width: textmetrics.width,
            height: LINEHEIGHT,
            sdfg: sdfg,
            state: sdfg_state
        };
        g.setEdge(edge.src, edge.dst, edge.attributes.layout, id);
    });

    dagre.layout(g);


    sdfg_state.edges.forEach(function (edge, id) {
        edge = check_and_redirect_edge(edge, drawn_nodes, sdfg_state);
        if (!edge) return;
        let gedge = g.edge(edge.src, edge.dst, id);
        let bb = calculateEdgeBoundingBox(gedge);
        edge.attributes.layout.width = bb.width;
        edge.attributes.layout.height = bb.height;
        edge.attributes.layout.x = bb.x;
        edge.attributes.layout.y = bb.y;
        edge.attributes.layout.points = gedge.points;
    });

    return g;
}

function calculateNodeSize(sdfg_state, node, controller_state = undefined) {
    if(controller_state === undefined) controller_state = global_state;
    let ctx = controller_state.ctx;
    let labelsize = ctx.measureText(node.label).width;
    let inconnsize = 0;
    let outconnsize = 0;
    node.attributes.in_connectors.forEach(function(conn) {
        // add 10px of margin around each connector
        inconnsize += ctx.measureText(conn).width + 10;
    });
    node.attributes.out_connectors.forEach(function(conn) {
        // add 10px of margin around each connector
        outconnsize += ctx.measureText(conn).width + 10;
    });
    let maxwidth = Math.max(labelsize, inconnsize, outconnsize);
    let maxheight = 2*LINEHEIGHT;
    if (node.attributes.in_connectors.length + node.attributes.out_connectors.length > 0) {
        maxheight += 4*LINEHEIGHT;
    }

    let size = { width: maxwidth, height: maxheight }

    // add something to the size based on the shape of the node
    if (node.type == "AccessNode") {
        size.width += size.height;
    }
    else if (node.type.endsWith("Entry")) {
        size.width += 2.0 * size.height;
        size.height /= 1.75;
    }
    else if (node.type.endsWith("Exit")) {
        size.width += 2.0 * size.height;
        size.height /= 1.75;
    }
    else if (node.type == "Tasklet") {
        size.width += 2.0 * (size.height / 3.0);
        size.height /= 1.75;
    }
    else if (node.type == "EmptyTasklet") {
        size.width = 0.0;
        size.height = 0.0;
    }
    else if (node.type == "Reduce") {
        size.width *= 2;
        size.height = size.width / 3.0;
    }
    else {
    }

    return size
}



function calculateBoundingBox(g) {
    // iterate over all objects, calculate the size of the bounding box
    let bb = {};
    bb.width = 0;
    bb.height = 0;

    g.nodes().forEach(function (v) {
        let x = g.node(v).x + g.node(v).width / 2.0;
        let y = g.node(v).y + g.node(v).height / 2.0;
        if (x > bb.width) bb.width = x;
        if (y > bb.height) bb.height = y;
    });

    return bb;
}

function calculateEdgeBoundingBox(edge) {
    // iterate over all points, calculate the size of the bounding box
    let bb = {};
    bb.x1 = edge.points[0].x;
    bb.y1 = edge.points[0].y;
    bb.x2 = edge.points[0].x;
    bb.y2 = edge.points[0].y;

    edge.points.forEach(function (p) {
        bb.x1 = p.x < bb.x1 ? p.x : bb.x1;
        bb.y1 = p.y < bb.y1 ? p.y : bb.y1;
        bb.x2 = p.x > bb.x2 ? p.x : bb.x2;
        bb.y2 = p.y > bb.y2 ? p.y : bb.y2;
    });

    bb = {'x': bb.x1, 'y': bb.y1, 'width': (bb.x2 - bb.x1),
          'height': (bb.y2 - bb.y1)};
    if (bb.width <= 5) {
        bb.width = 10;
        bb.x -= 5;
    }
    if (bb.height <= 5) {
        bb.height = 10;
        bb.y -= 5;
    }
    return bb;
}


function paint_state(g, drawnodestate) {
    g.nodes().forEach(function (v) {
        drawnodestate.draw_node(g.node(v), v);
    });
    g.edges().forEach(function (e) {
        let edge = g.edge(e);
        ObjectHelper.assert("edge invalid", edge);
        drawnodestate.draw_edge(g.edge(e), e.name);
    });
}

function create_global() {
    window.global_state = new SdfgState();
    return window.global_state;
}

function create_local() {
    return new SdfgState();
}

export { create_global, create_local, SdfgState, message_handler};
