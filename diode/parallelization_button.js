

class ParallelizationButton extends Button {
    constructor(ctx, targetsection, all_analyses, critical_path_analysis) {
        super(ctx);

        this._display_image = {
            "1": "data:image/png;base64,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",
            "-1": "data:image/png;base64,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"
        };

        this.setButtonImage(this._display_image[critical_path_analysis.judgement]);

        let databinding = {};

        let layout = new Layout(this.button_subwindow);
        layout.setRect("Title", new Pos(0, 0), new Pos(100, 10), new RU_DataViewText());
        layout.setRect("PathInfo", new Pos(0, 20), new Pos(70, 30), new RU_DataViewFormLayout().setTitle("PathInfo").setDataAnalysisFunction(d => {
            let x = d;

            let efficiencies = x.data.efficiency;
            let target_thread_num = max_func(efficiencies, y => y.thread_num);

            let path_1_thread = 0;
            if(toplevel_use_mean)
                path_1_thread = MathHelper.mean(x.data.critical_paths.find(x => x.thread_num == 1).value);
            else if(toplevel_use_median)
                path_1_thread = MathHelper.median(x.data.critical_paths.find(x => x.thread_num == 1).value);
            else ObjectHelper.assert("Undefined mode", false);

            let path_max_thread = 0;
            if(toplevel_use_mean)
                path_max_thread = MathHelper.mean(x.data.critical_paths.find(x => x.thread_num == target_thread_num).value);
            else if(toplevel_use_median)
                path_max_thread = MathHelper.median(x.data.critical_paths.find(x => x.thread_num == target_thread_num).value);
            else ObjectHelper.assert("Undefined mode", false);
            
            let descstr_1 = "unknown";
            let descstr_max = "unknown";
            if(toplevel_use_mean) {
                descstr_1 = "mean";
                descstr_max = "mean";
            }
            else if(toplevel_use_median) {
                descstr_1 = "median";
                descstr_max = "median";
            }

            return {
                fontsize: 16,
                rows: [
                    { title: "Threads", value: target_thread_num },
                    { title: "Serial Path (" + descstr_1 + ")", value: path_1_thread.toFixed(3).toString() + " cycles" },
                    { title: "Critical Path (" + descstr_max + ")", value: path_max_thread.toFixed(3).toString() + " cycles" } // As of yet unknown. 
                ],
                padding: {left: 10, right: 10, top: 0, bottom: 0 },
                rawdata: d
            };
        }));
        layout.setRect("Balance", new Pos(70, 50), new Pos(30, 20), new RU_DataViewNumberBlock().setTitle("Balance").setDataAnalysisFunction(x => {
            let balance_max = x.data.balance_max * 100.0;
            let p = Math.round(balance_max);

            return p;
        }).setColorScaling(x => Math.min(Math.pow(x, 2.) / 10, 100.)));
        layout.setRect("Efficiency", new Pos(70, 20), new Pos(30, 20), new RU_DataViewNumberBlock().setTitle("Efficiency").setOptions({
            draw_bar: ["left"],
            padding: { left: 10 },
            display_title: true,
            text_align: "center",
        }).setDataAnalysisFunction(x => {
            let efficiencies = x.data.efficiency;
            let target_thread_num = max_func(efficiencies, y => y.thread_num);
            
            let efficiency = 0;
            
            if(toplevel_use_mean)
                efficiency = MathHelper.mean(efficiencies.find(y => y.thread_num == target_thread_num).value);
            else if(toplevel_use_median)
                efficiency = MathHelper.median(efficiencies.find(y => y.thread_num == target_thread_num).value);
            else ObjectHelper.assert("Unknown mode", false);

            return Math.round(100. * efficiency);
        }).setColorScaling(x => 100 - x).setInformationFilePath("optimization_hints/efficiency.html"));

        let thread_graph = new RU_DataViewBarGraph({
            type: 'bar',
            yAxes: [{
                type: "linear",
                display: true,
                position: 'left',
                id: 'axis-1'
            }
            ]
        }).setDataAnalysisFunction(x => {
            
            let tcs = [];
            if (x != null) {
                ObjectHelper.logObject("x.data", x.data);
                tcs = x.data.map(x => x.data.cycles_per_thread);
            }

            ObjectHelper.logObject("tcs", tcs);

            let colors = RU_DataViewBarGraph.colorList().slice(0, tcs.length + 1);

            let datasets = [];

            let i = 0;
            for (let tc of tcs) {
                datasets.push({ label: "run " + i.toString(), yAxisID: "axis-1", data: tc, backgroundColor: colors[i] });
                i++;
            }

            let chartData = {
                labels: [...Array(tcs[0].length).keys()],
                "datasets": datasets,

            };

            return chartData;
        }).linkMouse(layout._layout_clickable).changeGraphOptions(x => {
            x.options.title.text = "PAPI_TOT_CYC per thread";
            x.options.scales.yAxes.find(x => x.id == 'axis-1').scaleLabel = { labelString: "Cycles", display: true };
            x.options.scales.yAxes.find(x => x.id == 'axis-1').ticks.beginAtZero = true;
            x.options.scales.xAxes = [{ scaleLabel: { labelString: "Thread", display: true } }];
        });

        let efficiency_graph = new RU_DataViewBarGraph({
            type: 'line',
            yAxes: [{
                type: "linear",
                display: true,
                position: 'left',
                id: 'axis-1'
            }, {
                type: "linear",
                display: true,
                position: 'right',
                id: 'axis-2'
            }
            ]
        }).setDataAnalysisFunction(x => {
            let critical_paths = [];
            if (x != null) {
                critical_paths = x.data.critical_paths;
            }

            let speedup = [];
            if (x != null) {
                speedup = x.data.speedup;
            }

            let efficiency = [];
            if (x != null) {
                efficiency = x.data.efficiency;
            }

            let colors = RU_DataViewBarGraph.colorList().slice(0, 4);

            let datasets = [];

            let graphcp = 0;
            if(toplevel_use_mean) {
                graphcp = critical_paths.map(cp => MathHelper.mean(cp.value));
            }
            else if(toplevel_use_median) {
                graphcp = critical_paths.map(cp => MathHelper.median(cp.value));
            }
            else ObjectHelper.assert("Unknown mode", false);
            ObjectHelper.logObject("gcp", graphcp);

            let i = 0;
            // Add the critical paths
            datasets.push({ label: "Critical path", fill: false, yAxisID: "axis-1", data: graphcp, backgroundColor: colors[0], borderColor: colors[0] });
            
            let agg_func = undefined;
            if(toplevel_use_mean)
                agg_func = x => MathHelper.mean(x);
            else if(toplevel_use_median)
                agg_func = x => MathHelper.median(x);
            else
                ObjectHelper.assert("undefined mode", false);

            // Add the speedup
            datasets.push({ label: "Speedup", fill: false, yAxisID: "axis-2", data: speedup.map(sp => agg_func(sp.value)), backgroundColor: colors[1], borderColor: colors[1] });

            datasets.push({ label: "Efficiency", fill: false, yAxisID: "axis-2", data: efficiency.map(sp => agg_func(sp.value)), backgroundColor: colors[2], borderColor: colors[2] });

            let chartData = {
                labels: critical_paths.map(x => x.thread_num),
                "datasets": datasets,

            };

            return chartData;
        }).linkMouse(layout._layout_clickable).changeGraphOptions(x => {
            x.options.title.text = "Parallel efficiency";
            x.options.scales.yAxes.find(x => x.id == 'axis-1').scaleLabel = { labelString: "Cycles", display: true };
            x.options.scales.yAxes.find(x => x.id == 'axis-1').ticks.beginAtZero = true;
            x.options.scales.yAxes.find(x => x.id == 'axis-2').scaleLabel = { labelString: "Relative Perf.", display: true };
            x.options.scales.yAxes.find(x => x.id == 'axis-2').ticks.beginAtZero = true;
            x.options.scales.xAxes = [{ scaleLabel: { labelString: "OMP_NUM_THREADS", display: true } }];
        }).setInformationFilePath("optimization_hints/efficiency.html");


        layout.setMultiviewRect("Graph", new Pos(0, 50), new Pos(70, 50), [thread_graph, efficiency_graph]);
        // Otherwise, we have a section that we should still process
        let section = new Section(targetsection);


        let ta = new ThreadAnalysis(section);
        let db = ta.analyze();


        databinding["Balance"] = db;
        databinding["Efficiency"] = critical_path_analysis;
        databinding["Title"] = new DataBlock({ fontsize: 32, text: "Parallelization results", color: "black", align: "center" }, "Text");
        databinding['Graph'] = [all_analyses, critical_path_analysis];
        databinding['PathInfo'] = critical_path_analysis;

        this.dataparams = [targetsection, all_analyses, critical_path_analysis];
        layout.setDataBinding(databinding);


        this.button_subwindow.setLayout(layout);

        this.setOnEnterHover(p => { this.color = "#00FF00"; this.button_subwindow_state = 'open'; })
        this.setOnLeaveHover(p => { this.color = "orange"; if (!this.is_locked_open) this.button_subwindow_state = 'collapsed'; })
        this.setOnClick((p, mb) => { this.is_locked_open = !this.is_locked_open; });
    }
}
