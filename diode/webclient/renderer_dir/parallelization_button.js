import { Button, Layout, Pos, max_func, max_func_obj, min_func, DataBlock,
    RU_DataViewText, RU_DataViewSuggestedActionBlock, RU_DataViewNumberBlock, RU_DataViewBarGraph, RU_DataViewFormLayout, } from "./renderer_util.js";
import { MathHelper, ObjectHelper } from "./datahelper.js";


class ParallelizationButton extends Button {
    constructor(ctx, targetsection_analysis, all_analyses, critical_path_analysis, communicator) {
        super(ctx);
        this.communicator = communicator;

        this._display_image = {
            "1": "data:image/png;base64,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CAgEBAICAYGAQACBgEBAICAQEAgIBAQCAgGBgEAAgYBAQCAgEBAICAQEAgIBgYBAAIGAQEAgIBAQCAgEBAICAYEAAgGBgEBAICAQEAgIBAQCAgGBAAIBgYBAQCAgEBAICAQEAgIBBAICAYGAQEAgIBAQCAgEBAICAQQCAgGBgEBAICAQEAgIBAQCCAQEAgIBgYBAQCAgEBAICAQEAggEBAICAYGAQEAgIBAQCAgEEAgIBAQCAgGBgEBAICAQEAgIBBAIzOj8AK7x3IUsj+oSBRLD0pJLxEssEAgIBAQCAgGBgEBAIIBAQCAgEBAICAQEAgIBgYBAQCCAQEAgIBAQCAgEBAICAYGAQACBgEBAICAQEAgIBAQCAgGBgEAAgYBAQCAgEBAICAQEAgIBgQACAYGAQEAgIBAQCAgEBAICAYEAAgGBgEBAICAQEAgIBAQCAgEEAgIBgYBAQCAgEBAICAQEAgIBBAICAYGAQEAgIBAQCAgEBAIIBAQCAgGBgEBAICAQEAgIBAQCCAQEAgIBgYBAQCAgEBAICAQEAggEBAICAYGAQEAgIBAQCAgEEAgIBAQCAgGBgEBAICAQEAgIBBAICAQEAgIBgYBAQCAgEBAIIBAQCAgEBAICAYGAQEAgIBAQCCAQEAgIBAQCAoFy/QOsBSVilqJCrgAAAABJRU5ErkJggg==",
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
                    { title: "Serial Path (" + descstr_1 + ")", value: ObjectHelper.valueToSensibleString(path_1_thread) + " cycles" },
                    { title: "Critical Path (" + descstr_max + ")", value: ObjectHelper.valueToSensibleString(path_max_thread) + " cycles" }
                ],
                padding: {left: 10, right: 10, top: 0, bottom: 0 },
                rawdata: d
            };
        }));
        layout.setRect("Imbalance", new Pos(70, 50), new Pos(30, 20), new RU_DataViewNumberBlock().setTitle("Imbalance").setDataAnalysisFunction(x => {
            let balance_max = x.data.balance_max * 100.0;
            let p = Math.round(balance_max);
            return p;
        }).setColorScaling(x => Math.min(Math.pow(x, 2.) / 10, 100.)));

        let suggested_action = new RU_DataViewSuggestedActionBlock().setDataAnalysisFunction(x => {
            
            let returnarr = [];
            let data = x.data;

            let speedup = data.speedup;
            let max_thread_num = max_func(speedup, y => y.thread_num);
            let min_thread_num = min_func(speedup, y => y.thread_num);
            
            let max_speedup = 0;
            let min_speedup = 0;
            if(toplevel_use_mean)
                min_speedup = MathHelper.mean(speedup.find(y => y.thread_num == min_thread_num).value);
            else if(toplevel_use_median)
                min_speedup = MathHelper.median(speedup.find(y => y.thread_num == min_thread_num).value);
            else ObjectHelper.assert("Unknown mode", false);
            
            if(toplevel_use_mean)
                max_speedup = MathHelper.mean(speedup.find(y => y.thread_num == max_thread_num).value);
            else if(toplevel_use_median)
                max_speedup = MathHelper.median(speedup.find(y => y.thread_num == max_thread_num).value);
            else ObjectHelper.assert("Unknown mode", false);


            if(Math.abs(max_speedup - 1.0) < 0.1) {
                returnarr.push("SpeedupHigherThanOne");
            }

            
            return returnarr;
        }).linkMouse(layout._layout_clickable);
        suggested_action.setHint("SpeedupHigherThanOne", "MapTiling");

        layout.setRect("SuggestedAction", new Pos(70, 70), new Pos(30, 20), suggested_action);

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
                tcs = x.data.map(x => x.data.cycles_per_thread);
            }

            let colors = RU_DataViewBarGraph.colorList().slice(0, tcs.length + 1);

            let datasets = [];
            // So now we have a mapping of thread -> cycles.
            
            if(all_analyses_global) {
                let chunksize = Math.round(tcs.length / all_analyses.repcount);

                // Just reset and it should be fine already
                tcs = ObjectHelper.createChunks(tcs, chunksize, MathHelper.sumArray);
            }

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

        databinding["Imbalance"] = targetsection_analysis;
        databinding["SuggestedAction"] = critical_path_analysis;
        databinding["Efficiency"] = critical_path_analysis;
        databinding["Title"] = new DataBlock({ fontsize: 32, text: "Parallelization results", color: "black", align: "center" }, "Text");

        databinding['Graph'] = [all_analyses, critical_path_analysis];
        databinding['PathInfo'] = critical_path_analysis;

        this.dataparams = [targetsection_analysis, all_analyses, critical_path_analysis];
        layout.setDataBinding(databinding);


        this.button_subwindow.setLayout(layout);

        this.setOnEnterHover(p => { this.color = "#00FF00"; this.button_subwindow_state = 'open'; })
        this.setOnLeaveHover(p => { this.color = "orange"; if (!this.is_locked_open) this.button_subwindow_state = 'collapsed'; })
        this.setOnClick((p, mb) => { this.is_locked_open = !this.is_locked_open; });

        this.setDefaultDblClick();
    }
}

export { ParallelizationButton };