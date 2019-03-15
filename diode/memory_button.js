


class MemoryButton extends Button {
    constructor(ctx, all_mem_analyses, target_bw) {
        super(ctx);

        ObjectHelper.logObject("target_bw", target_bw);
        this.Memory_Target_Bandwidth = new Number(target_bw);
        ObjectHelper.logObject("Memory_Target_Bandwidth", this.Memory_Target_Bandwidth);

        this._display_image = {
            "1": "data:image/png;base64,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",
            "-1": "data:image/png;base64,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IAABCEAAAhCAAAQgAAEIQABCAAEIAQQgBBCAEEAAQgABCEAAAhCAAAQgAAEIQAACEIAABCAAAQgBBCAEEIAQQABCAAEIQAACEIAABCAAAQhAAAIQgAAEIAABCAEEIAQQgBBAXAPCAjpxigV0WIKNVogl2ABCAAEIAQQgBBCAEEAAAhCAAAQgAAEIQAACEIAABCAAAQhAAEIAAQgBBCAEEIAQQABCAAEIQAACEIAABCAAAQhAAAIQgAAEIAABCAEEIAQQgBBAAEIAAQhAABJLILxZMU7xZkXezUsrxLt5AUIAAQgBBCAEEIAQQAACEIAABCAAAQhAAAIQgAAEIAABCEAAQgABCAEEIAQQgBBAAEIAAQhAAAIQgAAEIAABCEAAAhCAAAQgAAEIAQQgBBCAEEAAQgABCEAAAhCAAAQgAAEIQAACEIAABCAAAQgBBCAEEIAQQFwDwgI6cYoFdFiCjVaIJdgAQgABCAEEIAQQgBBAAAIQgAAEIAABCEAAAhCAAAQgAAEIQABCAAEIAQQgBBCAEEAAQgABCEAAAhCAAAQgAAEIQAACEAIIQMj1vrFnzze/Reb75p493+DrR0RERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERESUmP4/PVupnBEseEQAAAAASUVORK5CYII="
        };

        let databinding = {};

        let judgements = all_mem_analyses.data.map(x => x.judgement);
        let majority = MathHelper.majority(judgements);
        this.setButtonImage(this._display_image[majority]);

        this.dataparams = [all_mem_analyses, target_bw];

        let layout = new Layout(this.button_subwindow);
        layout.setRect("Title", new Pos(0, 0), new Pos(100, 10), new RU_DataViewText());
        layout.setRect("Balance", new Pos(70, 60), new Pos(30, 20), new RU_DataViewNumberBlock().setTitle("Balance").setOptions({
            display_title: false,
            text_align: "left",
            draw_bar: ["left"],
            padding: { left: 10 },
        }).setDataAnalysisFunction(d => {
            let x = d;
          
            let expected_bandwidth = x.data.map(x => x.data.expected_bandwidth);
            let target_bw = x.data[0].data.Memory_Target_Bandwidth;

            let ret = undefined;

            if(toplevel_use_mean)
                ret = Math.round(MathHelper.mean(expected_bandwidth) * 100. / target_bw);
            else if(toplevel_use_median)
                ret = Math.round(MathHelper.median(expected_bandwidth) * 100. / target_bw);
            else ObjectHelper.assert("Undefined mode", false);

            ObjectHelper.assert("is number", ret instanceof Number || typeof ret == "number");

            return ret;
        }).setColorScaling(x => 100 - x));
        layout.setRect("Bandwidth", new Pos(0, 60), new Pos(70, 20), new RU_DataViewFormLayout().setTitle("Bandwidth").setDataAnalysisFunction(d => {
            let x = d;

            let expected_bandwidth = x.data.map(x => x.data.expected_bandwidth);

            //#TODO: For B/s, we should try to use PAPI_REF_CYC. It seems to 
            //       vary between chips.

            let val = {};

            if(toplevel_use_mean) {
                val = { title: "Bandwidth", value: MathHelper.mean(expected_bandwidth).toFixed(3).toString() + " B/c" };
            }
            else if(toplevel_use_median) {
                val = { title: "Bandwidth", value: MathHelper.median(expected_bandwidth).toFixed(3).toString() + " B/c" };
            }
            else ObjectHelper.assert("Undefined mode", false);
            return {
                fontsize: 16,
                rows: [
                    val,
                    { title: "Target Bandwidth", value: this.Memory_Target_Bandwidth.toFixed(3).toString() + " B/c" } 
                ],
                padding: {left: 10, right: 10, top: 0, bottom: 0 },
                rawdata: d
            };
        }));

        layout.setRect("Graph", new Pos(0, 10), new Pos(70, 50), new RU_DataViewBarGraph(
            {
                type: "horizontalBar",
                xAxes: [{
                    type: 'linear',
                    display: false,
                    position: 'bottom',
                    id: 'data-axis',
                    gridLines: {
                        display: false
                    },
                    ticks: { 
                        beginAtZero: true
                    },
                    scaleLabel: { labelString: "Misses", display: true }
                }, {
                    type: 'linear',
                    display: false,
                    ticks: {
                        max: 1,
                        min: -1,
                    },
                    id: 'corr-axis',
                    position: 'top',
                    gridLines: {
                        display: false
                    },
                    scaleLabel: { labelString: "Correlation", display: true }
                }
            ],
                yAxes: [{
                    display: false,
                    position: 'left',
                    id: 'thread-axis',
                    gridLines: {
                        display: true
                    },
                    scaleLabel: { labelString: "Thread", display: true }
                }
            ],
        }).setDataAnalysisFunction(x => {
            if (x == null) return ;
            let l2misses = x.data.map(x => x.data.L2_TCM);
            let l3misses = x.data.map(x => x.data.L3_TCM);
            let tot_cyc = x.data.map(x => x.data.TOT_CYC);
            let bw_mem = x.data.map(x => x.data.mem_bandwidth);
            let bw_l3 = x.data.map(x => x.data.l3_bandwidth);

            console.log("l2 miss length: " + l2misses.length);

            let colors = RU_DataViewBarGraph.colorList().slice(0, l2misses[0].length + 1 + 2);

            let datasets = [];

            ObjectHelper.logObject("l2misses", l2misses);
            let z1 = MathHelper.zip(l2misses, tot_cyc);
            ObjectHelper.logObject("z1", z1);

            let l2corr = z1.map(x => MathHelper.sample_corr(x[0], x[1]));
            let l3corr = MathHelper.zip(l3misses, tot_cyc).map(x => MathHelper.sample_corr(x[0], x[1]));

            ObjectHelper.logObject("l2corr", l2corr);
            ObjectHelper.logObject("l3corr", l3corr);

            // Now we average the correlations
            let avg_l2_corr = MathHelper.mean(l2corr);
            let avg_l3_corr = MathHelper.mean(l3corr);

            let i = 0;
            for (let tcm of [l2misses, l3misses]) {
                let labelstr = "[Unknown]";
                if(tcm == l2misses) labelstr = "L2_TCM";
                if(tcm == l3misses) labelstr = "L3_TCM";

                let tmp = 0;

                if(toplevel_use_mean)
                    tmp = MathHelper.zip2d(tcm).map(x => MathHelper.mean(x));
                else if(toplevel_use_median)
                    tmp = MathHelper.zip2d(tcm).map(x => MathHelper.median(x));
                else ObjectHelper.assert("Undefined mode", false);
                ObjectHelper.logObject("tmp", tmp);

                datasets.push({ label: labelstr, xAxisID: "data-axis", yAxisID: "thread-axis", data: tmp, backgroundColor: colors[i] });
                i++;
            }

            let rho_col1 = colors[i];
            let rho_col2 = colors[i + 1];
            datasets.push({ label: "L2 corr.", xAxisID: "corr-axis", data: l2corr, backgroundColor: colors[i], hidden: true });
            datasets.push({ label: "L3 corr.", xAxisID: "corr-axis", data: l3corr, backgroundColor: colors[i + 1], hidden: true });

            let chartData = {
                labels: [...(Array(l2misses[0].length).keys())].map(x => "Thread " + x.toString()),
                "datasets": datasets,

            };


            return chartData;
        }).linkMouse(this.button_subwindow).changeGraphOptions(x => {
            x.options.title.text = "Cache misses (mean over all runs)";
            x.options.title.display = false;
            x.options.legend = {
                position: 'top',
                display: true
            };
        }));

        
        databinding["Balance"] = all_mem_analyses;
        databinding["Title"] = new DataBlock({ fontsize: 32, text: "Memory performance", color: "black", align: "center" }, "Text");
        databinding['Graph'] = all_mem_analyses;
        databinding['Bandwidth'] = all_mem_analyses;

        layout.setDataBinding(databinding);


        this.button_subwindow.setLayout(layout);

        this.setOnEnterHover(p => { this.color = "#FF0000"; this.button_subwindow_state = 'open'; })
        this.setOnLeaveHover(p => { this.color = "orange"; if (!this.is_locked_open) this.button_subwindow_state = 'collapsed'; })
        this.setOnClick((p, mb) => { this.is_locked_open = !this.is_locked_open; });
    }



}