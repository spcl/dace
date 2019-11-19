import { Button, Layout, Pos, RU_DataViewText, RU_DataViewFormLayout, RU_DataViewNumberBlock, RU_DataViewBarGraph, DataBlock } from "./renderer_util.js";
import { MathHelper, ObjectHelper } from "./datahelper.js";



class MemoryButton extends Button {
    constructor(ctx, all_mem_analyses, target_bw) {
        super(ctx);

        ObjectHelper.assert("Valid bandwidth", target_bw > 0.0);
        this.Memory_Target_Bandwidth = new Number(target_bw);

        this._display_image = {
            "1": "data:image/png;base64,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tftm2QMEEEBy6uWyBwgggGTViZIHCCCA5HWUNVHuAAEEkLxad7VPH5Pd7jQEBJCc2tznUdaeUUAAaUM7+zrKOjYKCCDtaH8fPi6NAQJIWzpe3gABBJAMj7LOlzZAAKki1x/U3NorZQ0Q1x8kkQt0ej3KulHSAHGBThKNeOR7bMfNcgYIIIDk2b5yBggggGTasTIGCCCAZHuUdS7m4+IYIIC0sfHQUdaNDaOAANLKNkYuRtg1CgggLW377EdZR0cBAaS1zXrXwYUxQABpcd8acIAAAkjWfWXAAQIIIHn/UtaAAwQQQPLuzGADBBBA8m7bYAMEEEAy7/xAAwQQQDJvw2QXH0dGAQFEo199uI9vjgECiD5s/8P+//TzUR+AAJL9z5AHhZwaHwUEEH3cpvt+831ybw//MCCA5N/uz3xk76sH144CAojuPc16+eTEtasTZw/tXNPbPwgIIAIEEAECiAABRIAAAggggAACCCCAAAIIIIAAAggggDTyS/KcApIREBfoZJQLdMrPFWw55Qo2QAQIIAIEEAECiAABBBBAAAEEEEAAAQQQQAABBBBAAAFEgAAiQAARIIAIEEAECCCAAAIIIIAAAggggAACCCCAAAIIIAIEEAECiAABRIAAAggggAACCCCAAAIIIIAAAggggAAiQKrL9QcZ5fqDBHKBTn25QCeBRjyngAACCCCAAAIIIIAAAggggAACCCCAACJAABEggAgQQAQIIIAAAggggAACCCCAAAIIIIAAAgggAgQQAQKIAAFEgAAiQAABBBBAAAEEEEAAAQQQQAABBBBAABEggAgQQAQIIAIEEEAAASTyJXlOAckIiAt0MsoFOuXnCraccgUbIAIEEAECiAABRIAAAggggAACCCCAAAIIIIAAAggggAAiQAARIIAIEEAECCACBBBAAAEEEEAAAQQQQAABBBBAAAEEEAECiAABRIAAIkAAAQQQQAABBBBAAAEEEEAAAQQQQAARINXl+oOMcv1BArlAp75coJNAI55TQAABBBBAAAEEEEAAAQQQQAABBBBAABEggAgQQAQIIAIEEEAAAQQQQAABBBBAAAEEEEAAAQQQAQKIAAFEgAAiQAARIIAAAggggAACCCCAAAIIIIAAAggggAgQQAQIIAIEEAECCCCAABL5kjyngGQExAU6gADyiFzBBggggAACCCCAAAKIAAFEgAAiQAARIIAAAggggAACCCCAAAIIIIAAAggggOgzvVn2b55+AAggOVX29/J9QAABBBBAAAEEEAECiAABRIAAIkAAESCAAAIIIIAAAggggAACCCCAAAIIIIAIEEAECCACBBABkhYQ1x9klL9RmEAu0Kkvfyc9gUY8p4AAAggggAACCCCAAAIIIIAAAggggAAiQAARIIAIEEAECCCAAAIIIIAAAggggAACCCCAAAIIIPq0EUAAESCACBBABAggAgQQQAABBBBAAAEEEEAAAQQQQAABBBABktiX5DmtrccAKT0fXp1R7/rw6tJz/UFGzbj+ABABAogAAUSAACJAAAEEEEAAAQQQQAABBBBAAAEEEEAAESCACBBABAggAgQQAQIIIIAAAggggAACCCCAAAIIIIAAAogAAUSAACJAABEggAACCCCAAAIIIIAAAggggAACCCCACJDqcv1BRrn+IIFcoFNfLtBJoBHPaW25gg0QAQKIAAFEgAAiQAABBBBAAAEEEEAAAQQQQAABBBBAABEggAgQQAQIIAIEEAECCCCAAAIIIIAAAggggAACCCCAAAKIAAFEgAAiQAARIIAAAggggAACCCCAAFLll+Q5rS2fzVt+Pt09o3y6e/m5HySj3A8CiAABRIAAIkAAESCAAAIIIIAAAggggAACCCCAAAIIIIAIEEAECCACBBABAogAAQQQQAABBBBAAAEEEEAAAQQQQAABRIAAIkAAESCACBBAAAEEEEAAAQQQQAABBBBAAAEEEEAESIW5/iCjXH+QQC7QqS8X6CTQiOe0tlzBBogAAUSAACJAABEggAACCCCAAAIIIIAAAggggAACCCCACBBABAggAgQQAQKIAAEEEEAAAQQQQAABBBBAAAEEEEAAAUSAACJAABEggAgQQAABBJDYl+Q5rS2frFh+Pps3o3w2b/n5dPeM8unugAgQQAQIIAIEEAECCCCAAAIIIIAAAggggAACCCCAAAKIAAFEgAAiQAARIIAIEEAAAQQQQAABBBBAAAEEEEAAAQQQQAQIIAIEEAECiAABBBBAAAEEEEAAAQQQQAABBBBAAAFEgFSY6w8yyvUHCeQCnfpygU4CjXhOa8sVbIAIEEAECCACBBABAggggAACCCCAAAIIIIAAAggggAACiAABRIAAIkAAESCACBBAAAEEEEAAAQQQQAABBBBAAAEEkKb39MjI81/S8Ht+ZORpQKQ2BYgEiASIBIgEiASIBIgEiASIBIgkQCRAJEAkQCRAJEAkQCRAJEAkQCQBIgEiASIBIgEiASIBIgEiASIJEAkQCRAJEAkQCRAJEAkQCRAJEEmASIBIgEiASIBIgEiASIBIgEgCRAJEAkQCRAJEAkQCRAJEAkQCRBIgEiASIBIgEiASIBIgEiASIJIAkQCRAJEAkQCRAJEAkQCRAJEAkQSIBIgEiASIBIgEiASIBIgEiCRAJEAkQCRAJEAkQCRAJEAkQCRAJAEiASIBIgEiASIBIgEiASIBIrW8/wGZ0JDqvSQHhgAAAABJRU5ErkJggg==",
            "-1": "data:image/png;base64,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"
        };

        let databinding = {};

        let judgements = all_mem_analyses.data.map(x => x.judgement);
        let majority = MathHelper.majority(judgements);
        this.setButtonImage(this._display_image[majority]);

        this.dataparams = [all_mem_analyses, target_bw];

        let repcount = all_mem_analyses.repcount;


        let layout = new Layout(this.button_subwindow);
        layout.setRect("Title", new Pos(0, 0), new Pos(100, 10), new RU_DataViewText());
        layout.setRect("PercentBandwidth", new Pos(70, 60), new Pos(30, 20), new RU_DataViewNumberBlock().setTitle("PercentBandwidth").setOptions({
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
        let transthis = this;
        ObjectHelper.assert("Valid bandwidth", this.Memory_Target_Bandwidth > 0.0);
        layout.setRect("Bandwidth", new Pos(0, 60), new Pos(70, 20), new RU_DataViewFormLayout().setTitle("Bandwidth").setDataAnalysisFunction(d => {
            let x = d;

            let expected_bandwidth = x.data.map(x => x.data.expected_bandwidth);
            let datasize = x.data.map(x => x.data.datasize);
            let input_datasize = x.data.map(x => x.data.input_datasize);

            let bytes_from_l3 = x.data.map(x => x.data.bytes_from_l3).map(x => MathHelper.sum(x));
            let bytes_from_mem = x.data.map(x => x.data.bytes_from_mem).map(x => MathHelper.sum(x));


            // Differentiate between global and local analysis here
            let repcount = all_mem_analyses.repcount;
            if(all_analyses_global) {
                let chunksize = datasize.length / repcount;
                datasize = ObjectHelper.createChunks(datasize, chunksize, MathHelper.sum);
                input_datasize = ObjectHelper.createChunks(input_datasize, chunksize, MathHelper.sum);
                datasize = ObjectHelper.createChunks(datasize, chunksize, MathHelper.sum);

                bytes_from_l3 = ObjectHelper.createChunks(bytes_from_l3, chunksize, MathHelper.sum);
                bytes_from_mem = ObjectHelper.createChunks(bytes_from_mem, chunksize, MathHelper.sum);
            }

            let val = {};

            if(toplevel_use_mean) {
                val = { title: "Bandwidth", value: ObjectHelper.valueToSensibleString(MathHelper.mean(expected_bandwidth), "programmer", "B/c") };
            }
            else if(toplevel_use_median) {
                val = { title: "Bandwidth", value: ObjectHelper.valueToSensibleString(MathHelper.median(expected_bandwidth), "programmer", "B/c") };
            }
            else ObjectHelper.assert("Undefined mode", false);
            return {
                fontsize: 16,
                rows: [
                    val,
                    { title: "Target Bandwidth", value: ObjectHelper.valueToSensibleString(transthis.Memory_Target_Bandwidth, "programmer", "B/c") },
                    { title: "Bytes processed (SA)", value: ObjectHelper.valueToSensibleString(MathHelper.median(datasize), "programmer", "B")},
                    { title: "Bytes imported into scope (SA)", value: ObjectHelper.valueToSensibleString(MathHelper.median(input_datasize), "programmer", "B")},
                    { title: "Bytes requested from L3", value: ObjectHelper.valueToSensibleString(MathHelper.median(bytes_from_l3), "programmer", "B")},
                    { title: "Bytes requested from memory", value: ObjectHelper.valueToSensibleString(MathHelper.median(bytes_from_mem), "programmer", "B")},
                    
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
                    display: true,
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

            let colors = RU_DataViewBarGraph.colorList().slice(0, l2misses[0].length + 1 + 2);

            let datasets = [];

            let thread_group_func = z1 => {
                let thread_grouped_z1 = [];
                for(let x of l2misses[0]) {
                    thread_grouped_z1.push([]);
                }
                for(let run_tuple of z1) {
                    let l2ms_per_thread = run_tuple[0];
                    let cycs_per_thread = run_tuple[1];

                    for(let i = 0; i < l2ms_per_thread.length; ++i) {
                        thread_grouped_z1[i].push([l2ms_per_thread[i], cycs_per_thread[i]]);
                    }
                }
                return thread_grouped_z1;
            };

            let z1 = MathHelper.zip(l2misses, tot_cyc);
            let thread_grouped_z1 = thread_group_func(z1);


            let l2corr = thread_grouped_z1.map(x => MathHelper.sample_corr(x[0], x[1]));
            
            let thread_grouped_z2 = thread_group_func(MathHelper.zip(l3misses, tot_cyc));
            let l3corr = thread_grouped_z2.map(x => MathHelper.sample_corr(x[0], x[1]));

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

                datasets.push({ label: labelstr, xAxisID: "data-axis", yAxisID: "thread-axis", data: tmp, backgroundColor: colors[i] });
                i++;
            }

            let rho_col1 = colors[i];
            let rho_col2 = colors[i + 1];

            if(display_memory_correlation) {
                datasets.push({ label: "L2 corr.", xAxisID: "corr-axis",/* yAxisID: "corr-y-axis",*/ data: l2corr, backgroundColor: colors[i], hidden: true });
                datasets.push({ label: "L3 corr.", xAxisID: "corr-axis", /*yAxisID: "corr-y-axis",*/ data: l3corr, backgroundColor: colors[i + 1], hidden: true });
            }

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

        
        databinding["PercentBandwidth"] = all_mem_analyses;
        databinding["Title"] = new DataBlock({ fontsize: 32, text: "Memory performance", color: "black", align: "center" }, "Text");
        databinding['Graph'] = all_mem_analyses;
        databinding['Bandwidth'] = all_mem_analyses;

        layout.setDataBinding(databinding);


        this.button_subwindow.setLayout(layout);

        this.setOnEnterHover(p => { this.color = "#FF0000"; this.button_subwindow_state = 'open'; })
        this.setOnLeaveHover(p => { this.color = "orange"; if (!this.is_locked_open) this.button_subwindow_state = 'collapsed'; })
        this.setOnClick((p, mb) => { this.is_locked_open = !this.is_locked_open; });

        this.setDefaultDblClick();
    }



}

export { MemoryButton };