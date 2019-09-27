import { Button, Layout, Pos, RU_DataViewText, RU_DataViewFormLayout, RU_DataViewNumberBlock, RU_DataViewBarGraph, DataBlock } from "./renderer_util.js";
import { MathHelper, ObjectHelper } from "./datahelper.js";


class MemoryOpButton extends Button {
    constructor(ctx, supersection_all_vec_analyses, path_analysis) {
        super(ctx);

        this.supersection_all_vec_analyses = supersection_all_vec_analyses;

        const baseimgstr = "data:image/png;base64,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";
        this._display_image = {
            "1": baseimgstr,
            "-1": baseimgstr
        };

        let databinding = {};

        // This is not really easy to judge because of machine-dependent measures
        // TODO: Find a way to create a judgement value
        let judgements = [1];
        let majority = MathHelper.majority(judgements);
        this.setButtonImage(this._display_image[majority]);

        this.dataparams = [supersection_all_vec_analyses, path_analysis]; // Fill in parameters

        let layout = new Layout(this.button_subwindow);
        layout.setRect("Title", new Pos(0, 0), new Pos(100, 10), new RU_DataViewText());

        
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
            
            let write_stall = [];
            let load_ins = [];
            let store_ins = [];
            if (x != null) {
                write_stall = x.data.map(x => x.data.write_stall);
                load_ins = x.data.map(x => x.data.load_ins);
                store_ins = x.data.map(x => x.data.store_ins);
            }


            let colors = RU_DataViewBarGraph.colorList().slice(0, write_stall.length + 1);

            let datasets = [];
            // So now we have a mapping of thread -> cycles.

            let repcount = supersection_all_vec_analyses.repcount;
            if(all_analyses_global) {
                // We need to group and add
                let chunksize = write_stall.length / repcount;

                write_stall = ObjectHelper.createChunks(write_stall, chunksize, MathHelper.sumArray);
                load_ins = ObjectHelper.createChunks(load_ins, chunksize, MathHelper.sumArray);
                store_ins = ObjectHelper.createChunks(store_ins, chunksize, MathHelper.sumArray);
            }


            // All runs should be identical, so we take the first element from each set
            datasets.push({ label: "Cycles stalled on writes", yAxisID: "axis-1", data: write_stall[0], backgroundColor: colors[0] });
            datasets.push({ label: "Load ins", yAxisID: "axis-1", data: load_ins[0], backgroundColor: colors[1] });
            datasets.push({ label: "Store ins", yAxisID: "axis-1", data: store_ins[0], backgroundColor: colors[2] });

            let chartData = {
                labels: [...Array(write_stall[0].length).keys()],
                "datasets": datasets,

            };
            return chartData;
        }).linkMouse(layout._layout_clickable).changeGraphOptions(x => {
            x.options.title.text = "Memory operations per thread";
            x.options.scales.yAxes.find(x => x.id == 'axis-1').scaleLabel = { labelString: "Operations", display: true };
            x.options.scales.yAxes.find(x => x.id == 'axis-1').ticks.beginAtZero = true;
            x.options.scales.xAxes = [{ scaleLabel: { labelString: "Thread", display: true } }];
        });

        layout.setRect("Graph", new Pos(0, 10), new Pos(70, 50), thread_graph);

        

        databinding["Title"] = new DataBlock({ fontsize: 32, text: "Memory operations", color: "black", align: "center" }, "Text");
        databinding['Graph'] = supersection_all_vec_analyses;

        layout.setDataBinding(databinding);
        this.button_subwindow.setLayout(layout);

        this.setOnEnterHover(p => { this.color = "#FF0000"; this.button_subwindow_state = 'open'; })
        this.setOnLeaveHover(p => { this.color = "orange"; if (!this.is_locked_open) this.button_subwindow_state = 'collapsed'; })
        this.setOnClick((p, mb) => { this.is_locked_open = !this.is_locked_open; });

        this.setDefaultDblClick();
    }



}
// Specialized version of the class in datahelper
class SuperSectionMemoryOpAnalysis {
    constructor(section, nodeid, stateid, critical_path_analysis) {
        this.section = section;
        this.critical_path_analysis = critical_path_analysis;
        this.for_node = nodeid;
        this.for_state = stateid;
        this.analysis_result = null;

        if(!(stateid == 0xFFFF || stateid == 65535))
            ObjectHelper.assert("for_node defined", this.for_node != undefined && new Number(this.for_node) != NaN);
    }

    judgement(analysis = null) {
        if (analysis == null) analysis = this.analysis_result;

        // TODO: Find a good judgement variable
        return -1;
    }

    analyze() {
        ObjectHelper.assert("Not available over old interface", false);
    }
}

class LazySuperSectionMemoryOpAnalysis extends SuperSectionMemoryOpAnalysis {

    constructor(communicator, section, nodeid, stateid, critical_path_analysis) {
        super(section, nodeid, stateid, critical_path_analysis, null);
        this.communicator = communicator;
    }

    async analyze() {
        // We differ from the eager analysis here: We let the python/sql-side do the hard work

        // Project to a section from the supersection.
        let section = this.section.toSection(this.for_node, this.for_state);
        section = await section;
        if(section === undefined) {
            return undefined;
        }
        let tmp = await this.communicator.runAnalysis("MemoryOpAnalysis", [new Number(section.unified_id), new Number(section.supersection_id)]).get();

        let data = tmp;

        let ret = new DataBlock(data, "MemoryOp");
        this.analysis_result = ret;
        ret.judgement = this.judgement();
        return ret;
    }
}

function AutoSuperSectionMemoryOpAnalysis(communicator, section, nodeid, stateid, critical_path_analysis) {
    if(section instanceof LazySuperSection) {
        return new LazySuperSectionMemoryOpAnalysis(communicator, section, nodeid, stateid);
    }
    else {
        ObjectHelper.assert("Do not use", false);
        return new SuperSectionMemoryOpAnalysis(section, nodeid, stateid, critical_path_analysis);
    }
}

export { MemoryOpButton, AutoSuperSectionMemoryOpAnalysis };