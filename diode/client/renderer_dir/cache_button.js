import { Button, Layout, Pos, RU_DataViewText, RU_DataViewFormLayout, RU_DataViewNumberBlock, RU_DataViewBarGraph, DataBlock } from "./renderer_util.js";
import { MathHelper, ObjectHelper } from "./datahelper.js";


class CacheOpButton extends Button {
    constructor(ctx, supersection_all_vec_analyses, path_analysis) {
        super(ctx);

        this.supersection_all_vec_analyses = supersection_all_vec_analyses;

        const baseimgstr = "data:image/png;base64,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";
        this._display_image = {
            "1": baseimgstr,
            "-1": baseimgstr
        };

        let databinding = {};

        // TODO: Since the results are hard to analyze, try to find a better way to visualze
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
            
            let cache_snoop = [];
            let cache_shr2ex = [];
            let cache_cln2ex = [];
            let cache_intervention = [];
            if (x != null) {
                cache_snoop = x.data.map(x => x.data.cache_snoop);
                cache_shr2ex = x.data.map(x => x.data.cache_shr2ex);
                cache_cln2ex = x.data.map(x => x.data.cache_cln2ex);
                cache_intervention = x.data.map(x => x.data.cache_intervention);
            }


            let colors = RU_DataViewBarGraph.colorList().slice(0, cache_snoop.length + 1);

            let datasets = [];
            // mapping of thread -> cycles.

            let repcount = supersection_all_vec_analyses.repcount;
            if(all_analyses_global) {
                // We need to group and add
                let chunksize = cache_snoop.length / repcount;

                cache_snoop = ObjectHelper.createChunks(cache_snoop, chunksize, MathHelper.sumArray);
                cache_shr2ex = ObjectHelper.createChunks(cache_shr2ex, chunksize, MathHelper.sumArray);
                cache_cln2ex = ObjectHelper.createChunks(cache_cln2ex, chunksize, MathHelper.sumArray);
                cache_intervention = ObjectHelper.createChunks(cache_intervention, chunksize, MathHelper.sumArray);
            }

            // We only take the first element from each set since they should all be equivalent
            datasets.push({ label: "Snoop", yAxisID: "axis-1", data: cache_snoop[0], backgroundColor: colors[0] });
            datasets.push({ label: "Shared to Exclusive", yAxisID: "axis-1", data: cache_shr2ex[0], backgroundColor: colors[1] });
            datasets.push({ label: "Clean to Exclusive", yAxisID: "axis-1", data: cache_cln2ex[0], backgroundColor: colors[2] });
            datasets.push({ label: "Intervention", yAxisID: "axis-1", data: cache_intervention[0], backgroundColor: colors[4] });

            let chartData = {
                labels: [...Array(cache_snoop[0].length).keys()],
                "datasets": datasets,

            };

            return chartData;
        }).linkMouse(layout._layout_clickable).changeGraphOptions(x => {
            x.options.title.text = "Cache operations per thread";
            x.options.scales.yAxes.find(x => x.id == 'axis-1').scaleLabel = { labelString: "Operations", display: true };
            x.options.scales.yAxes.find(x => x.id == 'axis-1').ticks.beginAtZero = true;
            x.options.scales.xAxes = [{ scaleLabel: { labelString: "Thread", display: true } }];
        });

        layout.setRect("Graph", new Pos(0, 10), new Pos(70, 50), thread_graph);

        

        databinding["Title"] = new DataBlock({ fontsize: 32, text: "Cache operations", color: "black", align: "center" }, "Text");
        databinding['Graph'] = supersection_all_vec_analyses;

        layout.setDataBinding(databinding);
        this.button_subwindow.setLayout(layout);

        this.setOnEnterHover(p => { this.color = "#FF0000"; this.button_subwindow_state = 'open'; })
        this.setOnLeaveHover(p => { this.color = "orange"; if (!this.is_locked_open) this.button_subwindow_state = 'collapsed'; })
        this.setOnClick((p, mb) => { this.is_locked_open = !this.is_locked_open; });

        this.setDefaultDblClick();
    }



}
// Specialized version of class in datahelper.
class SuperSectionCacheOpAnalysis {
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

class LazySuperSectionCacheOpAnalysis extends SuperSectionCacheOpAnalysis {

    constructor(communicator, section, nodeid, stateid, critical_path_analysis) {
        super(section, nodeid, stateid, critical_path_analysis, null);
        this.communicator = communicator;
    }

    async analyze() {
        // We differ from the eager analysis here: We let the python/sql-side do the hard work
        let section = this.section.toSection(this.for_node, this.for_state);
        section = await section;
        if(section === undefined) {
            return undefined;
        }
        let tmp = await this.communicator.runAnalysis("CacheOpAnalysis", [new Number(section.unified_id), new Number(section.supersection_id)]).get();

        let data = tmp;

        let ret = new DataBlock(data, "CacheOp");
        this.analysis_result = ret;
        ret.judgement = this.judgement();
        return ret;
    }
}

function AutoSuperSectionCacheOpAnalysis(communicator, section, nodeid, stateid, critical_path_analysis) {
    if(section instanceof LazySuperSection) {
        return new LazySuperSectionCacheOpAnalysis(communicator, section, nodeid, stateid);
    }
    else {
        return new SuperSectionCacheOpAnalysis(section, nodeid, stateid, critical_path_analysis);
    }
}

export { CacheOpButton, AutoSuperSectionCacheOpAnalysis };