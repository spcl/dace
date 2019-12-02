function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Communicator class to bridge the gap between host socket and client elements
class Communicator {
    constructor(socket) {

        this.seqid = 0;
        this.pendings = {};
    }

    send(data) {
        let sid = this.seqid;
        this.seqid = (this.seqid + 1) % 4096;
        let obj = {
            msg_type: "fetcher",
            seqid: sid,
            msg: data
        };
        
        this.pendings[sid] = new RequestedValue();
        socket.send(JSON.stringify(obj));
        
        return this.pendings[sid];
    }

    receive(msg) {
        let sid = msg.seqid;
        let data = msg.msg;

        this.pendings[sid].setReady(data);
        delete this.pendings[sid];
    }

    getRepetitions() {
        let data = {
            method: "getRepetitionCount",
            params: []
        };
        return this.send(data);
    }

    getRunOptionsFromMeta(meta_string) {
        let data = {
            method: "getRunOptionsFromMeta",
            params: [meta_string]
        };
        return this.send(data);
    }

    getSuperSectionCount(run_selection_string) {

        let data = {
            method: "getSuperSectionCount",
            params: [run_selection_string]
        };
        return this.send(data);
    }
    getSuperSectionDBIDS(run_selection_string) {

        let data = {
            method: "getSuperSectionDBIDS",
            params: [run_selection_string]
        };
        return this.send(data);
    }
    getAllSectionStateIds(run_selection_string) {

        let data = {
            method: "getAllSectionStateIds",
            params: [run_selection_string]
        };
        return this.send(data);
    }

    getAllSectionNodeIds(stateid) {

        let data = {
            method: "getAllSectionNodeIds",
            params: [stateid]
        };
        return this.send(data);
    }


    containsSection(ssid, unified_id) {
        let data = {
            method: "containsSection",
            params: [ssid, unified_id]
        };
        return this.send(data);
    }

    toSectionValid(ssid, unified_id) {
        let data = {
            method: "toSectionValid",
            params: [ssid, unified_id]
        };
        return this.send(data);
    }

    sections(ssid) {
        let data = {
            method: "sections",
            params: [ssid]
        };
        return this.send(data);
    }

    
    arbitaryQuery(query, param_array) {
        let data = {
            method: "SimpleQuery",
            params: [query, ...param_array]
        };
        return this.send(data);
    }

    runAnalysis(analysis_name, param_array) {
        let data = {
            method: "Analysis",
            params: [analysis_name, ...param_array]
        };
        return this.send(data);
    }

}

class RequestedValue {
    constructor() {
        this.value = "_null";
    }

    setReady(data) {
        this.value = data;
    }

    is_ready() {
        let ret = this.value !== "_null";
        return ret;
    }

    async get() {
        let i = 0;
        while(!this.is_ready()) {
            ++i;

            await sleep(1);

            i = i % 1000;
        }
        return this.value;
    }
}

// Lazily fetches supersections of a given run
class SuperSectionFetcher {

    constructor(communicator, type, supersections = undefined, run_selection_string = undefined) {
        this.communicator = communicator;
        if(type == "DataReady") {
            // Lazy
            this.lazy = true;
            this.run_selection_string = run_selection_string;
            this.count = 0;

            this.init_task = this.init();
        }
        else {
            // Eager
            this.lazy = false;
            this.supersections = supersections;
        }
    }

    async init() {
        if(this.run_selection_string === "meta:most_cores") {
            let tmp = await ObjectHelper.valueFromPromise(this.communicator.getRunOptionsFromMeta(this.run_selection_string).get());
            this.run_selection_string = tmp['options'];
        }

        this.supersection_db_ids = this.communicator.getSuperSectionDBIDS(this.run_selection_string);
        this.count = this.communicator.getSuperSectionCount(this.run_selection_string);

        this.count = await this.count.get();
        this.supersection_db_ids = await this.supersection_db_ids.get();
    }

    async wait_ready() {
        if(this.init_task === undefined) return;
        await this.init_task;
        delete this.init_task;
    }

    async allSectionStateIds() {
        if(this.lazy) {
            return this.communicator.getAllSectionStateIds(this.run_selection_string).get();
        }
        else {
            // Legacy code (expensive)
            return this.supersections.map(x => new SuperSection(x).allSectionStateIds());
        }
    }

    async allSectionNodeIds(stateid) {
        if(this.lazy) {
            return this.communicator.getAllSectionNodeIds(stateid).get();
        }
        else {
            return this.supersections.map(x => new SuperSection(x).allSectionNodeIds(stateid));
        }
    }

    async allMemoryAnalyses(sections, stateid, target_memory_speed) {
        let tmp = sections.map(x => AutoSuperSectionMemoryAnalysis(x.realize(), k, stateid, target_memory_speed).analyze());
        
        for(let i = 0; i < tmp.length; ++i) {
            tmp[i] = await ObjectHelper.valueFromPromise(tmp[i]);
        }
        tmp = tmp.filter(x => x != null);

        let ret = new DataBlock(tmp, "all_thread_mem");

        return ret;
    }

    elem(i) {
        if(this.lazy) {
            return new LazySuperSection(this.communicator, this.supersection_db_ids[i]);
        }
        else {
            
            return new EagerSuperSection(this.supersections[i]);
        }
    }

    map(func) {
        let ret = [];
        for(let x of this.iterator()) {
            ret.push(func(x));
        }
        return ret;
    }

    // Generator
    *iterator() {
        if(this.lazy) {
            // Fetch from host; but only the number of supersections, not the supersections themselves.
            for(let x of this.supersection_db_ids) {
                yield new LazySuperSection(this.communicator, x);
            }
        }
        else {
            // We have the data present.
            for(let x of this.supersections) {
                yield new EagerSuperSection(x);
            }
        }

        return;

    }

    [Symbol.iterator]() {
        return this.iterator();
    }
}

// SuperSection wrapper
class VirtualSuperSection {

    map(func) {
        ObjectHelper.assert("Abstract method", false);
    }

    allSectionStateIds() {
        ObjectHelper.assert("Abstract method", false);
    }

    // Create real class from contained data
    realize() {
        ObjectHelper.assert("Abstract method", false);
    }

    async containsSection(nodeid, stateid) {
        ObjectHelper.assert("Abstract method", false);
    }

}

class EagerSuperSection extends VirtualSuperSection {
    constructor(data) {
        super();
        this.data = data;
    }

    map(func) {
        return this.data.map(func);
    }

    realize() {
        return new SuperSection(this.data);
    }

}

class LazySuperSection extends VirtualSuperSection {

    constructor(communicator, index) {
        super();
        this.communicator = communicator;
        this.index = index;
    }

    realize() {
        // This should be kept lazy
        return this;
    }

    async sections() {
        let x = await ObjectHelper.valueFromPromiseFull(this.communicator.sections(this.index).get());

        return x.map(x => new LazySection(this.communicator, this.index, x.unified_id));
    }

    async containsSection(nodeid, stateid) {
        nodeid = new Number(nodeid);
        stateid = new Number(stateid);
        return this.communicator.containsSection(this.index, (stateid << 16) | nodeid).get();
    }

    async toSection(nodeid, stateid) {

        let unified_id = undefined;
        if(nodeid === undefined) {
            unified_id = 0x0FFFFFFFF;
        }
        else {
            nodeid = new Number(nodeid);
            stateid = new Number(stateid);
            unified_id = (stateid << 16) | nodeid;
        }
        let ls = new LazySection(this.communicator, this.index, unified_id);
        let tmp = await ls.is_valid();
        if(!tmp)
            return undefined;
        return ls;
    }

    async getSections(nodeid, stateid=undefined) {
        
        nodeid = new Number(nodeid);
        if(stateid == undefined) {
            stateid = 0;
            console.warn("Invalid parameters");
        }
        stateid = new Number(stateid);
        return this.toSection(nodeid, stateid);
    }
}

class LazySection {
    constructor(communicator, supersection_id, unified_id) {
        this.communicator = communicator;
        this.supersection_id = supersection_id;
        this.unified_id = new Number(unified_id);
    }

    async is_valid() {
        return await this.communicator.toSectionValid(this.supersection_id, this.unified_id).get();
    }
}

export {LazySection, LazySuperSection, Communicator};