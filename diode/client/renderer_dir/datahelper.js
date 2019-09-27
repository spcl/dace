import { max_func, min_func, max_func_obj } from "./renderer_util.js";

// Similar to the same class in python
class Entry {
    constructor(entryobj) {
        this.data = entryobj;
    }

    nodeid() {
        return new Number(this.data.node) & 0xFFFF;
        return this.data.node;
    }

    stateid() {
        return (new Number(this.data.node) >> 16) & 0xFFFF;
        return this.data.node;
    }

    thread() {
        return this.data.thread;
    }
    iteration() {
        return this.data.iteration;
    }
    values() {
        return this.data.values;
    }

    getKeys() {
        return this.data.values.map(x => ObjectHelper.listKeys(x)[0]);
    }
    getValue(papi_code, nofail = undefined) {
        let vals = this.data.values;
        for (let it of vals) {
            let keys = Object.keys(it);
            if (keys.some(x => x == papi_code)) {
                return it[papi_code];
            }
        }
        if(papi_code == "-2147483589") {
            // Instead of TOT_CYC, we allow REF_CYC with a warning
            console.warn("Fallback used (TOT_CYC => REF_CYC)");
            return this.getValue("-2147483541");
        }
        if(nofail == undefined) {
            ObjectHelper.logObject("this", this);
            ObjectHelper.logObject("keys", this.getKeys());
            ObjectHelper.assert("got value from key " + papi_code, false);
        }
        return "N/A";
    }
}
// Similar to the same class in Python
class Section {
    constructor(sectionobj = null, flags = undefined) {
        if (sectionobj == null) {
            return;
        }
        if(sectionobj instanceof Section) {
            
            Object.assign(this, sectionobj);
            this.private_fix_entries();
            return;
        }
        if(flags == "from_raw") {
            Object.assign(this, sectionobj);
            this.private_fix_entries();
            return;
        }
        this.node = sectionobj['entry_node'];
        this.datasize = sectionobj['static_movement'];

        ObjectHelper.assert("this is valid", this != undefined);
        let entries = sectionobj['entries']; // this is an array
        if(entries == undefined) {
            entries = sectionobj['_entries']; // Try with the other format
            if(entries != undefined) {
                ObjectHelper.logObject("sectionobj", sectionobj);
                console.trace("Dangerous assignment");
                Object.assign(this, sectionobj);
                this.private_fix_entries();
                ObjectHelper.assert("Correct subelements", this._entries.every(x => x instanceof Entry));
                return;
            }
            else {
                // Throw exception.
            }

        }
        ObjectHelper.assert("entries is valid", entries != undefined && entries != "undefined" && typeof entries != "function");
        this._entries = [];

        for (let e of entries) {
            // Construct and push
            this._entries.push(new Entry(e));
        }
    }

    private_fix_entries() {
        if(this._entries == undefined) {
            return;
        }
        if(!this._entries.every(x => x instanceof Entry)) {
            // Fix the entries
            this._entries = this._entries.map(x => new Entry(x.data));
        }
    }

    nodeid() {
        return new Number(this.node) & 0xFFFF;
        return this.node;
    }

    stateid() {
        return (new Number(this.node) >> 16) & 0xFFFF;
        return this.node;
    }

    entries() {
        return this._entries.map(x => x.data);
    }

    /* Returns true if this section is a serial section, i.e. only one thread 
       number occurs in all entries.
       This does not mean that the section iself was not inside a parallel 
       block, but that the section itself was not parallelized.
    */
    isSerial() {
        return this.threadid() != null;
    }

    threadid() {
        if(this._entries.length == 0) {
            return null;
        }
        let tid = this._entries[0].thread();
        if(this._entries.every(x => x.thread() == tid)) {
            return tid;
        }
        
        return null; // Not everyone had the same thread
    }

    // Returns the full cost of this section on one particular CPU, grouped 
    // by nodeid.
    sumIteration() {
        ObjectHelper.assert("object is serial", this.isSerial());

        let ret = {
            "entry_node": this.nodeid().toString(),
            "thread": this.threadid(),
            "iteration": "mixed",
            "flags": "mixed",
            "values": []
        };

        let keys = this.list_events();

        let vals = keys.map(x => { 
            let ret = {}; 
            ret[x] = MathHelper.sum(this.select_event(x));
            return ret;
        });

        ret['values'] = vals;

        let e = new Entry(ret);

        let retobj = new Section();
        Object.assign(retobj, this);
        retobj._entries = [e];

        return retobj;
    }

    numIterations() {
        let keys = this.list_events();

        let entry_count = 0;
        keys.forEach(x => { 
            entry_count = Math.max(entry_count, this.select_event(x).length);
        });
        return entry_count;
    }

    select_event(event, nofail = undefined) {

        return this._entries.filter(x => x.getValue(event, nofail) != "N/A").map(x => x.getValue(event, nofail));
    }

    list_events() {
        let keys = this._entries.map(x => x.getKeys());
        if(keys.length == 0) {
            return [];
        }
        let first = keys[0];
        let all_equal = keys.every(x => ObjectHelper.arraysEqual(x, first));
        if(!all_equal)
        {
            ObjectHelper.logObject("first", first);
            let first_different = keys.find(x => !ObjectHelper.arraysEqual(x, first));
            ObjectHelper.logObject("first different", first_different);
        }
        ObjectHelper.assert("same keys", all_equal);

        return first;
    }

    select_thread(threadnum) {
        return this.filter(x => x.thread() == threadnum);
    }
    select_node(nodeid) {
        return this.filter(x => x.nodeid() == nodeid);
    }

    get_max_thread_num() {
        ObjectHelper.assert("entries are defined", this._entries != undefined);
        for(let x of this._entries)
            ObjectHelper.assert("Must be correct object", x.thread != undefined);
        if(this._entries.length === 0)
            return undefined;
        return max_func(this._entries, x => x.thread());
    }

    get_min_thread_num() {
        ObjectHelper.assert("entries are defined", this._entries != undefined);
        if(this._entries.length === 0)
            return undefined;
        return min_func(this._entries, x => x.thread());
    }

    filter(predicate) {
        let ret = new Section();
        ret.node = this.nodeid();
        ObjectHelper.assert("entries valid", this._entries.every(x => x instanceof Entry));
        ret._entries = this._entries.filter(predicate);

        return ret;
    }
}

class SuperSection {

    constructor(supersection_obj = null) {
        let s = supersection_obj;
        if(s == null) {
            return;
        }

        this._sections = supersection_obj.sections.map(x => new Section(x));
        this._nodeid = supersection_obj.supernode;
    }

    // Downgrade to a section by flattening 
    toSection(nodeid = undefined, stateid = undefined) {
        let input = this.sections().filter(x => (nodeid == undefined || x.nodeid() == nodeid) && (stateid == undefined || x.stateid() == stateid));
        let rawobj = ObjectHelper.merge(input, {"datasize": (x, y) => x + y});
        let ret =  new Section(rawobj, "from_raw");

        let max_thread_num = 0;
        if(ret._entries != undefined) {

            max_thread_num = ret.get_max_thread_num();

            if(max_thread_num === undefined) {
                return undefined;
            }
        }

        for(let t = 0; t < max_thread_num; ++t) {
            let pre = input.map(x => x.select_thread(t).select_event('-2147483589'));
            if(pre.length == 0) {
                // We don't like it...
                continue;
            }
            else {
                let unwrapped = ObjectHelper.flatten(pre);
                let oldsum = MathHelper.sum(unwrapped);
                let newsum = MathHelper.sum(ret.select_thread(t).select_event('-2147483589'));
                ObjectHelper.assert("sum cyc equal", newsum == oldsum);
            }
        }

        return ret;
    }

    getSections(nodeid, stateid = undefined) {
        ObjectHelper.assert("nodeid provided", nodeid != undefined);

        let ret = this.sections();

        return ret.filter(x => x.nodeid() == nodeid && (stateid == undefined || x.stateid() == stateid));
    }

    // Get mean of threads
    toThreadMean(nodeid) {
        ObjectHelper.assert("nodeid defined", nodeid != undefined);

        // If we have a supersection of many sections and each of the 
        // sections contains only one thread, we are inside a parallel section.
        // Otherwise, if a supersection contains sections with mixed threads, 
        // we are outside.

        let all_serial = this.sections().every(x => x.isSerial());
        if(all_serial) {
            console.log("all sections are serial");
        }
        else {
            console.log("not all subsections are serial.");
            // This operation does not make any sense then.
            return null;
        }

        let means = this.sections().map(x => x.sumIteration());
        // means now has the means of all super-iterations 

        return means;
    }

    get_max_thread_num() {
        // Returns the maximum thread number of all sections
        ObjectHelper.assert("sections are defined", this.sections() != undefined);
        return max_func(this.sections(), x => x.get_max_thread_num());
    }

    filter(predicate) {
        let ret = new SuperSection();
        ret._nodeid = this.nodeid();
        
        ret._sections = this.sections().filter(predicate);

        return ret;
    }

    nodeid() {
        return this._nodeid & 0xFFFF;
    }

    stateid() {
        return new Number(this._nodeid) >> 16;
    }

    containsSection(nodeid, stateid = undefined) {
        return this.sections().filter(x => x.nodeid() == nodeid && (stateid == undefined || x.stateid() == stateid)).length > 0;
    }

    sections() {
        return this._sections;
    }

    allSectionNodeIds(for_state = undefined) {
        if(for_state == undefined) {
            return this.sections().map(x => x.nodeid());
        }
        else {
            // Only return the nodes of a given state
            return this.sections().filter(x => x.stateid() == for_state).map(x => x.nodeid());
        }
    }
    allSectionStateIds() {
        return this.sections().map(x => x.stateid());
    }

}
class MathHelper {
    static stdev(array) {
        return Math.sqrt(this.var(array));
    }

    static normalizedToHexByte(input) {
        ObjectHelper.assert("Input is normalized", input >= 0.0 && input <= 255.0);

        let denorm = input * 255;

        let tmp = denorm.toString(16).toUpperCase();
        ObjectHelper.assert("Correct Length", tmp.length == 1 || tmp.length == 2);
        return tmp.length == 2 ? tmp : "0" + tmp;
    }

    static majority(array) {

        ObjectHelper.assert("array valid", array != undefined && array.length != undefined);
        let i = 0;
        let dict = {};
        for (let x of array) {
            if (dict[x] == undefined) {
                dict[x] = 0;
            }
            dict[x]++;
            i++;
        }
        let a = [];
        for (let k of Object.keys(dict)) {
            let v = dict[k];

            a.push([k, v]);
        }
        ObjectHelper.assert("a sensible", a.length > 0);
        return max_func_obj(a, x => x[1], x => x[0]);
    }

    // Pearson's correlation coefficient
    static corr(X, Y) {
        return this.cov(X, Y) / (this.stdev(X) * this.stdev(Y));
    }

    // sample correlation
    static sample_corr(X, Y) {
        let xmean = this.mean(X);
        let ymean = this.mean(Y);

        let num = this.sum(this.zip(X, Y).map(x => (x[0] - xmean) * (x[1] - ymean)));
        let denom = Math.sqrt(this.sum(X.map(x => (x - xmean) * (x - xmean))) * this.sum(Y.map(y => (y - ymean) * (y - ymean))));
        return num / denom;
    }

    static cov(X, Y) {
        let n = X.length;

        let acc = 0;
        for (let i = 0; i < n; ++i) {
            for (let j = i + 1; j < n; ++j) {
                let tmp = (X[i] - X[j]) * (Y[i] - Y[j]);
                acc += tmp;
            }
        }
        return tmp / (n * n);
    }

    static var(array) {
        let mean = this.mean(array);
        let sum = array.reduce((a, b) => a + Math.pow(b - mean, 2.0), 0);
        return sum / array.length;
    }

    static mean(array) {
        if (array.length != 0) {
            return this.sum(array) / (array.length);
        }
        else {
            return 0;
        }
    }

    // This is the upper median
    static median(array) {
        if (array.length != 0) {
            let sorted = array.map(x => new Number(x));
            sorted.sort((a, b) => a - b);
            let index = sorted.length / 2;
            index = Math.floor(index);
            let ret = sorted[index];

            ret = new Number(ret);
            
            ObjectHelper.assert("is_number", ret instanceof Number);
            return ret;
        }
        else {
            return 0;
        }
    }

    static sum(array) {
        ObjectHelper.assert("Input valid", array != undefined && array != null);
        return array.reduce((a, b) => Number(a) + Number(b), 0);
    }
    static sumArray(array_of_arrays) {
        let base = null;
        for(let x of array_of_arrays) {
            if (base == null) {
                base = x;
            }
            else {
                for(let i = 0; i < base.length; ++i) {
                    base[i] = base[i] + x[i];
                }
            }
        }
        return base;
    }

    // Zips all elements of a 2d array 
    // (example: zip2d([[a, b, c], [1,2,3]]) -> [[a, 1], [b, 2], [c, 3]]). 
    // Restrictions: All sumelements must be of the same size!
    static zip2d(array) {
        if (array == []) return [];

        let ret = [];
        let outersize = array.length;
        let innersize = array[0].length;

        for (let i = 0; i < innersize; ++i) {
            let tmp = [];
            for (let j = 0; j < outersize; ++j) {
                tmp.push(array[j][i]);
            }
            ret.push(tmp);
        }
        return ret;
    }

    static zip(X, Y) {
        let ret = [];
        for (let i = 0; i < X.length; ++i) {
            ret.push([X[i], Y[i]]);
        }
        return ret;
    }

    static unique(array) {
        return array.filter((x, index, a) => a.indexOf(x) == index);
    }
}

class ObjectHelper {

    static createChunks(in_array, chunksize, aggregate=undefined) {
        ObjectHelper.assert("chunksize valid", !isNaN(chunksize) && chunksize != 0.0 && chunksize != undefined);
        let ret = []
        if(aggregate === undefined) aggregate = x => x;
        for(let i = 0; i < in_array.length; i += chunksize) {
            let tmp = in_array.slice(i, i + chunksize);
            ret.push(aggregate(tmp));
        }
        return ret;
    }

    static toUnicodeSuperscript(number) {
        number = new Number(number);
        const valarr = ["\u2070","\u00B9", "\u00B2", "\u00B3", "\u2074", "\u2075", "\u2076", "\u2077", "\u2078", "\u2079"];
        let valstr = "";

        if(number == 0) {
            valstr = valarr[0];
        }
        // Note: ~~(i / 10) is an integer division in JS
        // (bitwise operators make sense only on ints, so JS casts the float to 
        // an int, and then negates twice, which gives the int again)
        for(let i = parseInt(number.toFixed(0)); i > 0; i = ~~(i / 10)) {
            let index = Math.round(i) % 10;
            let c = valarr[index];
            
            valstr = c + valstr;
        }
        return valstr;
    }

    static toUnicodeSubscript(number) {
        number = new Number(number);
        const valarr = ["\u2080","\u2081", "\u2082", "\u2083", "\u2084", "\u2085", "\u2086", "\u2087", "\u2088", "\u2089"];
        let valstr = "";

        if(number == 0) {
            valstr = valarr[0];
        }
        
        for(let i = parseInt(number.toFixed(0)); i > 0; i = ~~(i / 10)) {
            let index = Math.round(i) % 10;
            let c = valarr[index];
            
            valstr = c + valstr;
        }
        return valstr;
    }

    // Number to sensible String
    static valueToSensibleString(value, mode="scientific", unit="", digits=4) {
        // valid modes:
        // scientific: x*10^(exp)
        // programmer: Ki/Mi/Gi/...
        value = new Number(value);
        let exp = 0;
        let base1024table = {
            0: "",
            1: "Ki",
            2: "Mi",
            3: "Gi",
            4: "Ti",
            5: "Pi"
        };
        let base1000table = {
            0: "",
            1: "K",
            2: "M",
            3: "G",
            4: "T",
            5: "P"
        };
        let ret = "";
        if(mode == "programmer") {
            // Go in 1024-Chunks
            while(value >= 1024.) {
                value = value / 1024.;
                ++exp;
            }
            ret = value.toFixed(digits - 1);

            ret += " " + base1024table[exp] + unit;
        }
        else if(mode == "scientific") {
            // Normal Scientific notation
            while(value >= 10.) {
                value /= 10;
                ++exp;
            }
            ret = value.toFixed(digits - 1);
            let valstr = exp.toString();
            // We need to replace numbers by their unicode equivalent for superscript
            valstr = ObjectHelper.toUnicodeSuperscript(exp);
            

            ret += " \u22C5 10" + valstr;
        }
        else if(mode == "fraction") {
            // Transform value into fraction (makes sense mostly for values \in [0, 1])

            // Simple (and slow) algorithm: Just multiply by 10 until the numerator is (about) integer
            const thresh = 0.00001;

            let numerator = new Number(value);
            let denominator = 1;
            while(Math.abs(Math.round(numerator) - numerator) > thresh)
            {
                numerator *= 10;
                denominator *= 10;
            }
            // Now it's really likely that the result is not rational (just a fraction, not normalized)
            let gcd = (a, b) => { 
                a = Math.abs(Math.round(a));
                b = Math.abs(Math.round(b));
                if (b > a) {let temp = a; a = b; b = temp;}
                while (a >= 0 && b >= 0) {
                    if (b == 0) return a;
                    a %= b;
                    if (a == 0) return b;
                    b %= a;
                }
                return 1;
            };

            let gcd_val = gcd(numerator, denominator);
            numerator /= gcd_val;
            denominator /= gcd_val;

            // We have the values - now translate them to the unicode equivalents
            let super_n = ObjectHelper.toUnicodeSuperscript(numerator.toString());
            let sub_d = ObjectHelper.toUnicodeSubscript(denominator.toString());

            ret = super_n + "/" + sub_d;

        }
        else {
            ObjectHelper.assert("unknown mode", false);
        }

        
        return ret;
    }

    // Applies operations to a key recursively (i.e. find a key and apply a function to the corresponding value). This function is in-place
    static modifyingMapRecursive(obj, key, func) {
        if(obj == undefined || obj == null || obj instanceof String || typeof obj == "string") {
            return;
        }
        if(obj instanceof Array) {
            for(let x of obj) {
                ObjectHelper.modifyingMapRecursive(x, key, func);
            }
            return;
        }
        let keys = ObjectHelper.listKeys(obj);
        if(keys == undefined) {
            return;
        }
        for(let k of keys) {
            if(k == key) {
                obj[k] = func(obj[k]);
            }
            else {
                ObjectHelper.modifyingMapRecursive(obj[k], key, func);
            }
        }
    }

    static async valueFromPromise(obj) {
        if (obj.then != undefined) {
            let x = await obj;
            return x;
        }
        else {
            return obj;
        }

    }

    // Same as valueFromPromise, but fully recursive (this will never return a promise)
    static async valueFromPromiseFull(obj) {
        if (obj.then != undefined) {
            let x = await obj;
            return await ObjectHelper.valueFromPromiseFull(x);
        }
        else {
            return obj;
        }

    }

    static async waitArray(obj_array) {
        for(let i = 0; i < obj_array.length; ++i) {
            obj_array[i] = await ObjectHelper.valueFromPromise(obj_array[i]);
        }
        return obj_array;
    }

    static arraysEqual(arr1, arr2) {
        this.assert("1 is array", arr1 instanceof Array);
        this.assert("2 is array", arr2 instanceof Array);
        if(arr1.length != arr2.length) {
            return false;
        }

        for(let i = 0; i < arr1.length; ++i) {
            if(arr1[i] != arr2[i]) {
                return false;
            }
        }
        return true;
    }

    static listKeys(obj) {
        let ret = [];
        for (let k of Object.keys(obj)) {
            ret.push(k);
        }

        return ret;
    }

    // Merges all objects together. Primitive values must be the same, and 
    // list elements will be appended. All prepended underscores of keys are
    // removed.
    static merge(in_array, onconflicts = {}) {
        // onconflicts: dict of key -> function(x, y), where the function 
        // resolves conflicts if the values are not identical.
        if(in_array.length == 0) return [];

        let keys = this.listKeys(in_array[0]);

        let ret = {};
        // Prime by adding keys
        for(let k of keys) {
            ret[k] = undefined;
        }
        for(let x of in_array) {
            for(let k of keys) {
                let v = x[k];
                if(v instanceof Array) {
                    if(ret[k] == undefined) {
                        ret[k] = [];
                    }
                    ret[k].push(...v);
                }
                else {
                    if(ret[k] == undefined) {
                        ret[k] = v;
                    }
                    if(ret[k] != v) {
                        console.log("Different elements for key " + k + ": " + ret[k] + " vs " + v);
                        if(onconflicts[k] != undefined) {
                            ret[k] = onconflicts[k](ret[k], v);
                            continue;
                        }
                    }
                    this.assert("Same primitive values", ret[k] == v);
                }
            }
        }
        return ret;
    }

    // Merges two objects into the return value. 'specials' applies the function in the value to the matching key.
    static mergeRecursive(o1, o2, specials = {}) {
        if(o1 == null || o1 == undefined) return o2;
        if(o2 == null || o2 == undefined) return o1;
        if(typeof o1 == "string") return o1;
        if(((typeof o1 == "array") || (o1 instanceof Array)) && ((typeof o2 == "array") || (o2 instanceof Array))) {
            if(o2.every(x => o1.includes(x))) {
                // If every element of o2 is already in o1, why bother?
                return o1;
            }
            return [...o1, ...o2];
        }

        let ret = o1;
        let keys = ObjectHelper.listKeys(o2);

        for(let k of keys) {
            let v = o2[k];

            if(specials[k] != undefined) {
                let specfunc = specials[k];
                ret[k] = specfunc(ret[k], v, specials);
                continue;
            }

            let o1ref = ret[k];
            if(o1ref == undefined) {
                // If the key is not present in o1, just add key + value
                ret[k] = v;
            }
            else {
                ret[k] = ObjectHelper.mergeRecursive(ret[k], v);
            }
        }
        return ret;
    }

    // Flattens an array of arrays to a single array
    static flatten(in_array) {
        return [].concat.apply([], in_array);
    }

    static logObject(title, obj) {
        if(obj == undefined) {
            obj = title;
            title = "(anon)";
        }
        return console.log(title + ": " + JSON.stringify(obj));
    }

    /* Groups elements by selecting a key using func(x) for every element x 
       in in_array. Returns a dict of key => [objects] */
    static groupBy(in_array, func) {

        let ret = {};

        for(let x of in_array) {
            ObjectHelper.assert("key not undefined", func(x) !== undefined);
            if(ObjectHelper.listKeys(ret).includes(func(x))) {
                ret[func(x)].push(x);
            }
            else {
                ret[func(x)] = [x];
            }
        }

        return Object.values(ret);
    }

    static assert(name, expr) {
        if(!expr) {
            console.log("Assertion \"" + name + "\" failed");
            console.trace();
            window.alert("Assertion failed. Check console");
            throw new Error();
        }
    }

    static stringify_circular(obj) {
        const getCircularReplacer = () => {
            const seen = new WeakSet;
            return (key, value) => {
                if (typeof value === "object" && value !== null) {
                    if (seen.has(value)) {
                        return;
                    }
                    seen.add(value);
                }
                return value;
            };
        };

        return JSON.stringify(obj, getCircularReplacer());
    }
}

// Class providing analysis of threads (mainly used in balance)
class ThreadAnalysis {
    constructor(section) {
        if(section instanceof Section) {
            this.section = section;
        }
        else if(section instanceof SuperSection) {
            ObjectHelper.assert("this is undesired.", false);
            // A supersection has many subsections. For now, let's try if 
            // merging works for this.
            this.section = section.toSection();
        }
        else if(section instanceof LazySuperSection) {
            this.section = section;
        }
        else if(section instanceof LazySection) {
            this.section = section;
        }
    }

    judgement(analysis=null) {
        return undefined;
    }

    analyze() {
        let section = this.section;

        let b_print_analysis = false;   // Set to true to debug.

        let data = {};

        let max_thread_num = Number(section.get_max_thread_num());
        if (b_print_analysis)
            console.log("max_thread_num: " + max_thread_num);
        let tot_cyc = [];
        let tot_l3_miss = [];
        let tot_l2_miss = [];
        let t = 0;
        for (t = 0; t < max_thread_num + 1; t++) {
            let ts = section.select_thread(t);
            let tc = ts.select_event('-2147483589'); // PAPI_TOT_CYC
            tot_cyc.push(MathHelper.sum(tc));

            let tl3 = ts.select_event('-2147483640'); // PAPI_L3_TCM
            tot_l3_miss.push(MathHelper.sum(tl3));

            let tl2 = ts.select_event('-2147483641');   //PAPI_L2_TCM
            tot_l2_miss.push(MathHelper.sum(tl2));
        }

        //Now we can get the balance
        let i = 0;
        if (b_print_analysis)
            for (let t of tot_cyc) {
                console.log("Thread " + i + " took " + t + " cycles");
                i++;
            }

        data.cycles_per_thread = tot_cyc;

        if(toplevel_use_mean)
            data.balance_stdev = (MathHelper.stdev(tot_cyc) / MathHelper.mean(tot_cyc));
        else if(toplevel_use_median) {
            data.balance_stdev = (MathHelper.stdev(tot_cyc) / MathHelper.median(tot_cyc));
        }
        else ObjectHelper.assert("Undefined mode", false);
        if (b_print_analysis)
            if (tot_cyc.length > 1 && MathHelper.mean(tot_cyc) != 0) {

                console.log("stdev: " + MathHelper.stdev(tot_cyc));
                console.log("Balance (stdev): " + data.balance_stdev);
            }


        // We need different means of balance calculations.
        let max_elem = Math.max(...tot_cyc);
        let min_elem = Math.min(...tot_cyc);
        let max_diff = max_func(tot_cyc, x => Math.max(Math.abs(max_elem - x), Math.abs(min_elem - x)));
        
        let biggest_unbalance = 0;
        if(toplevel_use_mean)
            biggest_unbalance = max_diff / MathHelper.mean(tot_cyc);
        else if(toplevel_use_median)
            biggest_unbalance = max_diff / MathHelper.median(tot_cyc);
        else ObjectHelper.assert("Undefined mode", false);

        if (b_print_analysis)
            console.log("max_diff: " + max_diff);
        if (b_print_analysis)
            console.log("Balance (max): " + biggest_unbalance);

        data.balance_max = biggest_unbalance;

        i = 0;
        if (b_print_analysis)
            for (let t of tot_l3_miss) {
                console.log("Thread " + i + " had " + t + " L3 misses");
                i++;
            }
        let sum_l3 = MathHelper.sum(tot_l3_miss);
        if (b_print_analysis)
            console.log("\n" + section.datasize + " bytes (presumably) accessed\n" + sum_l3 + " L3 misses over all threads\n" + (sum_l3 * 64) + " bytes loaded from memory");

        i = 0;
        if (b_print_analysis)
            for (let t of tot_l2_miss) {
                console.log("Thread " + i + " had " + t + " L2 misses");
                i++;
            }

        let sum_l2 = MathHelper.sum(tot_l2_miss);
        if (b_print_analysis)
            console.log("\n" + section.datasize + " bytes (presumably) accessed\n" + sum_l2 + " L3 misses over all threads\n" + (sum_l2 * 64) + " bytes loaded from memory");


        return new DataBlock(data, "thread");
    }


}

class LazyThreadAnalysis extends ThreadAnalysis {

    constructor(communicator, section) {
        super(section);
        this.section = section;
        this.communicator = communicator;
        ObjectHelper.assert("Parameter valid", this.communicator != undefined);
        ObjectHelper.assert("Parameter valid", this.section != undefined);
    }

    async analyze() {
        
        let tmp = await this.communicator.runAnalysis("ThreadAnalysis", [new Number(this.section.unified_id), new Number(this.section.supersection_id)]).get();

        let data = tmp;

        let ret = new DataBlock(data, "thread");
        this.analysis_result = ret;
        ret.judgement = this.judgement();
        return ret;
    }
}

function AutoThreadAnalysis(communicator, section) {
    if(section instanceof Section) {
        return new ThreadAnalysis(section);
    }
    else {
        return new LazyThreadAnalysis(communicator, section);
    }
}

class MemoryAnalysis {
    constructor(section, target_bw) {
        this.section = section;
        this.analysis_result = null;
        this.Memory_Target_Bandwidth = target_bw;
        if(this.Memory_Target_Bandwidth == undefined) {
            this.Memory_Target_Bandwidth = 20;
        }
    }

    judgement(analysis = null) {
        if (analysis == null) analysis = this.analysis_result;

        return MemoryAnalysis.sjudgement(analysis);
    }
    static sjudgement(analysis) {
        // We say memory was slow if the achieved bandwidth is below 50% of the target bandwidth
        let bandwidth = analysis.data.expected_bandwidth;
        let m = bandwidth / analysis.data.Memory_Target_Bandwidth;
        if (m < 0.5) {
            return -1;
        }
        // Otherwise, everything is fine
        return 1;
    }

    analyze() {
        let section = this.section;

        ObjectHelper.assert("section is Section", section instanceof Section);

        let b_print_analysis = false;   // Set to true to debug.

        let data = {};

        data.expected_memory_movement = section.datasize;

        if(b_print_analysis)
            console.log("Expected data movement: " + data.expected_memory_movement);

        let max_thread_num = Number(section.get_max_thread_num());
        let min_thread_num = Number(section.get_min_thread_num());

        if(max_thread_num == undefined || min_thread_num == undefined) {
            return undefined;
        }
        if (b_print_analysis)
            console.log("max_thread_num: " + max_thread_num);
        let tot_cyc = [];
        let tot_l3_miss = [];
        let tot_l2_miss = [];

        let mem_bw = []; // Bandwidth from mem to L3
        let l3_bw = [];  // Bandwidth from L3 to L2

        let critical_path_cyc = 0; // Critical path cycles

        let t = 0;
        for (t = min_thread_num; t < max_thread_num + 1; t++) {
            let ts = section.select_thread(t);
            let tc = ts.select_event('-2147483589'); // PAPI_TOT_CYC
            let tc_sum = MathHelper.sum(tc);
            tot_cyc.push(tc_sum);

            let tl3 = ts.select_event('-2147483640'); // PAPI_L3_TCM
            let tl3_sum = MathHelper.sum(tl3);
            tot_l3_miss.push(tl3_sum);

            let tl2 = ts.select_event('-2147483641');   //PAPI_L2_TCM
            let tl2_sum = MathHelper.sum(tl2);
            tot_l2_miss.push(tl2_sum);

            // Add the bandwidths for this element
            mem_bw.push(tl3_sum / tc_sum);
            l3_bw.push(tl2_sum / tc_sum);
        }

        if(tot_cyc.length == 0) {
            return undefined;
        }
        critical_path_cyc = max_func(tot_cyc, x => x);

        data.critical_path_cyc = critical_path_cyc;

        data.mem_bandwidth = mem_bw;
        data.l3_bandwidth = l3_bw;

        // Now we can get the balance
        let i = 0;
        if (b_print_analysis)
            for (let t of tot_cyc) {
                console.log("Thread " + i + " took " + t + " cycles");
                i++;
            }

        data.TOT_CYC = tot_cyc;
        data.L3_TCM = tot_l3_miss;
        data.L2_TCM = tot_l2_miss;


        i = 0;
        if (b_print_analysis)
            for (let t of tot_l3_miss) {
                console.log("Thread " + i + " had " + t + " L3 misses");
                i++;
            }
        let sum_l3 = MathHelper.sum(tot_l3_miss);
        if (b_print_analysis)
            console.log("\n" + section.datasize + " bytes accessed\n" + sum_l3 + " L3 misses over all threads\n" + (sum_l3 * 64) + " bytes loaded from memory");

        i = 0;
        if (b_print_analysis)
            for (let t of tot_l2_miss) {
                console.log("Thread " + i + " had " + t + " L2 misses");
                i++;
            }

        let sum_l2 = MathHelper.sum(tot_l2_miss);
        if (b_print_analysis)
            console.log("\n" + section.datasize + " bytes accessed\n" + sum_l2 + " L3 misses over all threads\n" + (sum_l2 * 64) + " bytes loaded from memory");

        {
            let datasize = data.expected_memory_movement;
            let crit_cyc = data.critical_path_cyc;

            let expected_bandwidth = datasize / crit_cyc;

            data.expected_bandwidth = expected_bandwidth;
        }
        data.Memory_Target_Bandwidth = this.Memory_Target_Bandwidth;
        let ret = new DataBlock(data, "memory");
        this.analysis_result = ret;
        ret.judgement = this.judgement();
        return ret;
    }
}

class SuperSectionMemoryAnalysis {
    constructor(section, nodeid, stateid, target_bw) {
        this.section = section;
        this.for_node = nodeid;
        this.for_state = stateid;
        this.analysis_result = null;

        this.Memory_Target_Bandwidth = target_bw;
        if(this.Memory_Target_Bandwidth == undefined) {
            this.Memory_Target_Bandwidth = 20;
        }

        if(!(stateid == 0xFFFF || stateid == 65535))
            ObjectHelper.assert("for_node defined", this.for_node != undefined && new Number(this.for_node) != NaN);
    }

    judgement(analysis = null) {
        if (analysis == null) analysis = this.analysis_result;

        // We say memory was slow if the achieved bandwidth is below 50% of 
        // the target bandwidth
        let bandwidth = analysis.data.expected_bandwidth;

        let m = bandwidth / analysis.data.Memory_Target_Bandwidth;
        if (m < 0.5) {
            return -1;
        }
        // Otherwise, everything is fine
        return 1;
    }

    analyze() {
        let section = this.section;

        ObjectHelper.assert("section is SuperSection", section instanceof SuperSection);

        // We have a supersection, so we should try to get individual 
        // sections out.
        section = section.toSection(this.for_node, this.for_state);
        if(section == undefined) {
            return null;
        }
        if(section['_entries'] == undefined) {
            return null;
        }

        let b_print_analysis = false;   // Set to true to debug.

        let data = {};

        data.expected_memory_movement = section.datasize;

        if(b_print_analysis)
            console.log("Expected data movement: " + data.expected_memory_movement);

        

        let max_thread_num = Number(section.get_max_thread_num());
        let min_thread_num = Number(section.get_min_thread_num());
        if (b_print_analysis)
            console.log("max_thread_num: " + max_thread_num);
        let tot_cyc = [];
        let tot_l3_miss = [];
        let tot_l2_miss = [];

        let mem_bw = []; // Bandwidth from mem to L3
        let l3_bw = [];  // Bandwidth from L3 to L2

        let critical_path_cyc = 0; // Critical path cycles

        let t = 0;
        for (t = min_thread_num; t < max_thread_num + 1; t++) {
            //console.log("iteration " + t);
            let ts = section.select_thread(t);
            let tc = ts.select_event('-2147483589'); // PAPI_TOT_CYC
            //console.log(tc);
            let tc_sum = MathHelper.sum(tc);
            tot_cyc.push(tc_sum);

            let tl3 = ts.select_event('-2147483640'); // PAPI_L3_TCM
            let tl3_sum = MathHelper.sum(tl3);
            tot_l3_miss.push(tl3_sum);

            let tl2 = ts.select_event('-2147483641');   //PAPI_L2_TCM
            let tl2_sum = MathHelper.sum(tl2);
            tot_l2_miss.push(tl2_sum);

            // Add the bandwidths for this element
            mem_bw.push(tl3_sum / tc_sum);
            l3_bw.push(tl2_sum / tc_sum);
        }

        critical_path_cyc = max_func(tot_cyc, x => x);

        data.critical_path_cyc = critical_path_cyc;

        data.mem_bandwidth = mem_bw;
        data.l3_bandwidth = l3_bw;

        let i = 0;
        if (b_print_analysis)
            for (let t of tot_cyc) {
                console.log("Thread " + i + " took " + t + " cycles");
                i++;
            }

        data.TOT_CYC = tot_cyc;
        data.L3_TCM = tot_l3_miss;
        data.L2_TCM = tot_l2_miss;


        i = 0;
        if (b_print_analysis)
            for (let t of tot_l3_miss) {
                console.log("Thread " + i + " had " + t + " L3 misses");
                i++;
            }
        let sum_l3 = MathHelper.sum(tot_l3_miss);
        if (b_print_analysis)
            console.log("\n" + section.datasize + " bytes accessed\n" + sum_l3 + " L3 misses over all threads\n" + (sum_l3 * 64) + " bytes loaded from memory");

        i = 0;
        if (b_print_analysis)
            for (let t of tot_l2_miss) {
                console.log("Thread " + i + " had " + t + " L2 misses");
                i++;
            }

        let sum_l2 = MathHelper.sum(tot_l2_miss);
        if (b_print_analysis)
            console.log("\n" + section.datasize + " bytes accessed\n" + sum_l2 + " L3 misses over all threads\n" + (sum_l2 * 64) + " bytes loaded from memory");

        {
            let datasize = data.expected_memory_movement;
            let crit_cyc = data.critical_path_cyc;

            let expected_bandwidth = datasize / crit_cyc;

            data.expected_bandwidth = expected_bandwidth;
        }

        data.Memory_Target_Bandwidth = this.Memory_Target_Bandwidth;
        let ret = new DataBlock(data, "memory");
        this.analysis_result = ret;
        ret.judgement = this.judgement();
        return ret;
    }
}

class LazySuperSectionMemoryAnalysis extends SuperSectionMemoryAnalysis {
    constructor(communicator, section, nodeid, stateid, target_bw) {
        super(section, nodeid, stateid, target_bw);
        this.communicator = communicator;
    }

    async analyze() {

        // We differ from the eager analysis here: We let the python/sql-side do the hard work
        let n = undefined;
        let s = new Number(this.for_state);
        if(this.for_node !== undefined)
            n = new Number(this.for_node);
        else
        {
            // Global
            s = 0;
            n = 0x0FFFFFFFF;
        }
        let tmp = await this.communicator.runAnalysis("MemoryAnalysis", [(s << 16) | n, this.section.index]).get();

        let data = tmp;

        if(data == null) {
            return null;
        }

        data.Memory_Target_Bandwidth = this.Memory_Target_Bandwidth;
        let ret = new DataBlock(data, "memory");
        this.analysis_result = ret;
        ret.judgement = this.judgement();
        return ret;
    }
}

function AutoSuperSectionMemoryAnalysis(communicator, section, nodeid, stateid, target_bw) {

    if(section instanceof LazySuperSection) {
        return new LazySuperSectionMemoryAnalysis(communicator, section, nodeid, stateid, target_bw);
    }
    else {
        return new SuperSectionMemoryAnalysis(section, nodeid, stateid, target_bw);
    }
}

class CriticalPathAnalysis {

    constructor(rundata, entry_node, stateid) {
        this.rundata = rundata;
        this.analysis_result = null;
        this.section_entry_node = entry_node;
        this.for_state = stateid;
    }

    judgement(analysis = null) {
        if (analysis == null) analysis = this.analysis_result;
        
        return CriticalPathAnalysis.sjudgement(analysis);
    }

    static sjudgement(analysis) {

        let e = analysis.data.efficiency;
        let max_thread_num = max_func(e, x => x.thread_num);

        let eff = 0;

        if(toplevel_use_mean) {
            eff = MathHelper.mean(e.find(x => x.thread_num == max_thread_num).value);
        }
        else if(toplevel_use_median) {
            eff = MathHelper.median(e.find(x => x.thread_num == max_thread_num).value);
        }
        else {
            ObjectHelper.assert("Undefined mode", false);
        }
        if (eff < 0.5) {
            return -1;
        }

        return 1;
    }

    analyze() {
        let data = {};

        // We want to compare different runs.
        let runs = this.rundata.map(x => x.data);

        
        if(this.section_entry_node !== undefined) {
            let single_threaded = new SuperSection(runs[0][0]).toThreadMean(this.section_entry_node, this.for_state);
        }

        let filtered_to_section = runs.map(x => x.map(y => new SuperSection(y).toSection(this.section_entry_node, this.for_state)).filter(x => x != undefined && x._entries != undefined));
        

       let thread_analyzed = filtered_to_section.map(x => x.map(y => { 
           return new ThreadAnalysis(y).analyze(); 
        }));

        // Now map to cycles_per_thread
        let cycles_per_thread = thread_analyzed.map(x => x.map(y => y.data.cycles_per_thread));

        // From here, we have all thread analyses (including tot_cyc for 
        // each element)
        let critical_paths = cycles_per_thread.map(x => x.map(y => max_func(y, z => z)));

        data.critical_paths = critical_paths.map((x, index) => ({ thread_num: index + 1, value: x }));

        let T1 = data.critical_paths.find(x => x.thread_num == 1).value;
        data.speedup = critical_paths.map((x, index) => ({ thread_num: index + 1, value: x.map((y, yi) => T1[yi] / y) }));

        data.efficiency = data.speedup.map((x, index) => ({ thread_num: index + 1, value: x.value.map((y, yi) => y / (index + 1)) }));

        let ret = new DataBlock(data, "path");
        this.analysis_result = ret;
        ret.judgement = this.judgement();
        return ret;
    }

}

class LazyCriticalPathAnalysis extends CriticalPathAnalysis {

    constructor(communicator, entry_node, stateid) {
        super(undefined, entry_node, stateid);
        this.communicator = communicator;
    }

    async analyze() {
        // We differ from the eager analysis here: We let the python/sql-side do the hard work
        ObjectHelper.assert("Valid state", this.for_state !== undefined);
        let p1 = undefined;
        if(this.section_entry_node !== undefined)
            p1 = new Number(this.section_entry_node);
        let tmp = await this.communicator.runAnalysis("CriticalPathAnalysis", [p1, new Number(this.for_state)]).get();

        let data = tmp;

        let ret = new DataBlock(data, "path");
        this.analysis_result = ret;
        ret.judgement = this.judgement();
        return ret;
    }
    
}


function AutoCriticalPathAnalysis(communicator, rundata, entry_node, stateid) {

    if(rundata != undefined) {
        return new CriticalPathAnalysis(rundata, entry_node, stateid);
    }
    else {
        // This is a lazy analysis.
        return new LazyCriticalPathAnalysis(communicator, entry_node, stateid)
    }
}



// Class to check if a result is reasonable
class ResultVerifier {
    constructor(all_perf_data) {
        this.all_data = all_perf_data;

        // Extract the list of supersections from the runs
        let rundata = this.all_data.map(x => x.data);

        ObjectHelper.assert("Data received", rundata.length > 0);
        if(!global_disable_verifier)
            ResultVerifier.assert_all_runs_same_number_of_entries(rundata);
    }

    static assert_all_runs_same_number_of_entries(runs) {
        
        ObjectHelper.assert("all_runs_same_number_of_supersection", runs.every(x => x.length == runs[0].length));

        // Now get all sections and check again.
        ObjectHelper.assert("all_runs_same_number_of_sections", runs.every(x => MathHelper.sum(x.map(y => new SuperSection(y).sections().length)) == MathHelper.sum(runs[0].map(y => new SuperSection(y).sections().length))));

        // Now check number of entries (this must also be the same)
        ObjectHelper.assert("all_runs_same_number_of_entries", runs.every(x => MathHelper.sum(x.map(y => MathHelper.sum(new SuperSection(y).sections().map(x => x.entries().length)))) == MathHelper.sum(runs[0].map(y => MathHelper.sum(new SuperSection(y).sections().map(x => x.entries().length))))));

    }


}

export {Entry, Section, SuperSection, MathHelper, ObjectHelper,
CriticalPathAnalysis, MemoryAnalysis};