class ValueTreeNode {

    constructor(label, data=null) {
        this._label = label;
        this._prev = null;
        this._data = data;
        this._children = [];

        this._on_activate = null;
        this._representative = null;
    }

    label() {return this._label;}
    setPathLabel(l) { this._path_label = l; }
    pathLabel() { if(this._path_label === undefined) return this.label(); else return this._path_label; }
    data() {return this._data;}
    children() {return this._children;}
    parent() {return this._prev; }

    getChild(pred) {
        for(let x of this.children()) {
            if(pred(x)) return x;

            let tmp = x.getChild(pred);
            if(tmp != null) return tmp;
        }
        return null;
    }

    clearChildren() {
        this._children = [];
    }

    head() {
        let it = this;
        while(it._prev != null) {
            it = it._prev;
        }
        return it;
    }

    allLabelsInTree() {
        return this.head().asPreOrderArray(x => x.pathLabel());
    }

    asPreOrderArray(f = x => x) {

        let ret = [f(this)];
        
        for(let x of this.children()) {
            ret.push(...x.asPreOrderArray(f));
        }

        return ret;
    }



    path(map_func = undefined) {
        let ret = [];
        if(map_func === undefined) {
            map_func = current => { return {'name': current.pathLabel(), 'params': current.data()} };
        }
        let current = this;
        while(current._prev != null) {
            let e = map_func(current);
            if(e.name != " <virtual>") {
                // " <virtual>" (with the space) is a virtual identifier, i.e. do not include the node in the path
                ret = [e, ...ret];
            }
            current = current._prev;
        }

        return ret;
    }

    addNode(label, data=null, options={}) {
        /*
            options:
                .LabelConflictSameLevel
                    Function(new_node, labels) => new_node: 
                        Called before every new node is added to the list of children.
                        Can be used to assign different names on conflict in the same
                        group of children. (NOT globally!)
                .LabelConflictGlobal
                    Function(new_node, allNodeLabels) => new_node:
                        Called before every new node is added to the list of children.
                        Can be used to assign different names on conflict over all nodes.
                        
        */
        let new_node = new ValueTreeNode(label, data);
        new_node._prev = this;
        if(options.LabelConflictSameLevel != undefined) {
            new_node = options.LabelConflictSameLevel(new_node, this.children().map(x => x.label()));
        }
        if(options.LabelConflictGlobal != undefined) {
            new_node = options.LabelConflictGlobal(new_node, this.allLabelsInTree());
        }
        new_node._prev = this; // Make sure that after potential reordering, the basic order is still enforced
        this._children.push(new_node);
        return new_node;
    }

    setHandler(type, handler) {
        // type: 'activate'
        // handler: function (this, level)
        if(type == "activate") {
            this._on_activate = handler;
        }
        else {
            console.assert(false, "type " + type + " is unknown");
        }
    }
    
    setRepresentative(obj) {
        this._representative = obj;
    }

    representative() { return this._representative; }

    activate(level) {
        if(this._on_activate != null) {
            this._on_activate(this, level);
        }
    }
}

/*
Classic TreeView implementation
*/
class TreeView {
    constructor(value_tree_node) {
        this._tree = value_tree_node;
        this._debouncing = null;
    }

    setDebouncing(obj) {
        this._debouncing = obj;
    }

    /*
        parent: Used for parent.append()
        depth: Depth information - unused
        node: Overriding the starting node. If undefined, this._tree is used
    */
    create_html_in(parent, depth = 0, node = undefined) {
        
        let current = node === undefined ? this._tree : node;

        let listitem = document.createElement("li");
        let listitemspan = document.createElement("span");
        listitemspan.innerText = current.label();
        listitem.append(listitemspan);

        let nextparent = document.createElement("ul");
        nextparent.classList.add("tree_view");
        nextparent.classList.add("collapsed_sublist");
        
        let onclickfunc = () => {
            nextparent.classList.toggle("collapsed_sublist");
            current.activate(1);
        };
        let ondblclickfunc = () => {
            current.activate(2);
        };
        let passed_click_func = onclickfunc;
        let passed_dblclick_func = onclickfunc;

        if(this._debouncing != null) {
            passed_click_func = this._debouncing.debounce("treeview-click", onclickfunc, 100);
            passed_dblclick_func = this._debouncing.debounce("treeview-click", ondblclickfunc, 10);
        }
        listitem.addEventListener("click", passed_click_func);

        listitem.addEventListener("mouseenter", () => {
            current.activate(0);
        });
        listitem.addEventListener("mouseleave", () => {
            current.activate(-1);
        });

        listitem.addEventListener("dblclick", passed_dblclick_func);

        current.setRepresentative(listitem);
    

        nextparent.append(listitem);

        let children = current.children();
        if(children.length == 0) {
            // current is a leaf node
            
        }
        else {
            // current is the root of a non-trivial subtree
            for(let n of children) {
                this.create_html_in(nextparent, depth + 1, n);
            }
        }
        parent.append(nextparent);
    }
    
}
