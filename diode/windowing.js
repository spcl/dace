// Contains resources used to provide a "native" multi-window interface

// Class to create the window on the parent side
class DiodeWindow {
    constructor(parent) {
        this.parent = parent;
        this.window = null;

        this.message_userdef = null;

        this._msg_func = x => this.message(x);
        this.parent.addEventListener('message', this._msg_func);
    }

    open(url = '', target = '', features = '', replace = false) {
        this.window = parent.open(url, target, features, replace);

        if(!this.window) return this.window; // some error occurred

        return this.window;
    }

    setCallback(x) {
        this.message_userdef = x;
    }

    passMessage(msg_obj) {
        return this.window.postMessage(msg_obj, "*");
    }

    reply(win, origin, msg_obj) {
        if(win != this.window && this.window != null) {
            console.log("We do not talk to strangers");
        }
        return this.passMessage(msg_obj);
    }

    setSenderData(data) {
        //this.layout_data = data;
        this.window_data = data;
    }

    destroy() {
        this.parent.removeEventListener('message', this._msg_func)
    }

    serialize_dataview(dv) {
        return dv.getSendableObject();
    }

    serialize_array(arr) {
        let ret = [];
        for(let x of arr) {
            ret.push(this.serialize(x));
        }
        return ret;
    }
    
    serialize_function(func) {
        return func.toString();
    }

    serialize_default(def) {
        if(def == undefined) {
            return "";
        }
        let ret = {};
        let keys = ObjectHelper.listKeys(def);
        for(let x of keys) {
            let val = def[x];

            ret[x] = this.serialize(val);
        }

        return ret;
    }



    serialize(obj) {
        if(obj instanceof RU_DataView) {
            return this.serialize_dataview(obj);
        }
        else if(obj instanceof Array) {
            return this.serialize_array(obj);
        }
        else if(obj instanceof Object)
            return this.serialize_default(obj);
        else if(typeof obj == "function")
            return this.serialize_function(obj);
        else
            return obj;
    }

    message(event) {
        let _data = event.data;
        if(event.source != this.window) {
            return; 
        }

        if(_data.type == "ClientOpened") {
            // Respond with the data
           
            if(this.message_userdef == null) {
                let answer = {
                    type: "DisplayData",
                    data: this.window_data
                };
                this.reply(event.source, event.origin, answer);
            }
            else {
                // Pass to custom function.
                this.message_userdef(_data);
            }

        }
        else if(_data.type == "close") {
            console.log("Parent received close message");
            this.window.close();
            this.destroy();
        }
        else if(this.message_userdef != null) {
            // Custom callback method.

            
            this.message_userdef(_data);
        }
        else {
            console.log("Unknown type " + JSON.stringify(_data));
        }
    }

}


// Class managing the client side
class ClientSide {
    constructor(thiswindow, userfunc = null) {
        this.thiswindow = thiswindow;

        this.owner = thiswindow.opener;

        this.thiswindow.addEventListener('message', x => this.message(x));

        this.thiswindow.addEventListener('onbeforeunload', x => this.destroy(x));

        this.subwindow = null;

        this.message_userdef = userfunc;

        this.passMessage({
            type: "ClientOpened"
        });
    }

    destroy(x) {
        x.preventDefault();
        x.returnValue = '';
        console.log("Destroying client window");
    }

    setCallback(x) {
        this.message_userdef = x;
    }

    passMessage(msg_obj) {
        return this.owner.postMessage(msg_obj, "*");
    }

    reply(win, origin, msg_obj) {
        if(win != this.owner) {
            return;
        }
        return this.passMessage(msg_obj);
    }

    // Reads the event coming from the parent window
    message(event) {
        
        let data = event.data;

        if(data.type == "DisplayData") {

            let canvas = window.document.getElementById('subwindowcanvas');
            if(!canvas) {
                alert("Didn't find canvas!");
            }
            let ctx = canvas.getContext('2d');
            if(!ctx) {
                alert("Didn't find context!");
            }
            
            let content = data.data;
            let classname = content.className;
            if(!classname) {
                alert("Classname is not set!");
            }
            let dataparams = content.dataParams;

            let class_obj = null;
            if(classname == "ParallelizationButton") {
                class_obj = ParallelizationButton;
            }
            else if(classname == "MemoryButton") {
                class_obj = MemoryButton;
            }

            let new_obj = new class_obj(ctx, ...(dataparams));

            if(!new_obj) {
                alert("Failed to instantiate class " + classname);
            }

            {
                let subwindow_width = new_obj.button_subwindow.targetwidth;
                let subwindow_height = new_obj.button_subwindow.targetheight;
                let subwindow_left = new_obj.topleft.x;
                let subwindow_top = new_obj.topleft.y;

                ctx.canvas.width = subwindow_width;
                ctx.canvas.height = subwindow_height;

                window.resizeTo(subwindow_width, subwindow_height);

                new_obj.button_subwindow_state = 'open'; // Open Sesame
                new_obj.is_locked_open = true;
                let b = new Bracket(ctx);

                b.setupEventListeners();

                b.addButton(new_obj);

                b.drawEx(new Pos(-20, 0), new Pos(0, 0), 0, 0, true);

            }
        }
        else if(this.message_userdef != null) {
            this.message_userdef(data);
        }
        else  {
            console.log("Unknown event caught");
            ObjectHelper.logObject("message", data);
        }
        
    }
}