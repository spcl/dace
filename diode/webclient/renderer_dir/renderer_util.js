import {ObjectHelper, MathHelper} from "./datahelper.js"
import { DiodeWindow } from "./windowing.js";

// Renderer utilities.

// Render a button to a canvas and return an image
function render_button_to_graphic(button, mode="canvas_element") {
    let button_class = button.constructor.name;
    let c = document.createElement("canvas");
    let ctx = c.getContext("2d");

    c.width = button.button_subwindow.targetwidth;
    c.height = button.button_subwindow.targetheight;

    // Recreate the button from the type name
    let new_obj = eval("new " + button_class + "(ctx, " + "...(button.dataparams)" + ");");

    new_obj.button_subwindow_state = 'open';
    new_obj.is_locked_open = true;
    // Skip animation
    new_obj.setFullyOpen();
    let b = new Bracket(ctx);

    b.setupEventListeners();

    b.addButton(new_obj);

    b.drawEx(new Pos(-20, 0), new Pos(0, 0), 0, 0, true);


    let imgdat = c.toDataURL("image/jpeg");

    if(mode === "canvas_element")
        return c;
    
    return imgdat;
}

function createImageDownload(brackets, prefix="") {
    // Some difficulties here: First, we have to merge the images to one big image.
    // We default to the following convention:
    // There are 10 columns with a width of 800px each. The first column is reserved to (textual) information about the node
    // Any more buttons would be wrapping around to newlines.

    // Every bracket begins a newline

    // Assuming WebKit=Firefox, there's a maximum of 32767px in either direction, or a total of 472'907'776px (e.g., 22'528 x 20'992)

    // TODO: Reevalutate this code and adjust to the case that there are more than 10 buttons (implement wrap-around, basically)

    let c = document.createElement("canvas");

    let ctx = c.getContext("2d");

    const colsize = 700;
    const rowsize = 500;
    let ypos = 0;
    let maxxpos = 0;

    // Cut height
    c.height = Object.entries(brackets).length * rowsize;

    // Cut width
    let tmp = Object.entries(brackets)[0][1];
    c.width = (1 + tmp.buttons.length) * colsize;


    // Fill white (will appear as transparent otherwise, which is viewer-dependent)
    ctx.save();
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.restore();

    for(let bracket_entry of Object.entries(brackets)) {
        
        let key = bracket_entry[0];
        let bracket = bracket_entry[1];

        ctx.save();

        ctx.textBaseline = "middle";
        ctx.textAlign = "center";

        ctx.font = "60px Arial";

        let unified_id = new Number(key);
        let stateid = (unified_id >> 16) & 0xFFFF;
        let nodeid = (unified_id) & 0xFFFF;

        ctx.fillText("Affected node " + stateid.toString() + "|" + nodeid.toString(), colsize / 2.0, ypos + rowsize / 2.0);

        ctx.restore();
        
        let xpos = colsize;
        for(let button of bracket.buttons) {
            // Draw the buttons accordingly

            let src_canvas = render_button_to_graphic(button);

            ctx.drawImage(src_canvas, xpos, ypos);

            xpos += colsize;
        }
        maxxpos = Math.max(maxxpos, xpos);

        ypos += rowsize;
        xpos = 0;
    }

    let imgdat = c.toDataURL("image/jpeg");
    
    let d = document.createElement("a");
    d.setAttribute("href", imgdat);
    d.setAttribute("download", prefix + "buttons.jpg");

    d.style.display = 'none';
    document.body.appendChild(d);
  
    d.click();
  
    document.body.removeChild(d);

}

function max_func(array, func) {
    if(array == undefined) {
        console.trace("undefined parameter");
    }
    ObjectHelper.assert("Array is non-empty", array.length > 0);
    let max_obj = null;
    let max_val = func(array[0]);
    for(let x of array) {
        let m = func(x);
        if(m > max_val) {
            max_val = m;
            max_obj = x;
        }
    }

    return max_val;
}

function max_func_obj(array, func, objfunc) {
    if(array.length == 0) {
        return null;
    }
    let max_obj = objfunc(array[0]);
    let max_val = func(array[0]);
    for(let x of array) {
        let m = func(x);
        if(m > max_val) {
            max_val = m;
            max_obj = objfunc(x);
        }
    }

    return max_obj;
}

function min_func(array, func) {
    let min_obj = null;
    let min_val = func(array[0]);
    for(let x of array) {
        let m = func(x);
        if(m < min_val) {
            min_val = m;
            min_obj = x;
        }
    }

    return min_val;
}

class Pos {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    toString() {
        return "(" + this.x + ", " + this.y + ")";
    }

    minus(other) {
        return new Pos(this.x - other.x, this.y - other.y);
    }
    plus(other) {
        return new Pos(this.x + other.x, this.y + other.y);
    }
    times(other) {
        return new Pos(this.x * other.x, this.y * other.y);
    }
    
    multiply(num) {
        return new Pos(this.x * num, this.y * num);
    }

    dist() {
        return Math.sqrt(this.x * this.x + this.y * this.y);
    }

    inRect(topleft, bottomright) {
        if (this.x < topleft.x || this.y < topleft.y) return false;
        if (this.x > bottomright.x || this.y > bottomright.y) return false;
        return true;
    }
}


class Clickable {

    constructor() {

        this.children = Array();

        this.onEnterHover = function () { return false; }
        this.onLeaveHover = function () { return false; }
        this.onClick = function () { return false; }
        this.onDoubleClick = function () { return false; }
        this.onMouseMove = function () { return false; }
        

        this.clickable_state = "not_hovered";

        this.enable_func = () => true;
    }

    destroy() {
        this.children = [];
    }

    addChild(clickable) {
        this.children.push(clickable);
    }

    addVIPChild(clickable) {
        this.children.unshift(clickable);
    }

    setEnableFunc(func) {
        this.enable_func = func;
    }

    onUpdateDoubleClick(mousepos, mb) {
        if(!this.enable_func()) return false;
        if(this.is_inside(mousepos)) {
            if(this.onDoubleClick())
                return true;
        }
        else {
            // Deselect?
        }

        for(let c of this.children) {
            if(c.onUpdateDoubleClick(mousepos, mb)) {
                return true;
            }
        }
        return false;
    }

    onUpdateClick(mousepos, mb) {
        if(!this.enable_func()) return false;
        if(this.is_inside(mousepos)) {
            if(this.onClick())
                return true;
        }
        else {
            // Deselect?
        }

        for(let c of this.children) {
            if(c.onUpdateClick(mousepos, mb)) {
                return true;
            }
        }
        return false;
    }

    onUpdateMove(mousepos) {
        if(!this.enable_func()) return false;
        if (this.is_inside(mousepos)) {
            this.onMouseMove(mousepos);
            if (this.clickable_state == "not_hovered") {
                this.clickable_state = "hovered";
                if (this.onEnterHover(mousepos)) {
                    return true;
                }
            }
        }
        else {
            if (this.clickable_state == "hovered") {
                this.clickable_state = "not_hovered";
                if (this.onLeaveHover()) {
                    return true;
                }
            }
        }
        // Else pass to children
        for (let c of this.children) {
            if (c.onUpdateMove(mousepos)) {
                return true;
            }
        }

        return false;
    }

    is_inside(pos) {
        // Abstract
        console.log("Abstract function called (is_inside)");
        return false
    }

    setOnEnterHover(func) {
        this.onEnterHover = func;
    }

    setOnLeaveHover(func) {
        this.onLeaveHover = func;
    }

    setOnClick(func) {
        this.onClick = func;
    }

    setOnDoubleClick(func) {
        this.onDoubleClick = func;
    }

    setOnMouseMove(func) {
        this.onMouseMove = func;
    }

}

class SubWindow extends Clickable {
    constructor(ctx, x, y, width, height) {
        super();

        this.ctx = ctx;
        this.topleft = new Pos(x, y);
        this.targetwidth = width;
        this.targetheight = height;
        this._sizetrans = 0;

        this.subwindow_trans_change = 10;

        this.subwindow_popped_out = false;

        this.layout = null;

        this.background_color = "white"; // Set to "transparent" if no background is requested
    }

    setLayout(layout) {
        this.layout = layout;
        return this;
    }

    width() {
        return this.targetwidth * this._sizetrans / 100.;
    }

    height() {
        return this.targetheight * this._sizetrans / 100.;
    }

    draw(topleft, state, ctx_override = null) {

        if (topleft != null)
            this.topleft = topleft;

        let ctx = this.ctx;
        if(ctx_override) {
            ctx = ctx_override;
        }

        ctx.save();

        ctx.beginPath();
        ctx.strokeStyle = "black";
        ctx.lineWidth = 2;
        let oldfill = ctx.fillStyle;
        ctx.fillStyle = this.background_color;
        ctx.rect(this.topleft.x, this.topleft.y, this.width(), this.height());
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = oldfill;

        if (state == 'open') {
            this._sizetrans += this.subwindow_trans_change;
            if (this._sizetrans > 100.)
                this._sizetrans = 100.;
        } else if (state == 'collapsed') {
            this._sizetrans -= this.subwindow_trans_change;
            if (this._sizetrans < 0.)
                this._sizetrans = 0.;
        }

        if(this.layout != null) {
            if(this._sizetrans != 0)
                this.layout.draw(ctx);
        }

        ctx.restore();
    }

    is_inside(pos) {
        let offsetpos = pos.minus(this.topleft);

        if (offsetpos.x < 0 || offsetpos.y < 0) return false;
        if (offsetpos.x > this.width() || this.height()) return false;
        return true;
    }


}

class Button extends Clickable {
    constructor(ctx) {
        super();
        this.ctx = ctx;
        this.state = 0;
        this.update = 0;

        this.color = "orange";
        this.button_subwindow_state = "collapsed";
        this.button_subwindow = new SubWindow(ctx, 0, 0, 600, 400);
        this.addChild(this.button_subwindow);
        this.is_locked_open = false;

        this.topleft = new Pos(0,0);
        this.size = new Pos(0,0);

        this.button_image = null;
    }

    setDefaultDblClick() {
        this.setOnDoubleClick(p => {
            let newwin = new DiodeWindow(window);
            newwin.setSenderData({ 
                className: this.constructor.name,
                dataParams: this.dataparams
            });
            let subwin = newwin.open("renderer_dir/subwindow.html", "_blank");
            if(!subwin) {
                console.log("Failed to open subwindow");
                alert("failed to open subwindow");
            }
            
            return true;
        });
    }

    // Bypass the opening animation (for saving results)
    setFullyOpen() {
        this.button_subwindow_state = "open";
        this.button_subwindow._sizetrans = 100.0;
        this.is_locked_open = true;
    }

    // Set a button image from data_url, and resize to the given side length (the image must be square)
    setButtonImage(data_url, side_len = 100) {
        if(data_url == undefined) {
            return this;
        }
        
        this.button_image = new Image();
        this.button_image.src = data_url;

        // Apparently you cannot assume that the image element is ready synchronously (without onload, it could happen that an empty image is set instead)
        this.button_image.onload = () => {

            let c = document.createElement("canvas");
            let ctx = c.getContext("2d");

            c.width = side_len;
            c.height = side_len;

            ctx.drawImage(this.button_image, 0, 0, side_len, side_len);

            // Now set the resized image to gain some performance.
            this.button_image.src = c.toDataURL();
        }
    }

    is_inside(pos) {
        let offsetpos = pos.minus(this.topleft);

        if (offsetpos.x < 0 || offsetpos.y < 0) return false;
        if (offsetpos.x > this.size.x || offsetpos.y > this.size.y) return false;
        return true;
    }

    draw(topleft, size) {
        this.update++;
        let now = new Date();
        let ctx = this.ctx;

        this.topleft = topleft
        this.size = size

        // Draw the window first (this way, the button is not occluded)
        this.button_subwindow.draw(topleft, this.button_subwindow_state);
        ctx.save();

        ctx.beginPath();
        // Draw the rect
        if(this.button_image == null) {
            if (this.state == 0) {
            
                let cfseg = 2 * Math.PI / 1000.;
                let mid = size.x / 2;
                let fac = now.getMilliseconds();
                let xval = Math.cos(cfseg * fac) * mid;
                let yval = Math.sin(cfseg * fac) * mid;


                let grad = ctx.createLinearGradient(topleft.x + mid + xval, topleft.y + mid + yval, topleft.x + mid - xval, topleft.y + mid - yval);
                grad.addColorStop(0, this.color);
                grad.addColorStop(1, "white");
                ctx.fillStyle = grad;
                
            }

            ctx.strokeStyle = "#000000";
            ctx.lineWidth = 2;
            ctx.rect(topleft.x, topleft.y, size.x, size.y);
            ctx.stroke();
            if (this.state == 0) {
                ctx.fill();
            }
        }
        else {
            ctx.drawImage(this.button_image, topleft.x, topleft.y, size.x, size.y);
            ctx.strokeStyle = "#000000";
            ctx.lineWidth = 2;
            ctx.rect(topleft.x, topleft.y, size.x, size.y);
            ctx.stroke();
        }

        ctx.restore();

        topleft = null;
        size = null;
        now = null;

    }


}

// Class to specify where which elements are located
class Layout {
    constructor(subwindow) {
        this.parent = subwindow;
        this._layout = {};
        this.databinding = null;
        let _this = this;
        this._layout_clickable = new class extends Clickable {
            is_inside(pos) {
                for(let x of Object.keys(_this._layout)) {
                    let val = _this._layout[x];
                    let realsize = new Pos(_this.parent.width() * val.size.x / 100.0, _this.parent.height() * val.size.y / 100.0);
                    let realtopleft = _this.parent.topleft.plus(val.topleft.times(new Pos(_this.parent.width() / 100.0, _this.parent.height() / 100.0)));

                    let inside = realtopleft.x < pos.x && pos.x < (realtopleft.x + realsize.x) && realtopleft.y < pos.y && pos.y < (realtopleft.y + realsize.y);
                    let has_handlers = true;
                    if(inside && has_handlers) {
                        this._tmp_hovered = x;
                        this._mouse_pos = pos;
                        return true;
                    }
                }
                delete this._tmp_hovered;
                return false;
            }
        }

        this._layout_clickable.setOnClick(() => {
            let target = this._layout_clickable._tmp_hovered;
            let val = this._layout[target];

            let realsize = new Pos(this.parent.width() * val.size.x / 100.0, this.parent.height() * val.size.y / 100.0);
            let realtopleft = this.parent.topleft.plus(val.topleft.times(new Pos(this.parent.width() / 100.0, this.parent.height() / 100.0)));

            let pos = this._layout_clickable._mouse_pos;

            // check if the "information-button" is clicked
            {
                let dv = this.getCurrentDataView(target);
                if(dv.information_button_pos != null) {
                    if(dv.information_button_pos.x < pos.x && dv.information_button_pos.y < pos.y) {
                        if(dv.information_button_pos.x + dv.information_button_size.x > pos.x && dv.information_button_pos.y + dv.information_button_size.y > pos.y) {
                            console.log("Information Button clicked!");
                            dv.showOptimizationHints();
                            return true;
                        }
                    }
                }
            }

            let left = realtopleft.x + realsize.x - 20;
            let right = realtopleft.x + realsize.x;
            let top = realtopleft.y;
            let bottom = realtopleft.y + 20;
            if(val.is_multiview) {
                if(left < pos.x && pos.x < right && top < pos.y && pos.y < bottom) {
                    // Multiview switcher pressed
                    val.multiview_selector = (val.multiview_selector + 1) % val.dataviews.length;
                    return true;
                }
                else {
                }
            }
            
            return false; // Pass the event to the next handler
            

        });

        this._layout_clickable.setOnMouseMove((pos) => {
            // We want to know the position and which element is hovered so 
            // that we can forward events
            let target = this._layout_clickable._tmp_hovered;
            if(target) {
                let val = this.getCurrentDataView(target);
                val.mouseInside(pos);
                return true;
            }
            else {
            }
        });

        this._layout_clickable.setOnDoubleClick(() => {
            let target = this._layout_clickable._tmp_hovered;
            let val = this._layout[target];

            let realsize = new Pos(this.parent.width() * val.size.x / 100.0, this.parent.height() * val.size.y / 100.0);
            let realtopleft = this.parent.topleft.plus(val.topleft.times(new Pos(this.parent.width() / 100.0, this.parent.height() / 100.0)));

            let pos = this._layout_clickable._mouse_pos;

            let left = realtopleft.x + realsize.x - 20;
            let right = realtopleft.x + realsize.x;
            let top = realtopleft.y;
            let bottom = realtopleft.y + 20;
            if(val.is_multiview) {
                if(left < pos.x && pos.x < right && top < pos.y && pos.y < bottom) {
                    return false;
                }
                else {
                    // Trigger the options for the underlying dataview
                    let dv = val.dataviews[val.multiview_selector];

                    dv.openSettingsWindow();
                    return true;
                }
            }
            else {
                // Trigger the options for the underlying dataview
                let dv = val.dataview;

                dv.openSettingsWindow();

                return true;
            }
            
            return false; // Not reached, but leave here (in case the handling of above cases changes)
            

        });

        subwindow.addVIPChild(this._layout_clickable);
    }

    setEventHandlers(click, enter, leave) {
        this._layout_clickable.setOnClick(click);
        this._layout_clickable.setOnEnterHover(enter);
        this._layout_clickable.setOnLeaveHover(leave);
    }

    getCurrentDataView(rect_name) {
        let val = this._layout[rect_name];
        if(val.is_multiview) {
            return val.dataviews[val.multiview_selector];
        }
        else {
            return val.dataview;
        }
    }

    getLayoutObject() {
        let ret = {};
        for(let k of Object.keys(this._layout)) {
            if(k == 'dataview') continue;
            if(k == 'dataviews') continue;
            ret[k] = this._layout[k];
        }
        return this._layout;
    }

    // Sizes all must be in percentages.
    setRect(name, topleft, size, dataview) {
        this._layout[name] = {};
        this._layout[name].topleft = topleft;
        this._layout[name].size = size;
        this._layout[name].dataview = dataview;
    }

    setMultiviewRect(name, topleft, size, dataview_array) {
        if(dataview_array.length < 1) {
            alert("You cannot set a multiview with no contents!");
            return;
        }
        this._layout[name] = {};
        this._layout[name].topleft = topleft;
        this._layout[name].size = size;
        this._layout[name].dataviews = dataview_array;
        this._layout[name].is_multiview = true;
        this._layout[name].multiview_selector = dataview_array.length - 1;

        // Set enable functions so that charts that are hidden behind others 
        // don't get the events propagated
        for(let i = 0; i < dataview_array.length; ++i) {
            let x = dataview_array[i];
            if(x instanceof RU_DataViewBarGraph) {
                x.dvbg_clickable.setEnableFunc(() => {
                    return this._layout[name].multiview_selector == i;
                });
            }
        }
    }

    setDataBinding(db) {
        this.databinding = db;
    }

    draw(ctx) {
        let databinding = this.databinding;
        for(let x of Object.keys(this._layout)) {
            let val = this._layout[x];

            let realsize = new Pos(this.parent.width() * val.size.x / 100.0, this.parent.height() * val.size.y / 100.0);
            let realtopleft = this.parent.topleft.plus(val.topleft.times(new Pos(this.parent.width() / 100.0, this.parent.height() / 100.0)));

            let dv = null;
            if(val.is_multiview) {
                dv = val.dataviews[val.multiview_selector];
            }
            else {
                dv = val.dataview;
            }
            if(dv == undefined) {
                console.log("Undefined dv for key " + x);
            }
            dv.setRect(realtopleft, realsize);

            if(val.is_multiview) {
                dv.draw(ctx, databinding[x][val.multiview_selector], this.parent._sizetrans / 100.);
                ctx.save();
                ctx.beginPath();
                ctx.fillStyle="lime";
                ctx.rect(realtopleft.x + realsize.x - 20, realtopleft.y, 20, 20);
                ctx.fill();
                ctx.restore();
            }
            else {
                dv.draw(ctx, databinding[x], this.parent._sizetrans / 100.);
            }
        }
    }
}

// Class holding data
class DataBlock {
    constructor(data, type) {
        this.setData(data, type);
    }
    setData(data, type) {
        this.data = data;
        this.datatype = type;

        return this;
    }
}

// Class providing an interface to plug some graphs into.
class RU_DataView {
    constructor() {
        this.topleft = null;
        this.size = null;
        this.analyze = x => x;
        this.color_scaling_func = x => x;

        this.update_data = true;

        /* String that contains an html markup with general information about 
           the implications of this dataview (i.e. how to interpret the 
           displayed results) */
        this.information_html_string = "";

        this.information_overlay_age = 0;
        this.information_overlay_alpha = 0.0;
        this.information_overlay_timer = null;

        this.information_button_pos = null;
        this.information_button_size = null;
        this.optimization_hint_path = "error";
    }

    setInformationFilePath(path) {
        this.optimization_hint_path = path;

        return this;
    }

    showOptimizationHints() {
        
        console.log("Opening " + this.optimization_hint_path);
        let win = window.open(this.optimization_hint_path, "", "width=800,height=600");
        
        ObjectHelper.assert("win valid", win != null);

        return this;
    }

    mouseInside(pos) {
        ObjectHelper.assert("pos valid", pos);
        ObjectHelper.assert("topleft valid", this.topleft);
        ObjectHelper.assert("size valid", this.size);
        
        let diff = pos.minus(this.topleft.plus(new Pos(this.size.x / 2, 0)));

        let dist = diff.dist();

        if(dist <= 30) {
            this.information_overlay_age = -1;
            window.clearTimeout(this.information_overlay_timer);
            this.information_overlay_timer = window.setTimeout(() => this.information_overlay_age = 0,  3000);
        }
        else {
            this.information_overlay_age = 0;
        }

        if(dist <= 10) {
            this.information_overlay_alpha = 1.0;
        }
        else {
            // Scale squared
            let d = (dist) / 10;
            this.information_overlay_alpha = 1.0 / (d);
        }
    }

    // Draw an overlay (containing the info-button at least). This function 
    // should be called by all subclasses after drawing.
    drawOverlay(ctx, scale = 1.0) {

        if(this.information_overlay_age >= 0) {

            ++this.information_overlay_age;
            if(this.information_overlay_age > 1) {
                this.information_overlay_alpha -= 0.05;
                if(this.information_overlay_alpha < 0) {
                    this.information_overlay_alpha = 0;
                }
                this.information_overlay_age = 0; 
            }
        }

        let alpha = 0.9; // Transparency value
        alpha *= this.information_overlay_alpha;
        let subalpha = 1.0;
        subalpha *= this.information_overlay_alpha;
        
        let information_sign = "\u{1F6C8}"; // This is the circled information icon.
        let base_fontsize = 18;
        ctx.save();
        ctx.beginPath();

        // Setup font
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.font = Math.round(base_fontsize * scale) + "px sans-serif";

        // Get the textwidth
        let textwidth = ctx.measureText(information_sign).width;
        let textheight = textwidth;
        // Draw background first
        ctx.fillStyle = "rgba(128, 128, 128, " + alpha + ")";
        ctx.strokeStyle = "rgba(128, 128, 128, " + subalpha + ")";
        this.information_button_pos = new Pos(this.topleft.x + (this.size.x - textwidth) * scale / 2 - textwidth, this.topleft.y);
        this.information_button_size = new Pos(textwidth * 3, textheight * 1.1);
        ctx.rect(this.information_button_pos.x, this.information_button_pos.y, this.information_button_size.x, this.information_button_size.y);
        ctx.fill();
        ctx.stroke();

        ctx.beginPath();
        
        alpha = 3.0 * this.information_overlay_alpha; 
        ctx.fillStyle = "rgba(0, 0, 255, " + alpha + ")";
        let center_x = this.topleft.x + this.size.x * scale / 2;
        ctx.fillText(information_sign, center_x, this.topleft.y);


        ctx.restore();
    }

    // Gets the settings (data analysis function and subclass-specific settings)
    getSettingsDict() {
        let ret = {
            "analysis_func": {
                type: "code",
                value: this.analyze.toString(),
                description: "#TODO"
            }
        };

        return ret;
    }

    setSettingsDict(dict) {
        let func = eval(dict['analysis_func'].value.toString());
        ObjectHelper.assert("Object type function", func instanceof Function);
        this.setDataAnalysisFunction(func);
        this.update_data = true; // Trigger an update request
    }

    setInformationHTMLString(str) {
        this.information_html_string = str;
        return this;
    }

    getInformationHTMLString() {
        return this.information_html_string;
    }

    openSettingsWindow() {
        let sw = new DiodeWindow(window);
        // Set the callback function for user-defined data
        sw.setCallback(x => {
            
            if(x.type == "ClientOpened") {
                sw.reply(x.source, x.origin, {
                    type: "settings-data",
                    data: this.getSettingsDict()
                });
            }
            else if(x.type == "save-settings") {
                this.setSettingsDict(x.data);
                console.log("Applied new settings");
            }
            else {
                console.log("Undefined operation reached! " + JSON.stringify(x));
            }
        });
        let cw = sw.open("DataViewSettings.html", "_blank");
        if(!cw) {
            alert("Failed to open child window...");
        }
    }

    setTitle(title) {
        // Abstract
        return this;
    }
    draw(ctx, datablock, scale) {
        console.log("Abstract function called");
    }

    getSendableObject() {
        return this;
    }

    fromSendableObject(object) {
        console.log("Abstract function fromSendableObject() called");
        return this;
    }

    setDataAnalysisFunction(func) {
        this.analyze = func;
        return this;
    }

    setRect(topleft, size) {
        this.topleft = topleft;
        this.size = size;
        return this;
    }
    
    drawRect(ctx) {
        ctx.beginPath();
        ctx.rect(this.topleft.x, this.topleft.y, this.size.x, this.size.y);
        ctx.stroke();
    }
}

// Class to draw text
class RU_DataViewText extends RU_DataView {
    constructor() {
        super();
    }

    fromSendableObject(obj) {
        // We don't need to do anything here because the data is in the datablock...
        return this;
    }
    draw(ctx, datablock, scale) {

        if(scale == 0.)
            return;


        ctx.save();

        ctx.beginPath();


        ctx.font = Math.round(scale * datablock.data.fontsize).toString() + "px sans-serif";
        ctx.fillStyle = datablock.data.color;
        ctx.textAlign=datablock.data.align;
        let textheight = ctx.measureText('M').width;
        ctx.fillText(datablock.data.text, this.topleft.x + this.size.x / 2, this.topleft.y + textheight);

        ctx.restore();

    }

}

// Class to bar graphs (using chart.js)
class RU_DataViewBarGraph extends RU_DataView {
    constructor(optstruct = null) {
        super();

        let typevar = "bar";
        let xAxisVar = undefined;
        let yAxisVar = undefined;

        this.dvbg_chart_data = null;

        if(optstruct != null) {
            if(optstruct.type != undefined) {
                typevar = optstruct.type;
            }
            if(optstruct.xAxes != undefined) {
                xAxisVar = optstruct.xAxes;
            }
            if(optstruct.yAxes != undefined) {
                yAxisVar = optstruct.yAxes;
            }
        }

        // We create a new canvas just for the chart (mainly to avoid 
        // potential trouble with different update handlers)
        this.dvbg_canvas_scaler = document.createElement("div");
        this.dvbg_canvas = document.createElement("canvas");
        this.dvbg_ctx = this.dvbg_canvas.getContext("2d");

        this.dvbg_canvas_scaler.appendChild(this.dvbg_canvas);
        document.body.appendChild(this.dvbg_canvas_scaler);

        let chartsettings = {
            type: typevar,
            data: this.analyze(null),
            options: {
                responsive: true,
                title: {
                    display: true,
                    text: "Chart test..."
                },
                tooltips: {
                    mode: 'index',
                    intersect: true
                },
                scales: {
                    
                }
            }
        };

        if(xAxisVar != undefined) {
            chartsettings.options.scales.xAxes = xAxisVar;
        }
        if(yAxisVar != undefined) {
            chartsettings.options.scales.yAxes = yAxisVar;
        }

        this.dvbg_chart = new Chart(this.dvbg_ctx, chartsettings);

        this.dvbg_canvas_scaler.style.visibility = "hidden";
        this.dvbg_canvas_scaler.style.position = "fixed";

        // A Clickable instance, for forwarding events.
        let parent_this = this;
        this.dvbg_clickable = new class extends Clickable {
            constructor() {
                super();
            }

            is_inside(pos) {
                if(parent_this.topleft == null) return false;
                return (parent_this.topleft.x < pos.x && pos.x < parent_this.topleft.x + parent_this.size.x) && (parent_this.topleft.y < pos.y && pos.y < parent_this.topleft.y + parent_this.size.y);
            }

            onUpdateMove(pos) {
                if(!this.enable_func()) return false;
                if(!this.is_inside(pos))
                    return false;
                // We have to translate the event to the target.
                let rect = parent_this.dvbg_canvas.getBoundingClientRect();
                let event = new MouseEvent("mousemove", {
                    offsetX: pos.x,
                    offsetY: pos.y,
                    clientX: rect.left + pos.x - parent_this.topleft.x,
                    clientY: rect.top + pos.y - parent_this.topleft.y
                });

                // Now pass to other canvas
                parent_this.dvbg_canvas.dispatchEvent(event);
            }

            onUpdateClick(pos) {
                if(!this.enable_func()) return false;
                if(!this.is_inside(pos))
                    return false;
                // We have to translate the event to the target.
                let rect = parent_this.dvbg_canvas.getBoundingClientRect();
                let event = new MouseEvent("click", {
                    offsetX: pos.x,
                    offsetY: pos.y,
                    clientX: rect.left + pos.x - parent_this.topleft.x,
                    clientY: rect.top + pos.y - parent_this.topleft.y
                });

                // Now pass to other canvas
                parent_this.dvbg_canvas.dispatchEvent(event);
            }
        }
    }

    getSettingsDict() {
        let basedict = super.getSettingsDict();

        // Collect the information
        let xAxes = this.dvbg_chart.options.scales.xAxes;
        let yAxes = this.dvbg_chart.options.scales.yAxes;
        let display_xAxes = xAxes.some(x => x == undefined || x.display);
        let display_yAxes = yAxes.some(x => x == undefined || x.display);

        let display_legend = this.dvbg_chart.options.legend.display;

        // Now we can append to the basedict
        basedict['graph_general_options'] = {
            type: "group", // Specify subgroup
            value: {
                // new subgroup of all graph options
                "display_horizonal_axis": {
                    type: "bool",
                    value: display_xAxes,
                    description: "Display the horizontal axes"
                },
                "display_vertical_axis": {
                    type: "bool",
                    value: display_yAxes,
                    description: "Display the vertical axes"
                },
                "display_legend": {
                    type: "bool",
                    value: display_legend,
                    description: "Display the legend"
                }
            },
            description: "General graph display options"

        };

        return basedict;
    }

    setSettingsDict(dict) {
        super.setSettingsDict(dict);

        // Get the desired settings
        let general_graph_options = dict['graph_general_options'].value;
        let dha = general_graph_options['display_horizonal_axis'].value;
        let dva = general_graph_options['display_vertical_axis'].value;
        let dl = general_graph_options['display_legend'].value;

        // Apply the settings
        this.dvbg_chart.options.scales.xAxes.forEach(x => x.display = dha);
        this.dvbg_chart.options.scales.yAxes.forEach(x => x.display = dva);

        this.dvbg_chart.options.legend.display = dl;

        this.dvbg_chart.update();
    }

    getSendableObject() {
        let ret = {};

        ret.chartsettings = this.dvbg_chart.options;
        ret.analyze = this.analyze.toString();

        return ret;
    }

    fromSendableObject(obj) {
        this.setDataAnalysisFunction(new Function(obj.analyze));
        return this;
    }


    // Provides an array of some pre-selected colors.
    static colorList() {
        let ret = new Array();

        //ret.push("red");
        ret.push("rgba(255, 0, 0, 0.7)");
        //ret.push("green");
        ret.push("rgba(0, 128, 0, 0.7)");
        //ret.push("blue");
        ret.push("rgba(0, 0, 255, 0.7)");
        //ret.push("#E8ADAA"); // rose
        ret.push("rgba(232, 173, 170, 0.7)");
        //ret.push("cyan");
        ret.push("rgba(0, 255, 255, 0.7)");
        //ret.push("black");
        ret.push("rgba(0, 0, 0, 0.7)");
        //ret.push("aqua");
        ret.push("rgba(16, 255, 255, 0.7)");
        //ret.push("lime");
        ret.push("rgba(0, 255, 0, 0.7)");
        //ret.push("fuchsia");
        ret.push("rgba(255, 0, 255, 0.7)");
        //ret.push("navy");
        ret.push("rgba(0, 0, 128, 0.7)");
        //ret.push("purple");
        ret.push("rgba(128, 0, 128, 0.7)");
        //ret.push("gray");
        ret.push("rgba(128, 128, 128, 0.7)");

        return ret;
    }

    linkMouse(parent) {
        parent.addChild(this.dvbg_clickable);
        return this;
    }

    changeGraphOptions(func) {
        func(this.dvbg_chart);
        return this;
    }

    draw(ctx, datablock, scale) {
        if(scale == 0.)
            return;

        let update = this.prev_scale != scale;
        this.prev_scale = scale;
        this.dvbg_canvas_scaler.style.height = Math.max(this.size.y * scale, 1) + "px";
        this.dvbg_canvas_scaler.style.width = Math.max(this.size.x * scale, 1) + "px";


        if(this.update_data)
        {
            this.dvbg_chart_data = this.analyze(datablock);
            let datacpy = JSON.parse(JSON.stringify(this.dvbg_chart_data)); 
            this.dvbg_chart.data = datacpy;
            this.update_data = false;
            this.dvbg_chart.update();
        }

        if(datablock == this.dvbg_cached_data) {
            // no update needed
        }
        else {
            this.dvbg_cached_data = datablock;

            this.dvbg_chart.update();
        }

        if(update) {
            // Force refresh
            this.dvbg_chart.resize();
        }
        
        
        ctx.save();

        ctx.beginPath();

        // Blit this onto our canvas
        ctx.drawImage(this.dvbg_canvas, this.topleft.x, this.topleft.y, this.size.x, this.size.y);

        ctx.restore();

    }

}

class RU_DataViewNumberBlock extends RU_DataView {
    constructor() {
        super();

        this.opt = {
            display_title: true,
            text_align: "center",
            draw_bar: undefined, // Defines where to draw a bar. Options: "left", "right", "top", "bottom" or any combination thereof
            padding: undefined
        };

        this.dvnb_cached_data = undefined;
    }

    setTitle(title) {
        this.title = title;
        return this;
    }

    setOptions(opt) {
        this.opt = opt;
        return this;
    }

    setStringFormatter(func) {
        this.stringformatter = func;
    }

    static percent2color(percent) {
        let step = 255 / 100.;

        let val = Math.floor(step * percent) * 2;

        let ret = 0;

        if(percent <= 50) {
            ret = 0x0000ff00 + 0x00010000 * val;
        }
        else {
            ret = 0x00ff0000 + 0x00000100 * Math.round(255.0 - val);
        }
        return '#' + ret.toString(16).padStart(6, '0');
    }
    setColorScaling(func) {
        this.color_scaling_func = func;
        return this;
    }
    draw(ctx, datablock, scale) {

        if(this.update_data)
        {
            this.dvnb_cached_data = this.analyze(datablock);
            this.update_data = false;
        }

        if(scale == 0.)
            return; //Nothing to do if we are size 0
        
        ctx.save();

        ctx.beginPath();
        ctx.textAlign="center";
        ctx.font= Math.round(16 * scale).toString() + "px sans-serif";
        
        let textheight = ctx.measureText('M').width; // Approximating text height by width of M (which should be about square)
        let x = this.topleft.x + this.size.x / 2;
        let y = this.topleft.y + textheight;

        if(this.opt.display_title) {
            ctx.fillText(this.title, x, y); // Set the title
        }

        // We'll only be interested in balance_max.
        let p = this.dvnb_cached_data;

        ctx.beginPath();
        ctx.textAlign = this.opt.text_align;
        ctx.font="bold " + (50 * scale).toString() + "px sans-serif";
        ctx.fillStyle=RU_DataViewNumberBlock.percent2color(this.color_scaling_func(p));
        textheight = ctx.measureText('M').width;
        ctx.textBaseline = "middle";

        let left_pad = 0;
        if(this.opt.padding) {
            if(this.opt.padding.left) {
                left_pad = this.opt.padding.left;
            }
        }
        if(this.opt.text_align == "center")
            ctx.fillText(String(p) + "%", x + left_pad, this.topleft.y + this.size.y / 2);
        else if(this.opt.text_align == "left")
            ctx.fillText(String(p) + "%", this.topleft.x + left_pad, this.topleft.y + this.size.y / 2);
        
        
        if(this.opt.draw_bar) {
            for(x of this.opt.draw_bar) {
                if(x == "left") {
                    ctx.beginPath();
                    ctx.strokeStyle="black";
                    ctx.lineWidth = 3;
                    ctx.moveTo(this.topleft.x, this.topleft.y);
                    ctx.lineTo(this.topleft.x, this.topleft.y + this.size.y * scale);
                    ctx.stroke();
                }
                else {
                    console.log("#TODO: Implement this other DataViewNumberBlock bar: " + x);
                }
            }
        }

        this.drawOverlay(ctx, scale);

        ctx.restore();
    }
}

// Class to suggest actions based on input values
class RU_DataViewSuggestedActionBlock extends RU_DataView {
    constructor() {
        super();

        this.opt = {
            
        };

        let parent_this = this;
        this.dvsa_clickable = new class extends Clickable {
            constructor() {
                super();
            }

            is_inside(pos) {
                if(parent_this.topleft == null) return false;
                return (parent_this.topleft.x < pos.x && pos.x < parent_this.topleft.x + parent_this.size.x) && (parent_this.topleft.y < pos.y && pos.y < parent_this.topleft.y + parent_this.size.y);
            }

            getButtonIndex(pos) {
                // Wrapper to look up the index
                let tl_x = parent_this.topleft.x;
                let tl_y = parent_this.topleft.y;

                let size_x = parent_this.size.x;
                let size_y = parent_this.size.y;

                let hp = parent_this.heightpadding;

                let local_pos = {x: pos.x - tl_x, y: pos.y - tl_y};

                if(local_pos.x < 0 || local_pos.x > size_x) return -1;
                if(local_pos.y < 0 || local_pos.y > size_y) return -1;

                // We can just divide through ellipseheight (approximated by a rect) + heightpadding
                let eachheight = parent_this.textheight + parent_this.heightpadding;

                let index = Math.floor(local_pos.y / eachheight);

                return index;
            }

            onUpdateMove(pos) {
                if(!this.enable_func()) return false;
                if(!this.is_inside(pos))
                {
                    parent_this.dvsa_button_hovered = -1;
                    return false;
                }
                
                let index = this.getButtonIndex(pos);
                // Pass to parent
                parent_this.buttonHovered(index);

                // No children here
                return true;
            }

            onUpdateClick(pos) {
                if(!this.enable_func()) return false;
                if(!this.is_inside(pos))
                    return false;
                
                return true;
            }
        };

        this.hints = {}; // Dict of condition => hint text

        this.dvsab_cached_data = [];

        this.heightpadding = 20;

        //this.analyze() expected return value:
        //list of condition strings that evaluated to 'true' (those will be displayed)
    }

    buttonHovered(index) {
        this.dvsa_button_hovered = index;
    }

    setTitle(title) {
        this.title = title;
        return this;
    }

    setHint(cond, text) {
        this.hints[cond] = text;
        return this;
    }

    setOptions(opt) {
        this.opt = opt;
        return this;
    }

    setStringFormatter(func) {
        this.stringformatter = func;
    }

    linkMouse(parent) {
        parent.addChild(this.dvsa_clickable);
        return this;
    }

    
    draw(ctx, datablock, scale) {

        if(this.update_data)
        {
            this.dvsab_cached_data = this.analyze(datablock);
            ObjectHelper.assert("Data evaluated to undefined", this.dvsab_cached_data != undefined);
            this.update_data = false;
        }

        if(scale == 0.)
            return; //Nothing to do if we are size 0
        
        ctx.save();


        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.strokeStyle = "black";
        ctx.lineWidth = 3;


        ctx.font = Math.round(12 * scale).toString() + "px sans-serif";


        let textheight = ctx.measureText("M").width; // Approximate height by width
        this.textheight = textheight;

        let xpos = this.topleft.x + this.size.x / 2.0;
        let ypos = this.topleft.y + textheight;

        let i = 0;
        for(let x of this.dvsab_cached_data) {
            let display_text = this.hints[x];

            ctx.beginPath();
            if(i == this.dvsa_button_hovered) {
                ctx.strokeStyle = "red";
            }
            else {
                ctx.strokeStyle = "black";
            }
            ctx.fillText(display_text, xpos, ypos);
            ctx.ellipse(xpos, ypos, this.size.x / 2.0, (textheight + this.heightpadding / 2.0) / 2.0, 0.0, 0.0, 2 * Math.PI);
            ctx.stroke();


            ypos += (textheight + this.heightpadding) * scale;
            ++i;
        }

        
        this.drawOverlay(ctx, scale);

        ctx.restore();
    }
}


// This is a text-based "form-layout"-style dataview
class RU_DataViewFormLayout extends RU_DataView {
    constructor() {
        super();
    }

    draw(ctx, datablock_in, scale) {

        if(scale == 0.) return;

        ctx.save();
        ctx.beginPath();

        // Transform to drawable
        let datablock = this.analyze(datablock_in);


        let fontsize = datablock.fontsize;
        let form_entries = datablock.rows;

        let padding = datablock.padding;

        let realfontsize = Math.round(fontsize * scale);

        ctx.font = realfontsize + "px sans-serif";

        // Determine the max width of the left side
        let name_width = max_func(form_entries.map(x => x.title), x => ctx.measureText(x + ": ").width);

        let text_height = ctx.measureText('M').width;
        let line_spacing = 10;

        // Determine the length of the right side (to allow for alignment)
        let val_width = max_func(form_entries.map(x => x.val), x => ctx.measureText(x).width);

        let i = 0;
        for(let line of form_entries) {
            let name = line.title;
            let val = line.value;

            // Draw title
            ctx.beginPath();
            ctx.textAlign = "left";
            ctx.fillText(name, this.topleft.x + padding.left * scale, this.topleft.y + padding.top + (i+1) * (text_height + line_spacing));

            // Align the text on the right side
            ctx.beginPath();
            ctx.textAlign = "right";
            // Draw value
            ctx.fillText(val, this.topleft.x + (this.size.x - padding.right) * scale, this.topleft.y + padding.top + (i+1) * (text_height + line_spacing));

            i++;
        }
        ctx.restore();
    }
}

class Bracket extends Clickable {
    constructor(ctx) {
        super();
        this.ctx = ctx;
        this.buttons = [];

        this.start = new Pos(0, 0);
        this.end = new Pos(0, 0);
        this.offset = 0;
        this.startoffset = 0;

        this.listeners = [];

        this.button_alpha = 1.;
        this.bracket_alpha = 1.;
    }

    hide() {
        this.button_alpha = 0.;
        this.bracket_alpha = 0.;
    }

    show(opacity = 1.0) {
        this.button_alpha = opacity;
        this.bracket_alpha = opacity;
    }

    setupEventListeners(canvas_draw_mgr=undefined) {
        let canvas = this.ctx.canvas;

        let mouseXtrans = x => x;
        let mouseYtrans = y => y;

        if(window.get_zoom != undefined) {
            mouseXtrans = x => x / window.get_zoom();
            mouseYtrans = y => y / window.get_zoom();
        }

        if(canvas_draw_mgr != undefined) {
            mouseXtrans = x => canvas_draw_mgr.mapPixelToCoordsX(x);
            mouseYtrans = y => canvas_draw_mgr.mapPixelToCoordsY(y);
        }

        let mm_lis = e => {
            this.onUpdateMove(new Pos(mouseXtrans(e.offsetX), mouseYtrans(e.offsetY)));
        };
        let c_lis = e => {
            this.onUpdateClick(new Pos(mouseXtrans(e.offsetX), mouseYtrans(e.offsetY)), e.button);
        };
        let dc_lis = e => {
            this.onUpdateDoubleClick(new Pos(mouseXtrans(e.offsetX), mouseYtrans(e.offsetY)), e.button);
        };

        // Add all event listeners to an array (this allows for easier un-setting)
        this.listeners.push(['mousemove', mm_lis]);
        this.listeners.push(['click', c_lis]);
        this.listeners.push(['dblclick', dc_lis]);

        let ctx = this.ctx;
        for(let lis of this.listeners) {
            ctx.canvas.addEventListener(...lis);
        }
    }

    destroy() {
        super.destroy();

        for(let lis of this.listeners) {
            this.ctx.canvas.removeEventListener(...lis);
        }

        this.buttons = null;
        this.start = null;
        this.end = null;
        this.ctx = null;
    }

    drawEx(start, end, offset, startoffset, animate, updatefunc) {
        this.start = start;
        this.end = end;
        this.offset = offset;
        this.startoffset = startoffset;
        this.animate = animate;
        this.updatefunc = updatefunc;

        this.draw();
    }

    is_inside_buttons(pos) {
        if(this.button_alpha == 0.) {
            // Hidden buttons should not be clickable
            return false;
        }

        let max_x = 0;
        let max_y = 0;

        for (let b of this.buttons) {
            if(b == undefined) {
                console.log("Undefined button!");
            }
            else if(b.topleft == undefined) {
                console.log("Undefined topleft");
            }
            max_x = Math.max(b.topleft.x + b.size.x, max_x);
            max_y = Math.max(b.topleft.y + b.size.y, max_y);
        }

        if (pos.inRect(this.start, new Pos(max_x, max_y))) {
            return true;
        }

        return false;
    }

    bracket_clicked() {
        if(this.button_alpha == 0.0) {
            this.button_alpha = 1.0;
        }
        else {
            this.button_alpha = 0.0;
        }
    }

    is_inside_bracket(pos) {
        const tol = 5;
        if(pos.x < this.start.x + this.startoffset - tol || pos.x > this.start.x + this.offset + tol) {
            return false;
        }
        if(pos.y < this.start.y - tol || pos.y > this.end.y + tol) {
            console.log("outside y");
            return false;
        }
        
        return true;
    }

    is_inside(pos) {
        return this.is_inside_buttons(pos);
    }

    onUpdateClick(pos, mb) {
        if(this.is_inside_bracket(pos, mb)) {
            this.bracket_clicked();
            return true;
        }

        if(this.button_alpha == 0.0) {
            // Buttons are hidden, don't process events on them
            return false;
        }

        return super.onUpdateClick(pos, mb);
    }

    onUpdateDoubleClick(pos, mb) {
        if(this.is_inside_bracket(pos, mb)) {
            this.bracket_clicked();
            return true;
        }

        if(this.button_alpha == 0.0) {
            // Buttons are hidden, don't process events on them
            return false;
        }

        return super.onUpdateDoubleClick(pos, mb);
    }

    draw() {
        let start = this.start;
        let end = this.end;
        let offset = this.offset;
        let startoffset = this.startoffset;
        let ctx = this.ctx;

        if(this.animate && this.updatefunc == null)
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

        let button_offset = 20;

        ctx.save();
        ctx.beginPath();
        ctx.lineWidth = 5;
        ctx.strokeStyle = "rgba(0,0,255," + this.bracket_alpha.toString() + ")";
        ctx.moveTo(start.x + startoffset, start.y);

        ctx.lineTo(start.x + offset, start.y);
        ctx.stroke();

        ctx.lineTo(start.x + offset, end.y);
        ctx.stroke();

        ctx.lineTo(end.x + startoffset, end.y);
        ctx.stroke();

        ctx.restore();

        let oldalpha = ctx.globalAlpha;
        ctx.globalAlpha = this.button_alpha;

        if(this.button_alpha > 0.0) {
            // only draw if it is visible to humans.
            this.drawButtons(new Pos(start.x + offset + button_offset, start.y));
        }

        ctx.globalAlpha = oldalpha;


        if(this.animate) {
            let _this = this;

            
            window.requestAnimationFrame(() => {
                if(this.updatefunc == null) {
                    _this.draw();
                }
                else {
                    this.updatefunc();
                }
            });
        }
    }

    drawButtons(startpos) {
        let xpos = startpos.x;
        let ypos = startpos.y;

        startpos = null;

        let size = 30;
        let b;
        let yoffset = 0;
        for (b of this.buttons) {
            let tl = new Pos(xpos, ypos);
            let tl_higher = new Pos(xpos, ypos + yoffset); // Slightly offset position to avoid collisions

            let tlsel = tl;
            if(b.button_subwindow_state == "open") {
                // Select the offset position (so that the buttons / windows don't overlap)
                yoffset -= size + 5;
            }
            tlsel = tl_higher;
            b.draw(tlsel, new Pos(size, size), this.ctx);
            xpos += size + 10;
        }
    }

    addButton(b) {
        this.buttons.push(b);
        this.addChild(b);
    }

};

var _canvas_manager_counter = 0;

// Class to manage drawing of all resources. This ensures that one object 
// does not overwrite another by mistake.
class CanvasDrawManager {
    static counter() {
        return _canvas_manager_counter++;
    }
    constructor(ctx, ref_global_state) {
        this.ctx = ctx;
        this.anim_id = null;
        this.drawables = [];
        this.ref_global_state = ref_global_state;
        this.indices = [];

        this.request_scale = false;
        this.scale_factor = {x: 1, y: 1};
        this.last_scale_factor = {x: 1, y: 1};
        
        this._destroying = false;

        this.scale_origin = {x: 0, y: 0};

        this.contention = 0;

        this._svg = document.createElementNS("http://www.w3.org/2000/svg",'svg');

        this.user_transform = this._svg.createSVGMatrix();

        this.addCtxTransformTracking();
    }

    svgPoint(x, y) {
        let pt  = this._svg.createSVGPoint();
        pt.x=x; pt.y=y;
        return pt;
    }

    applyUserTransform() {
        let ut = this.user_transform;
        this.ctx.setTransform(ut.a, ut.b, ut.c, ut.d, ut.e, ut.f);
    }

    get translation() {
        return { x: this.user_transform.e, y: this.user_transform.f };
    }

    addCtxTransformTracking() {
        /* This function is a hack to provide the non-standardized functionality
        of getting the current transform from a RenderingContext.
        When (if) this is standardized, the standard should be used instead.
        This is made for "easy" transforms and does not support saving/restoring
        */

        let svg = document.createElementNS("http://www.w3.org/2000/svg",'svg');
        this.ctx._custom_transform_matrix = svg.createSVGMatrix();
        // Save/Restore is not supported.

        let checker = () => {
            console.assert(!isNaN(this.ctx._custom_transform_matrix.f));
        };
        let _ctx = this.ctx;
        let scale_func = _ctx.scale;
        _ctx.scale = function(sx,sy) {
            _ctx._custom_transform_matrix = _ctx._custom_transform_matrix.scaleNonUniform(sx,sy);
            checker();
            return scale_func.call(_ctx, sx, sy);
        };
        let translate_func = _ctx.translate;
        _ctx.translate = function(sx,sy) {
            _ctx._custom_transform_matrix = _ctx._custom_transform_matrix.translate(sx,sy);
            checker();
            return translate_func.call(_ctx, sx, sy);
        };
        let rotate_func = _ctx.rotate;
        _ctx.rotate = function(r) {
            _ctx._custom_transform_matrix = _ctx._custom_transform_matrix.rotate(r * 180.0 / Math.PI);
            checker();
            return rotate_func.call(_ctx, r);
        };
        let transform_func = _ctx.scale;
        _ctx.transform = function(a,b,c,d,e,f){
			let m2 = svg.createSVGMatrix();
			m2.a=a; m2.b=b; m2.c=c; m2.d=d; m2.e=e; m2.f=f;
            _ctx._custom_transform_matrix = _ctx._custom_transform_matrix.multiply(m2);
            checker();
			return transform_func.call(_ctx,a,b,c,d,e,f);
		};

        let setTransform_func = _ctx.setTransform;
		_ctx.setTransform = function(a,b,c,d,e,f){
			_ctx._custom_transform_matrix.a = a;
			_ctx._custom_transform_matrix.b = b;
			_ctx._custom_transform_matrix.c = c;
			_ctx._custom_transform_matrix.d = d;
			_ctx._custom_transform_matrix.e = e;
            _ctx._custom_transform_matrix.f = f;
            checker();
			return setTransform_func.call(_ctx,a,b,c,d,e,f);
		};
		
		_ctx.custom_inverseTransformMultiply = function(x,y){
            let pt  = svg.createSVGPoint();
            pt.x=x; pt.y=y;
            checker();
			return pt.matrixTransform(_ctx._custom_transform_matrix.inverse());
		}
    }

    destroy() {
        this._destroying = true;
        this.clearDrawables();
    }

    addDrawable(obj) {
        this.drawables.push(obj);
        this.indices.push({"c": CanvasDrawManager.counter(), "d": obj});
    }

    removeDrawable(drawable) {
        this.drawables = this.drawables.filter(x => x != drawable);
    }

    clearDrawables() {
        for(let x of this.drawables) {
            x.destroy();
        }
        this.drawables = [];
        this.indices = [];
    }

    scale(diff, x=0, y=0) {
        
        if(this.request_scale || Math.abs(diff) < 0.0001 || this.contention > 0) {
            console.log("Blocking potential race");
            return;
        }
        this.contention++;
        this.request_scale = true;

        this.scale_origin.x = x;
        this.scale_origin.y = y;
        this.scale_factor.x += diff;
        this.scale_factor.y += diff;
        
        this.scale_factor.x = Math.max(0.001, this.scale_factor.x);
        this.scale_factor.y = Math.max(0.001, this.scale_factor.y);
        {
            let sv = diff < 0 ? 0.9 : 1.1;
            let pt = this.svgPoint(this.scale_origin.x, this.scale_origin.y).matrixTransform(this.user_transform.inverse());
            this.user_transform = this.user_transform.translate(pt.x, pt.y);
            this.user_transform = this.user_transform.scale(sv, sv, 1, 0, 0, 0);
            this.user_transform = this.user_transform.translate(-pt.x, -pt.y);
        }
        this.contention--;
    }

    translate(x, y) {
        this.user_transform = this.user_transform.translate(x / this.user_transform.a, y / this.user_transform.d);
    }

    mapPixelToCoordsX(xpos) {
        return this.svgPoint(xpos, 0).matrixTransform(this.user_transform.inverse()).x;
    }

    mapPixelToCoordsY(ypos) {
        return this.svgPoint(0, ypos).matrixTransform(this.user_transform.inverse()).y;
    }

    getScale() {
        ObjectHelper.assert("Uniform scale", this.scale_factor.x == this.scale_factor.y);
        return this.noJitter(this.scale_factor.x); // We don't allow non-uniform scaling.
    }

    getLastScale() {
        return this.noJitter(this.last_scale_factor.x); // We don't allow non-uniform scaling.
    }

    noJitter(x) {
        x = parseFloat(x.toFixed(3));
        x = Math.round(x * 100) / 100;
        return x;
    }


    draw() {
        if(this._destroying) {
            return;
        }
        if(this.contention > 0) return;
        this.contention += 1;
        let ctx = this.ctx;

        // Clear with default transform
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.restore();

        let mx = 0;
        let my = 0;
        if(this.request_scale && this.contention == 1) {
            
            mx = this.mapPixelToCoordsX(this.scale_origin.x);
            my = this.mapPixelToCoordsY(this.scale_origin.y);

            // Reset the translation
            this.applyUserTransform();
            this.request_scale = false;
        }
        else
        {
            mx = this.mapPixelToCoordsX(this.scale_origin.x);
            my = this.mapPixelToCoordsY(this.scale_origin.y);
            
            
        }
        this.applyUserTransform();
            
        

        this.ref_global_state.drawSDFG();
        for(let d of this.drawables) {
            d.draw();
        }

        if(false) // Comment this line to show debug values at cursor position
        {
            ctx.fillText("(" + mx.toFixed(1) + "|" + my.toFixed(1) + ")",  mx, my);
            ctx.fillText("(" + this.translation.x.toFixed(1) + "|" + this.translation.y.toFixed(1) + ")",  mx, my + 10);
            ctx.fillText("s:" + this.getScale().toFixed(1),  mx, my + 20);
        }
        this.contention -= 1;
    }

    draw_async() {
        let _this = this;
        this.anim_id = window.requestAnimationFrame(() => _this.draw());
    }
}

export { CanvasDrawManager, Bracket, Button, Layout, Pos, min_func, max_func, max_func_obj,
RU_DataView, RU_DataViewBarGraph, RU_DataViewFormLayout, RU_DataViewNumberBlock, RU_DataViewSuggestedActionBlock, RU_DataViewText,
createImageDownload, DataBlock };