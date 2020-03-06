
class ContextMenu {
    /*
        Implementation for a custom context menu
    */
    constructor(html_content = null) {

        this._html_content = html_content;
        this._click_close_handlers = [];

        this._ = setTimeout(() => {
            this._click_close_handlers = [
                ['click', x => {
                    this.destroy();
                }],
                ['contextmenu', x => {
                    this.destroy();
                }]

            ];

            for(let x of this._click_close_handlers) {
                window.addEventListener(...x);
            }
        }, 30);

        this._options = [];
        this._cmenu_elem = null;
    }

    width() {
        return this._cmenu_elem.offsetWidth;
    }

    visible() { return this._cmenu_elem != null; }

    addOption(name, onselect, onhover=null) {
        this._options.push({
            name: name,
            func: onselect,
            onhover: onhover
        });
    }

    destroy() {
        if (!this._cmenu_elem)
            return;
        // Clear everything

        // Remove the context menu

        document.body.removeChild(this._cmenu_elem);

        for(let x of this._click_close_handlers) {
            window.removeEventListener(...x);
        }

        this._cmenu_elem = null;
    }

    show(x, y) {
        /*
            Shows the context menu originating at point (x,y)
        */

        let cmenu_div = document.createElement('div');
        cmenu_div.id = "contextmenu";
        $(cmenu_div).css('left', x + "px");
        $(cmenu_div).css('top', y + "px");
        cmenu_div.classList.add("context_menu");


        if(this._html_content == null) {
            // Set default context menu

            for(let x of this._options) {

                let elem = document.createElement('div');
                elem.addEventListener('click', x.func);
                elem.classList.add("context_menu_option");

                elem.innerText = x.name;
                cmenu_div.appendChild(elem);
            }

        }
        else {
            cmenu_div.innerHTML = this._html_content;
        }

        this._cmenu_elem = cmenu_div;
        document.body.appendChild(cmenu_div);
    }
}
