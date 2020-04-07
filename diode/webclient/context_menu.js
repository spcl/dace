
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

    addCheckableOption(name, checked, onselect, onhover=null) {
        this._options.push({
            name: name,
            checkbox: true,
            checked: checked,
            func: onselect,
            onhover: onhover
        });
    }

    destroy() {
        if (!this._cmenu_elem)
            return;
        // Clear everything

        // Remove the context menu

        this._cmenu_elem.remove();

        for(let x of this._click_close_handlers) {
            window.removeEventListener(...x);
        }

        this._cmenu_elem = null;
    }

    show(x, y) {
        const sdfv_menu = $('<div>', {
            id: 'sdfv-menu',
            'class': 'sdfv-menu',
            'css': {
                'left': x + 'px',
                'top': y + 'px',
            },
        });

        if (this._html_content === null) {
            for (const option of this._options) {
                let option_html = '';
                let option_click_handler = null;
                if (option.checkbox) {
                    let tick_classes = 'cm material-icons';
                    if (!option.checked)
                        tick_classes += ' hidden';
                    option_html = $('<i>', {
                        'class': tick_classes,
                        'html': 'check',
                    })[0].outerHTML + option.name;
                    option_click_handler = function(element) {
                        option.checked = !option.checked;
                        option.func(element, option.checked);
                    };
                } else {
                    option_html = option.name;
                    option_click_handler = option.func;
                }
                $('<div>', {
                    'class': 'sdfv-menu-option',
                    'html': option_html,
                    'click': option_click_handler,
                }).appendTo(sdfv_menu);
            }
        } else {
            sdfv_menu.html(this._html_content);
        }

        this._cmenu_elem = sdfv_menu;
        sdfv_menu.appendTo(document.body);
    }

}
