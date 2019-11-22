

class Appearance  {

    constructor(config) {
        if(config === null) {
            config = {};
            config.style = Appearance.default();
        }
        if(typeof(config) == 'string')
            config = JSON.parse(config);


        this._change_callback = null;
        this._config = config;
        this.insertStylesheet(config.style);

        this._constructor_values = {};
        if(config.vals != undefined) {
            this._constructor_values = config.vals;
        }
    }

    static getClassProperties(cssclassname) {
        let elem = document.createElement('div');
        elem.classList = cssclassname;
        elem.style = "width: 0; height: 0;";
        document.body.appendChild(elem);

        let style = JSON.parse(JSON.stringify(window.getComputedStyle(elem)));
        document.body.removeChild(elem);

        return style;
    }


    static fonts() {
        return [
            "Georgia", "Palatino", "Times",
            "Arial", "ArialBlack", "ComicSans", "Impact", "Lucida", "Tahoma", "Trebuchet", "Verdana",
            "Courier", "LucidaConsole"
        ];
    }

    static font(shortname) {
        switch(shortname) {
            // Serif
            case "Georgia": return 'Georgia, serif';
            case "Palatino": return '"Palatino Linotype", "Book Antiqua", Palatino, serif';
            case "Times": return '"Times New Roman", Times, serif';

            // Sans-Serif
            case "Arial": return "Arial, Helvetica, sans-serif";
            case "ArialBlack": return '"Arial Black", Gadget, sans-serif';
            case "ComicSans": return '"Comic Sans MS", cursive, sans-serif';
            case "Impact": return 'Impact, Charcoal, sans-serif';
            case "Lucida": return '"Lucida Sans Unicode", "Lucida Grande", sans-serif';
            case "Tahoma": return 'Tahoma, Geneva, sans-serif';
            case "Trebuchet": return '"Trebuchet MS", Helvetica, sans-serif';
            case "Verdana": return 'Verdana, Geneva, sans-serif';

            // Monospace
            case "Courier": return '"Courier New", Courier, monospace';
            case "LucidaConsole": return '"Lucida Console", Monaco, monospace';

            default: throw "Unknown font (not websafe?)"
        }
    }

    setOnChange(func) {
        this._change_callback = func;
    }

    setChanged() {
        if(this._change_callback != null) {
            this._change_callback(this);
        }
    }

    apply() {
        this.insertStylesheet(this.getCSS());
    }

    insertStylesheet(stylestring) {

        // This function does work in all browsers but IE (<= 8).
        let stylesheet = document.getElementById("appearance-stylesheet");
        let update = true;
        if(stylesheet == null || stylesheet == undefined) {
            stylesheet = document.createElement('style');
            update = false;
            stylesheet.id = "appearance-stylesheet";
            stylesheet.type = "text/css";
        }
        
        stylesheet.innerText = stylestring;

        if(!update) {
            document.head.appendChild(stylesheet);
        }

        this.setChanged();
    }

    setValue(css_type, css_value) {
        this._constructor_values[css_type] = css_value;
        return this;
    }
    setFont(shortname) {
        let f = Appearance.font(shortname);
        this._constructor_values['font-family'] = f;
        return this;
    }

    toString() {
        return JSON.stringify(this._config);
    }

    getCSS() {
        let inner = "";

        for(let x of Object.entries(this._constructor_values)) {
            inner += x[0] + ": " + x[1] + ";"; 
        }

        let ret = ".diode_appearance { ";
        ret += inner;
        ret += "}";

        // Since the lm_content overrides the background, we re-override the background
        // Note that this is legal and works because local <style></style> always overrides imports from other files
        ret += ".lm_content { ";
        ret += inner;
        ret += "}";
        return ret;
    }

    toStorable() {
        return {
            style: this.getCSS(),
            vals: this._constructor_values
        };
    }

    setFromAceEditorTheme(theme_name) {
        /*
            Sets background and foreground according to the theme identifed by `theme_name`
            Note: The css class selector used to determine current values is: '.ace-' + `theme_name`
        */

        let s = Appearance.getClassProperties("ace-" + theme_name);

        let bgc = s['background-color'];
        let bg = s['background'];
        if(bgc) this.setValue('background-color', bgc);
        if(bg) this.setValue('background', bg + " !important");
        this.setValue('color', s['color']);

        this.insertStylesheet(this.getCSS());
    }

    static default() {
        /*
            Builds a default configuration
        */
        let cssstring = `
        .diode_appearance {
            font-family: ` + Appearance.font('Arial') + `
        }
        `;


        return cssstring;
    }

}

export {Appearance}