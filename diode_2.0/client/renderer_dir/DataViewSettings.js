// This is a file that provides functionality of View Settings
import { ClientSide } from "./windowing.js";
import {ObjectHelper, MathHelper} from "./datahelper.js";

class SettingsWindow extends ClientSide {

    parseSettings(data, parentObject) {
        "use strict";
        let set = data;
        

        console.log("set: " + JSON.stringify(set));
        for(let k of ObjectHelper.listKeys(set)) {
            let val = set[k];
            console.log("With key " + k);
            if(val.type == "code") {
                let label = window.document.createElement('label');
                let codearea = window.document.createElement('textarea');
                label.id = "label_" + k;
                codearea.id = "value_" + k;
                label.htmlFor = codearea.id;
                label.innerHTML = k;

                codearea.textContent = val.value;

                parentObject.appendChild(label);
                parentObject.appendChild(window.document.createElement("br"));
                parentObject.appendChild(codearea);
            }
            else if(val.type == "group") {
                // Subgroup requested...
                let fieldset = window.document.createElement("fieldset");
                fieldset.id = "value_" + k;
                let legend = window.document.createElement("legend");
                legend.innerHTML = val["description"];

                console.log("Starting recursion");
                // Recurse.
                this.parseSettings(val.value, fieldset);

                fieldset.appendChild(legend);

                parentObject.appendChild(fieldset);

            }
            else if(val.type == "bool") {
                // We have to add a checkbox
                
                let checkbox = window.document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.id = "value_" + k;
                checkbox.checked = val.value;

                
                let label = window.document.createElement("label");
                label.htmlFor = checkbox.id;
                label.innerHTML = k;

                parentObject.appendChild(label);
                
                parentObject.appendChild(checkbox);
                parentObject.appendChild(window.document.createElement("br"));
            }
            else {
                console.log("Unimplemented type " + val.type + " encountered.");
            }
        }
    }

    restoreSettings(set, parentObject) {
        "use strict";
        let retset = {};
        for(let k of ObjectHelper.listKeys(set)) {
            let val = set[k];

            let _value = {};

            let elem = window.document.getElementById('value_' + k);

            ObjectHelper.assert("Element must exist", elem);

            if(val.type == "code") {
                _value = {
                    type: val.type,
                    value: elem.value.toString()
                };
                
            }
            else if(val.type == "bool") {
                _value = {
                    type: val.type,
                    value: elem.checked
                };
                console.log("The id of elem is " + elem.id);
                console.log("value of " + k + " is " + elem.value);
                
            }
            else if(val.type == "group") {
                // We have to recurse
                let sub = this.restoreSettings(val.value, elem);

                _value = {
                    type: val.type,
                    value: sub
                };
            }
            retset[k] = _value;
        }

        return retset;
    }

    constructor(thiswindow) {
        super(thiswindow, data => {
            if(data.type == "settings-data") {
                
                let form = window.document.createElement('form');
                form.onsubmit = x => x.preventDefault();
                this.parseSettings(data.data, form);

                let savebutton = window.document.createElement('button');
                savebutton.innerText = "Save";
                let _this = this;
                savebutton.onclick = x => {
                    console.log("\"Save\" clicked");

                    let retset = this.restoreSettings(data.data, form);
                    
                    // Send information back
                    _this.passMessage({
                        type: "save-settings",
                        data: retset
                    });
                };
                form.appendChild(window.document.createElement("br"));
                form.appendChild(savebutton);

                window.document.body.appendChild(form);


                
            }
            else {
                console.log("Undefined type encountered " + JSON.stringify(data));
            }

            
        });
    }
}

export { SettingsWindow };