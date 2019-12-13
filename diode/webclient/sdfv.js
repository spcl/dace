var fr;
var file = null;
var renderer = null;

function init_sdfv(sdfg) {
    $('input[type="file"]').change(function(e){
        if (e.target.files.length < 1)
            return;
        file = e.target.files[0];
        reload_file();
    });
    $('#reload').click(function(e){
        reload_file();
    });

    if (sdfg !== null)
        renderer = new SDFGRenderer(sdfg, document.getElementById('contents'),
                                    mouse_event);
}

function reload_file() {
    if (!file)
        return;
    fr = new FileReader();
    fr.onload = file_read_complete;
    fr.readAsText(file);
}

function file_read_complete() {
    let sdfg = parse_sdfg(fr.result);
    if (renderer)
        renderer.destroy();
    renderer = new SDFGRenderer(sdfg, document.getElementById('contents'), mouse_event);
}

function mouse_event(evtype, event, mousepos, elements, renderer, elem) {
    if (evtype === 'click' || evtype === 'dblclick') {
        if (elem) {
            // Change header
            document.getElementById("sidebar-header").innerText = elem.type() + " " + elem.label();

            // Change contents
            let contents = document.getElementById("sidebar-contents");
            let html = "";
            if (elem instanceof Edge && elem.data.type === "Memlet") {
                let sdfg_edge = elem.sdfg.nodes[elem.parent_id].edges[elem.id];
                html += "<h4>Connectors: " + sdfg_edge.src_connector + " &rarr; " + sdfg_edge.dst_connector + "</h4>";
            }
            html += "<hr />";

            for (let attr of Object.entries(elem.attributes())) {
                if (attr[0] === "layout" || attr[0] === "sdfg") continue;
                html += "<b>" + attr[0] + "</b>:&nbsp;&nbsp;";
                html += sdfg_property_to_string(attr[1], attr[0]) + "</p>";
            }
            contents.innerHTML = html;
            document.getElementById("sidebar").style.display = "block";
        } else {
            document.getElementById("sidebar-contents").innerHTML = "";
            document.getElementById("sidebar-header").innerText = "Nothing selected";
        }
    }
}

function close_menu() {
  document.getElementById("sidebar").style.display = "none";
}