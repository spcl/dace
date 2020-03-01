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
    $('#outline').click(function(e){
        if (renderer)
            setTimeout(() => outline(renderer, renderer.graph), 1);
    });
    $('#search-btn').click(function(e){
        if (renderer)
            setTimeout(() => {find_in_graph(renderer, renderer.graph, $('#search').val(),
                                                    $('#search-case')[0].checked);}, 1);
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
    close_menu();
}

function find_recursive(graph, query, results, case_sensitive) {
    for (let nodeid of graph.nodes()) {
        let node = graph.node(nodeid);
        let label = node.label();
        if (!case_sensitive)
            label = label.toLowerCase();
        if (label.indexOf(query) !== -1)
            results.push(node);
        // Enter states or nested SDFGs recursively
        if (node.data.graph)
            find_recursive(node.data.graph, query, results, case_sensitive);
    }
    for (let edgeid of graph.edges()) {
        let edge = graph.edge(edgeid);
        let label = edge.label();
        if (!case_sensitive)
            label = label.toLowerCase();
        if (label.indexOf(query) !== -1)
            results.push(edge);
    }
}

function find_in_graph(renderer, sdfg, query, case_sensitive=false) {
    // Modify sidebar header
    document.getElementById("sidebar-header").innerText = 'Search Results for "' + query + '"';

    let results = [];
    if (!case_sensitive)
        query = query.toLowerCase();
    find_recursive(sdfg, query, results, case_sensitive);

    // Zoom to bounding box of all results first
    if (results.length > 0)
        renderer.zoom_to_view(results);

    // Show clickable results in sidebar
    let sidebar = document.getElementById("sidebar-contents");
    sidebar.innerHTML = '';
    for (let result of results) {
        let d = document.createElement('div');
        d.className = 'context_menu_option';
        d.innerHTML = result.type() + ' ' + result.label();
        d.onclick = () => {renderer.zoom_to_view([result])};
        sidebar.appendChild(d);
    }

    // Open sidebar if closed
    document.getElementById("sidebar").style.display = "flex";
}

function outline_recursive(renderer, graph, elements) {
    for (let nodeid of graph.nodes()) {
        let node = graph.node(nodeid);
        let d = document.createElement('div');
        d.className = 'context_menu_option';
        let is_collapsed = node.attributes().is_collapsed;
        is_collapsed = (is_collapsed === undefined) ? false : is_collapsed;
        d.innerHTML = node.type() + ' ' + node.label() + (is_collapsed ? " (collapsed)" : "");
        d.onclick = (e) => {
            renderer.zoom_to_view([node]);

            // Ensure that the innermost div is the one that handles the event
            if (!e) e = window.event;
            e.cancelBubble = true;
            if (e.stopPropagation) e.stopPropagation();
        };

        // Traverse states or nested SDFGs
        if (node.data.graph && !is_collapsed)
            outline_recursive(renderer, node.data.graph, d);

        elements.appendChild(d);
    }
}

function outline(renderer, sdfg) {
    // Modify sidebar header
    document.getElementById("sidebar-header").innerText = 'SDFG Outline';

    let sidebar = document.getElementById("sidebar-contents");
    sidebar.innerHTML = '';

    // Entire SDFG
    let d = document.createElement('div');
    d.className = 'context_menu_option';
    d.innerHTML = '<i class="material-icons" style="font-size: inherit">filter_center_focus</i> SDFG ' +
        renderer.sdfg.attributes.name;
    d.onclick = () => renderer.zoom_to_view();
    sidebar.appendChild(d);

    // Add elements to tree view in sidebar
    outline_recursive(renderer, sdfg, sidebar);

    // Open sidebar if closed
    document.getElementById("sidebar").style.display = "flex";
}

function mouse_event(evtype, event, mousepos, elements, renderer, elem) {
    if (evtype === 'click' || evtype === 'dblclick') {
        if (renderer.menu)
            renderer.menu.destroy();
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
                if (attr[0] === "layout" || attr[0] === "sdfg" || attr[0].startsWith("_meta_")) continue;
                html += "<b>" + attr[0] + "</b>:&nbsp;&nbsp;";
                html += sdfg_property_to_string(attr[1], attr[0]) + "</p>";
            }

            // If access node, add array information too
            if (elem instanceof AccessNode) {
                let sdfg_array = elem.sdfg.attributes._arrays[elem.attributes().data];
                html += "<br /><h4>Array properties:</h4>";
                for (let attr of Object.entries(sdfg_array.attributes)) {
                    if (attr[0] === "layout" || attr[0] === "sdfg" || attr[0].startsWith("_meta_")) continue;
                    html += "<b>" + attr[0] + "</b>:&nbsp;&nbsp;";
                    html += sdfg_property_to_string(attr[1], attr[0]) + "</p>";
                }
            }

            contents.innerHTML = html;
            document.getElementById("sidebar").style.display = "flex";
        } else {
            document.getElementById("sidebar-contents").innerHTML = "";
            document.getElementById("sidebar-header").innerText = "Nothing selected";
        }
    }
}

function close_menu() {
  document.getElementById("sidebar").style.display = "none";
}


var right = document.getElementById('sidebar');
var bar = document.getElementById('dragbar');

const drag = (e) => {
  document.selection ? document.selection.empty() : window.getSelection().removeAllRanges();
  right.style.width = Math.max(((e.view.innerWidth - e.pageX)), 20) + 'px';
}

bar.addEventListener('mousedown', () => {
    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', () => {
        document.removeEventListener('mousemove', drag);
    });
});

