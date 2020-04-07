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
    $('#outline-btn').click(function(e){
        if (renderer)
            setTimeout(() => outline(renderer, renderer.graph), 1);
    });
    $('#search-bar').show();
    $('#search-btn').click(function(e){
        if (renderer)
            setTimeout(() => {find_in_graph(renderer, renderer.graph, $('#search').val(),
                                            $('#search-case')[0].checked);}, 1);
    });
    $('#search').on('keydown', function(e) {
        if (e.key == 'Enter' || e.which == 13) {
            if (renderer)
                setTimeout(() => {find_in_graph(renderer, renderer.graph, $('#search').val(),
                                                $('#search-case')[0].checked);}, 1);
            e.preventDefault();
        }
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

// https://stackoverflow.com/a/901144/6489142
function getParameterByName(name) {
    let url = window.location.href;
    name = name.replace(/[\[\]]/g, '\\$&');
    var regex = new RegExp('[?&]' + name + '(=([^&#]*)|&|#|$)'),
        results = regex.exec(url);
    if (!results) return null;
    if (!results[2]) return '';
    return decodeURIComponent(results[2].replace(/\+/g, ' '));
}

function load_sdfg_from_url(url) {
    let request = new XMLHttpRequest();
    request.responseType = 'text'; // Will be parsed as JSON by parse_sdfg
    request.onload = () => {
        if (request.status == 200) {
            let sdfg = parse_sdfg(request.response);
            if (renderer)
                renderer.destroy();
            init_sdfv(sdfg);
        } else {
            alert("Failed to load SDFG from URL");
            init_sdfv(null);
        }
    };
    request.onerror = () => {
        alert("Failed to load SDFG from URL: " + request.status);
        init_sdfv(null);
    };
    request.open('GET', url + ((/\?/).test(url) ? "&" : "?") + (new Date()).getTime(), true);
    request.send();
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
        if (label !== undefined) {
            if (!case_sensitive)
                label = label.toLowerCase();
            if (label.indexOf(query) !== -1)
                results.push(edge);
        }
    }
}

function find_in_graph(renderer, sdfg, query, case_sensitive=false) {
    // Adapt the info-container header to show the search query
    $('#info-title').text('Search Results for "' + query + '":');

    // Recursively search for all matches
    let results = [];
    if (!case_sensitive)
        query = query.toLowerCase();
    find_recursive(sdfg, query, results, case_sensitive);

    // Zoom to bounding box of all results first
    if (results.length > 0)
        renderer.zoom_to_view(results);

    // Show clickable results in the info box
    const info_content = $('#info-contents');
    info_content.html('');
    for (const result of results) {
        $('<div>', {
            'class': 'info-menu-option',
            'click': function () {
                renderer.zoom_to_view([result]);
            },
            'html': result.type() + ' ' + result.label(),
        }).appendTo(info_content);
    }

    // Show the clear info-box button if it's hidden
    $('#info-clear-btn').show();
}

function outline(renderer, sdfg) {
    // Set the info-box title
    $('#info-title').text('SDFG Outline');

    let info_content = document.getElementById('info-contents');
    info_content.innerHTML = '';

    // Entire SDFG
    let d = document.createElement('div');
    d.className = 'info-menu-option';
    d.innerHTML = '<i class="material-icons" style="font-size: inherit">filter_center_focus</i> SDFG ' +
        renderer.sdfg.attributes.name;
    d.onclick = () => renderer.zoom_to_view();
    info_content.appendChild(d);

    let stack = [info_content];
    traverse_sdfg_scopes(sdfg, (node, parent) => {
        // Skip exit nodes when scopes are known
        if (node.type().endsWith('Exit') && node.data.node.scope_entry >= 0) {
            stack.push(null);
            return true;
        }

        // Create element
        let d = document.createElement('div');
        d.className = 'info-menu-option';
        let is_collapsed = node.attributes().is_collapsed;
        is_collapsed = (is_collapsed === undefined) ? false : is_collapsed;
        let node_type = node.type();

        // If a scope has children, remove the name "Entry" from the type
        if (node.type().endsWith('Entry')) {
            let state = node.sdfg.nodes[node.parent_id];
            if (state.scope_dict[node.id] !== undefined) {
                node_type = node_type.slice(0, -5);
            }
        }

        d.innerHTML = node_type + ' ' + node.label() + (is_collapsed ? " (collapsed)" : "");
        d.onclick = (e) => {
            // Show node or entire scope
            let nodes_to_display = [node];
            if (node.type().endsWith('Entry')) {
                let state = node.sdfg.nodes[node.parent_id];
                if (state.scope_dict[node.id] !== undefined) {
                    for (let subnode_id of state.scope_dict[node.id])
                        nodes_to_display.push(parent.node(subnode_id));
                }
            }

            renderer.zoom_to_view(nodes_to_display);

            // Ensure that the innermost div is the one that handles the event
            if (!e) e = window.event;
            e.cancelBubble = true;
            if (e.stopPropagation) e.stopPropagation();
        };
        stack.push(d);

        // If is collapsed, don't traverse further
        if (is_collapsed)
            return false;
                        
    }, (node, parent) => {
        // After scope ends, pop ourselves as the current element 
        // and add to parent
        let elem = stack.pop();
        if (elem)
            stack[stack.length - 1].appendChild(elem);
    });

    // Show the clear info-box button if it's hidden
    $('#info-clear-btn').show();
}

function mouse_event(evtype, event, mousepos, elements, renderer, elem) {
    if (evtype === 'click' || evtype === 'dblclick') {
        if (renderer.menu)
            renderer.menu.destroy();
        if (elem) {
            // Correctly set the info-box title.
            $('#info-title').text(elem.type() + ' ' + elem.label());

            const contents = $('#info-contents');
            contents.html('');
            if (elem instanceof Edge && elem.data.type === 'Memlet') {
                const sdfg_edge =
                    elem.sdfg.nodes[elem.parent_id].edges[elem.id];
                $('<p>', {
                    'class': 'info-subtitle',
                    'html': 'Connectors: ' + sdfg_edge.src_connector +
                    ' <i class="material-icons">arrow_forward</i> ' +
                    sdfg_edge.dst_connector,
                }).appendTo(contents);
                $('<hr>').appendTo(contents);
            }

            const attr_table = $('<table>', {
                id: 'sdfg-attribute-table',
                'class': 'info-table',
            }).appendTo(contents);
            const attr_table_header = $('<thead>').appendTo(attr_table);
            const attr_table_header_row =
                $('<tr>').appendTo(attr_table_header);
            $('<th>', {
                'class': 'key-col',
                'text': 'Attribute',
            }).appendTo(attr_table_header_row);
            $('<th>', {
                'class': 'val-col',
                'text': 'Value',
            }).appendTo(attr_table_header_row);

            const attr_table_body = $('<tbody>').appendTo(attr_table);
            for (const attr of Object.entries(elem.attributes())) {
                if (attr[0] === 'layout' ||
                    attr[0] === 'sdfg' ||
                    attr[0].startsWith('_meta_'))
                    continue;
                const val = sdfg_property_to_string(
                    attr[1],
                    renderer.view_settings()
                );
                if (val === null || val === '')
                    continue;
                const row = $('<tr>').appendTo(attr_table_body);
                $('<th>', {
                    'class': 'key-col',
                    'text': attr[0],
                }).appendTo(row);
                $('<td>', {
                    'class': 'val-col',
                    'html': val,
                }).appendTo(row);
            }

            // If we're processing an access node, add array information too.
            if (elem instanceof AccessNode) {
                const sdfg_array = elem.sdfg.attributes._arrays[
                    elem.attributes().data
                ];
                $('<br>').appendTo(contents);
                $('<p>', {
                    'class': 'info-subtitle',
                    'text': 'Array properties:'
                }).appendTo(contents);

                const array_table = $('<table>', {
                    id: 'sdfg-array-table',
                    'class': 'info-table',
                }).appendTo(contents);
                const array_table_header = $('<thead>').appendTo(array_table);
                const array_table_header_row =
                    $('<tr>').appendTo(array_table_header);
                $('<th>', {
                    'class': 'key-col',
                    'text': 'Property',
                }).appendTo(array_table_header_row);
                $('<th>', {
                    'class': 'val-col',
                    'text': 'Value',
                }).appendTo(array_table_header_row);

                const array_table_body = $('<tbody>').appendTo(array_table);
                for (const attr of Object.entries(sdfg_array.attributes)) {
                    if (attr[0] === 'layout' ||
                        attr[0] === 'sdfg' ||
                        attr[0].startsWith('_meta_'))
                        continue;
                    const val = sdfg_property_to_string(
                        attr[1],
                        renderer.view_settings()
                    );
                    if (val === null || val === '')
                        continue;
                    const row = $('<tr>').appendTo(array_table_body);
                    $('<th>', {
                        'class': 'key-col',
                        'text': attr[0],
                    }).appendTo(row);
                    $('<td>', {
                        'class': 'val-col',
                        'html': val,
                    }).appendTo(row);
                }
            }

            $('#info-clear-btn').show();
        } else {
            clear_info_box();
        }
    }
}

/**
 * Clear the info container and its title.
 * This also hides the clear button again.
 */
function clear_info_box() {
    $('#info-contents').html('');
    $('#info-title').text('');
    $('#info-clear-btn').hide();
}