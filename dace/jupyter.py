""" Jupyter Notebook support for DaCe. """

import os


# From https://stackoverflow.com/a/39662359/6489142
def _isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


# Code that runs on "import dace"
if _isnotebook():
    from IPython.display import display, HTML

    # Emit javascript headers for SDFG renderer
    sdfv_deps = [
        'renderer_dir/dagre.js', 'renderer_dir/global_vars.js',
        'renderer_elements.js', 'sdfg_utils.js', 'renderer.js'
    ]
    result = ''

    # Load dependencies
    root_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'diode', 'client')
    for dep in sdfv_deps:
        file = os.path.join(root_path, dep)
        with open(file, 'r') as fp:
            result += '<script>%s</script>\n' % fp.read()

    # Rely on internet connection for Material icons
    result += '<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">'

    # Run this code once
    display(HTML(result))
