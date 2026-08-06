# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import os
import urllib.request, urllib.parse
import pathlib
import dace
import numpy as np


def expand_library_nodes_for_autodiff(sdfg):
    """Expand all library nodes, but lower any ``Reduce`` through its ``pure`` expansion.

    ``Reduce``'s default ``auto`` expansion dispatches on schedule and lands on the OpenMP
    lowering for an unscheduled CPU reduce -- a single C++ tasklet carrying a ``reduction()``
    clause, which autodiff cannot reverse. The ``pure`` expansion is a map + Python tasklet
    that it can. (The same pin is used in the einsum-lifting tests.)
    """
    from dace.libraries.standard.nodes.reduce import Reduce
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, Reduce):
            node.implementation = 'pure'
    sdfg.expand_library_nodes()


def get_data_file(url, directory_name=None) -> str:
    """ Get a data file from ``url``, cache it locally and return the local file path to it.

        :param url: the url to download from.
        :param directory_name: an optional relative directory path where the file will be downloaded to.
        :return: the path of the downloaded file.
    """

    data_directory = (pathlib.Path(dace.__file__).parent.parent / 'tests' / 'data')

    if directory_name is not None:
        data_directory /= directory_name

    data_directory.mkdir(exist_ok=True, parents=True)

    file_name = os.path.basename(urllib.parse.urlparse(url).path)
    file_path = str(data_directory / file_name)

    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
    return file_path


def tensors_close(name, expected, result, rtol=1e-5, atol=1e-5):

    def to_numpy(x):
        if hasattr(x, 'detach'):
            x = x.detach()
        if hasattr(x, 'cpu'):
            x = x.cpu()
        if hasattr(x, 'numpy'):
            x = x.numpy()
        return x

    expected = to_numpy(expected)
    result = to_numpy(result)
    np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol, err_msg=f'{name} not close')


def torch_tensors_close(name, torch_v, dace_v, rtol=1e-5, atol=1e-4):
    """
    Assert that the two torch tensors are close. Prints a nice error string if not.
    """
    # check that the device is correct
    assert torch_v.device == dace_v.device, "Tensors are on different devices"

    torch_v = torch_v.detach().cpu().numpy()
    dace_v = dace_v.detach().cpu().numpy()
    np.testing.assert_allclose(dace_v, torch_v, rtol=rtol, atol=atol, err_msg=f'{name} not close')
