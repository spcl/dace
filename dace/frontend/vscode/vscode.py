# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import os
import json
import socket
import time
import dace


def is_available() -> bool:
    """ Checks if vscode is listening to messages
        :return: if VSCode is listening to messages
    """
    return "DACE_port" in os.environ


def load_or_transform() -> bool:
    """ Checks if the user want's to apply transformations before the
        codegen runs on the SDFG
        :return: true if the env variable DACE_sdfg_edit is set
        to transform or load
    """
    if "DACE_sdfg_edit" in os.environ:
        return (os.environ["DACE_sdfg_edit"] == 'transform'
                or os.environ["DACE_sdfg_edit"] == 'load')
    return False


def stop_and_load(sdfg):
    """ Stops and loads a SDFG file to create the code with
        :param sdfg: The current SDFG
        :return: The loaded SDFG
    """
    send({'type': 'loadSDFG'})
    breakpoint()
    # If the env variable isn't set or none then the user doesn't want
    # to load an SDFG so return the current one
    if ("DACE_load_filename" not in os.environ
            or os.environ["DACE_load_filename"] == "none"):
        return sdfg
    return dace.SDFG.from_file(os.environ["DACE_load_filename"])


def stop_and_transform(sdfg):
    """ Stops the compilation to allow transformations to the sdfg
        :param sdfg: The sdfg to allow transformations on
        :return: Transformed sdfg
    """
    filename = os.path.abspath(os.path.join(sdfg.build_folder, 'program.sdfg'))
    sdfg.save(filename)
    send({'type': 'openSDFG', 'filename': filename})
    breakpoint()
    return dace.SDFG.from_file(filename)


def pre_codegen_action(sdfg):
    """ Before the code generation we either load a saved SDFG
        or apply transformations o the current one
        :param sdfg: The current SDFG
        :return: The newly loaded SDFG 
    """
    if os.environ["DACE_sdfg_edit"] == 'load':
        return stop_and_load(sdfg)
    elif os.environ["DACE_sdfg_edit"] == 'transform':
        return stop_and_transform(sdfg)


def send(data: json):
    """ Sends a json object to the port given as the env variable DACE_port.
        If the port isn't set we don't send anything.
        :param data: json object to send
    """

    if not is_available():
        return

    HOST = socket.gethostname()
    PORT = os.environ["DACE_port"]

    data_bytes = bytes(json.dumps(data), "utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, int(PORT)))
        s.sendall(data_bytes)