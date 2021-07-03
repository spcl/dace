# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import os
import json
import socket
import time
import dace


def pre_codegen_transform() -> bool:
    """ Checks if the user want's to apply transformations before the
        codegen runs on the SDFG
        :return: true if the env variable DACE_sdfg_edit is set
    """
    if "DACE_sdfg_edit" in os.environ:
        return os.environ["DACE_sdfg_edit"]
    return False


def stop_and_transform(sdfg):
    """ Stops the compilation to allow transformations to the sdfg
        :param sdfg: The sdfg to allow transformations on
        :return: Transformed sdfg
    """
    filename = os.path.abspath(os.path.join(sdfg.build_folder, 'program.sdfg'))
    sdfg.save(filename)
    send({'type': 'stopForTransformation', 'filename': filename})
    breakpoint()
    return dace.SDFG.from_file(filename)


def send(data: json):
    """ Sends a json object to the port given as the env variable DACE_port.
        If the port isn't set we don't send anything.
        :param data: json object to send
    """

    if "DACE_port" not in os.environ:
        return

    HOST = socket.gethostname()
    PORT = os.environ["DACE_port"]

    data_bytes = bytes(json.dumps(data), "utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, int(PORT)))
        s.sendall(data_bytes)